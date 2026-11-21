# SPDX-License-Identifier: Apache-2.0
"""Remote connections: TLS (daemon mode) and SSH pipe mode (PLAN.md §15).

Server authentication pins the SHA-256 of the server's SubjectPublicKeyInfo
from the invite. A mismatch is a hard failure before any request is sent
(§12). Client authentication is an Ed25519 signature over the server nonce
and the server key fingerprint.
"""

from __future__ import annotations

import base64
import collections
import hashlib
import hmac
import os
import secrets
import shlex
import shutil
import socket
import ssl
import struct
import subprocess
import threading
import time

from . import _debug, _sshhost, keys
from ._pb import auth_pb2 as apb
from ._pb import compute_pb2 as cpb
from ._pb import graph_pb2 as gpb
from .config import ClientConfig, ServerEntry
from .connection import Connection
from .errors import NotPairedError, ReflopError, ReflopRemoteError, ServerKeyMismatch, SessionLostError, sanitize

PROTOCOL_VERSION = 1
AUTH_CONTEXT = b"reflop-auth-v1\x00"
PAIR_CONTEXT = b"reflop-pair-v1\x00"
MAX_FRAME = 64 << 20
# The largest frame the client accepts, whatever the server advertises
# (reflopd's own limit is at most 1 GiB, plus 1 MiB of envelope).
MAX_FRAME_CEILING = (1 << 30) + (1 << 20)


def frame_limit(server_max_frame: int) -> int:
    """The frame cap after authentication: the server's limit plus envelope
    room, never above the client's own ceiling (SEC_SWEEP M3)."""
    if server_max_frame <= 0:
        return MAX_FRAME
    return min(int(server_max_frame) + (1 << 20), MAX_FRAME_CEILING)

def _repeated(fd) -> bool:
    """FieldDescriptor.label is gone in protobuf 7; is_repeated is newer."""
    if hasattr(fd, "is_repeated"):
        return bool(fd.is_repeated)
    return fd.label == fd.LABEL_REPEATED

_HDR = struct.Struct(">I")
INVITE_PREFIX = "reflop1-"


# ---------------------------------------------------------------- framing


class Framer:
    """Strict framing on a byte stream (socket or pipe)."""

    def __init__(self, rfile, wfile, close) -> None:
        self.rfile = rfile
        self.wfile = wfile
        self._close = close
        self.wlock = threading.Lock()
        self.max_frame = MAX_FRAME
        self.stderr_tail: SshStderr | None = None  # SSH pipe mode only

    def diagnostics(self) -> str:
        """What ssh last wrote to stderr, as " (ssh: ...)", or "" (TLS, or
        ssh said nothing). For error messages only."""
        t = self.stderr_tail.text() if self.stderr_tail is not None else ""
        return f" (ssh: {t})" if t else ""

    def read(self) -> cpb.Envelope:
        hdr = self._exact(4)
        (n,) = _HDR.unpack(hdr)
        if n > self.max_frame:
            raise ReflopError(f"server sent a {n}-byte frame (limit {self.max_frame})")
        env = cpb.Envelope()
        env.ParseFromString(self._exact(n))
        _check_unknown(env)
        return env

    def _exact(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self.rfile.read(n - len(buf))
            if not chunk:
                raise EOFError("connection closed")
            buf += chunk
        return bytes(buf)

    def write(self, env: cpb.Envelope) -> None:
        data = env.SerializeToString(deterministic=True)
        with self.wlock:
            self.wfile.write(_HDR.pack(len(data)) + data)
            self.wfile.flush()

    def close(self) -> None:
        self._close()


def _check_unknown(msg) -> None:
    from google.protobuf import unknown_fields

    if len(unknown_fields.UnknownFieldSet(msg)) > 0:
        raise ReflopError("server message has unknown fields")
    for fd, v in msg.ListFields():
        if fd.message_type is None or (_repeated(fd) and fd.message_type.GetOptions().map_entry):
            continue
        for x in (v if _repeated(fd) else [v]):
            _check_unknown(x)


def spki_fingerprint(der_cert: bytes) -> bytes:
    from cryptography import x509
    from cryptography.hazmat.primitives import serialization

    cert = x509.load_der_x509_certificate(der_cert)
    spki = cert.public_key().public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
    return hashlib.sha256(spki).digest()


def dial_tls(addresses: list[str], fingerprint: bytes, timeout: float = 15.0) -> Framer:
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_3
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE  # replaced by the pinned SPKI check below
    ctx.set_alpn_protocols(["reflop/1"])
    last: Exception | None = None
    for addr in addresses:
        host, _, port = addr.rpartition(":")
        host = host.strip("[]")
        try:
            _debug.log(f"TLS: connecting to {addr}")
            raw = socket.create_connection((host, int(port)), timeout=timeout)
            tls = ctx.wrap_socket(raw, server_hostname=None)
            der = tls.getpeercert(binary_form=True)
            if der is None or not hmac.compare_digest(spki_fingerprint(der), fingerprint):
                tls.close()
                raise ServerKeyMismatch(
                    f"the server at {addr} presented a different key than the one pinned at pairing. "
                    "This could be an impersonation attempt; nothing was sent. If the Owner reinstalled reflopd, "
                    "pair again with a new invite.")
            _debug.log(f"TLS: connected to {addr}, server key matches the pinned key")
            tls.settimeout(None)
            tls.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            f = tls.makefile("rwb", buffering=1 << 20)

            def close(tls=tls, f=f):
                # Unblock the reader thread first: closing a buffered file
                # whose read() is in progress would deadlock on its lock.
                for step in (lambda: tls.shutdown(socket.SHUT_RDWR), tls.close, f.close):
                    try:
                        step()
                    except OSError:
                        pass
            return Framer(f, f, close)
        except ServerKeyMismatch:
            raise
        except OSError as e:
            _debug.log(f"TLS: {addr}: {e}")
            last = e
    raise ReflopError(f"could not reach the server ({', '.join(addresses)}): {last}")


def ssh_command(target: str, keyfile, fingerprint_hex: str, host_keys: list[str],
                port: int | None = None) -> list[str]:
    """The ssh argv for SSH pipe mode. With host keys recorded at pairing,
    ssh trusts only those (SEC_SWEEP L6); without, it checks the user's
    known_hosts and never accepts an unknown key."""
    ssh = os.environ.get("REFLOP_SSH_COMMAND") or shutil.which("ssh")
    if ssh is None:
        raise ReflopError("ssh is not installed")
    cmd = [ssh, "-T", *_sshhost.BASE_OPTIONS, *_sshhost.known_hosts_options(fingerprint_hex, host_keys),
           "-i", str(keyfile)]
    if port:
        cmd += ["-p", str(port)]
    return cmd + ["--", target, "reflop-stdio"]


class SshStderr:
    """The last few lines ssh wrote to stderr (ADR-026).

    ssh says why it failed there (host key mismatch, "Permission denied
    (publickey)", connection refused, DNS), and reflopd's own stderr on the
    server arrives the same way. All of it is untrusted: memory is bounded,
    every line is sanitized and cut short, and the text is only ever shown
    in an error message. Nothing parses it or acts on it.
    """

    LINES = 8
    LINE_BYTES = 300

    def __init__(self, stream) -> None:
        self._stream = stream
        self._lines: collections.deque[str] = collections.deque(maxlen=self.LINES)
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()

    def _drain(self) -> None:
        partial = b""
        try:
            while True:
                chunk = self._stream.read(4096)
                if not chunk:
                    break
                *done, partial = (partial + chunk).split(b"\n")
                for line in done:
                    self._add(line)
                # Keep the start of an overlong line, drop the rest of it.
                partial = partial[:self.LINE_BYTES]
        except (OSError, ValueError):
            pass
        self._add(partial)

    def _add(self, raw: bytes) -> None:
        line = sanitize(raw[:self.LINE_BYTES].decode("utf-8", "replace"), self.LINE_BYTES).strip()
        if line:
            _debug.log(f"ssh: {line}")
            with self._lock:
                self._lines.append(line)

    def text(self, wait: float = 1.0) -> str:
        """The kept lines joined by "; ". Waits up to `wait` seconds for ssh
        to finish writing: its stdout can close before its stderr is read."""
        self._thread.join(wait)
        with self._lock:
            return "; ".join(self._lines)


def dial_ssh(target: str, key, fingerprint_hex: str, host_keys: list[str], port: int | None = None) -> Framer:
    cmd = ssh_command(target, keys.ssh_key_file(key), fingerprint_hex, host_keys, port)
    _debug.log(f"ssh: starting {shlex.join(cmd)}")
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)

    def close():
        try:
            p.stdin.close()
        except OSError:
            pass
        p.terminate()

    fr = Framer(p.stdout, p.stdin, close)
    fr.stderr_tail = SshStderr(p.stderr)
    return fr


def ssh_closed(target: str, fr: Framer, hint: str = "") -> ReflopError:
    """The error for an ssh connection that ended before the handshake did."""
    return ReflopError(f"ssh to {target} closed before the reflop handshake{fr.diagnostics()}.{hint}")


# ---------------------------------------------------------------- handshake


def _hello(fr: Framer) -> cpb.HelloResponse:
    fr.write(cpb.Envelope(request_id=1, hello=cpb.Hello(protocol_versions=[PROTOCOL_VERSION], opset_versions=[1],
                                                        client_kind="python", client_version=_version())))
    env = fr.read()
    if env.WhichOneof("body") == "error":
        raise ReflopRemoteError.from_pb(env.error)
    if env.WhichOneof("body") != "hello_response":
        raise ReflopError("unexpected reply to Hello")
    return env.hello_response


def _version() -> str:
    from . import __version__

    return __version__


def _authenticate(fr: Framer, key, nonce: bytes, fp: bytes) -> apb.AuthResult:
    pub = keys.public_bytes(key)
    sig = key.sign(AUTH_CONTEXT + nonce + fp + pub)
    fr.write(cpb.Envelope(request_id=3, authenticate=apb.Authenticate(public_key=pub, signature=sig)))
    env = fr.read()
    if env.WhichOneof("body") == "error":
        raise ReflopRemoteError.from_pb(env.error)
    if env.WhichOneof("body") != "auth_result":
        raise ReflopError("unexpected reply to Authenticate")
    return env.auth_result


def decode_invite(code: str) -> apb.InviteCode:
    code = code.strip()
    if not code.startswith(INVITE_PREFIX):
        raise ValueError("not a reflop invite code (it starts with reflop1-)")
    body = code[len(INVITE_PREFIX):].upper()
    raw = base64.b32decode(body + "=" * (-len(body) % 8))
    inv = apb.InviteCode()
    inv.ParseFromString(raw)
    _check_unknown(inv)
    if inv.version != 1:
        raise ValueError("unsupported invite version")
    return inv


def _expired_message(expires_unix: int, now: float) -> str:
    """Issue #5: show both times, so a wrong clock on this machine shows."""
    def utc(t: float) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(t))

    late = int(now - expires_unix)
    days, rest = divmod(late, 86400)
    hours, rest = divmod(rest, 3600)
    minutes, seconds = divmod(rest, 60)
    parts = [f"{n}{u}" for n, u in ((days, "d"), (hours, "h"), (minutes, "m")) if n] or [f"{seconds}s"]
    return (f"this invite expired at {utc(expires_unix)}, and this machine's clock says {utc(now)} "
            f"({' '.join(parts)} later). If the Owner made the invite recently, this machine's clock or date "
            "is probably wrong: compare `date -u` here and on the Owner's machine. Otherwise ask the Owner "
            "for a new invite.")


def pair(code: str, name: str | None = None) -> ServerEntry:
    """Redeem an invite and store the server (pinned key) in the config."""
    inv = decode_invite(code)
    now = time.time()
    if inv.expires_unix and inv.expires_unix < now:
        raise ReflopError(_expired_message(inv.expires_unix, now))
    fp = inv.server_key_fingerprint
    if len(fp) != 32:
        raise ReflopError("invite has no server key fingerprint")
    _check_not_paired(ClientConfig.load(), fp.hex())
    key = keys.load_or_create()
    host_keys: list[str] = []
    if inv.transport == apb.TRANSPORT_SSH:
        try:
            host_keys = _sshhost.normalize(inv.ssh_host_keys)
        except ValueError as e:
            raise ReflopError(f"invite: SSH host key: {e}") from None
        fr = dial_ssh(inv.addresses[0], key, fp.hex(), host_keys)
    else:
        fr = dial_tls(list(inv.addresses), fp)
    try:
        try:
            hr = _hello(fr)
        except (EOFError, OSError):
            if inv.transport != apb.TRANSPORT_SSH:
                raise
            raise ssh_closed(inv.addresses[0], fr, " Check that the invite is for this machine's key "
                             "(`reflop key --ssh`) and that `ssh` to that address works.") from None
        _debug.log("handshake: server key matches the invite")
        if not hmac.compare_digest(hr.server_key_fingerprint, fp):
            raise ServerKeyMismatch("the server's key does not match the invite")
        if inv.transport == apb.TRANSPORT_TLS:
            pub = keys.public_bytes(key)
            msg = PAIR_CONTEXT + hr.server_nonce + fp + inv.invite_id + pub
            mac = hmac.new(inv.secret, msg, hashlib.sha256).digest()
            fr.write(cpb.Envelope(request_id=2, pair=apb.Pair(invite_id=inv.invite_id, public_key=pub, mac=mac,
                                                              signature=key.sign(msg))))
            env = fr.read()
            if env.WhichOneof("body") == "error":
                raise ReflopRemoteError.from_pb(env.error)
        ar = _authenticate(fr, key, hr.server_nonce, fp)
    finally:
        fr.close()
    cfg = ClientConfig.load()
    entry = ServerEntry(
        name=name or _default_name(inv, cfg), transport="ssh" if inv.transport == apb.TRANSPORT_SSH else "tls",
        addresses=list(inv.addresses), fingerprint=fp.hex(), identity=sanitize(ar.identity_name, 64),
        identity_id=ar.identity_id.hex(), ssh_host_keys=host_keys)
    cfg.servers[entry.name] = entry
    if not cfg.default:
        cfg.default = entry.name
    cfg.save()
    return entry


def _check_not_paired(cfg: ClientConfig, fingerprint_hex: str) -> None:
    """One machine has one key, and a server registers a key once: pairing
    again with a server this machine is paired with needs an unpair on both
    sides first (issue #6). Checked before the invite is used up."""
    for n, s in cfg.servers.items():
        if hmac.compare_digest(s.fingerprint, fingerprint_hex):
            raise ReflopError(
                f"this machine is already paired with that server, as {s.identity!r} (local name {n!r}). "
                f"To pair again, run `reflop unpair {n}` first. If the Owner has not unpaired you yet, they "
                f"run `reflopd unpair {s.identity}`; then ask them for a new invite.")


def _default_name(inv, cfg) -> str:
    base = "gpu"
    if inv.addresses:
        host = inv.addresses[0].split("@")[-1].rsplit(":", 1)[0].strip("[]")
        if host and host not in ("127.0.0.1", "localhost", "::1"):
            base = "".join(ch for ch in host if ch.isalnum() or ch in "-_.")[:32] or "gpu"
    name, i = base, 2
    while name in cfg.servers:
        name, i = f"{base}{i}", i + 1
    return name


# ---------------------------------------------------------------- connection


class RemoteConnection(Connection):
    def __init__(self, entry: ServerEntry, kind: str = "auto", *, key=None) -> None:
        super().__init__()
        self.entry = entry
        self.key = keys.load_or_create() if key is None else key
        self.fp = bytes.fromhex(entry.fingerprint)
        self.session_id = b""
        self._open()
        self.caps = self.auth.capabilities
        if self.caps.limits.upload_chunk_bytes:
            self.chunk_bytes = int(self.caps.limits.upload_chunk_bytes)
        try:
            self.gdevice = self._choose_device(kind)
            self.open_session()
        except BaseException:
            self.close()
            raise
        threading.Thread(target=self._keepalive, daemon=True).start()

    def same_identity(self, other: Connection) -> bool:
        return (isinstance(other, RemoteConnection) and self.fp == other.fp
                and bool(self.auth.identity_id) and self.auth.identity_id == other.auth.identity_id)

    def cpu_peer(self) -> RemoteConnection:
        """Open the CPU using the pinned Owner and Guest key of this session."""
        peer = RemoteConnection(self.entry, kind="cpu", key=self.key)
        if not self.same_identity(peer):
            peer.close()
            raise ReflopError("the authenticated Guest identity changed while opening the Owner's CPU session")
        return peer

    def _open(self) -> None:
        _debug.log(f"connecting to {self.entry.name} ({self.entry.transport}, {', '.join(self.entry.addresses)})")
        if self.entry.transport == "ssh":
            self.fr = dial_ssh(self.entry.addresses[0], self.key, self.entry.fingerprint, self.entry.ssh_host_keys,
                               self.entry.ssh_port)
        else:
            self.fr = dial_tls(self.entry.addresses, self.fp)
        try:
            hr = _hello(self.fr)
            if not hmac.compare_digest(hr.server_key_fingerprint, self.fp):
                self.fr.close()
                raise ServerKeyMismatch("the server's key does not match the pinned key")
            _debug.log("handshake: server key matches the pinned key")
            try:
                self.auth = _authenticate(self.fr, self.key, hr.server_nonce, self.fp)
            except ReflopRemoteError as e:
                self.fr.close()
                if e.code != cpb.ERROR_UNAUTHENTICATED:
                    raise
                raise NotPairedError(e.code, not_paired_message(self.entry)) from None
        except (EOFError, OSError):
            self.fr.close()
            if self.entry.transport != "ssh":
                raise
            raise ssh_closed(
                self.entry.addresses[0], self.fr,
                f" If sshd rejected this machine's key, the Owner has probably unpaired you: run `reflop unpair "
                f"{self.entry.name}` to forget this server. `reflop doctor` checks the connection.") from None
        _debug.log(f"authenticated as {sanitize(self.auth.identity_name, 64)!r}")
        lim = self.auth.capabilities.limits
        self.fr.max_frame = frame_limit(lim.max_frame_bytes)
        if lim.max_fetch_bytes:
            self.max_response_bytes = min(self.max_response_bytes, int(lim.max_fetch_bytes))
        self.closed = False
        self._reader = threading.Thread(target=self._read_loop, args=(self.fr,), daemon=True)
        self._reader.start()
        self._subscribe()

    def _subscribe(self) -> None:
        rid = next(self._ids)
        self.fr.write(cpb.Envelope(request_id=rid, subscribe_events=cpb.SubscribeEvents()))

    def _choose_device(self, kind: str) -> gpb.Device:
        devs = [d.device for d in self.caps.devices]
        if kind in ("auto", "cuda"):
            for d in devs:
                if d.kind == gpb.DEVICE_CUDA:
                    return d
            if kind == "cuda":
                raise ReflopError("the server has no GPU")
        for d in devs:
            if d.kind == gpb.DEVICE_CPU:
                return d
        if kind == "cpu":
            raise ReflopError("the selected Owner exposes no CPU device")
        raise ReflopError("the server exposes no usable device")

    def capabilities(self):
        return self.caps

    def open_session(self, seed: int | None = None) -> None:
        req = cpb.OpenSession(device=self.gdevice)
        if seed is not None:
            req.has_seed, req.seed = True, seed
        _debug.log("opening a session")
        r = self.call(cpb.Envelope(open_session=req), timeout=300)
        _debug.log("session opened")
        self.session_id = r.session_opened.session_id
        if r.session_opened.enabled_op_ids:
            # This session's device may support fewer operators than the
            # server's first device (e.g. CUDA-only fused kernels on the CPU).
            del self.caps.enabled_op_ids[:]
            self.caps.enabled_op_ids.extend(r.session_opened.enabled_op_ids)

    def new_session(self) -> None:
        """Open a fresh server session after the old one was lost (reflop.recover)."""
        if self.closed:
            self._open()
        self.open_session()

    def _read_loop(self, fr: Framer) -> None:
        try:
            while True:
                self._deliver(fr.read())
        except Exception as e:  # noqa: BLE001 - connection ended
            if fr is self.fr:
                # Mark it closed first: diagnostics() may wait for ssh, and a
                # request sent meanwhile must reconnect, not write to a dead pipe.
                on_purpose, self.closed = self.closed, True
                why = "" if on_purpose else fr.diagnostics()
                _debug.log(f"connection ended: {e}{why}")
                self._fail_all(SessionLostError(cpb.ERROR_SESSION_LOST, f"connection to the server lost: {e}{why}"))

    def _send(self, env: cpb.Envelope) -> None:
        if self.closed:
            self._reconnect()
        self.fr.write(env)

    def _reconnect(self) -> None:
        """Re-attach to the same server session after an idle disconnect
        (the server keeps it for the session TTL, 30 minutes by default)."""
        with self._lock:
            if self._pending:
                raise SessionLostError(cpb.ERROR_SESSION_LOST, "connection lost with requests in flight")
        _debug.log("reconnecting to resume the session")
        self._open()
        r = self.call(cpb.Envelope(open_session=cpb.OpenSession(device=self.gdevice, resume_session_id=self.session_id)),
                      timeout=60)
        if not r.session_opened.resumed:
            raise SessionLostError(cpb.ERROR_SESSION_LOST, "the server session could not be resumed")
        _debug.log("session resumed")

    def _keepalive(self) -> None:
        while not self.closed:
            time.sleep(30)
            try:
                if not self.closed:
                    self.request(cpb.Envelope(ping=cpb.Ping(nonce=secrets.randbits(63))))
            except Exception:  # noqa: BLE001
                return

    def close(self) -> None:
        try:
            if not self.closed and self.session_id:
                self.request(cpb.Envelope(close_session=cpb.CloseSession())).result(10)
        except Exception:  # noqa: BLE001
            pass
        self.closed = True
        self.fr.close()


def not_paired_message(entry: ServerEntry) -> str:
    return (f"{entry.name} no longer accepts this machine's key: the Owner has revoked or unpaired "
            f"{entry.identity!r}. Run `reflop unpair {entry.name}` to forget this server; to use it again, "
            "pair with a new invite from the Owner.")


# ---------------------------------------------------------------- unpair

# What the server side did when `reflop unpair` told it (ADR-025).
UNPAIR_REVOKED = "revoked"          # the server revoked this machine's key now
UNPAIR_ALREADY = "already"          # the server had already stopped accepting the key
UNPAIR_UNREACHABLE = "unreachable"  # could not connect
UNPAIR_MISMATCH = "mismatch"        # a different server key: nothing was sent
UNPAIR_UNSUPPORTED = "unsupported"  # reflopd too old to know Unpair
UNPAIR_SKIPPED = "skipped"          # --local
UNPAIR_FAILED = "failed"            # anything else; detail says what


def unpair(name: str | None = None, local_only: bool = False) -> tuple[ServerEntry, str, str]:
    """Forget a paired server. Unless local_only, first tell it, which
    revokes this machine's key there. The local entry is removed whatever
    the server says: a server that cannot be reached must not keep a Guest
    from forgetting it. Returns (entry, outcome, detail)."""
    cfg = ClientConfig.load()
    entry = cfg.get(name)
    outcome, detail = (UNPAIR_SKIPPED, "") if local_only else _tell_unpair(entry)
    forget(cfg, entry.name)
    return entry, outcome, detail


def _tell_unpair(entry: ServerEntry) -> tuple[str, str]:
    key = keys.load_or_create()
    fp = bytes.fromhex(entry.fingerprint)
    try:
        if entry.transport == "ssh":
            fr = dial_ssh(entry.addresses[0], key, entry.fingerprint, entry.ssh_host_keys, entry.ssh_port)
        else:
            fr = dial_tls(entry.addresses, fp)
    except ServerKeyMismatch:
        return UNPAIR_MISMATCH, ""
    except (ReflopError, OSError) as e:
        return UNPAIR_UNREACHABLE, sanitize(str(e), 300)
    try:
        hr = _hello(fr)
        if not hmac.compare_digest(hr.server_key_fingerprint, fp):
            return UNPAIR_MISMATCH, ""
        try:
            _authenticate(fr, key, hr.server_nonce, fp)
        except ReflopRemoteError as e:
            if e.code == cpb.ERROR_UNAUTHENTICATED:
                return UNPAIR_ALREADY, ""
            raise
        fr.write(cpb.Envelope(request_id=4, unpair=cpb.Unpair()))
        env = fr.read()
        body = env.WhichOneof("body")
        if body == "unpaired":
            return UNPAIR_REVOKED, ""
        if body == "error":
            if env.error.code == cpb.ERROR_PROTOCOL:
                return UNPAIR_UNSUPPORTED, ""
            if env.error.code == cpb.ERROR_UNAUTHENTICATED:
                return UNPAIR_ALREADY, ""
            raise ReflopRemoteError.from_pb(env.error)
        return UNPAIR_FAILED, "unexpected reply to Unpair"
    except (EOFError, BrokenPipeError):
        # SSH: sshd no longer lets this key in, or the connection dropped.
        return UNPAIR_UNREACHABLE, "the connection closed" + fr.diagnostics()
    except (ReflopError, OSError) as e:
        return UNPAIR_FAILED, sanitize(str(e), 300)
    finally:
        fr.close()


def forget(cfg: ClientConfig, name: str) -> None:
    """Remove a server from the config, with its pinned SSH host keys."""
    entry = cfg.servers.pop(name)
    if cfg.default == name:
        cfg.default = next(iter(cfg.servers), "")
    cfg.save()
    if not any(s.fingerprint == entry.fingerprint for s in cfg.servers.values()):
        _sshhost.forget(entry.fingerprint)


def connect(server: str | None = None, kind: str = "auto") -> RemoteConnection:
    entry = ClientConfig.load().get(server)
    return RemoteConnection(entry, kind=kind)


def admin(server: str | None = None):
    """An authenticated connection without a session (data and snapshot admin)."""
    entry = ClientConfig.load().get(server)
    conn = RemoteConnection.__new__(RemoteConnection)
    Connection.__init__(conn)
    conn.entry = entry
    conn.key = keys.load_or_create()
    conn.fp = bytes.fromhex(entry.fingerprint)
    conn._open()
    conn.caps = conn.auth.capabilities
    if conn.caps.limits.upload_chunk_bytes:
        conn.chunk_bytes = int(conn.caps.limits.upload_chunk_bytes)
    conn.session_id = b""
    return conn

