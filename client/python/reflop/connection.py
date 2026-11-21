# SPDX-License-Identifier: Apache-2.0
"""Request/response plumbing shared by every transport.

A Connection multiplexes requests by request_id. Responses arrive on a
reader thread and resolve Futures. Streams (fetch) collect frames until the
one marked `end`. Events are fixed-schema and informational: they are
printed (sanitized) and never trigger client actions (§12).
"""

from __future__ import annotations

import hashlib
import itertools
import os
import sys
import threading
from concurrent.futures import Future

from ._pb import compute_pb2 as cpb
from .errors import OutOfMemoryError, ReflopError, ReflopRemoteError, SessionLostError, sanitize


def _err(e: cpb.Error) -> ReflopRemoteError:
    r = ReflopRemoteError.from_pb(e)
    if e.code == cpb.ERROR_OUT_OF_MEMORY:
        r.__class__ = OutOfMemoryError
    elif e.code == cpb.ERROR_SESSION_LOST:
        r.__class__ = SessionLostError
    return r


class _StreamCheck:
    """Bounds one streamed response (SEC_SWEEP M3).

    A server must not be able to grow the Guest process without limit: the
    total is capped at max_response_bytes, and fetch chunks must match the
    size their metadata declares (numel x itemsize), arrive in order, and
    name only the tensors that were requested.
    """

    def __init__(self, n_tensors: int, max_bytes: int) -> None:
        self.n = n_tensors
        self.max = max_bytes
        self.total = 0
        self.expect: dict[int, int] = {}
        self.got: dict[int, int] = {}

    @staticmethod
    def _nbytes(meta) -> int:
        from .trace import PB_TO_DTYPE

        dt = PB_TO_DTYPE.get(meta.dtype)
        if dt is None:
            raise ReflopError("the server sent a tensor with an unknown dtype")
        n = 1
        for d in meta.shape:
            if d < 0:
                raise ReflopError("the server sent a tensor with a negative dimension")
            n *= d
        return n * dt.itemsize

    def add(self, env: cpb.Envelope) -> None:
        if env.WhichOneof("body") != "fetch_chunk":
            raise ReflopError("unexpected message in a fetch response")
        c = env.fetch_chunk
        if not c.HasField("meta") and c.index not in self.expect and not c.data and env.end and self.n == 0:
            return  # the empty end marker of a fetch with nothing to send
        if c.index >= self.n:
            raise ReflopError("the server sent data for a tensor that was not requested")
        if c.HasField("meta"):
            if c.index in self.expect:
                raise ReflopError("the server sent a tensor's metadata twice")
            self.expect[c.index] = self._nbytes(c.meta)
            self.got[c.index] = 0
            if sum(self.expect.values()) > self.max:
                raise ReflopError(f"the server's response exceeds {self.max} bytes")
        elif c.index not in self.expect:
            raise ReflopError("the server sent tensor data before its metadata")
        if c.offset != self.got[c.index]:
            raise ReflopError("the server sent tensor data out of order")
        self.got[c.index] += len(c.data)
        self.total += len(c.data)
        if self.got[c.index] > self.expect[c.index]:
            raise ReflopError("the server sent more data than the tensor's metadata declares")
        if self.total > self.max:
            raise ReflopError(f"the server's response exceeds {self.max} bytes")

    def complete(self) -> None:
        for i in range(self.n):
            if i not in self.expect or self.got[i] != self.expect[i]:
                raise ReflopError("the server's fetch response ended before every tensor arrived")


class Connection:
    """Base class; subclasses implement _send(Envelope)."""

    def __init__(self, chunk_bytes: int = 4 << 20, max_response_bytes: int = 1 << 34) -> None:
        self._ids = itertools.count(1)
        self._pending: dict[int, tuple[Future, list | None, _StreamCheck | None]] = {}
        self._lock = threading.Lock()
        self.chunk_bytes = chunk_bytes
        self.max_response_bytes = max_response_bytes
        self.closed = False
        self.on_event = self._print_event

    @staticmethod
    def _print_event(ev: cpb.Event) -> None:
        kind = cpb.Event.Kind.Name(ev.kind).removeprefix("KIND_").lower().replace("_", " ")
        sys.stderr.write(f"reflop: server {kind}: {sanitize(ev.text, 500)}\n")

    # -------------------------------------------------------------- subclass API

    def _send(self, env: cpb.Envelope) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def _deliver(self, env: cpb.Envelope) -> None:
        """Called by the transport for every incoming envelope."""
        if env.WhichOneof("body") == "event":
            try:
                self.on_event(env.event)
            except Exception:  # noqa: BLE001 - events must never break the client
                pass
            return
        with self._lock:
            entry = self._pending.get(env.request_id)
        if entry is None:
            return  # unknown or late response: ignore (never act on it)
        fut, frames, check = entry
        if env.WhichOneof("body") == "error":
            with self._lock:
                self._pending.pop(env.request_id, None)
            if not fut.done():
                fut.set_exception(_err(env.error))
            return
        if frames is not None:
            try:
                if check is not None:
                    check.add(env)
                    if env.end:
                        check.complete()
            except ReflopError as e:
                with self._lock:
                    self._pending.pop(env.request_id, None)
                frames.clear()
                if not fut.done():
                    fut.set_exception(e)
                return
            frames.append(env)
            if not env.end:
                return
        with self._lock:
            self._pending.pop(env.request_id, None)
        if not fut.done():
            fut.set_result(frames if frames is not None else env)

    def _fail_all(self, exc: BaseException) -> None:
        with self._lock:
            pending, self._pending = self._pending, {}
        for fut, _, _ in pending.values():
            if not fut.done():
                fut.set_exception(exc)

    # -------------------------------------------------------------- requests

    def request(self, env: cpb.Envelope, stream: bool = False) -> Future:
        if self.closed:
            raise SessionLostError(cpb.ERROR_SESSION_LOST, "connection is closed")
        rid = next(self._ids)
        env.request_id = rid
        fut: Future = Future()
        check = None
        if stream:
            n = len(env.fetch.handles) if env.WhichOneof("body") == "fetch" else 0
            check = _StreamCheck(n, self.max_response_bytes)
        with self._lock:
            self._pending[rid] = (fut, [] if stream else None, check)
        try:
            self._send(env)
        except Exception as e:  # noqa: BLE001
            with self._lock:
                self._pending.pop(rid, None)
            fut.set_exception(e)
        return fut

    def request_stream(self, env: cpb.Envelope) -> Future:
        return self.request(env, stream=True)

    def call(self, env: cpb.Envelope, timeout: float | None = None) -> cpb.Envelope:
        return self.request(env).result(timeout)

    def upload(self, handle: bytes, meta, data: bytes) -> Future:
        """Stream one tensor; the future resolves on the final frame's reply."""
        if self.closed:
            raise SessionLostError(cpb.ERROR_SESSION_LOST, "connection is closed")
        rid = next(self._ids)
        fut: Future = Future()
        with self._lock:
            self._pending[rid] = (fut, None, None)
        sha = hashlib.sha256(data).digest()
        n = len(data)
        step = self.chunk_bytes
        off = 0
        while True:
            chunk = data[off:off + step]
            final = off + len(chunk) >= n
            u = cpb.UploadTensor(handle=handle, offset=off, data=chunk, final=final)
            if off == 0:
                u.meta.CopyFrom(meta)
            if final:
                u.sha256 = sha
            try:
                self._send(cpb.Envelope(request_id=rid, upload_tensor=u))
            except Exception as e:  # noqa: BLE001
                with self._lock:
                    self._pending.pop(rid, None)
                fut.set_exception(e)
                return fut
            off += len(chunk)
            if final:
                break
        return fut

    def reseed(self, seed: int) -> None:
        self.call(cpb.Envelope(reseed=cpb.Reseed(seed=seed)))

    def close(self) -> None:
        self.closed = True


class LocalConnection(Connection):
    """Runs the reflop worker's request handling in-process.

    For tests and `reflop verify`: same validation and execution code as the
    server, with no network and no sandbox. Requires the worker package
    (server/worker) on sys.path. Snapshots and checkpoints go to a temporary
    directory, handled the way reflopd handles them (checkpoint_interval_s
    > 0 turns checkpoints on, as `reflopd checkpoints on` does).
    """

    def __init__(self, device: str = "cpu", seed: int = 0, limits=None, checkpoint_interval_s: int = 0) -> None:
        super().__init__()
        from reflop_worker._pb import capabilities_pb2 as kpb
        from reflop_worker._pb import graph_pb2 as gpb
        from reflop_worker._pb import worker_pb2 as wpb
        from reflop_worker.core import Session

        init = wpb.WorkerInit(seed=seed)
        if device == "cpu":
            init.device.kind = gpb.DEVICE_CPU
        else:
            init.device.kind = gpb.DEVICE_CUDA
            init.device.index = int(device.split(":")[1]) if ":" in device else 0
        init.limits.CopyFrom(limits or default_limits(kpb))
        init.numerics.tf32_cudnn = os.environ.get("REFLOP_LOCAL_TF32", "1") == "1"
        self._init = init
        self._Session = Session
        self.worker = self._new_worker()
        self.gdevice = init.device
        self.ready = self.worker.ready(wpb.SandboxReport())
        self._wlock = threading.Lock()
        self.session_id = os.urandom(16)
        self.checkpoint_interval_s = int(checkpoint_interval_s or os.environ.get("REFLOP_LOCAL_CHECKPOINT_S", "0"))
        self._dir: str | None = None
        self._dead = False
        self._last_ckpt: float | None = None
        self._gen = 0

    def capabilities(self):
        from ._pb import capabilities_pb2 as kpb

        c = kpb.ServerCapabilities(protocol_versions=[1], opset_versions=[1],
                                   enabled_op_ids=self.ready.enabled_op_ids, capabilities=self.ready.capabilities)
        c.capabilities.extend(["snapshots", "checkpoints"])
        c.checkpoint_interval_s = self.checkpoint_interval_s
        c.devices.extend(self.ready.devices)
        c.limits.CopyFrom(self.worker.limits)
        c.numerics.CopyFrom(self.ready.numerics)
        return c

    # -------------------------------------------------------------- test hooks

    def kill_worker(self) -> None:
        """Tests: the worker dies; requests fail with SESSION_LOST until
        new_session()."""
        self._dead = True

    def _new_worker(self):
        from . import session as _s

        # Worker startup seeds PyTorch. Skip the client seed hook because this
        # worker lives in the same process as the client.
        prev = getattr(_s._seeding, "active", False)
        _s._seeding.active = True
        try:
            return self._Session(self._init)
        finally:
            _s._seeding.active = prev

    def new_session(self) -> None:
        worker = self._new_worker()
        with self._wlock:
            self.worker = worker
            self.session_id = os.urandom(16)
            self._dead = False
            self._last_ckpt = None

    def close(self) -> None:
        super().close()
        if self._dir is not None:
            import shutil

            shutil.rmtree(self._dir, ignore_errors=True)

    # -------------------------------------------------------------- requests

    def _send(self, env: cpb.Envelope) -> None:
        which = env.WhichOneof("body")
        with self._wlock:
            if which in ("list_snapshots", "delete_snapshot", "describe_snapshot"):
                return self._admin(env, which)
            if self._dead:
                return self._error(env, cpb.ERROR_SESSION_LOST, "session lost: the worker died")
            if which in ("snapshot", "checkpoint"):
                return self._write(env, which)
            if which in ("restore", "recover_session"):
                return self._read(env, which)
            self.worker.handle(env, self._deliver)

    def _error(self, env, code, msg) -> None:
        self._deliver(cpb.Envelope(request_id=env.request_id, end=True, error=cpb.Error(code=code, message=msg)))

    def _root(self) -> str:
        if self._dir is None:
            import tempfile

            self._dir = tempfile.mkdtemp(prefix="reflop-local-snapshots-")
        os.makedirs(os.path.join(self._dir, ".checkpoints"), mode=0o700, exist_ok=True)
        return self._dir

    def _ckpt_path(self, sid: bytes) -> str:
        return os.path.join(self._root(), ".checkpoints", sid.hex() + ".safetensors")

    def _write(self, env, which) -> None:
        import time

        from reflop_worker._pb import worker_pb2 as wpb

        rid = env.request_id
        if which == "checkpoint":
            if self.checkpoint_interval_s <= 0:
                return self._error(env, cpb.ERROR_PERMISSION_DENIED, "the Owner has not enabled checkpoints on this server")
            if self._last_ckpt is not None and time.time() - self._last_ckpt < self.checkpoint_interval_s:
                return self._error(env, cpb.ERROR_RESOURCE_LIMIT, "the Owner allows one checkpoint every "
                                   f"{self.checkpoint_interval_s}s")
            final = self._ckpt_path(self.session_id)
            msg = wpb.SnapshotWrite(request_id=rid, checkpoint=True, tag=env.checkpoint.tag, generation=self._gen + 1)
        else:
            name = env.snapshot.name
            final = os.path.join(self._root(), name + ".safetensors")
            msg = wpb.SnapshotWrite(request_id=rid, handles=env.snapshot.handles, keys=env.snapshot.keys)
        tmp = final + ".tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        sid = self.session_id

        def cut(r, _n):
            self.worker.writer.on_go(wpb.SnapshotGo(request_id=r, proceed=True))

        def done(resp):
            body = resp.WhichOneof("body")
            ok = body == "snapshot_saved" or (body == "checkpoint_saved" and not resp.checkpoint_saved.skipped)
            if ok and sid == self.session_id:
                with open(tmp, "rb") as f:
                    os.fsync(f.fileno())
                os.replace(tmp, final)
                if body == "checkpoint_saved":
                    self._last_ckpt, self._gen = time.time(), resp.checkpoint_saved.generation
                else:
                    resp.snapshot_saved.name = env.snapshot.name
            else:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            self._deliver(resp)

        try:
            self.worker.snapshot_write(msg, fd, cut, done)
        except Exception as e:  # noqa: BLE001 - as the worker's main loop
            from reflop_worker.errors import ReflopError as WorkerError

            if os.path.exists(tmp):
                os.remove(tmp)
            if isinstance(e, WorkerError):
                return self._deliver(cpb.Envelope(request_id=rid, end=True, error=cpb.Error.FromString(
                    e.to_pb().SerializeToString())))
            raise

    def _read(self, env, which) -> None:
        import base64
        import json

        from reflop_worker._pb import worker_pb2 as wpb
        from reflop_worker.errors import ReflopError as WorkerError
        from reflop_worker.safetensors import RESERVED, SafetensorsError

        rid = env.request_id
        if which == "restore":
            path = os.path.join(self._root(), env.restore.name + ".safetensors")
            msg = wpb.SnapshotRead(request_id=rid, keys=env.restore.keys, handles=env.restore.handles)
        else:
            old = bytes(env.recover_session.from_session_id)
            path = self._ckpt_path(old)
            if not os.path.exists(path):
                return self._error(env, cpb.ERROR_UNKNOWN_SESSION, "no checkpoint of that session")
            try:
                hdr, meta = self._header(path)
            except (SafetensorsError, ValueError) as e:
                return self._error(env, cpb.ERROR_PROTOCOL, f"checkpoint file: {e}")
            keys = sorted(k for k in hdr if not k.startswith(RESERVED))
            msg = wpb.SnapshotRead(request_id=rid, keys=keys, handles=[bytes.fromhex(k) for k in keys], restore_rng=True)
        if not os.path.exists(path):
            return self._error(env, cpb.ERROR_PROTOCOL, "no such snapshot")
        fd = os.open(path, os.O_RDONLY)
        try:
            self.worker.snapshot_read(msg, fd)
        except WorkerError as e:
            return self._deliver(cpb.Envelope(request_id=rid, end=True, error=cpb.Error.FromString(
                e.to_pb().SerializeToString())))
        if which == "restore":
            return self._deliver(cpb.Envelope(request_id=rid, end=True, restored=cpb.Restored()))
        from reflop_worker import dtypes

        info = json.loads(meta)
        out = cpb.SessionRecovered(tag=base64.b64decode(info["tag"]), generation=info["generation"],
                                   created_unix=info["created_unix"])
        for k in keys:
            dt, shape, _b, _e = hdr[k]
            out.handles.append(bytes.fromhex(k))
            out.metas.add(dtype=dtypes.TORCH_TO_PB[dt], shape=list(shape))
        os.replace(path, self._ckpt_path(self.session_id))
        self._deliver(cpb.Envelope(request_id=rid, end=True, session_recovered=out))

    def _header(self, path):
        from reflop_worker.checkpoint import META_KEY
        from reflop_worker.safetensors import parse_header

        with open(path, "rb") as f:
            data = f.read()
        start, hdr = parse_header(memoryview(data), len(data), 1 << 40)
        meta = b""
        if META_KEY in hdr:
            _dt, _shape, b, e = hdr[META_KEY]
            meta = data[start + b:start + e]
        return hdr, meta

    def _admin(self, env, which) -> None:
        from reflop_worker import dtypes
        from reflop_worker.safetensors import RESERVED

        rid = env.request_id
        root = self._root()
        if which == "list_snapshots":
            out = cpb.SnapshotList()
            for n in sorted(os.listdir(root)):
                p = os.path.join(root, n)
                if n.endswith(".safetensors") and not n.startswith(".") and os.path.isfile(p):
                    st = os.stat(p)
                    out.snapshots.add(name=n[:-len(".safetensors")], bytes=st.st_size, created_unix=int(st.st_mtime))
            return self._deliver(cpb.Envelope(request_id=rid, end=True, snapshot_list=out))
        name = env.delete_snapshot.name if which == "delete_snapshot" else env.describe_snapshot.name
        path = os.path.join(root, name + ".safetensors")
        if "/" in name or name.startswith(".") or not os.path.exists(path):
            return self._error(env, cpb.ERROR_PROTOCOL, "no such snapshot")
        if which == "delete_snapshot":
            os.remove(path)
            return self._deliver(cpb.Envelope(request_id=rid, end=True, freed=cpb.Freed()))
        hdr, _ = self._header(path)
        out = cpb.SnapshotDescription(name=name, bytes=os.path.getsize(path))
        for k in sorted(hdr):
            if k.startswith(RESERVED):
                continue
            dt, shape, _b, _e = hdr[k]
            out.keys.append(k)
            out.metas.add(dtype=dtypes.TORCH_TO_PB[dt], shape=list(shape))
        return self._deliver(cpb.Envelope(request_id=rid, end=True, snapshot_description=out))


def default_limits(kpb):
    return kpb.Limits(
        max_frame_bytes=64 << 20, max_nodes=1_000_000, max_graph_inputs=65536, max_dims=16,
        max_tensor_numel=1 << 40, max_inline_tensor_bytes=1 << 20, max_constant_bytes=64 << 10,
        max_fetch_bytes=16 << 30, max_list_length=65536, upload_chunk_bytes=4 << 20,
        max_handles_per_session=1 << 22, max_graphs_per_session=65536, max_manifest_bytes=1 << 20)
