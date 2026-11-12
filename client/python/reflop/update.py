# SPDX-License-Identifier: Apache-2.0
"""`reflop update`: install the latest client release into the environment
reflop runs in. The release's checksums.txt must carry a valid signature by a
reflop release key, and every file installed must match it (SEC_SWEEP M4).

How depends on the install: a uv project upgrades uv.lock and syncs (from
PyPI, after checking the locked hashes), a uv tool reinstalls from PyPI, and
anything else installs the release's verified wheel."""

from __future__ import annotations

import base64
import hashlib
import ipaddress
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request

DEFAULT_RELEASES = "https://github.com/chaseungjoon/reflop/releases"
DEFAULT_PYPI = "https://pypi.org"
TAG = re.compile(r"^v\d+\.\d+\.\d+([-+][0-9A-Za-z.-]+)?$")
MAX_DOWNLOAD = 256 << 20

# The public keys that sign releases; the same list is in reflopd
# (selfupdate.ReleaseKeys) and install.sh, and a test keeps them in sync.
RELEASE_KEYS = (
    "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIKqs9C0dx4nGfqCL1Vw0hgwleOGNaDOPYyMBCwYU3lsW reflop-release",
)
SIGNATURE_NAMESPACE = "reflop-release"


class SignatureError(RuntimeError):
    """A release's signature is missing or invalid."""


def check_base_url(raw: str) -> str:
    """https only; plain http only to this machine (a local test mirror)."""
    u = urllib.parse.urlsplit(raw)
    if u.scheme == "https" and u.hostname:
        return raw.rstrip("/")
    if u.scheme == "http" and u.hostname:
        h = u.hostname
        try:
            loop = ipaddress.ip_address(h).is_loopback
        except ValueError:
            loop = h == "localhost"
        if loop:
            return raw.rstrip("/")
    raise RuntimeError(f"REFLOP_UPDATE_URL must use https (http is allowed only for localhost), got {raw!r}")


def _releases() -> str:
    v = os.environ.get("REFLOP_UPDATE_URL")
    return check_base_url(v) if v else DEFAULT_RELEASES


def _ssh_string(b: bytes, off: int) -> tuple[bytes, int]:
    if off + 4 > len(b):
        raise SignatureError("malformed signature")
    (n,) = struct.unpack(">I", b[off:off + 4])
    off += 4
    if off + n > len(b):
        raise SignatureError("malformed signature")
    return b[off:off + n], off + n


def _key_blob(line: str) -> bytes:
    f = line.split()
    return base64.b64decode(f[1])


def verify_signature(msg: bytes, armored: bytes, keys=RELEASE_KEYS) -> None:
    """Check an armored OpenSSH signature (ssh-keygen -Y sign, SSHSIG
    format) over msg. Raises SignatureError."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    text = armored.decode("ascii", "replace").strip().splitlines()
    if len(text) < 3 or text[0] != "-----BEGIN SSH SIGNATURE-----" or text[-1] != "-----END SSH SIGNATURE-----":
        raise SignatureError("not an SSH signature")
    try:
        blob = base64.b64decode("".join(text[1:-1]), validate=True)
    except ValueError:
        raise SignatureError("malformed signature") from None
    if blob[:6] != b"SSHSIG" or len(blob) < 10 or struct.unpack(">I", blob[6:10])[0] != 1:
        raise SignatureError("malformed signature")
    off = 10
    pub, off = _ssh_string(blob, off)
    ns, off = _ssh_string(blob, off)
    reserved, off = _ssh_string(blob, off)
    hash_alg, off = _ssh_string(blob, off)
    sig, off = _ssh_string(blob, off)
    if off != len(blob):
        raise SignatureError("malformed signature")
    if ns != SIGNATURE_NAMESPACE.encode():
        raise SignatureError("the signature is not for reflop releases")
    if not any(_key_blob(k) == pub for k in keys):
        raise SignatureError("signed by a key that is not a reflop release key")
    ktype, o = _ssh_string(pub, 0)
    raw_pub, o = _ssh_string(pub, o)
    sfmt, o2 = _ssh_string(sig, 0)
    raw_sig, o2 = _ssh_string(sig, o2)
    if ktype != b"ssh-ed25519" or sfmt != b"ssh-ed25519" or len(raw_pub) != 32 or len(raw_sig) != 64:
        raise SignatureError("unsupported signature algorithm")
    if hash_alg == b"sha512":
        h = hashlib.sha512(msg).digest()
    elif hash_alg == b"sha256":
        h = hashlib.sha256(msg).digest()
    else:
        raise SignatureError("unsupported signature hash")
    s = lambda x: struct.pack(">I", len(x)) + x  # noqa: E731
    signed = b"SSHSIG" + s(ns) + s(reserved) + s(hash_alg) + s(h)
    try:
        Ed25519PublicKey.from_public_bytes(raw_pub).verify(raw_sig, signed)
    except InvalidSignature:
        raise SignatureError("bad signature") from None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


def latest() -> str:
    opener = urllib.request.build_opener(_NoRedirect)
    req = urllib.request.Request(_releases() + "/latest", method="HEAD", headers={"User-Agent": "reflop"})
    try:
        opener.open(req, timeout=30)
    except urllib.error.HTTPError as e:
        loc = e.headers.get("Location", "")
        tag = loc.rsplit("/", 1)[-1]
        if TAG.match(tag):
            return tag
    raise RuntimeError("could not determine the latest reflop release (none published yet?)")


def _get(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "reflop"}), timeout=300) as r:
        data = r.read(MAX_DOWNLOAD + 1)
    if len(data) > MAX_DOWNLOAD:
        raise RuntimeError(f"download {url}: larger than {MAX_DOWNLOAD} bytes")
    return data


def _pypi() -> str:
    v = os.environ.get("REFLOP_PYPI_URL")
    return check_base_url(v) if v else DEFAULT_PYPI


def _vtuple(v: str) -> tuple[int, int, int] | None:
    m = re.match(r"^(\d+)\.(\d+)\.(\d+)", v)
    return (int(m[1]), int(m[2]), int(m[3])) if m else None


def signed_sums(tag: str) -> dict[str, str]:
    """The release's checksums.txt as {file: sha256}, after its signature
    checks out against RELEASE_KEYS."""
    base = f"{_releases()}/download/{tag}/"
    raw = _get(base + "checksums.txt")
    try:
        sig = _get(base + "checksums.txt.sig")
    except urllib.error.HTTPError as e:
        raise SignatureError(f"release {tag} has no checksums.txt.sig (HTTP {e.code}); "
                             "releases before 0.1.6 are unsigned") from None
    verify_signature(raw, sig)
    sums = {}
    for ln in raw.decode().splitlines():
        f = ln.split()
        if len(f) == 2:
            sums[f[1].lstrip("*")] = f[0]
    return sums


def pypi_sha256(version: str, filename: str) -> str | None:
    """SHA-256 of one file of reflop==version on PyPI, or None if PyPI does
    not have it (yet: the upload follows the release by a few minutes)."""
    try:
        meta = json.loads(_get(f"{_pypi()}/pypi/reflop/{version}/json"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    for u in meta.get("urls", []):
        if u.get("filename") == filename:
            return u.get("digests", {}).get("sha256")
    return None


def lock_entry(text: str) -> tuple[str | None, set[str]] | None:
    """reflop's (version, file hashes) in a uv.lock, or None."""
    for block in re.split(r"(?m)^\[\[package\]\]\s*$", text)[1:]:
        if re.search(r'(?m)^name = "reflop"\s*$', block):
            v = re.search(r'(?m)^version = "([^"]+)"\s*$', block)
            return (v[1] if v else None, set(re.findall(r"sha256:([0-9a-f]{64})", block)))
    return None


def install_kind(prefix: str) -> tuple[str, str | None]:
    """How the running reflop was installed: ("uv-tool" | "pipx" | "private"
    | "uv-project" | "venv", project dir for uv-project)."""
    p = os.path.normpath(prefix).replace(os.sep, "/")
    if "/uv/tools/" in p:
        return "uv-tool", None
    if "/pipx/venvs/" in p:
        return "pipx", None
    if p.endswith("/reflop/client-venv"):
        return "private", None
    if os.path.basename(p) == ".venv":
        proj = os.path.dirname(os.path.normpath(prefix))
        lock = os.path.join(proj, "uv.lock")
        if os.path.isfile(lock) and os.path.isfile(os.path.join(proj, "pyproject.toml")):
            with open(lock, encoding="utf-8") as f:
                if lock_entry(f.read()):
                    return "uv-project", proj
    return "venv", None


def _run(cmd: list[str]) -> int:
    print("reflop: " + " ".join(cmd))
    return subprocess.call(cmd)


def _installed_version() -> str:
    out = subprocess.run([sys.executable, "-c", "import importlib.metadata as m; print(m.version('reflop'))"],
                         capture_output=True, text=True)
    return out.stdout.strip()


def _update_uv_project(uv: str, proj: str, want: str, sums: dict[str, str]) -> int:
    """Upgrade reflop in uv.lock, check the locked files against the signed
    release, then sync, so `uv run` and `uv sync` keep the new version."""
    lock = os.path.join(proj, "uv.lock")
    with open(lock, "rb") as f:
        before = f.read()

    def restore():
        with open(lock, "wb") as f:
            f.write(before)

    r = _run([uv, "lock", "--project", proj, "--upgrade-package", "reflop"])
    if r != 0:
        return r
    with open(lock, encoding="utf-8") as f:
        got, hashes = lock_entry(f.read()) or (None, set())
    if got != want:
        restore()
        print(f"reflop: {proj}/pyproject.toml keeps reflop at {got}; to allow {want}, run there:\n"
              f"  uv add 'reflop>={want}'", file=sys.stderr)
        return 1
    if not hashes or not hashes <= set(sums.values()):
        restore()
        raise RuntimeError("uv.lock's files for reflop do not match the signed release; uv.lock was restored")
    return _run([uv, "sync", "--project", proj, "--inexact"])


def update(check: bool = False, version: str | None = None) -> int:
    from . import __version__

    tag = version if version else latest()
    if not tag.startswith("v"):
        tag = "v" + tag
    if not TAG.match(tag):
        raise RuntimeError(f"invalid version {tag!r} (expected e.g. v0.2.0)")
    want = tag[1:]
    cur, new = _vtuple(__version__), _vtuple(want)
    if not version and cur and new and cur >= new:
        print(f"reflop {__version__} is up to date (latest release: {want})")
        return 0
    if check:
        print(f"update available: reflop {__version__} → {want} (run `reflop update`)")
        return 0
    sums = signed_sums(tag)
    wheel = f"reflop-{want}-py3-none-any.whl"
    digest = sums.get(wheel)
    if digest is None:
        raise RuntimeError(f"release {tag} has no {wheel}")
    kind, proj = install_kind(sys.prefix)
    uv = shutil.which("uv")

    if kind == "uv-project":
        if version:
            print(f"reflop: this environment belongs to the uv project {proj}; pin the version there:\n"
                  f"  uv add reflop=={want}", file=sys.stderr)
            return 1
        if not uv:
            raise RuntimeError(f"{proj} is a uv project, but uv is not on PATH; run there: uv add 'reflop>={want}'")
        on_pypi = pypi_sha256(want, wheel)
        if on_pypi is None:
            raise RuntimeError(f"reflop {want} is not on PyPI yet (it follows the release by a few minutes); "
                               "try again shortly")
        if on_pypi != digest:
            raise RuntimeError(f"PyPI's {wheel} does not match the signed release; not installing it")
        r = _update_uv_project(uv, proj, want, sums)
    else:
        # A uv tool installed from PyPI stays a PyPI install, so `uv tool
        # upgrade` keeps working. Otherwise (or before PyPI has it), install
        # the release's verified wheel.
        on_pypi = pypi_sha256(want, wheel) if kind == "uv-tool" and uv else None
        if on_pypi is not None and on_pypi != digest:
            raise RuntimeError(f"PyPI's {wheel} does not match the signed release; not installing it")
        if on_pypi is not None:
            r = _run([uv, "tool", "install", "--force", f"reflop=={want}"])
        else:
            data = _get(f"{_releases()}/download/{tag}/{wheel}")
            if hashlib.sha256(data).hexdigest() != digest:
                raise RuntimeError(f"checksum mismatch for {wheel}")
            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, wheel)
                with open(path, "wb") as f:
                    f.write(data)
                if kind == "uv-tool" and uv:
                    cmd = [uv, "tool", "install", "--force", path]
                elif uv and os.environ.get("VIRTUAL_ENV") == sys.prefix:
                    cmd = [uv, "pip", "install", "--python", sys.executable, "--upgrade", path]
                else:
                    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", path]
                r = _run(cmd)
    if r != 0:
        return r
    got = _installed_version()
    if got != want:
        print(f"reflop: expected {want} after the update, but this environment has {got or 'no reflop'}",
              file=sys.stderr)
        return 1
    print(f"updated reflop {__version__} → {want}")
    if kind == "venv":
        print(f"note: if a requirements file pins reflop, update it too: reflop=={want}")
    return 0
