# SPDX-License-Identifier: Apache-2.0
"""`reflop update`: install the latest client release from GitHub into the
environment reflop runs in. The release's checksums.txt must carry a valid
signature by a reflop release key before the wheel is checked against it
(SEC_SWEEP M4)."""

from __future__ import annotations

import base64
import hashlib
import ipaddress
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


def update(check: bool = False, version: str | None = None) -> int:
    from . import __version__

    tag = version if version else latest()
    if not tag.startswith("v"):
        tag = "v" + tag
    want = tag[1:]
    if not version and want == __version__:
        print(f"reflop {__version__} is up to date")
        return 0
    if check:
        print(f"update available: reflop {__version__} → {want}")
        return 0
    wheel = f"reflop-{want}-py3-none-any.whl"
    base = f"{_releases()}/download/{tag}/"
    sums_raw = _get(base + "checksums.txt")
    try:
        sig = _get(base + "checksums.txt.sig")
    except urllib.error.HTTPError as e:
        raise SignatureError(f"release {tag} has no checksums.txt.sig (HTTP {e.code}); "
                             "releases before 0.1.6 are unsigned") from None
    verify_signature(sums_raw, sig)
    sums = sums_raw.decode()
    digest = next((ln.split()[0] for ln in sums.splitlines() if ln.split()[-1:] == [wheel]), None)
    if digest is None:
        raise RuntimeError(f"release {tag} has no {wheel}")
    data = _get(base + wheel)
    if hashlib.sha256(data).hexdigest() != digest:
        raise RuntimeError(f"checksum mismatch for {wheel}")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, wheel)
        with open(path, "wb") as f:
            f.write(data)
        if "/uv/tools/" in sys.prefix and shutil.which("uv"):
            cmd = ["uv", "tool", "install", "--force", path]
        elif shutil.which("uv") and os.environ.get("VIRTUAL_ENV") == sys.prefix:
            cmd = ["uv", "pip", "install", "--python", sys.executable, "--upgrade", path]
        else:
            cmd = [sys.executable, "-m", "pip", "install", "--upgrade", path]
        print("reflop: " + " ".join(cmd[:4]) + " …")
        r = subprocess.call(cmd)
    if r == 0:
        print(f"updated reflop {__version__} → {want}")
    return r
