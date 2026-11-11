# SPDX-License-Identifier: Apache-2.0
"""SSH host-key pinning for SSH pipe mode (SEC_SWEEP L6).

SSH invites carry the server's sshd host keys. They go into a known_hosts
file owned by reflop, under a HostKeyAlias tied to the server's reflop key,
and ssh runs with StrictHostKeyChecking=yes. The user's own known_hosts and
ssh_config settings therefore cannot vouch for an impostor. Mirrors
pkg/sshhost (reflopd, libreflop).
"""

from __future__ import annotations

import base64
import os
import struct
import tempfile
from pathlib import Path

from .config import config_dir

TYPES = ("ssh-ed25519", "ecdsa-sha2-nistp256", "ecdsa-sha2-nistp384", "ecdsa-sha2-nistp521", "ssh-rsa")
MAX_KEYS = 4

# Options for every SSH pipe connection, pinned or not.
BASE_OPTIONS = ("-o", "BatchMode=yes", "-o", "ServerAliveInterval=30", "-o", "IdentitiesOnly=yes",
                "-o", "ForwardAgent=no", "-o", "ForwardX11=no", "-o", "ClearAllForwardings=yes",
                "-o", "PermitLocalCommand=no", "-o", "StrictHostKeyChecking=yes", "-o", "UpdateHostKeys=no")


def parse(line: str) -> tuple[str, bytes]:
    """Check a "type base64 [comment]" line; return (type, key blob)."""
    f = line.split()
    if len(f) < 2:
        raise ValueError('expected "type base64"')
    if f[0] not in TYPES:
        raise ValueError(f"unsupported host key type {f[0][:40]!r}")
    try:
        blob = base64.b64decode(f[1], validate=True)
    except ValueError:
        raise ValueError("malformed host key") from None
    if len(blob) < 4 or len(blob) > 2048:
        raise ValueError("malformed host key")
    (n,) = struct.unpack(">I", blob[:4])
    if n > len(blob) - 4 or blob[4:4 + n] != f[0].encode():
        raise ValueError("host key type does not match its data")
    return f[0], blob


def normalize(lines) -> list[str]:
    """Validate invite host keys and return them as "type base64" lines."""
    if len(lines) > MAX_KEYS:
        raise ValueError("too many SSH host keys in the invite")
    out = []
    for line in lines:
        t, blob = parse(line)
        out.append(f"{t} {base64.b64encode(blob).decode()}")
    return out


def alias(fingerprint_hex: str) -> str:
    return "reflop-" + fingerprint_hex


def quote_option(path: str) -> str:
    if any(c in path for c in '"\n\r'):
        raise ValueError(f"unsupported character in path {path!r}")
    return '"' + path.replace("%", "%%") + '"'


def known_hosts_options(fingerprint_hex: str, host_keys: list[str]) -> list[str]:
    """ssh options that trust only host_keys; [] if none were recorded."""
    if not host_keys:
        return []
    a = alias(fingerprint_hex)
    data = "".join(f"{a} {k}\n" for k in normalize(host_keys))
    d = config_dir() / "ssh_known_hosts"
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = d / fingerprint_hex
    try:
        same = path.read_text() == data
    except OSError:
        same = False
    if not same:
        _write_atomic(path, data)
    return ["-o", "UserKnownHostsFile=" + quote_option(str(path)), "-o", "GlobalKnownHostsFile=/dev/null",
            "-o", "HostKeyAlias=" + a, "-o", "CheckHostIP=no"]


def _write_atomic(path: Path, data: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix="." + path.name + ".tmp-")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(data)
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
