# SPDX-License-Identifier: Apache-2.0
"""`reflop update`: install the latest client release from GitHub, verified
against the release's checksums.txt, into the environment reflop runs in."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request

RELEASES = os.environ.get("REFLOP_UPDATE_URL", "https://github.com/chaseungjoon/reflop/releases")
TAG = re.compile(r"^v\d+\.\d+\.\d+([-+][0-9A-Za-z.-]+)?$")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


def latest() -> str:
    opener = urllib.request.build_opener(_NoRedirect)
    req = urllib.request.Request(RELEASES + "/latest", method="HEAD", headers={"User-Agent": "reflop"})
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
        return r.read()


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
    base = f"{RELEASES}/download/{tag}/"
    sums = _get(base + "checksums.txt").decode()
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
