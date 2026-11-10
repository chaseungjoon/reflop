#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pin the server compute runtime (PLAN.md §4.2, ADR-008).

Resolves each runtime variant with `uv pip compile`, then picks the exact
wheel file per architecture and records its URL and SHA-256. The output,
server/reflopd/internal/runtime/lock.json, is embedded into reflopd, so a
reflopd release fixes every byte of the runtime it installs. Guests can
never trigger, choose or modify this.

    python3 codegen/lock_runtime.py            # rewrite lock.json
"""

from __future__ import annotations

import html.parser
import json
import re
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "server/reflopd/internal/runtime/lock.json"
PY_MINOR = "3.12"
ARCHES = ("x86_64", "aarch64")
# Packages never installed into the runtime. triton is a JIT compiler; the
# worker never compiles kernels (invariant I1), so it is left out entirely.
EXCLUDE = {"triton"}

VARIANTS = {
    # name: (torch requirement, index, cuda, min driver, min cc, max cc)
    "cu132": ("torch==2.15.0", "https://download.pytorch.org/whl/cu132", "13.2", "580.65.06", [7, 5], [12, 1]),
    "cu126": ("torch==2.14.1", "https://download.pytorch.org/whl/cu126", "12.6", "525.60.13", [5, 0], [9, 0]),
    "cpu": ("torch==2.15.0", "https://download.pytorch.org/whl/cpu", "", "", None, None),
}
EXTRA = ["protobuf>=5.29,<7"]


class Links(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            d = dict(attrs)
            if "href" in d:
                self.links.append((d["href"], ""))

    def handle_data(self, data):
        if self.links and not self.links[-1][1]:
            self.links[-1] = (self.links[-1][0], data.strip())


def fetch(url: str, accept: str = "text/html") -> bytes:
    req = urllib.request.Request(url, headers={"Accept": accept, "User-Agent": "reflop-lock/1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def files_for(index: str, name: str) -> list[tuple[str, str, str]]:
    """Return (filename, url, sha256) for every file of a project."""
    out = []
    if "pypi.org" in index:
        data = json.loads(fetch(f"https://pypi.org/simple/{norm(name)}/", "application/vnd.pypi.simple.v1+json"))
        for f in data["files"]:
            if "sha256" in f.get("hashes", {}):
                out.append((f["filename"], f["url"], f["hashes"]["sha256"]))
        return out
    base = f"{index}/{norm(name)}/"
    try:
        page = fetch(base).decode()
    except Exception:
        return out
    p = Links()
    p.feed(page)
    for href, text in p.links:
        url = urllib.parse.urljoin(base, href)
        m = re.search(r"#sha256=([0-9a-f]{64})", url)
        if not m:
            continue
        fn = urllib.parse.unquote(text or url.split("/")[-1].split("#")[0])
        out.append((fn, url.split("#")[0], m.group(1)))
    return out


def wheel_score(fn: str, version: str, arch: str) -> int | None:
    if not fn.endswith(".whl"):
        return None
    parts = fn[:-4].split("-")
    if len(parts) < 5:
        return None
    ver = urllib.parse.unquote(parts[1])
    if ver != version:
        return None
    py, abi, plat = parts[-3], parts[-2], parts[-1]
    cp = "cp" + PY_MINOR.replace(".", "")
    pys = py.split(".")
    if cp in pys and abi in (cp, "abi3", "none"):
        s = 30
    elif any(x.startswith("cp3") for x in pys) and abi == "abi3":
        s = 20
    elif ("py3" in pys or "py2.py3" in py) and abi == "none":
        s = 10
    else:
        return None
    ok = False
    for pl in plat.split("."):
        if pl == "any":
            ok = True
        m = re.fullmatch(r"manylinux_2_(\d+)_" + arch, pl)
        if m and int(m.group(1)) <= 28:
            ok = True
            s += int(m.group(1)) / 100
        if pl in (f"manylinux2014_{arch}", f"manylinux1_{arch}", f"manylinux2010_{arch}"):
            ok = True
    return int(s * 100) if ok else None


def resolve(variant: str, arch: str) -> list[dict]:
    req, index, *_ = VARIANTS[variant]
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("\n".join([req, *EXTRA]) + "\n")
        reqfile = f.name
    out = subprocess.run(
        ["uv", "pip", "compile", "-q", reqfile, "--python-version", PY_MINOR,
         "--python-platform", f"{arch}-manylinux_2_28", "--index-url", index,
         "--extra-index-url", "https://pypi.org/simple", "--index-strategy", "unsafe-best-match",
         "--no-header", "--no-annotate"],
        check=True, capture_output=True, text=True).stdout
    pins = []
    for line in out.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, version = line.split("==")
        if norm(name) in EXCLUDE:
            continue
        best = None
        for idx in (index, "https://pypi.org/simple"):
            for fn, url, sha in files_for(idx, name):
                sc = wheel_score(fn, version, arch)
                if sc is not None and (best is None or sc > best[0]):
                    best = (sc, fn, url, sha)
            if best:
                break
        if best is None:
            sys.exit(f"lock_runtime: no wheel for {name}=={version} on {arch}")
        pins.append({"name": norm(name), "version": version, "file": best[1], "url": best[2], "sha256": best[3]})
        print(f"  {variant}/{arch}: {best[1]}", file=sys.stderr)
    return pins


def python_build() -> dict:
    rel = json.loads(fetch("https://api.github.com/repos/astral-sh/python-build-standalone/releases/latest", "application/json"))
    tag = rel["tag_name"]
    sums = fetch(f"https://github.com/astral-sh/python-build-standalone/releases/download/{tag}/SHA256SUMS").decode()
    res = {"release": tag}
    for arch in ARCHES:
        pat = re.compile(rf"^([0-9a-f]{{64}})\s+(cpython-({re.escape(PY_MINOR)}\.\d+)\+{tag}-{arch}-unknown-linux-gnu-install_only\.tar\.gz)$")
        for line in sums.splitlines():
            m = pat.match(line.strip())
            if m:
                res[arch] = {"file": m.group(2), "sha256": m.group(1),
                             "url": f"https://github.com/astral-sh/python-build-standalone/releases/download/{tag}/{m.group(2)}"}
                res["version"] = m.group(3)
    for arch in ARCHES:
        if arch not in res:
            sys.exit(f"lock_runtime: no CPython {PY_MINOR} build for {arch} in {tag}")
    return res


def main() -> None:
    lock = {"format": 1, "python": python_build(), "variants": {}}
    for v, (req, index, cuda, drv, ccmin, ccmax) in VARIANTS.items():
        lock["variants"][v] = {
            "torch": req.split("==")[1], "cuda": cuda, "min_driver": drv,
            "min_cc": ccmin, "max_cc": ccmax,
            "wheels": {arch: resolve(v, arch) for arch in ARCHES},
        }
    OUT.write_text(json.dumps(lock, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}", file=sys.stderr)


if __name__ == "__main__":
    main()
