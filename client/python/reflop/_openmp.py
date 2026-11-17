# SPDX-License-Identifier: Apache-2.0
"""Find the OpenMP runtimes in this environment (issue #7).

On macOS the PyTorch wheel ships its own libomp. A second copy, from
scikit-learn, xgboost, conda's llvm-openmp or Homebrew, in the same process
makes the two runtimes fight over threads: `OMP: Error #15`, a segfault or a
hang. `reflop doctor` uses this module to name the copies and the packages
they came from. It only looks at file names and at the list of libraries the
process has loaded; it never loads anything itself.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass

# libomp (LLVM), libiomp5 (Intel) and libgomp (GNU), as shipped in wheels:
# plain (torch/lib/libomp.dylib), versioned (libgomp.so.1) or renamed by
# auditwheel/delocate (sklearn.libs/libgomp-a34b3233.so.1.0.0).
_NAME = re.compile(r"^(libomp|libiomp5|libiomp5md|libgomp)([.-][^/\\]*)?\.(dylib|so|dll)(\.\d+)*$")


def is_runtime(path: str) -> bool:
    return bool(_NAME.match(os.path.basename(path)))


@dataclass(frozen=True)
class Runtime:
    package: str  # distribution name, or "" when no installed package owns the file
    path: str


def installed() -> list[Runtime]:
    """OpenMP runtimes shipped by the Python packages installed here, one
    entry per file (symlinks to the same file count once)."""
    from importlib import metadata

    out: list[Runtime] = []
    seen: set[str] = set()
    for dist in metadata.distributions():
        try:
            files = dist.files or ()
            name = dist.metadata["Name"] or "?"
        except Exception:  # noqa: BLE001 - a broken RECORD must not break doctor
            continue
        for f in files:
            if not is_runtime(str(f)):
                continue
            try:
                p = os.path.realpath(dist.locate_file(f))
            except Exception:  # noqa: BLE001
                continue
            if p in seen or not os.path.exists(p):
                continue
            seen.add(p)
            out.append(Runtime(name, p))
    return sorted(out, key=lambda r: (r.package.lower(), r.path))


def _loaded_paths() -> list[str]:
    if sys.platform == "darwin":
        import ctypes

        libc = ctypes.CDLL(None)
        libc._dyld_image_count.restype = ctypes.c_uint32
        libc._dyld_get_image_name.restype = ctypes.c_char_p
        libc._dyld_get_image_name.argtypes = [ctypes.c_uint32]
        names = (libc._dyld_get_image_name(i) for i in range(libc._dyld_image_count()))
        return [n.decode(errors="replace") for n in names if n]
    if sys.platform.startswith("linux"):
        out = []
        with open("/proc/self/maps", encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.split(None, 5)
                if len(parts) == 6 and parts[5].startswith("/"):
                    out.append(parts[5].rstrip("\n"))
        return out
    return []


def loaded() -> list[str]:
    """OpenMP runtimes loaded in this process, as resolved paths."""
    try:
        paths = _loaded_paths()
    except Exception:  # noqa: BLE001 - diagnostics only
        return []
    out: list[str] = []
    for p in paths:
        if is_runtime(p):
            r = os.path.realpath(p)
            if r not in out:
                out.append(r)
    return out


def owner(path: str, runtimes: list[Runtime]) -> str:
    """The package that installed `path`, or "" if none of `runtimes` is it."""
    for r in runtimes:
        if r.path == path:
            return r.package
    return ""


def report(installed_rts: list[Runtime], loaded_paths: list[str], platform: str,
           omp_num_threads: str | None) -> list[tuple[str, str]]:
    """Doctor lines as (level, text), level being "ok", "warn" or "fail".

    Only macOS is judged: there torch's libomp clashes with any other copy.
    On Linux torch uses libgomp, several copies of which coexist, so the
    runtimes are only listed.
    """
    def label(path: str) -> str:
        pkg = owner(path, installed_rts)
        return f"{pkg} ({path})" if pkg else path

    if not installed_rts and not loaded_paths:
        return [("ok", "OpenMP: no runtime found")]
    lines: list[tuple[str, str]] = []
    loaded_desc = "; ".join(label(p) for p in loaded_paths) or "none"
    others = [r for r in installed_rts if r.path not in loaded_paths]
    if platform != "darwin":
        lines.append(("ok", f"OpenMP loaded with torch: {loaded_desc}"))
        if others:
            lines.append(("ok", "OpenMP also installed: " + "; ".join(f"{r.package} ({r.path})" for r in others)))
        return lines
    if len(loaded_paths) > 1:
        lines.append(("fail", f"{len(loaded_paths)} OpenMP runtimes loaded with torch alone: {loaded_desc}. "
                              "They clash unless OMP_NUM_THREADS=1 (`reflop run` sets it); torch from PyPI "
                              "in a fresh virtual environment loads only its own (issue #7)"))
    else:
        lines.append(("ok", f"OpenMP loaded with torch: {loaded_desc}"))
    if others:
        names = sorted({r.package or r.path for r in others})
        lines.append(("warn", "other OpenMP runtimes installed: "
                              + "; ".join(f"{r.package} ({r.path})" for r in others)
                              + f". A program that imports {', '.join(names)} next to torch loads two runtimes, "
                              "which crashes or hangs on macOS (issue #7)"))
    if (len(loaded_paths) > 1 or others) and omp_num_threads != "1":
        lines.append(("warn", "`reflop run` sets OMP_NUM_THREADS=1 for you; for `python train.py` set it "
                              "yourself: OMP_NUM_THREADS=1 python train.py"))
    return lines
