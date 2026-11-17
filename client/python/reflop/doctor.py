# SPDX-License-Identifier: Apache-2.0
"""`reflop doctor`: client-side diagnostics."""

from __future__ import annotations

import os
import platform
import sys


def doctor(server: str | None = None) -> int:
    import torch

    from . import __version__, keys
    from .config import ClientConfig, config_dir
    from .errors import sanitize

    bad = 0

    def line(ok, msg, warn=False):
        """warn: worth reading, but does not make doctor fail."""
        nonlocal bad
        bad += 0 if ok or warn else 1
        print(f"  {'WARN' if warn else 'ok  ' if ok else 'FAIL'} {msg}")

    print(f"reflop {__version__}, python {sys.version.split()[0]}, {platform.system()} {platform.machine()}")
    tv = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    line(tv >= (2, 6), f"torch {torch.__version__} (reflop needs ≥ 2.6; no CUDA needed on this machine)")
    openmp(line, sys.platform)
    try:
        k = keys.load_or_create()
        where = "macOS Keychain" if keys._keychain_ok() and keys._keychain_get() else str(keys._key_file())
        line(True, f"identity key: {keys.public_bytes(k).hex()[:16]}… stored in {where}")
    except Exception as e:  # noqa: BLE001
        line(False, f"identity key: {e}")
    cfg = ClientConfig.load()
    line(bool(cfg.servers), f"{len(cfg.servers)} paired server(s) in {config_dir()}")
    if cfg.servers:
        try:
            from .client import admin

            c = admin(server)
            line(True, f"connected to {c.entry.name}: key matches the pinned fingerprint, authenticated as "
                       f"{sanitize(c.auth.identity_name, 64)!r}")
            c.close()
        except Exception as e:  # noqa: BLE001
            line(False, f"connection: {sanitize(str(e))}")
    return 1 if bad else 0


def openmp(line, plat: str) -> None:
    """List the OpenMP runtimes; on macOS, flag a second one (issue #7).
    torch is already imported, so its runtime is among the loaded ones."""
    from . import _openmp

    try:
        rts, loaded = _openmp.installed(), _openmp.loaded()
    except Exception as e:  # noqa: BLE001 - diagnostics must not end doctor
        line(True, f"OpenMP: could not list the runtimes ({e})", warn=True)
        return
    for level, msg in _openmp.report(rts, loaded, plat, os.environ.get("OMP_NUM_THREADS")):
        line(level == "ok", msg, warn=level == "warn")
