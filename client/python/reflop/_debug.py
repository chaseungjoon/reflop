# SPDX-License-Identifier: Apache-2.0
"""REFLOP_DEBUG: tracebacks from the CLI and connection steps on stderr.

Set REFLOP_DEBUG=1 to turn it on; unset, empty or "0" leaves it off. Log
lines never carry key material, invite secrets or tensor data: only server
names, addresses and the step reached.
"""

from __future__ import annotations

import os
import sys
import time

_T0 = time.monotonic()


def enabled() -> bool:
    return os.environ.get("REFLOP_DEBUG", "") not in ("", "0")


def log(msg: str) -> None:
    if enabled():
        sys.stderr.write(f"reflop[debug +{time.monotonic() - _T0:.3f}s]: {msg}\n")
        sys.stderr.flush()
