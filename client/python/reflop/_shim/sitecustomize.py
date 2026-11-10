# SPDX-License-Identifier: Apache-2.0
# Loaded by `reflop run`: maps "cuda" to the reflop device in this process.
import os

if os.environ.get("REFLOP_SHIM") == "1":
    try:
        from reflop import shim

        shim.install()
    except Exception as e:  # noqa: BLE001
        import sys

        sys.stderr.write(f"reflop: shim not installed: {e}\n")
