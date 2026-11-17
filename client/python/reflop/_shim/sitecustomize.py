# SPDX-License-Identifier: Apache-2.0
# Loaded by `reflop run`: maps "cuda" to the reflop device in this process.
import os

if os.environ.get("REFLOP_SHIM") == "1":
    if os.environ.get("REFLOP_DEBUG", "") not in ("", "0"):
        # A hung program prints nothing (issue #8): `kill -USR1 <pid>` or
        # Ctrl-\ dumps every thread's Python stack and lets it run on.
        import faulthandler
        import signal

        for _name in ("SIGUSR1", "SIGQUIT"):
            _sig = getattr(signal, _name, None)
            if _sig is not None:
                try:
                    faulthandler.register(_sig, all_threads=True, chain=False)
                except (RuntimeError, ValueError, OSError):
                    pass
    try:
        from reflop import shim

        shim.install()
    except Exception as e:  # noqa: BLE001
        import sys

        sys.stderr.write(f"reflop: shim not installed: {e}\n")
