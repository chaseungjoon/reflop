# SPDX-License-Identifier: Apache-2.0
"""Post-warm-up sandbox step: install the seccomp filter built by reflopd.

This is the only worker module allowed to use ctypes, and only with the
literal libc symbol names below (static analysis rule W-CTYPES, §17.4).
Nothing here depends on client data.

Landlock and no_new_privs were already applied by `reflopd __sandbox-exec`
before this interpreter started (ADR-004).
"""

from __future__ import annotations

import ctypes
import os
import platform

_SECCOMP_SET_MODE_FILTER = 1
_SECCOMP_FILTER_FLAG_TSYNC = 1
_PR_GET_NO_NEW_PRIVS = 39
_PR_GET_SECCOMP = 21
_SYS_SECCOMP = {"x86_64": 317, "aarch64": 277}


class _SockFprog(ctypes.Structure):
    _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.c_void_p)]


def _libc() -> ctypes.CDLL:
    return ctypes.CDLL(None, use_errno=True)


def no_new_privs() -> bool:
    return _libc().prctl(_PR_GET_NO_NEW_PRIVS, 0, 0, 0, 0) == 1


def seccomp_mode() -> int:
    return _libc().prctl(_PR_GET_SECCOMP, 0, 0, 0, 0)


def landlock_status() -> tuple[bool, int]:
    """Report whether this process runs inside a Landlock domain.

    /proc/self/status has no Landlock field, so probe: a path outside every
    rule (the filesystem root listing) must be denied.
    """
    try:
        os.listdir("/")
    except PermissionError:
        return True, 0
    except OSError:
        return False, 0
    return False, 0


def install_seccomp(program: bytes) -> None:
    """Install a classic-BPF seccomp filter on all threads (TSYNC)."""
    if not program or len(program) % 8 != 0:
        raise ValueError("invalid seccomp program")
    n = len(program) // 8
    if n > 4096:
        raise ValueError("seccomp program too long")
    arch = platform.machine()
    if arch not in _SYS_SECCOMP:
        raise OSError(f"seccomp: unsupported architecture {arch}")
    buf = ctypes.create_string_buffer(program, len(program))
    prog = _SockFprog(n, ctypes.cast(buf, ctypes.c_void_p))
    libc = _libc()
    rc = libc.syscall(_SYS_SECCOMP[arch], _SECCOMP_SET_MODE_FILTER, _SECCOMP_FILTER_FLAG_TSYNC, ctypes.byref(prog))
    if rc != 0:
        err = ctypes.get_errno()
        # With TSYNC, a positive return is the id of a thread that could
        # not be synchronized.
        raise OSError(err, f"seccomp(SET_MODE_FILTER, TSYNC) failed: rc={rc} {os.strerror(err) if err else ''}")
