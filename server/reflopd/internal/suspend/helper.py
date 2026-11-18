# SPDX-License-Identifier: Apache-2.0
"""reflopd's suspend helper (ADR-030). Run by reflopd only, with the pinned
runtime's Python in isolated mode, as a child of reflopd and inside reflopd's
own sandbox:

    python -I -B -c <this file> suspend|resume <pid> <lock-timeout-ms>

It calls four CUDA driver functions on one process: lock and checkpoint
(suspend: the driver moves the process's GPU memory into that process's own
host memory and releases the GPU), or restore and unlock (resume). It reads
no Guest data and receives nothing from a Guest: reflopd passes the operation
and the pid of one of its own workers.

It refuses any pid that is not a reflop worker started by the same reflopd:
the target's parent must be this helper's parent, and its command line must
be the worker's. Exit codes: 0 ok, 2 usage, 3 the driver has no checkpoint
API, 4 not a reflop worker, 5 lock timed out (the process keeps running),
6 another driver error.
"""

import ctypes
import sys

CU_ERRORS = {1: "invalid value", 2: "out of memory", 3: "not initialized", 101: "invalid device",
             201: "invalid context", 401: "illegal state", 801: "not supported", 909: "timeout"}


class LockArgs(ctypes.Structure):
    _fields_ = [("timeoutMs", ctypes.c_uint), ("reserved0", ctypes.c_uint), ("reserved1", ctypes.c_uint64 * 7)]


def fail(code, msg):
    sys.stderr.write(msg + "\n")
    sys.exit(code)


def is_our_worker(pid, parent):
    try:
        with open(f"/proc/{pid}/status") as f:
            ppid = next((int(ln.split()[1]) for ln in f if ln.startswith("PPid:")), -1)
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            cmd = f.read()
    except (OSError, ValueError):
        return False
    return ppid == parent and b"reflop_worker.main" in cmd


def main():
    if len(sys.argv) != 4 or sys.argv[1] not in ("suspend", "resume"):
        fail(2, "usage: suspend|resume <pid> <lock-timeout-ms>")
    op, pid, timeout_ms = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    import os

    if pid <= 1 or not is_our_worker(pid, os.getppid()):
        fail(4, f"pid {pid} is not a reflop worker of this reflopd")
    try:
        cu = ctypes.CDLL("libcuda.so.1")
    except OSError as e:
        fail(3, f"libcuda: {e}")
    names = ["cuCheckpointProcessLock", "cuCheckpointProcessCheckpoint"] if op == "suspend" else \
        ["cuCheckpointProcessRestore", "cuCheckpointProcessUnlock"]
    for name in names:
        if not hasattr(cu, name):
            fail(3, f"the NVIDIA driver has no {name} (driver 550 or newer is needed)")
    for name in names:
        arg = ctypes.byref(LockArgs(timeoutMs=timeout_ms)) if name == "cuCheckpointProcessLock" else None
        rc = getattr(cu, name)(pid, arg)
        if rc != 0:
            if name == "cuCheckpointProcessLock" and rc == 909:
                fail(5, f"{name}: timed out after {timeout_ms} ms (a kernel is still running)")
            if rc == 801:
                fail(3, f"{name}: not supported")
            # A failed checkpoint can leave the process locked; reflopd ends it.
            fail(6, f"{name}: CUDA error {rc} ({CU_ERRORS.get(rc, 'unknown')})")


main()
