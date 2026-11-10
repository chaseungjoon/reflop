# SPDX-License-Identifier: Apache-2.0
"""Isolation probes (PLAN.md §17.5). TEST BUILDS ONLY.

reflopd embeds this file only when built with `-tags reflop_probes`; release
builds do not contain it. Each probe attempts something the sandbox must
prevent and returns 0 when the attempt FAILED (the sandbox held), 1 when it
succeeded (a sandbox hole), 2 on a probe error.
"""

from __future__ import annotations

import ctypes
import os
import socket
import sys


def _tcp():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.close()


def _udp():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.sendto(b"x", ("127.0.0.1", 9))


def _unix_new():
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.close()


def _connect_loopback():
    socket.create_connection(("127.0.0.1", 22), timeout=1)


def _read(path):
    def f():
        with open(path, "rb") as fh:
            fh.read(1)
    return f


def _listdir(path):
    def f():
        os.listdir(path)
    return f


def _write(path):
    def f():
        with open(path, "wb") as fh:
            fh.write(b"x")
    return f


def _exec():
    os.execv("/bin/true", ["/bin/true"])


def _fork():
    pid = os.fork()
    if pid == 0:
        os._exit(0)
    os.waitpid(pid, 0)


def _ptrace_parent():
    libc = ctypes.CDLL(None, use_errno=True)
    PTRACE_ATTACH = 16
    if libc.ptrace(PTRACE_ATTACH, os.getppid(), None, None) != 0:
        raise OSError(ctypes.get_errno(), "ptrace failed")


def _kill_parent():
    os.kill(os.getppid(), 0)


def _kill_init():
    os.kill(1, 0)


def _mount():
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.mount(b"none", b"/tmp", b"tmpfs", 0, None) != 0:
        raise OSError(ctypes.get_errno(), "mount failed")


def _unshare():
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.unshare(0x10000000) != 0:  # CLONE_NEWUSER
        raise OSError(ctypes.get_errno(), "unshare failed")


def _setuid0():
    os.setuid(0)


def _dev(path):
    def f():
        fd = os.open(path, os.O_RDONLY)
        os.close(fd)
    return f


PROBES = {
    "tcp_socket": _tcp,
    "udp_socket": _udp,
    "unix_socket": _unix_new,
    "connect_loopback": _connect_loopback,
    "read_etc_shadow": _read("/etc/shadow"),
    "read_etc_passwd": _read("/etc/passwd"),
    "list_root": _listdir("/"),
    "write_tmp": _write("/tmp/reflop-probe"),
    "write_var_tmp": _write("/var/tmp/reflop-probe"),
    "exec": _exec,
    "fork": _fork,
    "ptrace_parent": _ptrace_parent,
    "signal_parent": _kill_parent,
    "signal_init": _kill_init,
    "mount": _mount,
    "unshare": _unshare,
    "setuid0": _setuid0,
    "open_dev_mem": _dev("/dev/mem"),
    "open_dev_sda": _dev("/dev/sda"),
    "open_dev_input": _dev("/dev/input/event0"),
}


def run(name: str) -> int:
    # read:<path>, list:<path>, write:<path> probe arbitrary locations
    # (the test passes the Owner's home, ~/.ssh, other workers' scratch …).
    if name.startswith(("read:", "list:", "write:")):
        kind, _, path = name.partition(":")
        fn = {"read": _read, "list": _listdir, "write": _write}[kind](path)
    else:
        fn = PROBES.get(name)
        if fn is None:
            sys.stderr.write(f"unknown probe {name}\n")
            return 2
    try:
        fn()
    except (OSError, PermissionError) as e:
        sys.stderr.write(f"probe {name}: blocked ({e})\n")
        return 0
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"probe {name}: blocked ({type(e).__name__}: {e})\n")
        return 0
    sys.stderr.write(f"probe {name}: SUCCEEDED (sandbox hole)\n")
    return 1
