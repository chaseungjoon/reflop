# SPDX-License-Identifier: Apache-2.0
"""reflop-worker entry point (PLAN.md §4.3 start-up order).

1. Started by `reflopd __sandbox-exec` with a minimal environment, after
   no_new_privs and Landlock were applied (so the filesystem is already
   confined for every thread).
2. Import the runtime, initialize the device, warm up kernels.
3. Install the seccomp filter (all threads, TSYNC).
4. Report ready, then serve requests on the inherited control socket.
"""

from __future__ import annotations

import os
import socket
import sys
import warnings

FD_CONTROL, FD_PASS = 3, 4


def _log(msg: str) -> None:
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def warm_up(device) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.init()
        torch.cuda.set_device(device)
    a = torch.ones(64, 64, device=device)
    b = (a @ a).relu_().sum()  # cuBLAS
    if device.type == "cuda":
        x = torch.ones(1, 3, 8, 8, device=device)
        w = torch.ones(4, 3, 3, 3, device=device)
        torch.nn.functional.conv2d(x, w)  # cuDNN
        torch.nn.functional.scaled_dot_product_attention(
            torch.ones(1, 1, 8, 16, device=device), torch.ones(1, 1, 8, 16, device=device),
            torch.ones(1, 1, 8, 16, device=device))
        torch.randn(4, device=device)
        torch.cuda.synchronize(device)
    float(b)


def main() -> None:
    warnings.filterwarnings("ignore", message="The given buffer is not writable")
    probe = os.environ.get("REFLOP_TEST_PROBE") or None

    ctrl = socket.socket(fileno=FD_CONTROL)
    fdpass = socket.socket(fileno=FD_PASS)
    from ._pb import compute_pb2 as cpb
    from ._pb import worker_pb2 as wpb
    from .wire import Channel

    ch = Channel(ctrl, 64 << 20)
    first = ch.recv()
    if first.WhichOneof("body") != "init":
        _log("expected WorkerInit")
        sys.exit(2)
    init = first.init
    if init.limits.max_frame_bytes:
        ch.max_frame = int(init.limits.max_frame_bytes)

    from . import sandbox
    from .core import Session, torch_device
    from .errors import FatalError, ReflopError

    def fatal(msg: str, code: int = cpb.ERROR_INTERNAL) -> None:
        try:
            ch.send(wpb.WorkerEnvelope(fatal=cpb.Error(code=code, message=msg[:2000])))
        except OSError:
            pass
        _log(msg)
        os._exit(3)

    try:
        warm_up(torch_device(init.device))
        session = Session(init)
    except Exception as e:  # noqa: BLE001
        fatal(f"worker initialization failed: {e}")
        return

    rep = wpb.SandboxReport(no_new_privs=sandbox.no_new_privs())
    ll, _ = sandbox.landlock_status()
    rep.landlock = ll
    if init.seccomp_filter:
        try:
            sandbox.install_seccomp(init.seccomp_filter)
            rep.seccomp = sandbox.seccomp_mode() == 2
        except OSError as e:
            if init.require_seccomp:
                fatal(f"seccomp could not be applied: {e}")
            rep.note = f"seccomp unavailable: {e}"
    if probe is not None:
        # Test builds only: run one probe under the full sandbox, then exit.
        try:
            from . import _probes  # absent from release builds (§17.5)
        except ImportError:
            _log("isolation probes are not part of this build")
            os._exit(2)
        os._exit(_probes.run(probe))
    ch.send(wpb.WorkerEnvelope(ready=session.ready(rep)))

    def send_client(env: cpb.Envelope) -> None:
        ch.send(wpb.WorkerEnvelope(client=env))

    def recv_fds(n: int) -> tuple[bytes, list[int]]:
        msg, fds, _flags, _addr = socket.recv_fds(fdpass, 256, max(n, 1))
        return msg, list(fds)

    while True:
        try:
            env = ch.recv()
        except EOFError:
            os._exit(0)
        except Exception as e:  # noqa: BLE001 - malformed internal message
            fatal(f"control channel: {e}", cpb.ERROR_PROTOCOL)
        which = env.WhichOneof("body")
        try:
            if which == "client":
                session.handle(env.client, send_client)
            elif which == "dataset_open":
                tag, fds = recv_fds(len(env.dataset_open.shards))
                if tag != env.dataset_open.dataset_id:
                    for fd in fds:
                        os.close(fd)
                    fatal("fd channel out of sync")
                try:
                    session.open_dataset(env.dataset_open, fds)
                except ReflopError as e:
                    _log(f"dataset open failed: {e}")
            elif which == "dataset_close":
                session.close_dataset(env.dataset_close.dataset_id)
            elif which in ("snapshot_write", "snapshot_read"):
                m = env.snapshot_write if which == "snapshot_write" else env.snapshot_read
                tag, fds = recv_fds(1)
                rid = m.request_id
                try:
                    if len(fds) != 1:
                        raise ReflopError(cpb.ERROR_INTERNAL, "missing snapshot descriptor")
                    if which == "snapshot_write":
                        n = session.snapshot_write(m, fds[0])
                        send_client(cpb.Envelope(request_id=rid, end=True,
                                                 snapshot_saved=cpb.SnapshotSaved(bytes=n)))
                    else:
                        session.snapshot_read(m, fds[0])
                        send_client(cpb.Envelope(request_id=rid, end=True, restored=cpb.Restored()))
                except ReflopError as e:
                    send_client(cpb.Envelope(request_id=rid, end=True, error=e.to_pb()))
                except Exception as e:  # noqa: BLE001
                    send_client(cpb.Envelope(request_id=rid, end=True, error=ReflopError(
                        cpb.ERROR_INTERNAL, f"snapshot failed: {e}").to_pb()))
            elif which == "stats_request":
                ch.send(wpb.WorkerEnvelope(stats=session.stats()))
            else:
                fatal(f"unexpected control message {which}", cpb.ERROR_PROTOCOL)
        except FatalError as e:
            fatal(str(e), cpb.ERROR_SESSION_LOST)
        except Exception as e:  # noqa: BLE001 - a bug; keep the worker alive if possible
            _log(f"internal error: {type(e).__name__}: {e}")
            if which == "client":
                send_client(cpb.Envelope(request_id=env.client.request_id, end=True, error=ReflopError(
                    cpb.ERROR_INTERNAL, f"internal error: {type(e).__name__}").to_pb()))
