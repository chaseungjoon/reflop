# SPDX-License-Identifier: Apache-2.0
"""Public Python API (PLAN.md §16.2)."""

from __future__ import annotations

import contextlib
import os

import torch

from . import session as _session
from .device import register, torch_device


def _open_session(conn, index: int, allow_local_fallback: bool):
    from ._pb import graph_pb2 as gpb

    caps = conn.capabilities()
    gdev = conn.gdevice if hasattr(conn, "gdevice") else gpb.Device(kind=gpb.DEVICE_CUDA)
    return _session.Session(conn, index, gdev, caps, allow_local_fallback=allow_local_fallback)


def device(index: int = 0, *, server: str | None = None, kind: str = "auto", local: str | None = None,
           allow_local_fallback: bool = True, checkpoint: bool = False, checkpoint_interval=None) -> torch.device:
    """Connect (once) and return the remote device.

    server: a name from `reflop pair` (default: the default server).
    kind:   "auto" (GPU if the server has one), "cuda" or "cpu" (server CPU).
    local:  run the reflop worker in-process instead ("cpu" or "cuda"); for
            tests and diagnostics only, gives no isolation.
    checkpoint: save the session on the server at step boundaries, at most
            once per interval, so reflop.recover() can continue after the
            worker is lost. The Owner must allow it.
    checkpoint_interval: seconds (or a timedelta) between checkpoints; the
            default and the minimum are the Owner's interval.
    """
    register()
    from . import natives

    natives.register()
    try:
        sess = _session.by_index(index)
        if checkpoint:
            from . import checkpoints

            checkpoints.enable(sess, checkpoint_interval)
        return torch_device(index)
    except KeyError:
        pass
    local = local or os.environ.get("REFLOP_LOCAL") or None
    if local:
        from .connection import LocalConnection

        conn = LocalConnection(local)
    else:
        from .client import connect

        conn = connect(server=server, kind=kind)
    sess = _open_session(conn, index, allow_local_fallback)
    if checkpoint:
        from . import checkpoints

        try:
            checkpoints.enable(sess, checkpoint_interval)
        except BaseException:
            sess.close()
            raise
    return torch_device(index)


def step_boundary() -> None:
    """Flush the trace (keeps traces step-aligned → high graph cache hits).

    With checkpoints on, this is also where they are taken."""
    for s in list(_session._sessions.values()):
        s.flush()
        s.maybe_checkpoint()


def synchronize() -> None:
    _session.synchronize()


def seed(n: int) -> None:
    """Seed the session RNG on the server (reproducible dropout, randn, ...)."""
    _session.seed(n)


def close() -> None:
    for s in list(_session._sessions.values()):
        s.close()


@contextlib.contextmanager
def placement(mode: str):
    """Override placement (PLAN.md §13.3):

    "auto"        default: ops on reflop tensors run remotely; an op that
                  mixes big client tensors with small reflop tensors may run
                  locally (cost model); unsupported ops fall back locally
    "remote"      never run locally: unsupported ops raise
    "local"       ops on reflop tensors run on the client and return client
                  tensors (fetches inputs)
    "remote-cpu"  new tensors and .to(reflop.device()) go to the server CPU
                  (opens a second session on demand)
    """
    from . import _placement as pl

    with pl.override(mode):
        yield


def fetch(obj, path: str | None = None):
    """Download tensors to the client. With `path`, save them as safetensors
    (never pickle). Accepts a tensor, a state_dict, or any dict of tensors."""
    from .tensor import RemoteTensor

    def get(t):
        if isinstance(t, RemoteTensor):
            return t.detach().cpu()
        return t.detach().cpu() if isinstance(t, torch.Tensor) else t

    if isinstance(obj, torch.Tensor):
        out = get(obj)
        if path:
            from safetensors.torch import save_file

            save_file({"tensor": out.contiguous()}, path)
        return out
    if isinstance(obj, dict):
        out = {k: get(v) for k, v in obj.items()}
        if path:
            from safetensors.torch import save_file

            tensors = {k: v.contiguous() for k, v in out.items() if isinstance(v, torch.Tensor)}
            save_file(_dedupe(tensors), path)
        return out
    raise TypeError("reflop.fetch takes a tensor or a dict of tensors")


def _dedupe(tensors: dict) -> dict:
    seen = {}
    out = {}
    for k, v in tensors.items():
        key = (v.data_ptr(), v.numel(), v.dtype) if v.numel() else None
        if key is not None and key in seen:
            v = v.clone()
        seen[key] = k
        out[k] = v
    return out


def _sess(index: int | None):
    return _session.current() if index is None else _session.by_index(index)


def save(name: str, tensors) -> int:
    """Save reflop tensors on the server as the snapshot `name`.

    `tensors` is a dict (nested dicts allowed), for example
    model.state_dict() or optimizer.state_dict()["state"]. Only tensors are
    stored. Returns the snapshot's size in bytes; an existing snapshot with
    the same name is replaced once the new one is safely written."""
    from . import checkpoints

    return checkpoints.save(name, tensors)


def load(name: str, index: int | None = None) -> dict:
    """Load the snapshot `name` into new reflop tensors: {name: tensor}.

    Works in any later session of the same identity, for example
    model.load_state_dict(reflop.load("run1"))."""
    from . import checkpoints

    return checkpoints.load(name, _sess(index))


def snapshots(index: int | None = None) -> list[dict]:
    """Your snapshots on the server: name, bytes, created."""
    from . import checkpoints

    return checkpoints.snapshots(_sess(index))


def delete_snapshot(name: str, index: int | None = None) -> None:
    from . import checkpoints

    checkpoints.delete(name, _sess(index))


def checkpoint_tag(obj, index: int | None = None) -> None:
    """Store a small JSON value (for example {"step": 1200}) with the next
    checkpoints; reflop.recover() returns it."""
    from . import checkpoints

    checkpoints.set_tag(_sess(index), obj)


def recover(index: int | None = None):
    """After SessionLostError: start a new session from the last checkpoint.

    Tensors that existed at the checkpoint have their values from then;
    tensors created later raise SessionLostError when used. Returns the
    reflop.checkpoint_tag() value of that checkpoint (None if none was set).
    Raises ReflopError when there is no checkpoint (none was saved yet, or
    the Owner ended the session on purpose)."""
    from . import checkpoints

    return checkpoints.recover(_sess(index))


def stats() -> dict:
    """Per-session counters (flushes, registrations, executes, bytes)."""
    return {i: dict(s.stats) for i, s in _session._sessions.items()}


def _usage_dict(rep) -> dict:
    from datetime import datetime, timezone

    from .errors import sanitize

    def when(t):
        return datetime.fromtimestamp(t, timezone.utc) if t else None

    return {
        "identity": sanitize(rep.identity_name, 64),
        "since": when(rep.since_unix),
        "device_seconds": rep.device_us / 1e6,
        "by_device": {sanitize(d.device, 80): d.device_us / 1e6 for d in rep.by_device},
        "executes": rep.executes,
        "flops": rep.flops,
        "peak_memory_bytes": rep.peak_memory_bytes,
        "sessions": rep.sessions,
        "session_seconds": rep.session_us / 1e6,
        "bytes_up": rep.bytes_up,
        "bytes_down": rep.bytes_down,
        "first": when(rep.first_unix),
        "last": when(rep.last_unix),
    }


def _usage_on(conn) -> dict:
    from ._pb import compute_pb2 as cpb
    from .errors import ReflopError, sanitize

    if "usage" not in conn.caps.capabilities:
        ver = sanitize(conn.caps.runtime_info.reflopd_version, 40) or "unknown"
        raise ReflopError(f"this server (reflopd {ver}) does not report usage; it needs reflopd 0.1.4 or newer")
    return _usage_dict(conn.call(cpb.Envelope(get_usage=cpb.GetUsage()), timeout=30).usage_report)


def usage(server: str | None = None) -> dict:
    """Compute you (your identity) have used on a server, cumulative.

    ``device_seconds`` is the main measure: time the server's devices spent
    running your executes. ``flops`` is the server's estimate."""
    from .client import admin

    c = admin(server)
    try:
        return _usage_on(c)
    finally:
        c.close()
