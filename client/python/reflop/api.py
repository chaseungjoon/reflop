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
           allow_local_fallback: bool = True) -> torch.device:
    """Connect (once) and return the remote device.

    server: a name from `reflop pair` (default: the default server).
    kind:   "auto" (GPU if the server has one), "cuda" or "cpu" (server CPU).
    local:  run the reflop worker in-process instead ("cpu" or "cuda"); for
            tests and diagnostics only, gives no isolation.
    """
    register()
    from . import natives

    natives.register()
    try:
        _session.by_index(index)
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
    _open_session(conn, index, allow_local_fallback)
    return torch_device(index)


def step_boundary() -> None:
    """Flush the trace (keeps traces step-aligned → high graph cache hits)."""
    for s in list(_session._sessions.values()):
        s.flush()


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
    """Override placement: "auto" (default), "remote", "local", "remote-cpu"."""
    from . import placement as pl

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


def stats() -> dict:
    """Per-session counters (flushes, registrations, executes, bytes)."""
    return {i: dict(s.stats) for i, s in _session._sessions.items()}
