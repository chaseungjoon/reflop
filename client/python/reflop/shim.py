# SPDX-License-Identifier: Apache-2.0
"""Shim mode (PLAN.md §16.2, best effort): unmodified code that uses "cuda"
runs its tensor math on the reflop device. The explicit API
(`reflop.device()`) is the supported path; this covers the common calls:
`.cuda()`, `.to("cuda")`, `device="cuda"` in factories,
`torch.cuda.is_available()`, and `torch.autocast("cuda")`.
"""

from __future__ import annotations

import torch
from torch.overrides import TorchFunctionMode

_installed = False
_dev: torch.device | None = None
_pending_seed: int | None = None


def _target() -> torch.device:
    global _dev
    if _dev is None:
        import reflop

        _dev = reflop.device()
        if _pending_seed is not None:
            reflop.seed(_pending_seed)
    return _dev


def _seed(seed: int) -> None:
    # Seeding happens before the first CUDA use; connect lazily and apply then.
    global _pending_seed
    from . import session

    if getattr(session._seeding, "active", False):
        # A reseed in progress: the in-process worker (local mode) seeds this
        # same torch, which lands here again and would deadlock the connection.
        return
    if _dev is None:
        _pending_seed = int(seed)
    else:
        import reflop

        reflop.seed(int(seed))


def _map(x):
    if isinstance(x, str) and (x == "cuda" or x.startswith("cuda:")):
        return _target()
    if isinstance(x, torch.device) and x.type == "cuda":
        return _target()
    return x


def _amp_device(device_type):
    """autocast's device type: "cuda" means the reflop device. Autocast is
    per device type, so CUDA autocast would never touch reflop tensors."""
    if device_type == "cuda":
        from .device import BACKEND, register

        register()  # torch.reflop must exist; no connection needed
        return BACKEND
    return device_type


def _bf16_supported(including_emulation: bool = True) -> bool:
    # Asked of the server's device; the real torch.cuda would need local CUDA.
    from . import session
    from ._pb import graph_pb2 as gpb

    sess = session.by_index(_target().index or 0)
    return any(d.device.kind == sess.gdevice.kind and gpb.DTYPE_BFLOAT16 in d.dtypes for d in sess.caps.devices)


def _install_amp() -> None:
    from torch.amp.autocast_mode import autocast

    init = autocast.__init__

    # torch.autocast, torch.amp.autocast and torch.cuda.amp.autocast (a
    # subclass that passes "cuda") all run this __init__.
    def __init__(self, device_type, *args, **kwargs):
        init(self, _amp_device(device_type), *args, **kwargs)

    autocast.__init__ = __init__
    enabled, get_dtype = torch.is_autocast_enabled, torch.get_autocast_dtype

    def is_autocast_enabled(device_type="cuda"):
        # No argument means CUDA (PyTorch's legacy form), so reflop here.
        return enabled(_amp_device(device_type))

    torch.is_autocast_enabled = is_autocast_enabled
    torch.get_autocast_dtype = lambda device_type: get_dtype(_amp_device(device_type))
    torch.cuda.is_bf16_supported = _bf16_supported


class _CudaToReflop(TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if "device" in kwargs:
            kwargs["device"] = _map(kwargs["device"])
        if func in (torch.Tensor.to, torch.nn.Module.to) or getattr(func, "__name__", "") == "to":
            args = tuple(_map(a) for a in args)
        return func(*args, **kwargs)


def install() -> None:
    global _installed
    if _installed:
        return
    _installed = True
    torch.Tensor.cuda = lambda self, device=None, non_blocking=False, **kw: self.to(_target(), non_blocking=non_blocking)
    torch.nn.Module.cuda = lambda self, device=None: self.to(_target())
    torch.cuda.is_available = lambda: True
    torch.cuda.device_count = lambda: 1
    torch.cuda.synchronize = lambda device=None: __import__("reflop").synchronize()
    torch.cuda.manual_seed = torch.cuda.manual_seed_all = _seed
    _install_amp()
    _mode = _CudaToReflop()
    _mode.__enter__()
