# SPDX-License-Identifier: Apache-2.0
"""Shim mode (PLAN.md §16.2, best effort): unmodified code that uses "cuda"
runs its tensor math on the reflop device. The explicit API
(`reflop.device()`) is the supported path; this covers the common calls:
`.cuda()`, `.to("cuda")`, `device="cuda"` in factories, and
`torch.cuda.is_available()`.
"""

from __future__ import annotations

import torch
from torch.overrides import TorchFunctionMode

_installed = False
_dev: torch.device | None = None


def _target() -> torch.device:
    global _dev
    if _dev is None:
        import reflop

        _dev = reflop.device()
    return _dev


def _map(x):
    if isinstance(x, str) and (x == "cuda" or x.startswith("cuda:")):
        return _target()
    if isinstance(x, torch.device) and x.type == "cuda":
        return _target()
    return x


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
    torch.cuda.manual_seed = torch.cuda.manual_seed_all = lambda seed: __import__("reflop").seed(seed)
    _mode = _CudaToReflop()
    _mode.__enter__()
