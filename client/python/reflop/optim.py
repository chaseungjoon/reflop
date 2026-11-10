# SPDX-License-Identifier: Apache-2.0
"""Fused optimizer support: PyTorch's `_fused_*_` kernels become one
functional reflop-native node each (PLAN.md §11.3)."""

from __future__ import annotations

import torch

from .natives import op


def fused(sess, func, vals):
    """Functionalize aten._fused_adam(w)_ / _fused_sgd_ / _fused_adagrad_."""
    from .tensor import _assign, _rebind, dispatch

    name = str(func)
    if name in ("aten._fused_adam_.default", "aten._fused_adamw_.default"):
        params, grads, m, v, mx, steps, lr, b1, b2, wd, eps, ams, mxz, gs, fi = vals
        if isinstance(lr, torch.Tensor):
            lr = float(lr.item())
        f = op("fused_adamw" if "adamw" in name else "fused_adam")
        out = dispatch(sess, f, tuple([params, grads, m, v, mx, steps, lr, b1, b2, wd, eps, ams, mxz]),
                       {"grad_scale": gs, "found_inf": fi})
        for targets, results in zip((params, m, v, mx, steps), out):
            for t, r in zip(targets, results):
                _assign(sess, t, r)
        return None
    if name == "aten._fused_sgd_.default":
        params, grads, mb, wd, mom, lr, damp, nest, mxz, first, gs, fi = vals
        if isinstance(lr, torch.Tensor):
            lr = float(lr.item())
        out = dispatch(sess, op("fused_sgd"), tuple([params, grads, mb, wd, mom, lr, damp, nest, mxz, first]),
                       {"grad_scale": gs, "found_inf": fi})
        for targets, results in zip((params, mb), out):
            for t, r in zip(targets, results):
                _assign(sess, t, r)
        return None
    if name == "aten._fused_adagrad_.default":
        params, grads, ss, steps, lr, lrd, wd, eps, mxz, gs, fi = vals
        out = dispatch(sess, op("fused_adagrad"), tuple([params, grads, ss, steps, lr, lrd, wd, eps, mxz]),
                       {"grad_scale": gs, "found_inf": fi})
        for targets, results in zip((params, ss, steps), out):
            for t, r in zip(targets, results):
                _assign(sess, t, r)
        return None
    from .trace import UnsupportedOperatorError

    _ = _rebind
    raise UnsupportedOperatorError(f"{name}: unsupported fused optimizer")
