# SPDX-License-Identifier: Apache-2.0
"""Client registration of reflop-native operators as PyTorch custom ops.

`torch.ops.reflop.<name>` behaves like any aten op: on RemoteTensors it is
recorded in the trace; on meta tensors it infers shapes; on CPU tensors it
runs a reference implementation (used by local fallback and tests).
"""

from __future__ import annotations

import torch

from . import _opset_generated as opset

_lib: torch.library.Library | None = None


# Schema strings must match opset/v1/opset.yaml exactly.
_TEXT = {
    "dataset_gather": "dataset_gather(Tensor dataset, Tensor indices) -> Tensor",
    "bounds_check": "bounds_check(Tensor index, int size, int? sentinel=None) -> Tensor",
    "nonzero_count": "nonzero_count(Tensor self) -> Tensor",
    "mask_count": "mask_count(Tensor mask) -> Tensor",
    "randperm_seeded": "randperm_seeded(int n, int seed, ScalarType dtype=long) -> Tensor",
    "fused_adam": "fused_adam(Tensor[] params, Tensor[] grads, Tensor[] exp_avgs, Tensor[] exp_avg_sqs, "
                  "Tensor[] max_exp_avg_sqs, Tensor[] state_steps, float lr, float beta1, float beta2, "
                  "float weight_decay, float eps, bool amsgrad, bool maximize, Tensor? grad_scale=None, "
                  "Tensor? found_inf=None) -> (Tensor[], Tensor[], Tensor[], Tensor[], Tensor[])",
    "fused_adamw": "fused_adamw(Tensor[] params, Tensor[] grads, Tensor[] exp_avgs, Tensor[] exp_avg_sqs, "
                   "Tensor[] max_exp_avg_sqs, Tensor[] state_steps, float lr, float beta1, float beta2, "
                   "float weight_decay, float eps, bool amsgrad, bool maximize, Tensor? grad_scale=None, "
                   "Tensor? found_inf=None) -> (Tensor[], Tensor[], Tensor[], Tensor[], Tensor[])",
    "fused_sgd": "fused_sgd(Tensor[] params, Tensor[] grads, Tensor[] momentum_buffers, float weight_decay, "
                 "float momentum, float lr, float dampening, bool nesterov, bool maximize, bool is_first_step, "
                 "Tensor? grad_scale=None, Tensor? found_inf=None) -> (Tensor[], Tensor[])",
    "fused_adagrad": "fused_adagrad(Tensor[] params, Tensor[] grads, Tensor[] state_sums, Tensor[] state_steps, "
                     "float lr, float lr_decay, float weight_decay, float eps, bool maximize, "
                     "Tensor? grad_scale=None, Tensor? found_inf=None) -> (Tensor[], Tensor[], Tensor[])",
    "amp_unscale_and_check": "amp_unscale_and_check(Tensor[] grads, Tensor found_inf, Tensor inv_scale) "
                             "-> (Tensor[], Tensor)",
    "amp_update_scale": "amp_update_scale(Tensor scale, Tensor growth_tracker, Tensor found_inf, "
                        "float growth_factor, float backoff_factor, int growth_interval) -> (Tensor, Tensor)",
}


def _likes(ts):
    return [torch.empty_like(t) for t in ts]


def _meta_impl(name):
    def f(*a, **k):
        if name == "dataset_gather":
            ds, idx = a
            return ds.new_empty((idx.shape[0], *ds.shape[1:]))
        if name == "bounds_check":
            return torch.empty_like(a[0])
        if name in ("nonzero_count", "mask_count"):
            return a[0].new_empty((), dtype=torch.int64)
        if name == "randperm_seeded":
            dt = a[2] if len(a) > 2 else k.get("dtype", torch.int64)
            return torch.empty((a[0],), dtype=dt, device="meta")
        if name in ("fused_adam", "fused_adamw"):
            return tuple(_likes(a[i]) for i in (0, 2, 3, 4, 5))
        if name == "fused_sgd":
            return _likes(a[0]), _likes(a[2])
        if name == "fused_adagrad":
            return _likes(a[0]), _likes(a[2]), _likes(a[3])
        if name == "amp_unscale_and_check":
            return _likes(a[0]), torch.empty_like(a[1])
        if name == "amp_update_scale":
            return torch.empty_like(a[0]), torch.empty_like(a[1])
        raise NotImplementedError(name)
    return f


def _cpu_impl(name):
    def f(*a, **k):
        if name == "dataset_gather":
            return a[0].index_select(0, a[1].long())
        if name == "bounds_check":
            idx, size = a[0], a[1]
            sentinel = a[2] if len(a) > 2 else k.get("sentinel")
            ok = (idx >= 0) & (idx < size)
            if sentinel is not None:
                ok |= idx == sentinel
            if not bool(ok.all()):
                raise IndexError("index out of bounds")
            return idx.clone()
        if name == "nonzero_count":
            return torch.count_nonzero(a[0]).to(torch.int64)
        if name == "mask_count":
            return a[0].to(torch.bool).sum(dtype=torch.int64)
        if name == "randperm_seeded":
            g = torch.Generator().manual_seed(a[1])
            return torch.randperm(a[0], generator=g, dtype=a[2] if len(a) > 2 else torch.int64)
        if name in ("fused_adam", "fused_adamw"):
            p, g, m, v, mx, st = [list(x) for x in a[:6]]
            p, m, v, mx, st = ([t.clone() for t in x] for x in (p, m, v, mx, st))
            fn = torch._fused_adamw_ if name == "fused_adamw" else torch._fused_adam_
            lr, b1, b2, wd, eps, ams, mxz = a[6:13]
            if p:
                fn(p, g, m, v, mx, st, lr=lr, beta1=b1, beta2=b2, weight_decay=wd, eps=eps, amsgrad=ams,
                   maximize=mxz, grad_scale=k.get("grad_scale"), found_inf=k.get("found_inf"))
            return p, m, v, mx, st
        raise NotImplementedError(f"reflop::{name} has no CPU reference implementation")
    return f


def register() -> None:
    global _lib
    if _lib is not None:
        return
    _lib = torch.library.Library("reflop", "DEF")
    for name, text in _TEXT.items():
        if f"reflop.{name}.default" not in opset.OPS:
            continue
        _lib.define(text)
        _lib.impl(name, _meta_impl(name), "Meta")
        _lib.impl(name, _cpu_impl(name), "CPU")


def op(name: str):
    register()
    return getattr(torch.ops.reflop, name).default

