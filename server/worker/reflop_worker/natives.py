# SPDX-License-Identifier: Apache-2.0
"""Trusted server implementations of reflop-native operators (PLAN.md §8.5).

The table below maps literal names to functions defined in this file. The
generated op table refers to these names; nothing is looked up from client
data.
"""

from __future__ import annotations

import torch

aten = torch.ops.aten


def is_native(op) -> bool:
    return isinstance(op.fn, tuple) and op.fn[0] == "native"


def _short(name: str) -> str:
    return name.split(".")[1]


def _clone_list(ts):
    return [t.clone() for t in ts]


# ---------------------------------------------------------------- implementations


def dataset_gather(dataset, indices):
    return dataset.gather(indices)


def bounds_check(index, size, sentinel=None):
    return index  # the check itself runs in validators (E2)


def nonzero_count(t):
    return torch.count_nonzero(t).to(torch.int64)


def mask_count(mask):
    return mask.to(torch.bool).sum(dtype=torch.int64)


def randperm_seeded(n, seed, dtype=torch.int64, *, device=None):
    g = torch.Generator(device=device or "cpu")
    g.manual_seed(seed)
    return torch.randperm(n, generator=g, dtype=dtype, device=device or "cpu")


def _scalar_tensor(v, like):
    return torch.tensor(v, dtype=torch.float32, device=like.device)


def fused_adam(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, lr, beta1, beta2,
               weight_decay, eps, amsgrad, maximize, grad_scale=None, found_inf=None, *, _adamw=False):
    p, m, v, mx, st = (_clone_list(params), _clone_list(exp_avgs), _clone_list(exp_avg_sqs),
                       _clone_list(max_exp_avg_sqs), _clone_list(state_steps))
    if not p:
        return p, m, v, mx, st
    f = torch._fused_adamw_ if _adamw else torch._fused_adam_
    f(p, list(grads), m, v, mx, st, lr=lr, beta1=beta1, beta2=beta2, weight_decay=weight_decay, eps=eps,
      amsgrad=amsgrad, maximize=maximize, grad_scale=grad_scale, found_inf=found_inf)
    return p, m, v, mx, st


def fused_adamw(*args, **kwargs):
    return fused_adam(*args, _adamw=True, **kwargs)


def fused_sgd(params, grads, momentum_buffers, weight_decay, momentum, lr, dampening, nesterov, maximize,
              is_first_step, grad_scale=None, found_inf=None):
    p, mb = _clone_list(params), _clone_list(momentum_buffers)
    if p:
        torch._fused_sgd_(p, list(grads), mb, weight_decay=weight_decay, momentum=momentum, lr=lr,
                          dampening=dampening, nesterov=nesterov, maximize=maximize, is_first_step=is_first_step,
                          grad_scale=grad_scale, found_inf=found_inf)
    return p, mb


def fused_adagrad(params, grads, state_sums, state_steps, lr, lr_decay, weight_decay, eps, maximize,
                  grad_scale=None, found_inf=None):
    p, ss, st = _clone_list(params), _clone_list(state_sums), _clone_list(state_steps)
    if p:
        torch._fused_adagrad_(p, list(grads), ss, st, lr=lr, lr_decay=lr_decay, weight_decay=weight_decay,
                              eps=eps, maximize=maximize, grad_scale=grad_scale, found_inf=found_inf)
    return p, ss, st


def amp_unscale_and_check(grads, found_inf, inv_scale):
    g = _clone_list(grads)
    fi = found_inf.clone()
    if g:
        torch._amp_foreach_non_finite_check_and_unscale_(g, fi, inv_scale)
    return g, fi


def amp_update_scale(scale, growth_tracker, found_inf, growth_factor, backoff_factor, growth_interval):
    s, gt = scale.clone(), growth_tracker.clone()
    torch._amp_update_scale_(s, gt, found_inf, growth_factor, backoff_factor, growth_interval)
    return s, gt


IMPL = {
    "dataset_gather": dataset_gather, "bounds_check": bounds_check, "nonzero_count": nonzero_count,
    "mask_count": mask_count, "randperm_seeded": randperm_seeded, "fused_adam": fused_adam,
    "fused_adamw": fused_adamw, "fused_sgd": fused_sgd, "fused_adagrad": fused_adagrad,
    "amp_unscale_and_check": amp_unscale_and_check, "amp_update_scale": amp_update_scale,
}


def call(name: str, pos, kw, device):
    short = _short(name)
    f = IMPL[short]
    if short == "randperm_seeded":
        kw = dict(kw, device=device)
    return f(*pos, **kw)


# ---------------------------------------------------------------- meta


def _meta_like(ts):
    return [torch.empty_like(t, device="meta") for t in ts]


def meta(name: str, pos, kw):
    short = _short(name)
    a = list(pos)
    if short == "dataset_gather":
        ds, idx = a
        if idx.dim() != 1:
            raise ValueError("indices must be 1-D")
        return torch.empty((idx.shape[0], *ds.shape[1:]), dtype=ds.dtype, device="meta")
    if short == "bounds_check":
        return torch.empty_like(a[0], device="meta")
    if short in ("nonzero_count", "mask_count"):
        return torch.empty((), dtype=torch.int64, device="meta")
    if short == "randperm_seeded":
        n, _seed = a[0], a[1]
        dt = a[2] if len(a) > 2 else torch.int64
        if n < 0:
            raise ValueError("n must be non-negative")
        return torch.empty((n,), dtype=dt, device="meta")
    if short in ("fused_adam", "fused_adamw"):
        params, grads, m, v, mx, st = a[:6]
        _same_lists(params, grads, m, v, st)
        return _meta_like(params), _meta_like(m), _meta_like(v), _meta_like(mx), _meta_like(st)
    if short == "fused_sgd":
        params, grads, mb = a[:3]
        _same_lists(params, grads)
        return _meta_like(params), _meta_like(mb)
    if short == "fused_adagrad":
        params, grads, ss, st = a[:4]
        _same_lists(params, grads, ss, st)
        return _meta_like(params), _meta_like(ss), _meta_like(st)
    if short == "amp_unscale_and_check":
        grads, fi, _inv = a
        return _meta_like(grads), torch.empty_like(fi, device="meta")
    if short == "amp_update_scale":
        return torch.empty_like(a[0], device="meta"), torch.empty_like(a[1], device="meta")
    raise NotImplementedError(name)


def _same_lists(*lists):
    n = len(lists[0])
    for lst in lists:
        if len(lst) != n:
            raise ValueError("tensor lists must have equal lengths")
    for i in range(n):
        shapes = {tuple(lst[i].shape) for lst in lists[:2]}
        if len(shapes) != 1:
            raise ValueError("param and grad shapes differ")
