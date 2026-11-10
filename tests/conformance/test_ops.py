# SPDX-License-Identifier: Apache-2.0
"""Conformance (PLAN.md §17.2): PyTorch calls on reflop tensors (in-process
worker) must match plain CPU results. Edge cases: zero-size dims, size-1
broadcast, non-contiguous inputs via views, in-place ops on views, ints,
bools and half precision."""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() and os.environ.get("REFLOP_TEST_DEVICE", "cuda") == "cuda" else "cpu"


@pytest.fixture(scope="module")
def dev():
    import reflop

    d = reflop.device(index=3, local=DEVICE)
    yield d


def g(*shape, dtype=torch.float32, seed=0, low=-2.0, high=2.0):
    gen = torch.Generator().manual_seed(seed + sum(shape))
    if dtype.is_floating_point:
        return (torch.rand(shape, generator=gen) * (high - low) + low).to(dtype)
    if dtype == torch.bool:
        return torch.rand(shape, generator=gen) > 0.5
    return torch.randint(int(low), int(high) + 1, shape, generator=gen, dtype=dtype)


def _close(a, b, tol):
    if isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _close(x, y, tol)
        return
    if not isinstance(a, torch.Tensor):
        assert a == b or abs(a - b) <= tol
        return
    a = a.detach().cpu()
    assert a.shape == b.shape, (a.shape, b.shape)
    assert a.dtype == b.dtype, (a.dtype, b.dtype)
    if a.dtype.is_floating_point:
        torch.testing.assert_close(a, b, atol=tol, rtol=tol, equal_nan=True)
    else:
        assert torch.equal(a, b)


CASES = {
    # unary
    "exp": (lambda t: t.exp(), [g(5, 7)]),
    "sigmoid_tanh": (lambda t: t.sigmoid() * t.tanh(), [g(4, 4)]),
    "log1p_abs": (lambda t: t.abs().log1p(), [g(9)]),
    "rsqrt": (lambda t: (t.abs() + 0.1).rsqrt(), [g(3, 5)]),
    "round_trunc": (lambda t: t.round() + t.trunc() + t.floor() + t.ceil(), [g(6)]),
    "erf_gelu_silu": (lambda t: F.gelu(t) + F.silu(t) + torch.erf(t) + F.gelu(t, approximate="tanh"), [g(8)]),
    # binary / broadcasting / scalars
    "broadcast": (lambda a, b: a * b + a / (b.abs() + 1) - 2.5, [g(4, 1, 3), g(1, 5, 3)]),
    "pow": (lambda a: a.abs() ** 1.5 + 2 ** a, [g(5)]),
    "minmax": (lambda a, b: torch.maximum(a, b) - torch.minimum(a, b), [g(3, 3), g(3, 3, seed=1)]),
    "where": (lambda a, b: torch.where(a > 0, a, b), [g(4, 4), g(4, 4, seed=2)]),
    "clamp": (lambda a: a.clamp(-1, 0.5) + a.clamp(min=0.2), [g(10)]),
    "int_ops": (lambda a, b: (a * b + a // (b.abs() + 1) - a % 3, a & b, a | b, a ^ b), [g(6, dtype=torch.int64, low=-9, high=9), g(6, dtype=torch.int64, seed=3, low=-9, high=9)]),
    "bool_ops": (lambda a, b: (a & b, a | b, ~a, a.logical_xor(b)), [g(7, dtype=torch.bool), g(7, dtype=torch.bool, seed=1)]),
    "compare": (lambda a, b: (a < b, a >= b, a == a, a != b), [g(5), g(5, seed=4)]),
    # reductions
    "sum_mean": (lambda a: (a.sum(), a.mean(1), a.sum((0, 2), keepdim=True)), [g(3, 4, 5)]),
    "var_std": (lambda a: (a.var(0), a.std(1, correction=0)), [g(6, 5)]),
    "amax_argmax": (lambda a: (a.amax(1), a.argmax(0), a.max(1).values, a.min(0).indices), [g(5, 7)]),
    "prod_cumsum": (lambda a: (a.prod(1), a.cumsum(0), a.cumprod(1)), [g(3, 4, low=0.5, high=1.5)]),
    "any_all": (lambda a: (a.any(), a.all(1)), [g(4, 5, dtype=torch.bool)]),
    "logsumexp_norm": (lambda a: (a.logsumexp(1), a.norm(dim=1), torch.linalg.vector_norm(a, 3)), [g(4, 6)]),
    "sort_topk": (lambda a: (a.sort(1).values, a.topk(3, 1).values, a.sort(0, descending=True).indices), [g(5, 6)]),
    # matmul family
    "mm_bmm": (lambda a, b, c: (a @ b, torch.bmm(c, c.transpose(1, 2)), torch.addmm(b[0], a, b)), [g(4, 6), g(6, 6), g(2, 3, 5)]),
    "linear_einsum": (lambda x, w: (F.linear(x, w), torch.einsum("bi,oi->bo", x, w)), [g(5, 8), g(3, 8)]),
    # conv / pool / norm
    "conv2d": (lambda x, w: F.conv2d(x, w, padding=1, stride=2), [g(2, 3, 9, 9), g(4, 3, 3, 3)]),
    "conv1d_groups": (lambda x, w: F.conv1d(x, w, groups=2), [g(2, 4, 10), g(6, 2, 3)]),
    "conv_transpose": (lambda x, w: F.conv_transpose2d(x, w, stride=2), [g(1, 2, 4, 4), g(2, 3, 3, 3)]),
    "pools": (lambda x: (F.max_pool2d(x, 2), F.avg_pool2d(x, 3, 1, 1), F.adaptive_avg_pool2d(x, 2)), [g(2, 3, 8, 8)]),
    "layer_group_norm": (lambda x: (F.layer_norm(x, x.shape[-1:]), F.group_norm(x, 2)), [g(3, 4, 6)]),
    "batch_norm_eval": (lambda x, m, v: F.batch_norm(x, m, v.abs() + 0.1, training=False), [g(4, 3, 5), g(3), g(3)]),
    # softmax / losses
    "softmax": (lambda x: (x.softmax(-1), F.log_softmax(x, 0)), [g(4, 7)]),
    "cross_entropy": (lambda x, y: (F.cross_entropy(x, y), F.cross_entropy(x, y, reduction="none")), [g(6, 5), torch.tensor([0, 4, 2, 1, 3, 0])]),
    "losses": (lambda a, b: (F.mse_loss(a, b), F.smooth_l1_loss(a, b), F.binary_cross_entropy_with_logits(a, b.sigmoid()), F.huber_loss(a, b)), [g(10), g(10, seed=5)]),
    "nll_ignore": (lambda x, y: F.nll_loss(x.log_softmax(1), y, ignore_index=-100), [g(4, 3), torch.tensor([0, -100, 2, 1])]),
    # shape / views
    "views": (lambda a: (a.view(6, 4), a.permute(2, 0, 1), a.transpose(0, 1).contiguous(), a.flatten(1), a[:, 1:3, ::2], a[1]), [g(2, 3, 4)]),
    "noncontig_math": (lambda a: (a.t() * 2).sum(0) + a[::2].mean(), [g(6, 5)]),
    "cat_stack_split": (lambda a, b: (torch.cat([a, b], 1), torch.stack([a, b]), a.split([1, 2], 1), b.chunk(3, 1)), [g(2, 3), g(2, 3, seed=6)]),
    "expand_repeat": (lambda a: (a.expand(3, 4, 5), a.repeat(2, 1, 3), a.flip(-1), a.roll(2, -1)), [g(1, 4, 5)]),
    "tril_triu": (lambda a: (a.tril(), a.triu(1), torch.diag_embed(a[0])), [g(4, 4)]),
    "zero_size": (lambda a: (a.sum(0), a * 2, torch.cat([a, a])), [g(0, 3)]),
    # indexing
    "gather_scatter": (lambda a, i: (a.gather(1, i), torch.zeros(3, 5).scatter_add(1, i, a[:, :2]), a.index_select(1, i[0])), [g(3, 5), torch.tensor([[0, 4], [1, 1], [3, 2]])]),
    "advanced_index": (lambda a, i: (a[i], a[:, i], a[i, i]), [g(5, 5), torch.tensor([4, 0, 2])]),
    "masked": (lambda a, m: (a.masked_fill(m, 0.0), a[m], a.masked_select(m)), [g(4, 4), g(4, 4, dtype=torch.bool, seed=9)]),
    "embedding": (lambda w, i: F.embedding(i, w, padding_idx=0), [g(10, 4), torch.tensor([[1, 0, 9], [3, 3, 2]])]),
    "nonzero": (lambda a: (a > 0).nonzero(), [g(4, 5)]),
    # creation (on device)
    "factories": (None, []),
    # in-place and views
    "inplace_view": (None, []),
    # dtype conversions
    "to_dtype": (lambda a: (a.to(torch.float16).float(), a.to(torch.bfloat16).float(), a.to(torch.int32), a.double()), [g(7)]),
    "half_math": (lambda a: (a * a).sum(), [g(16, dtype=torch.float16)]),
    # upsample / pad / grid
    "interpolate": (lambda x: (F.interpolate(x, scale_factor=2, mode="nearest"), F.interpolate(x, size=(5, 7), mode="bilinear", align_corners=False)), [g(1, 2, 3, 4)]),
    "pad": (lambda x: (F.pad(x, (1, 2, 0, 1)), F.pad(x, (1, 1, 1, 1), mode="reflect"), F.pad(x, (2, 2, 0, 0), mode="replicate")), [g(1, 2, 4, 4)]),
    "sdpa": (lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True), [g(2, 2, 5, 8), g(2, 2, 5, 8, seed=1), g(2, 2, 5, 8, seed=2)]),
}


def _run(dev, fn, args):
    local = fn(*[a.clone() for a in args])
    remote = fn(*[a.to(dev) for a in args])
    return remote, local


@pytest.mark.parametrize("name", [k for k, v in CASES.items() if v[0] is not None])
def test_op(dev, name):
    fn, args = CASES[name]
    remote, local = _run(dev, fn, args)
    tol = 2e-2 if name in ("half_math", "to_dtype") else (2e-3 if DEVICE == "cuda" else 1e-5)
    if DEVICE == "cuda" and name in ("conv2d", "conv1d_groups", "conv_transpose", "mm_bmm", "linear_einsum", "sdpa"):
        tol = 2e-2  # TF32 / reduced-precision kernels
    _close(remote, local, tol)


def test_factories(dev):
    _close(torch.zeros(3, 4, device=dev), torch.zeros(3, 4), 0)
    _close(torch.full((2, 2), 7.0, device=dev), torch.full((2, 2), 7.0), 0)
    _close(torch.arange(10, device=dev), torch.arange(10), 0)
    _close(torch.linspace(0, 1, 5, device=dev), torch.linspace(0, 1, 5), 1e-6)
    _close(torch.eye(3, device=dev), torch.eye(3), 0)
    _close(torch.ones(2, 3, dtype=torch.int64, device=dev), torch.ones(2, 3, dtype=torch.int64), 0)
    e = torch.empty(64, device=dev)
    assert e.cpu().abs().sum() == 0  # UNINIT → zeros (I13)


def test_inplace_on_views(dev):
    a = g(4, 5)
    r = a.clone().to(dev)
    a[1:3] *= 2
    r[1:3] *= 2
    a[:, 0] = 7
    r[:, 0] = 7
    a.t()[2].add_(1)
    r.t()[2].add_(1)
    a.view(-1)[::3].zero_()
    r.view(-1)[::3].zero_()
    _close(r, a, 1e-6)


def test_inplace_ops_and_out(dev):
    a = g(3, 3)
    r = a.to(dev)
    r.mul_(2).add_(1).clamp_(max=2)
    a.mul_(2).add_(1).clamp_(max=2)
    _close(r, a, 1e-6)
    o = torch.empty(3, 3, device=dev)
    torch.add(r, r, out=o)
    _close(o, a + a, 1e-6)


def test_autograd_matches(dev):
    w = g(5, 3).requires_grad_(True)
    x = g(4, 5)
    wr = w.detach().clone().to(dev).requires_grad_(True)
    (torch.tanh(x @ w).pow(2).sum()).backward()
    (torch.tanh(x.to(dev) @ wr).pow(2).sum()).backward()
    _close(wr.grad, w.grad, 1e-3 if DEVICE == "cuda" else 1e-5)


def test_item_and_bool_sync(dev):
    a = torch.tensor([1.5, -2.0]).to(dev)
    assert a.sum().item() == -0.5
    assert bool((a > 0).any())
    assert a.tolist() == [1.5, -2.0]
    assert torch.equal(a, a)


def test_inplace_view_ops(dev):
    a = torch.arange(6.0).reshape(2, 3)
    r = a.to(dev)
    for t in (a, r):
        t.transpose_(0, 1)
        t.unsqueeze_(0)
        t.squeeze_(0)
    _close(r, a, 0)
    _close(r * 2, a * 2, 0)


def test_view_reused_across_flush(dev):
    a = torch.arange(12.0).reshape(3, 4)
    r = a.to(dev)
    v = r.t()[1:]
    _close(v, a.t()[1:], 0)  # flush
    _close(v + 1, a.t()[1:] + 1, 0)  # replays the view in a new trace


def test_rnn_batch_first(dev):
    import copy

    torch.manual_seed(0)
    m = torch.nn.LSTM(4, 6, batch_first=True, num_layers=2)
    r = copy.deepcopy(m).to(dev)
    x = g(3, 5, 4)
    _close(r(x.to(dev))[0], m(x)[0], 1e-4 if DEVICE == "cuda" else 1e-5)
