# SPDX-License-Identifier: Apache-2.0
"""BatchNorm running statistics match plain PyTorch.

Regression: native_batch_norm's schema does not mark running_mean and
running_var as written, so the client recorded it as a pure function. The
statistics only changed through the kernel's in-place side effect on the
server, and not at all when the output was dropped (a BN recalibration loop):
num_batches_tracked went up, running_mean and running_var stayed.
"""

from __future__ import annotations

import contextlib
import copy
import os

import pytest
import reflop
import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() and os.environ.get("REFLOP_TEST_DEVICE", "cuda") == "cuda" else "cpu"
TOL = 1e-4 if DEVICE == "cuda" else 1e-5


@pytest.fixture(scope="module")
def dev():
    return reflop.device(index=9, local=DEVICE)


def _bns(m):
    return [x for x in m.modules() if isinstance(x, nn.modules.batchnorm._BatchNorm)]


def _same_stats(remote, local):
    for r, lo in zip(_bns(remote), _bns(local), strict=True):
        if lo.running_mean is None:
            assert r.running_mean is None and r.running_var is None
            continue
        torch.testing.assert_close(r.running_mean.cpu(), lo.running_mean, atol=TOL, rtol=TOL)
        torch.testing.assert_close(r.running_var.cpu(), lo.running_var, atol=TOL, rtol=TOL)
        assert int(r.num_batches_tracked.cpu()) == int(lo.num_batches_tracked)


def _model(kind, momentum=0.1, affine=True, stack=False, track=True):
    torch.manual_seed(0)
    cls = {"1d": nn.BatchNorm1d, "2d": nn.BatchNorm2d, "3d": nn.BatchNorm3d}[kind]
    bn = cls(8, momentum=momentum, affine=affine, track_running_stats=track)
    if stack:
        return nn.Sequential(nn.Linear(8, 8), bn, nn.ReLU(), nn.Linear(8, 8), nn.BatchNorm1d(8))
    return bn


def _batch(kind, g):
    shape = {"1d": (16, 8), "2d": (16, 8, 4, 4), "3d": (4, 8, 2, 3, 3)}[kind]
    return torch.randn(*shape, generator=g) * 3 + 1


CASES = {
    "dropped": {},
    "kept": {"keep": True},
    "no_grad": {"no_grad": True},
    "cumulative_momentum": {"momentum": None},
    "bn2d": {"kind": "2d"},
    "bn3d": {"kind": "3d"},
    "no_affine": {"affine": False},
    "step_boundary": {"boundary": True},
    "stacked": {"stack": True},
    "no_grad_kept": {"no_grad": True, "keep": True},
    "untracked": {"track": False},
}


@pytest.mark.parametrize("case", list(CASES))
def test_running_stats_match_pytorch(dev, case):
    o = dict(kind="1d", momentum=0.1, affine=True, stack=False, track=True, keep=False, no_grad=False,
             boundary=False)
    o.update(CASES[case])
    local = _model(o["kind"], o["momentum"], o["affine"], o["stack"], o["track"])
    remote = copy.deepcopy(local).to(dev)
    g = torch.Generator().manual_seed(1)
    held = []
    for _ in range(5):
        x = _batch(o["kind"], g)
        with torch.no_grad() if o["no_grad"] else contextlib.nullcontext():
            lo = local(x)
            out = remote(x.to(dev))
        if o["keep"]:
            held.append((out, lo))
        del out
        if o["boundary"]:
            reflop.step_boundary()
    _same_stats(remote, local)
    for out, lo in held:
        torch.testing.assert_close(out.detach().cpu(), lo.detach(), atol=TOL, rtol=TOL)


def test_recalibration_loop_then_eval(dev):
    # The reported use: reset the statistics, run batches for their side
    # effect only, then evaluate with the new statistics.
    local = _model("2d")
    remote = copy.deepcopy(local).to(dev)
    for m in (local, remote):
        m.reset_running_stats()
        m.momentum = None
    g = torch.Generator().manual_seed(2)
    with torch.no_grad():
        for _ in range(4):
            x = _batch("2d", g)
            local(x)
            remote(x.to(dev))
    _same_stats(remote, local)
    local.eval()
    remote.eval()
    x = _batch("2d", g)
    with torch.no_grad():
        torch.testing.assert_close(remote(x.to(dev)).cpu(), local(x), atol=TOL, rtol=TOL)


def test_training_outputs_and_gradients_match(dev):
    local = _model("1d", stack=True)
    remote = copy.deepcopy(local).to(dev)
    g = torch.Generator().manual_seed(3)
    for _ in range(3):
        x = _batch("1d", g)
        for m in (local, remote):
            m.zero_grad()
        lo = local(x)
        out = remote(x.to(dev))
        (lo * lo).sum().backward()
        (out * out).sum().backward()
        torch.testing.assert_close(out.detach().cpu(), lo.detach(), atol=TOL, rtol=TOL)
        for pr, pl in zip(remote.parameters(), local.parameters(), strict=True):
            torch.testing.assert_close(pr.grad.cpu(), pl.grad, atol=TOL, rtol=TOL)
    _same_stats(remote, local)


def test_eval_mode_gradients_and_frozen_stats(dev):
    # Fine-tuning with frozen BN: eval mode, gradients still flow through it
    # (now through _native_batch_norm_legit_no_training).
    local = _model("2d")
    for _ in range(2):
        local(_batch("2d", torch.Generator().manual_seed(9)))
    local.eval()
    remote = copy.deepcopy(local).to(dev)
    x = _batch("2d", torch.Generator().manual_seed(4)).requires_grad_()
    xr = x.detach().to(dev).requires_grad_()
    lo = local(x)
    out = remote(xr)
    (lo * lo).sum().backward()
    (out * out).sum().backward()
    torch.testing.assert_close(out.detach().cpu(), lo.detach(), atol=TOL, rtol=TOL)
    torch.testing.assert_close(xr.grad.cpu(), x.grad, atol=TOL, rtol=TOL)
    for pr, pl in zip(remote.parameters(), local.parameters(), strict=True):
        torch.testing.assert_close(pr.grad.cpu(), pl.grad, atol=TOL, rtol=TOL)
    _same_stats(remote, local)


def test_captured_step_updates_running_stats(dev):
    torch.manual_seed(0)
    local = nn.Sequential(nn.Linear(8, 8), nn.BatchNorm1d(8))
    remote = copy.deepcopy(local).to(dev)
    opt_l = torch.optim.SGD(local.parameters(), 0.1)
    opt_r = torch.optim.SGD(remote.parameters(), 0.1)

    def step(m, o, x):
        o.zero_grad(set_to_none=False)
        loss = m(x).pow(2).mean()
        loss.backward()
        o.step()
        return loss

    s = reflop.capture(lambda x: step(remote, opt_r, x))
    g = torch.Generator().manual_seed(6)
    for _ in range(6):
        x = _batch("1d", g)
        lo = step(local, opt_l, x)
        out = s(x.to(dev))
        torch.testing.assert_close(out.detach().cpu(), lo.detach(), atol=TOL, rtol=TOL)
    assert s.replays >= 3
    _same_stats(remote, local)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_eval_mode_low_precision_input(dev, dtype):
    local = _model("2d")
    for _ in range(2):
        local(_batch("2d", torch.Generator().manual_seed(8)))
    local.eval()
    remote = copy.deepcopy(local).to(dev)
    x = _batch("2d", torch.Generator().manual_seed(7)).to(dtype)
    with torch.no_grad():
        lo = local.float()(x.float()).to(dtype)  # the kernels compute in float32
        out = remote(x.to(dev))
    assert out.dtype == dtype
    torch.testing.assert_close(out.cpu().float(), lo.float(), atol=2e-2, rtol=2e-2)


def test_switching_to_train_before_backward(dev):
    # save_mean from an eval forward must not alias running_mean: a training
    # step before the backward updates running_mean in place. PyTorch itself
    # differs by device here: the CPU backward reads the running statistics
    # as they are at backward time, the CUDA backward the ones saved by the
    # forward (the same as without the training step).
    local = _model("1d")
    remote = copy.deepcopy(local).to(dev)
    for m in (local, remote):
        m.eval()
    x = _batch("1d", torch.Generator().manual_seed(10))
    if DEVICE == "cuda":
        ref = copy.deepcopy(local)
        xc = x.clone().requires_grad_()
        ref(xc).sum().backward()
        want = xc.grad
    lo = local(x.requires_grad_())
    xr = x.detach().to(dev).requires_grad_()
    out = remote(xr)
    for m, inp in ((local, x.detach()), (remote, x.detach().to(dev))):
        m.train()
        m(inp)
    lo.sum().backward()
    out.sum().backward()
    torch.testing.assert_close(xr.grad.cpu(), want if DEVICE == "cuda" else x.grad, atol=TOL, rtol=TOL)
    _same_stats(remote, local)
