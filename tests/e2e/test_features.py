# SPDX-License-Identifier: Apache-2.0
"""Optimizer, capture and mixed-precision paths match plain PyTorch."""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import reflop

DEVICE = "cuda" if torch.cuda.is_available() and os.environ.get("REFLOP_TEST_DEVICE", "cuda") == "cuda" else "cpu"
X = torch.randn(32, 16, generator=torch.Generator().manual_seed(1))
Y = torch.randint(0, 4, (32,), generator=torch.Generator().manual_seed(2))


@pytest.fixture(scope="module")
def dev():
    return reflop.device(index=5, local=DEVICE)


def mk():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))


def tol():
    return 1e-4 if DEVICE == "cuda" else 1e-6


def test_fused_adamw(dev):
    m, r = mk().to(dev), mk()
    o = torch.optim.AdamW(m.parameters(), 1e-2, fused=True)
    ro = torch.optim.AdamW(r.parameters(), 1e-2)
    for _ in range(3):
        loss = F.cross_entropy(m(X.to(dev)), Y.to(dev))
        o.zero_grad()
        loss.backward()
        o.step()
        rl = F.cross_entropy(r(X), Y)
        ro.zero_grad()
        rl.backward()
        ro.step()
    assert abs(loss.item() - rl.item()) < tol()


def test_capture_with_capturable_adamw(dev):
    m, r = mk().to(dev), mk()
    o = torch.optim.AdamW(m.parameters(), 1e-2, capturable=True)
    ro = torch.optim.AdamW(r.parameters(), 1e-2)

    def step(x, y):
        o.zero_grad(set_to_none=False)
        loss = F.cross_entropy(m(x), y)
        loss.backward()
        o.step()
        return loss

    s = reflop.capture(step)
    x, y = X.to(dev), Y.to(dev)
    for _ in range(6):
        loss = s(x, y)
        rl = F.cross_entropy(r(X), Y)
        ro.zero_grad()
        rl.backward()
        ro.step()
    assert s.replays >= 3
    assert abs(loss.item() - rl.item()) < tol()


def test_capture_refuses_drifting_scalars(dev):
    m = mk().to(dev)
    o = torch.optim.Adam(m.parameters(), 1e-2)

    def step(x, y):
        o.zero_grad(set_to_none=False)
        loss = F.cross_entropy(m(x), y)
        loss.backward()
        o.step()
        return loss

    s = reflop.capture(step)
    with pytest.raises(reflop.CaptureError):
        for _ in range(4):
            s(X.to(dev), Y.to(dev))


def test_autocast_bf16(dev):
    m = mk().to(dev)
    with torch.autocast("reflop", dtype=torch.bfloat16):
        out = m(X.to(dev))
        loss = F.cross_entropy(out, Y.to(dev))
    assert out.dtype == torch.bfloat16 and loss.dtype == torch.float32
    loss.backward()
    assert m[0].weight.grad.dtype == torch.float32


def test_grad_scaler(dev):
    m = mk().to(dev)
    o = torch.optim.SGD(m.parameters(), 0.1)
    sc = torch.amp.GradScaler("reflop")
    first = None
    for _ in range(5):
        with torch.autocast("reflop", dtype=torch.float16):
            loss = F.cross_entropy(m(X.to(dev)), Y.to(dev))
        o.zero_grad()
        sc.scale(loss).backward()
        sc.step(o)
        sc.update()
        first = first if first is not None else loss.item()
    assert loss.item() < first
    assert sc.get_scale() > 0
