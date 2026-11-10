# SPDX-License-Identifier: Apache-2.0
"""`reflop verify`: cross-check small graphs against local CPU execution
(diagnostics only; results are trusted by design, §12)."""

from __future__ import annotations

import copy

import torch


def verify(server: str | None = None) -> bool:
    import reflop

    dev = reflop.device(server=server)
    torch.manual_seed(0)
    ok = True

    def check(name, a, b, tol=1e-3):
        nonlocal ok
        err = (a.detach().cpu().double() - b.detach().cpu().double()).abs().max().item()
        good = err <= tol * max(1.0, b.detach().abs().max().item())
        ok &= good
        print(f"  {'ok  ' if good else 'FAIL'} {name:28} max abs error {err:.2e}")

    x = torch.randn(64, 32)
    w = torch.randn(32, 16)
    check("matmul", (x.to(dev) @ w.to(dev)), x @ w)
    check("softmax", torch.softmax(x.to(dev), -1), torch.softmax(x, -1))
    check("conv2d", torch.nn.functional.conv2d(torch.randn(2, 3, 16, 16).to(dev) * 0 + 1, torch.ones(4, 3, 3, 3).to(dev)),
          torch.nn.functional.conv2d(torch.ones(2, 3, 16, 16), torch.ones(4, 3, 3, 3)))
    m = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.GELU(), torch.nn.Linear(64, 8))
    mr = copy.deepcopy(m).to(dev)
    y = torch.randint(0, 8, (64,))
    loss_l = torch.nn.functional.cross_entropy(m(x), y)
    loss_l.backward()
    loss_r = torch.nn.functional.cross_entropy(mr(x.to(dev)), y.to(dev))
    loss_r.backward()
    check("MLP loss", loss_r, loss_l)
    check("MLP grad", mr[0].weight.grad, m[0].weight.grad)
    idx = torch.randint(0, 64, (100,))
    check("embedding", torch.nn.functional.embedding(idx.to(dev), x.to(dev)), torch.nn.functional.embedding(idx, x))
    print("verify:", "all checks passed" if ok else "some checks FAILED")
    return ok
