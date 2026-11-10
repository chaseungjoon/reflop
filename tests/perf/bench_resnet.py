#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""§17.6 throughput: ResNet-18 / CIFAR-10 shapes, reflop vs local GPU.

    python tests/perf/bench_resnet.py [--local-worker] [--steps N] [--batch B]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "client/python"), str(ROOT / "server/worker")]

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torchvision.models import resnet18  # noqa: E402


def step_fn(model, opt, x, y):
    opt.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    opt.step()
    return loss


def bench(dev, steps, batch, label, sync, captured=False):
    torch.manual_seed(0)
    model = resnet18(num_classes=10).to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    x = torch.randn(batch, 3, 32, 32).to(dev)
    y = torch.randint(0, 10, (batch,)).to(dev)
    def fn(x, y):
        return step_fn(model, opt, x, y)
    if captured:
        import reflop

        fn = reflop.capture(fn)
    for _ in range(3):
        fn(x, y)
    sync()
    import reflop

    global _before
    _before = dict(reflop.stats().get(0, {})) if dev.type == "reflop" else {}
    t0 = time.perf_counter()
    for _ in range(steps):
        loss = fn(x, y)
        if dev.type == "reflop" and not captured:
            reflop.step_boundary()
    loss_v = float(loss)
    sync()
    dt = (time.perf_counter() - t0) / steps
    print(f"{label:28} {dt * 1000:8.2f} ms/step  {batch / dt:9.1f} img/s  loss {loss_v:.4f}")
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--local-worker", action="store_true", help="in-process worker instead of reflopd")
    ap.add_argument("--capture", action="store_true", help="use reflop.capture (static replay)")
    a = ap.parse_args()
    import reflop

    base = bench(torch.device("cuda"), a.steps, a.batch, "local GPU (plain PyTorch)", torch.cuda.synchronize)
    dev = reflop.device(local="cuda" if a.local_worker else None)
    t = bench(dev, a.steps, a.batch, "reflop" + (" (in-process)" if a.local_worker else " (reflopd)")
              + (" capture" if a.capture else " eager"), reflop.synchronize, a.capture)
    st = reflop.stats()[0]
    before = _before
    n = a.steps
    print(f"throughput vs local GPU: {base / t * 100:.1f}%")
    print(f"upload per step: {(st['bytes_up'] - before['bytes_up']) / n / 1024:.1f} KiB; "
          f"registrations: {st['registers'] - before['registers']}; executes: {st['executes'] - before['executes']}")
    reflop.close()


if __name__ == "__main__":
    main()
