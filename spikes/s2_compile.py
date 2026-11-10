#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""M0 spike S2: torch.compile custom backend + AOTAutograd with reflop tensors."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "client/python"), str(ROOT / "server/worker")]
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import reflop  # noqa: E402

seen = {}


def backend(gm, example_inputs):
    seen["nodes"] = len(gm.graph.nodes)
    seen["inputs"] = [type(x).__name__ for x in example_inputs][:4]
    return gm.forward


dev = reflop.device(local="cpu")
m = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 2)).to(dev)
x = torch.randn(4, 8).to(dev)
for name, fn in (("dynamo-only", torch.compile(m, backend=backend)),
                 ("aot", torch.compile(m, backend="aot_eager"))):
    try:
        y = fn(x)
        loss = F.cross_entropy(y, torch.zeros(4, dtype=torch.long).to(dev))
        loss.backward()
        print(f"{name}: OK loss={loss.item():.4f} {seen}")
    except Exception as e:  # noqa: BLE001
        print(f"{name}: FAILED {type(e).__name__}: {str(e).splitlines()[0][:200]}")
