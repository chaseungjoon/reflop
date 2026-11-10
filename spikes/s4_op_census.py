#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""M0 spike S4: op census. Trains reference models for a few steps on the
reflop device (in-process worker) and lists every op that reached the
client dispatcher, every decomposition, and every local fallback."""

from __future__ import annotations

import collections
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "client/python"), str(ROOT / "server/worker")]

import reflop  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from reflop import tensor as T  # noqa: E402

seen = collections.Counter()
recorded = collections.Counter()
fallback = collections.Counter()
orig_dispatch = T._dispatch
orig_fb = T._run_locally
orig_rec = None


def spy_dispatch(sess, func, args, kwargs):
    seen[str(func)] += 1
    return orig_dispatch(sess, func, args, kwargs)


def spy_fb(sess, func, args, kwargs):
    fallback[str(func)] += 1
    return orig_fb(sess, func, args, kwargs)


T._dispatch = spy_dispatch
T._run_locally = spy_fb


def mlp():
    return nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 10)), (torch.randn(16, 32),), torch.randint(0, 10, (16,))


def resnet18():
    from torchvision.models import resnet18 as r

    return r(num_classes=10), (torch.randn(4, 3, 32, 32),), torch.randint(0, 10, (4,))


class TinyGPT(nn.Module):
    def __init__(self, v=64, d=64, h=4, n=2, t=32):
        super().__init__()
        self.emb = nn.Embedding(v, d)
        self.pos = nn.Parameter(torch.zeros(1, t, d))
        self.blocks = nn.ModuleList(nn.TransformerEncoderLayer(d, h, 4 * d, dropout=0.0, batch_first=True,
                                                               norm_first=True) for _ in range(n))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, v)
        self.t = t

    def forward(self, idx):
        x = self.emb(idx) + self.pos[:, : idx.shape[1]]
        mask = nn.Transformer.generate_square_subsequent_mask(idx.shape[1], device=x.device)
        for b in self.blocks:
            x = b(x, src_mask=mask, is_causal=True)
        return self.head(self.ln(x)).flatten(0, 1)


def gpt():
    return TinyGPT(), (torch.randint(0, 64, (4, 32)),), torch.randint(0, 64, (4 * 32,))


class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU())
        self.d2 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(8, 16, 3, padding=1), nn.ReLU())
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.u1 = nn.Conv2d(24, 8, 3, padding=1)
        self.out = nn.Conv2d(8, 2, 1)

    def forward(self, x):
        a = self.d1(x)
        b = self.d2(a)
        return self.out(F.relu(self.u1(torch.cat([a, self.up(b)], 1)))).permute(0, 2, 3, 1).reshape(-1, 2)


def unet():
    return TinyUNet(), (torch.randn(2, 3, 16, 16),), torch.randint(0, 2, (2 * 16 * 16,))


class LSTMNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(8, 16, batch_first=True)
        self.fc = nn.Linear(16, 4)

    def forward(self, x):
        y, _ = self.lstm(x)
        return self.fc(y[:, -1])


def lstm():
    return LSTMNet(), (torch.randn(4, 10, 8),), torch.randint(0, 4, (4,))


MODELS = {"mlp": mlp, "resnet18": resnet18, "gpt": gpt, "unet": unet, "lstm": lstm}


def run(name, device):
    seen.clear()
    fallback.clear()
    torch.manual_seed(0)
    model, inputs, target = MODELS[name]()
    import copy

    ref = copy.deepcopy(model)
    dev = reflop.device(local=device)
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ropt = torch.optim.AdamW(ref.parameters(), lr=1e-3)
    losses = []
    for _ in range(3):
        opt.zero_grad()
        ropt.zero_grad()
        out = model(*[x.to(dev) for x in inputs])
        loss = F.cross_entropy(out, target.to(dev))
        loss.backward()
        opt.step()
        rl = F.cross_entropy(ref(*inputs), target)
        rl.backward()
        ropt.step()
        losses.append((loss.item(), rl.item()))
    return {"seen": dict(seen), "fallback": dict(fallback), "losses": losses}


if __name__ == "__main__":
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda"
    names = sys.argv[2:] or list(MODELS)
    report = {}
    for n in names:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                report[n] = run(n, device)
                r = report[n]
                print(f"{n}: {len(r['seen'])} distinct ops, fallbacks {sorted(r['fallback'])}, "
                      f"losses {[(round(a, 4), round(b, 4)) for a, b in r['losses']]}")
            except Exception as e:  # noqa: BLE001
                import traceback

                traceback.print_exc()
                report[n] = {"error": f"{type(e).__name__}: {e}"}
                print(f"{n}: ERROR {type(e).__name__}: {str(e)[:300]}")
        reflop.close()
    (ROOT / "spikes/out").mkdir(exist_ok=True)
    (ROOT / "spikes/out/op_census.json").write_text(json.dumps(report, indent=1))
