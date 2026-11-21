# SPDX-License-Identifier: Apache-2.0
"""`reflop run` shim: CUDA autocast reaches the reflop device.

Regression: torch.autocast("cuda", float16) did nothing (outputs stayed
float32) and bfloat16 raised "Torch not compiled with CUDA enabled". The shim
patches torch globally, so it runs in a child process with no local CUDA.
The timeout catches the local-mode seeding deadlock.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SCRIPT = r"""
import json, warnings
import torch, torch.nn as nn
from reflop import shim

shim.install()
# Regression: in local mode the in-process worker's own reseed re-entered the
# shim's torch.cuda.manual_seed and deadlocked the first .cuda().
torch.manual_seed(0)
m =nn.Linear(8, 4).cuda()
x = torch.randn(2, 8).cuda()
r = {"bf16_supported": torch.cuda.is_bf16_supported()}
with torch.autocast("cuda", dtype=torch.float16):
    r["fp16"] = str(m(x).dtype)
    r["enabled"] = torch.is_autocast_enabled("cuda")
    r["enabled_legacy"] = torch.is_autocast_enabled()
    r["dtype"] = str(torch.get_autocast_dtype("cuda"))
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    y = m(x)
    r["bf16"] = str(y.dtype)
    r["loss"] = str(y.float().pow(2).mean().dtype)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    with torch.cuda.amp.autocast():
        r["cuda_amp"] = str(m(x).dtype)
r["after"] = str(m(x).dtype)
r["enabled_after"] = torch.is_autocast_enabled("cuda")
with torch.autocast("cpu", dtype=torch.bfloat16):
    r["cpu"] = str(nn.Linear(8, 4)(torch.randn(2, 8)).dtype)
print(json.dumps(r))
"""


def test_cuda_autocast_runs_on_the_reflop_device():
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT / "client/python"), str(ROOT / "server/worker")]),
               REFLOP_NO_KEYCHAIN="1", REFLOP_LOCAL="cpu", CUDA_VISIBLE_DEVICES="")
    p = subprocess.run([sys.executable, "-c", SCRIPT], env=env, capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    r = json.loads(p.stdout.strip().splitlines()[-1])
    assert r == {
        "bf16_supported": True,  # the server CPU lists bfloat16
        "fp16": "torch.float16", "enabled": True, "enabled_legacy": True, "dtype": "torch.float16",
        "bf16": "torch.bfloat16", "loss": "torch.float32",
        "cuda_amp": "torch.float16",
        "after": "torch.float32", "enabled_after": False,
        "cpu": "torch.bfloat16",  # other device types are left alone
    }
