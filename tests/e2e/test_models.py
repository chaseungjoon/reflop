# SPDX-License-Identifier: Apache-2.0
"""§17.2 end-to-end: reference models train on the reflop device (in-process
worker on the CPU, so results are deterministic) exactly like local CPU."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "spikes"))


@pytest.mark.parametrize("name", ["mlp", "gpt", "unet", "lstm", "resnet18"])
def test_model_trains_like_local(name):
    if name == "resnet18":
        pytest.importorskip("torchvision")
    import s4_op_census as census

    import reflop

    r = census.run(name, "cpu")
    reflop.close()
    assert not r["fallback"], r["fallback"]
    for remote, local in r["losses"]:
        assert abs(remote - local) <= 1e-4 * max(1.0, abs(local)), r["losses"]
