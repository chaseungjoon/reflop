# SPDX-License-Identifier: Apache-2.0
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "client/python"), str(ROOT / "server/worker")]
os.environ.setdefault("REFLOP_NO_KEYCHAIN", "1")


def cuda_ok() -> bool:
    import torch

    return torch.cuda.is_available()


@pytest.fixture(scope="session")
def devname():
    return "cuda" if cuda_ok() and os.environ.get("REFLOP_TEST_DEVICE", "cuda") == "cuda" else "cpu"
