# SPDX-License-Identifier: Apache-2.0
"""Issue #7 regression: a program that loads torch's OpenMP runtime and
scikit-learn's (a second libomp on macOS) runs to the end under `reflop run`,
with reflop tensors in between. The macOS CI job installs scikit-learn for
this; elsewhere it is skipped when scikit-learn is missing."""

import os
import sys
from pathlib import Path

import pytest
from reflop.cli.main import main

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).with_name("openmp_clash.py")

pytest.importorskip("sklearn")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX only")
def test_torch_and_sklearn_runtimes_under_reflop_run(monkeypatch, capfd):
    paths = [str(ROOT / "client/python"), str(ROOT / "server/worker")]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(paths + [p for p in [os.environ.get("PYTHONPATH")] if p]))
    monkeypatch.setenv("REFLOP_NO_KEYCHAIN", "1")
    monkeypatch.setenv("REFLOP_LOCAL", "cpu")
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.delenv("KMP_DUPLICATE_LIB_OK", raising=False)

    rc = main(["run", sys.executable, str(SCRIPT)])

    out, err = capfd.readouterr()
    assert rc == 0, err[-4000:]
    assert "openmp-ok" in out
    assert "OMP: Error" not in err
