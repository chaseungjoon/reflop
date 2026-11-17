# SPDX-License-Identifier: Apache-2.0
"""Issue #7: on macOS a second OpenMP runtime next to torch's libomp crashes
or hangs the Guest's program. `reflop doctor` names every runtime installed
and loaded, and the package each came from."""

import sys
from pathlib import Path

import pytest
import torch
from reflop import _openmp, doctor
from reflop._openmp import Runtime, is_runtime, report


@pytest.mark.parametrize("path", [
    "torch/lib/libomp.dylib",
    "sklearn/.dylibs/libomp.dylib",
    "torch/lib/libiomp5.dylib",
    "torch/lib/libgomp.so.1",
    "scikit_learn.libs/libgomp-a34b3233.so.1.0.0",
    "lib/libiomp5md.dll",
    "/opt/homebrew/opt/libomp/lib/libomp.dylib",
])
def test_runtime_names(path):
    assert is_runtime(path)


@pytest.mark.parametrize("path", [
    "torch/lib/libtorch_cpu.dylib", "libompitrace.so", "libomptarget.so", "sklearn/_openmp_helpers.py",
    "torch/lib/libgomp.so.1/README", "libomp.dylib.txt",
])
def test_not_runtime_names(path):
    assert not is_runtime(path)


TORCH = Runtime("torch", "/env/torch/lib/libomp.dylib")
SKLEARN = Runtime("scikit-learn", "/env/sklearn/.dylibs/libomp.dylib")


def levels(lines):
    return [lvl for lvl, _ in lines]


def test_macos_one_runtime_is_fine():
    lines = report([TORCH], [TORCH.path], "darwin", None)
    assert levels(lines) == ["ok"]
    assert "torch (/env/torch/lib/libomp.dylib)" in lines[0][1]


def test_macos_second_installed_runtime_warns_and_names_the_package():
    lines = report([SKLEARN, TORCH], [TORCH.path], "darwin", None)
    assert levels(lines) == ["ok", "warn", "warn"]
    assert "scikit-learn (/env/sklearn/.dylibs/libomp.dylib)" in lines[1][1]
    assert "imports scikit-learn next to torch" in lines[1][1]
    assert "OMP_NUM_THREADS=1 python train.py" in lines[2][1]


def test_macos_no_env_hint_once_the_user_set_one_thread():
    assert levels(report([SKLEARN, TORCH], [TORCH.path], "darwin", "1")) == ["ok", "warn"]


def test_macos_two_runtimes_loaded_with_torch_fails():
    brew = "/opt/homebrew/opt/libomp/lib/libomp.dylib"
    lines = report([TORCH], [TORCH.path, brew], "darwin", None)
    assert levels(lines)[0] == "fail"
    assert "2 OpenMP runtimes loaded" in lines[0][1] and brew in lines[0][1] and "torch (" in lines[0][1]


def test_linux_lists_but_does_not_judge():
    """torch on Linux uses libgomp, and several copies of it coexist."""
    gomp = Runtime("torch", "/env/torch/lib/libgomp.so.1")
    other = Runtime("scikit-learn", "/env/scikit_learn.libs/libgomp-a34b3233.so.1.0.0")
    lines = report([other, gomp], [gomp.path], "linux", None)
    assert levels(lines) == ["ok", "ok"]
    assert "scikit-learn" in lines[1][1]


def test_nothing_found():
    assert report([], [], "darwin", None) == [("ok", "OpenMP: no runtime found")]


class _Dist:
    def __init__(self, name, root, files):
        self.metadata = {"Name": name}
        self.files = [Path(f) for f in files]
        self._root = root

    def locate_file(self, f):
        return self._root / f


def test_installed_reads_package_records(monkeypatch, tmp_path):
    for f in ("torch/lib/libomp.dylib", "torch/lib/libtorch.dylib", "sklearn/.dylibs/libomp.dylib"):
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / f).write_bytes(b"")
    (tmp_path / "torch/lib/libiomp5.dylib").symlink_to("libomp.dylib")  # the same runtime twice

    class Broken:
        @property
        def files(self):
            raise OSError("bad RECORD")

    dists = [
        _Dist("torch", tmp_path, ["torch/lib/libomp.dylib", "torch/lib/libiomp5.dylib", "torch/lib/libtorch.dylib"]),
        _Dist("scikit-learn", tmp_path, ["sklearn/.dylibs/libomp.dylib", "sklearn/.dylibs/libgomp.so.1"]),
        Broken(),
    ]
    from importlib import metadata

    monkeypatch.setattr(metadata, "distributions", lambda: dists)

    got = _openmp.installed()

    assert got == [
        Runtime("scikit-learn", str((tmp_path / "sklearn/.dylibs/libomp.dylib").resolve())),
        Runtime("torch", str((tmp_path / "torch/lib/libomp.dylib").resolve())),
    ]


def test_loaded_finds_torchs_runtime_in_this_process():
    """No fakes: torch is imported, so its OpenMP runtime is mapped."""
    lib = Path(torch.__file__).parent / "lib"
    shipped = [p for p in lib.iterdir() if is_runtime(str(p))] if lib.is_dir() else []
    if not shipped or sys.platform not in ("darwin", "linux"):
        pytest.skip("this torch build ships no OpenMP runtime of its own")
    got = _openmp.loaded()
    assert any(Path(p).parent == lib.resolve() for p in got), got
    assert all(is_runtime(p) for p in got)


def test_installed_finds_torchs_runtime_in_this_environment():
    lib = Path(torch.__file__).parent / "lib"
    if not lib.is_dir() or not any(is_runtime(str(p)) for p in lib.iterdir()):
        pytest.skip("this torch build ships no OpenMP runtime of its own")
    assert any(r.package == "torch" for r in _openmp.installed())


def _doctor(monkeypatch, tmp_path, lines, plat="darwin"):
    """doctor() with a healthy pairing faked, so only the OpenMP lines decide
    the exit status."""
    from types import SimpleNamespace

    from reflop import client, config

    monkeypatch.setenv("REFLOP_CLIENT_HOME", str(tmp_path))
    monkeypatch.setattr(config.ClientConfig, "load", classmethod(lambda cls: SimpleNamespace(servers={"gpu": 1})))
    conn = SimpleNamespace(entry=SimpleNamespace(name="gpu"), auth=SimpleNamespace(identity_name="me"),
                           close=lambda: None)
    monkeypatch.setattr(client, "admin", lambda server=None: conn)
    monkeypatch.setattr(sys, "platform", plat)
    monkeypatch.setattr(_openmp, "installed", lambda: [])
    monkeypatch.setattr(_openmp, "loaded", lambda: [])
    monkeypatch.setattr(_openmp, "report", lambda *a: lines)
    return doctor.doctor()


def test_doctor_warning_is_shown_but_does_not_fail(monkeypatch, tmp_path, capsys):
    """Two installed runtimes are a risk, not a broken setup."""
    assert _doctor(monkeypatch, tmp_path, [("ok", "fine")]) == 0
    assert _doctor(monkeypatch, tmp_path, [("ok", "fine"), ("warn", "second runtime")]) == 0
    assert "WARN second runtime" in capsys.readouterr().out


def test_doctor_fails_on_two_loaded_runtimes(monkeypatch, tmp_path, capsys):
    assert _doctor(monkeypatch, tmp_path, [("fail", "two loaded")]) == 1
    assert "FAIL two loaded" in capsys.readouterr().out


def test_doctor_survives_a_listing_error(monkeypatch, capsys):
    def boom():
        raise OSError("no dyld")
    monkeypatch.setattr(_openmp, "installed", boom)
    seen = []
    doctor.openmp(lambda ok, msg, warn=False: seen.append((ok, msg, warn)), "darwin")
    assert seen and seen[0][2] and "no dyld" in seen[0][1]


def test_doctor_runs_the_real_check(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("REFLOP_CLIENT_HOME", str(tmp_path))
    doctor.doctor()
    assert "OpenMP" in capsys.readouterr().out
