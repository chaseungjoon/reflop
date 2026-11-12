# SPDX-License-Identifier: Apache-2.0
"""`reflop update` picks the right installer for how reflop was installed,
never downgrades, and installs only files that match the signed release."""

import hashlib
import sys

import pytest
from reflop import update as up

WHEEL = b"wheel-bytes"
SHA = hashlib.sha256(WHEEL).hexdigest()
SDIST_SHA = "b" * 64


def lock_text(version, hashes):
    files = "\n".join(f'    {{ url = "https://files.pythonhosted.org/x", hash = "sha256:{h}" }},' for h in hashes)
    return f'''version = 1

[[package]]
name = "numpy"
version = "2.0.0"
source = {{ registry = "https://pypi.org/simple" }}
wheels = [
    {{ url = "https://files.pythonhosted.org/n", hash = "sha256:{"c" * 64}" }},
]

[[package]]
name = "reflop"
version = "{version}"
source = {{ registry = "https://pypi.org/simple" }}
dependencies = [
    {{ name = "numpy" }},
]
wheels = [
{files}
]
'''


@pytest.fixture
def release(monkeypatch):
    """A signed release 9.9.9 whose wheel is WHEEL; installs are recorded."""
    calls = []
    state = {"installed": "9.9.9", "pypi": SHA}
    monkeypatch.setattr(up, "latest", lambda: "v9.9.9")
    monkeypatch.setattr(up, "signed_sums", lambda tag: {f"reflop-{tag[1:]}-py3-none-any.whl": SHA,
                                                         f"reflop-{tag[1:]}.tar.gz": SDIST_SHA})
    monkeypatch.setattr(up, "pypi_sha256", lambda v, f: state["pypi"])
    monkeypatch.setattr(up, "_get", lambda url: WHEEL)
    monkeypatch.setattr(up, "_installed_version", lambda: state["installed"])
    monkeypatch.setattr(up.shutil, "which", lambda n: "/usr/bin/uv" if n == "uv" else None)
    monkeypatch.setattr("reflop.__version__", "0.1.6")

    def run(cmd):
        calls.append(cmd)
        return state.get("on_run", lambda c: 0)(cmd)

    monkeypatch.setattr(up, "_run", run)
    return calls, state


def uv_project(tmp_path, monkeypatch, version="0.1.6"):
    proj = tmp_path / "proj"
    (proj / ".venv").mkdir(parents=True)
    (proj / "pyproject.toml").write_text('[project]\nname = "p"\nversion = "0"\ndependencies = ["reflop"]\n')
    (proj / "uv.lock").write_text(lock_text(version, ["a" * 64]))
    monkeypatch.setattr(sys, "prefix", str(proj / ".venv"))
    return proj


def test_install_kind(tmp_path, monkeypatch):
    assert up.install_kind("/home/u/.local/share/uv/tools/reflop") == ("uv-tool", None)
    assert up.install_kind("/home/u/.local/share/pipx/venvs/reflop") == ("pipx", None)
    assert up.install_kind("/home/u/.local/share/reflop/client-venv") == ("private", None)
    assert up.install_kind("/home/u/proj/.venv") == ("venv", None)
    proj = uv_project(tmp_path, monkeypatch)
    assert up.install_kind(str(proj / ".venv")) == ("uv-project", str(proj))
    # A uv project whose lock does not list reflop (installed with `uv pip`).
    (proj / "uv.lock").write_text('version = 1\n\n[[package]]\nname = "numpy"\nversion = "2.0.0"\n')
    assert up.install_kind(str(proj / ".venv")) == ("venv", None)


def test_lock_entry():
    assert up.lock_entry(lock_text("1.2.3", ["a" * 64, "d" * 64])) == ("1.2.3", {"a" * 64, "d" * 64})
    assert up.lock_entry('version = 1\n\n[[package]]\nname = "reflopx"\nversion = "1"\n') is None


def test_never_downgrades(release, monkeypatch, capsys):
    calls, _ = release
    for cur in ("9.9.9", "10.0.0"):
        monkeypatch.setattr("reflop.__version__", cur)
        assert up.update() == 0
    assert calls == []
    assert "up to date" in capsys.readouterr().out


def test_uv_project_upgrades_the_lock_then_syncs(release, tmp_path, monkeypatch):
    calls, state = release
    proj = uv_project(tmp_path, monkeypatch)

    def on_run(cmd):
        if cmd[1] == "lock":
            (proj / "uv.lock").write_text(lock_text("9.9.9", [SHA, SDIST_SHA]))
        return 0

    state["on_run"] = on_run
    assert up.update() == 0
    assert [c[1] for c in calls] == ["lock", "sync"]
    assert calls[0][-2:] == ["--upgrade-package", "reflop"] and "--inexact" in calls[1]
    assert up.lock_entry((proj / "uv.lock").read_text())[0] == "9.9.9"


def test_uv_project_lock_with_unsigned_hash_is_restored(release, tmp_path, monkeypatch):
    calls, state = release
    proj = uv_project(tmp_path, monkeypatch)
    old = (proj / "uv.lock").read_text()
    state["on_run"] = lambda cmd: (proj / "uv.lock").write_text(lock_text("9.9.9", ["e" * 64])) and 0
    with pytest.raises(RuntimeError, match="do not match the signed release"):
        up.update()
    assert (proj / "uv.lock").read_text() == old
    assert [c[1] for c in calls] == ["lock"]


def test_uv_project_pinned_below_the_release(release, tmp_path, monkeypatch, capsys):
    calls, state = release
    proj = uv_project(tmp_path, monkeypatch)
    old = (proj / "uv.lock").read_text()
    assert up.update() == 1  # the fake `uv lock` leaves reflop at 0.1.6
    assert (proj / "uv.lock").read_text() == old
    assert "uv add 'reflop>=9.9.9'" in capsys.readouterr().err
    assert [c[1] for c in calls] == ["lock"]


def test_uv_project_waits_for_pypi(release, tmp_path, monkeypatch):
    calls, state = release
    uv_project(tmp_path, monkeypatch)
    state["pypi"] = None
    with pytest.raises(RuntimeError, match="not on PyPI yet"):
        up.update()
    state["pypi"] = "f" * 64
    with pytest.raises(RuntimeError, match="does not match the signed release"):
        up.update()
    assert calls == []


def test_uv_tool_reinstalls_from_pypi(release, monkeypatch):
    calls, state = release
    monkeypatch.setattr(sys, "prefix", "/home/u/.local/share/uv/tools/reflop")
    assert up.update() == 0
    assert calls == [["/usr/bin/uv", "tool", "install", "--force", "reflop==9.9.9"]]
    # Before PyPI has the release: the verified wheel from the release.
    calls.clear()
    state["pypi"] = None
    assert up.update() == 0
    assert calls[0][:4] == ["/usr/bin/uv", "tool", "install", "--force"] and calls[0][4].endswith(".whl")
    # PyPI serving a different file: refused.
    calls.clear()
    state["pypi"] = "f" * 64
    with pytest.raises(RuntimeError, match="does not match"):
        up.update()
    assert calls == []


def test_plain_venv_installs_the_verified_wheel(release, monkeypatch):
    calls, _ = release
    monkeypatch.setattr(sys, "prefix", "/home/u/proj/.venv")
    monkeypatch.setenv("VIRTUAL_ENV", "/home/u/proj/.venv")
    assert up.update() == 0
    assert calls[0][:3] == ["/usr/bin/uv", "pip", "install"] and calls[0][-1].endswith("reflop-9.9.9-py3-none-any.whl")
    monkeypatch.setattr(up, "_get", lambda url: b"tampered")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        up.update()


def test_reports_an_update_that_did_not_take(release, monkeypatch, capsys):
    _, state = release
    monkeypatch.setattr(sys, "prefix", "/home/u/.local/share/reflop/client-venv")
    state["installed"] = "0.1.6"
    assert up.update() == 1
    assert "expected 9.9.9" in capsys.readouterr().err
