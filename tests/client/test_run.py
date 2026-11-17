# SPDX-License-Identifier: Apache-2.0
"""Issue #8: `reflop run` never ends silently. A child killed by a signal is
named and exits 128 + n, as the shell would report it; Python children get
faulthandler; with REFLOP_DEBUG=1 a hung child can be made to dump its
stacks; Ctrl-C reaches the child instead of killing reflop first."""

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
from reflop.cli import main as cli
from reflop.cli.main import _signal_name, main

ROOT = Path(__file__).resolve().parents[2]
posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals only")


def _fake_child(monkeypatch, rc):
    seen = {}

    def run(command, env):
        seen["command"], seen["env"] = command, env
        return rc
    monkeypatch.setattr(cli, "_run_child", run)
    return seen


@pytest.fixture
def child_path(monkeypatch):
    """Children are new interpreters: give them the client and worker
    packages that conftest puts on this process's sys.path."""
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(ROOT / "client/python"), str(ROOT / "server/worker")]))
    monkeypatch.setenv("REFLOP_NO_KEYCHAIN", "1")
    monkeypatch.delenv("REFLOP_DEBUG", raising=False)
    monkeypatch.delenv("PYTHONFAULTHANDLER", raising=False)


def test_signal_death_is_reported(monkeypatch, capsys):
    _fake_child(monkeypatch, -signal.SIGSEGV)

    rc = main(["run", "python", "-c", "raise"])

    assert rc == 128 + signal.SIGSEGV
    err = capsys.readouterr().err
    assert "python was killed by SIGSEGV (signal 11)" in err


def test_only_the_program_name_is_echoed(monkeypatch, capsys):
    """The argv tail can carry secrets, so it must not reach stderr."""
    _fake_child(monkeypatch, -signal.SIGTERM)

    main(["run", "hf", "login", "--token", "hf_secret"])

    err = capsys.readouterr().err
    assert "hf was killed by SIGTERM" in err
    assert "hf_secret" not in err


def test_exit_status_passes_through(monkeypatch):
    _fake_child(monkeypatch, 3)

    assert main(["run", "false"]) == 3


def test_ctrl_c_and_closed_pipe_are_quiet_like_a_shell(monkeypatch, capsys):
    for sig in (signal.SIGINT, signal.SIGPIPE):
        _fake_child(monkeypatch, -sig)
        assert main(["run", "python", "x.py"]) == 128 + sig
        assert "killed" not in capsys.readouterr().err


def test_child_gets_faulthandler_unless_the_user_chose(monkeypatch):
    monkeypatch.delenv("PYTHONFAULTHANDLER", raising=False)
    seen = _fake_child(monkeypatch, 0)
    main(["run", "python", "x.py"])
    assert seen["env"]["PYTHONFAULTHANDLER"] == "1"

    monkeypatch.setenv("PYTHONFAULTHANDLER", "")
    main(["run", "python", "x.py"])
    assert seen["env"]["PYTHONFAULTHANDLER"] == ""


def test_debug_reaches_the_child(monkeypatch):
    monkeypatch.setenv("REFLOP_DEBUG", "1")
    seen = _fake_child(monkeypatch, 0)
    main(["run", "python", "x.py"])
    assert seen["env"]["REFLOP_DEBUG"] == "1"


def test_macos_child_gets_one_openmp_thread(monkeypatch, capsys):
    """Issue #7, ADR-027: a second OpenMP runtime next to torch's libomp
    crashes or hangs the program unless OpenMP stays on one thread."""
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    seen = _fake_child(monkeypatch, 0)

    main(["run", "python", "train.py"])

    assert seen["env"]["OMP_NUM_THREADS"] == "1"
    assert "OMP_NUM_THREADS=1 (macOS default" in capsys.readouterr().err
    assert "OMP_NUM_THREADS" not in os.environ  # reflop's own process is unchanged


@pytest.mark.parametrize("value", ["4", "1", ""])
def test_macos_user_openmp_setting_wins(monkeypatch, capsys, value):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setenv("OMP_NUM_THREADS", value)
    seen = _fake_child(monkeypatch, 0)

    main(["run", "python", "train.py"])

    assert seen["env"]["OMP_NUM_THREADS"] == value
    assert "macOS default" not in capsys.readouterr().err


def test_other_platforms_keep_their_openmp_threads(monkeypatch, capsys):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    seen = _fake_child(monkeypatch, 0)

    main(["run", "python", "train.py"])

    assert "OMP_NUM_THREADS" not in seen["env"]
    assert "macOS default" not in capsys.readouterr().err


def test_macos_crash_points_at_the_openmp_issue(monkeypatch, capsys):
    monkeypatch.setattr(sys, "platform", "darwin")
    _fake_child(monkeypatch, -signal.SIGSEGV)

    # One thread already (reflop's default or the user's): send them to doctor.
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    main(["run", "python", "train.py"])
    err = capsys.readouterr().err
    assert "reflop doctor" in err and cli.OPENMP_ISSUE in err
    assert "try OMP_NUM_THREADS=1" not in err

    # The user chose more threads: suggest one.
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    main(["run", "python", "train.py"])
    err = capsys.readouterr().err
    assert "try OMP_NUM_THREADS=1" in err and cli.OPENMP_ISSUE in err

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    main(["run", "python", "train.py"])
    assert cli.OPENMP_ISSUE not in capsys.readouterr().err


def test_macos_clean_exit_has_no_openmp_hint(monkeypatch, capsys):
    monkeypatch.setattr(sys, "platform", "darwin")
    _fake_child(monkeypatch, 0)
    main(["run", "python", "train.py"])
    assert cli.OPENMP_ISSUE not in capsys.readouterr().err


def test_run_help_documents_debug(capsys):
    with pytest.raises(SystemExit):
        main(["run", "--help"])
    out = capsys.readouterr().out
    assert "REFLOP_DEBUG=1" in out and "kill -USR1" in out
    assert "OMP_NUM_THREADS=1" in out and "reflop doctor" in out


@posix_only
def test_a_real_child_killed_by_a_signal(capsys):
    """No fake here: the child really dies from a signal, so the path that
    turns a negative wait status into 128 + n is exercised end to end."""
    rc = main(["run", sys.executable, "-c", "import os, signal; os.kill(os.getpid(), signal.SIGSEGV)"])

    assert rc == 128 + signal.SIGSEGV
    err = capsys.readouterr().err
    assert "was killed by SIGSEGV (signal 11)" in err
    assert sys.executable in err


@posix_only
def test_a_native_crash_prints_python_stacks(child_path, capfd):
    rc = main(["run", sys.executable, "-c",
               "import os, signal\ndef crash():\n    os.kill(os.getpid(), signal.SIGSEGV)\ncrash()"])

    assert rc == 128 + signal.SIGSEGV
    err = capfd.readouterr().err
    assert "Fatal Python error: Segmentation fault" in err
    assert "in crash" in err


@posix_only
def test_debug_sigusr1_dumps_stacks_and_the_child_runs_on(child_path, monkeypatch, capfd):
    monkeypatch.setenv("REFLOP_DEBUG", "1")
    rc = main(["run", sys.executable, "-c",
               "import os, signal\ndef stuck():\n    os.kill(os.getpid(), signal.SIGUSR1)\nstuck()\nprint('alive')"])

    assert rc == 0
    out, err = capfd.readouterr()
    assert "alive" in out
    assert "in stuck" in err  # faulthandler's dump of the main thread
    assert "is pid" in err and "kill -USR1" in err


@posix_only
def test_without_debug_sigusr1_is_left_alone(child_path, capfd):
    """Programs may use SIGUSR1 themselves: only REFLOP_DEBUG takes it over."""
    rc = main(["run", sys.executable, "-c", "import os, signal; os.kill(os.getpid(), signal.SIGUSR1)"])

    assert rc == 128 + signal.SIGUSR1
    assert "killed by SIGUSR1" in capfd.readouterr().err


@posix_only
def test_ctrl_c_does_not_kill_reflop_before_the_child(child_path, capfd):
    """A terminal sends SIGINT to reflop and the child together. Here the
    child sends it to reflop alone and then finishes its work: reflop must
    wait for it rather than die and take the child down."""
    rc = main(["run", sys.executable, "-c",
               "import os, signal, time\nos.kill(os.getppid(), signal.SIGINT)\ntime.sleep(0.3)\nprint('finished')"])

    assert rc == 0
    assert "finished" in capfd.readouterr().out
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler  # restored afterwards


@posix_only
def test_sigterm_to_reflop_reaches_the_child(child_path, capfd):
    rc = main(["run", sys.executable, "-c",
               "import os, signal, time\nos.kill(os.getppid(), signal.SIGTERM)\ntime.sleep(30)"])

    assert rc == 128 + signal.SIGTERM
    assert "killed by SIGTERM" in capfd.readouterr().err


@posix_only
def test_ignored_signals_stay_ignored(monkeypatch):
    """A background job starts with SIGINT ignored; the child must inherit
    that rather than get a default handler from reflop."""
    seen = {}
    real = subprocess.Popen

    def popen(command, env):
        seen["sigint"] = signal.getsignal(signal.SIGINT)
        return real([sys.executable, "-c", "pass"], env=env)
    monkeypatch.setattr(subprocess, "Popen", popen)
    old = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        assert cli._run_child(["x"], dict(os.environ)) == 0
        assert seen["sigint"] is signal.SIG_IGN
    finally:
        signal.signal(signal.SIGINT, old)


def test_signal_name_falls_back_for_a_nameless_number():
    assert _signal_name(1_000_000) == "signal 1000000"
