# SPDX-License-Identifier: Apache-2.0
"""Issue #8: when `reflop run` sees its child die from a signal it names the
signal and exits 128 + n, the status the shell would have reported itself."""

import signal
import subprocess
import sys

import pytest
from reflop.cli.main import _signal_name, main


def test_signal_death_is_reported(monkeypatch, capsys):
    monkeypatch.setattr(subprocess, "call", lambda command, env=None: -signal.SIGSEGV)

    rc = main(["run", "python", "-c", "raise"])

    assert rc == 128 + signal.SIGSEGV
    err = capsys.readouterr().err
    assert "python was killed by SIGSEGV (signal 11)" in err


def test_only_the_program_name_is_echoed(monkeypatch, capsys):
    """The argv tail can carry secrets, so it must not reach stderr."""
    monkeypatch.setattr(subprocess, "call", lambda command, env=None: -signal.SIGTERM)

    main(["run", "hf", "login", "--token", "hf_secret"])

    err = capsys.readouterr().err
    assert "hf was killed by SIGTERM" in err
    assert "hf_secret" not in err


def test_exit_status_passes_through(monkeypatch):
    monkeypatch.setattr(subprocess, "call", lambda command, env=None: 3)

    assert main(["run", "false"]) == 3


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals only: Windows reports a plain exit status")
def test_a_real_child_killed_by_a_signal(capsys):
    """No fake here: the child really dies from a signal, so the path that
    turns a negative wait status into 128 + n is exercised end to end."""
    rc = main(["run", sys.executable, "-c", "import os, signal; os.kill(os.getpid(), signal.SIGSEGV)"])

    assert rc == 128 + signal.SIGSEGV
    err = capsys.readouterr().err
    assert "was killed by SIGSEGV (signal 11)" in err
    assert sys.executable in err


def test_signal_name_falls_back_for_a_nameless_number():
    assert _signal_name(1_000_000) == "signal 1000000"
