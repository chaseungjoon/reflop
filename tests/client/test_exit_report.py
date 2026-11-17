# SPDX-License-Identifier: Apache-2.0
"""Issue #8: a request error that the program never waited for is written to
stderr at exit, instead of being dropped with the session."""

import os
import subprocess
import sys
import threading
from concurrent.futures import Future
from pathlib import Path

from reflop._pb import compute_pb2 as cpb
from reflop.errors import ReflopRemoteError
from reflop.session import Session

ROOT = Path(__file__).resolve().parents[2]


def _session(*futures):
    s = Session.__new__(Session)
    s.inflight, s.async_error, s.async_what = [], None, ""
    for f, what in futures:
        s._track(f, what)
    return s


def _remote_error():
    return ReflopRemoteError(cpb.ERROR_INDEX_OUT_OF_BOUNDS, "index out of range")


def test_an_error_arriving_while_exiting_is_printed(capsys):
    f = Future()
    s = _session((f, "execute"))
    threading.Timer(0.05, f.set_exception, [_remote_error()]).start()

    s.report_at_exit(5)

    err = capsys.readouterr().err
    assert "an error was not reported before the program ended" in err
    assert "running a graph failed: index out of range" in err


def test_an_error_stored_earlier_is_printed(capsys):
    f = Future()
    s = _session((f, "upload"))
    f.set_exception(_remote_error())

    s.report_at_exit(5)

    assert "uploading a tensor failed" in capsys.readouterr().err


def test_an_error_already_raised_is_not_repeated(capsys):
    """The program caught it (say an OOM it recovered from): not news at exit."""
    f = Future()
    s = _session((f, "execute"))
    f.set_exception(_remote_error())
    try:
        s.raise_async()
    except ReflopRemoteError:
        pass

    s.report_at_exit(5)

    assert capsys.readouterr().err == ""


def test_requests_still_running_are_counted_not_waited_for_forever(capsys):
    s = _session((Future(), "execute"), (Future(), "execute"))

    s.report_at_exit(0.05)

    assert "exiting with 2 request(s) still running" in capsys.readouterr().err


def test_a_clean_exit_says_nothing(capsys):
    f = Future()
    s = _session((f, "execute"))
    f.set_result(None)

    s.report_at_exit(5)

    assert capsys.readouterr().err == ""


def test_end_to_end_a_script_whose_last_step_failed():
    """Before: exit status 0 and no output at all."""
    script = (
        "import torch, reflop\n"
        "dev = reflop.device(local='cpu')\n"
        "x = torch.ones(3, device=dev)\n"
        "y = x[torch.tensor([5]).to(dev)]\n"
        "reflop.step_boundary()\n"
    )
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT / "client/python"), str(ROOT / "server/worker")]),
               REFLOP_NO_KEYCHAIN="1", CUDA_VISIBLE_DEVICES="")
    r = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=120)

    assert "an error was not reported before the program ended" in r.stderr
    assert "index out of range" in r.stderr
