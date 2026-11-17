# SPDX-License-Identifier: Apache-2.0
"""Issue #8: when ssh fails or drops, the error carries what ssh said
(ADR-026) instead of only "connection lost" or nothing."""

import io
import os
import stat
import sys
from concurrent.futures import Future

import pytest
from reflop import client, config
from reflop.connection import Connection
from reflop.errors import ReflopError, SessionLostError

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the fake ssh is a shell script")


@pytest.fixture
def fake_ssh(tmp_path, monkeypatch):
    """Point REFLOP_SSH_COMMAND at a script that fails the way ssh does."""
    monkeypatch.setenv("REFLOP_CLIENT_HOME", str(tmp_path))
    monkeypatch.delenv("REFLOP_SERVER", raising=False)

    def make(stderr: str, code: int = 255):
        f = tmp_path / "ssh"
        f.write_text(f"#!/bin/sh\nprintf '%s' '{stderr}' >&2\nexit {code}\n")
        f.chmod(f.stat().st_mode | stat.S_IXUSR)
        monkeypatch.setenv("REFLOP_SSH_COMMAND", str(f))
    return make


def _entry():
    return config.ServerEntry(name="lab", transport="ssh", addresses=["me@gpu.example"], fingerprint="ab" * 32,
                              identity="alice", identity_id="00" * 16)


def test_ssh_failure_says_why(fake_ssh):
    fake_ssh("me@gpu.example: Permission denied (publickey).\n")

    with pytest.raises(ReflopError) as e:
        client.RemoteConnection(_entry())

    msg = str(e.value)
    assert "closed before the reflop handshake" in msg
    assert "(ssh: me@gpu.example: Permission denied (publickey).)" in msg
    assert "reflop unpair lab" in msg


def test_ssh_failure_with_no_output_keeps_the_old_message(fake_ssh):
    fake_ssh("")

    with pytest.raises(ReflopError) as e:
        client.RemoteConnection(_entry())

    assert "handshake. If sshd rejected" in str(e.value)
    assert "(ssh:" not in str(e.value)


def test_unpair_over_ssh_says_why(fake_ssh):
    fake_ssh("ssh: connect to host gpu.example port 22: Connection refused\n")

    outcome, detail = client._tell_unpair(_entry())

    assert outcome == client.UNPAIR_UNREACHABLE
    assert "Connection refused" in detail


def test_a_dropped_connection_names_the_ssh_reason():
    r, w = os.pipe()
    os.write(w, b"Connection to gpu.example closed by remote host.\n")
    os.close(w)
    fr = client.Framer(io.BytesIO(b""), io.BytesIO(), lambda: None)
    fr.stderr_tail = client.SshStderr(os.fdopen(r, "rb", buffering=0))
    conn = client.RemoteConnection.__new__(client.RemoteConnection)
    Connection.__init__(conn)
    conn.fr = fr
    fut = Future()
    conn._pending[1] = (fut, None, None)

    conn._read_loop(fr)

    with pytest.raises(SessionLostError) as e:
        fut.result(1)
    assert "connection to the server lost" in str(e.value)
    assert "(ssh: Connection to gpu.example closed by remote host.)" in str(e.value)


def test_a_connection_closed_on_purpose_does_not_wait_for_ssh():
    class Never:
        def text(self, wait=1.0):
            raise AssertionError("diagnostics read for a deliberate close")

    fr = client.Framer(io.BytesIO(b""), io.BytesIO(), lambda: None)
    fr.stderr_tail = Never()
    conn = client.RemoteConnection.__new__(client.RemoteConnection)
    Connection.__init__(conn)
    conn.fr, conn.closed = fr, True

    conn._read_loop(fr)


def test_tls_has_no_diagnostics():
    assert client.Framer(io.BytesIO(), io.BytesIO(), lambda: None).diagnostics() == ""


def test_debug_logs_connection_steps(fake_ssh, monkeypatch, capsys):
    fake_ssh("Permission denied (publickey).\n")
    monkeypatch.setenv("REFLOP_DEBUG", "1")

    with pytest.raises(ReflopError):
        client.RemoteConnection(_entry())

    err = capsys.readouterr().err
    assert "reflop[debug" in err
    assert "connecting to lab (ssh, me@gpu.example)" in err
    assert "ssh: starting" in err
    assert "ssh: Permission denied (publickey)." in err


def test_debug_off_logs_nothing(fake_ssh, monkeypatch, capsys):
    fake_ssh("Permission denied (publickey).\n")
    monkeypatch.setenv("REFLOP_DEBUG", "0")

    with pytest.raises(ReflopError):
        client.RemoteConnection(_entry())

    assert "reflop[debug" not in capsys.readouterr().err


def test_a_dropped_connection_is_marked_closed_before_waiting_for_ssh():
    """A request sent while the client waits for ssh's last words must take
    the reconnect path, not write to the dead pipe."""
    seen = {}

    class Slow:
        def text(self, wait=1.0):
            seen["closed"] = conn.closed
            return "bye"

    fr = client.Framer(io.BytesIO(b""), io.BytesIO(), lambda: None)
    fr.stderr_tail = Slow()
    conn = client.RemoteConnection.__new__(client.RemoteConnection)
    Connection.__init__(conn)
    conn.fr = fr

    conn._read_loop(fr)

    assert seen["closed"] is True
