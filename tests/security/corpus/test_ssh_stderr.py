# SPDX-License-Identifier: Apache-2.0
"""ssh's stderr is shown in error messages (issue #8, ADR-026). Everything
on it is untrusted, reflopd's stderr on the server included: it must not
reach the terminal with control sequences, and must not grow the client
without limit (invariant I9)."""

import os
import threading

from reflop.client import SshStderr


def _tail(data: bytes, chunks: int = 1) -> str:
    r, w = os.pipe()
    t = SshStderr(os.fdopen(r, "rb", buffering=0))

    def feed():
        step = max(1, len(data) // chunks)
        for i in range(0, len(data), step):
            os.write(w, data[i:i + step])
        os.close(w)
    threading.Thread(target=feed).start()
    return t.text(wait=10)


def test_control_sequences_are_stripped():
    out = _tail(b"\x1b]0;owned\x07\x1b[2J\x1b[31mPermission denied\x1b[0m\r\x08\xe2\x80\xae(publickey)\n")
    assert out == "Permission denied(publickey)"
    assert "\x1b" not in out and "\r" not in out


def test_only_the_last_lines_are_kept():
    out = _tail(b"".join(b"line %d\n" % i for i in range(1000)), chunks=50)
    lines = out.split("; ")
    assert len(lines) == SshStderr.LINES
    assert lines[-1] == "line 999"


def test_an_endless_line_is_cut_short():
    """No newline ever: memory stays bounded and the start is kept."""
    out = _tail(b"start" + b"A" * (8 << 20), chunks=2048)
    assert out.startswith("start")
    assert len(out) <= SshStderr.LINE_BYTES


def test_long_lines_are_cut_short():
    out = _tail((b"B" * 100_000 + b"\n") * 20, chunks=100)
    assert all(len(line) <= SshStderr.LINE_BYTES for line in out.split("; "))
    assert len(out) <= SshStderr.LINES * (SshStderr.LINE_BYTES + 2)


def test_invalid_utf8_does_not_break_the_reader():
    assert _tail(b"\xff\xfe bad bytes \xc3\n ok\n") .endswith("ok")
