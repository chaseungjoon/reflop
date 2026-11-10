# SPDX-License-Identifier: Apache-2.0
"""Client protection (§12, I9, I12)."""

from reflop.errors import sanitize


def test_strips_ansi_and_osc():
    s = "ok\x1b[31mred\x1b[0m \x1b]0;pwned\x07title \x1b]8;;http://evil\x1b\\link\x1b]8;;\x1b\\ \x07bell\x00nul\x9bcsi"
    out = sanitize(s)
    assert "\x1b" not in out and "\x07" not in out and "\x00" not in out and "\x9b" not in out
    assert "red" in out and "link" in out


def test_strips_bidi_overrides():
    assert "‮" not in sanitize("abc‮def")


def test_length_cap():
    assert len(sanitize("x" * 100000)) <= 4000


def test_events_never_act(capsys):
    from reflop._pb import compute_pb2 as cpb
    from reflop.connection import Connection

    c = Connection()
    c._deliver(cpb.Envelope(event=cpb.Event(kind=cpb.Event.KIND_OWNER_MESSAGE, text="\x1b[2Jrm -rf ~")))
    err = capsys.readouterr().err
    assert "\x1b" not in err and "rm -rf" in err  # printed as inert text only


def test_unsolicited_responses_ignored():
    from reflop._pb import compute_pb2 as cpb
    from reflop.connection import Connection

    c = Connection()
    c._deliver(cpb.Envelope(request_id=12345, end=True, pong=cpb.Pong()))  # nothing pending: no effect


def test_client_never_sends_environment(monkeypatch):
    """Recorded traffic contains no env vars, paths or hostnames (I12)."""
    import os
    import socket

    import reflop
    import torch
    from reflop.connection import LocalConnection

    secret = "REFLOP_TEST_SECRET_9f8a7b"
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", secret)
    sent = []
    orig = LocalConnection._send

    def spy(self, env):
        sent.append(env.SerializeToString())
        return orig(self, env)
    monkeypatch.setattr(LocalConnection, "_send", spy)
    dev = reflop.device(index=7, local="cpu")
    x = torch.ones(3).to(dev)
    (x * 2).sum().item()
    blob = b"".join(sent)
    for needle in (secret, os.getcwd(), socket.gethostname(), os.environ.get("HOME", "/nonexistent"), __file__):
        assert needle.encode() not in blob, needle
