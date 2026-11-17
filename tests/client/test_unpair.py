# SPDX-License-Identifier: Apache-2.0
"""Issue #6: `reflop unpair` forgets a server and tells it, which revokes
this machine's key there (ADR-025); `reflop pair` with a server this machine
is still paired with says to unpair first; a server that no longer accepts
the key says to run `reflop unpair`."""

import pytest
from reflop import client, config
from reflop._pb import auth_pb2 as apb
from reflop._pb import compute_pb2 as cpb
from reflop.cli.main import main
from reflop.errors import NotPairedError, ReflopError, ServerKeyMismatch

FP = bytes(range(32))


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("REFLOP_CLIENT_HOME", str(tmp_path))
    monkeypatch.delenv("REFLOP_SERVER", raising=False)
    cfg = config.ClientConfig()
    for name, fp in (("gpu", FP), ("other", b"\x09" * 32)):
        cfg.servers[name] = config.ServerEntry(name=name, transport="tls", addresses=["127.0.0.1:7460"],
                                               fingerprint=fp.hex(), identity="alice", identity_id="00" * 16)
    cfg.default = "gpu"
    cfg.save()
    return tmp_path


class FakeFramer:
    """Plays the server's side of one connection."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.sent = []
        self.closed = False

    def write(self, env):
        self.sent.append(env.WhichOneof("body"))

    def read(self):
        if not self.replies:
            raise EOFError("connection closed")
        r = self.replies.pop(0)
        if isinstance(r, BaseException):
            raise r
        return r

    def close(self):
        self.closed = True

    def diagnostics(self):
        return ""


def hello(fp=FP):
    return cpb.Envelope(hello_response=cpb.HelloResponse(protocol_version=1, server_nonce=b"\x07" * 32,
                                                         server_key_fingerprint=fp))


def auth_ok():
    return cpb.Envelope(auth_result=apb.AuthResult(identity_name="alice"))


def err(code, msg="x"):
    return cpb.Envelope(error=cpb.Error(code=code, message=msg))


def serve(monkeypatch, *replies):
    fr = FakeFramer(replies)
    monkeypatch.setattr(client, "dial_tls", lambda addrs, fp: fr)
    return fr


@pytest.mark.parametrize("replies,outcome", [
    ([hello(), auth_ok(), cpb.Envelope(unpaired=cpb.Unpaired(identity_name="alice"))], client.UNPAIR_REVOKED),
    ([hello(), err(cpb.ERROR_UNAUTHENTICATED, "authentication failed")], client.UNPAIR_ALREADY),
    ([hello(), auth_ok(), err(cpb.ERROR_PROTOCOL, "unexpected message")], client.UNPAIR_UNSUPPORTED),
    ([hello(b"\x05" * 32)], client.UNPAIR_MISMATCH),
    ([hello(), auth_ok(), err(cpb.ERROR_INTERNAL, "disk full")], client.UNPAIR_FAILED),
    ([hello(), auth_ok()], client.UNPAIR_UNREACHABLE),  # closed before replying
])
def test_unpair_outcomes_always_forget(home, monkeypatch, replies, outcome):
    fr = serve(monkeypatch, *replies)
    entry, got, _ = client.unpair("gpu")
    assert (entry.name, got) == ("gpu", outcome)
    assert fr.closed
    if outcome == client.UNPAIR_REVOKED:
        assert fr.sent == ["hello", "authenticate", "unpair"]
    if outcome == client.UNPAIR_MISMATCH:
        assert fr.sent == ["hello"]  # nothing signed for a server with another key
    cfg = config.ClientConfig.load()
    assert "gpu" not in cfg.servers and cfg.default == "other"


@pytest.mark.parametrize("exc,outcome", [
    (ReflopError("could not reach the server"), client.UNPAIR_UNREACHABLE),
    (ServerKeyMismatch("different key"), client.UNPAIR_MISMATCH),
])
def test_unpair_when_dial_fails(home, monkeypatch, exc, outcome):
    def dial(addrs, fp):
        raise exc
    monkeypatch.setattr(client, "dial_tls", dial)
    _, got, _ = client.unpair()  # the default server
    assert got == outcome
    assert "gpu" not in config.ClientConfig.load().servers


def test_unpair_local_does_not_connect_and_drops_pinned_ssh_keys(home, monkeypatch):
    monkeypatch.setattr(client, "dial_tls", lambda *a: pytest.fail("connected with --local"))
    pinned = home / "ssh_known_hosts" / FP.hex()
    pinned.parent.mkdir()
    pinned.write_text("reflop-x ssh-ed25519 AAAA\n")
    _, got, _ = client.unpair("gpu", local_only=True)
    assert got == client.UNPAIR_SKIPPED
    assert not pinned.exists()


def test_unpair_unknown_server(home):
    with pytest.raises(RuntimeError, match="no paired server named 'nope'"):
        client.unpair("nope")
    assert set(config.ClientConfig.load().servers) == {"gpu", "other"}


def invite(fp):
    inv = apb.InviteCode(version=1, transport=apb.TRANSPORT_TLS, addresses=["127.0.0.1:7460"],
                         server_key_fingerprint=fp, invite_id=b"\x02" * 16, secret=b"\x03" * 32,
                         expires_unix=0, identity_name="alice")
    import base64

    return client.INVITE_PREFIX + base64.b32encode(inv.SerializeToString()).decode().rstrip("=").lower()


def test_pair_with_a_paired_server_says_unpair_first(home, monkeypatch):
    # Refused before the key is loaded or the invite is sent, so it stays usable.
    monkeypatch.setattr(client.keys, "load_or_create", lambda: pytest.fail("touched the key"))
    monkeypatch.setattr(client, "dial_tls", lambda *a: pytest.fail("connected"))
    with pytest.raises(ReflopError) as e:
        client.pair(invite(FP))
    assert "reflop unpair gpu" in str(e.value) and "reflopd unpair alice" in str(e.value)


def test_revoked_key_says_run_unpair(home, monkeypatch):
    serve(monkeypatch, hello(), err(cpb.ERROR_UNAUTHENTICATED, "authentication failed"))
    with pytest.raises(NotPairedError) as e:
        client.admin("gpu")
    assert "reflop unpair gpu" in str(e.value)
    assert "gpu" in config.ClientConfig.load().servers  # only told, not forgotten


def test_cli_unpair_and_servers_remove(home, monkeypatch, capsys):
    serve(monkeypatch, hello(), auth_ok(), cpb.Envelope(unpaired=cpb.Unpaired(identity_name="alice")))
    assert main(["unpair", "--yes"]) == 0
    out = capsys.readouterr().out
    assert "forgot gpu" in out and "reflopd unpair alice" in out
    # `servers remove` used to drop only the local entry (issue #6); now it unpairs.
    monkeypatch.setattr(client, "dial_tls", lambda *a: (_ for _ in ()).throw(ReflopError("offline")))
    assert main(["servers", "remove", "other"]) == 0
    assert "could not reach the server" in capsys.readouterr().out
    assert config.ClientConfig.load().servers == {}
