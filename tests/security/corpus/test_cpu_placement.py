# SPDX-License-Identifier: Apache-2.0
"""Implicit CPU placement stays with the selected Owner and Guest identity."""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import reflop
import torch
from reflop import client, session
from reflop._pb import auth_pb2 as apb
from reflop._pb import graph_pb2 as gpb
from reflop.config import ClientConfig, ServerEntry


class StubConnection(client.RemoteConnection):
    """Exercise the connection and placement logic with stub authentication."""

    def _open(self):
        self.auth = apb.AuthResult(identity_id=self.key)
        self.auth.capabilities.devices.add(device=gpb.Device(kind=gpb.DEVICE_CUDA))
        if self.entry.extra.get("cpu", True):
            self.auth.capabilities.devices.add(device=gpb.Device(kind=gpb.DEVICE_CPU))

    def open_session(self):
        self.session_id = b"stub"

    def _keepalive(self):
        pass

    def close(self):
        self.closed = True


@pytest.fixture
def owners(tmp_path, monkeypatch):
    monkeypatch.setenv("REFLOP_CLIENT_HOME", str(tmp_path))
    monkeypatch.delenv("REFLOP_SERVER", raising=False)
    monkeypatch.delenv("REFLOP_LOCAL", raising=False)
    monkeypatch.setattr(client, "RemoteConnection", StubConnection)
    monkeypatch.setattr(client.keys, "load_or_create", lambda: b"a" * 16)
    monkeypatch.setattr(session, "_sessions", {})
    monkeypatch.setattr(session, "_default", [])
    monkeypatch.setattr(session, "_seed", None)
    alpha = ServerEntry("alpha", "tls", ["alpha.test:7460"], "aa" * 32, "guest")
    beta = ServerEntry("beta", "tls", ["beta.test:7460"], "bb" * 32, "guest")
    cfg = ClientConfig(default="beta", servers={"alpha": alpha, "beta": beta})
    cfg.save()
    yield cfg
    reflop.close()


def test_existing_cpu_from_another_owner_is_ignored(owners):
    beta = reflop.device(1, server="beta", kind="cpu")
    alpha = reflop.device(2, server="alpha", kind="cuda")
    with reflop.placement("remote-cpu"):
        actual = torch.empty((4,), device=alpha).device
        assert actual != beta
        assert torch.empty((4,), device=alpha).device == actual
        assert torch.empty((4,), device=beta).device == beta
    assert session.by_index(actual.index).conn.entry.name == "alpha"
    assert session.current() is session.by_index(alpha.index)


def test_open_cpu_uses_the_pinned_connection_after_config_changes(owners, monkeypatch):
    alpha = reflop.device(2, server="alpha", kind="cuda")
    conn = session.by_index(alpha.index).conn
    owners.servers["alpha"] = owners.servers["beta"]
    owners.save()

    def unexpected_key_load():
        pytest.fail("implicit CPU placement must use the original Guest key")

    monkeypatch.setattr(client.keys, "load_or_create", unexpected_key_load)
    with reflop.placement("remote-cpu"):
        actual = torch.empty((4,), device="reflop").device
    peer = session.by_index(actual.index).conn
    assert peer.entry == conn.entry
    assert peer.key == conn.key


def test_existing_cpu_uses_the_authenticated_guest_identity(owners, monkeypatch):
    alpha = reflop.device(2, server="alpha", kind="cuda")
    monkeypatch.setattr(client.keys, "load_or_create", lambda: b"b" * 16)
    foreign = reflop.device(3, server="alpha", kind="cpu")
    with reflop.placement("remote-cpu"):
        actual = torch.empty((4,), device=alpha).device
    assert actual != foreign
    assert session.by_index(actual.index).conn.auth.identity_id == b"a" * 16


def test_missing_owner_cpu_fails_without_switching_servers(owners):
    owners.servers["alpha"].extra["cpu"] = False
    owners.save()
    reflop.device(1, server="beta", kind="cpu")
    alpha = reflop.device(2, server="alpha", kind="cuda")
    with reflop.placement("remote-cpu"), pytest.raises(reflop.ReflopError, match="Owner exposes no CPU"):
        torch.empty((4,), device=alpha)
    assert len(session._sessions) == 2


def test_new_cpu_with_a_changed_guest_identity_is_rejected(owners, monkeypatch):
    alpha = reflop.device(2, server="alpha", kind="cuda")
    open_ = StubConnection._open
    peers = []

    def changed_auth(conn):
        open_(conn)
        conn.auth.identity_id = b"b" * 16
        peers.append(conn)

    monkeypatch.setattr(StubConnection, "_open", changed_auth)
    with reflop.placement("remote-cpu"), pytest.raises(reflop.ReflopError, match="Guest identity changed"):
        torch.empty((4,), device=alpha)
    assert len(peers) == 1 and peers[0].closed
    assert list(session._sessions) == [alpha.index]


def test_concurrent_cpu_peers_keep_distinct_device_indexes(owners, monkeypatch):
    alpha = reflop.device(126, server="alpha", kind="cuda")
    beta = reflop.device(127, server="beta", kind="cuda")
    barrier = threading.Barrier(2, timeout=5)
    open_ = StubConnection.open_session

    def open_together(conn):
        barrier.wait()
        open_(conn)

    def cpu_for(gpu):
        with reflop.placement("remote-cpu"):
            return torch.empty((4,), device=gpu).device

    monkeypatch.setattr(StubConnection, "open_session", open_together)
    with ThreadPoolExecutor(max_workers=2) as pool:
        alpha_cpu, beta_cpu = pool.map(cpu_for, (alpha, beta))

    assert alpha_cpu != beta_cpu
    assert session.by_index(alpha_cpu.index).conn.entry.name == "alpha"
    assert session.by_index(beta_cpu.index).conn.entry.name == "beta"
