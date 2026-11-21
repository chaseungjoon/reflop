# SPDX-License-Identifier: Apache-2.0
"""Implicit CPU placement and reproducible RNG across sessions."""

import os

import pytest
import reflop
import torch
from reflop import client, session
from reflop._pb import graph_pb2 as gpb


@pytest.fixture(params=["local", "server"])
def open_device(request, monkeypatch):
    mode = request.param
    if mode == "server" and os.environ.get("REFLOP_E2E") != "1":
        pytest.skip("needs a running reflopd (REFLOP_E2E=1)")
    if mode == "local":
        def unexpected_connect(*args, **kwargs):
            pytest.fail("an in-process session tried to open a network connection")

        monkeypatch.setattr(client, "connect", unexpected_connect)
    monkeypatch.setattr(session, "_sessions", {})
    monkeypatch.setattr(session, "_default", [])
    monkeypatch.setattr(session, "_seed", None)

    def open_(index, kind="cpu"):
        return reflop.device(index, kind=kind, local="cpu" if mode == "local" else None)

    yield open_
    reflop.close()


def test_seed_reaches_every_open_session(open_device, monkeypatch):
    first = open_device(1)
    second = open_device(2)
    calls = []
    for dev in (first, second):
        conn = session.by_index(dev.index).conn
        original = conn.reseed

        def reseed(n, *, dev=dev, original=original):
            calls.append((dev.index, n))
            original(n)

        monkeypatch.setattr(conn, "reseed", reseed)

    reflop.seed(584)
    assert sorted(calls) == [(1, 584), (2, 584)]
    first_draws = [torch.randn(12, device=dev).cpu() for dev in (first, second)]
    reflop.seed(584)
    for dev, want in zip((first, second), first_draws, strict=True):
        assert torch.equal(torch.randn(12, device=dev).cpu(), want)


def test_new_sessions_use_the_last_reflop_seed(open_device):
    reflop.seed(72)
    first = open_device(1)
    want = torch.randn(12, generator=torch.Generator().manual_seed(72))
    assert torch.equal(torch.randn(12, device=first).cpu(), want)
    second = open_device(2)
    assert torch.equal(torch.randn(12, device=second).cpu(), want)

    reflop.seed(0)
    want_zero = torch.randn(12, device=first).cpu()
    third = open_device(3)
    assert torch.equal(torch.randn(12, device=third).cpu(), want_zero)


def test_implicit_cpu_session_keeps_the_explicit_default(open_device, monkeypatch):
    primary = open_device(1, kind="cuda")
    primary_session = session.by_index(primary.index)
    model = torch.nn.Linear(4, 2).to(primary)
    weights = {k: v.cpu() for k, v in model.state_dict().items()}
    reflop.save("session-default-regression", model.state_dict())

    reflop.seed(83)
    first = torch.randn(12, device=primary).cpu()
    with reflop.placement("remote-cpu"), monkeypatch.context() as placement_patch:
        if primary_session.gdevice.kind == gpb.DEVICE_CPU:
            # Exercise CPU session selection with CPU workers in CPU-only CI.
            placement_patch.setattr(primary_session, "gdevice", gpb.Device(kind=gpb.DEVICE_CUDA))
        cpu = torch.zeros(3, device=primary).device
    assert cpu != primary
    assert session.by_index(cpu.index).gdevice.kind == gpb.DEVICE_CPU
    reflop.seed(83)
    assert torch.equal(torch.randn(12, device=primary).cpu(), first)
    assert torch.ones(3, device="reflop").device == primary

    state = reflop.load("session-default-regression")
    assert all(t.device == primary for t in state.values())
    model.load_state_dict(state)
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor.cpu(), weights[name])
    reflop.delete_snapshot("session-default-regression")
    explicit_cpu = open_device(cpu.index + 1)
    assert session.current() is session.by_index(explicit_cpu.index)
    session.by_index(explicit_cpu.index).close()
    assert session.current() is primary_session
