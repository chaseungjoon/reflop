# SPDX-License-Identifier: Apache-2.0
"""Unflushed tensor reads that exceed the inline execute response budget."""

import os
from concurrent.futures import Future

import pytest
import reflop
import torch
from reflop import session
from reflop._pb import compute_pb2 as cpb
from reflop.tensor import value_of
from reflop.trace import NODE

MIB = 1 << 20
INDEX = 17


@pytest.fixture(params=["local", "server"])
def dev(request):
    if request.param == "server" and os.environ.get("REFLOP_E2E") != "1":
        pytest.skip("needs a running reflopd (REFLOP_E2E=1)")
    # Exercise CUDA transfers in the worker process when running over TLS.
    d = reflop.device(INDEX, local="cpu" if request.param == "local" else None)
    yield d
    session.by_index(INDEX).close()


@pytest.mark.parametrize("size", [64 * MIB, 65 * MIB])
def test_cpu_on_large_unflushed_tensor(dev, size):
    count = size // torch.int32.itemsize
    x = torch.arange(count, dtype=torch.int32, device=dev)
    assert x.reflop_handle is None
    assert torch.equal(x.cpu(), torch.arange(count, dtype=torch.int32))
    assert x[-1].item() == count - 1


@pytest.fixture
def small_frame_session(monkeypatch):
    d = reflop.device(INDEX, local="cpu")
    s = session.by_index(INDEX)
    # Force the same routing decisions with small tensors in CI.
    limits = type(s.limits)()
    limits.CopyFrom(s.limits)
    limits.max_frame_bytes = MIB
    monkeypatch.setattr(s, "limits", limits)
    yield d, s
    s.close()


def test_selects_inline_and_chunked_returns_within_one_budget(small_frame_session, monkeypatch):
    dev, sess = small_frame_session
    fetched = []
    fetch = sess.fetch_handles

    def track_fetch(handles):
        fetched.extend(handles)
        return fetch(handles)

    monkeypatch.setattr(sess, "fetch_handles", track_fetch)
    budget = sess.limits.max_frame_bytes // 4
    sizes = [budget - 1024, budget + 1024, budget - 1024]
    want = [torch.full((n,), i + 1, dtype=torch.uint8) for i, n in enumerate(sizes)]
    inputs = [torch.full_like(t, i + 1, device=dev) for i, t in enumerate(want)]

    assert all(value_of(t).kind == NODE for t in inputs)
    got = sess.flush([value_of(t) for t in inputs])
    assert all(torch.equal(a, b) for a, b in zip(got, want, strict=True))
    assert fetched == [inputs[1].reflop_handle, inputs[2].reflop_handle]


@pytest.mark.parametrize("fail_fetch", [False, True])
def test_temp_handle_is_freed_after_fetch(small_frame_session, monkeypatch, fail_fetch):
    dev, sess = small_frame_session
    count = sess.limits.max_frame_bytes // 4 + 1024
    base = torch.tensor([7], dtype=torch.uint8, device=dev)
    sess.synchronize()
    handles = sess.conn.worker.handles.t
    before = set(handles)
    request = sess.conn.request_stream
    fetched = []

    def request_stream(env):
        fetched.extend(env.fetch.handles)
        assert set(fetched).isdisjoint(before)
        assert all(h in handles for h in fetched)
        if fail_fetch:
            future = Future()
            future.set_exception(reflop.ReflopError("test fetch failed"))
            return future
        return request(env)

    monkeypatch.setattr(sess.conn, "request_stream", request_stream)
    view = base.expand(count)
    if fail_fetch:
        with pytest.raises(reflop.ReflopError, match="test fetch failed"):
            view.cpu()
    else:
        assert torch.equal(view.cpu(), torch.full((count,), 7, dtype=torch.uint8))
    assert len(fetched) == 1
    assert set(handles) == before
    assert sess.conn.worker.handles.get(base.reflop_handle).item() == 7


def test_failed_execute_is_reported_before_fetch(small_frame_session, monkeypatch):
    dev, sess = small_frame_session
    count = sess.limits.max_frame_bytes // 4 + 1024
    base = torch.tensor([7], dtype=torch.uint8, device=dev)
    bad_index = torch.tensor([1]).to(dev)
    sess.synchronize()

    def unexpected_fetch(env):
        pytest.fail("the client fetched a handle from a failed execute")

    monkeypatch.setattr(sess.conn, "request_stream", unexpected_fetch)
    with pytest.raises(reflop.ReflopRemoteError, match="index"):
        base[bad_index].expand(count).cpu()
    assert (base + 1).item() == 8


def test_oversize_read_keeps_the_fetch_limit(small_frame_session):
    dev, sess = small_frame_session
    limit = MIB
    sess.limits.max_fetch_bytes = limit
    sess.conn.worker.limits.max_fetch_bytes = limit
    base = torch.tensor([7], dtype=torch.uint8, device=dev)
    sess.synchronize()
    before = set(sess.conn.worker.handles.t)
    with pytest.raises(reflop.ReflopRemoteError, match="too large to return") as err:
        base.expand(limit + 1).cpu()
    assert err.value.stage == cpb.STAGE_E1_BINDINGS
    assert set(sess.conn.worker.handles.t) == before
