# SPDX-License-Identifier: Apache-2.0
"""A value read after a request that failed asynchronously raises that
request's error; it is never returned stale.

Regression (benchmark B2): executes are sent without waiting. When one failed
on the server, the handles it would have rebound kept their old contents, and
the next fetch returned that old data with no error: a captured step whose
replay failed returned the previous batch's loss exactly. The error surfaced
one call later. The in-process worker answers synchronously, so these tests
deliver its answers from another thread after a delay, in order, as the
network does.
"""

from __future__ import annotations

import os
import queue
import threading
import time

import pytest
import reflop
import torch
import torch.nn as nn
import torch.nn.functional as F
from reflop import session as rsession

DEVICE = "cuda" if torch.cuda.is_available() and os.environ.get("REFLOP_TEST_DEVICE", "cuda") == "cuda" else "cpu"
INDEX = 11
DELAY_S = 0.02


@pytest.fixture(scope="module", params=["local", "server"])
def dev(request):
    """local: the in-process worker, its answers delayed and delivered from
    another thread. server: a running reflopd over the network (REFLOP_E2E=1,
    a paired client)."""
    if request.param == "server":
        if os.environ.get("REFLOP_E2E") != "1":
            pytest.skip("needs a running reflopd (REFLOP_E2E=1)")
        d = reflop.device(index=INDEX + 1)
        sess = rsession.by_index(INDEX + 1)
        yield d
        try:
            sess.close()
        except Exception:  # noqa: BLE001 - a test may leave an error unread
            pass
        return
    d = reflop.device(index=INDEX, local=DEVICE)
    sess = rsession.by_index(INDEX)
    conn = sess.conn
    deliver = conn._deliver
    q: queue.Queue = queue.Queue()

    def pump():
        while (env := q.get()) is not None:
            time.sleep(DELAY_S)
            deliver(env)

    t = threading.Thread(target=pump, daemon=True)
    t.start()
    conn._deliver = q.put  # every answer now arrives later, in order
    yield d
    try:
        sess.close()
    except Exception:  # noqa: BLE001 - a test may leave an error unread
        pass
    q.put(None)
    t.join(5)


@pytest.fixture(autouse=True)
def clean(dev):
    """Each test starts with no error pending and every answer delivered."""
    try:
        reflop.synchronize()
    except Exception:  # noqa: BLE001
        pass
    yield


def _fail_async(dev, target):
    """Send a step that rebinds `target` and fails on the server (an index
    out of range, found at run time), without waiting for it."""
    src = torch.ones(4).to(dev)
    bad = torch.tensor([7]).to(dev)
    target.add_(src[bad].sum())
    reflop.step_boundary()


def _flushed_zeros(dev, n=3):
    a = torch.zeros(n).to(dev)
    reflop.step_boundary()
    reflop.synchronize()
    return a


def test_answers_really_arrive_later(dev):
    # The fixture's premise: without it these tests would pass on the old code.
    if dev.index == INDEX + 1:
        pytest.skip("a real connection is asynchronous anyway")
    before = time.monotonic()
    torch.ones(2).to(dev).sum().item()
    assert time.monotonic() - before >= DELAY_S


def test_fetch_after_a_failed_step_raises(dev):
    a = _flushed_zeros(dev)
    _fail_async(dev, a)
    with pytest.raises(reflop.ReflopRemoteError, match="running a graph failed") as e:
        a.cpu()  # before: [0, 0, 0] and no error
    assert e.value.code == 9 or "INDEX_OUT_OF_BOUNDS" in str(e.value)


def test_unrelated_value_after_a_failed_step_raises(dev):
    a = _flushed_zeros(dev)
    b = torch.full((3,), 5.0).to(dev)
    reflop.step_boundary()
    reflop.synchronize()
    _fail_async(dev, a)
    with pytest.raises(reflop.ReflopRemoteError, match="index"):
        b.sum().item()


def test_returned_output_after_a_failed_step_raises(dev):
    # The value is computed in the same flush and returned with the execute
    # (no separate fetch).
    a = _flushed_zeros(dev)
    _fail_async(dev, a)
    with pytest.raises(reflop.ReflopRemoteError):
        (torch.ones(3).to(dev) * 2).cpu()


@pytest.mark.parametrize("read", ["item", "bool", "tolist", "numpy", "nonzero", "fetch"])
def test_every_kind_of_read_raises(dev, read):
    a = _flushed_zeros(dev)
    _fail_async(dev, a)
    with pytest.raises(reflop.ReflopRemoteError):
        if read == "item":
            a.sum().item()
        elif read == "bool":
            bool(a.sum() > -1)
        elif read == "tolist":
            a.tolist()
        elif read == "numpy":
            a.numpy()
        elif read == "nonzero":
            torch.nonzero(a + 1)
        else:
            reflop.fetch({"a": a})


def test_print_shows_the_error_and_the_next_read_still_raises(dev):
    # repr must not raise (debuggers, tracebacks); it shows the error instead
    # of stale values, and does not swallow it.
    a = _flushed_zeros(dev)
    _fail_async(dev, a)
    text = repr(a)
    assert "unavailable" in text and "running a graph failed" in text and "0." not in text
    with pytest.raises(reflop.ReflopRemoteError, match="running a graph failed"):
        a.sum().item()
    assert a.sum().item() == 0  # raised once


def test_the_error_is_raised_once_and_the_failed_step_left_nothing(dev):
    a = _flushed_zeros(dev)
    _fail_async(dev, a)
    with pytest.raises(reflop.ReflopRemoteError):
        a.cpu()
    # The failed step changed nothing on the server; the session still works.
    assert a.cpu().tolist() == [0.0, 0.0, 0.0]
    a.add_(1)
    assert a.cpu().tolist() == [1.0, 1.0, 1.0]


def test_synchronize_raises(dev):
    a = _flushed_zeros(dev)
    _fail_async(dev, a)
    with pytest.raises(reflop.ReflopRemoteError):
        reflop.synchronize()


def test_captured_step_whose_replay_fails_does_not_return_the_previous_loss(dev):
    # B2 in general form: the replay is sent without waiting and the step
    # returns the same output objects as before.
    w = torch.arange(1.0, 9.0).to(dev)

    def step(x, idx):
        return (x[idx] * w).sum()

    s = reflop.capture(step)
    x = torch.arange(16.0).reshape(2, 8)
    good = torch.tensor([1])
    want = float((x[good] * torch.arange(1.0, 9.0)).sum())
    for _ in range(4):
        assert s(x.to(dev), good).item() == want
    assert s.replays >= 2
    loss = s(x.to(dev), torch.tensor([5]))  # same shape and dtype, out of range
    with pytest.raises(reflop.ReflopRemoteError, match="index"):
        loss.item()  # before: the previous loss, exactly
    assert s(x.to(dev), good).item() == want  # later replays work


def test_out_of_memory_keeps_its_type(dev):
    big = torch.empty(1 << 40, dtype=torch.uint8, device=dev)
    big.add_(1)
    reflop.step_boundary()  # rejected at validation, answered later
    with pytest.raises(torch.OutOfMemoryError):
        torch.ones(2).to(dev).sum().item()
    assert torch.ones(2, device=dev).sum().item() == 2


def test_save_after_a_failed_step_raises(dev):
    if "snapshots" not in rsession.by_index(dev.index).caps.capabilities:
        pytest.skip("the server does not offer snapshots")
    a = _flushed_zeros(dev)
    _fail_async(dev, a)
    with pytest.raises(reflop.ReflopRemoteError):
        reflop.save("async-errors-test", {"a": a})


def test_training_without_errors_is_unaffected(dev):
    torch.manual_seed(0)
    local = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))
    remote = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))
    remote.load_state_dict(local.state_dict())
    remote = remote.to(dev)
    o_l = torch.optim.SGD(local.parameters(), 0.1, momentum=0.9)
    o_r = torch.optim.SGD(remote.parameters(), 0.1, momentum=0.9)

    def step(m, o, x, y):
        o.zero_grad(set_to_none=False)
        loss = F.cross_entropy(m(x), y)
        loss.backward()
        o.step()
        return loss

    s = reflop.capture(lambda x, y: step(remote, o_r, x, y))
    g = torch.Generator().manual_seed(1)
    for i in range(8):
        x, y = torch.randn(32, 16, generator=g), torch.randint(0, 4, (32,), generator=g)
        lo = step(local, o_l, x, y)
        if i < 4:
            out = step(remote, o_r, x.to(dev), y.to(dev))
            reflop.step_boundary()
        else:
            out = s(x.to(dev), y.to(dev))
        assert abs(out.item() - lo.item()) < (1e-4 if DEVICE == "cuda" else 1e-5)
