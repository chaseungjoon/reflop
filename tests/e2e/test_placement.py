# SPDX-License-Identifier: Apache-2.0
"""Placement: ops that run on the client still update reFLOP tensors.

Regression: running an op locally ("local" placement, or "auto" when its cost
model prefers the client) works on fetched copies of the reFLOP inputs. An
op that wrote to one (x.add_(1), out=x, a write through a view) changed the
copy and the reFLOP tensor kept its old value, silently. In "auto", the
default, a small reFLOP tensor updated from large client tensors
(x.scatter_add_(0, big_index, big_src)) was lost; in "local", every in-place
op was, including autograd's gradient accumulation and the optimizer step.
"""

from __future__ import annotations

import copy
import os
import warnings

import pytest
import reflop
import torch
import torch.nn as nn
import torch.nn.functional as F
from reflop.tensor import RemoteTensor

DEVICE = "cuda" if torch.cuda.is_available() and os.environ.get("REFLOP_TEST_DEVICE", "cuda") == "cuda" else "cpu"
INDEX = 13
MODES = ["auto", "local", "remote"]


@pytest.fixture(scope="module")
def dev():
    return reflop.device(index=INDEX, local=DEVICE)


def _stats():
    reflop.synchronize()
    return dict(reflop.stats()[INDEX])


def _big_scatter(x):
    # A small target updated from large client tensors: what "auto" runs
    # on the client (fetching 8 values beats uploading 2M indices).
    idx = torch.randint(0, 8, (2_000_000,), generator=torch.Generator().manual_seed(0))
    return x.scatter_add_(0, idx, torch.ones(2_000_000))


WRITES = {
    "add_": lambda x: x.add_(1),
    "mul_ by a cpu tensor": lambda x: x.mul_(torch.full((8,), 2.0)),
    "clamp_": lambda x: x.clamp_(min=3),
    "out=": lambda x: torch.add(x, 1, out=x),
    "out= of another op": lambda x: torch.mul(x, x, out=x),
    "through a slice": lambda x: x[2:4].add_(5),
    "through a reshaped view": lambda x: x.view(2, 4)[1].mul_(3),
    "through a transpose": lambda x: x.view(2, 4).t()[0].sub_(1),
    "index_add_ with a cpu index": lambda x: x.index_add_(0, torch.tensor([0, 1]), torch.ones(2)),
    "masked_fill_": lambda x: x.masked_fill_(torch.arange(8) % 2 == 0, -1.0),
    "scatter_add_ from big cpu tensors": _big_scatter,
    "two writes in a row": lambda x: x.add_(1).mul_(2),
}


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", list(WRITES))
def test_writes_reach_the_reflop_tensor(dev, mode, name):
    f = WRITES[name]
    want = torch.arange(8.0)
    f(want)
    x = torch.arange(8.0).to(dev)
    with warnings.catch_warnings(), reflop.placement(mode):
        warnings.simplefilter("ignore")
        f(x)
    assert torch.equal(x.cpu(), want)


@pytest.mark.parametrize("mode", MODES)
def test_an_in_place_op_returns_the_tensor_itself(dev, mode):
    x = torch.zeros(3).to(dev)
    with reflop.placement(mode):
        r = x.add_(1)
    assert r is x


def test_auto_still_runs_the_big_scatter_on_the_client(dev):
    x = torch.zeros(8).to(dev)
    before = _stats()
    _big_scatter(x)
    total = x.sum().item()
    after = _stats()
    assert total == 2_000_000
    # Uploading the 2M indices and values would be 24 MB.
    assert after["bytes_up"] - before["bytes_up"] < 1 << 20


def test_local_placement_still_computes_on_the_client(dev):
    x = torch.arange(8.0).to(dev)
    reflop.synchronize()
    before = _stats()
    with reflop.placement("local"):
        y = x * 2 + 1
    assert type(y) is torch.Tensor  # a client tensor, as "local" promises
    assert torch.equal(y, torch.arange(8.0) * 2 + 1)
    assert _stats()["executes"] == before["executes"]


def test_local_placement_writes_without_running_the_op_on_the_server(dev):
    # The functional form runs on the client; only its result is sent back.
    x = torch.arange(8.0).to(dev)
    with reflop.placement("local"):
        x.exp_()
    torch.testing.assert_close(x.cpu(), torch.arange(8.0).exp())


def test_a_view_stays_a_view_and_reads_only_its_part(dev):
    x = torch.arange(1024.0).to(dev)
    reflop.synchronize()
    with reflop.placement("local"):
        v = x[10:12]
    assert isinstance(v, RemoteTensor)
    before = _stats()
    assert v.cpu().tolist() == [10.0, 11.0]
    assert _stats()["bytes_down"] - before["bytes_down"] == 8  # not the 4 KiB base


@pytest.mark.parametrize("mode", ["auto", "local"])
def test_a_cpu_target_with_reflop_inputs(dev, mode):
    y = torch.ones(4).to(dev)
    c = torch.zeros(4)
    with reflop.placement(mode):
        c.add_(y)
        torch.mul(y[:2], 3, out=c[2:])
    assert c.tolist() == [1.0, 1.0, 3.0, 3.0]
    assert type(c) is torch.Tensor


def test_training_under_local_placement_matches_pytorch(dev):
    torch.manual_seed(0)
    local = nn.Sequential(nn.Linear(8, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Linear(16, 3))
    remote = copy.deepcopy(local).to(dev)
    o_l = torch.optim.SGD(local.parameters(), 0.1, momentum=0.9)
    o_r = torch.optim.SGD(remote.parameters(), 0.1, momentum=0.9)
    g = torch.Generator().manual_seed(1)
    for _ in range(4):
        x, y = torch.randn(32, 8, generator=g), torch.randint(0, 3, (32,), generator=g)
        o_l.zero_grad()
        lo = F.cross_entropy(local(x), y)
        lo.backward()
        o_l.step()
        with reflop.placement("local"):
            o_r.zero_grad()
            out = F.cross_entropy(remote(x.to(dev)), y.to(dev))
            out.backward()
            o_r.step()
        assert abs(out.item() - lo.item()) < 1e-5
    for (k, r), lo in zip(remote.state_dict().items(), local.state_dict().values(), strict=True):
        torch.testing.assert_close(r.detach().cpu(), lo.detach(), atol=1e-5, rtol=1e-5, msg=k)


def test_gradient_accumulation_under_local_placement(dev):
    w = torch.ones(4).to(dev).requires_grad_()
    with reflop.placement("local"):
        for _ in range(3):
            (w * torch.arange(4.0)).sum().backward()
    assert w.grad.cpu().tolist() == [0.0, 3.0, 6.0, 9.0]
