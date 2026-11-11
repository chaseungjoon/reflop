# SPDX-License-Identifier: Apache-2.0
"""Two sessions at once (a GPU session and the server CPU): new tensors and
moves follow the session a tensor lives in."""

from __future__ import annotations

import os

import pytest
import reflop
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() and os.environ.get("REFLOP_TEST_DEVICE", "cuda") == "cuda" else "cpu"


@pytest.fixture(scope="module")
def devs():
    from reflop import session

    d0 = reflop.device(0, local=DEVICE)
    d1 = reflop.device(1, kind="cpu", local="cpu")
    yield d0, d1
    for i in (0, 1):
        session.by_index(i).close()


def sess(t):
    return t._rf_storage.session.device_index


def test_new_tensors_follow_the_tensor_session(devs):
    _, d1 = devs
    x = torch.randn(4, 3).to(d1)
    assert x.device == d1
    for t in (torch.zeros(3, device=x.device), torch.arange(3, device=x.device), torch.tensor([1.0], device=x.device),
              torch.randn(2, device=x.device)):
        assert sess(t) == 1
    assert (x + torch.ones(3, device=x.device)).sum().item() == pytest.approx((x.cpu() + 1).sum().item(), rel=1e-5)


class SeqNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(10, 8)
        self.pos = nn.Embedding(6, 8)
        self.lstm = nn.LSTM(8, 16, 2, batch_first=True)
        self.gru = nn.GRU(16, 8, batch_first=True)
        self.head = nn.Linear(8, 3)

    def forward(self, idx):
        x = self.emb(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        out, _ = self.gru(self.lstm(x)[0])
        return self.head(out[:, -1])


def test_lstm_gru_on_the_server_cpu_with_a_gpu_session_open(devs):
    _, d1 = devs
    idx = torch.randint(0, 10, (4, 6), generator=torch.Generator().manual_seed(0))
    y = torch.tensor([0, 1, 2, 1])
    losses = []
    for dev in (None, d1):
        torch.manual_seed(0)
        m = SeqNet() if dev is None else SeqNet().to(dev)
        o = torch.optim.SGD(m.parameters(), 0.1)
        run = []
        for _ in range(3):
            loss = F.cross_entropy(m(idx if dev is None else idx.to(dev)), y if dev is None else y.to(dev))
            o.zero_grad()
            loss.backward()
            o.step()
            run.append(loss.item())
        losses.append(run)
        if dev is not None:
            assert {sess(p) for p in m.parameters()} == {1}
    assert losses[0] == pytest.approx(losses[1], rel=1e-5)


def test_composite_ops_create_tensors_in_their_inputs_session(devs):
    # The causal mask is built in C++ with the inputs' C++ device (reflop:0).
    _, d1 = devs
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(2, 4, 8, 16, generator=g) for _ in range(3))
    qr, kr, vr = (t.to(d1).requires_grad_() for t in (q, k, v))
    out = F.scaled_dot_product_attention(qr, kr, vr, is_causal=True)
    assert sess(out) == 1
    out.sum().backward()
    assert sess(qr.grad) == 1
    want = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(out.cpu(), want, atol=1e-5)


def test_moves_between_sessions(devs):
    d0, d1 = devs
    m = nn.Linear(3, 2).to(d0)
    w = m.weight.detach().cpu()
    m.to(d1)
    assert {sess(p) for p in m.parameters()} == {1}
    m.to(d0)
    assert {sess(p) for p in m.parameters()} == {0}
    assert torch.equal(m.weight.detach().cpu(), w)
    a = torch.ones(3).to(d1)
    b = a.to(d0)
    assert sess(b) == 0 and sess(a.to(b.device)) == 0 and sess(b.to(a)) == 1
    assert a.to(a.device) is a
    with pytest.raises(RuntimeError, match="different server sessions"):
        a + b
    # Moves are differentiable; the gradient comes back to the source session.
    p = torch.ones(3).to(d0).requires_grad_()
    (p.to(d1) * 2).sum().backward()
    assert sess(p.grad) == 0 and p.grad.cpu().tolist() == [2.0, 2.0, 2.0]


def test_assigning_data(devs):
    d0, _ = devs
    m = nn.Linear(3, 2).to(d0)
    m.weight.data = torch.zeros(2, 3, device=d0)
    assert m.weight.abs().sum().item() == 0


def test_capture_uses_the_session_of_the_step(devs):
    # d1 was opened last; the step works in d0 with CPU arguments.
    d0, _ = devs
    m = nn.Linear(16, 4).to(d0)
    o = torch.optim.SGD(m.parameters(), 0.1, momentum=0.9)

    def step(x, y):
        o.zero_grad(set_to_none=True)
        loss = F.cross_entropy(m(x.to(d0)), y.to(d0))
        loss.backward()
        o.step()
        return loss

    s = reflop.capture(step)
    for _ in range(5):
        s(torch.randn(32, 16), torch.randint(0, 4, (32,))).item()
    assert s.replays >= 2


def test_to_a_reference_tensor_follows_its_session(devs):
    d0, d1 = devs
    x1 = torch.randn(3).to(d1)
    for t in (torch.arange(3).to(x1), torch.arange(3).to(tensor=x1), torch.ones(3).to(d0).to(x1),
              torch.ones(3).to(d0).to(tensor=x1)):
        assert sess(t) == 1 and t.dtype == x1.dtype
    c = x1.to(d1, None, False, True)
    assert c is not x1 and sess(c) == 1


def test_capture_does_not_replay_stale_uploads(devs):
    d0, _ = devs
    w = torch.ones(4).to(d0)

    def scaled(x):
        x.mul_(10)
        return (x.to(d0) * w).sum()

    s = reflop.capture(scaled)
    with pytest.warns(RuntimeWarning, match="modifies a CPU argument in place"):
        for i in range(5):
            assert s(torch.full((4,), float(i + 1))).item() == 40.0 * (i + 1)
