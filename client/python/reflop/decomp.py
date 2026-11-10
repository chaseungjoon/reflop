# SPDX-License-Identifier: Apache-2.0
"""reflop's own decompositions (PLAN.md §11.2 step 2), tried before
PyTorch's table. They express ops that PyTorch only implements as fused
device kernels in terms of opset operators, keeping each op's exact output
layout so autograd formulas that consume them still work."""

from __future__ import annotations

import torch

aten = torch.ops.aten


def _thnn_fused_lstm_cell(input_gates, hidden_gates, cx, input_bias=None, hidden_bias=None):
    # Same math and workspace layout as aten/src/ATen/native/cuda/RNN.cu.
    gates = input_gates + hidden_gates
    if input_bias is not None:
        gates = gates + input_bias + hidden_bias
    i, f, g, o = gates.chunk(4, 1)
    i, f, g, o = i.sigmoid(), f.sigmoid(), g.tanh(), o.sigmoid()
    cy = f * cx + i * g
    hy = o * cy.tanh()
    return hy, cy, torch.cat([i, f, g, o], 1)


def _thnn_fused_lstm_cell_backward_impl(grad_hy, grad_cy, cx, cy, workspace, has_bias):
    i, f, g, o = workspace.chunk(4, 1)
    tcy = cy.tanh()
    gcy = torch.zeros_like(cy) if grad_cy is None else grad_cy
    if grad_hy is not None:
        gcy = gcy + grad_hy * o * (1 - tcy * tcy)
        go = grad_hy * tcy * o * (1 - o)
    else:
        go = torch.zeros_like(o)
    gi = gcy * g * i * (1 - i)
    gf = gcy * cx * f * (1 - f)
    gg = gcy * i * (1 - g * g)
    grad_gates = torch.cat([gi, gf, gg, go], 1)
    grad_cx = gcy * f
    grad_bias = grad_gates.sum(0) if has_bias else None
    return grad_gates, grad_cx, grad_bias


def _thnn_fused_gru_cell(input_gates, hidden_gates, hx, input_bias=None, hidden_bias=None):
    if input_bias is not None:
        input_gates = input_gates + input_bias
        hidden_gates = hidden_gates + hidden_bias
    ir, iz, inn = input_gates.chunk(3, 1)
    hr, hz, hn = hidden_gates.chunk(3, 1)
    r = (ir + hr).sigmoid()
    z = (iz + hz).sigmoid()
    n = (inn + r * hn).tanh()
    hy = n + z * (hx - n)
    return hy, torch.cat([r, z, n, hx, hn], 1)


def _thnn_fused_gru_cell_backward(grad_hy, workspace, has_bias):
    r, z, n, hx, hn = workspace.chunk(5, 1)
    ghx = grad_hy * z
    gn = grad_hy * (1 - z)
    gz = grad_hy * (hx - n)
    gin = gn * (1 - n * n)
    ghn = gin * r
    gr = gin * hn
    gir = gr * r * (1 - r)
    giz = gz * z * (1 - z)
    grad_input_gates = torch.cat([gir, giz, gin], 1)
    grad_hidden_gates = torch.cat([gir, giz, ghn], 1)
    gib = grad_input_gates.sum(0) if has_bias else None
    ghb = grad_hidden_gates.sum(0) if has_bias else None
    return grad_input_gates, grad_hidden_gates, ghx, gib, ghb


TABLE = {
    aten._thnn_fused_lstm_cell.default: _thnn_fused_lstm_cell,
    aten._thnn_fused_lstm_cell_backward_impl.default: _thnn_fused_lstm_cell_backward_impl,
    aten._thnn_fused_gru_cell.default: _thnn_fused_gru_cell,
    aten._thnn_fused_gru_cell_backward.default: _thnn_fused_gru_cell_backward,
}
