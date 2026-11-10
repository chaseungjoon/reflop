# SPDX-License-Identifier: Apache-2.0
"""Stage V5: cost and peak-memory estimates (PLAN.md §9.2, §9.6).

The FLOP estimate feeds the optional per-execute FLOP quota and the audit
log. The peak-memory estimate (liveness analysis over meta sizes) is
checked against the device's total memory: a graph that cannot fit fails
fast, before any allocation.
"""

from __future__ import annotations

import math

import torch

from ._pb import graph_pb2 as gpb
from .graph import Program, Ref


def _numel(t) -> int:
    return int(t.numel()) if isinstance(t, torch.Tensor) else 0


def _bytes(t) -> int:
    if not isinstance(t, torch.Tensor):
        return 0
    return int(t.numel()) * t.element_size()


def node_flops(node, ins: list, outs: list) -> int:
    kind = node.op.cost
    name = node.op.name
    out0 = outs[0] if outs else None
    if kind == "matmul":
        tens = [t for t in ins if isinstance(t, torch.Tensor)]
        if name.startswith(("aten.mm", "aten._int_mm", "aten.addmm", "aten._addmm_activation", "aten._scaled_mm")):
            a, b = [t for t in tens if t.dim() == 2][-2:]
            return 2 * a.shape[0] * a.shape[1] * b.shape[1]
        if name.startswith(("aten.bmm", "aten.baddbmm", "aten.addbmm")):
            a, b = [t for t in tens if t.dim() == 3][-2:]
            return 2 * a.shape[0] * a.shape[1] * a.shape[2] * b.shape[2]
        if name.startswith(("aten.dot", "aten.vdot")):
            return 2 * _numel(tens[0])
        if name.startswith(("aten.mv", "aten.addmv")):
            m = [t for t in tens if t.dim() == 2][0]
            return 2 * m.numel()
        return _numel(out0)
    if kind == "conv":
        if name == "aten.convolution.default" and out0 is not None:
            w = ins[1]  # (out, in/groups, *kernel)
            k = math.prod(w.shape[2:]) if w.dim() > 2 else 1
            return 2 * out0.numel() * (w.shape[1] if w.dim() > 1 else 1) * k
        return 2 * sum(_numel(t) for t in outs) * 9
    if kind == "attention":
        q, k = ins[0], ins[1]
        if isinstance(q, torch.Tensor) and isinstance(k, torch.Tensor) and q.dim() >= 2:
            f = 4 * q.shape[:-1].numel() * k.shape[-2] * q.shape[-1]
            return f * (2 if "backward" in name else 1)
        return _numel(out0)
    if kind == "reduce":
        return sum(_numel(t) for t in ins if isinstance(t, torch.Tensor))
    if kind == "sort":
        n = _numel(ins[0])
        return int(n * max(1.0, math.log2(max(n, 2))))
    if kind == "cubic":
        a = ins[0]
        if isinstance(a, torch.Tensor) and a.dim() >= 2:
            n = max(a.shape[-1], a.shape[-2])
            return a.shape[:-2].numel() * n ** 3
        return 0
    return sum(_numel(t) for t in outs)


def analyze(prog: Program, input_bytes: int = 0) -> None:
    """Fill prog.flops and prog.peak_bytes."""
    metas = prog.metas
    flops = 0
    live = input_bytes
    peak = live
    # Bytes for every node output; views alias their input.
    out_bytes: dict = {}
    expiring: dict[int, list] = {}
    for r, last in prog.last_use.items():
        expiring.setdefault(last, []).append(r)
    for node in prog.nodes:
        ins = []
        for a in node.args:
            if isinstance(a, Ref):
                ins.append(metas.get(a))
            elif isinstance(a, list):
                ins.extend(metas.get(x) for x in a if isinstance(x, Ref))
            else:
                ins.append(a)
        outs = [metas[Ref(node.index, k)] for k in range(node.outputs)]
        flops += node_flops(node, ins, outs)
        is_view = getattr(node.op.fn, "is_view", False)
        for k, m in enumerate(outs):
            b = 0 if is_view else _bytes(m)
            out_bytes[Ref(node.index, k)] = b
            live += b
        peak = max(peak, live)
        for r in expiring.get(node.index, []):
            if r.node >= 0:
                live -= out_bytes.get(r, 0)
        # Outputs nobody reads are dropped right away.
        for k in range(node.outputs):
            r = Ref(node.index, k)
            if r not in prog.last_use:
                live -= out_bytes[r]
    prog.flops = flops
    prog.peak_bytes = peak


def input_bytes(prog: Program) -> int:
    return sum(_bytes(prog.metas[Ref(-1, i)]) for i, s in enumerate(prog.inputs)
               if s.kind == gpb.GraphInput.KIND_TENSOR)
