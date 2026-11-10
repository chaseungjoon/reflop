# SPDX-License-Identifier: Apache-2.0
"""Stage V4: meta inference (PLAN.md §9.2).

Runs every node on PyTorch `meta` tensors to obtain all intermediate shapes
and dtypes without touching device memory. An op that has no meta kernel
cannot be validated and is rejected (fail closed, I11).
"""

from __future__ import annotations

import torch
from torch.utils._pytree import tree_flatten

from . import natives, validators
from ._pb import compute_pb2 as cpb
from ._pb import graph_pb2 as gpb
from .errors import ReflopError, invalid, limit
from .graph import Program, Ref, ScalarRef, placeholder


class DatasetMeta:
    """Stand-in for a dataset input during meta inference."""

    def __init__(self, dtype: torch.dtype, shape: tuple) -> None:
        self.dtype = dtype
        self.shape = torch.Size(shape)


def call_args(node, resolve):
    """Split resolved arguments into positional and keyword arguments by the
    static schema (kwarg-only names come from the generated table)."""
    pos, kw = [], {}
    for spec, a in zip(node.op.args, node.args):
        v = resolve(a)
        if spec.kwarg_only:
            kw[spec.name] = v
        else:
            pos.append(v)
    return pos, kw


def infer(prog: Program, limits: cpb.Limits) -> None:
    S = cpb.STAGE_V4_META
    metas: dict = {}
    for i, inp in enumerate(prog.inputs):
        if inp.kind == gpb.GraphInput.KIND_TENSOR:
            metas[Ref(-1, i)] = torch.empty(inp.shape, dtype=inp.dtype, device="meta")
        elif inp.kind == gpb.GraphInput.KIND_DATASET_REF:
            metas[Ref(-1, i)] = DatasetMeta(inp.dtype, inp.shape)
    last_use: dict = {}

    for node in prog.nodes:
        def resolve(a, node=node):
            if isinstance(a, Ref):
                if a.node >= 0 and a.index >= prog.nodes[a.node].outputs:
                    raise invalid(f"reference to missing output {a.index} of node {a.node}", S, node.index, node.op.id)
                last_use[a] = node.index
                return metas[a]
            if isinstance(a, ScalarRef):
                return placeholder(prog.inputs[a.input].scalar_kind)
            if isinstance(a, list):
                return [resolve(x) for x in a]
            if a == "session-device":
                return torch.device("meta")
            return a

        pos, kw = call_args(node, resolve)
        validators.check_static(node, pos, kw)
        try:
            with torch.inference_mode():
                if natives.is_native(node.op):
                    out = natives.meta(node.op.name, pos, kw)
                else:
                    out = node.op.fn(*pos, **kw)
        except ReflopError:
            raise
        except NotImplementedError as e:
            raise invalid(f"{node.op.name} cannot be validated (no meta kernel): {str(e)[:200]}", S,
                          node.index, node.op.id) from None
        except Exception as e:  # noqa: BLE001 - every kernel error becomes a structured rejection
            raise ReflopError(cpb.ERROR_SHAPE_MISMATCH, f"{node.op.name}: {str(e)[:500]}", S, node.index,
                              node.op.id) from None
        flat = [x for x in tree_flatten(out)[0] if isinstance(x, torch.Tensor)]
        for k, m in enumerate(flat):
            if m.dim() > limits.max_dims:
                raise limit(f"{node.op.name} output has {m.dim()} dims", S, node.index, node.op.id)
            if m.numel() > limits.max_tensor_numel:
                raise limit(f"{node.op.name} output has {m.numel()} elements; the limit is "
                            f"{limits.max_tensor_numel}", S, node.index, node.op.id)
            metas[Ref(node.index, k)] = m
        node.outputs = len(flat)

    for r, (dt, shape) in zip(prog.outputs, prog.output_meta):
        if r.node >= 0 and r.index >= prog.nodes[r.node].outputs:
            raise invalid(f"graph output refers to missing output {r.index} of node {r.node}", S)
        m = metas.get(r)
        if m is None or isinstance(m, DatasetMeta):
            raise invalid("graph output is not a tensor", S)
        if m.dtype != dt or tuple(m.shape) != tuple(shape):
            raise ReflopError(cpb.ERROR_SHAPE_MISMATCH,
                              f"declared output {dt}{list(shape)} but inferred {m.dtype}{list(m.shape)}", S)
        last_use[r] = len(prog.nodes)
    prog.metas = metas
    prog.last_use = last_use
