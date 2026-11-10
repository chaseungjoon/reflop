# SPDX-License-Identifier: Apache-2.0
"""Stages V2 (structure) and V3 (schema/attributes) of PLAN.md §9.2.

Turns a decoded protobuf Graph into a Program: a list of nodes whose
arguments are plain Python values and references. Nothing here allocates
device memory or calls a kernel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from . import dtypes
from ._pb import compute_pb2 as cpb
from ._pb import graph_pb2 as gpb
from .errors import ReflopError, invalid, limit
from .ops_generated import FLAG_BANNED, FLAG_CLIENT, OPS, OPSET_VERSION

SUPPORTED_OPSETS = (OPSET_VERSION,)

MEMFMT = {
    gpb.MEMORY_FORMAT_CONTIGUOUS: torch.contiguous_format,
    gpb.MEMORY_FORMAT_PRESERVE: torch.preserve_format,
    gpb.MEMORY_FORMAT_CHANNELS_LAST: torch.channels_last,
    gpb.MEMORY_FORMAT_CHANNELS_LAST_3D: torch.channels_last_3d,
}


@dataclass(frozen=True)
class Ref:
    """Reference to a graph input (node == -1) or a node output."""

    node: int
    index: int


@dataclass(frozen=True)
class ScalarRef:
    """A SCALAR graph input used where the schema expects a number."""

    input: int


@dataclass
class InputSpec:
    kind: int
    dtype: torch.dtype | None
    shape: tuple
    scalar_kind: str | None = None  # "bool" | "int" | "float" | "complex"


@dataclass
class PNode:
    index: int
    op: object             # OpSpec
    args: list             # Python values: Ref, ScalarRef, list[Ref|None], ints, floats, ...
    outputs: int = 0       # filled by meta inference (flattened tensor outputs)


@dataclass
class Program:
    device: gpb.Device
    inputs: list[InputSpec]
    nodes: list[PNode]
    outputs: list[Ref]
    output_meta: list[tuple]   # (dtype, shape)
    graph_hash: bytes = b""
    # Filled by later stages:
    metas: dict = field(default_factory=dict)       # Ref -> meta tensor
    last_use: dict = field(default_factory=dict)    # Ref -> node index of last use
    flops: int = 0
    peak_bytes: int = 0


def _base(t: str) -> str:
    out, depth = [], 0
    for ch in t:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0:
            out.append(ch)
    return "".join(out)


def _fixed_len(core: str) -> int | None:
    """int[2] -> 2, SymInt[] -> None."""
    i = core.find("[")
    inner = core[i + 1:-1]
    return int(inner) if inner else None


class Builder:
    def __init__(self, limits: cpb.Limits, enabled: frozenset, capabilities: frozenset, device: gpb.Device) -> None:
        self.limits = limits
        self.enabled = enabled
        self.caps = capabilities
        self.device = device

    # -------------------------------------------------------------- V2

    def build(self, g: gpb.Graph) -> Program:
        S = cpb.STAGE_V2_STRUCTURE
        L = self.limits
        if g.opset_version not in SUPPORTED_OPSETS:
            raise ReflopError(cpb.ERROR_UNSUPPORTED_VERSION, f"opset version {g.opset_version} is not supported", S)
        if g.device.kind != self.device.kind or g.device.index != self.device.index:
            raise invalid("graph device does not match the session device", S)
        if len(g.inputs) > L.max_graph_inputs:
            raise limit(f"graph has {len(g.inputs)} inputs; the limit is {L.max_graph_inputs}", S)
        if len(g.nodes) > L.max_nodes:
            raise limit(f"graph has {len(g.nodes)} nodes; the limit is {L.max_nodes}", S)
        if len(g.outputs) == 0:
            raise invalid("graph has no outputs", S)
        if len(g.outputs) != len(g.output_meta):
            raise invalid("output_meta must describe every output", S)
        inputs = [self._input(i, gi) for i, gi in enumerate(g.inputs)]
        nodes = []
        for ni, n in enumerate(g.nodes):
            op = OPS.get(n.op_id)
            if op is None or op.fn is None:
                raise invalid(f"unknown operator id {n.op_id}", S, ni, n.op_id)
            if op.flags & (FLAG_CLIENT | FLAG_BANNED):
                raise invalid(f"{op.name} must never be sent to the server", S, ni, n.op_id)
            if n.op_id not in self.enabled:
                why = f" (requires capability {op.capability!r})" if op.capability else ""
                raise invalid(f"{op.name} is not enabled on this server{why}", S, ni, n.op_id)
            dev = "cuda" if self.device.kind == gpb.DEVICE_CUDA else "cpu"
            if dev not in op.devices:
                raise invalid(f"{op.name} is not available on {dev}", S, ni, n.op_id)
            if len(n.args) != len(op.args):
                raise invalid(f"{op.name} takes {len(op.args)} arguments, got {len(n.args)}", S, ni, n.op_id)
            nodes.append(PNode(ni, op, [self._arg(ni, op, spec, a, inputs) for spec, a in zip(op.args, n.args)]))
        outputs = [self._ref(len(g.nodes), r, inputs, allow_scalar=False) for r in g.outputs]
        out_meta = [self._meta(m, cpb.STAGE_V2_STRUCTURE) for m in g.output_meta]
        for r in outputs:
            if r.node == -1 and inputs[r.index].kind != gpb.GraphInput.KIND_TENSOR:
                raise invalid("only tensors can be graph outputs", S)
        return Program(g.device, inputs, nodes, outputs, out_meta)

    def _meta(self, m: gpb.TensorMeta, stage: int) -> tuple:
        if m.dtype not in dtypes.PB_TO_TORCH:
            raise invalid(f"unknown dtype {m.dtype}", stage)
        cap = dtypes.CAPABILITY.get(m.dtype)
        if cap is not None and cap not in self.caps:
            raise invalid(f"dtype requires capability {cap!r}", stage)
        if len(m.shape) > self.limits.max_dims:
            raise limit(f"tensor has {len(m.shape)} dims; the limit is {self.limits.max_dims}", stage)
        dt = dtypes.PB_TO_TORCH[m.dtype]
        try:
            dtypes.nbytes_checked(m.shape, dt, self.limits.max_tensor_numel)
        except (ValueError, OverflowError) as e:
            raise limit(str(e), stage) from None
        return dt, tuple(m.shape)

    def _input(self, i: int, gi: gpb.GraphInput) -> InputSpec:
        S = cpb.STAGE_V2_STRUCTURE
        K = gpb.GraphInput
        if gi.kind in (K.KIND_TENSOR, K.KIND_DATASET_REF):
            dt, shape = self._meta(gi.meta, S)
            return InputSpec(gi.kind, dt, shape)
        if gi.kind == K.KIND_SCALAR:
            sk = {gpb.DTYPE_BOOL: "bool", gpb.DTYPE_INT64: "int", gpb.DTYPE_FLOAT64: "float",
                  gpb.DTYPE_COMPLEX128: "complex"}.get(gi.meta.dtype)
            if sk is None or gi.meta.shape:
                raise invalid(f"input {i}: scalar inputs must be 0-d bool, int64, float64 or complex128", S)
            if sk == "complex" and "complex" not in self.caps:
                raise invalid("complex scalars require capability 'complex'", S)
            return InputSpec(gi.kind, None, (), sk)
        raise invalid(f"input {i}: unsupported input kind {gi.kind}", S)

    def _ref(self, ni: int, r: gpb.ValueRef, inputs, allow_scalar: bool, allow_dataset: bool = False):
        which = r.WhichOneof("ref")
        if which == "graph_input":
            if r.graph_input >= len(inputs):
                raise invalid(f"reference to missing graph input {r.graph_input}", cpb.STAGE_V2_STRUCTURE, ni)
            spec = inputs[r.graph_input]
            K = gpb.GraphInput
            if spec.kind == K.KIND_SCALAR:
                if not allow_scalar:
                    raise invalid("a scalar input cannot be used as a tensor", cpb.STAGE_V3_SCHEMA, ni)
                return ScalarRef(r.graph_input)
            if spec.kind == K.KIND_DATASET_REF and not allow_dataset:
                raise invalid("a dataset input can only be read with reflop.dataset_gather", cpb.STAGE_V3_SCHEMA, ni)
            return Ref(-1, r.graph_input)
        if which == "node_output":
            if r.node_output.node >= ni:
                # Strictly backward references: the graph is a DAG by construction.
                raise invalid("node references a later or the same node", cpb.STAGE_V2_STRUCTURE, ni)
            return Ref(r.node_output.node, r.node_output.output)
        raise invalid("empty value reference", cpb.STAGE_V2_STRUCTURE, ni)

    # -------------------------------------------------------------- V3

    def _arg(self, ni: int, op, spec, a: gpb.Arg, inputs):
        S = cpb.STAGE_V3_SCHEMA
        t = _base(spec.type)
        opt = t.endswith("?")
        core = t[:-1] if opt else t
        which = a.WhichOneof("value")

        def bad(why: str):
            return invalid(f"{op.name} argument {spec.name!r} ({spec.type}): {why}", S, ni, op.id)

        if which is None:
            raise bad("missing value")
        if which == "none":
            if opt or core == "Generator":
                return None
            raise bad("None is not allowed")
        if core == "Tensor" and which in ("i", "f", "b", "complex"):
            # Wrapped Python number (weak-scalar promotion semantics).
            if which == "complex" and "complex" not in self.caps:
                raise bad("complex scalars require capability 'complex'")
            return {"i": lambda: int(a.i), "f": lambda: float(a.f), "b": lambda: bool(a.b),
                    "complex": lambda: complex(a.complex.real, a.complex.imag)}[which]()
        if core == "Tensor" and which == "tensor" and a.tensor.WhichOneof("ref") == "graph_input" \
                and a.tensor.graph_input < len(inputs) and inputs[a.tensor.graph_input].kind == gpb.GraphInput.KIND_SCALAR:
            return self._scalar_input(ni, op, spec, a.tensor, inputs, ("float", "int", "bool", "complex"))
        if core == "Tensor":
            if which != "tensor":
                raise bad(f"expected a tensor, got {which}")
            ds_ok = op.name == "reflop.dataset_gather.default" and spec.name == "dataset"
            r = self._ref(ni, a.tensor, inputs, allow_scalar=False, allow_dataset=ds_ok)
            if ds_ok and not (r.node == -1 and inputs[r.index].kind == gpb.GraphInput.KIND_DATASET_REF):
                raise bad("expected a dataset input")
            return r
        if core in ("Tensor[]", "Tensor?[]"):
            if which != "tensors":
                raise bad(f"expected a tensor list, got {which}")
            if len(a.tensors.items) > self.limits.max_list_length:
                raise limit(f"{op.name}: list longer than {self.limits.max_list_length}", S, ni, op.id)
            out = []
            for it in a.tensors.items:
                if it.WhichOneof("ref") is None:
                    if core != "Tensor?[]":
                        raise bad("None inside a non-optional tensor list")
                    out.append(None)
                else:
                    out.append(self._ref(ni, it, inputs, allow_scalar=False))
            return out
        if core in ("int", "SymInt"):
            if which != "i":
                raise bad(f"expected an int, got {which}")
            return int(a.i)
        if core.startswith(("int[", "SymInt[")):
            if which != "ints":
                raise bad(f"expected an int list, got {which}")
            vals = list(a.ints.values)
            # int[N] is PyTorch's broadcast hint ("one int means N copies"), not
            # a fixed length; meta inference checks each op's real rule.
            if len(vals) > self.limits.max_list_length:
                raise bad(f"list of {len(vals)} exceeds the limit")
            return vals
        if core == "float":
            if which == "f":
                return float(a.f)
            if which == "tensor":
                return self._scalar_input(ni, op, spec, a.tensor, inputs, ("float", "int"))
            raise bad(f"expected a float, got {which}")
        if core.startswith("float["):
            if which != "floats":
                raise bad(f"expected a float list, got {which}")
            if len(a.floats.values) > self.limits.max_list_length:
                raise bad("list too long")
            return list(a.floats.values)
        if core == "bool":
            if which != "b":
                raise bad(f"expected a bool, got {which}")
            return bool(a.b)
        if core.startswith("bool["):
            if which != "bools":
                raise bad(f"expected a bool list, got {which}")
            vals = list(a.bools.values)
            if len(vals) > self.limits.max_list_length:
                raise bad(f"list of {len(vals)} exceeds the limit")
            return vals
        if core == "Scalar":
            if which == "i":
                return int(a.i)
            if which == "f":
                return float(a.f)
            if which == "b":
                return bool(a.b)
            if which == "complex":
                if "complex" not in self.caps:
                    raise bad("complex scalars require capability 'complex'")
                return complex(a.complex.real, a.complex.imag)
            if which == "tensor":
                return self._scalar_input(ni, op, spec, a.tensor, inputs, ("float", "int", "bool", "complex"))
            raise bad(f"expected a scalar, got {which}")
        if core == "Scalar[]":
            if which == "tensors":
                if len(a.tensors.items) > self.limits.max_list_length:
                    raise bad("list too long")
                return [self._scalar_input(ni, op, spec, it, inputs, ("float", "int", "bool", "complex"))
                        for it in a.tensors.items]
            if which == "floats":
                return list(a.floats.values)
            if which == "ints":
                return list(a.ints.values)
            raise bad(f"expected a scalar list, got {which}")
        if core == "ScalarType":
            if which != "dtype" or a.dtype not in dtypes.PB_TO_TORCH:
                raise bad("expected a known dtype")
            cap = dtypes.CAPABILITY.get(a.dtype)
            if cap is not None and cap not in self.caps:
                raise bad(f"dtype requires capability {cap!r}")
            return dtypes.PB_TO_TORCH[a.dtype]
        if core == "MemoryFormat":
            if which != "memory_format" or a.memory_format not in MEMFMT:
                raise bad("expected a known memory format")
            return MEMFMT[a.memory_format]
        if core == "Layout":
            if which != "layout" or a.layout != gpb.LAYOUT_STRIDED:
                raise bad("only the strided layout is supported")
            return torch.strided
        if core == "Device":
            if which != "device" or a.device.kind != self.device.kind or a.device.index != self.device.index:
                raise bad("device must be the session device")
            return "session-device"
        if core == "Generator":
            raise bad("generators are server-managed; send None")
        if core == "str":
            codes = op.enums.get(spec.name)
            if which != "enum_code" or codes is None or a.enum_code >= len(codes):
                raise bad("unknown enum code")
            v = codes[a.enum_code]
            # For optional string args ("str?"), the code for "none" means None;
            # for plain "str" args (gelu's approximate="none") it is the string.
            return None if (v == "none" and opt) else v
        raise bad("unsupported schema type")

    def _scalar_input(self, ni, op, spec, ref: gpb.ValueRef, inputs, kinds):
        if ref.WhichOneof("ref") != "graph_input" or ref.graph_input >= len(inputs):
            raise invalid(f"{op.name} argument {spec.name!r}: scalars must come from scalar graph inputs",
                          cpb.STAGE_V3_SCHEMA, ni, op.id)
        s = inputs[ref.graph_input]
        if s.kind != gpb.GraphInput.KIND_SCALAR or s.scalar_kind not in kinds:
            raise invalid(f"{op.name} argument {spec.name!r}: expected a {'/'.join(kinds)} scalar input",
                          cpb.STAGE_V3_SCHEMA, ni, op.id)
        return ScalarRef(ref.graph_input)


def placeholder(kind: str):
    """Value used for SCALAR inputs during meta inference."""
    return {"bool": True, "int": 1, "float": 1.0, "complex": complex(1.0, 0.0)}[kind]


def finite_or_raise(x: float, what: str) -> float:
    if not math.isfinite(x):
        raise ValueError(f"{what} must be finite")
    return x
