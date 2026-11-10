# SPDX-License-Identifier: Apache-2.0
"""Lazy trace: SSA values and nodes recorded by RemoteTensor, and their
encoding into a FlopIR Graph (PLAN.md §7, §11.2)."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any

import torch

from . import _opset_generated as opset
from ._pb import graph_pb2 as gpb

# ---------------------------------------------------------------- dtypes

DTYPE_TO_PB = {
    torch.bool: gpb.DTYPE_BOOL, torch.uint8: gpb.DTYPE_UINT8, torch.int8: gpb.DTYPE_INT8,
    torch.int16: gpb.DTYPE_INT16, torch.int32: gpb.DTYPE_INT32, torch.int64: gpb.DTYPE_INT64,
    torch.float16: gpb.DTYPE_FLOAT16, torch.bfloat16: gpb.DTYPE_BFLOAT16, torch.float32: gpb.DTYPE_FLOAT32,
    torch.float64: gpb.DTYPE_FLOAT64, torch.complex64: gpb.DTYPE_COMPLEX64,
    torch.complex128: gpb.DTYPE_COMPLEX128, torch.float8_e4m3fn: gpb.DTYPE_FLOAT8_E4M3FN,
    torch.float8_e5m2: gpb.DTYPE_FLOAT8_E5M2,
}
PB_TO_DTYPE = {v: k for k, v in DTYPE_TO_PB.items()}
MEMFMT_TO_PB = {
    torch.contiguous_format: gpb.MEMORY_FORMAT_CONTIGUOUS, torch.preserve_format: gpb.MEMORY_FORMAT_PRESERVE,
    torch.channels_last: gpb.MEMORY_FORMAT_CHANNELS_LAST, torch.channels_last_3d: gpb.MEMORY_FORMAT_CHANNELS_LAST_3D,
}


class UnsupportedOperatorError(RuntimeError):
    """An operator could not be expressed in the server's opset."""


def tensor_meta(t: torch.Tensor) -> gpb.TensorMeta:
    if t.dtype not in DTYPE_TO_PB:
        raise UnsupportedOperatorError(f"dtype {t.dtype} is not supported by reflop")
    return gpb.TensorMeta(dtype=DTYPE_TO_PB[t.dtype], shape=list(t.shape))


# ---------------------------------------------------------------- values

NODE, HANDLE, CPU, SCALAR, DATASET = "node", "handle", "cpu", "scalar", "dataset"


class Value:
    """One SSA value. `meta` is a meta tensor (or a Python scalar type for
    lifted scalars)."""

    __slots__ = ("kind", "node", "index", "handle", "meta", "payload", "__weakref__")

    def __init__(self, kind: str, meta: Any, node: int = -1, index: int = 0, handle: bytes | None = None,
                 payload: Any = None) -> None:
        self.kind = kind
        self.meta = meta
        self.node = node
        self.index = index
        self.handle = handle
        self.payload = payload  # CPU tensor, Python scalar, or (dataset_id, tensor_name)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Value({self.kind}, node={self.node}.{self.index}, handle={self.handle and self.handle.hex()[:8]})"


@dataclass
class Node:
    name: str               # opset name, e.g. "aten.add.Tensor"
    func: Any               # OpOverload (for local reference execution)
    args: list              # encoded args, see Trace.encode_arg
    outs: list[Value]
    rng: bool = False


# Python floats in these schema types are lifted to SCALAR graph inputs so
# that values that change every step (learning rate, bias corrections, loss
# scale) do not change the graph hash (§11.2 step 5).
_LIFT_TYPES = {"float", "float?", "Scalar", "Scalar?"}


@dataclass
class Trace:
    nodes: list[Node] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.nodes)

    def record(self, name: str, func: Any, args: list, out_metas: list, rng: bool = False) -> list[Value]:
        idx = len(self.nodes)
        outs = [Value(NODE, m, node=idx, index=i) for i, m in enumerate(out_metas)]
        self.nodes.append(Node(name, func, args, outs, rng))
        return outs


# ---------------------------------------------------------------- graph building


def _is_alloc(entry) -> bool:
    return bool(entry[1] & opset.FLAG_ALLOC)


@dataclass
class Built:
    graph: gpb.Graph
    graph_hash: bytes
    input_values: list[Value]       # in graph input order
    output_values: list[Value]      # in graph output order
    scalar_values: list[Any]        # per input (None for tensors)


def canonical_hash(graph: gpb.Graph) -> bytes:
    data = graph.SerializeToString(deterministic=True)
    return hashlib.sha256(b"reflop-graph-v1\x00" + graph.opset_version.to_bytes(4, "big") + data).digest()


class GraphBuilder:
    """Turns a slice of the trace into a Graph with dense input/node numbering.

    Inputs are created on first use, so equal programs produce identical
    graphs. Dead nodes (not reaching any output) are removed first.
    """

    def __init__(self, trace: Trace, device: gpb.Device, outputs: list[Value]) -> None:
        self.trace = trace
        self.device = device
        self.outputs = outputs
        self.inputs: list[Value] = []
        self.input_index: dict[tuple, int] = {}
        self.node_index: dict[int, int] = {}
        self.g = gpb.Graph(opset_version=opset.OPSET_VERSION, device=device)

    def _live_nodes(self) -> list[int]:
        need = set()
        stack = [v.node for v in self.outputs if v.kind == NODE]
        while stack:
            n = stack.pop()
            if n in need:
                continue
            need.add(n)
            for a in self.trace.nodes[n].args:
                for v in _arg_values(a):
                    if v.kind == NODE:
                        stack.append(v.node)
        return sorted(need)

    def _input_ref(self, v: Value) -> gpb.ValueRef:
        k = ("obj", id(v)) if v.kind in (CPU, SCALAR) else (v.kind, v.handle, v.payload if v.kind == DATASET else None)
        i = self.input_index.get(k)
        if i is None:
            i = len(self.inputs)
            self.input_index[k] = i
            self.inputs.append(v)
            gi = gpb.GraphInput()
            if v.kind == SCALAR:
                gi.kind = gpb.GraphInput.KIND_SCALAR
                gi.meta.dtype = _scalar_dtype(v.payload)
            elif v.kind == DATASET:
                gi.kind = gpb.GraphInput.KIND_DATASET_REF
                gi.meta.CopyFrom(tensor_meta(v.meta))
            else:
                gi.kind = gpb.GraphInput.KIND_TENSOR
                gi.meta.CopyFrom(tensor_meta(v.meta))
            self.g.inputs.append(gi)
        return gpb.ValueRef(graph_input=i)

    def ref(self, v: Value) -> gpb.ValueRef:
        if v.kind == NODE:
            return gpb.ValueRef(node_output=gpb.NodeOutput(node=self.node_index[v.node], output=v.index))
        return self._input_ref(v)

    def build(self) -> Built:
        for n in self._live_nodes():
            node = self.trace.nodes[n]
            entry = opset.OPS[node.name]
            pn = gpb.Node(op_id=entry[0])
            for spec, a in zip(entry[3], node.args, strict=True):
                pn.args.append(self._encode(spec, a))
            self.node_index[n] = len(self.g.nodes)
            self.g.nodes.append(pn)
        for v in self.outputs:
            self.g.outputs.append(self.ref(v))
            self.g.output_meta.append(tensor_meta(v.meta))
        scalars = [v.payload if v.kind == SCALAR else None for v in self.inputs]
        return Built(self.g, canonical_hash(self.g), list(self.inputs), list(self.outputs), scalars)

    def _encode(self, spec, a) -> gpb.Arg:
        kind, val = a
        if kind == "none":
            return gpb.Arg(none=True)
        if kind == "t":
            return gpb.Arg(tensor=self.ref(val))
        if kind == "tl":
            return gpb.Arg(tensors=gpb.TensorList(items=[gpb.ValueRef() if x is None else self.ref(x) for x in val]))
        if kind == "i":
            return gpb.Arg(i=val)
        if kind == "f":
            return gpb.Arg(f=val)
        if kind == "b":
            return gpb.Arg(b=val)
        if kind == "ints":
            return gpb.Arg(ints=gpb.Int64List(values=val))
        if kind == "floats":
            return gpb.Arg(floats=gpb.DoubleList(values=val))
        if kind == "bools":
            return gpb.Arg(bools=gpb.BoolList(values=val))
        if kind == "dtype":
            return gpb.Arg(dtype=DTYPE_TO_PB[val])
        if kind == "mf":
            return gpb.Arg(memory_format=MEMFMT_TO_PB[val])
        if kind == "layout":
            return gpb.Arg(layout=gpb.LAYOUT_STRIDED)
        if kind == "device":
            return gpb.Arg(device=self.device)
        if kind == "enum":
            return gpb.Arg(enum_code=val)
        if kind == "complex":
            return gpb.Arg(complex=gpb.ComplexValue(real=val.real, imag=val.imag))
        raise AssertionError(f"unknown arg kind {kind}")


def _scalar_dtype(x) -> int:
    if isinstance(x, bool):
        return gpb.DTYPE_BOOL
    if isinstance(x, int):
        return gpb.DTYPE_INT64
    if isinstance(x, complex):
        return gpb.DTYPE_COMPLEX128
    return gpb.DTYPE_FLOAT64


def _arg_values(a):
    kind, val = a
    if kind == "t":
        yield val
    elif kind == "tl":
        for x in val:
            if x is not None:
                yield x


# ---------------------------------------------------------------- arg normalization


def _base(t: str) -> str:
    """Strip alias annotations: 'Tensor(a!)' -> 'Tensor', 'Tensor(a)[]' -> 'Tensor[]'."""
    out, depth = [], 0
    for ch in t:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0:
            out.append(ch)
    return "".join(out)


def normalize(func, args: tuple, kwargs: dict) -> list:
    """Bind args/kwargs to the schema, filling defaults, in schema order."""
    schema = func._schema
    vals = []
    for i, a in enumerate(schema.arguments):
        if a.kwarg_only:
            if a.name in kwargs:
                vals.append(kwargs[a.name])
            elif a.has_default_value():
                vals.append(a.default_value)
            else:
                raise TypeError(f"{func}: missing keyword argument {a.name}")
        elif i < len(args):
            vals.append(args[i])
        elif a.name in kwargs:
            vals.append(kwargs[a.name])
        elif a.has_default_value():
            vals.append(a.default_value)
        else:
            raise TypeError(f"{func}: missing argument {a.name}")
    return vals


class ArgEncoder:
    """Converts normalized Python values to trace args for one op entry."""

    def __init__(self, value_of, cpu_value, scalar_value) -> None:
        self.value_of = value_of          # RemoteTensor -> Value
        self.cpu_value = cpu_value        # CPU tensor -> Value
        self.scalar_value = scalar_value  # python scalar -> Value (lifted)

    def tensor(self, x) -> Value:
        from .tensor import RemoteTensor

        if isinstance(x, Value):
            return x
        if isinstance(x, RemoteTensor):
            return self.value_of(x)
        if isinstance(x, torch.Tensor):
            return self.cpu_value(x)
        raise TypeError(f"expected a tensor, got {type(x).__name__}")

    def _number(self, v, alloc: bool):
        if isinstance(v, bool):
            return ("b", v)
        if isinstance(v, int):
            return ("i", v)
        if isinstance(v, complex):
            return ("complex", v)
        if not alloc and math.isfinite(v):
            return ("t", self.scalar_value(float(v)))
        return ("f", float(v))

    def encode(self, name: str, entry, vals: list) -> list:
        out = []
        enums = entry[5]
        alloc = _is_alloc(entry)
        for (aname, atype, _kw, _hd, _d), v in zip(entry[3], vals, strict=True):
            t = _base(atype)
            opt = t.endswith("?")
            core = t.rstrip("?")
            if v is None and core == "str" and "none" in enums.get(aname, []):
                out.append(("enum", enums[aname].index("none")))
                continue
            if v is None:
                if not opt and core not in ("Generator",):
                    raise UnsupportedOperatorError(f"{name}: argument {aname} is None")
                out.append(("none", None))
                continue
            if core == "Tensor" and isinstance(v, (bool, int, float, complex)):
                # A wrapped Python number: keep it a number so the server applies
                # PyTorch's weak-scalar type promotion, exactly as here.
                out.append(self._number(v, alloc))
            elif core == "Tensor":
                out.append(("t", self.tensor(v)))
            elif core in ("Tensor[]", "Tensor?[]"):
                out.append(("tl", [None if x is None else self.tensor(x) for x in v]))
            elif core in ("int", "SymInt"):
                if isinstance(v, torch.Tensor):
                    v = int(v.item())
                out.append(("i", int(v)))
            elif core.startswith(("int[", "SymInt[")):
                if isinstance(v, int):
                    v = [v]
                out.append(("ints", [int(x) for x in v]))
            elif core == "float":
                if not alloc and isinstance(v, float) and math.isfinite(v):
                    out.append(("t", self.scalar_value(float(v))))
                else:
                    out.append(("f", float(v)))
            elif core.startswith("float["):
                out.append(("floats", [float(x) for x in v]))
            elif core == "bool":
                out.append(("b", bool(v)))
            elif core.startswith("bool["):
                out.append(("bools", [bool(x) for x in v]))
            elif core == "Scalar":
                if isinstance(v, torch.Tensor):
                    v = v.item()
                out.append(self._number(v, alloc))
            elif core == "Scalar[]":
                items = []
                for x in v:
                    if isinstance(x, torch.Tensor):
                        x = x.item()
                    items.append(self.scalar_value(x if isinstance(x, (bool, int, complex)) else float(x)))
                out.append(("tl", items))
            elif core == "ScalarType":
                out.append(("dtype", v))
            elif core == "MemoryFormat":
                out.append(("mf", v))
            elif core == "Layout":
                if v != torch.strided:
                    raise UnsupportedOperatorError(f"{name}: only strided layout is supported")
                out.append(("layout", v))
            elif core == "Device":
                out.append(("device", v))
            elif core == "Generator":
                raise UnsupportedOperatorError(
                    f"{name}: explicit torch.Generator objects are not supported on reflop tensors; "
                    "use reflop.seed(n)")
            elif core == "str":
                codes = enums.get(aname)
                if codes is None or v not in codes:
                    raise UnsupportedOperatorError(f"{name}: unsupported value {v!r} for {aname}")
                out.append(("enum", codes.index(v)))
            else:
                raise UnsupportedOperatorError(f"{name}: unsupported argument type {atype}")
        return out
