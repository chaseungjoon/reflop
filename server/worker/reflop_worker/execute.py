# SPDX-License-Identifier: Apache-2.0
"""Stages E1–E3 (PLAN.md §9.2–9.3): bind, check, execute.

Dispatch is through the static table only (I2): node.op.fn is a PyTorch
OpOverload referenced by literal name in generated code, or a native
implementation from natives.IMPL.
"""

from __future__ import annotations

import hashlib
import time

import torch
from torch.utils._pytree import tree_flatten

from . import dtypes, natives, validators
from ._pb import compute_pb2 as cpb
from ._pb import graph_pb2 as gpb
from .errors import FatalError, ReflopError
from .graph import Program, Ref, ScalarRef
from .handles import check_new
from .ops_generated import FLAG_UNINIT

E1 = cpb.STAGE_E1_BINDINGS
E3 = cpb.STAGE_E3_EXECUTE


def binding_error(msg: str) -> ReflopError:
    return ReflopError(cpb.ERROR_BINDING, msg, E1)


def tensor_from_bytes(data: bytes, dt: torch.dtype, shape, device) -> torch.Tensor:
    n = dtypes.numel_checked(shape, 1 << 62)
    need = n * dtypes.itemsize(dt)
    if len(data) != need:
        raise binding_error(f"tensor data has {len(data)} bytes, expected {need}")
    if need == 0:
        return torch.empty(shape, dtype=dt, device=device)
    buf = bytearray(data)
    t = torch.frombuffer(buf, dtype=torch.uint8).view(dt).reshape(shape)
    r = validators.measure(t) if dt in validators.INT_DTYPES else None  # on the host: no device sync later
    out = t.to(device)
    validators.set_range(out, r)
    return out


# Ops whose integer output takes values from their first input only, so a
# verified range carries over (no device synchronization needed).
RANGE_PRESERVING = {
    "aten.view.default", "aten._unsafe_view.default", "aten.reshape.default", "aten.clone.default",
    "aten.contiguous.default", "aten.alias.default", "aten.detach.default", "aten.expand.default",
    "aten.permute.default", "aten.t.default", "aten.transpose.int", "aten.squeeze.dim", "aten.squeeze.dims",
    "aten.unsqueeze.default", "aten.select.int", "aten.slice.Tensor", "aten.flatten.using_ints",
    "aten.index_select.default", "aten.gather.default", "aten.repeat.default", "aten.cat.default",
    "aten.split_with_sizes.default", "aten.split.Tensor", "aten.unbind.int", "aten.flip.default",
    "aten.roll.default", "aten.lift_fresh_copy.default", "aten.diagonal.default", "aten.narrow.default",
    "reflop.dataset_gather.default",
}


def propagate_range(node, pos: list, outs: list) -> None:
    name = node.op.name
    src = None
    if name in RANGE_PRESERVING and pos:
        a = pos[0]
        if isinstance(a, list):  # cat
            rs = [validators.known_range(x) for x in a if isinstance(x, torch.Tensor)]
            if rs and all(r is not None for r in rs):
                src = (min(r[0] for r in rs), max(r[1] for r in rs))
        elif isinstance(a, torch.Tensor) or hasattr(a, "value_range"):
            src = a.value_range if hasattr(a, "value_range") else validators.known_range(a)
            if isinstance(a, torch.Tensor) and outs and outs[0].dtype != a.dtype:
                src = None
    elif name == "aten._to_copy.default" and pos and isinstance(pos[0], torch.Tensor) and outs:
        r = validators.known_range(pos[0])
        o = outs[0]
        if r is not None and o.dtype in validators.INT_DTYPES and o.dtype != torch.bool:
            info = torch.iinfo(o.dtype)
            if info.min <= r[0] and r[1] <= info.max:
                src = r
    elif name in ("aten.randperm.default", "reflop.randperm_seeded.default") and pos:
        src = (0, max(int(pos[0]) - 1, 0))
    elif name in ("aten.argmax.default", "aten.argmin.default") and pos:
        src = (0, max(pos[0].numel() - 1, 0))
    if src is not None:
        for o in outs:
            validators.set_range(o, src)


def oom_message(e: BaseException) -> str:
    """Keep only the failed allocation: PyTorch's full OOM text names other
    processes and their memory use, which would leak across tenants."""
    import re

    m = re.search(r"Tried to allocate [0-9.]+ [KMGT]?i?B", str(e))
    return (m.group(0) if m else "allocation failed") + "; the device is shared with other sessions"


def is_cuda_fatal(e: BaseException) -> bool:
    s = str(e)
    return any(k in s for k in ("device-side assert", "CUDA error: an illegal", "CUDA error: unspecified launch",
                                "CUBLAS_STATUS_EXECUTION_FAILED", "cudaErrorLaunchFailure",
                                "an illegal memory access", "CUDA error: misaligned"))


class Executor:
    def __init__(self, device: torch.device, handles, datasets: dict, limits: cpb.Limits,
                 flops_quota: int = 0, device_memory_quota: int = 0) -> None:
        self.device = device
        self.handles = handles
        self.datasets = datasets
        self.limits = limits
        self.flops_quota = flops_quota
        self.device_memory_quota = device_memory_quota

    # -------------------------------------------------------------- E1

    def bind(self, prog: Program, ex: cpb.Execute) -> dict:
        if len(ex.bindings) != len(prog.inputs):
            raise binding_error(f"graph has {len(prog.inputs)} inputs, got {len(ex.bindings)} bindings")
        if len(ex.outputs) != len(prog.outputs):
            raise binding_error(f"graph has {len(prog.outputs)} outputs, got {len(ex.outputs)} output specs")
        env: dict = {}
        scalars: dict[int, object] = {}
        K = gpb.GraphInput
        for i, (spec, b) in enumerate(zip(prog.inputs, ex.bindings)):
            which = b.WhichOneof("value")
            if spec.kind == K.KIND_TENSOR:
                if which == "handle":
                    t = self.handles.get(b.handle)
                elif which == "inline":
                    if len(b.inline.data) > self.limits.max_inline_tensor_bytes:
                        raise ReflopError(cpb.ERROR_RESOURCE_LIMIT, "inline tensor exceeds the limit", E1)
                    if b.inline.meta.dtype != dtypes.TORCH_TO_PB[spec.dtype] or tuple(b.inline.meta.shape) != spec.shape:
                        raise binding_error(f"input {i}: inline tensor meta does not match the graph")
                    t = tensor_from_bytes(b.inline.data, spec.dtype, spec.shape, self.device)
                else:
                    raise binding_error(f"input {i}: expected a tensor binding")
                if t.dtype != spec.dtype or tuple(t.shape) != spec.shape:
                    raise binding_error(f"input {i}: bound tensor is {t.dtype}{list(t.shape)}, the graph expects "
                                        f"{spec.dtype}{list(spec.shape)}")
                env[Ref(-1, i)] = t
            elif spec.kind == K.KIND_SCALAR:
                if which != "scalar":
                    raise binding_error(f"input {i}: expected a scalar binding")
                sw = b.scalar.WhichOneof("value")
                want = {"bool": "b", "int": "i", "float": "f", "complex": "c"}[spec.scalar_kind]
                if sw != want:
                    raise binding_error(f"input {i}: expected a {spec.scalar_kind} scalar")
                if sw == "b":
                    scalars[i] = bool(b.scalar.b)
                elif sw == "i":
                    scalars[i] = int(b.scalar.i)
                elif sw == "f":
                    scalars[i] = float(b.scalar.f)
                else:
                    scalars[i] = complex(b.scalar.c.real, b.scalar.c.imag)
            elif spec.kind == K.KIND_DATASET_REF:
                if which != "dataset":
                    raise binding_error(f"input {i}: expected a dataset binding")
                ds = self.datasets.get(b.dataset.dataset_id)
                if ds is None:
                    raise ReflopError(cpb.ERROR_UNKNOWN_DATASET, "dataset is not open in this session", E1)
                t = ds.tensors.get(b.dataset.tensor)
                if t is None:
                    raise binding_error(f"dataset has no tensor {b.dataset.tensor!r}")
                if t.dtype != spec.dtype or tuple(t.shape) != spec.shape:
                    raise binding_error(f"input {i}: dataset tensor does not match the graph")
                env[Ref(-1, i)] = t
        seen = set()
        for o, spec in enumerate(ex.outputs):
            k = cpb.OutputSpec
            if spec.kind == k.KIND_STORE:
                check_new(spec.handle)
                if spec.handle in self.handles.t or spec.handle in seen:
                    raise ReflopError(cpb.ERROR_HANDLE_EXISTS, f"output {o}: handle already exists", E1)
                seen.add(spec.handle)
            elif spec.kind == k.KIND_REBIND:
                old = self.handles.get(spec.handle)
                dt, shape = prog.output_meta[o]
                if old.dtype != dt or tuple(old.shape) != tuple(shape):
                    raise binding_error(f"output {o}: rebind changes the tensor's dtype or shape")
                if spec.handle in seen:
                    raise binding_error(f"output {o}: handle used twice")
                seen.add(spec.handle)
            elif spec.kind == k.KIND_RETURN:
                dt, shape = prog.output_meta[o]
                if dtypes.nbytes_checked(shape, dt, self.limits.max_tensor_numel) > self.limits.max_fetch_bytes:
                    raise ReflopError(cpb.ERROR_RESOURCE_LIMIT, f"output {o} is too large to return", E1)
            elif spec.kind != k.KIND_DISCARD:
                raise binding_error(f"output {o}: unknown output kind")
        env["__scalars__"] = scalars
        return env

    # -------------------------------------------------------------- E2 + E3

    def compile(self, prog: Program) -> list:
        """Flatten a validated program into per-node plans (once per graph)."""
        if getattr(prog, "_compiled", None) is not None:
            return prog._compiled
        expiring: dict[int, list] = {}
        for r, last in prog.last_use.items():
            if r.node >= 0 and r not in set(prog.outputs):
                expiring.setdefault(last, []).append(r)
        plans = []
        for node in prog.nodes:
            args = []
            for spec, a in zip(node.op.args, node.args):
                if isinstance(a, Ref):
                    kind, val = 0, a
                elif isinstance(a, ScalarRef):
                    kind, val = 1, a.input
                elif isinstance(a, list) and any(isinstance(x, (Ref, ScalarRef)) for x in a):
                    kind, val = 2, a
                elif isinstance(a, str) and a == "session-device":
                    kind, val = 3, None
                else:
                    kind, val = 4, a
                args.append((spec.kwarg_only, spec.name, kind, val))
            outs = []
            for k in range(node.outputs):
                r = Ref(node.index, k)
                m = prog.metas[r]
                outs.append((r, m.dtype, tuple(m.shape), r in prog.last_use))
            plans.append((node, node.op.fn, natives.is_native(node.op), args, outs,
                          validators.has_runtime(node.op), bool(node.op.flags & FLAG_UNINIT),
                          expiring.get(node.index, ())))
        prog._compiled = plans
        return plans

    def run(self, prog: Program, ex: cpb.Execute) -> cpb.ExecuteResult:
        if self.flops_quota and prog.flops > self.flops_quota:
            raise ReflopError(cpb.ERROR_QUOTA, f"graph needs ~{prog.flops} FLOPs; the per-execute quota is "
                              f"{self.flops_quota}", cpb.STAGE_V5_SAFETY)
        plans = self.compile(prog)
        env = self.bind(prog, ex)
        scalars = env.pop("__scalars__")
        t0 = time.perf_counter()
        dev = self.device
        node = None
        try:
            with torch.inference_mode():
                for node, fn, native, args, outs, check, uninit, free in plans:
                    pos, kw = [], {}
                    for is_kw, name, kind, val in args:
                        if kind == 0:
                            v = env[val]
                        elif kind == 4:
                            v = val
                        elif kind == 1:
                            v = scalars[val]
                        elif kind == 3:
                            v = dev
                        else:
                            v = [None if x is None else (env[x] if isinstance(x, Ref) else scalars[x.input])
                                 for x in val]
                        if is_kw:
                            kw[name] = v
                        else:
                            pos.append(v)
                    if check:
                        validators.check_runtime(node, pos, kw)
                    out = natives.call(node.op.name, pos, kw, dev) if native else fn(*pos, **kw)
                    if isinstance(out, torch.Tensor):
                        flat = [out]
                    else:
                        flat = [x for x in tree_flatten(out)[0] if isinstance(x, torch.Tensor)]
                    if len(flat) != len(outs):
                        raise ReflopError(cpb.ERROR_INTERNAL, f"{node.op.name}: produced {len(flat)} outputs, "
                                          f"validated {len(outs)}", E3, node.index, node.op.id)
                    propagate_range(node, pos, flat)
                    for t, (r, dt, shape, used) in zip(flat, outs):
                        # A few kernels give unused auxiliary outputs a different
                        # shape than their meta kernel (e.g. _embedding_bag's
                        # offset2bag). Anything that is consumed must match.
                        if used and (t.dtype != dt or tuple(t.shape) != shape):
                            raise ReflopError(cpb.ERROR_INTERNAL, f"{node.op.name}: output is {t.dtype}"
                                              f"{list(t.shape)}, validated {dt}{list(shape)}", E3,
                                              node.index, node.op.id)
                        if uninit:
                            t.zero_()  # never return stale memory (I13)
                        env[r] = t
                    for r in free:
                        env.pop(r, None)
                # No device sync here: the next execute's Python dispatch overlaps
                # with this one's kernels. RETURN outputs sync when copied out;
                # asynchronous kernel faults surface on the next sync and are
                # treated as fatal (SESSION_LOST).
        except ReflopError:
            raise
        except torch.OutOfMemoryError as e:
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            raise ReflopError(cpb.ERROR_OUT_OF_MEMORY, f"out of device memory: {oom_message(e)}", E3) from None
        except Exception as e:  # noqa: BLE001 - every kernel error becomes a structured error
            if is_cuda_fatal(e):
                raise FatalError(f"CUDA context lost: {e}") from None
            idx = node.index if node is not None else None
            op_id = node.op.id if node is not None else 0
            raise ReflopError(cpb.ERROR_EXECUTION, f"{str(e)[:500]}", E3, idx, op_id) from None
        res = cpb.ExecuteResult()
        K = cpb.OutputSpec
        for spec, r in zip(ex.outputs, prog.outputs):
            t = env[r]
            if spec.kind == K.KIND_STORE:
                self.handles.put_new(spec.handle, t)
            elif spec.kind == K.KIND_REBIND:
                self.handles.replace(spec.handle, t)
            elif spec.kind == K.KIND_RETURN:
                from .safetensors import tensor_bytes

                res.returned.append(cpb.TensorData(
                    meta=gpb.TensorMeta(dtype=dtypes.TORCH_TO_PB[t.dtype], shape=list(t.shape)),
                    data=tensor_bytes(t)))
        res.duration_us = int((time.perf_counter() - t0) * 1e6)
        res.flops = prog.flops
        res.peak_memory_bytes = prog.peak_bytes
        return res


def graph_hash(g: gpb.Graph) -> bytes:
    """Server-computed hash; a client-claimed hash is never trusted (§7.3)."""
    data = g.SerializeToString(deterministic=True)
    return hashlib.sha256(b"reflop-graph-v1\x00" + g.opset_version.to_bytes(4, "big") + data).digest()
