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
from .meta import DatasetMeta, call_args
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
    return t.to(device)


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
        for o, (spec, r) in enumerate(zip(ex.outputs, prog.outputs)):
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

    def run(self, prog: Program, ex: cpb.Execute) -> cpb.ExecuteResult:
        if self.flops_quota and prog.flops > self.flops_quota:
            raise ReflopError(cpb.ERROR_QUOTA, f"graph needs ~{prog.flops} FLOPs; the per-execute quota is "
                              f"{self.flops_quota}", cpb.STAGE_V5_SAFETY)
        env = self.bind(prog, ex)
        scalars = env.pop("__scalars__")
        t0 = time.perf_counter()
        expiring: dict[int, list] = {}
        for r, last in prog.last_use.items():
            expiring.setdefault(last, []).append(r)
        dev = self.device
        protected = set(prog.outputs)
        node = None
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
        try:
            with torch.inference_mode():
                for node in prog.nodes:
                    def resolve(a):
                        if isinstance(a, Ref):
                            return env[a]
                        if isinstance(a, ScalarRef):
                            return scalars[a.input]
                        if isinstance(a, list):
                            return [resolve(x) for x in a]
                        if a == "session-device":
                            return dev
                        return a

                    pos, kw = call_args(node, resolve)
                    validators.check_runtime(node, pos, kw)
                    if natives.is_native(node.op):
                        out = natives.call(node.op.name, pos, kw, dev)
                    else:
                        out = node.op.fn(*pos, **kw)
                    flat = [x for x in tree_flatten(out)[0] if isinstance(x, torch.Tensor)]
                    if len(flat) != node.outputs:
                        raise ReflopError(cpb.ERROR_INTERNAL, f"{node.op.name}: produced {len(flat)} outputs, "
                                          f"validated {node.outputs}", E3, node.index, node.op.id)
                    uninit = node.op.flags & FLAG_UNINIT
                    for k, t in enumerate(flat):
                        r = Ref(node.index, k)
                        m = prog.metas[r]
                        # A few kernels give unused auxiliary outputs a different
                        # shape than their meta kernel (e.g. _embedding_bag's
                        # offset2bag). Anything that is consumed must match.
                        if (t.dtype != m.dtype or tuple(t.shape) != tuple(m.shape)) and r in prog.last_use:
                            raise ReflopError(cpb.ERROR_INTERNAL, f"{node.op.name}: output {k} is {t.dtype}"
                                              f"{list(t.shape)}, validated {m.dtype}{list(m.shape)}", E3,
                                              node.index, node.op.id)
                        if uninit:
                            t.zero_()  # never return stale memory (I13)
                        env[Ref(node.index, k)] = t
                    for r in expiring.get(node.index, ()):
                        if r not in protected and r.node >= 0:
                            env.pop(r, None)
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
        except ReflopError:
            raise
        except torch.OutOfMemoryError as e:
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            raise ReflopError(cpb.ERROR_OUT_OF_MEMORY, f"out of device memory: {str(e)[:300]}", E3) from None
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
            if isinstance(t, DatasetMeta):  # pragma: no cover - rejected in V2
                raise ReflopError(cpb.ERROR_INTERNAL, "dataset output", E3)
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
        if dev.type == "cuda":
            res.peak_memory_bytes = torch.cuda.max_memory_allocated(dev)
        return res


def graph_hash(g: gpb.Graph) -> bytes:
    """Server-computed hash; a client-claimed hash is never trusted (§7.3)."""
    data = g.SerializeToString(deterministic=True)
    return hashlib.sha256(b"reflop-graph-v1\x00" + g.opset_version.to_bytes(4, "big") + data).digest()
