# SPDX-License-Identifier: Apache-2.0
"""Request handling for one session. Used by the worker process and,
in-process, by the client's test backend."""

from __future__ import annotations

import collections
import hashlib
import os

import torch

from . import dtypes
from ._pb import capabilities_pb2 as kpb
from ._pb import compute_pb2 as cpb
from ._pb import graph_pb2 as gpb
from ._pb import worker_pb2 as wpb
from .cost import analyze, input_bytes
from .datasets import open_dataset
from .errors import FatalError, ReflopError
from .execute import Executor, graph_hash, tensor_from_bytes
from .graph import Builder
from .handles import HandleTable, check_new
from .meta import infer
from .ops_generated import OPS
from .safetensors import parse_header, tensor_bytes
from .safetensors import write as st_write

# Tier 2 capability keys and how to detect them on a device.
CAPABILITY_KEYS = ("linalg", "complex", "fp8", "rnn_fused", "ctc", "cdist_backward")


def torch_device(d: gpb.Device) -> torch.device:
    if d.kind == gpb.DEVICE_CUDA:
        return torch.device("cuda", d.index)
    return torch.device("cpu")


def detect_capabilities(device: torch.device, allowed: set[str] | None) -> list[str]:
    caps = {"linalg", "complex", "ctc", "cdist_backward", "rnn_fused"}
    if device.type == "cuda":
        major, minor = torch.cuda.get_device_capability(device)
        if (major, minor) >= (8, 9):
            caps.add("fp8")
        try:
            from torch.backends.cuda import is_flash_attention_available

            if is_flash_attention_available():
                caps.add("flash_attention")
        except ImportError:
            pass
        caps.add("efficient_attention")
        if torch.backends.cudnn.is_available():
            caps.add("cudnn_attention")
    keep = set()
    for c in caps:
        if c in CAPABILITY_KEYS and allowed is not None and c not in allowed:
            continue
        keep.add(c)
    return sorted(keep)


def enabled_ops(device: torch.device, caps: list[str], requested: list[int]) -> list[int]:
    dev = device.type
    req = set(requested) if requested else None
    out = []
    for i, op in OPS.items():
        if op.fn is None or dev not in op.devices:
            continue
        if op.capability and op.capability not in caps:
            continue
        if req is not None and i not in req:
            continue
        out.append(i)
    return out


def device_info(device: torch.device, gd: gpb.Device) -> kpb.DeviceInfo:
    di = kpb.DeviceInfo(device=gd)
    base = [torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.float16,
            torch.bfloat16, torch.float32, torch.float64, torch.complex64, torch.complex128]
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        di.name = p.name
        di.total_memory_bytes = p.total_memory
        di.compute_capability_major, di.compute_capability_minor = p.major, p.minor
        if (p.major, p.minor) >= (8, 9):
            base += [torch.float8_e4m3fn, torch.float8_e5m2]
        if p.major < 8:
            base.remove(torch.bfloat16)
    else:
        di.name = "CPU"
        try:
            pages = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except (ValueError, OSError):
            pages = 0
        di.total_memory_bytes = pages
    di.dtypes.extend(dtypes.TORCH_TO_PB[d] for d in base)
    return di


class Session:
    """All state of one (identity, session)."""

    def __init__(self, init: wpb.WorkerInit) -> None:
        self.init = init
        self.limits = init.limits
        self.gdevice = init.device
        self.device = torch_device(init.device)
        allowed = None  # reflopd already filtered Tier 2 capabilities in enabled_op_ids
        self.caps = detect_capabilities(self.device, allowed)
        self.enabled = enabled_ops(self.device, self.caps, list(init.enabled_op_ids))
        self.enabled_set = frozenset(self.enabled)
        self.builder = Builder(self.limits, self.enabled_set, frozenset(self.caps), self.gdevice)
        self.handles = HandleTable(self.limits.max_handles_per_session or (1 << 22))
        self.datasets: dict = {}
        self.programs: collections.OrderedDict[bytes, object] = collections.OrderedDict()
        self.max_programs = self.limits.max_graphs_per_session or 65536
        self.uploads: dict[int, dict] = {}
        self.executes = 0
        self.dev_info = device_info(self.device, self.gdevice)
        self.executor = Executor(self.device, self.handles, self.datasets, self.limits,
                                 flops_quota=init.execute_flops_quota,
                                 device_memory_quota=init.device_memory_quota_bytes)
        self._apply_numerics(init.numerics)
        torch.manual_seed(init.seed)

    def _apply_numerics(self, n: kpb.Numerics) -> None:
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = n.tf32_matmul
            torch.backends.cudnn.allow_tf32 = n.tf32_cudnn
            torch.backends.cudnn.benchmark = n.cudnn_benchmark
        torch.use_deterministic_algorithms(n.deterministic, warn_only=True)

    def numerics(self) -> kpb.Numerics:
        return kpb.Numerics(
            tf32_matmul=bool(torch.backends.cuda.matmul.allow_tf32), tf32_cudnn=bool(torch.backends.cudnn.allow_tf32),
            deterministic=torch.are_deterministic_algorithms_enabled(),
            cudnn_benchmark=bool(torch.backends.cudnn.benchmark))

    def ready(self, sandbox: wpb.SandboxReport) -> wpb.WorkerReady:
        r = wpb.WorkerReady(sandbox=sandbox, enabled_op_ids=self.enabled, capabilities=self.caps,
                            numerics=self.numerics())
        r.devices.append(self.dev_info)
        r.runtime_info.torch_version = str(torch.__version__)
        r.runtime_info.cuda_version = str(torch.version.cuda or "")
        return r

    # -------------------------------------------------------------- requests

    def handle(self, env: cpb.Envelope, send) -> None:
        """Process one client envelope; call send(Envelope) for responses."""
        rid = env.request_id
        which = env.WhichOneof("body")
        try:
            if which == "register_graph":
                send(cpb.Envelope(request_id=rid, end=True, graph_registered=self.register(env.register_graph.graph)))
            elif which == "execute":
                send(cpb.Envelope(request_id=rid, end=True, execute_result=self.execute(env.execute)))
            elif which == "upload_tensor":
                done = self.upload(rid, env.upload_tensor)
                if done is not None:
                    send(cpb.Envelope(request_id=rid, end=True, uploaded=done))
            elif which == "fetch":
                self.fetch(rid, env.fetch, send)
            elif which == "free":
                for h in env.free.handles:
                    self.handles.free(h)
                send(cpb.Envelope(request_id=rid, end=True, freed=cpb.Freed()))
            elif which == "reseed":
                torch.manual_seed(env.reseed.seed)
                send(cpb.Envelope(request_id=rid, end=True, reseeded=cpb.Reseeded()))
            elif which == "ping":
                send(cpb.Envelope(request_id=rid, end=True, pong=cpb.Pong(nonce=env.ping.nonce)))
            else:
                raise ReflopError(cpb.ERROR_PROTOCOL, f"unexpected request {which}")
        except FatalError:
            raise
        except ReflopError as e:
            if which == "upload_tensor":
                self.uploads.pop(rid, None)
            send(cpb.Envelope(request_id=rid, end=True, error=e.to_pb()))
        except torch.OutOfMemoryError as e:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            send(cpb.Envelope(request_id=rid, end=True, error=ReflopError(
                cpb.ERROR_OUT_OF_MEMORY, f"out of memory: {str(e)[:300]}").to_pb()))

    def register(self, g: gpb.Graph) -> cpb.GraphRegistered:
        h = graph_hash(g)
        prog = self.programs.get(h)
        cached = prog is not None
        if prog is None:
            # Same hash ⇒ same structural validation result (R7). Per-execute
            # checks are never cached.
            prog = self.builder.build(g)
            infer(prog, self.limits)
            analyze(prog, input_bytes(prog))
            total = self.dev_info.total_memory_bytes
            if total and prog.peak_bytes > total:
                raise ReflopError(cpb.ERROR_RESOURCE_LIMIT, f"graph needs ~{prog.peak_bytes} bytes of device memory; "
                                  f"the device has {total}", cpb.STAGE_V5_SAFETY)
            q = self.init.device_memory_quota_bytes
            if q and prog.peak_bytes > q:
                raise ReflopError(cpb.ERROR_QUOTA, f"graph needs ~{prog.peak_bytes} bytes; this identity's device "
                                  f"memory quota is {q}", cpb.STAGE_V5_SAFETY)
            prog.graph_hash = h
            self.programs[h] = prog
            while len(self.programs) > self.max_programs:
                self.programs.popitem(last=False)
        else:
            self.programs.move_to_end(h)
        r = cpb.GraphRegistered(graph_hash=h, peak_memory_bytes=prog.peak_bytes, flops=prog.flops, cached=cached)
        for dt, shape in prog.output_meta:
            r.output_meta.append(gpb.TensorMeta(dtype=dtypes.TORCH_TO_PB[dt], shape=list(shape)))
        return r

    def execute(self, ex: cpb.Execute) -> cpb.ExecuteResult:
        prog = self.programs.get(ex.graph_hash)
        if prog is None:
            raise ReflopError(cpb.ERROR_UNKNOWN_GRAPH, "graph is not registered in this session", cpb.STAGE_E1_BINDINGS)
        self.executes += 1
        return self.executor.run(prog, ex)

    def upload(self, rid: int, u: cpb.UploadTensor):
        st = self.uploads.get(rid)
        if st is None:
            check_new(u.handle)
            if u.handle in self.handles.t:
                raise ReflopError(cpb.ERROR_HANDLE_EXISTS, "handle already exists", cpb.STAGE_E1_BINDINGS)
            if u.meta.dtype not in dtypes.PB_TO_TORCH or len(u.meta.shape) > self.limits.max_dims:
                raise ReflopError(cpb.ERROR_BINDING, "invalid tensor meta", cpb.STAGE_E1_BINDINGS)
            dt = dtypes.PB_TO_TORCH[u.meta.dtype]
            try:
                n = dtypes.nbytes_checked(u.meta.shape, dt, self.limits.max_tensor_numel)
            except (ValueError, OverflowError) as e:
                raise ReflopError(cpb.ERROR_RESOURCE_LIMIT, str(e), cpb.STAGE_E1_BINDINGS) from None
            budget = self.init.host_memory_budget_bytes
            if budget and n > budget:
                raise ReflopError(cpb.ERROR_SAFETY_RAIL, "upload would exceed the host memory safety rail",
                                  cpb.STAGE_V5_SAFETY)
            st = {"handle": u.handle, "dt": dt, "shape": tuple(u.meta.shape), "buf": bytearray(n), "fill": 0,
                  "sha": hashlib.sha256()}
            self.uploads[rid] = st
        elif u.handle != st["handle"]:
            raise ReflopError(cpb.ERROR_PROTOCOL, "upload handle changed mid-stream")
        if u.offset != st["fill"] or st["fill"] + len(u.data) > len(st["buf"]):
            raise ReflopError(cpb.ERROR_PROTOCOL, "upload chunk out of order or past the end")
        st["buf"][st["fill"]:st["fill"] + len(u.data)] = u.data
        st["sha"].update(u.data)
        st["fill"] += len(u.data)
        if not u.final:
            return None
        del self.uploads[rid]
        if st["fill"] != len(st["buf"]):
            raise ReflopError(cpb.ERROR_PROTOCOL, "upload ended before all bytes arrived")
        if u.sha256 and u.sha256 != st["sha"].digest():
            raise ReflopError(cpb.ERROR_PROTOCOL, "upload checksum mismatch")
        t = tensor_from_bytes(bytes(st["buf"]), st["dt"], st["shape"], self.device)
        self.handles.put_new(st["handle"], t)
        return cpb.Uploaded(handle=st["handle"])

    def fetch(self, rid: int, f: cpb.Fetch, send) -> None:
        tensors = [self.handles.get(h) for h in f.handles]
        total = sum(t.numel() * t.element_size() for t in tensors)
        if total > self.limits.max_fetch_bytes:
            raise ReflopError(cpb.ERROR_RESOURCE_LIMIT, f"fetch of {total} bytes exceeds the limit")
        chunk = int(self.limits.upload_chunk_bytes) or (4 << 20)
        frames = []
        for i, t in enumerate(tensors):
            data = tensor_bytes(t)
            meta = gpb.TensorMeta(dtype=dtypes.TORCH_TO_PB[t.dtype], shape=list(t.shape))
            if not data:
                frames.append(cpb.FetchChunk(index=i, meta=meta, offset=0, data=b"", last_of_tensor=True))
                continue
            for off in range(0, len(data), chunk):
                c = cpb.FetchChunk(index=i, offset=off, data=data[off:off + chunk],
                                   last_of_tensor=off + chunk >= len(data))
                if off == 0:
                    c.meta.CopyFrom(meta)
                frames.append(c)
        if not frames:
            send(cpb.Envelope(request_id=rid, end=True, fetch_chunk=cpb.FetchChunk()))
            return
        for k, c in enumerate(frames):
            send(cpb.Envelope(request_id=rid, end=k == len(frames) - 1, fetch_chunk=c))

    # -------------------------------------------------------------- reflopd-internal

    def open_dataset(self, msg: wpb.DatasetOpen, fds: list[int]) -> None:
        old = self.datasets.pop(msg.dataset_id, None)
        if old is not None:
            old.close()
        budget = 0
        if self.device.type == "cuda":
            free, _total = torch.cuda.mem_get_info(self.device)
            budget = free // 4
        self.datasets[msg.dataset_id] = open_dataset(msg, fds, self.device, self.limits.max_tensor_numel, budget)

    def close_dataset(self, dataset_id: bytes) -> None:
        ds = self.datasets.pop(dataset_id, None)
        if ds is not None:
            ds.close()

    def snapshot_write(self, msg: wpb.SnapshotWrite, fd: int) -> int:
        if len(msg.handles) != len(msg.keys):
            raise ReflopError(cpb.ERROR_PROTOCOL, "handles and keys differ in length")
        tensors = {}
        for h, k in zip(msg.handles, msg.keys):
            if k in tensors:
                raise ReflopError(cpb.ERROR_PROTOCOL, f"duplicate key {k!r}")
            tensors[k] = self.handles.get(h)
        with os.fdopen(fd, "wb", closefd=True) as f:
            return st_write(f, tensors)

    def snapshot_read(self, msg: wpb.SnapshotRead, fd: int) -> None:
        if len(msg.handles) != len(msg.keys):
            raise ReflopError(cpb.ERROR_PROTOCOL, "handles and keys differ in length")
        for h in msg.handles:
            check_new(h)
            if h in self.handles.t:
                raise ReflopError(cpb.ERROR_HANDLE_EXISTS, "handle already exists")
        with os.fdopen(fd, "rb", closefd=True) as f:
            data = f.read()
        start, hdr = parse_header(memoryview(data), len(data), self.limits.max_tensor_numel)
        out = {}
        for h, k in zip(msg.handles, msg.keys):
            if k not in hdr:
                raise ReflopError(cpb.ERROR_PROTOCOL, f"snapshot has no tensor {k!r}")
            dt, shape, b, e = hdr[k]
            out[h] = tensor_from_bytes(data[start + b:start + e], dt, shape, self.device)
        for h, t in out.items():
            self.handles.put_new(h, t)

    def stats(self) -> wpb.WorkerStats:
        s = wpb.WorkerStats(handles=len(self.handles), graphs=len(self.programs), executes=self.executes)
        if self.device.type == "cuda":
            s.device_memory_allocated = torch.cuda.memory_allocated(self.device)
            s.device_memory_reserved = torch.cuda.memory_reserved(self.device)
        try:
            with open("/proc/self/statm") as f:
                s.host_rss_bytes = int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        except OSError:
            pass
        return s

