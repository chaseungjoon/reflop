# SPDX-License-Identifier: Apache-2.0
"""Client session: owns the trace, flushes it into graphs, tracks handles.

One Session ↔ one server session (one worker) ↔ one reflop device index.
"""

from __future__ import annotations

import atexit
import os
import secrets
import threading
import weakref
from typing import Any

import torch

from . import _opset_generated as opset
from ._pb import compute_pb2 as cpb
from ._pb import graph_pb2 as gpb
from .errors import ReflopRemoteError
from .trace import CPU, DATASET, HANDLE, NODE, SCALAR, ArgEncoder, GraphBuilder, Trace, Value, tensor_meta

FLUSH_NODES = int(os.environ.get("REFLOP_FLUSH_NODES", "20000"))

_sessions: dict[int, Session] = {}
_default: list[Session] = []
_glock = threading.Lock()


def current() -> Session:
    if not _default:
        raise RuntimeError("no reflop session: call reflop.device() first")
    return _default[-1]


def has_default() -> bool:
    return bool(_default)


def device_count() -> int:
    return 1  # see RemoteTensor.__new__: tensors always report reflop:0


def by_index(i: int) -> Session:
    return _sessions[i]


def seed(n: int) -> None:
    current().set_seed(n)


_seeding = threading.local()


def seed_if_open(n: int) -> None:
    if getattr(_seeding, "active", False) or not _default:
        return
    _seeding.active = True
    try:
        for s in list(_sessions.values()):
            s.set_seed(n)
    finally:
        _seeding.active = False


def synchronize() -> None:
    for s in list(_sessions.values()):
        s.synchronize()


_atexit_done = False


def _register_atexit() -> None:
    global _atexit_done
    if not _atexit_done:
        _atexit_done = True
        atexit.register(_close_at_exit)


def _close_at_exit() -> None:
    """End the server sessions when the process exits. A dead process can never
    resume them, and a detached session keeps its worker (and GPU memory) for the
    Owner's session TTL. Unflushed work is dropped: nothing can read it anymore."""
    with _glock:
        sessions = list(_sessions.values())
        _sessions.clear()
        _default.clear()
    for s in sessions:
        try:
            s.conn.close()
        except Exception:  # noqa: BLE001 - best effort at exit
            pass


def _tensor_from_data(td: cpb.TensorData | None, meta: gpb.TensorMeta, data: bytes) -> torch.Tensor:
    from .trace import PB_TO_DTYPE

    dt = PB_TO_DTYPE[meta.dtype]
    shape = tuple(meta.shape)
    if not data:
        return torch.empty(shape, dtype=dt)
    buf = bytearray(data)
    return torch.frombuffer(buf, dtype=torch.uint8).view(dt).reshape(shape).clone()


def tensor_bytes(t: torch.Tensor) -> bytes:
    t = t.detach().contiguous().cpu()
    n = t.numel() * t.element_size()
    if n == 0:
        return b""
    buf = bytearray(n)
    torch.frombuffer(buf, dtype=torch.uint8).copy_(t.reshape(-1).view(torch.uint8))
    return bytes(buf)


class Session:
    def __init__(self, conn, device_index: int, gdevice: gpb.Device, caps, *, allow_local_fallback: bool = True) -> None:
        self.conn = conn
        self.device_index = device_index
        self.gdevice = gdevice
        self.caps = caps
        self.limits = caps.limits
        id_to_name = {v[0]: k for k, v in opset.OPS.items()}
        self.enabled = {id_to_name[i] for i in caps.enabled_op_ids if i in id_to_name}
        self.unsupported_names: set[str] = set()
        self.allow_local_fallback = allow_local_fallback
        self.trace = Trace()
        self.dirty: weakref.WeakSet = weakref.WeakSet()
        self.registered: set[bytes] = set()
        self.pending_free: list[bytes] = []
        self.upload_log: list | None = None  # set by reflop.capture while it records a step
        self.async_error: BaseException | None = None
        self.inflight: list = []
        self.depth = 0
        self.epoch = 0  # increments at every flush (invalidates cached view replays)
        self.lock = threading.RLock()
        self._prefix = secrets.token_bytes(8)
        self._ctr = 0
        self.stats = {"flushes": 0, "registers": 0, "executes": 0, "bytes_up": 0, "bytes_down": 0}
        self.encoder = ArgEncoder(self._value_of, self.cpu_value, self.scalar_value)
        with _glock:
            _sessions[device_index] = self
            _default.append(self)
            _register_atexit()

    # -------------------------------------------------------------- op recording

    def _value_of(self, t) -> Value:
        from .tensor import value_of

        return value_of(t)

    def op_enabled(self, name: str) -> bool:
        return name in self.enabled

    def record_values(self, name: str, func, vals: list, metas: list, self_is_value: bool = False) -> list[Value]:
        entry = opset.OPS.get(name)
        if entry is None or name not in self.enabled:
            from .trace import UnsupportedOperatorError

            raise UnsupportedOperatorError(f"{name} is not available on this server")
        args = self.encoder.encode(name, entry, vals)
        outs = self.trace.record(name, func, args, metas, bool(entry[1] & opset.FLAG_RNG))
        return outs

    def record_simple(self, name: str, func, vals: list, meta) -> list[Value]:
        return self.record_values(name, func, vals, [meta])

    def mark_dirty(self, storage) -> None:
        self.dirty.add(storage)

    def new_handle(self) -> bytes:
        self._ctr += 1
        h = self._prefix + self._ctr.to_bytes(8, "big")
        if h[0] == 0xFF:  # reserved range
            h = b"\x00" + h[1:]
        return h

    def cpu_value(self, t: torch.Tensor) -> Value:
        from .tensor import _to_meta

        return Value(CPU, _to_meta(t), payload=t.detach().clone())

    def scalar_value(self, x) -> Value:
        return Value(SCALAR, None, payload=x)

    def dataset_value(self, dataset_id: bytes, name: str, meta: torch.Tensor) -> Value:
        return Value(DATASET, meta, payload=(dataset_id, name))

    def upload_value(self, t: torch.Tensor, source: torch.Tensor | None = None) -> Value:
        """Upload a CPU tensor. `source` is the tensor the program moved (before
        any dtype cast); reflop.capture uses it to rebind per-step uploads."""
        h = self.new_handle()
        self._upload(h, t)
        if self.upload_log is not None and source is not None:
            self.upload_log.append((h, source, (source._version, tuple(source.shape), source.stride(),
                                                source.dtype), t.dtype))
        return Value(HANDLE, t.to("meta") if t.dim() else torch.empty((), dtype=t.dtype, device="meta"), handle=h)

    def upload_tensor(self, t: torch.Tensor, source: torch.Tensor | None = None):
        from .tensor import RemoteTensor, Storage

        v = self.upload_value(t.detach(), source)
        st = Storage(self, v.meta, v)
        return RemoteTensor(v.meta, st)

    def _upload(self, h: bytes, t: torch.Tensor) -> None:
        data = tensor_bytes(t)
        self.stats["bytes_up"] += len(data)
        fut = self.conn.upload(h, tensor_meta(t), data)
        self._track(fut, "upload")

    def release(self, h: bytes) -> None:
        self.pending_free.append(h)
        if len(self.pending_free) >= 4096 and self.depth == 0 and len(self.trace) == 0:
            self._send_frees()

    def _send_frees(self) -> None:
        if self.pending_free:
            hs, self.pending_free = self.pending_free, []
            self._track(self.conn.request(cpb.Envelope(free=cpb.Free(handles=hs))), "free")

    # -------------------------------------------------------------- async errors

    def _track(self, fut, what: str) -> None:
        def done(f):
            e = f.exception()
            if e is not None and self.async_error is None:
                self.async_error = e
        fut.add_done_callback(done)
        self.inflight.append(fut)
        if len(self.inflight) > 256:
            self.inflight = [f for f in self.inflight if not f.done()]

    def raise_async(self) -> None:
        e = self.async_error
        if e is not None:
            self.async_error = None
            if isinstance(e, ReflopRemoteError):
                raise ReflopRemoteError(e.code, f"an earlier asynchronous operation failed: {e.message}",
                                        e.stage, e.node, e.op_name) from e
            raise e

    def synchronize(self) -> None:
        with self.lock:
            self.flush()
            for f in list(self.inflight):
                try:
                    f.result()
                except Exception:  # noqa: BLE001 - surfaced by raise_async
                    pass
            self.inflight.clear()
            self.raise_async()

    # -------------------------------------------------------------- flush

    def maybe_flush(self) -> None:
        if self.depth == 0 and len(self.trace) >= FLUSH_NODES:
            self.flush()

    def flush(self, returns: list[Value] | None = None, plan: dict | None = None) -> list[torch.Tensor]:
        """Execute the pending trace. Live tensors become server handles; the
        values in `returns` come back as CPU tensors. With `plan`, the built
        graph, output specs and freed handles are recorded (reflop.capture)."""
        returns = returns or []
        self._plan = plan
        with self.lock:
            self.raise_async()
            outputs: list[Value] = []
            specs: list[cpb.OutputSpec] = []
            owners: list = []
            # Deterministic output order keeps the graph hash stable across steps.
            def order(st):
                v = st.value
                return (0, v.node, v.index, st.seq) if v is not None and v.kind == NODE else (1, 0, 0, st.seq)
            for st in sorted(self.dirty, key=order):
                v = st.value
                if v is None or v.kind == HANDLE:
                    continue
                if v.kind == CPU:
                    st.set(self.upload_value(v.payload.expand(st.meta.shape).to(st.meta.dtype).contiguous()))
                    st.handle = st.value.handle
                    continue
                if v.kind != NODE:
                    continue
                if st.handle is None:
                    h = self.new_handle()
                    specs.append(cpb.OutputSpec(kind=cpb.OutputSpec.KIND_STORE, handle=h))
                else:
                    h = st.handle
                    specs.append(cpb.OutputSpec(kind=cpb.OutputSpec.KIND_REBIND, handle=h))
                outputs.append(v)
                owners.append((st, h))
            ret_pos = []
            for i, v in enumerate(returns):
                if v.kind == NODE:
                    ret_pos.append(i)
                    outputs.append(v)
                    specs.append(cpb.OutputSpec(kind=cpb.OutputSpec.KIND_RETURN))
            self.dirty = weakref.WeakSet()
            results: list = [None] * len(returns)
            if outputs:
                self._execute(outputs, specs, owners, ret_pos, results)
            self.trace = Trace()
            self.epoch += 1
            if plan is not None:
                plan["freed"] = set(self.pending_free)
                plan.setdefault("specs", specs)
                plan.setdefault("owners", owners)
            self._send_frees()
            self._plan = None
            for i, v in enumerate(returns):
                if results[i] is not None:
                    continue
                if v.kind == HANDLE:
                    results[i] = self.fetch_handles([v.handle])[0]
                elif v.kind == CPU:
                    results[i] = v.payload
                elif v.kind == SCALAR:
                    results[i] = torch.tensor(v.payload)
                elif v.kind == DATASET:
                    raise RuntimeError("cannot fetch a whole dataset")
            return results

    def _execute(self, outputs, specs, owners, ret_pos, results) -> None:
        built = GraphBuilder(self.trace, self.gdevice, outputs).build()
        if getattr(self, "_plan", None) is not None:
            self._plan["built"] = built
        self.stats["flushes"] += 1
        if built.graph_hash not in self.registered:
            fut = self.conn.request(cpb.Envelope(register_graph=cpb.RegisterGraph(graph=built.graph)))
            self.stats["registers"] += 1
            expect = built.graph_hash

            def check(f, expect=expect):
                if f.exception() is not None:
                    self.registered.discard(expect)
                elif f.result().graph_registered.graph_hash != expect and self.async_error is None:
                    self.async_error = RuntimeError("the server computed a different graph hash")
            fut.add_done_callback(check)
            self._track(fut, "register")
            self.registered.add(built.graph_hash)
        ex = cpb.Execute(graph_hash=built.graph_hash)
        temp = self.bind_inputs(ex, built.input_values)
        ex.outputs.extend(specs)
        fut = self.conn.request(cpb.Envelope(execute=ex))
        self.stats["executes"] += 1
        for st, h in owners:
            st.set(Value(HANDLE, st.meta, handle=h))
            st.handle = h
        self.pending_free.extend(temp)
        if ret_pos:
            try:
                r = fut.result()  # raises ReflopRemoteError
            except ReflopRemoteError:
                self.raise_async()  # a failed registration explains a failed execute
                raise
            for k, pos in enumerate(ret_pos):
                td = r.execute_result.returned[k]
                results[pos] = _tensor_from_data(td, td.meta, td.data)
                self.stats["bytes_down"] += len(td.data)
        else:
            self._track(fut, "execute")

    def bind_inputs(self, ex: cpb.Execute, values: list) -> list[bytes]:
        """Fill ex.bindings for graph input values; returns temporary handles."""
        temp: list[bytes] = []
        inline_max = int(self.limits.max_inline_tensor_bytes or (1 << 20))
        for v in values:
            b = ex.bindings.add()
            if v.kind == HANDLE:
                b.handle = v.handle
            elif v.kind == CPU:
                t = v.payload
                nb = t.numel() * t.element_size()
                if nb <= inline_max:
                    b.inline.meta.CopyFrom(tensor_meta(t))
                    b.inline.data = tensor_bytes(t)
                    self.stats["bytes_up"] += nb
                else:
                    h = self.new_handle()
                    self._upload(h, t)
                    temp.append(h)
                    b.handle = h
            elif v.kind == SCALAR:
                x = v.payload
                if isinstance(x, bool):
                    b.scalar.b = x
                elif isinstance(x, int):
                    b.scalar.i = x
                elif isinstance(x, complex):
                    b.scalar.c.real, b.scalar.c.imag = x.real, x.imag
                else:
                    b.scalar.f = float(x)
            elif v.kind == DATASET:
                b.dataset.dataset_id, b.dataset.tensor = v.payload
        return temp

    # -------------------------------------------------------------- fetch

    def fetch_handles(self, handles: list[bytes]) -> list[torch.Tensor]:
        frames = self.conn.request_stream(cpb.Envelope(fetch=cpb.Fetch(handles=handles))).result()
        parts: dict[int, list] = {}
        metas: dict[int, gpb.TensorMeta] = {}
        for env in frames:
            c = env.fetch_chunk
            if c.HasField("meta"):
                metas[c.index] = c.meta
            parts.setdefault(c.index, []).append(c.data)
        out = []
        for i in range(len(handles)):
            data = b"".join(parts.get(i, []))
            self.stats["bytes_down"] += len(data)
            out.append(_tensor_from_data(None, metas[i], data))
        return out

    def fetch(self, v: Value, meta=None) -> torch.Tensor:
        return self.flush([v])[0]

    def fetch_scalar(self, v: Value) -> Any:
        return self.fetch(v).item()

    def set_seed(self, n: int) -> None:
        self.flush()
        _seeding.active, prev = True, getattr(_seeding, "active", False)
        try:
            self.conn.reseed(n)
        finally:
            _seeding.active = prev

    def close(self) -> None:
        try:
            self.synchronize()
        finally:
            with _glock:
                _sessions.pop(self.device_index, None)
                if self in _default:
                    _default.remove(self)
            self.conn.close()
