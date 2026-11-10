# SPDX-License-Identifier: Apache-2.0
"""Request/response plumbing shared by every transport.

A Connection multiplexes requests by request_id. Responses arrive on a
reader thread and resolve Futures. Streams (fetch) collect frames until the
one marked `end`. Events are fixed-schema and informational: they are
printed (sanitized) and never trigger client actions (§12).
"""

from __future__ import annotations

import hashlib
import itertools
import sys
import threading
from concurrent.futures import Future

from ._pb import compute_pb2 as cpb
from .errors import OutOfMemoryError, ReflopRemoteError, SessionLostError, sanitize


def _err(e: cpb.Error) -> ReflopRemoteError:
    r = ReflopRemoteError.from_pb(e)
    if e.code == cpb.ERROR_OUT_OF_MEMORY:
        r.__class__ = OutOfMemoryError
    elif e.code == cpb.ERROR_SESSION_LOST:
        r.__class__ = SessionLostError
    return r


class Connection:
    """Base class; subclasses implement _send(Envelope)."""

    def __init__(self, chunk_bytes: int = 4 << 20, max_response_bytes: int = 1 << 34) -> None:
        self._ids = itertools.count(1)
        self._pending: dict[int, tuple[Future, list | None]] = {}
        self._lock = threading.Lock()
        self.chunk_bytes = chunk_bytes
        self.max_response_bytes = max_response_bytes
        self.closed = False
        self.on_event = self._print_event

    @staticmethod
    def _print_event(ev: cpb.Event) -> None:
        kind = cpb.Event.Kind.Name(ev.kind).removeprefix("KIND_").lower().replace("_", " ")
        sys.stderr.write(f"reflop: server {kind}: {sanitize(ev.text, 500)}\n")

    # -------------------------------------------------------------- subclass API

    def _send(self, env: cpb.Envelope) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def _deliver(self, env: cpb.Envelope) -> None:
        """Called by the transport for every incoming envelope."""
        if env.WhichOneof("body") == "event":
            try:
                self.on_event(env.event)
            except Exception:  # noqa: BLE001 - events must never break the client
                pass
            return
        with self._lock:
            entry = self._pending.get(env.request_id)
        if entry is None:
            return  # unknown or late response: ignore (never act on it)
        fut, frames = entry
        if env.WhichOneof("body") == "error":
            with self._lock:
                self._pending.pop(env.request_id, None)
            if not fut.done():
                fut.set_exception(_err(env.error))
            return
        if frames is not None:
            frames.append(env)
            if not env.end:
                return
        with self._lock:
            self._pending.pop(env.request_id, None)
        if not fut.done():
            fut.set_result(frames if frames is not None else env)

    def _fail_all(self, exc: BaseException) -> None:
        with self._lock:
            pending, self._pending = self._pending, {}
        for fut, _ in pending.values():
            if not fut.done():
                fut.set_exception(exc)

    # -------------------------------------------------------------- requests

    def request(self, env: cpb.Envelope, stream: bool = False) -> Future:
        if self.closed:
            raise SessionLostError(cpb.ERROR_SESSION_LOST, "connection is closed")
        rid = next(self._ids)
        env.request_id = rid
        fut: Future = Future()
        with self._lock:
            self._pending[rid] = (fut, [] if stream else None)
        try:
            self._send(env)
        except Exception as e:  # noqa: BLE001
            with self._lock:
                self._pending.pop(rid, None)
            fut.set_exception(e)
        return fut

    def request_stream(self, env: cpb.Envelope) -> Future:
        return self.request(env, stream=True)

    def call(self, env: cpb.Envelope, timeout: float | None = None) -> cpb.Envelope:
        return self.request(env).result(timeout)

    def upload(self, handle: bytes, meta, data: bytes) -> Future:
        """Stream one tensor; the future resolves on the final frame's reply."""
        if self.closed:
            raise SessionLostError(cpb.ERROR_SESSION_LOST, "connection is closed")
        rid = next(self._ids)
        fut: Future = Future()
        with self._lock:
            self._pending[rid] = (fut, None)
        sha = hashlib.sha256(data).digest()
        n = len(data)
        step = self.chunk_bytes
        off = 0
        while True:
            chunk = data[off:off + step]
            final = off + len(chunk) >= n
            u = cpb.UploadTensor(handle=handle, offset=off, data=chunk, final=final)
            if off == 0:
                u.meta.CopyFrom(meta)
            if final:
                u.sha256 = sha
            try:
                self._send(cpb.Envelope(request_id=rid, upload_tensor=u))
            except Exception as e:  # noqa: BLE001
                with self._lock:
                    self._pending.pop(rid, None)
                fut.set_exception(e)
                return fut
            off += len(chunk)
            if final:
                break
        return fut

    def reseed(self, seed: int) -> None:
        self.call(cpb.Envelope(reseed=cpb.Reseed(seed=seed)))

    def close(self) -> None:
        self.closed = True


class LocalConnection(Connection):
    """Runs the reflop worker's request handling in-process.

    For tests and `reflop verify`: same validation and execution code as the
    server, with no network and no sandbox. Requires the worker package
    (server/worker) on sys.path.
    """

    def __init__(self, device: str = "cpu", seed: int = 0, limits=None) -> None:
        super().__init__()
        from reflop_worker._pb import capabilities_pb2 as kpb
        from reflop_worker._pb import graph_pb2 as gpb
        from reflop_worker._pb import worker_pb2 as wpb
        from reflop_worker.core import Session

        init = wpb.WorkerInit(seed=seed)
        if device == "cpu":
            init.device.kind = gpb.DEVICE_CPU
        else:
            init.device.kind = gpb.DEVICE_CUDA
            init.device.index = int(device.split(":")[1]) if ":" in device else 0
        init.limits.CopyFrom(limits or default_limits(kpb))
        init.numerics.tf32_cudnn = True
        self.worker = Session(init)
        self.gdevice = init.device
        self.ready = self.worker.ready(wpb.SandboxReport())
        self._wlock = threading.Lock()

    def capabilities(self):
        from ._pb import capabilities_pb2 as kpb

        c = kpb.ServerCapabilities(protocol_versions=[1], opset_versions=[1],
                                   enabled_op_ids=self.ready.enabled_op_ids, capabilities=self.ready.capabilities)
        c.devices.extend(self.ready.devices)
        c.limits.CopyFrom(self.worker.limits)
        c.numerics.CopyFrom(self.ready.numerics)
        return c

    def _send(self, env: cpb.Envelope) -> None:
        with self._wlock:
            self.worker.handle(env, self._deliver)


def default_limits(kpb):
    return kpb.Limits(
        max_frame_bytes=64 << 20, max_nodes=1_000_000, max_graph_inputs=65536, max_dims=16,
        max_tensor_numel=1 << 40, max_inline_tensor_bytes=1 << 20, max_constant_bytes=64 << 10,
        max_fetch_bytes=16 << 30, max_list_length=65536, upload_chunk_bytes=4 << 20,
        max_handles_per_session=1 << 22, max_graphs_per_session=65536, max_manifest_bytes=1 << 20)
