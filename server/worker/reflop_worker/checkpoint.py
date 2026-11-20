# SPDX-License-Identifier: Apache-2.0
"""Snapshot and checkpoint files, written in the background (ADR-029).

Server tensors are never modified in place (a rebind replaces the table
entry), so holding references to the current tensors gives a consistent
point-in-time view. The cut is taken on the request thread between two
requests: requests before it are in the file, later ones are not. A
background thread then copies the tensors to the host through a small
pinned ring buffer on a separate CUDA stream and writes them to the file
descriptor reflopd passed. Execution continues meanwhile.

Costs, measured on an RTX 5070 (docs/fault_tolerance.md):
- GPU memory: a tensor rebound during the copy keeps its old version
  allocated until it has been copied, so the session's state can need up to
  twice its size. The cut is refused when the GPU does not have that much
  free memory, and each reference is dropped as soon as its tensor has been
  copied.
- Host memory: two RING-byte buffers per session, pinned once.
- Nothing is cached between checks; the file is plain safetensors and is
  validated again in full when it is read back (I6, I9).
"""

from __future__ import annotations

import base64
import json
import os
import threading
import time

import torch

from ._pb import compute_pb2 as cpb
from ._pb import worker_pb2 as wpb
from .errors import ReflopError
from .safetensors import RESERVED, SafetensorsError, check_key, header

RING = 64 << 20
META_KEY = RESERVED + "meta"
RNG_CPU_KEY = RESERVED + "rng/cpu"
RNG_CUDA_KEY = RESERVED + "rng/cuda"
MAX_TAG = 4096
# Headroom the GPU keeps beyond the state's size before a cut is allowed.
GPU_MARGIN = 256 << 20


class Skip(Exception):
    """The checkpoint is not taken now; the previous one stays."""


def free_device_bytes(device: torch.device) -> int:
    """Memory a new allocation can use: free on the device plus what the
    caching allocator holds but does not use."""
    free, _total = torch.cuda.mem_get_info(device)
    return free + torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)


class Ring:
    """Two host buffers for staging chunks; pinned when the device is CUDA.

    They are bytearrays so that os.write can take them directly (the runtime
    has no numpy and the worker uses no ctypes); CUDA pins them with
    cudaHostRegister."""

    def __init__(self, cuda: bool) -> None:
        self.bufs = [bytearray(RING) for _ in range(2)]
        self.views = [torch.frombuffer(b, dtype=torch.uint8) for b in self.bufs]
        self.pinned = False
        if cuda:
            rt = torch.cuda.cudart()
            ok = True
            for v in self.views:
                ok = ok and int(rt.cudaHostRegister(v.data_ptr(), RING, 0)) == 0
            self.pinned = ok


class Writer:
    """One snapshot or checkpoint write at a time per session."""

    def __init__(self, sess) -> None:
        self.sess = sess
        self.ring: Ring | None = None
        self.thread: threading.Thread | None = None
        self.go: dict[int, tuple[threading.Event, list]] = {}
        self.lock = threading.Lock()

    def busy(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    # -------------------------------------------------------------- the cut

    def plan(self, msg: wpb.SnapshotWrite) -> tuple[list, bytes, int]:
        """Take the cut. Returns (items, header bytes, total file bytes).

        items = [(key, tensor)] in file order. Runs on the request thread."""
        sess = self.sess
        if self.busy():
            raise ReflopError(cpb.ERROR_BUSY, "a snapshot or checkpoint of this session is still being written")
        items: list = []
        if msg.checkpoint:
            if msg.handles or msg.keys:
                raise ReflopError(cpb.ERROR_PROTOCOL, "a checkpoint names no handles")
            if len(msg.tag) > MAX_TAG:
                raise ReflopError(cpb.ERROR_RESOURCE_LIMIT, f"checkpoint tags are at most {MAX_TAG} bytes")
            for h, t in sorted(sess.handles.t.items()):
                items.append((h.hex(), t))
            meta = json.dumps({"generation": int(msg.generation), "created_unix": int(time.time()),
                               "tag": base64.b64encode(bytes(msg.tag)).decode()}).encode()
            items.append((META_KEY, torch.frombuffer(bytearray(meta), dtype=torch.uint8)))
            items.append((RNG_CPU_KEY, torch.get_rng_state()))
            if sess.device.type == "cuda":
                items.append((RNG_CUDA_KEY, torch.cuda.get_rng_state(sess.device)))
        else:
            if len(msg.handles) != len(msg.keys) or not msg.handles:
                raise ReflopError(cpb.ERROR_PROTOCOL, "handles and keys must be non-empty and equal in length")
            seen = set()
            for h, k in zip(msg.handles, msg.keys):
                try:
                    check_key(k)
                except SafetensorsError as e:
                    raise ReflopError(cpb.ERROR_PROTOCOL, f"snapshot key {k[:64]!r}: {e}") from None
                if k in seen:
                    raise ReflopError(cpb.ERROR_PROTOCOL, f"duplicate key {k[:64]!r}")
                seen.add(k)
                items.append((k, sess.handles.get(h)))
            items.sort(key=lambda kv: kv[0])
        try:
            hb, data = header([(k, t.dtype, tuple(t.shape)) for k, t in items])
        except SafetensorsError as e:
            raise ReflopError(cpb.ERROR_RESOURCE_LIMIT, str(e)) from None
        if sess.device.type == "cuda":
            on_device = sum(t.numel() * t.element_size() for _k, t in items if t.device.type == "cuda")
            free = free_device_bytes(sess.device)
            if on_device + GPU_MARGIN > free:
                raise Skip(f"the GPU has {free >> 20} MiB free; a checkpoint of {on_device >> 20} MiB of state "
                           f"needs that much again (plus {GPU_MARGIN >> 20} MiB) while it is copied")
        return items, hb, len(hb) + data

    # -------------------------------------------------------------- start / go

    def start(self, msg: wpb.SnapshotWrite, fd: int, cut, done) -> None:
        """Cut now, report the size with cut(rid, bytes), wait for go, write.

        done(envelope) is called once, from the writer thread or from here."""
        rid = msg.request_id
        try:
            items, hb, total = self.plan(msg)
        except Skip as e:
            os.close(fd)
            done(cpb.Envelope(request_id=rid, end=True, checkpoint_saved=cpb.CheckpointSaved(
                skipped=True, reason=str(e), generation=msg.generation)))
            return
        except BaseException:
            os.close(fd)
            raise
        ev = None
        if self.sess.device.type == "cuda":
            ev = torch.cuda.Event()
            ev.record()  # all work queued so far on the request stream
        gate = threading.Event()
        verdict: list = []
        with self.lock:
            self.go[rid] = (gate, verdict)
        if self.ring is None:
            self.ring = Ring(self.sess.device.type == "cuda")
        self.thread = threading.Thread(target=self._run, args=(msg, fd, items, hb, total, ev, gate, verdict, done),
                                       name="reflop-snapshot", daemon=True)
        self.thread.start()
        cut(rid, total)

    def on_go(self, g: wpb.SnapshotGo) -> None:
        with self.lock:
            entry = self.go.pop(g.request_id, None)
        if entry is not None:
            entry[1].append(g)
            entry[0].set()

    # -------------------------------------------------------------- the thread

    def _run(self, msg, fd, items, hb, total, ev, gate, verdict, done) -> None:
        rid = msg.request_id
        try:
            gate.wait()
            g = verdict[0]
            if not g.proceed:
                items.clear()
                done(cpb.Envelope(request_id=rid, end=True, error=cpb.Error(
                    code=g.code or cpb.ERROR_SAFETY_RAIL, message=g.reason[:500])))
                return
            self._write(fd, items, hb, ev)
            os.close(fd)
            fd = None
            if msg.checkpoint:
                done(cpb.Envelope(request_id=rid, end=True, checkpoint_saved=cpb.CheckpointSaved(
                    bytes=total, generation=msg.generation)))
            else:
                done(cpb.Envelope(request_id=rid, end=True, snapshot_saved=cpb.SnapshotSaved(bytes=total)))
        except BaseException as e:  # noqa: BLE001 - the thread must always answer
            items.clear()
            what = "checkpoint" if msg.checkpoint else "snapshot"
            done(cpb.Envelope(request_id=rid, end=True, error=cpb.Error(
                code=cpb.ERROR_INTERNAL, message=f"{what} write failed: {type(e).__name__}: {str(e)[:300]}")))
        finally:
            if fd is not None:
                os.close(fd)

    def _write(self, fd: int, items: list, hb: bytes, ev) -> None:
        _write_all(fd, memoryview(hb))
        ring = self.ring
        cuda = ev is not None
        stream = torch.cuda.Stream(self.sess.device) if cuda else None
        evs = [torch.cuda.Event() for _ in range(2)] if cuda else None
        if cuda:
            stream.wait_event(ev)
        k = 0
        pending = None  # (slot, nbytes, item index if this chunk ends the item, else -1)

        def flush_pending():
            ps, pm, last = pending
            if cuda:
                evs[ps].synchronize()
            _write_all(fd, memoryview(ring.bufs[ps])[:pm])
            if last >= 0:
                items[last] = None  # copied: the GPU memory of an old version can go

        ctx = torch.cuda.stream(stream) if cuda else _null()
        with ctx:
            for i in range(len(items)):
                t = items[i][1]
                n = t.numel() * t.element_size()
                if n == 0:
                    items[i] = None
                    continue
                src = t.detach()
                if not src.is_contiguous():
                    src = src.contiguous()
                flat = src.reshape(-1).view(torch.uint8)
                for off in range(0, n, RING):
                    m = min(RING, n - off)
                    slot = k % 2
                    if pending is not None and pending[0] == slot:
                        flush_pending()
                        pending = None
                    ring.views[slot][:m].copy_(flat[off:off + m], non_blocking=cuda and ring.pinned)
                    if cuda:
                        evs[slot].record(stream)
                    if pending is not None:
                        flush_pending()
                    pending = (slot, m, i if off + m >= n else -1)
                    k += 1
                del src, flat, t
            if pending is not None:
                flush_pending()
        if cuda:
            stream.synchronize()


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _write_all(fd: int, mv: memoryview) -> None:
    while mv:
        n = os.write(fd, mv)
        if n <= 0:
            raise OSError("short write")
        mv = mv[n:]
