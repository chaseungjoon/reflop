# SPDX-License-Identifier: Apache-2.0
"""Worker host memory (issue #16): uploads in progress count together
against the host budget, finishing an upload makes no extra copies, and a
fetch holds one private copy plus its frames (about 2x, was about 4x).

Before: each upload stream was checked alone (ten parallel streams could each
allocate a whole budget), finishing an upload copied the data twice more
(about 3x the tensor at the peak), and a fetch built every chunk before
sending anything (about 4x).
"""

from __future__ import annotations

import tracemalloc

import pytest
import torch
from helpers import call, make_session
from reflop_worker._pb import compute_pb2 as cpb
from reflop_worker.dtypes import TORCH_TO_PB

MiB = 1 << 20
CHUNK = 4 * MiB


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA")
    return request.param


def _handle(i: int) -> bytes:
    return bytes([1]) + i.to_bytes(15, "big")


def _meta(dt, shape):
    from reflop_worker._pb import graph_pb2 as gpb

    return gpb.TensorMeta(dtype=TORCH_TO_PB[dt], shape=list(shape))


def _send(sess, rid, h, nbytes, offset, data, final):
    up = cpb.UploadTensor(handle=h, meta=_meta(torch.uint8, (nbytes,)), offset=offset, data=data, final=final)
    return call(sess, cpb.Envelope(request_id=rid, upload_tensor=up))


def _body(out):
    return out[0].WhichOneof("body") if out else None


# ---------------------------------------------------------------- uploads


def test_uploads_in_progress_count_together():
    sess = make_session("cpu", host_budget=64 * MiB)
    # Stream 1 declares 40 MiB and sends its first chunk.
    assert _send(sess, 1, _handle(1), 40 * MiB, 0, b"\1" * CHUNK, False) == []
    # Stream 2 would bring the uploads in progress to 80 MiB: refused.
    out = _send(sess, 2, _handle(2), 40 * MiB, 0, b"\1" * CHUNK, False)
    assert _body(out) == "error" and out[0].error.code == cpb.ERROR_SAFETY_RAIL
    assert "in uploads in progress" in out[0].error.message
    # A small one still fits next to stream 1.
    assert _body(_send(sess, 3, _handle(3), 8 * MiB, 0, b"\1" * (8 * MiB), True)) == "uploaded"
    # Finishing stream 1 releases its reservation.
    for off in range(CHUNK, 40 * MiB, CHUNK):
        out = _send(sess, 1, _handle(1), 40 * MiB, off, b"\1" * CHUNK, off + CHUNK >= 40 * MiB)
    assert _body(out) == "uploaded"
    assert _body(_send(sess, 4, _handle(4), 40 * MiB, 0, b"\1" * CHUNK, False)) is None  # accepted, in progress


def test_a_failed_stream_releases_its_reservation():
    sess = make_session("cpu", host_budget=64 * MiB)
    assert _send(sess, 1, _handle(1), 40 * MiB, 0, b"\1" * CHUNK, False) == []
    out = _send(sess, 1, _handle(1), 40 * MiB, 3 * CHUNK, b"\1" * CHUNK, False)  # out of order
    assert _body(out) == "error" and out[0].error.code == cpb.ERROR_PROTOCOL
    assert _body(_send(sess, 2, _handle(2), 40 * MiB, 0, b"\1" * CHUNK, False)) is None


def test_one_upload_larger_than_the_budget_is_refused():
    sess = make_session("cpu", host_budget=16 * MiB)
    out = _send(sess, 1, _handle(1), 17 * MiB, 0, b"\1" * CHUNK, False)
    assert _body(out) == "error" and out[0].error.code == cpb.ERROR_SAFETY_RAIL


def _finish_peak(device, n):
    """Host memory allocated while the last chunk of an n-byte upload is
    handled (the envelope is built before measuring)."""
    sess = make_session(device)
    h = _handle(7)
    data = bytes(range(256)) * (CHUNK // 256)
    for off in range(0, n - CHUNK, CHUNK):
        assert _send(sess, 1, h, n, off, data, False) == []
    last = cpb.Envelope(request_id=1, upload_tensor=cpb.UploadTensor(
        handle=h, meta=_meta(torch.uint8, (n,)), offset=n - CHUNK, data=data, final=True))
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        out = call(sess, last)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert _body(out) == "uploaded"
    t = sess.handles.get(h)
    assert t.device.type == device and t.numel() == n
    assert torch.equal(t[:256].cpu(), torch.arange(256, dtype=torch.uint8))
    return peak


def test_finishing_an_upload_makes_no_extra_host_copies(device):
    # What remains is a few copies of the last chunk (protobuf hands out a new
    # bytes object per read), the same for any tensor size. Before: 2n more.
    small, big = _finish_peak(device, 16 * MiB), _finish_peak(device, 64 * MiB)
    assert big < 3 * CHUNK, f"finishing a 64 MiB upload allocated {big >> 20} MiB on the host"
    assert big - small < MiB, f"the peak grows with the tensor: {small >> 20} MiB, then {big >> 20} MiB"


# ---------------------------------------------------------------- fetches


def _fetch(sess, handles):
    frames = []
    call_frames = call(sess, cpb.Envelope(request_id=9, fetch=cpb.Fetch(handles=handles)))
    frames.extend(call_frames)
    return frames


def _reassemble(frames, count):
    parts = {i: [] for i in range(count)}
    metas = {}
    for env in frames:
        c = env.fetch_chunk
        if c.HasField("meta"):
            metas[c.index] = c.meta
        parts[c.index].append((c.offset, c.data))
    out = {}
    for i in range(count):
        data = b"".join(d for _, d in sorted(parts[i]))
        out[i] = (metas[i], data)
    return out


def test_fetch_holds_at_most_two_copies(device):
    n = 64 * MiB
    sess = make_session(device)
    h = _handle(11)
    sess.handles.put_new(h, torch.arange(n // 4, dtype=torch.float32, device=device))
    sent = []
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()

        def send(env):
            sent.append((env, tracemalloc.get_traced_memory()[0]))

        sess.fetch(9, cpb.Fetch(handles=[h]), send)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    envs = [e for e, _ in sent]
    assert len(envs) == n // CHUNK
    assert [e.end for e in envs] == [False] * (len(envs) - 1) + [True]
    # One private copy of the tensor plus the frames built from it. Before:
    # a copy, a bytearray and a bytes of the whole tensor plus the frames,
    # about 4n.
    assert peak <= 2 * n + 2 * CHUNK, f"fetching {n >> 20} MiB allocated {peak >> 20} MiB on the host"
    meta, data = _reassemble(envs, 1)[0]
    assert list(meta.shape) == [n // 4]
    assert torch.equal(torch.frombuffer(bytearray(data), dtype=torch.float32), torch.arange(n // 4, dtype=torch.float32))


@pytest.mark.parametrize("case", ["several", "empty", "non_contiguous", "exact_chunks", "bool", "bfloat16"])
def test_fetch_returns_the_exact_bytes(device, case):
    sess = make_session(device, upload_chunk_bytes=1024)
    g = torch.Generator().manual_seed(0)
    if case == "several":
        ts = [torch.randn(1000, generator=g), torch.arange(7, dtype=torch.int64), torch.randn(3, 300, generator=g)]
    elif case == "empty":
        ts = [torch.empty(0, 5), torch.randn(4, generator=g), torch.empty(0)]
    elif case == "non_contiguous":
        ts = [torch.randn(64, 64, generator=g).t()[:, ::3]]
    elif case == "exact_chunks":
        ts = [torch.randn(512, generator=g)]  # 2048 bytes: two full chunks
    elif case == "bool":
        ts = [torch.rand(3001, generator=g) > 0.5]
    else:
        ts = [torch.randn(777, generator=g).to(torch.bfloat16)]
    handles = []
    for i, t in enumerate(ts):
        h = _handle(100 + i)
        sess.handles.put_new(h, t.to(device))
        handles.append(h)
    frames = _fetch(sess, handles)
    assert frames[-1].end and not any(f.end for f in frames[:-1])
    got = _reassemble(frames, len(ts))
    for i, t in enumerate(ts):
        meta, data = got[i]
        assert list(meta.shape) == list(t.shape)
        want = t.contiguous().reshape(-1).view(torch.uint8)
        assert data == bytes(bytearray(want.tolist())), case


def test_fetch_of_nothing_sends_one_empty_frame():
    sess = make_session("cpu")
    frames = _fetch(sess, [])
    assert len(frames) == 1 and frames[0].end
