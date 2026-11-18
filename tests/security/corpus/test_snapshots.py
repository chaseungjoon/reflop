# SPDX-License-Identifier: Apache-2.0
"""Snapshots and checkpoints in the worker (ADR-028, ADR-029): reserved
keys, all-or-nothing restore, corrupt files fail closed, a cut that later
rebinds cannot change, and the suspend helper acts only on reflop workers."""

import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from reflop.connection import default_limits
from reflop_worker._pb import compute_pb2 as cpb
from reflop_worker._pb import graph_pb2 as gpb
from reflop_worker._pb import worker_pb2 as wpb
from reflop_worker.checkpoint import META_KEY, RNG_CPU_KEY
from reflop_worker.core import Session
from reflop_worker.errors import ReflopError
from reflop_worker.safetensors import SafetensorsError, parse_header

ROOT = Path(__file__).resolve().parents[3]
H1, H2, H3 = b"\x01" * 16, b"\x02" * 16, b"\x03" * 16


def session(max_handles=0) -> Session:
    from reflop_worker._pb import capabilities_pb2 as kpb

    init = wpb.WorkerInit(seed=0)
    init.device.kind = gpb.DEVICE_CPU
    init.limits.CopyFrom(default_limits(kpb))
    if max_handles:
        init.limits.max_handles_per_session = max_handles
    return Session(init)


def write(s: Session, msg: wpb.SnapshotWrite, path) -> list:
    """Run a write the way reflopd drives it: cut, go, wait."""
    out = []
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)

    def cut(rid, _n):
        s.writer.on_go(wpb.SnapshotGo(request_id=rid, proceed=True))

    s.snapshot_write(msg, fd, cut, out.append)
    if s.writer.thread is not None:
        s.writer.thread.join()
    return out


def read(s: Session, path, keys, handles, rng=False):
    fd = os.open(path, os.O_RDONLY)
    s.snapshot_read(wpb.SnapshotRead(request_id=1, keys=keys, handles=handles, restore_rng=rng), fd)


@pytest.fixture
def saved(tmp_path):
    s = session()
    s.handles.put_new(H1, torch.arange(12.0).reshape(3, 4))
    s.handles.put_new(H2, torch.tensor([1, 2, 3]))
    p = tmp_path / "snap.safetensors"
    out = write(s, wpb.SnapshotWrite(request_id=7, handles=[H1, H2], keys=["a", "b"]), p)
    assert out[-1].WhichOneof("body") == "snapshot_saved"
    assert out[-1].snapshot_saved.bytes == p.stat().st_size
    return p


@pytest.mark.parametrize("key", ["__metadata__", "__reflop__/meta", "__reflop__/rng/cpu", "", "k" * 257])
def test_reserved_and_invalid_keys_refused(tmp_path, key):
    s = session()
    s.handles.put_new(H1, torch.zeros(2))
    with pytest.raises(ReflopError):
        write(s, wpb.SnapshotWrite(request_id=1, handles=[H1], keys=[key]), tmp_path / "x")


def test_restore_reserved_key_refused(saved, tmp_path):
    s = session()
    with pytest.raises(ReflopError):
        read(s, saved, [RNG_CPU_KEY], [H3])


def test_round_trip(saved):
    s = session()
    read(s, saved, ["a", "b"], [H3, H1])
    assert torch.equal(s.handles.t[H3], torch.arange(12.0).reshape(3, 4))
    assert torch.equal(s.handles.t[H1], torch.tensor([1, 2, 3]))


def test_restore_is_all_or_nothing(saved):
    # Before 0.2: a two-tensor restore with room for one left the first one.
    s = session(max_handles=1)
    with pytest.raises(ReflopError):
        read(s, saved, ["a", "b"], [H1, H2])
    assert len(s.handles) == 0
    # The same handle twice was silently overwritten.
    s = session()
    with pytest.raises(ReflopError):
        read(s, saved, ["a", "b"], [H1, H1])
    assert len(s.handles) == 0
    # An existing handle is never replaced, and nothing else is added.
    s = session()
    s.handles.put_new(H2, torch.zeros(1))
    with pytest.raises(ReflopError):
        read(s, saved, ["a", "b"], [H1, H2])
    assert set(s.handles.t) == {H2}
    # A missing key adds nothing.
    with pytest.raises(ReflopError):
        read(s, saved, ["a", "zzz"], [H1, H3])
    assert set(s.handles.t) == {H2}


@pytest.mark.parametrize("damage", ["truncate", "header", "garbage"])
def test_corrupt_file_fails_closed(saved, damage):
    data = saved.read_bytes()
    if damage == "truncate":
        data = data[:-5]
    elif damage == "header":
        data = struct.pack("<Q", 1 << 40) + data[8:]
    else:
        data = os.urandom(len(data))
    saved.write_bytes(data)
    s = session()
    with pytest.raises(ReflopError):
        read(s, saved, ["a", "b"], [H1, H2])
    assert len(s.handles) == 0


def test_metadata_must_be_standard():
    h = json.dumps({"__metadata__": {"k": 1}, "x": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}).encode()
    b = struct.pack("<Q", len(h)) + h + b"\0"
    with pytest.raises(SafetensorsError):
        parse_header(memoryview(b), len(b))


def test_checkpoint_is_the_state_at_the_cut(tmp_path):
    """Rebinds after the cut never reach the file, and the RNG state and tag
    are saved; restore_rng brings the RNG back."""
    s = session()
    s.handles.put_new(H1, torch.ones(1000))
    s.handles.put_new(H2, torch.full((5,), 2.0))
    torch.manual_seed(1234)
    rng = torch.get_rng_state()
    p = tmp_path / "ck"
    out = []
    fd = os.open(p, os.O_WRONLY | os.O_CREAT, 0o600)
    gate = []
    s.snapshot_write(wpb.SnapshotWrite(request_id=3, checkpoint=True, tag=b'{"step": 9}', generation=4), fd,
                     lambda rid, n: gate.append(rid), out.append)
    for _ in range(20):  # training continues before the writer may go
        s.handles.replace(H1, s.handles.t[H1] * 3)
    torch.rand(10)  # the RNG moves on too
    s.writer.on_go(wpb.SnapshotGo(request_id=gate[0], proceed=True))
    s.writer.thread.join()
    assert out[-1].checkpoint_saved.generation == 4
    data = p.read_bytes()
    start, hdr = parse_header(memoryview(data), len(data))
    meta = json.loads(data[start + hdr[META_KEY][2]:start + hdr[META_KEY][3]])
    assert meta["generation"] == 4
    t = session()
    read(t, p, [H1.hex(), H2.hex()], [H1, H2], rng=True)
    assert torch.equal(t.handles.t[H1], torch.ones(1000))
    assert torch.equal(torch.get_rng_state(), rng)


def test_refused_go_writes_nothing(tmp_path):
    s = session()
    s.handles.put_new(H1, torch.ones(4))
    out = []
    fd = os.open(tmp_path / "ck", os.O_WRONLY | os.O_CREAT, 0o600)
    s.snapshot_write(wpb.SnapshotWrite(request_id=3, checkpoint=True), fd, lambda rid, n: s.writer.on_go(
        wpb.SnapshotGo(request_id=rid, proceed=False, code=cpb.ERROR_QUOTA, reason="quota")), out.append)
    s.writer.thread.join()
    assert out[-1].error.code == cpb.ERROR_QUOTA
    assert (tmp_path / "ck").stat().st_size == 0


def test_rng_restore_checks_size(tmp_path):
    from reflop_worker import safetensors as st

    p = tmp_path / "bad"
    with open(p, "wb") as f:
        hb, _ = st.header([(H1.hex(), torch.float32, (1,)), (RNG_CPU_KEY, torch.uint8, (3,))])
        f.write(hb + b"\0" * 4 + b"\0" * 3)
    s = session()
    with pytest.raises(ReflopError):
        read(s, p, [H1.hex()], [H1], rng=True)
    assert len(s.handles) == 0


HELPER = ROOT / "server/reflopd/internal/suspend/helper.py"


@pytest.mark.parametrize("target", ["self", "init", "child"])
def test_suspend_helper_refuses_non_workers(target):
    """The suspend helper (ADR-030) refuses any process that is not a reflop
    worker started by its own parent, before touching the driver."""
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        pid = {"self": os.getpid(), "init": 1, "child": child.pid}[target]
        r = subprocess.run([sys.executable, "-I", str(HELPER), "suspend", str(pid), "100"], capture_output=True, text=True)
        assert r.returncode == 4, r.stderr
    finally:
        child.kill()
