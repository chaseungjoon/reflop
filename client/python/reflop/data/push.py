# SPDX-License-Identifier: Apache-2.0
"""Upload a TensorPack: chunked, verified per chunk, resumable."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from .._pb import compute_pb2 as cpb
from .._pb import data_pb2 as dpb
from .tensorpack import read_manifest

WINDOW = 16


def push(conn, pack: str, name: str, progress=True) -> dpb.DatasetInfo:
    raw, man = read_manifest(pack)
    for i, sh in enumerate(man["shards"]):
        f = Path(pack) / sh["file"]
        if f.stat().st_size != sh["size_bytes"]:
            raise RuntimeError(f"{f} does not match the manifest; rebuild the pack")
    st = conn.call(cpb.Envelope(pack_begin=dpb.PackBegin(name=name, manifest=raw)), timeout=120).pack_status
    chunk = int(st.chunk_bytes)
    todo = [(s.shard, c) for s in st.shards for c in s.missing_chunks]
    total = len(todo)
    inflight = []
    done = 0
    for shard, c in todo:
        path = Path(pack) / man["shards"][shard]["file"]
        with open(path, "rb") as f:
            f.seek(c * chunk)
            data = f.read(chunk)
        inflight.append(conn.request(cpb.Envelope(pack_chunk=dpb.PackChunk(
            pack_id=st.pack_id, shard=shard, chunk=c, data=data, sha256=hashlib.sha256(data).digest()))))
        if len(inflight) >= WINDOW:
            inflight.pop(0).result()
            done += 1
            if progress:
                sys.stderr.write(f"\rreflop: uploading {done}/{total} chunks")
    for fut in inflight:
        fut.result()
        done += 1
    if progress and total:
        sys.stderr.write(f"\rreflop: uploaded {total} chunks{' ' * 10}\n")
    return conn.call(cpb.Envelope(pack_commit=dpb.PackCommit(pack_id=st.pack_id)), timeout=600).dataset_info
