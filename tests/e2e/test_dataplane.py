# SPDX-License-Identifier: Apache-2.0
"""M3 acceptance against a running reflopd (REFLOP_E2E=1, paired client).

TensorPack build → interrupted upload → resumed upload → remote training
where each step sends only batch indices."""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(os.environ.get("REFLOP_E2E") != "1", reason="needs a running reflopd (REFLOP_E2E=1)")


def make_pack(tmp_path, n=6000):
    from reflop.data import write_pack

    g = torch.Generator().manual_seed(0)
    images = torch.randint(0, 256, (n, 3, 32, 32), dtype=torch.uint8, generator=g)
    labels = (images.float().mean((1, 2, 3)) > 127.5).long()  # learnable rule
    return write_pack({"images": images, "labels": labels}, tmp_path / "pack", shard_bytes=4 << 20), images, labels


def test_push_resume_and_train(tmp_path):
    import reflop
    from reflop._pb import compute_pb2 as cpb
    from reflop._pb import data_pb2 as dpb
    from reflop.client import admin
    from reflop.data.push import push
    from reflop.data.tensorpack import read_manifest

    pack, images, labels = make_pack(tmp_path)
    raw, man = read_manifest(pack)
    assert len(man["shards"]) >= 4
    c = admin(None)
    name = f"e2e-{os.getpid()}"
    try:
        # Interrupted upload: send only the first two chunks, then drop the connection.
        st = c.call(cpb.Envelope(pack_begin=dpb.PackBegin(name=name, manifest=raw)), timeout=60).pack_status
        total = sum(len(s.missing_chunks) for s in st.shards)
        import hashlib

        sent = 0
        for s in st.shards[:1]:
            for ch in s.missing_chunks[:2]:
                with open(pack / man["shards"][s.shard]["file"], "rb") as f:
                    f.seek(ch * st.chunk_bytes)
                    data = f.read(st.chunk_bytes)
                c.call(cpb.Envelope(pack_chunk=dpb.PackChunk(pack_id=st.pack_id, shard=s.shard, chunk=ch, data=data,
                                                             sha256=hashlib.sha256(data).digest())), timeout=60)
                sent += 1
        c.close()
        c = admin(None)
        st2 = c.call(cpb.Envelope(pack_begin=dpb.PackBegin(name=name, manifest=raw)), timeout=60).pack_status
        assert sum(len(s.missing_chunks) for s in st2.shards) == total - sent
        info = push(c, str(pack), name, progress=False)
        assert info.num_samples == images.shape[0]
    finally:
        c.close()

    dev = reflop.device()
    ds = reflop.dataset(name)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 32 * 32, 64), torch.nn.ReLU(),
                                torch.nn.Linear(64, 2)).to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    mean = torch.tensor(127.5).to(dev)
    before = dict(reflop.stats()[0])
    steps = 0
    first = last = None
    for batch in ds.loader(batch_size=128, shuffle=True, seed=0, epochs=2):
        x = (batch["images"].float() - mean) / 64.0
        loss = F.cross_entropy(model(x), batch["labels"])
        opt.zero_grad()
        loss.backward()
        opt.step()
        steps += 1
        if steps == 1:
            first = loss.item()
    last = loss.item()
    after = reflop.stats()[0]
    per_step = (after["bytes_up"] - before["bytes_up"]) / steps
    assert last < first, (first, last)
    # Index-level batching: the indices once per step (batch 128 × int64), not
    # once per dataset tensor and not images.
    assert per_step < 128 * 8 + 64, per_step
    c = admin(None)
    lst = c.call(cpb.Envelope(list_datasets=dpb.ListDatasets()), timeout=30).dataset_list.datasets
    did = [d.dataset_id for d in lst if d.name == name][0]
    c.call(cpb.Envelope(delete_dataset=dpb.DeleteDataset(dataset_id=did)), timeout=30)
    c.close()
    reflop.close()
