# SPDX-License-Identifier: Apache-2.0
"""Server-resident datasets: per training step only batch indices cross the
wire (PLAN.md §10.1, §16.2)."""

from __future__ import annotations

import torch

from .. import session as _session
from .._pb import compute_pb2 as cpb
from .._pb import data_pb2 as dpb
from ..trace import PB_TO_DTYPE


class RemoteDataset:
    def __init__(self, name: str, device=None) -> None:
        idx = device.index if isinstance(device, torch.device) and device.index is not None else 0
        self.sess = _session.by_index(idx) if device is not None else _session.current()
        lst = self.sess.conn.call(cpb.Envelope(list_datasets=dpb.ListDatasets()), timeout=60).dataset_list
        found = [d for d in lst.datasets if d.name == name]
        if not found:
            names = ", ".join(d.name for d in lst.datasets) or "none"
            raise KeyError(f"no dataset named {name!r} on the server (have: {names}); push it with `reflop data push`")
        self.info = found[0]
        self.name = name
        self.id = self.info.dataset_id
        self.num_samples = int(self.info.num_samples)
        self._tensors = {}
        for t in self.info.tensors:
            meta = torch.empty(tuple(t.meta.shape), dtype=PB_TO_DTYPE[t.meta.dtype], device="meta")
            self._tensors[t.name] = self._handle_tensor(t.name, meta)

    def _handle_tensor(self, name: str, meta: torch.Tensor):
        from ..tensor import RemoteTensor, Storage

        v = self.sess.dataset_value(self.id, name, meta)
        st = Storage(self.sess, meta, None)
        st.value = v  # datasets are inputs only; never written, never outputs
        t = RemoteTensor(meta, st)
        t._rf_dataset = True
        return t

    def __len__(self) -> int:
        return self.num_samples

    @property
    def tensors(self) -> list[str]:
        return list(self._tensors)

    def batch(self, indices) -> dict[str, torch.Tensor]:
        """Gather rows on the server. `indices` is a CPU (or reflop) int tensor."""
        from ..natives import op

        if not isinstance(indices, torch.Tensor):
            indices = torch.as_tensor(indices, dtype=torch.int64)
        if indices.dtype not in (torch.int32, torch.int64):
            indices = indices.to(torch.int64)
        g = op("dataset_gather")
        return {k: g(t, indices) for k, t in self._tensors.items()}

    def loader(self, batch_size: int, shuffle: bool = True, seed: int = 0, drop_last: bool = True, epochs: int = 1):
        """Yield batches (dicts of RemoteTensors). Index permutations are made
        on the client and sent inline: ~8 bytes per sample per step."""
        g = torch.Generator().manual_seed(seed)
        n = self.num_samples
        for _epoch in range(epochs):
            order = torch.randperm(n, generator=g) if shuffle else torch.arange(n)
            stop = n - (n % batch_size) if drop_last else n
            for i in range(0, stop, batch_size):
                yield self.batch(order[i:i + batch_size])
