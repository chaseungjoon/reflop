# SPDX-License-Identifier: Apache-2.0
"""Server-resident datasets (PLAN.md §10.4).

reflopd passes one read-only file descriptor per shard; the worker has no
filesystem access to the store at all (ADR-004). Shards are memory-mapped,
their safetensors headers re-validated, and rows gathered either on the
host (pinned buffer, async copy) or, if the whole dataset fits comfortably,
from a device-resident copy.
"""

from __future__ import annotations

import mmap
import os

import torch

from . import safetensors
from ._pb import compute_pb2 as cpb
from .errors import ReflopError


class DatasetTensor:
    """One named tensor of a dataset, concatenated over shards on axis 0."""

    def __init__(self, name: str, dtype: torch.dtype, shape: tuple, parts: list[torch.Tensor], device) -> None:
        self.name = name
        self.dtype = dtype
        self.shape = torch.Size(shape)
        self.parts = parts            # CPU tensors backed by mmaps
        self.device = device
        self.offsets = [0]
        for p in parts:
            self.offsets.append(self.offsets[-1] + p.shape[0])
        self._whole_cpu: torch.Tensor | None = None
        self._on_device: torch.Tensor | None = None

    def nbytes(self) -> int:
        return self.shape.numel() * torch.empty((), dtype=self.dtype).element_size()

    def _cpu(self) -> torch.Tensor:
        if self._whole_cpu is None:
            self._whole_cpu = self.parts[0] if len(self.parts) == 1 else torch.cat(self.parts, 0)
        return self._whole_cpu

    def make_resident(self) -> None:
        if self._on_device is None and self.device.type == "cuda":
            self._on_device = self._cpu().to(self.device, non_blocking=False)

    def gather(self, indices: torch.Tensor) -> torch.Tensor:
        if self._on_device is not None:
            return self._on_device.index_select(0, indices.to(self._on_device.device))
        idx = indices.to("cpu", torch.int64)
        rows = self._cpu().index_select(0, idx)
        if self.device.type == "cuda":
            return rows.pin_memory().to(self.device, non_blocking=True)
        return rows


class Dataset:
    def __init__(self, dataset_id: bytes, tensors: dict[str, DatasetTensor], maps: list) -> None:
        self.id = dataset_id
        self.tensors = tensors
        self._maps = maps

    def close(self) -> None:
        self.tensors.clear()
        for m in self._maps:
            try:
                m.close()
            except (BufferError, ValueError):
                pass  # still referenced by a tensor; freed when collected


def open_dataset(msg, fds: list[int], device, max_numel: int, resident_budget: int) -> Dataset:
    """Build a Dataset from a DatasetOpen message and shard fds."""
    if len(fds) != len(msg.shards):
        for fd in fds:
            os.close(fd)
        raise ReflopError(cpb.ERROR_DATASET, "shard descriptors do not match the manifest")
    from .dtypes import PB_TO_TORCH

    want = {t.name: (PB_TO_TORCH[t.meta.dtype], tuple(t.meta.shape)) for t in msg.tensors}
    parts: dict[str, list] = {n: [] for n in want}
    maps = []
    try:
        for shard, fd in zip(msg.shards, fds):
            size = os.fstat(fd).st_size
            mm = mmap.mmap(fd, size, prot=mmap.PROT_READ) if size else None
            os.close(fd)
            if mm is None:
                raise ReflopError(cpb.ERROR_DATASET, "empty shard")
            maps.append(mm)
            start, hdr = safetensors.parse_header(memoryview(mm), size, max_numel)
            if set(hdr) != set(want):
                raise ReflopError(cpb.ERROR_DATASET, "shard tensors do not match the manifest")
            for name, (dt, shape, b, e) in hdr.items():
                wdt, wshape = want[name]
                if dt != wdt or tuple(shape[1:]) != tuple(wshape[1:]) or shape[0] != shard.num_samples:
                    raise ReflopError(cpb.ERROR_DATASET, f"shard tensor {name!r} does not match the manifest")
                if e > b:
                    t = torch.frombuffer(mm, dtype=dt, count=(e - b) // torch.empty((), dtype=dt).element_size(),
                                         offset=start + b)
                else:
                    t = torch.empty(0, dtype=dt)
                parts[name].append(t.view(shape))
    except Exception:
        for m in maps:
            try:
                m.close()
            except (BufferError, ValueError):
                pass
        raise
    tensors = {n: DatasetTensor(n, want[n][0], want[n][1], parts[n], device) for n in want}
    for n, t in tensors.items():
        if sum(p.shape[0] for p in t.parts) != t.shape[0]:
            raise ReflopError(cpb.ERROR_DATASET, f"tensor {n!r}: shard sizes do not add up")
    total = sum(t.nbytes() for t in tensors.values())
    if device.type == "cuda" and total <= resident_budget:
        for t in tensors.values():
            t.make_resident()
    return Dataset(msg.dataset_id, tensors, maps)
