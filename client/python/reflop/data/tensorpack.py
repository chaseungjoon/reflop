# SPDX-License-Identifier: Apache-2.0
"""Build TensorPacks on the client.

Decoding (images, audio, text tokenization) happens here, on the Guest's
machine; the server only ever sees tensors (non-goal N7).

    mypack/
      manifest.json
      shard-00000.safetensors
      ...
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import struct
from collections.abc import Iterable, Mapping
from pathlib import Path

import torch

NAME_RE = re.compile(r"^[a-z0-9_]{1,64}$")
DTYPES = {torch.bool: ("bool", "BOOL"), torch.uint8: ("uint8", "U8"), torch.int8: ("int8", "I8"),
          torch.int16: ("int16", "I16"), torch.int32: ("int32", "I32"), torch.int64: ("int64", "I64"),
          torch.float16: ("float16", "F16"), torch.bfloat16: ("bfloat16", "BF16"),
          torch.float32: ("float32", "F32"), torch.float64: ("float64", "F64")}
MANIFEST_TO_DTYPE = {v[0]: k for k, v in DTYPES.items()}


def _tensor_bytes(t: torch.Tensor) -> bytes:
    t = t.detach().contiguous().cpu()
    n = t.numel() * t.element_size()
    if n == 0:
        return b""
    buf = bytearray(n)
    torch.frombuffer(buf, dtype=torch.uint8).copy_(t.reshape(-1).view(torch.uint8))
    return bytes(buf)


def _write_shard(path: Path, tensors: dict[str, torch.Tensor]) -> tuple[str, int]:
    hdr, off, blobs = {}, 0, []
    for name in sorted(tensors):
        t = tensors[name]
        b = _tensor_bytes(t)
        hdr[name] = {"dtype": DTYPES[t.dtype][1], "shape": list(t.shape), "data_offsets": [off, off + len(b)]}
        off += len(b)
        blobs.append(b)
    h = json.dumps(hdr, separators=(",", ":")).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    sha = hashlib.sha256()
    with open(path, "wb") as f:
        for part in (struct.pack("<Q", len(h)), h, *blobs):
            f.write(part)
            sha.update(part)
    return sha.hexdigest(), 8 + len(h) + off


def write_pack(tensors: Mapping[str, torch.Tensor], out: str | os.PathLike, shard_bytes: int = 256 << 20) -> Path:
    """Write full tensors (sample axis first) as a TensorPack."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    names = list(tensors)
    if not names:
        raise ValueError("no tensors")
    n = None
    for k, t in tensors.items():
        if not NAME_RE.match(k):
            raise ValueError(f"tensor name {k!r} must match [a-z0-9_]{{1,64}}")
        if t.dtype not in DTYPES:
            raise ValueError(f"tensor {k!r}: dtype {t.dtype} is not supported in TensorPacks")
        if t.dim() == 0:
            raise ValueError(f"tensor {k!r} needs a sample axis")
        if n is None:
            n = t.shape[0]
        elif t.shape[0] != n:
            raise ValueError("all tensors must have the same number of samples")
    row = sum(t[0].numel() * t.element_size() for t in tensors.values()) if n else 1
    per_shard = max(1, shard_bytes // max(row, 1))
    shards = []
    for i, start in enumerate(range(0, max(n, 1), per_shard)):
        end = min(start + per_shard, n)
        part = {k: t[start:end] for k, t in tensors.items()}
        fn = f"shard-{i:05d}.safetensors"
        sha, size = _write_shard(out / fn, part)
        shards.append({"file": fn, "sha256": sha, "size_bytes": size, "num_samples": end - start})
    manifest = {"format_version": 1, "num_samples": n,
                "tensors": [{"name": k, "dtype": DTYPES[t.dtype][0], "shape": list(t.shape)} for k, t in tensors.items()],
                "shards": shards}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return out


def _collect(samples: Iterable) -> dict[str, torch.Tensor]:
    cols: dict[str, list] = {}
    for s in samples:
        if isinstance(s, Mapping):
            items = s.items()
        elif isinstance(s, (tuple, list)):
            items = ((("x", "y")[i] if len(s) == 2 else f"t{i}", v) for i, v in enumerate(s))
        else:
            items = (("x", s),)
        for k, v in items:
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v)
            cols.setdefault(k, []).append(v)
    return {k: torch.stack(v) for k, v in cols.items()}


def build(source: str, out: str, shard_bytes: int = 256 << 20) -> Path:
    """Run `module:function` locally and write its result as a TensorPack.

    The function may return a dict of full tensors, or an iterable of
    samples (dicts, (x, y) pairs, or tensors), e.g. a torchvision dataset.
    """
    mod, _, fn = source.partition(":")
    if not fn:
        raise ValueError("--from takes module:function")
    import sys

    if "" not in sys.path:
        sys.path.insert(0, "")
    obj = getattr(importlib.import_module(mod), fn)()
    tensors = dict(obj) if isinstance(obj, Mapping) else _collect(obj)
    return write_pack(tensors, out, shard_bytes)


def read_manifest(pack: str | os.PathLike) -> tuple[bytes, dict]:
    p = Path(pack) / "manifest.json"
    raw = p.read_bytes()
    return raw, json.loads(raw)
