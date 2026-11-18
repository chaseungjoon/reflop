# SPDX-License-Identifier: Apache-2.0
"""Strict safetensors reader and writer (PLAN.md §10.2).

safetensors is a data-only format, but the parser is ours and strict:
header ≤ 1 MiB, JSON object of known shape, whitelisted dtypes, in-bounds,
non-overlapping and size-consistent data_offsets, no trailing garbage.

The writer and the readers agree on every rule: the writer refuses what a
reader would reject, so a file this module wrote always reads back.
Keys that start with RESERVED are written only by checkpoints (ADR-029).
"""

from __future__ import annotations

import json
import math
import struct

import torch

MAX_HEADER = 1 << 20
DTYPES = {
    "BOOL": torch.bool, "U8": torch.uint8, "I8": torch.int8, "I16": torch.int16, "I32": torch.int32,
    "I64": torch.int64, "F16": torch.float16, "BF16": torch.bfloat16, "F32": torch.float32,
    "F64": torch.float64, "C64": torch.complex64, "C128": torch.complex128,
    "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2,
}
NAMES = {v: k for k, v in DTYPES.items()}


RESERVED = "__reflop__/"
MAX_KEY = 256


class SafetensorsError(ValueError):
    pass


def check_key(k: str, reserved_ok: bool = False) -> None:
    """A tensor name a snapshot may use (the same rule for writer and reader)."""
    if not isinstance(k, str) or not k or len(k.encode()) > MAX_KEY:
        raise SafetensorsError("tensor names must be 1 to 256 bytes")
    if k == "__metadata__":
        raise SafetensorsError("'__metadata__' is reserved by the safetensors format")
    if k.startswith(RESERVED) and not reserved_ok:
        raise SafetensorsError(f"names starting with {RESERVED!r} are reserved")


def _reject_dupes(pairs):
    d = {}
    for k, v in pairs:
        if k in d:
            raise SafetensorsError(f"duplicate key {k!r}")
        d[k] = v
    return d


def parse_header(buf, total_len: int, max_numel: int = 1 << 40) -> tuple[int, dict]:
    """Return (data_start, {name: (dtype, shape, begin, end)})."""
    if total_len < 8:
        raise SafetensorsError("file too short")
    (n,) = struct.unpack("<Q", bytes(buf[:8]))
    if n == 0 or n > MAX_HEADER or 8 + n > total_len:
        raise SafetensorsError("bad header length")
    try:
        hdr = json.loads(bytes(buf[8:8 + n]).decode("utf-8"), object_pairs_hook=_reject_dupes)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise SafetensorsError(f"header is not valid JSON: {e}") from None
    if not isinstance(hdr, dict):
        raise SafetensorsError("header must be a JSON object")
    data_start = 8 + n
    data_len = total_len - data_start
    out = {}
    spans = []
    for name, info in hdr.items():
        if name == "__metadata__":
            # Ignored (§10.2), but only in its standard form.
            if not isinstance(info, dict) or not all(isinstance(k, str) and isinstance(v, str)
                                                     for k, v in info.items()):
                raise SafetensorsError("__metadata__ must map strings to strings")
            continue
        if len(name.encode()) > MAX_KEY:
            raise SafetensorsError("tensor name too long")
        if not isinstance(info, dict) or set(info) != {"dtype", "shape", "data_offsets"}:
            raise SafetensorsError(f"tensor {name!r}: unexpected fields")
        dt = DTYPES.get(info["dtype"])
        if dt is None:
            raise SafetensorsError(f"tensor {name!r}: dtype {info['dtype']!r} not allowed")
        shape = info["shape"]
        offs = info["data_offsets"]
        if (not isinstance(shape, list) or not all(isinstance(d, int) and not isinstance(d, bool) and d >= 0
                                                  for d in shape) or len(shape) > 16):
            raise SafetensorsError(f"tensor {name!r}: bad shape")
        if (not isinstance(offs, list) or len(offs) != 2 or not all(isinstance(o, int) and not isinstance(o, bool)
                                                                    for o in offs)):
            raise SafetensorsError(f"tensor {name!r}: bad data_offsets")
        b, e = offs
        numel = math.prod(shape) if shape else 1
        if numel > max_numel:
            raise SafetensorsError(f"tensor {name!r}: too many elements")
        size = numel * torch.empty((), dtype=dt).element_size()
        if not 0 <= b <= e <= data_len or e - b != size:
            raise SafetensorsError(f"tensor {name!r}: data_offsets out of bounds or inconsistent with shape")
        spans.append((b, e, name))
        out[name] = (dt, tuple(shape), b, e)
    spans.sort()
    end = 0
    for b, e, name in spans:
        if b < end:
            raise SafetensorsError(f"tensor {name!r} overlaps another tensor")
        end = e
    if end != data_len:
        raise SafetensorsError("trailing or unreferenced bytes after the tensor data")
    return data_start, out


def tensor_bytes(t: torch.Tensor) -> bytes:
    """Raw little-endian bytes of a tensor without numpy (not in the runtime)."""
    t = t.detach().contiguous().cpu()
    n = t.numel() * t.element_size()
    if n == 0:
        return b""
    buf = bytearray(n)
    torch.frombuffer(buf, dtype=torch.uint8).copy_(t.reshape(-1).view(torch.uint8))
    return bytes(buf)


def header(items) -> tuple[bytes, int]:
    """safetensors header for items = [(name, dtype, shape)] in file order.

    Returns (header bytes including the 8-byte length, data length)."""
    hdr = {}
    off = 0
    for name, dt, shape in items:
        if dt not in NAMES:
            raise SafetensorsError(f"dtype {dt} cannot be stored")
        n = (math.prod(shape) if shape else 1) * torch.empty((), dtype=dt).element_size()
        hdr[name] = {"dtype": NAMES[dt], "shape": list(shape), "data_offsets": [off, off + n]}
        off += n
    h = json.dumps(hdr, separators=(",", ":")).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    if len(h) > MAX_HEADER:
        raise SafetensorsError(f"the header would be {len(h)} bytes; the limit is {MAX_HEADER} "
                               f"(about {MAX_HEADER // 100} tensors)")
    return struct.pack("<Q", len(h)) + h, off


def write(f, tensors: dict[str, torch.Tensor]) -> int:
    """Write tensors in safetensors format to a binary file (tests and tools).

    The worker streams snapshots with checkpoint.Writer instead."""
    names = sorted(tensors)
    for k in names:
        check_key(k)
    ts = [tensors[k].detach().contiguous().cpu() for k in names]
    hb, total = header([(k, t.dtype, tuple(t.shape)) for k, t in zip(names, ts)])
    f.write(hb)
    for t in ts:
        f.write(tensor_bytes(t))
    return len(hb) + total
