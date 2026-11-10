# SPDX-License-Identifier: Apache-2.0
"""Data-category corpus: the worker's own safetensors parser."""

import json
import struct

import pytest
from reflop_worker.safetensors import SafetensorsError, parse_header


def st(hdr, data=b""):
    h = json.dumps(hdr).encode() if not isinstance(hdr, bytes) else hdr
    return struct.pack("<Q", len(h)) + h + data


CASES = {
    "header bomb": struct.pack("<Q", 1 << 40) + b"{}",
    "not json": st(b"\xff\xfe"),
    "past end": st({"x": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}}, b"\0" * 8),
    "size mismatch": st({"x": {"dtype": "F32", "shape": [3], "data_offsets": [0, 8]}}, b"\0" * 8),
    "overlap": st({"a": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]},
                   "b": {"dtype": "U8", "shape": [4], "data_offsets": [2, 6]}}, b"\0" * 6),
    "trailing": st({"x": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]}}, b"\0" * 5),
    "object dtype": st({"x": {"dtype": "PY_OBJECT", "shape": [1], "data_offsets": [0, 1]}}, b"\0"),
    "bool as int": st({"x": {"dtype": "U8", "shape": [True], "data_offsets": [0, 1]}}, b"\0"),
    "duplicate": b"",
    "extra field": st({"x": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1], "exec": "rm -rf /"}}, b"\0"),
    "negative": st({"x": {"dtype": "U8", "shape": [1], "data_offsets": [-1, 0]}}, b"\0"),
}


@pytest.mark.parametrize("name", [k for k in CASES if k != "duplicate"])
def test_rejected(name):
    b = CASES[name]
    with pytest.raises(SafetensorsError):
        parse_header(memoryview(b), len(b))


def test_duplicate_keys():
    h = b'{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}'
    b = struct.pack("<Q", len(h)) + h + b"\0"
    with pytest.raises(SafetensorsError):
        parse_header(memoryview(b), len(b))


def test_valid():
    b = st({"x": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}, "__metadata__": {"k": "v"}}, b"\0" * 8)
    start, hdr = parse_header(memoryview(b), len(b))
    assert set(hdr) == {"x"}
