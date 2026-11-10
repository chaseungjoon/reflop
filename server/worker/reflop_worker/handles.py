# SPDX-License-Identifier: Apache-2.0
"""Session-scoped tensor handles (invariant I8).

A worker serves exactly one (identity, session), so this table is the
whole namespace an identity can reach: handles of other identities or
sessions do not exist in this process.
"""

from __future__ import annotations

import torch

from ._pb import compute_pb2 as cpb
from .errors import ReflopError

HANDLE_LEN = 16


def check_new(h: bytes) -> None:
    if len(h) != HANDLE_LEN:
        raise ReflopError(cpb.ERROR_BINDING, "handles are 16 bytes", cpb.STAGE_E1_BINDINGS)
    if h == bytes(HANDLE_LEN) or h[0] == 0xFF:
        raise ReflopError(cpb.ERROR_BINDING, "handle is in a reserved range", cpb.STAGE_E1_BINDINGS)


class HandleTable:
    def __init__(self, max_handles: int) -> None:
        self.max = max_handles
        self.t: dict[bytes, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.t)

    def get(self, h: bytes) -> torch.Tensor:
        try:
            return self.t[h]
        except KeyError:
            raise ReflopError(cpb.ERROR_UNKNOWN_HANDLE, f"unknown handle {h.hex()[:8]}…",
                              cpb.STAGE_E1_BINDINGS) from None

    def put_new(self, h: bytes, t: torch.Tensor) -> None:
        check_new(h)
        if h in self.t:
            raise ReflopError(cpb.ERROR_HANDLE_EXISTS, f"handle {h.hex()[:8]}… already exists", cpb.STAGE_E1_BINDINGS)
        if len(self.t) >= self.max:
            raise ReflopError(cpb.ERROR_RESOURCE_LIMIT, f"more than {self.max} live handles", cpb.STAGE_E1_BINDINGS)
        self.t[h] = t

    def replace(self, h: bytes, t: torch.Tensor) -> None:
        if h not in self.t:
            raise ReflopError(cpb.ERROR_UNKNOWN_HANDLE, f"unknown handle {h.hex()[:8]}…", cpb.STAGE_E1_BINDINGS)
        self.t[h] = t

    def free(self, h: bytes) -> bool:
        return self.t.pop(h, None) is not None

    def bytes(self) -> int:
        seen, n = set(), 0
        for t in self.t.values():
            k = t.untyped_storage().data_ptr()
            if k not in seen:
                seen.add(k)
                n += t.untyped_storage().nbytes()
        return n
