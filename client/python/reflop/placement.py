# SPDX-License-Identifier: Apache-2.0
"""Placement policy (PLAN.md §13.3). Granularity is a tensor or a trace,
never a single op."""

from __future__ import annotations

import contextlib
import threading

_state = threading.local()
MODES = ("auto", "remote", "local", "remote-cpu")


def mode() -> str:
    return getattr(_state, "mode", "auto")


@contextlib.contextmanager
def override(m: str):
    if m not in MODES:
        raise ValueError(f"placement must be one of {MODES}")
    prev = mode()
    _state.mode = m
    try:
        yield
    finally:
        _state.mode = prev
