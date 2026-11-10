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


# Cost model (§13.3): t_local = F / P_local; t_remote = F / P_server + bytes /
# bandwidth + RTT. Measured values come from .reflop.toml (`reflop init`);
# these defaults describe a LAN and a mid-range GPU. Hysteresis: local
# execution must win by HYSTERESIS to be chosen.
P_LOCAL = 5e10
P_SERVER = 2e13
BANDWIDTH = 1e8
RTT = 1e-3
HYSTERESIS = 4.0


def load_measurements(path: str = ".reflop.toml") -> None:
    global P_LOCAL, RTT
    try:
        import tomllib

        with open(path, "rb") as f:
            d = tomllib.load(f)
    except (OSError, ValueError, ImportError):
        return
    P_LOCAL = float(d.get("local_flops", P_LOCAL))
    RTT = float(d.get("rtt_ms", RTT * 1000)) / 1000


def prefer_local(flops: float, remote_bytes: int, local_bytes: int) -> bool:
    """For an op mixing client tensors and server tensors: run it locally
    (fetching remote_bytes) instead of remotely (uploading local_bytes)?"""
    t_local = flops / P_LOCAL + remote_bytes / BANDWIDTH + RTT
    t_remote = flops / P_SERVER + local_bytes / BANDWIDTH
    return t_local * HYSTERESIS < t_remote
