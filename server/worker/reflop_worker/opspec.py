# SPDX-License-Identifier: Apache-2.0
"""Operator table entry types (used by the generated ops_generated.py)."""

from __future__ import annotations

from typing import Any, NamedTuple


class A(NamedTuple):
    """One schema argument."""

    name: str
    type: str          # native schema type, e.g. "Tensor?", "SymInt[]", "ScalarType?"
    kwarg_only: bool
    has_default: bool
    default: Any


class OpSpec(NamedTuple):
    id: int
    name: str
    fn: Any            # torch OpOverload, ("native", name), or None if unavailable
    args: tuple
    returns: tuple
    flags: int
    tier: int
    validator: str | None
    cost: str
    capability: str | None
    enums: dict
    devices: tuple
