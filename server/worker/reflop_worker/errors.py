# SPDX-License-Identifier: Apache-2.0
"""Structured errors. Every failure maps to an ErrorCode and a stage."""

from __future__ import annotations

from ._pb import compute_pb2 as cpb


class ReflopError(Exception):
    def __init__(self, code: int, message: str, stage: int = cpb.STAGE_NONE, node: int | None = None,
                 op_id: int = 0) -> None:
        super().__init__(message)
        self.code = code
        self.stage = stage
        self.node = node
        self.op_id = op_id
        self.message = message

    def to_pb(self) -> cpb.Error:
        e = cpb.Error(code=self.code, stage=self.stage, message=self.message[:2000])
        if self.node is not None:
            e.has_node = True
            e.node_index = self.node
            e.op_id = self.op_id
        return e


class FatalError(Exception):
    """The worker must exit (e.g. CUDA context lost)."""


def invalid(msg: str, stage: int = cpb.STAGE_V2_STRUCTURE, node: int | None = None, op_id: int = 0) -> ReflopError:
    return ReflopError(cpb.ERROR_INVALID_GRAPH, msg, stage, node, op_id)


def limit(msg: str, stage: int = cpb.STAGE_V5_SAFETY, node: int | None = None, op_id: int = 0) -> ReflopError:
    return ReflopError(cpb.ERROR_RESOURCE_LIMIT, msg, stage, node, op_id)
