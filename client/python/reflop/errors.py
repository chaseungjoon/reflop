# SPDX-License-Identifier: Apache-2.0
"""Client-side errors. Every server-provided string is sanitized before it
can reach a terminal (invariant I9)."""

from __future__ import annotations

import re

# C0 controls except \t and \n, DEL, C1 controls, and complete ANSI/OSC
# escape sequences.
_ESC = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)?|\x1b[@-Z\\-_]")
_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f\x80-\x9f]")


def sanitize(s: str, limit: int = 4000) -> str:
    """Strip terminal control sequences from untrusted text."""
    if not isinstance(s, str):
        s = str(s)
    s = _ESC.sub("", s)
    s = _CTRL.sub("", s)
    s = s.replace("‮", "").replace("‭", "").replace("⁦", "").replace("⁧", "") \
        .replace("⁨", "").replace("⁩", "")
    return s[:limit]


class ReflopError(RuntimeError):
    """Base class for reflop client errors."""


class ReflopRemoteError(ReflopError):
    """The server rejected or failed a request."""

    def __init__(self, code: int, message: str, stage: int = 0, node: int | None = None,
                 op_name: str | None = None) -> None:
        from ._pb import compute_pb2 as cpb

        self.code = code
        self.message = sanitize(message)
        self.stage = stage
        self.node = node
        self.op_name = op_name
        try:
            cname = cpb.ErrorCode.Name(code).removeprefix("ERROR_")
        except ValueError:
            cname = str(code)
        try:
            sname = cpb.ValidationStage.Name(stage).removeprefix("STAGE_") if stage else ""
        except ValueError:
            sname = ""
        where = f" at node {node} ({op_name})" if node is not None and op_name else (
            f" at node {node}" if node is not None else "")
        super().__init__(f"[{cname}{'/' + sname if sname else ''}]{where}: {self.message}")

    @classmethod
    def from_pb(cls, e) -> ReflopRemoteError:
        from . import _opset_generated as opset

        op_name = None
        if e.has_node:
            for name, entry in opset.OPS.items():
                if entry[0] == e.op_id:
                    op_name = name
                    break
        return cls(e.code, e.message, e.stage, e.node_index if e.has_node else None, op_name)


class OutOfMemoryError(ReflopRemoteError):
    pass


class SessionLostError(ReflopRemoteError):
    pass


class ServerKeyMismatch(ReflopError):
    """The server presented a different key than the pinned one."""
