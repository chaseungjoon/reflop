# SPDX-License-Identifier: Apache-2.0
"""Framing on the inherited control socket: uint32 BE length + protobuf."""

from __future__ import annotations

import socket
import struct

from google.protobuf.message import Message

from ._pb import worker_pb2


def _repeated(fd) -> bool:
    """FieldDescriptor.label is gone in protobuf 7; is_repeated is newer."""
    if hasattr(fd, "is_repeated"):
        return bool(fd.is_repeated)
    return fd.label == fd.LABEL_REPEATED

_HDR = struct.Struct(">I")


class ProtocolError(Exception):
    pass


def check_no_unknown(msg: Message) -> None:
    """Reject unknown fields anywhere in the tree (invariant I11)."""
    from google.protobuf import unknown_fields

    if len(unknown_fields.UnknownFieldSet(msg)) > 0:
        raise ProtocolError(f"unknown fields in {msg.DESCRIPTOR.full_name}")
    for fd, value in msg.ListFields():
        if fd.message_type is None:
            continue
        if _repeated(fd):
            if fd.message_type.GetOptions().map_entry:
                continue
            for v in value:
                check_no_unknown(v)
        else:
            check_no_unknown(value)


def check_enums(msg: Message) -> None:
    """Reject enum values not declared in their enum (open proto3 enums)."""
    for fd, value in msg.ListFields():
        if fd.enum_type is not None:
            vals = value if _repeated(fd) else [value]
            for v in vals:
                if v not in fd.enum_type.values_by_number:
                    raise ProtocolError(f"unknown enum value {v} for {fd.full_name}")
        elif fd.message_type is not None:
            if _repeated(fd):
                if fd.message_type.GetOptions().map_entry:
                    continue
                for v in value:
                    check_enums(v)
            else:
                check_enums(value)


class Channel:
    """Blocking framed channel. Reads are single-threaded; writes are
    serialized by the caller (the worker is single-threaded)."""

    def __init__(self, sock: socket.socket, max_frame: int) -> None:
        self.sock = sock
        self.max_frame = max_frame
        self._buf = bytearray()

    def _read_exact(self, n: int) -> bytes:
        while len(self._buf) < n:
            chunk = self.sock.recv(max(1 << 20, n - len(self._buf)))
            if not chunk:
                raise EOFError("control socket closed")
            self._buf += chunk
        out = bytes(self._buf[:n])
        del self._buf[:n]
        return out

    def recv(self) -> worker_pb2.WorkerEnvelope:
        (n,) = _HDR.unpack(self._read_exact(4))
        if n > self.max_frame:
            raise ProtocolError(f"frame of {n} bytes exceeds limit {self.max_frame}")
        msg = worker_pb2.WorkerEnvelope()
        msg.ParseFromString(self._read_exact(n))
        check_no_unknown(msg)
        check_enums(msg)
        return msg

    def send(self, msg: Message) -> None:
        data = msg.SerializeToString(deterministic=True)
        if len(data) > self.max_frame:
            raise ProtocolError(f"outgoing frame of {len(data)} bytes exceeds limit")
        self.sock.sendall(_HDR.pack(len(data)) + data)
