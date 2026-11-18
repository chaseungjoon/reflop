# SPDX-License-Identifier: Apache-2.0
"""Named snapshots, automatic checkpoints and recovery (ADR-028, ADR-029).

Snapshots and checkpoints live on the Owner's machine, in this identity's
directory. Only tensors are stored; the files are plain safetensors.

- reflop.save / reflop.load: named snapshots you keep and load again, also
  from a new process.
- Checkpoints: with reflop.device(checkpoint=True), the server saves the
  whole session (every server tensor and the RNG state) at a step boundary,
  at most once per interval. After the worker is lost (a crash, a restart,
  an Owner pause that could not suspend it), reflop.recover() starts a new
  session from the last checkpoint, in the same process.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import timedelta

import torch

from ._pb import compute_pb2 as cpb
from .errors import ReflopError, ReflopRemoteError, SessionLostError, sanitize

MAX_TAG = 4096
# After a skipped or failed checkpoint, try again this soon (never sooner
# than the server allows: it counts from the last saved checkpoint).
RETRY_S = 120.0
_RESERVED = "__reflop__/"


def _yellow(s: str) -> str:
    if os.environ.get("NO_COLOR") or not sys.stderr.isatty():
        return s
    return f"\x1b[33m{s}\x1b[0m"


def _seconds(v) -> float:
    if isinstance(v, timedelta):
        return v.total_seconds()
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    raise TypeError("checkpoint_interval is a number of seconds or a datetime.timedelta")


def _minutes(s: float) -> str:
    return f"{s:g} s" if s < 60 else f"{s / 60:g} min"


class AutoCheckpoint:
    """Automatic checkpoints of one session (reflop.device(checkpoint=True))."""

    def __init__(self, sess, interval_s: float) -> None:
        self.sess = sess
        self.interval = interval_s
        self.start = time.monotonic()
        self.last_ok: float | None = None  # monotonic time of the last saved checkpoint
        self.last_try = 0.0
        self.inflight = None
        self.saved = 0  # checkpoints saved by this session (and the one it recovered from)
        self.generation = 0
        self.tag = b""
        self.disabled = False  # the Owner turned checkpoints off meanwhile
        self.warned: set[str] = set()
        self.lock = threading.Lock()

    def due(self, now: float) -> bool:
        if self.disabled or (self.inflight is not None and not self.inflight.done()):
            return False
        base = self.last_ok if self.last_ok is not None else self.start
        if now - base < self.interval:
            return False
        # A try since then was skipped or failed: wait a little before the next.
        return not (self.last_try > base and now - self.last_try < min(RETRY_S, self.interval))

    def maybe(self) -> None:
        now = time.monotonic()
        if not self.due(now):
            return
        sess = self.sess
        sess.flush()  # the cut must see every operation recorded so far
        sess._send_frees()
        self.last_try = now
        try:
            fut = sess.conn.request(cpb.Envelope(checkpoint=cpb.Checkpoint(tag=self.tag)))
        except SessionLostError:
            return  # the program finds out on its next request
        self.inflight = fut
        fut.add_done_callback(self._done)

    def _warn(self, key: str, msg: str) -> None:
        if key not in self.warned:
            self.warned.add(key)
            sys.stderr.write(f"reflop: {msg}\n")

    def _done(self, fut) -> None:
        e = fut.exception()
        if e is None:
            r = fut.result().checkpoint_saved
            if r.skipped:
                self._warn("skip:" + r.reason[:40], f"checkpoint skipped: {sanitize(r.reason, 300)} "
                           f"(trying again in {_minutes(min(RETRY_S, self.interval))})")
                return
            with self.lock:
                self.last_ok = time.monotonic()
                self.saved += 1
                self.generation = r.generation
            return
        if isinstance(e, SessionLostError):
            return
        if isinstance(e, ReflopRemoteError) and e.code == cpb.ERROR_PERMISSION_DENIED:
            self.disabled = True
            self._warn("denied", f"checkpoints turned off: {e.message}")
            return
        if isinstance(e, ReflopRemoteError) and e.code == cpb.ERROR_RESOURCE_LIMIT and "allows one checkpoint" in e.message:
            with self.lock:
                self.last_ok = time.monotonic()  # the server's clock says wait
            return
        self._warn("err:" + type(e).__name__, f"checkpoint failed: {sanitize(str(e), 300)}")


def enable(sess, interval=None) -> None:
    """Turn automatic checkpoints on for sess (reflop.device(checkpoint=True))."""
    caps = sess.caps
    if "checkpoints" not in caps.capabilities:
        ver = sanitize(caps.runtime_info.reflopd_version, 40) or "unknown"
        raise ReflopError(f"this server (reflopd {ver}) does not support checkpoints; it needs reflopd 0.2.0 or newer")
    owner = int(caps.checkpoint_interval_s)
    if owner <= 0:
        raise ReflopError("the Owner has not enabled checkpoints on this server (`reflopd checkpoints on`)")
    iv = float(owner) if interval is None else _seconds(interval)
    if iv < owner:
        raise ReflopError(f"checkpoint_interval is {_minutes(iv)}, but the Owner allows one checkpoint every "
                          f"{_minutes(owner)} at most: use {_minutes(owner)} or more (or leave it unset)")
    if sess.checkpoints is not None:
        sess.checkpoints.interval = iv
        sess.checkpoints.disabled = False
        return
    sess.checkpoints = AutoCheckpoint(sess, iv)
    sys.stderr.write(_yellow(
        f"reflop: checkpoints on (one every {_minutes(iv)}). While a checkpoint is copied, the GPU memory used by\n"
        "  your session's state (parameters, optimizer state, buffers) doubles. Leave that much GPU memory free,\n"
        "  or the checkpoint is skipped.") + "\n")


def set_tag(sess, obj) -> None:
    """Store a small JSON value (for example the step number) with the next checkpoints."""
    tag = json.dumps(obj, separators=(",", ":")).encode()
    if len(tag) > MAX_TAG:
        raise ValueError(f"the checkpoint tag is {len(tag)} bytes as JSON; the limit is {MAX_TAG}")
    if sess.checkpoints is None:
        raise ReflopError("checkpoints are off: pass checkpoint=True to reflop.device()")
    sess.checkpoints.tag = tag


def recover(sess):
    """Start a new server session from the last checkpoint of this one.

    Every tensor that existed at the checkpoint gets its value from then
    back; tensors created after it raise SessionLostError when used. Returns
    the tag set with reflop.checkpoint_tag() when the checkpoint was taken
    (None without one)."""
    conn = sess.conn
    if not hasattr(conn, "new_session"):
        raise ReflopError("this connection cannot open a new session")
    with sess.lock:
        old = conn.session_id
        conn.new_session()
        try:
            r = conn.call(cpb.Envelope(recover_session=cpb.RecoverSession(from_session_id=old)), timeout=None)
        except ReflopRemoteError as e:
            raise ReflopError(f"cannot recover: {e.message}") from None
        rec = r.session_recovered
        _adopt(sess, rec)
    tag = bytes(rec.tag)
    if not tag:
        return None
    try:
        return json.loads(tag)
    except ValueError:
        return tag


def _adopt(sess, rec) -> None:
    """Bring the client's view of the session in line with the restored one."""
    from .trace import HANDLE, Trace, Value

    restored = set(rec.handles)
    sess.trace = Trace()  # recorded after the checkpoint: the caller redoes that work
    sess.dirty.clear()
    sess.registered.clear()  # the new worker has no graphs; they are registered again on use
    sess.inflight.clear()
    sess.async_error = None
    sess.epoch += 1
    sess.pending_free = [h for h in sess.pending_free if h in restored]
    alive = set()
    for h, st in list(sess.storages.items()):
        if h in restored and st.handle == h:
            st.set(Value(HANDLE, st.meta, handle=h))
            alive.add(h)
        else:
            st.lost = ("this tensor was created after the last checkpoint, so reflop.recover() could not bring it "
                       "back: create it again")
            st._handle = None
    gone = [h for h in restored if h not in alive and h not in sess.pending_free]
    if gone:
        sess.pending_free.extend(gone)  # in the checkpoint, no longer used here
    ck = sess.checkpoints
    if ck is not None:
        with ck.lock:
            ck.last_ok = time.monotonic()
            ck.generation = rec.generation
            ck.inflight = None


# ---------------------------------------------------------------- named snapshots

def _need_snapshots(sess) -> None:
    if "snapshots" not in sess.caps.capabilities:
        ver = sanitize(sess.caps.runtime_info.reflopd_version, 40) or "unknown"
        raise ReflopError(f"this server (reflopd {ver}) does not support reflop.save/load; it needs reflopd 0.2.0 or newer")


def _flatten(obj, prefix: str, out: dict) -> None:
    if isinstance(obj, torch.Tensor):
        if not prefix:
            raise TypeError("reflop.save takes a dict of tensors (for example model.state_dict())")
        out[prefix] = obj
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, (str, int)):
                raise TypeError(f"reflop.save: key {k!r} is not a string")
            _flatten(v, f"{prefix}.{k}" if prefix else str(k), out)
        return
    raise TypeError(f"reflop.save stores tensors only; {prefix or 'the value'} is a {type(obj).__name__}. "
                    "Save other values (step counts, hyperparameters) on your own machine.")


def save(name: str, obj) -> int:
    """Save a dict of reflop tensors (nested dicts allowed, keys joined with
    '.') as the named snapshot `name` on the server. Returns its size."""
    from .tensor import RemoteTensor

    flat: dict = {}
    _flatten(obj, "", flat)
    if not flat:
        raise ValueError("nothing to save")
    for k in flat:
        if k == "__metadata__" or k.startswith(_RESERVED) or len(k.encode()) > 256:
            raise ValueError(f"reflop.save: {k!r} cannot be used as a name")
    sess = None
    bases: dict = {}
    for k, t in flat.items():
        if not isinstance(t, RemoteTensor):
            raise TypeError(f"reflop.save: {k} is not on the reflop device")
        s = t._rf_storage.session
        if sess is None:
            sess = s
        elif s is not sess:
            raise ValueError("reflop.save: all tensors must be on one reflop device")
        st = t._rf_storage
        whole = (not t._rf_chain and tuple(st.meta.shape) == tuple(t.shape) and st.meta.dtype == t.dtype)
        bases[k] = t if whole else t.detach().clone()  # a view is saved as its own tensor
    _need_snapshots(sess)
    from .tensor import value_of

    with sess.lock:
        for t in bases.values():
            value_of(t)  # a fresh (never written) tensor becomes a value to store
            sess.mark_dirty(t._rf_storage)
        sess.flush()
        keys, handles = [], []
        for k, t in bases.items():
            h = t._rf_storage.handle
            if h is None:
                raise ReflopError(f"reflop.save: {k} has no server value")
            keys.append(k)
            handles.append(h)
        r = sess.conn.call(cpb.Envelope(snapshot=cpb.Snapshot(name=name, handles=handles, keys=keys)), timeout=None)
    return int(r.snapshot_saved.bytes)


def load(name: str, sess) -> dict:
    """Load the named snapshot into new reflop tensors: {key: tensor}."""
    from .tensor import RemoteTensor, Storage
    from .trace import HANDLE, PB_TO_DTYPE, Value

    _need_snapshots(sess)
    desc = sess.conn.call(cpb.Envelope(describe_snapshot=cpb.DescribeSnapshot(name=name)), timeout=60).snapshot_description
    keys = list(desc.keys)
    handles = [sess.new_handle() for _ in keys]
    with sess.lock:
        sess.flush()
        sess.conn.call(cpb.Envelope(restore=cpb.Restore(name=name, keys=keys, handles=handles)), timeout=None)
    out = {}
    for k, h, m in zip(keys, handles, desc.metas):
        meta = torch.empty(tuple(m.shape), dtype=PB_TO_DTYPE[m.dtype], device="meta")
        st = Storage(sess, meta, Value(HANDLE, meta, handle=h))
        out[k] = RemoteTensor(meta, st)
    return out


def snapshots(sess) -> list[dict]:
    from datetime import datetime, timezone

    r = sess.conn.call(cpb.Envelope(list_snapshots=cpb.ListSnapshots()), timeout=60).snapshot_list
    return [{"name": sanitize(s.name, 64), "bytes": s.bytes,
             "created": datetime.fromtimestamp(s.created_unix, timezone.utc)} for s in r.snapshots]


def delete(name: str, sess) -> None:
    sess.conn.call(cpb.Envelope(delete_snapshot=cpb.DeleteSnapshot(name=name)), timeout=60)

