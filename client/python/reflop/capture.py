# SPDX-License-Identifier: Apache-2.0
"""reflop.capture: static step capture and replay.

Eager mode runs your Python every step and records every op, which costs
tens of microseconds per op on the client. For steady-state training steps
whose graph does not change, `reflop.capture` works like CUDA graphs:

    step = reflop.capture(train_step)     # train_step(x, y) -> loss
    for x, y in data:
        loss = step(x, y)                  # after 2 eager calls: one Execute

The first two calls run eagerly. If both produced the same graph with the
same baked-in values, later calls skip Python entirely and send a single
Execute that binds the new arguments. Requirements (checked, not assumed):

- tensor arguments keep their shapes and dtypes (else it re-captures);
- values that change every step must live in tensors, not Python numbers:
  use `capturable=True` (or `fused=True`) optimizers and tensor learning
  rates; a step whose Python scalars drift is refused with an explanation;
- the returned tensors are the same objects every call (their contents
  change), as with CUDA graphs.
"""

from __future__ import annotations

import functools

import torch

from . import session as _session
from ._pb import compute_pb2 as cpb
from .trace import CPU, DATASET, HANDLE, SCALAR


class CaptureError(RuntimeError):
    pass


def _arg_tensors(args):
    return [a for a in args if isinstance(a, torch.Tensor)]


def _state(t):
    return (t._version, tuple(t.shape), t.stride(), t.dtype)


def _same_cpu_tensor(t, state, ts, cpu_args, before):
    """Index of the CPU argument that `t` was, unchanged, when the step
    uploaded it (`state`, against the arguments' states at the call)."""
    for i in cpu_args:
        a = ts[i]
        if (a is t or a.data_ptr() == t.data_ptr()) and before[i] == state:
            return i
    return None


def _sig(args):
    return tuple((tuple(a.shape), a.dtype, type(a).__name__) if isinstance(a, torch.Tensor) else ("py", repr(a))
                 for a in args)


class Captured:
    def __init__(self, fn, sess=None) -> None:
        self.fn = fn
        self.sess = sess
        self.prev = None
        self.plan = None
        self.sig = None
        self.out = None
        self.replays = 0
        functools.update_wrapper(self, fn)

    def _candidates(self, args) -> list:
        """Sessions the step may use: the arguments' session if they pin one,
        else every open session (the step's own work decides)."""
        from .tensor import RemoteTensor

        if self.sess is not None:
            return [self.sess]
        ss = {id(a._rf_storage.session): a._rf_storage.session
              for a in _arg_tensors(args) if isinstance(a, RemoteTensor)}
        if len(ss) == 1:
            return list(ss.values())
        return list(_session._sessions.values()) or [_session.current()]

    def _arg_handles(self, args) -> list:
        from .tensor import RemoteTensor, value_of

        hs = []
        for a in _arg_tensors(args):
            if isinstance(a, RemoteTensor):
                v = value_of(a)
                if v.kind != HANDLE:
                    a._rf_storage.session.flush()
                    v = value_of(a)
                hs.append(("handle", v.handle))
            else:
                hs.append(("cpu", None))
        return hs

    def _record(self, args):
        cands = self._candidates(args)
        for c in cands:
            c.flush()
        arg_h = self._arg_handles(args)
        ts = _arg_tensors(args)
        cpu_args = [i for i, (k, _) in enumerate(arg_h) if k == "cpu"]
        before = {i: _state(ts[i]) for i in cpu_args}
        logs = {}
        for c in cands:
            c.upload_log = []
        try:
            out = self.fn(*args)
        finally:
            for c in cands:
                logs[id(c)], c.upload_log = c.upload_log, None
        used = [c for c in cands if len(c.trace) or c.dirty or logs[id(c)]]
        if len(used) != 1:
            # Nothing recorded, or work in several sessions: stays eager.
            for c in used:
                c.flush()
            return out, None
        sess = used[0]
        uploads = {h: (t, st, dt) for h, t, st, dt in logs[id(sess)]}
        plan: dict = {}
        sess.flush(plan=plan)
        built = plan.get("built")
        if built is None:
            return out, None
        sources = []
        for v in built.input_values:
            src = None
            if v.kind == HANDLE:
                for i, (k, h) in enumerate(arg_h):
                    if k == "handle" and h == v.handle:
                        src = ("arg", i)
                        break
                if src is None and v.handle in uploads:
                    # x.to(dev) inside the step: rebind the argument each replay.
                    t, st, dt = uploads[v.handle]
                    i = _same_cpu_tensor(t, st, ts, cpu_args, before)
                    if i is not None:
                        src = ("upload", i, dt)
                    elif t._version == st[0]:
                        src = ("const", sess.cpu_value(t.to(dt)))
                    else:
                        src = ("changed",)  # modified after the upload: never replayed
                if src is None:
                    if v.handle in plan["freed"]:
                        raise CaptureError("the step reads a server tensor that it also frees (a temporary made "
                                           "before the step and dropped inside it); run a few eager steps first "
                                           "(e.g. zero_grad(set_to_none=False))")
                    src = ("handle", v.handle)
            elif v.kind == CPU:
                for i in cpu_args:
                    if v.payload.shape == ts[i].shape and torch.equal(v.payload, ts[i].detach().cpu()):
                        src = ("arg", i)
                        break
                if src is None:
                    src = ("const", v)
            elif v.kind == SCALAR:
                src = ("scalar", v.payload)
            elif v.kind == DATASET:
                src = ("dataset", v)
            sources.append(src)
        specs = []
        for sp in plan["specs"]:
            if sp.kind == cpb.OutputSpec.KIND_STORE:
                specs.append(cpb.OutputSpec(kind=cpb.OutputSpec.KIND_REBIND, handle=sp.handle))
            else:
                specs.append(sp)
        return out, {"sess": sess, "hash": built.graph_hash, "sources": sources, "specs": specs,
                     "owners": [st for st, _ in plan["owners"]]}

    @staticmethod
    def _same(a, b) -> bool:
        if (a is None or b is None or a["sess"] is not b["sess"] or a["hash"] != b["hash"]
                or len(a["sources"]) != len(b["sources"])):
            return False
        for x, y in zip(a["sources"], b["sources"]):
            if x[0] != y[0] or x[0] == "changed":
                return False
            if x[0] == "scalar" and x[1] != y[1]:
                return False
            if x[0] == "const" and not torch.equal(x[1].payload, y[1].payload):
                return False
            if x[0] == "arg" and x[1] != y[1]:
                return False
            if x[0] == "upload" and x[1:] != y[1:]:
                return False
        return True

    def __call__(self, *args):
        sig = _sig(args)
        if self.plan is not None and sig == self.sig:
            return self._replay(self.plan["sess"], args)
        out, plan = self._record(args)
        if sig == self.sig and self._same(self.prev, plan):
            self.plan, self.out = plan, out
        elif self.prev is not None and sig == self.sig and plan is not None and self.prev["hash"] == plan["hash"]:
            drift = [i for i, (x, y) in enumerate(zip(self.prev["sources"], plan["sources"]))
                     if x[0] == "scalar" and x[1] != y[1]]
            if drift:
                raise CaptureError(
                    f"reflop.capture: {len(drift)} Python scalar(s) changed between steps (for example a "
                    "non-capturable optimizer's bias correction or a Python learning-rate schedule). Replaying "
                    "would freeze them. Use capturable=True or fused=True optimizers and tensor-valued "
                    "hyperparameters, or call the step without reflop.capture.")
        self.prev, self.sig = plan, sig
        return out

    def _replay(self, sess, args):
        if len(sess.trace):
            sess.flush()
        sess.raise_async()
        arg_h = self._arg_handles(args)
        ts = _arg_tensors(args)
        values = []
        from .trace import Value

        for src in self.plan["sources"]:
            kind = src[0]
            if kind == "arg":
                i = src[1]
                if arg_h[i][0] == "handle":
                    values.append(Value(HANDLE, None, handle=arg_h[i][1]))
                else:
                    values.append(sess.cpu_value(ts[i]))
            elif kind == "upload":
                values.append(sess.cpu_value(ts[src[1]].to(src[2])))
            elif kind == "handle":
                values.append(Value(HANDLE, None, handle=src[1]))
            elif kind == "const":
                values.append(src[1])
            elif kind == "scalar":
                values.append(Value(SCALAR, None, payload=src[1]))
            elif kind == "dataset":
                values.append(src[1])
        ex = cpb.Execute(graph_hash=self.plan["hash"])
        temp = sess.bind_inputs(ex, values)
        ex.outputs.extend(self.plan["specs"])
        fut = sess.conn.request(cpb.Envelope(execute=ex))
        sess._track(fut, "execute")
        sess.stats["executes"] += 1
        sess.pending_free.extend(temp)
        sess._send_frees()
        for st in self.plan["owners"]:
            st.version += 1
            st.view_cache.clear()
        self.replays += 1
        return self.out


def capture(fn=None):
    """Decorator/wrapper: capture a step function for static replay."""
    if fn is None:
        return capture
    return Captured(fn)
