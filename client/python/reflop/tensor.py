# SPDX-License-Identifier: Apache-2.0
"""RemoteTensor: a lazy, server-resident tensor (PLAN.md §11).

Every aten op on a RemoteTensor is intercepted by __torch_dispatch__ and
recorded in the session's trace instead of being executed. Shapes and dtypes
come from running the op on meta tensors, so shape-dependent Python code
works without round trips.

Functionalization: the wire format has no in-place ops and no aliasing.
Views share a Storage and remember the chain of view ops from their base.
Reading a view replays the chain; writing to a view (any in-place op) is
turned into a functional op plus an inverse "scatter" back into the base.
"""

from __future__ import annotations

import functools
import itertools
import threading
import warnings
from typing import Any

import torch
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

from . import _opset_generated as opset
from .trace import HANDLE, UnsupportedOperatorError, Value, _base, normalize

aten = torch.ops.aten


class Storage:
    """State shared by a base tensor and all its views.

    `value` is the current Value of the base, or None for a freshly
    allocated (logically zero) tensor. `handle` is set once the base lives
    in a server handle.
    """

    __slots__ = ("session", "meta", "value", "version", "handle", "view_cache", "seq", "__weakref__")
    _seq = itertools.count()

    def __init__(self, session, meta: torch.Tensor, value: Value | None) -> None:
        self.session = session
        self.seq = next(Storage._seq)
        self.meta = meta
        self.value = value
        self.version = 0
        self.handle: bytes | None = None
        self.view_cache: dict = {}
        if value is not None and value.kind == HANDLE:
            self.handle = value.handle
        elif value is not None:
            session.mark_dirty(self)

    def set(self, value: Value) -> None:
        self.value = value
        self.version += 1
        self.view_cache.clear()
        if value.kind != HANDLE:
            self.session.mark_dirty(self)

    def __del__(self) -> None:
        h = self.handle
        if h is not None:
            try:
                self.session.release(h)
            except Exception:  # noqa: BLE001 - interpreter shutdown
                pass


class RemoteTensor(torch.Tensor):
    """A tensor on the `reflop` device."""

    _rf_storage: Storage
    _rf_chain: tuple
    _rf_meta: torch.Tensor

    @staticmethod
    def __new__(cls, meta: torch.Tensor, storage: Storage, chain: tuple = (), requires_grad: bool = False):
        from .device import torch_device

        # Every reflop tensor reports reflop:0: PyTorch's Python backends
        # expose a single device to autograd. The tensor's server session
        # (GPU, or the server CPU from reflop.device(1, kind="cpu")) is
        # tracked in its Storage instead.
        r = torch.Tensor._make_wrapper_subclass(
            cls, meta.shape, strides=meta.stride(), storage_offset=meta.storage_offset(), dtype=meta.dtype,
            device=torch_device(0), requires_grad=requires_grad)
        r._rf_meta = meta
        r._rf_storage = storage
        r._rf_chain = chain
        return r

    def __init__(self, *args, **kwargs) -> None:  # noqa: D401 - torch calls __init__ with __new__ args
        pass

    # ---------------------------------------------------------------- dispatch

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        sess = None
        for x in itertools.chain(tree_flatten(args)[0], tree_flatten(kwargs)[0]):
            if isinstance(x, RemoteTensor):
                s = x._rf_storage.session
                if sess is None:
                    sess = s
                elif s is not sess and str(func) != "aten._to_copy.default":
                    raise RuntimeError(
                        f"{func}: these reflop tensors live in different server sessions (device "
                        f"{sess.device_index} and {s.device_index}); move one with .to(reflop.device(i))")
        if sess is None:  # pragma: no cover - torch only calls us with a RemoteTensor
            return func(*args, **kwargs)
        return dispatch(sess, func, args, kwargs)

    # ---------------------------------------------------------------- python API

    def __repr__(self, *, tensor_contents=None) -> str:
        try:
            body = repr(self.cpu())
        except Exception as e:  # noqa: BLE001 - repr must not raise
            body = f"<unavailable: {e}>"
        if body.startswith("tensor("):
            body = body[len("tensor("):-1]
        return f"tensor({body}, device='{self.device}')"

    def numpy(self, *, force: bool = False):
        return self.cpu().numpy()

    def __array__(self, dtype=None, copy=None):
        a = self.cpu().numpy()
        return a.astype(dtype) if dtype is not None else a

    def tolist(self):
        return self.cpu().tolist()

    def __reduce_ex__(self, proto):
        raise TypeError("reflop tensors cannot be pickled; use reflop.fetch(obj, path) to save them locally "
                        "as safetensors")

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        with torch.no_grad():
            c = self.clone()
        if self.requires_grad:
            c.requires_grad_(True)
        memo[id(self)] = c
        return c

    @property
    def reflop_handle(self) -> bytes | None:
        """Server handle of the base tensor, after the last flush."""
        return self._rf_storage.handle


def is_remote(x) -> bool:
    return isinstance(x, RemoteTensor)


# ---------------------------------------------------------------- meta helpers


def _to_meta(x):
    if isinstance(x, RemoteTensor):
        return x._rf_meta
    if isinstance(x, torch.Tensor):
        if x.device.type == "meta":
            return x
        if x.dim() == 0 and x.device.type == "cpu":
            return x
        return x.to("meta")
    if isinstance(x, torch.device) and x.type == "reflop":
        return torch.device("meta")
    return x


def run_meta(func, args, kwargs):
    with torch._C._DisableTorchDispatch():
        return func(*tree_map(_to_meta, args), **tree_map(_to_meta, kwargs))


# Meta inference cache for non-view ops. Output shapes/dtypes depend on input
# metas and on int/bool/dtype arguments, never on float values (ALLOC ops,
# whose sizes can depend on floats, are not cached). Python meta kernels
# dominate client time, so this is the main eager-mode speedup.
_META_CACHE: dict = {}
_META_CACHE_MAX = 1 << 16
_UNCACHED_FLAGS = None


def _sig(x):
    if isinstance(x, RemoteTensor):
        m = x._rf_meta
        return ("T", m.dtype, tuple(m.shape), m.stride(), m.storage_offset())
    if isinstance(x, torch.Tensor):
        return ("C", x.dtype, tuple(x.shape), x.stride(), x.device.type)
    if isinstance(x, float):
        return ("f",)
    if isinstance(x, (list, tuple)):
        return (type(x).__name__, tuple(_sig(y) for y in x))
    if isinstance(x, torch.device):
        return ("d", x.type)
    if isinstance(x, (int, bool, str, type(None), torch.dtype, torch.memory_format, torch.layout, complex)):
        return (type(x).__name__, x)
    raise TypeError  # unknown: do not cache


def meta_cached(func, args, kwargs):
    global _UNCACHED_FLAGS
    if _UNCACHED_FLAGS is None:
        _UNCACHED_FLAGS = opset.FLAG_ALLOC | opset.FLAG_RNG
    entry = opset.OPS.get(str(func))
    if entry is None or entry[1] & _UNCACHED_FLAGS:
        return run_meta(func, args, kwargs)
    try:
        key = (func, _sig(args), _sig(tuple(sorted(kwargs.items()))))
    except TypeError:
        return run_meta(func, args, kwargs)
    hit = _META_CACHE.get(key)
    if hit is not None:
        spec, leaves = hit
        return tree_unflatten([torch.empty_strided(sh, st, dtype=dt, device="meta") if sh is not None else v
                               for sh, st, dt, v in leaves], spec)
    out = run_meta(func, args, kwargs)
    flat, spec = tree_flatten(out)
    leaves = [(tuple(t.shape), t.stride(), t.dtype, None) if isinstance(t, torch.Tensor) else (None, None, None, t)
              for t in flat]
    if len(_META_CACHE) < _META_CACHE_MAX:
        _META_CACHE[key] = (spec, leaves)
    return out


def _flat_tensors(out) -> list:
    leaves, _ = tree_flatten(out)
    return [x for x in leaves if isinstance(x, torch.Tensor)]


# ---------------------------------------------------------------- values


def value_of(t: RemoteTensor) -> Value:
    """The current Value of t, replaying its view chain if needed."""
    st = t._rf_storage
    sess = st.session
    if st.value is None:
        st.set(sess.record_simple("aten.full.default", aten.full.default,
                                  [list(st.meta.shape), 0, st.meta.dtype, None, torch.device("reflop"), None],
                                  st.meta)[0])
    if not t._rf_chain:
        return st.value
    # Cached replays are only valid within one trace: node values from an
    # earlier (flushed) trace no longer exist.
    tag = (st.version, sess.epoch)
    key = t._rf_chain
    hit = st.view_cache.get(key)
    if hit is not None and hit[0] == tag:
        return hit[1]
    v = st.value
    cur_meta = st.meta
    for i, (func, rest, k) in enumerate(t._rf_chain):
        prefix = t._rf_chain[: i + 1]
        cached = st.view_cache.get(prefix)
        if cached is not None and cached[0] == tag:
            v, cur_meta = cached[1], cached[2]
            continue
        a, kw = _rebind(func, [cur_meta, *rest])
        out_meta = run_meta(func, a, kw)
        metas = _flat_tensors(out_meta)
        outs = sess.record_values(str(func), func, [v, *rest], metas, self_is_value=True)
        v = outs[k if k is not None else 0]
        cur_meta = metas[k if k is not None else 0]
        st.view_cache[prefix] = (tag, v, cur_meta)
    return v


# Inverses for writing through a view (functionalization "view inverse").
def _inverse(sess, func, rest, k, base_meta, base_val: Value, new_val: Value, view_meta) -> Value:
    name = str(func)
    rec = sess.record_simple

    def scatter(sname, sfunc, args):
        return rec(sname, sfunc, [base_val, new_val, *args], base_meta)[0]

    if name in ("aten.alias.default", "aten.detach.default", "aten.lift_fresh.default"):
        return new_val
    if name in ("aten.view.default", "aten._unsafe_view.default", "aten.reshape.default",
                "aten.squeeze.dim", "aten.squeeze.dims", "aten.squeeze.default", "aten._reshape_alias.default",
                "aten.view_copy.default", "aten.flatten.using_ints", "aten.unflatten.int"):
        return rec("aten.view.default", aten.view.default, [new_val, list(base_meta.shape)], base_meta)[0] \
            if base_meta.is_contiguous() else \
            rec("aten.reshape.default", aten.reshape.default, [new_val, list(base_meta.shape)], base_meta)[0]
    if name == "aten.unsqueeze.default":
        return rec("aten.squeeze.dim", aten.squeeze.dim, [new_val, rest[0]], base_meta)[0]
    if name in ("aten.t.default",):
        return rec("aten.t.default", aten.t.default, [new_val], base_meta)[0]
    if name == "aten.transpose.int":
        return rec("aten.transpose.int", aten.transpose.int, [new_val, rest[0], rest[1]], base_meta)[0]
    if name == "aten.permute.default":
        dims = [d % base_meta.dim() for d in rest[0]]
        inv = [0] * len(dims)
        for i, d in enumerate(dims):
            inv[d] = i
        return rec("aten.permute.default", aten.permute.default, [new_val, inv], base_meta)[0]
    if name == "aten.slice.Tensor":
        dim, start, end, step = rest
        return scatter("aten.slice_scatter.default", aten.slice_scatter.default, [dim, start, end, step])
    if name == "aten.select.int":
        return scatter("aten.select_scatter.default", aten.select_scatter.default, [rest[0], rest[1]])
    if name == "aten.diagonal.default":
        return scatter("aten.diagonal_scatter.default", aten.diagonal_scatter.default, list(rest))
    if name == "aten.split_with_sizes.default":
        sizes, dim = rest
        start = sum(sizes[:k])
        return scatter("aten.slice_scatter.default", aten.slice_scatter.default, [dim, start, start + sizes[k], 1])
    if name == "aten.split.Tensor":
        size, dim = rest
        start = k * size
        end = min(start + size, base_meta.shape[dim])
        return scatter("aten.slice_scatter.default", aten.slice_scatter.default, [dim, start, end, 1])
    if name == "aten.unbind.int":
        return scatter("aten.select_scatter.default", aten.select_scatter.default, [rest[0], k])
    if name == "aten.view_as_real.default":
        return rec("aten.view_as_complex.default", aten.view_as_complex.default, [new_val], base_meta)[0]
    if name == "aten.view_as_complex.default":
        return rec("aten.view_as_real.default", aten.view_as_real.default, [new_val], base_meta)[0]
    if name == "aten.expand.default":
        raise RuntimeError("unsupported operation: in-place write to an expanded tensor (more than one element "
                           "of the written-to tensor refers to a single memory location)")
    # Generic fallback: describe the view relative to its base's storage.
    if base_meta.is_contiguous() and base_meta.storage_offset() == 0:
        return scatter("aten.as_strided_scatter.default", aten.as_strided_scatter.default,
                       [list(view_meta.shape), list(view_meta.stride()), view_meta.storage_offset()])
    raise UnsupportedOperatorError(f"cannot write through a view created by {name}")


def write(t: RemoteTensor, new_val: Value, new_meta: torch.Tensor | None = None) -> None:
    """Make new_val the content of t (functionalized in-place update)."""
    st = t._rf_storage
    sess = st.session
    tm = t._rf_meta
    if new_meta is not None:
        if new_meta.dtype != tm.dtype:
            new_val = sess.record_simple("aten._to_copy.default", aten._to_copy.default,
                                         [new_val, tm.dtype, None, None, None, False, None], tm)[0]
        if tuple(new_meta.shape) != tuple(tm.shape):
            new_val = sess.record_simple("aten.expand.default", aten.expand.default,
                                         [new_val, list(tm.shape), False], tm)[0]
    if not t._rf_chain:
        st.set(new_val)
        return
    # Values and metas of every intermediate base along the chain.
    if st.value is None:
        value_of(RemoteTensor(st.meta, st))
    vals, metas = [st.value], [st.meta]
    cur, cm = st.value, st.meta
    for func, rest, k in t._rf_chain[:-1]:
        a, kw = _rebind(func, [cm, *rest])
        om = _flat_tensors(run_meta(func, a, kw))
        cur = sess.record_values(str(func), func, [cur, *rest], om, self_is_value=True)[k or 0]
        cm = om[k or 0]
        vals.append(cur)
        metas.append(cm)
    v = new_val
    view_meta = tm
    for (func, rest, k), bv, bm in zip(reversed(t._rf_chain), reversed(vals), reversed(metas)):
        v = _inverse(sess, func, rest, k, bm, bv, v, view_meta)
        view_meta = bm
    st.set(v)


# ---------------------------------------------------------------- functional op lookup


def _arg_sig(a) -> tuple:
    return (a.name, _base(str(a.type)))


@functools.lru_cache(maxsize=4096)
def functional_of(func) -> tuple[Any, list[int]] | None:
    """Find the functional overload of a mutating op. Returns (func, list of
    mutated argument positions)."""
    schema = func._schema
    mutated = [i for i, a in enumerate(schema.arguments) if a.alias_info is not None and a.alias_info.is_write]
    base = schema.name.split("::")[1]
    out_args = [i for i in mutated if schema.arguments[i].kwarg_only]
    if out_args and len(out_args) == len(mutated):
        cand_pkt = getattr(aten, base, None)
        keep = [a for i, a in enumerate(schema.arguments) if i not in out_args]
    else:
        fname = base[:-1] if base.endswith("_") else base
        cand_pkt = getattr(aten, fname, None)
        keep = list(schema.arguments)
    if cand_pkt is None:
        return None
    want = [_arg_sig(a) for a in keep]
    pref = [func._overloadname.replace("_out", "").replace("out", "default") or "default", func._overloadname]
    names = list(cand_pkt.overloads())
    names.sort(key=lambda n: (n not in pref, n))
    loose = None
    for ov in names:
        f = getattr(cand_pkt, ov)
        fs = f._schema
        if any(a.alias_info is not None and a.alias_info.is_write for a in fs.arguments):
            continue
        sig = [_arg_sig(a) for a in fs.arguments]
        if sig == want:
            return f, mutated
        # Factory out= variants (arange.start_out): the functional form adds
        # keyword-only dtype/layout/device/pin_memory, filled from `out`.
        extra = fs.arguments[len(want):]
        if (loose is None and out_args and sig[:len(want)] == want
                and all(a.kwarg_only and a.name in _FACTORY_KW for a in extra)):
            loose = (f, mutated)
    return loose


_FACTORY_KW = {"dtype", "layout", "device", "pin_memory", "memory_format"}


# Mutating ops whose functional form is not found by name.
_SPECIAL_FUNCTIONAL = {
    "aten.normal_.default": ("aten.normal_functional.default", aten.normal_functional.default),
    "aten.bernoulli_.float": ("aten.bernoulli.p", aten.bernoulli.p),
    "aten.bernoulli_.Tensor": ("aten.bernoulli.Tensor", aten.bernoulli.Tensor),
}

_BATCH_NORM = {"aten._native_batch_norm_legit.default", "aten.native_batch_norm.default"}


# ---------------------------------------------------------------- decompositions


@functools.lru_cache(maxsize=1)
def _decomp_table():
    from torch._decomp import core_aten_decompositions, get_decompositions

    table = dict(core_aten_decompositions())
    table.update(get_decompositions([
        aten.nll_loss_forward, aten.masked_fill.Tensor, aten.fill.Tensor, aten.full_like, aten.zeros_like,
        aten.ones_like, aten.rand_like, aten.randn_like, aten.new_zeros, aten.new_ones, aten.new_full,
        aten.new_empty, aten.lift_fresh_copy, aten.addcmul, aten.addcdiv, aten.logsumexp, aten.norm,
    ]))
    return table


_warned_local: set[str] = set()


# ---------------------------------------------------------------- dispatch


def dispatch(sess, func, args, kwargs):
    if sess.depth == 0:
        sess.maybe_flush()
    sess.depth += 1
    prev = getattr(_active, "sess", None)
    _active.sess = sess
    try:
        return _dispatch(sess, func, args, kwargs)
    finally:
        sess.depth -= 1
        _active.sess = prev


def _placement_local(sess, func, args, kwargs):
    """placement("local"), or "auto" when the cost model prefers it: run the
    op on the client and return client tensors (granularity: this tensor)."""
    from . import _placement as placement

    mode = placement.mode()
    if mode == "local":
        return True
    if mode != "auto":
        return False
    leaves = tree_flatten((args, kwargs))[0]
    remote = [x for x in leaves if isinstance(x, RemoteTensor)]
    local = [x for x in leaves if isinstance(x, torch.Tensor) and not isinstance(x, RemoteTensor) and x.dim() > 0]
    if not local or not remote:
        return False
    rb = sum(x.numel() * x.element_size() for x in remote)
    lb = sum(x.numel() * x.element_size() for x in local)
    flops = max(x.numel() for x in remote + local)
    return placement.prefer_local(flops, rb, lb)


def _dispatch(sess, func, args, kwargs):
    name = str(func)
    if name not in _SPECIALS and _placement_local(sess, func, args, kwargs):
        return _run_locally(sess, func, args, kwargs, keep_local=True)
    special = _SPECIALS.get(name)
    if special is not None:
        return special(sess, func, args, kwargs)
    if torch.Tag.inplace_view in func.tags:
        return _inplace_view(sess, func, args, kwargs)
    if func.is_view:
        return _view(sess, func, args, kwargs)
    schema = func._schema
    if any(a.alias_info is not None and a.alias_info.is_write for a in schema.arguments):
        return _mutating(sess, func, args, kwargs)
    return _functional(sess, func, args, kwargs)


def _view(sess, func, args, kwargs):
    vals = normalize(func, args, kwargs)
    src = vals[0]
    if not isinstance(src, RemoteTensor):
        return _functional(sess, func, args, kwargs)
    out_meta = run_meta(func, args, kwargs)
    rest = tuple(_freeze(v) for v in vals[1:])
    st = src._rf_storage
    if str(func) in ("aten.alias.default", "aten.detach.default", "aten.lift_fresh.default"):
        return RemoteTensor(out_meta, st, src._rf_chain)
    if isinstance(out_meta, torch.Tensor):
        return RemoteTensor(out_meta, st, src._rf_chain + ((func, rest, None),))
    flat, spec = tree_flatten(out_meta)
    outs = [RemoteTensor(m, st, src._rf_chain + ((func, rest, i),)) for i, m in enumerate(flat)]
    return tree_unflatten(outs, spec)


def _inplace_view(sess, func, args, kwargs):
    """transpose_, squeeze_, unsqueeze_, t_, as_strided_, ...: change this
    tensor's metadata in place (PyTorch's C++ RNN code does this to outputs).

    The functional view joins the tensor's view chain, and the wrapper's own
    sizes/strides are updated by running the op with dispatch disabled (the
    approach of torch.utils._python_dispatch.return_and_correct_aliasing).
    """
    vals = normalize(func, args, kwargs)
    t = vals[0]
    found = functional_of(func)
    ffunc = found[0] if found else None
    if ffunc is None or not isinstance(t, RemoteTensor):
        raise UnsupportedOperatorError(f"{func}: unsupported in-place view operator")
    rest = tuple(_freeze(v) for v in vals[1:])
    fa, fk = _rebind(ffunc, [t._rf_meta, *vals[1:]])
    new_meta = run_meta(ffunc, fa, fk)
    t._rf_chain = t._rf_chain + ((ffunc, rest, None),)
    t._rf_meta = new_meta
    with torch._C._DisableTorchDispatch():
        func(*args, **kwargs)
    return t


def _freeze(v):
    if isinstance(v, list):
        return tuple(_freeze(x) for x in v)
    return v


def _wrap_outputs(sess, out_meta, values: list[Value]):
    flat, spec = tree_flatten(out_meta)
    it = iter(values)
    wrapped = []
    for m in flat:
        if isinstance(m, torch.Tensor):
            v = next(it)
            wrapped.append(RemoteTensor(m, Storage(sess, m, v)))
        else:
            wrapped.append(m)
    return tree_unflatten(wrapped, spec)


def _functional(sess, func, args, kwargs):
    name = str(func)
    entry = opset.OPS.get(name)
    if entry is not None and entry[1] & opset.FLAG_DDS:
        return _dds(sess, func, args, kwargs)
    if name in sess.unsupported_names or not sess.op_enabled(name):
        return _fallback(sess, func, args, kwargs)
    if entry[1] & opset.FLAG_DDS:
        return _dds(sess, func, args, kwargs)
    if entry[1] & (opset.FLAG_CLIENT | opset.FLAG_BANNED):
        return _fallback(sess, func, args, kwargs)
    out_meta = meta_cached(func, args, kwargs)
    vals = normalize(func, args, kwargs)
    outs = sess.record_values(name, func, vals, _flat_tensors(out_meta))
    return _wrap_outputs(sess, out_meta, outs)


def _fallback(sess, func, args, kwargs):
    """Not in the opset: decompose, else run locally, else fail (§11.2 step 2)."""
    name = str(func)
    from .decomp import TABLE as REFLOP_DECOMP

    d = REFLOP_DECOMP.get(func) or _decomp_table().get(func)
    if d is not None:
        r = d(*args, **kwargs)
        if r is not NotImplemented:
            return r
    if hasattr(func, "decompose"):
        try:
            r = func.decompose(*args, **kwargs)
            if r is not NotImplemented:
                return r
        except Exception:  # noqa: BLE001 - not every op can decompose
            pass
    from . import _placement as placement

    if sess.allow_local_fallback and placement.mode() != "remote":
        if name not in _warned_local:
            _warned_local.add(name)
            warnings.warn(f"reflop: {name} is not supported by the server; running it on the client CPU",
                          stacklevel=4)
        return _run_locally(sess, func, args, kwargs)
    entry = opset.OPS.get(name)
    why = "it is not in opset v1" if entry is None else (
        f"the server does not enable it (capability {entry[6]!r})" if entry[6] else "the server disabled it")
    raise UnsupportedOperatorError(f"{name} cannot run on the reflop device: {why}. "
                                   "Use a different operator, move this computation to the CPU, or enable "
                                   "local fallback with reflop.placement('auto').")


def _run_locally(sess, func, args, kwargs, keep_local=False):
    local = tree_map(lambda x: x._rf_storage.session.fetch(value_of(x), x._rf_meta) if isinstance(x, RemoteTensor) else
                     (torch.device("cpu") if isinstance(x, torch.device) and x.type == "reflop" else x), (args, kwargs))
    with torch._C._DisableTorchDispatch():
        out = func(*local[0], **local[1])
    if keep_local:
        return out
    return tree_map(lambda x: sess.upload_tensor(x) if isinstance(x, torch.Tensor) else x, out)


def _mutating(sess, func, args, kwargs):
    name = str(func)
    schema = func._schema
    vals = normalize(func, args, kwargs)
    if name in _BATCH_NORM:
        return _batch_norm(sess, func, vals)
    if name.startswith("aten._fused_") and name.endswith("_.default"):
        from . import optim

        return optim.fused(sess, func, vals)
    sp = _SPECIAL_FUNCTIONAL.get(name)
    if sp is not None:
        fname, ffunc = sp
        mutated = [0]
        fvals = vals
    else:
        found = functional_of(func)
        if found is None:
            d = _decomp_table().get(func)
            if d is not None:
                return d(*args, **kwargs)
            raise UnsupportedOperatorError(f"{name}: no functional form is known for this in-place operator")
        ffunc, mutated = found
        out_pos = [i for i in mutated if schema.arguments[i].kwarg_only]
        fvals = [v for i, v in enumerate(vals) if i not in out_pos]
        fschema = ffunc._schema.arguments
        if len(fschema) > len(fvals) and out_pos:
            out_t = vals[out_pos[0]]
            fill = {"dtype": out_t.dtype, "layout": torch.strided, "device": out_t.device, "pin_memory": None,
                    "memory_format": None}
            fvals = fvals + [fill[a.name] for a in fschema[len(fvals):]]
    targets = [vals[i] for i in mutated]
    # Run the functional op (records, decomposes, or falls back as usual).
    fargs, fkwargs = _rebind(ffunc, fvals)
    if any(isinstance(t, list) for t in targets):
        result = dispatch(sess, ffunc, fargs, fkwargs)
        flat_targets = [x for t in targets for x in (t if isinstance(t, list) else [t])]
        flat_results = _flat_tensors(result)
        for t, r in zip(flat_targets, flat_results, strict=True):
            _assign(sess, t, r)
        return None if schema.returns == [] else (targets[0] if len(targets) == 1 else tuple(targets))
    result = dispatch(sess, ffunc, fargs, fkwargs)
    results = _flat_tensors(result)
    for t, r in zip(targets, results):
        _assign(sess, t, r)
    if not schema.returns:
        return None
    if len(schema.returns) == 1:
        return targets[0]
    # e.g. sort.values(out=(values, indices)) returns the out tensors.
    return tuple(targets) if len(targets) == len(schema.returns) else result


def _rebind(func, vals):
    schema = func._schema
    args, kwargs = [], {}
    for a, v in zip(schema.arguments, vals, strict=True):
        if a.kwarg_only:
            kwargs[a.name] = v
        else:
            args.append(v)
    return tuple(args), kwargs


def _assign(sess, target, result) -> None:
    """target ← result, where result is a RemoteTensor or CPU tensor."""
    if isinstance(target, RemoteTensor):
        if isinstance(result, RemoteTensor):
            write(target, value_of(result), result._rf_meta)
        else:
            write(target, sess.cpu_value(result), _to_meta(result))
    elif isinstance(target, torch.Tensor):
        with torch._C._DisableTorchDispatch():
            target.copy_(result.cpu() if isinstance(result, RemoteTensor) else result)


def _batch_norm(sess, func, vals):
    # (input, weight, bias, running_mean, running_var, training, momentum, eps)
    inp, w, b, rm, rv, training, momentum, eps = vals
    if rm is None or rv is None:
        f = aten._native_batch_norm_legit.no_stats
        return dispatch(sess, f, (inp, w, b, training, momentum, eps), {})
    if not training:
        f = aten._native_batch_norm_legit_no_training.default
        return dispatch(sess, f, (inp, w, b, rm, rv, momentum, eps), {})
    f = aten._native_batch_norm_legit_functional.default
    out, mean, rstd, new_rm, new_rv = dispatch(sess, f, (inp, w, b, rm, rv, training, momentum, eps), {})
    _assign(sess, rm, new_rm)
    _assign(sess, rv, new_rv)
    return out, mean, rstd


# ---------------------------------------------------------------- data-dependent shapes (§9.3, ADR-012)


def _dds(sess, func, args, kwargs):
    name = str(func)
    if name == "aten.nonzero.default":
        (x,) = normalize(func, args, kwargs)
        cnt = dispatch(sess, _native("nonzero_count"), (x,), {})
        n = int(sess.fetch_scalar(value_of(cnt)))
        return dispatch(sess, aten.nonzero_static.default, (x,), {"size": n, "fill_value": -1})
    return _fallback(sess, func, args, kwargs)


def _native(name):
    from . import natives

    return natives.op(name)


# ---------------------------------------------------------------- specials


def _local_scalar(sess, func, args, kwargs):
    (t,) = args
    return sess.fetch_scalar(value_of(t))


def _to_copy(sess, func, args, kwargs):
    vals = normalize(func, args, kwargs)
    src, dtype, layout, device, pin, non_blocking, mf = vals
    if device is not None and torch.device(device).type != "reflop":
        if not isinstance(src, RemoteTensor):  # pragma: no cover
            return func(*args, **kwargs)
        cpu = sess.fetch(value_of(src), src._rf_meta)
        return cpu.to(dtype=dtype or cpu.dtype, device=device,
                      memory_format=mf or torch.preserve_format)
    if not isinstance(src, RemoteTensor):
        t = src.to(dtype=dtype or src.dtype)
        return _session_for(device).upload_tensor(t)
    if device is not None and _target_index(device) != src._rf_storage.session.device_index:
        # reflop:i -> reflop:j: different server sessions, move through the client.
        cpu = sess.fetch(value_of(src), src._rf_meta)
        return _session_for(device).upload_tensor(cpu.to(dtype=dtype or cpu.dtype))
    return _functional(sess, func, args, kwargs)


def _copy_(sess, func, args, kwargs):
    vals = normalize(func, args, kwargs)
    dst, src, _non_blocking = vals
    if not isinstance(dst, RemoteTensor):
        with torch._C._DisableTorchDispatch():
            dst.copy_(sess.fetch(value_of(src), src._rf_meta))
        return dst
    st = dst._rf_storage
    if not isinstance(src, RemoteTensor):
        if (st.value is None and not dst._rf_chain and tuple(src.shape) == tuple(dst.shape)
                and src.dtype == dst.dtype and src.dim() > 0):
            st.set(sess.upload_value(src))
            st.handle = st.value.handle
            return dst
        write(dst, sess.cpu_value(src), _to_meta(src))
        return dst
    write(dst, value_of(src), src._rf_meta)
    return dst


def _fill_(sess, func, args, kwargs):
    vals = normalize(func, args, kwargs)
    dst, value = vals[0], vals[1]
    st = dst._rf_storage
    if isinstance(value, torch.Tensor):
        value = value.item() if not isinstance(value, RemoteTensor) else value
    if st.value is None and not dst._rf_chain and not isinstance(value, torch.Tensor):
        v = sess.record_simple("aten.full.default", aten.full.default,
                               [list(dst.shape), value, dst.dtype, None, torch.device("reflop"), None],
                               dst._rf_meta)[0]
        st.set(v)
        return dst
    if isinstance(value, RemoteTensor):
        res = dispatch(sess, aten.expand.default, (value, list(dst.shape)), {})
        write(dst, value_of(res), res._rf_meta)
        return dst
    res = dispatch(sess, aten.fill.Scalar, (dst, value), {})
    write(dst, value_of(res), res._rf_meta)
    return dst


def _random_(sess, func, args, kwargs):
    """random_(from, to) → randint.low on the session generator (functional)."""
    vals = normalize(func, args, kwargs)
    dst = vals[0]
    name = str(func)
    if name == "aten.random_.from":
        lo, hi = vals[1], vals[2]
    elif name == "aten.random_.to":
        lo, hi = 0, vals[1]
    else:
        lo = 0
        hi = (1 << 63) - 1 if dst.dtype == torch.int64 else (
            2 if dst.dtype == torch.bool else int(torch.iinfo(dst.dtype).max) + 1
            if not dst.dtype.is_floating_point else 1 << torch.finfo(dst.dtype).nmant + 1)
    if hi is None:
        hi = int(torch.iinfo(dst.dtype).max) + 1 if not dst.dtype.is_floating_point else 1 << 24
    r = dispatch(sess, aten.randint.low, (lo, hi, list(dst.shape)),
                 {"dtype": dst.dtype, "device": dst.device})
    write(dst, value_of(r), r._rf_meta)
    return dst


def _zero_(sess, func, args, kwargs):
    return _fill_(sess, aten.fill_.Scalar, (args[0], 0), {})


def _equal(sess, func, args, kwargs):
    a, b = args[0], args[1]
    if tuple(a.shape) != tuple(b.shape):
        return False
    r = dispatch(sess, aten.all.default, (dispatch(sess, aten.eq.Tensor, (a, b), {}),), {})
    return bool(sess.fetch_scalar(value_of(r)))


def _is_nonzero(sess, func, args, kwargs):
    return bool(sess.fetch_scalar(value_of(args[0])))


def _is_same_size(sess, func, args, kwargs):
    return tuple(args[0].shape) == tuple(args[1].shape)


def _refuse(msg):
    def f(sess, func, args, kwargs):
        raise UnsupportedOperatorError(f"{func}: {msg}")
    return f


def _index_tensor(sess, func, args, kwargs):
    self_, indices = normalize(func, args, kwargs)
    if any(isinstance(i, torch.Tensor) and i.dtype == torch.bool for i in indices if i is not None):
        indices = _expand_bool_indices(sess, indices)
        args = (self_, indices)
    return _functional(sess, func, args, {})


def _index_put(sess, func, args, kwargs):
    vals = normalize(func, args, kwargs)
    if any(isinstance(i, torch.Tensor) and i.dtype == torch.bool for i in vals[1] if i is not None):
        vals[1] = _expand_bool_indices(sess, vals[1])
    a, kw = _rebind(func, vals)
    if name_is_mutating(func):
        return _mutating(sess, func, a, kw)
    return _functional(sess, func, a, kw)


def name_is_mutating(func) -> bool:
    return any(x.alias_info is not None and x.alias_info.is_write for x in func._schema.arguments)


def _expand_bool_indices(sess, indices):
    """Boolean masks have a data-dependent number of selected elements:
    count on the server, then use a static-size nonzero (two-phase)."""
    out = []
    for i in indices:
        if isinstance(i, torch.Tensor) and i.dtype in (torch.bool, torch.uint8):
            if not isinstance(i, RemoteTensor):
                i = sess.upload_tensor(i.to(torch.bool))
            nz = dispatch(sess, aten.nonzero.default, (i,), {})
            out.extend(dispatch(sess, aten.unbind.int, (nz, 1), {}))
        else:
            out.append(i)
    return out


def _masked_select(sess, func, args, kwargs):
    self_, mask = normalize(func, args, kwargs)
    shape = torch.broadcast_shapes(self_.shape, mask.shape)
    s = dispatch(sess, aten.expand.default, (self_, list(shape)), {}) if tuple(self_.shape) != shape else self_
    m = dispatch(sess, aten.expand.default, (mask, list(shape)), {}) if tuple(mask.shape) != shape else mask
    return _index_tensor(sess, aten.index.Tensor, (s, [m]), {})


_SPECIALS = {
    "aten._local_scalar_dense.default": _local_scalar,
    "aten._to_copy.default": _to_copy,
    "aten.copy_.default": _copy_,
    "aten.fill_.Scalar": _fill_,
    "aten.fill_.Tensor": _fill_,
    "aten.zero_.default": _zero_,
    "aten.random_.from": _random_,
    "aten.random_.to": _random_,
    "aten.random_.default": _random_,
    "aten.equal.default": _equal,
    "aten.is_nonzero.default": _is_nonzero,
    "aten.is_same_size.default": _is_same_size,
    "aten.index.Tensor": _index_tensor,
    "aten.index_put.default": _index_put,
    "aten.index_put_.default": _index_put,
    "aten._unsafe_index.Tensor": _index_tensor,
    "aten.masked_select.default": _masked_select,
    "aten.set_.source_Tensor": _refuse("rebinding a reflop tensor's storage is not supported"),
    "aten.set_.source_Storage": _refuse("rebinding a reflop tensor's storage is not supported"),
    "aten.set_.source_Storage_storage_offset": _refuse("rebinding a reflop tensor's storage is not supported"),
    "aten.resize_.default": _refuse("resizing a reflop tensor in place is not supported (BANNED)"),
    "aten.resize_as_.default": _refuse("resizing a reflop tensor in place is not supported (BANNED)"),
    "aten.record_stream.default": lambda sess, func, args, kwargs: None,
}


# ---------------------------------------------------------------- factory kernels


def _target_index(device) -> int:
    d = torch.device(device) if not isinstance(device, torch.device) else device
    return d.index or 0


_active = threading.local()


def _session_for(device):
    from . import _placement, session

    if _placement.mode() == "remote-cpu":
        return _cpu_session()
    # Inside an op's decomposition every new tensor belongs to the session
    # being dispatched (tensors all report reflop:0, see RemoteTensor.__new__).
    act = getattr(_active, "sess", None)
    if act is not None:
        return act

    if isinstance(device, (str, int)):
        device = torch.device(device) if isinstance(device, str) else torch.device("reflop", device)
    idx = device.index if isinstance(device, torch.device) and device.index is not None else 0
    try:
        return session.by_index(idx)
    except KeyError:
        if idx == 0 and session.has_default():
            return session.current()
        raise RuntimeError(f"no reflop session for device reflop:{idx}; call reflop.device({idx}) first") from None


def _cpu_session():
    """The server-CPU session, opened on demand (placement "remote-cpu")."""
    from . import session
    from ._pb import graph_pb2 as gpb

    for s in list(session._sessions.values()):
        if s.gdevice.kind == gpb.DEVICE_CPU:
            return s
    import reflop

    idx = max(session._sessions, default=-1) + 1
    reflop.device(idx, kind="cpu", local=getattr(session.current().conn, "local_kind", None) and "cpu")
    return session.by_index(idx)


def _new_empty(size, dtype, stride=None, device=None):
    sess = _session_for(device)
    with torch._C._DisableTorchDispatch():
        if stride is None:
            meta = torch.empty(size, dtype=dtype or torch.get_default_dtype(), device="meta")
        else:
            meta = torch.empty_strided(size, stride, dtype=dtype or torch.get_default_dtype(), device="meta")
    return RemoteTensor(meta, Storage(sess, meta, None))


def _empty_memory_format(size, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None):
    t = _new_empty(size, dtype, device=device)
    if memory_format not in (None, torch.contiguous_format, torch.preserve_format):
        with torch._C._DisableTorchDispatch():
            meta = torch.empty(size, dtype=t.dtype, device="meta", memory_format=memory_format)
        return RemoteTensor(meta, Storage(t._rf_storage.session, meta, None))
    return t


def _empty_strided(size, stride, dtype=None, layout=None, device=None, pin_memory=None):
    return _new_empty(size, dtype, stride, device=device)


# Factory ops get direct kernels: their composite forms allocate an empty
# tensor and resize it through out=, which a remote tensor cannot do.
FACTORIES = [
    "arange.default", "arange.start", "arange.start_step", "linspace.default", "logspace.default", "eye.default",
    "eye.m", "full.default", "zeros.default", "ones.default", "rand.default", "randn.default", "randint.default",
    "randint.low", "randperm.default", "scalar_tensor.default",
]


def _factory_kernel(op):
    name = str(op)

    def kernel(*args, **kwargs):
        sess = _session_for(kwargs.get("device"))
        if name not in sess.enabled:
            raise UnsupportedOperatorError(f"{name} is not enabled on this server")
        meta_kw = dict(kwargs, device=torch.device("meta"))
        with torch._C._DisableTorchDispatch():
            meta = op(*args, **meta_kw)
        vals = normalize(op, args, kwargs)
        out = sess.record_values(name, op, vals, [meta])[0]
        return RemoteTensor(meta, Storage(sess, meta, out))
    return kernel


_lib = None


def register_backend_kernels() -> None:
    """Allocation on the reflop device returns lazy, logically-zero tensors
    (UNINIT ops are zero-filled, invariant I13)."""
    global _lib
    if _lib is not None:
        return
    _lib = torch.library.Library("aten", "IMPL")
    _lib.impl("empty.memory_format", _empty_memory_format, "PrivateUse1")
    _lib.impl("empty_strided", _empty_strided, "PrivateUse1")
    for qual in FACTORIES:
        base, ov = qual.split(".")
        pkt = getattr(aten, base, None)
        if pkt is None or ov not in pkt.overloads():
            continue
        _lib.impl(base if ov == "default" else qual, _factory_kernel(getattr(pkt, ov)), "PrivateUse1")


__all__ = ["RemoteTensor", "Storage", "dispatch", "value_of", "write", "is_remote", "run_meta"]
