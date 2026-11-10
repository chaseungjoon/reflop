# SPDX-License-Identifier: Apache-2.0
"""Per-operator validators (PLAN.md §8.3, invariants I4, I13).

Two hooks per validator name (from the generated op table):

  static(node, args, kwargs)   during V4, on meta tensors and constants
  runtime(node, args, kwargs)  during E2, on the real tensors, right before
                               the kernel launches

Runtime checks that read tensor values (index bounds, zero divisors,
offsets) synchronize with the device once per check. They are mandatory;
caching a graph's validation never skips them (rule R7).
"""

from __future__ import annotations

import math

import torch

from ._pb import compute_pb2 as cpb
from .errors import ReflopError, invalid

S = cpb.STAGE_V4_META
R = cpb.STAGE_E2_RUNTIME
INT_DTYPES = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.bool}
INDEX_DTYPES = {torch.int32, torch.int64}


def _named(node, args, kwargs) -> dict:
    """Map argument names (static schema) to values."""
    out = {}
    pos = iter(args)
    for spec in node.op.args:
        out[spec.name] = kwargs[spec.name] if spec.kwarg_only else next(pos)
    return out


def _bad(node, msg, stage=S, code=cpb.ERROR_INVALID_GRAPH) -> ReflopError:
    return ReflopError(code, f"{node.op.name}: {msg}", stage, node.index, node.op.id)


def _oob(node, msg) -> ReflopError:
    return ReflopError(cpb.ERROR_INDEX_OUT_OF_BOUNDS, f"{node.op.name}: {msg}", R, node.index, node.op.id)


def _is_t(x) -> bool:
    return isinstance(x, torch.Tensor)


def _check_index_dtype(node, idx, what="index") -> None:
    if _is_t(idx) and idx.dtype not in INDEX_DTYPES:
        raise _bad(node, f"{what} must be int32 or int64, got {idx.dtype}")


def bounds(node, idx: torch.Tensor, lo: int, hi: int, sentinel: int | None = None, what: str = "index") -> None:
    """Fail unless every element is in [lo, hi) (or equals sentinel)."""
    if not _is_t(idx) or idx.numel() == 0:
        return
    if idx.dtype not in INDEX_DTYPES and idx.dtype not in INT_DTYPES:
        raise _oob(node, f"{what} has non-integer dtype {idx.dtype}")
    if sentinel is None:
        mn, mx = torch.aminmax(idx.reshape(-1))
        mn, mx = int(mn), int(mx)  # one device synchronization
        if mn < lo or mx >= hi:
            raise _oob(node, f"{what} out of range: values span [{mn}, {mx}], valid range is [{lo}, {hi})")
        return
    ok = (idx == sentinel) | ((idx >= lo) & (idx < hi))
    if not bool(ok.all()):
        raise _oob(node, f"{what} out of range: valid range is [{lo}, {hi}) or {sentinel}")


def _dim(d: int, rank: int) -> int:
    r = max(rank, 1)
    if not -r <= d < r:
        raise IndexError(f"dimension {d} out of range for rank {rank}")
    return d % r


# ---------------------------------------------------------------- validators


def dim_index_static(node, a):
    _check_index_dtype(node, a["index"])
    _dim(a["dim"], a["self"].dim())


def dim_index_runtime(node, a):
    self_ = a["self"]
    d = _dim(a["dim"], self_.dim())
    size = self_.shape[d] if self_.dim() > 0 else 1
    bounds(node, a["index"], 0, size)


def advanced_index_static(node, a):
    for i in a["indices"]:
        if i is None:
            continue
        if i.dtype in (torch.bool, torch.uint8):
            raise _bad(node, "boolean mask indices have a data-dependent shape; the client must convert them "
                             "(nonzero_count + nonzero_static)")
        _check_index_dtype(node, i)


def advanced_index_runtime(node, a):
    self_ = a["self"]
    for d, i in enumerate(a["indices"]):
        if i is None:
            continue
        if d >= self_.dim():
            raise _oob(node, "too many indices")
        n = self_.shape[d]
        bounds(node, i, -n, n)


def embedding_static(node, a):
    _check_index_dtype(node, a["indices"])
    if a.get("sparse"):
        raise _bad(node, "sparse gradients are not supported")


def embedding_runtime(node, a):
    bounds(node, a["indices"], 0, a["weight"].shape[0])


def embedding_backward_static(node, a):
    _check_index_dtype(node, a["indices"])
    if a["num_weights"] < 0:
        raise _bad(node, "num_weights must be non-negative")


def embedding_backward_runtime(node, a):
    bounds(node, a["indices"], 0, a["num_weights"])


def embedding_bag_static(node, a):
    _check_index_dtype(node, a["indices"])
    _check_index_dtype(node, a["offsets"], "offsets")
    if a["mode"] not in (0, 1, 2):
        raise _bad(node, "mode must be 0 (sum), 1 (mean) or 2 (max)")
    if a.get("sparse"):
        raise _bad(node, "sparse gradients are not supported")
    if a["indices"].dim() != 1 or a["offsets"].dim() != 1:
        raise _bad(node, "indices and offsets must be 1-D (flatten 2-D input on the client)")
    psw = a.get("per_sample_weights")
    if psw is not None and tuple(psw.shape) != tuple(a["indices"].shape):
        raise _bad(node, "per_sample_weights must have the same shape as indices")


def embedding_bag_runtime(node, a):
    idx, off = a["indices"], a["offsets"]
    bounds(node, idx, 0, a["weight"].shape[0])
    n = idx.numel()
    if off.numel() == 0:
        return
    if int(off[0]) != 0:
        raise _oob(node, "the first offset must be 0")
    if off.numel() > 1 and bool((off[1:] < off[:-1]).any()):
        raise _oob(node, "offsets must be non-decreasing")
    last = int(off[-1])
    if last > n or (a.get("include_last_offset") and last != n):
        raise _oob(node, f"offsets exceed the number of indices ({n})")


def embedding_bag_backward_runtime(node, a):
    bounds(node, a["indices"], 0, a["num_weights"])
    o2b = a.get("offset2bag")
    if o2b is not None:
        bounds(node, o2b, 0, max(a["grad"].shape[0], 1), what="offset2bag")
    mi = a.get("maximum_indices")
    if mi is not None:
        bounds(node, mi, -1, a["num_weights"], what="maximum_indices")


def embedding_bag_backward_static(node, a):
    _check_index_dtype(node, a["indices"])
    if a.get("sparse"):
        raise _bad(node, "sparse gradients are not supported")


def embedding_renorm_runtime(node, a):
    bounds(node, a["indices"], 0, a["self"].shape[0])


def pool_indices_runtime(node, a):
    self_ = a["self"]
    spatial = 3 if "3d" in node.op.name else 2
    n = math.prod(self_.shape[-spatial:]) if self_.dim() >= spatial else 0
    bounds(node, a["indices"], 0, n)


def pool_indices_static(node, a):
    if a["indices"].dtype != torch.int64:
        raise _bad(node, "pooling indices must be int64")


def max_unpool_runtime(node, a):
    n = math.prod(a["output_size"])
    bounds(node, a["indices"], 0, n)


def nll_loss_static(node, a):
    if a["reduction"] not in (0, 1, 2):
        raise _bad(node, "reduction must be 0 (none), 1 (mean) or 2 (sum)")
    _check_index_dtype(node, a["target"], "target")


def nll_loss_runtime(node, a):
    x = a["self"]
    c = x.shape[0] if x.dim() == 1 else x.shape[1]
    bounds(node, a["target"], 0, c, sentinel=a["ignore_index"], what="target")


def take_runtime(node, a):
    n = a["self"].numel()
    bounds(node, a["index"], -n, n)


def take_static(node, a):
    _check_index_dtype(node, a["index"])


def _divisor(node, a):
    for k in ("other", "divisor"):
        if k in a:
            return a[k]
    raise _bad(node, "no divisor argument")


def int_divisor_nonzero_runtime(node, a):
    x = a.get("self")
    d = _divisor(node, a)
    xd = x.dtype if _is_t(x) else None
    dd = d.dtype if _is_t(d) else (torch.int64 if isinstance(d, int) and not isinstance(d, bool) else None)
    if xd is None or xd not in INT_DTYPES or dd is None or dd not in INT_DTYPES:
        return  # floating division is defined for zero divisors
    if node.op.name.startswith("aten.div.") and a.get("rounding_mode") is None:
        return  # true division returns float
    if _is_t(d):
        if d.numel() and bool((d == 0).any()):
            raise ReflopError(cpb.ERROR_RUNTIME_CHECK, f"{node.op.name}: integer division by zero", R,
                              node.index, node.op.id)
    elif d == 0:
        raise ReflopError(cpb.ERROR_RUNTIME_CHECK, f"{node.op.name}: integer division by zero", R, node.index,
                          node.op.id)


def shift_amount_runtime(node, a):
    x = a.get("self")
    s = a.get("other")
    dt = x.dtype if _is_t(x) else None
    if dt is None or dt not in INT_DTYPES or dt == torch.bool:
        return
    bits = torch.iinfo(dt).bits
    if _is_t(s):
        if s.numel():
            mn, mx = int(s.min()), int(s.max())
            if mn < 0 or mx >= bits:
                raise ReflopError(cpb.ERROR_RUNTIME_CHECK, f"{node.op.name}: shift amount outside [0, {bits})",
                                  R, node.index, node.op.id)
    elif isinstance(s, int) and not 0 <= s < bits:
        raise ReflopError(cpb.ERROR_RUNTIME_CHECK, f"{node.op.name}: shift amount outside [0, {bits})", R,
                          node.index, node.op.id)


def masked_scatter_runtime(node, a):
    self_, mask, src = a["self"], a["mask"], a["source"]
    need = int(mask.expand(self_.shape).sum()) if mask.numel() else 0
    if src.numel() < need:
        raise ReflopError(cpb.ERROR_RUNTIME_CHECK,
                          f"{node.op.name}: source has {src.numel()} elements but the mask selects {need}", R,
                          node.index, node.op.id)


def multinomial_static(node, a):
    p = a["self"]
    if p.dim() not in (1, 2):
        raise _bad(node, "probabilities must be 1-D or 2-D")
    n = a["num_samples"]
    if n <= 0 or (not a["replacement"] and n > p.shape[-1]):
        raise _bad(node, "invalid num_samples")


def multinomial_runtime(node, a):
    p = a["self"]
    if p.numel() == 0:
        return
    ok = bool(torch.isfinite(p).all()) and bool((p >= 0).all()) and bool((p.sum(-1) > 0).all())
    if not ok:
        raise ReflopError(cpb.ERROR_RUNTIME_CHECK, f"{node.op.name}: probabilities must be finite, non-negative "
                          "and not all zero", R, node.index, node.op.id)


def _reach(size, stride, offset) -> int:
    """Largest reachable element index, checked arithmetic."""
    if any(s < 0 for s in stride) or any(n < 0 for n in size) or offset < 0:
        raise ValueError("sizes, strides and offset must be non-negative")
    if len(size) != len(stride):
        raise ValueError("size and stride must have the same length")
    if any(n == 0 for n in size):
        return offset - 1
    top = offset
    for n, s in zip(size, stride):
        top += (n - 1) * s
        if top > (1 << 62):
            raise OverflowError("as_strided reaches beyond 2^62 elements")
    return top


def _storage_numel(t: torch.Tensor) -> int:
    return t.untyped_storage().nbytes() // t.element_size()


def as_strided_static(node, a):
    t = a["self"]
    off = a.get("storage_offset")
    try:
        top = _reach(a["size"], a["stride"], t.storage_offset() if off is None else off)
    except (ValueError, OverflowError) as e:
        raise _bad(node, str(e)) from None
    if top >= _storage_numel(t):
        raise _bad(node, "view reaches past the end of the input's storage")


def as_strided_runtime(node, a):
    t = a["self"]
    off = a.get("storage_offset")
    try:
        top = _reach(a["size"], a["stride"], t.storage_offset() if off is None else off)
    except (ValueError, OverflowError) as e:
        raise _oob(node, str(e)) from None
    # Checked against the server-side storage of the actual tensor, never a
    # client claim (§8.4).
    if top >= _storage_numel(t):
        raise _oob(node, "view reaches past the end of the input's storage")


def empty_strided_static(node, a):
    size, stride = a["size"], a["stride"]
    if len(size) != len(stride):
        raise _bad(node, "size and stride must have the same length")
    dims = sorted((s, n) for n, s in zip(size, stride) if n != 1)
    expect = 1
    for s, n in dims:
        if s != expect:
            raise _bad(node, "strides must describe a dense, non-overlapping layout")
        expect *= n


def to_copy_static(node, a):
    if a.get("layout") not in (None, torch.strided):
        raise _bad(node, "only strided layout")


def cdist_static(node, a):
    p = a.get("p")
    if isinstance(p, float) and (p < 0 or math.isnan(p)):
        raise _bad(node, "p must be non-negative")
    cm = a.get("compute_mode")
    if cm is not None and cm not in (0, 1, 2):
        raise _bad(node, "invalid compute_mode")


def reduction_static(node, a):
    r = a.get("reduction")
    if r is not None and r not in (0, 1, 2):
        raise _bad(node, "reduction must be 0, 1 or 2")


def attention_static(node, a):
    p = a.get("dropout_p")
    if isinstance(p, float) and not 0.0 <= p < 1.0:
        raise _bad(node, "dropout_p must be in [0, 1)")


def int_mm_static(node, a):
    if a["self"].dtype != torch.int8 or a["mat2"].dtype != torch.int8:
        raise _bad(node, "_int_mm takes int8 matrices")


def linalg_static(node, a):
    for v in a.values():
        if _is_t(v) and v.dim() >= 2 and max(v.shape[-1], v.shape[-2]) > 32768:
            raise _bad(node, "matrix dimension exceeds 32768", code=cpb.ERROR_RESOURCE_LIMIT)


def scaled_mm_static(node, a):
    for k in ("scale_a", "scale_b"):
        s = a.get(k)
        if _is_t(s) and s.dtype != torch.float32:
            raise _bad(node, f"{k} must be float32")


def cudnn_rnn_static(node, a):
    if a.get("weight_buf") is not None:
        # Flat weight buffers would need layout validation against the RNN
        # configuration; v1 accepts only the per-parameter weight list.
        raise _bad(node, "flat weight buffers are not supported; call flatten_parameters() is not needed")


def ctc_static(node, a):
    _check_index_dtype(node, a["targets"], "targets")
    lp = a["log_probs"]
    if lp.dim() != 3:
        raise _bad(node, "log_probs must be (T, N, C)")
    if not 0 <= a["blank"] < lp.shape[2]:
        raise _bad(node, "blank index out of range")


def ctc_runtime(node, a):
    lp, tg = a["log_probs"], a["targets"]
    t_len, c = lp.shape[0], lp.shape[2]
    bounds(node, tg, 0, c, what="targets")
    il, tl = a["input_lengths"], a["target_lengths"]
    s = tg.shape[-1] if tg.dim() == 2 else tg.numel()
    if _is_t(il):
        bounds(node, il, 0, t_len + 1, what="input_lengths")
    elif any(not 0 <= x <= t_len for x in il):
        raise _oob(node, "input_lengths out of range")
    if _is_t(tl):
        bounds(node, tl, 0, s + 1, what="target_lengths")
    elif any(not 0 <= x <= s for x in tl):
        raise _oob(node, "target_lengths out of range")


def foreach_lengths_static(node, a):
    lens = {len(v) for v in a.values() if isinstance(v, list) and v and (_is_t(v[0]) or v[0] is None)}
    if len(lens) > 1:
        raise _bad(node, "tensor lists must have equal lengths")


def fused_optim_static(node, a):
    foreach_lengths_static(node, a)


def dataset_gather_static(node, a):
    _check_index_dtype(node, a["indices"])
    if a["indices"].dim() != 1:
        raise _bad(node, "indices must be 1-D")


def dataset_gather_runtime(node, a):
    bounds(node, a["indices"], 0, a["dataset"].shape[0])


def bounds_check_runtime(node, a):
    bounds(node, a["index"], 0, a["size"], sentinel=a.get("sentinel"))


def alloc_static(node, a):
    # Output sizes are computed by meta inference and checked against the
    # protocol limits there and the rails in V5; nothing extra here.
    return


STATIC = {
    "dim_index": dim_index_static, "advanced_index": advanced_index_static, "embedding": embedding_static,
    "embedding_backward": embedding_backward_static, "embedding_bag": embedding_bag_static,
    "embedding_bag_backward": embedding_bag_backward_static, "pool_indices": pool_indices_static,
    "nll_loss": nll_loss_static, "take": take_static, "multinomial": multinomial_static,
    "as_strided": as_strided_static, "empty_strided": empty_strided_static, "to_copy": to_copy_static,
    "cdist": cdist_static, "reduction": reduction_static, "attention": attention_static,
    "int_mm": int_mm_static, "linalg": linalg_static, "scaled_mm": scaled_mm_static,
    "cudnn_rnn": cudnn_rnn_static, "ctc": ctc_static, "foreach_lengths": foreach_lengths_static,
    "fused_optim": fused_optim_static, "dataset_gather": dataset_gather_static, "alloc": alloc_static,
}

RUNTIME = {
    "dim_index": dim_index_runtime, "advanced_index": advanced_index_runtime, "embedding": embedding_runtime,
    "embedding_backward": embedding_backward_runtime, "embedding_bag": embedding_bag_runtime,
    "embedding_bag_backward": embedding_bag_backward_runtime, "embedding_renorm": embedding_renorm_runtime,
    "pool_indices": pool_indices_runtime, "max_unpool": max_unpool_runtime, "nll_loss": nll_loss_runtime,
    "take": take_runtime, "int_divisor_nonzero": int_divisor_nonzero_runtime,
    "shift_amount": shift_amount_runtime, "masked_scatter": masked_scatter_runtime,
    "multinomial": multinomial_runtime, "as_strided": as_strided_runtime, "ctc": ctc_runtime,
    "dataset_gather": dataset_gather_runtime, "bounds_check": bounds_check_runtime,
}

# Every validator named in the opset must exist (checked at import and in CI).
KNOWN = set(STATIC) | set(RUNTIME)


DIM_ARGS = {"dim", "dim0", "dim1", "dim2", "start_dim", "end_dim", "dims", "source", "destination"}
# Ops whose dim may address one past the input's last dim (they add a dim).
DIM_PLUS_ONE = ("aten.unsqueeze", "aten.stack", "aten.unsqueeze_copy", "aten.diag_embed")


def check_dims(node, args, kwargs) -> None:
    """V3 for every op: dim-like arguments must address a real dimension of
    the first tensor input (§9.2: "dim within rank")."""
    named = _named(node, args, kwargs)
    first = next((v for v in named.values() if isinstance(v, torch.Tensor)), None)
    if first is None:
        lst = next((v for v in named.values() if isinstance(v, list) and v and isinstance(v[0], torch.Tensor)), None)
        first = lst[0] if lst else None
    if first is None:
        return
    rank = max(first.dim(), 1)
    if node.op.name.startswith(DIM_PLUS_ONE):
        rank += 1
    for spec in node.op.args:
        if spec.name not in DIM_ARGS:
            continue
        v = named.get(spec.name)
        vals = v if isinstance(v, list) else [v]
        for d in vals:
            if isinstance(d, int) and not isinstance(d, bool) and not -rank <= d < rank:
                raise invalid(f"{node.op.name}: {spec.name}={d} is out of range for a {first.dim()}-d input",
                              cpb.STAGE_V3_SCHEMA, node.index, node.op.id)


def check_static(node, args, kwargs) -> None:
    check_dims(node, args, kwargs)
    name = node.op.validator
    if name is None:
        return
    f = STATIC.get(name)
    if f is not None:
        try:
            f(node, _named(node, args, kwargs))
        except ReflopError:
            raise
        except (IndexError, ValueError, TypeError, KeyError) as e:
            raise invalid(f"{node.op.name}: {e}", S, node.index, node.op.id) from None


def check_runtime(node, args, kwargs) -> None:
    name = node.op.validator
    if name is None:
        return
    f = RUNTIME.get(name)
    if f is not None:
        try:
            f(node, _named(node, args, kwargs))
        except ReflopError:
            raise
        except (IndexError, ValueError, TypeError, KeyError) as e:
            raise ReflopError(cpb.ERROR_RUNTIME_CHECK, f"{node.op.name}: {e}", R, node.index,
                              node.op.id) from None


def has_runtime(op) -> bool:
    return op.validator in RUNTIME
