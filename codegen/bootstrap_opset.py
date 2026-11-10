#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""M0 spike S5: build opset/v1/opset.yaml from PLAN.md §8 and the pinned PyTorch.

This is a one-time bootstrap. After it has run, opset/v1/opset.yaml is the
single source of truth (invariant I10) and is edited by hand; this script is
kept so the derivation stays reviewable. It:

  1. lists every overload tagged torch.Tag.core in the pinned PyTorch,
  2. resolves every operator named in PLAN.md §8 (Tier 0/1/2) to an overload,
  3. records each overload's schema (argument names, types, defaults),
  4. attaches the security flags, validators and cost functions of §8,
  5. writes opset.yaml and a census (docs/opset_census.json) of mismatches.

    python codegen/bootstrap_opset.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "opset/v1/opset.yaml"
CENSUS = ROOT / "docs/opset_census.json"

# --------------------------------------------------------------------------
# PLAN.md §8, as data. Each group: (tier, category, names, flags, extra).
# A name without an overload means ".default". Wildcards ("_foreach_sub.*")
# expand to every overload the pinned PyTorch has.

UNARY = "abs acos acosh asin asinh atan atanh ceil cos cosh erf exp expm1 floor log log10 log1p log2 neg reciprocal round rsqrt sign sin sinh sqrt tan tanh trunc sigmoid".split()

TIER0 = [
    ("unary", UNARY, [], {}),
    ("binary", ["add.Scalar", "add.Tensor", "sub.Scalar", "sub.Tensor", "mul.Scalar", "mul.Tensor",
                "div.Scalar", "div.Tensor", "pow.Scalar", "pow.Tensor_Scalar", "pow.Tensor_Tensor",
                "atan2", "maximum", "minimum"], [], {}),
    ("binary", ["fmod.Scalar", "fmod.Tensor", "remainder.Scalar", "remainder.Tensor"], ["RESTRICTED"],
     {"validator": "int_divisor_nonzero"}),
    ("binary", ["div.Scalar_mode", "div.Tensor_mode"], ["ENUM", "RESTRICTED"],
     {"validator": "int_divisor_nonzero", "enum_args": {"rounding_mode": ["none", "trunc", "floor"]}}),
    ("binary", ["atan2.out"], ["FUNC"], {"functional_of": "aten.atan2.default", "client_rewrite": "aten.atan2.default"}),
    ("compare", ["eq.Scalar", "eq.Tensor", "ne.Scalar", "ne.Tensor", "lt.Scalar", "lt.Tensor", "le.Scalar",
                 "le.Tensor", "gt.Scalar", "gt.Tensor", "ge.Scalar", "ge.Tensor", "isinf", "isnan",
                 "logical_and", "logical_or", "logical_xor", "logical_not", "bitwise_and.Scalar",
                 "bitwise_and.Tensor", "bitwise_or.Scalar", "bitwise_or.Tensor", "bitwise_xor.Scalar",
                 "bitwise_xor.Tensor", "bitwise_not", "where.self"], [], {}),
    ("activation", ["clamp", "clamp.Tensor", "relu", "leaky_relu", "hardtanh", "elu"], [], {}),
    ("activation", ["gelu"], ["ENUM"], {"enum_args": {"approximate": ["none", "tanh"]}}),
    ("reduction", ["sum.dim_IntList", "mean", "mean.dim", "prod", "prod.dim_int", "amax", "amin", "any",
                   "any.dim", "any.dims", "argmax", "argmin", "max.dim", "min.dim", "var.correction",
                   "var.dim", "cumsum"], [], {"cost": "reduce"}),
    ("sort", ["sort", "topk"], [], {"cost": "sort"}),
    ("matmul", ["mm", "addmm", "bmm"], [], {"cost": "matmul"}),
    ("conv", ["convolution", "convolution_backward"], [], {"cost": "conv"}),
    ("norm", ["native_layer_norm", "native_layer_norm_backward", "native_group_norm",
              "native_group_norm_backward", "_native_batch_norm_legit.no_stats",
              "_native_batch_norm_legit_no_training"], [], {}),
    ("norm", ["_native_batch_norm_legit"], ["FUNC"],
     {"functional_of": "aten._native_batch_norm_legit.default",
      "client_rewrite": "aten._native_batch_norm_legit_functional.default"}),
    ("softmax", ["_softmax", "_log_softmax"], [], {}),
    ("rng", ["native_dropout"], ["RNG"], {}),
    ("rng", ["rand", "randn", "randperm"], ["RNG", "ALLOC"], {"validator": "alloc"}),
    ("pool", ["avg_pool1d", "avg_pool2d", "avg_pool2d_backward", "avg_pool3d", "adaptive_avg_pool1d",
              "_adaptive_avg_pool2d", "_adaptive_avg_pool2d_backward", "_adaptive_avg_pool3d",
              "max_pool2d_with_indices", "max_pool3d_with_indices"], [], {}),
    ("pool", ["max_pool2d_with_indices_backward"], ["IDX"],
     {"validator": "pool_indices", "index_inputs": [{"arg": "indices", "into": "spatial(self, 2)"}]}),
    ("pad", ["constant_pad_nd", "reflection_pad1d", "reflection_pad2d", "reflection_pad3d",
             "replication_pad2d", "replication_pad3d"], ["ALLOC"], {"validator": "alloc"}),
    ("resample", ["upsample_bilinear2d.vec", "upsample_nearest2d.vec", "col2im", "grid_sampler_2d"],
     ["ALLOC"], {"validator": "alloc"}),
    ("index", ["index.Tensor"], ["IDX"],
     {"validator": "advanced_index", "index_inputs": [{"arg": "indices", "into": "advanced(self)", "negative": True}]}),
    ("index", ["index_put"], ["IDX"],
     {"validator": "advanced_index", "index_inputs": [{"arg": "indices", "into": "advanced(self)", "negative": True}]}),
    ("index", ["index_select"], ["IDX"],
     {"validator": "dim_index", "index_inputs": [{"arg": "index", "into": "size(self, dim)"}]}),
    ("index", ["gather"], ["IDX"],
     {"validator": "dim_index", "index_inputs": [{"arg": "index", "into": "size(self, dim)"}]}),
    ("index", ["scatter.src", "scatter.value", "scatter_add"], ["IDX"],
     {"validator": "dim_index", "index_inputs": [{"arg": "index", "into": "size(self, dim)"}]}),
    ("index", ["scatter_reduce.two"], ["IDX", "ENUM"],
     {"validator": "dim_index", "index_inputs": [{"arg": "index", "into": "size(self, dim)"}],
      "enum_args": {"reduce": ["sum", "prod", "mean", "amax", "amin"]}}),
    ("index", ["embedding"], ["IDX"],
     {"validator": "embedding", "index_inputs": [{"arg": "indices", "into": "size(weight, 0)"}]}),
    ("index", ["embedding_dense_backward"], ["IDX"],
     {"validator": "embedding_backward", "index_inputs": [{"arg": "indices", "into": "num_weights"}]}),
    ("index", ["_embedding_bag"], ["IDX", "RESTRICTED"],
     {"validator": "embedding_bag", "index_inputs": [{"arg": "indices", "into": "size(weight, 0)"}]}),
    ("index", ["masked_scatter"], ["RESTRICTED"], {"validator": "masked_scatter"}),
    ("index", ["select.int", "select_scatter", "slice.Tensor", "slice_scatter"], [], {}),
    ("index", ["nonzero"], ["DDS", "CLIENT"], {"client_rewrite": "reflop.nonzero_count + aten.nonzero_static"}),
    ("view", ["view", "expand", "permute", "squeeze.dim", "squeeze.dims", "unsqueeze", "diagonal", "alias",
              "split_with_sizes", "cat", "flip", "clone"], [], {}),
    ("view", ["repeat"], ["ALLOC"], {"validator": "alloc"}),
    ("view", ["as_strided"], ["RESTRICTED"], {"validator": "as_strided"}),
    ("creation", ["full", "full_like", "scalar_tensor", "arange.start_step", "fill.Scalar"], ["ALLOC"],
     {"validator": "alloc"}),
    ("creation", ["empty.memory_format"], ["UNINIT", "ALLOC"], {"validator": "alloc"}),
    ("creation", ["empty_strided"], ["UNINIT", "ALLOC", "RESTRICTED"], {"validator": "empty_strided"}),
    ("device", ["_to_copy", "copy"], ["RESTRICTED"], {"validator": "to_copy"}),
    ("distance", ["_cdist_forward", "_pdist_forward"], ["RESTRICTED"], {"validator": "cdist"}),
    ("fft", ["_fft_r2c", "_fft_c2r"], [], {"capability": "complex"}),
    ("client", ["sym_size.int", "sym_stride.int", "sym_numel", "sym_storage_offset", "sym_is_contiguous",
                "_local_scalar_dense"], ["CLIENT"], {}),
    ("banned", ["resize_"], ["BANNED"], {}),
]

UPSAMPLE = ["upsample_nearest1d", "upsample_nearest2d", "upsample_nearest3d", "_upsample_nearest_exact1d",
            "_upsample_nearest_exact2d", "_upsample_nearest_exact3d", "upsample_linear1d", "upsample_bilinear2d",
            "upsample_bicubic2d", "upsample_trilinear3d", "_upsample_bilinear2d_aa"]

FOREACH = ["_foreach_add.*", "_foreach_sub.*", "_foreach_mul.*", "_foreach_div.*", "_foreach_addcmul.*",
           "_foreach_addcdiv.*", "_foreach_lerp.*", "_foreach_sqrt", "_foreach_neg", "_foreach_maximum.*",
           "_foreach_minimum.*", "_foreach_norm.Scalar", "_foreach_sign", "_foreach_abs", "_foreach_clamp_min.*",
           "_foreach_clamp_max.*", "_foreach_pow.*", "_foreach_reciprocal", "_foreach_exp", "_foreach_zero"]

TIER1 = [
    ("activation_bw", ["threshold_backward", "sigmoid_backward", "tanh_backward", "elu_backward",
                       "leaky_relu_backward", "hardtanh_backward", "silu", "silu_backward", "hardswish",
                       "hardswish_backward", "hardsigmoid", "hardsigmoid_backward", "softplus",
                       "softplus_backward", "mish", "mish_backward", "log_sigmoid_forward",
                       "log_sigmoid_backward", "_prelu_kernel", "_prelu_kernel_backward", "glu",
                       "glu_backward", "threshold"], [], {}),
    ("activation_bw", ["gelu_backward"], ["ENUM"], {"enum_args": {"approximate": ["none", "tanh"]}}),
    ("softmax", ["_softmax_backward_data", "_log_softmax_backward_data", "logsumexp"], [], {}),
    ("norm", ["native_batch_norm_backward", "_native_batch_norm_legit_functional",
              "_fused_rms_norm", "_fused_rms_norm_backward", "_weight_norm_interface",
              "_weight_norm_interface_backward", "native_batch_norm"], [], {}),
    ("attention", ["_scaled_dot_product_flash_attention", "_scaled_dot_product_flash_attention_backward",
                   "_scaled_dot_product_efficient_attention", "_scaled_dot_product_efficient_attention_backward",
                   "_scaled_dot_product_cudnn_attention", "_scaled_dot_product_cudnn_attention_backward",
                   "_scaled_dot_product_attention_math",
                   "_scaled_dot_product_flash_attention_for_cpu",
                   "_scaled_dot_product_flash_attention_for_cpu_backward"],
     ["RNG", "RESTRICTED"], {"validator": "attention", "cost": "attention"}),
    ("loss", ["nll_loss_forward", "nll_loss_backward", "nll_loss2d_forward", "nll_loss2d_backward"],
     ["IDX"], {"validator": "nll_loss", "index_inputs": [{"arg": "target", "into": "classes(self)", "sentinel": "ignore_index"}]}),
    ("loss", ["mse_loss", "mse_loss_backward", "smooth_l1_loss", "smooth_l1_loss_backward", "huber_loss",
              "huber_loss_backward", "binary_cross_entropy", "binary_cross_entropy_backward",
              "binary_cross_entropy_with_logits", "soft_margin_loss", "soft_margin_loss_backward"],
     ["RESTRICTED"], {"validator": "reduction"}),
    ("matmul", ["baddbmm", "addbmm", "addmv", "_addmm_activation", "dot", "vdot", "mv", "outer"], [],
     {"cost": "matmul"}),
    ("matmul", ["_int_mm"], ["RESTRICTED"], {"validator": "int_mm", "cost": "matmul"}),
    ("pool", ["avg_pool3d_backward", "_adaptive_avg_pool3d_backward", "adaptive_max_pool2d", "adaptive_max_pool3d"], [], {}),
    ("pool", ["max_pool3d_with_indices_backward"], ["IDX"],
     {"validator": "pool_indices", "index_inputs": [{"arg": "indices", "into": "spatial(self, 3)"}]}),
    ("pool", ["adaptive_max_pool2d_backward"], ["IDX"],
     {"validator": "pool_indices", "index_inputs": [{"arg": "indices", "into": "spatial(self, 2)"}]}),
    ("pool", ["adaptive_max_pool3d_backward"], ["IDX"],
     {"validator": "pool_indices", "index_inputs": [{"arg": "indices", "into": "spatial(self, 3)"}]}),
    ("pool", ["max_unpool2d", "max_unpool3d"], ["IDX", "ALLOC"],
     {"validator": "max_unpool", "index_inputs": [{"arg": "indices", "into": "prod(output_size)"}]}),
    ("upsample", [n + ".vec" for n in UPSAMPLE] + UPSAMPLE + [n + "_backward" for n in UPSAMPLE] +
     [n + "_backward.vec" for n in UPSAMPLE], ["ALLOC"], {"validator": "alloc"}),
    ("spatial", ["grid_sampler_2d_backward", "grid_sampler_3d", "grid_sampler_3d_backward",
                 "affine_grid_generator", "im2col", "pixel_shuffle", "pixel_unshuffle", "channel_shuffle",
                 "roll", "rot90"], ["ALLOC"], {"validator": "alloc"}),
    ("index", ["index_add", "index_copy", "index_fill.int_Scalar", "index_fill.int_Tensor"], ["IDX"],
     {"validator": "dim_index", "index_inputs": [{"arg": "index", "into": "size(self, dim)"}]}),
    ("index", ["masked_fill.Scalar", "masked_fill.Tensor", "searchsorted.Tensor", "bucketize.Tensor",
               "nonzero_static"], [], {}),
    ("index", ["take"], ["IDX"],
     {"validator": "take", "index_inputs": [{"arg": "index", "into": "numel(self)", "negative": True}]}),
    ("index", ["_unsafe_index.Tensor", "_unsafe_index_put"], ["IDX"],
     {"validator": "advanced_index", "index_inputs": [{"arg": "indices", "into": "advanced(self)", "negative": True}]}),
    ("index", ["_embedding_bag_backward", "_embedding_bag_dense_backward"], ["IDX", "RESTRICTED"],
     {"validator": "embedding_bag_backward", "index_inputs": [{"arg": "indices", "into": "num_weights"}]}),
    ("index", ["embedding_renorm"], ["IDX", "FUNC"],
     {"validator": "embedding_renorm", "functional_of": "aten.embedding_renorm_.default",
      "index_inputs": [{"arg": "indices", "into": "size(self, 0)"}]}),
    ("reduction", ["sum", "nansum", "std.correction", "var_mean.correction", "std_mean.correction", "cumprod",
                   "cummax", "cummin", "logcumsumexp", "linalg_vector_norm", "norm.ScalarOpt_dim", "all",
                   "all.dim", "all.dims", "aminmax", "median.dim", "nanmedian.dim", "mode", "kthvalue",
                   "max", "min", "median", "nanmedian"], [], {"cost": "reduce"}),
    ("elementwise", ["clamp_min", "clamp_max", "exp2", "erfc", "erfinv", "lgamma", "digamma", "polygamma",
                     "i0", "special_i0e", "special_i1", "special_i1e", "lerp.Scalar", "lerp.Tensor",
                     "addcmul", "addcdiv", "nan_to_num", "copysign.Tensor", "copysign.Scalar", "hypot",
                     "isfinite", "signbit", "fmax", "fmin", "xlogy.Tensor", "logit", "frac", "heaviside",
                     "sgn", "deg2rad", "rad2deg", "logaddexp", "logaddexp2", "isneginf", "isposinf",
                     "special_xlog1py", "special_entr", "sinc", "nextafter", "float_power.Tensor_Scalar",
                     "float_power.Tensor_Tensor", "float_power.Scalar", "trunc", "square", "relu6",
                     "clamp_min.Tensor", "clamp_max.Tensor", "true_divide.Tensor", "positive",
                     "rsub.Scalar", "rsub.Tensor"], [], {}),
    ("elementwise", ["floor_divide", "floor_divide.Scalar"], ["RESTRICTED"], {"validator": "int_divisor_nonzero"}),
    ("elementwise", ["bitwise_left_shift.Tensor", "bitwise_right_shift.Tensor", "bitwise_left_shift.Tensor_Scalar",
                     "bitwise_right_shift.Tensor_Scalar", "__lshift__.Scalar", "__rshift__.Scalar",
                     "__lshift__.Tensor", "__rshift__.Tensor"], ["RESTRICTED"], {"validator": "shift_amount"}),
    ("creation", ["zeros", "ones", "zeros_like", "ones_like", "eye", "eye.m", "linspace", "logspace",
                  "full_like", "new_zeros", "new_ones", "new_full", "arange", "arange.start", "scalar_tensor"],
     ["ALLOC"], {"validator": "alloc"}),
    ("creation", ["empty_like", "new_empty", "new_empty_strided"], ["UNINIT", "ALLOC"], {"validator": "alloc"}),
    ("shape", ["tril", "triu", "diag_embed", "narrow", "unbind.int", "stack", "transpose.int", "t", "reshape",
               "_unsafe_view", "unfold", "contiguous", "lift_fresh_copy", "detach", "tile", "movedim.int",
               "expand_as", "flatten.using_ints", "split.Tensor", "split_with_sizes", "chunk", "squeeze",
               "view_as", "select_backward", "slice_backward", "diagonal_backward", "diagonal_scatter",
               "as_strided_scatter", "unsqueeze_copy", "view_copy", "permute_copy", "expand_copy",
               "t_copy", "transpose_copy.int", "squeeze_copy.dims", "alias_copy", "slice_copy.Tensor",
               "select_copy.int", "detach_copy", "split_with_sizes_copy", "unbind_copy.int",
               "diagonal_copy", "unfold_copy", "_unsafe_view", "masked_fill_"], [], {}),
    ("shape", ["repeat_interleave.self_int"], ["ALLOC"], {"validator": "alloc"}),
    ("complex", ["view_as_real", "view_as_complex", "polar", "complex", "real", "imag", "conj_physical",
                 "_conj", "angle"], [], {"capability": "complex"}),
    ("rng", ["bernoulli.p", "bernoulli.default", "uniform", "normal_functional", "normal.Tensor_float",
             "normal.float_Tensor", "normal.Tensor_Tensor", "exponential", "randint", "randint.low",
             "rand_like", "randn_like", "randint_like", "randint_like.low_dtype", "native_dropout_backward",
             "normal.float_float", "bernoulli.Tensor"], ["RNG", "ALLOC"], {"validator": "alloc"}),
    ("rng", ["multinomial"], ["RNG", "RESTRICTED"], {"validator": "multinomial"}),
    ("optimizer", FOREACH, ["RESTRICTED"], {"validator": "foreach_lengths"}),
]

TIER2 = [
    ("linalg", ["linalg_cholesky_ex", "linalg_qr", "_linalg_svd", "_linalg_eigh", "_linalg_solve_ex",
                "linalg_solve_triangular", "linalg_inv_ex", "_linalg_det", "_linalg_slogdet",
                "linalg_lu_factor_ex", "linalg_lu_solve", "cholesky_solve", "triangular_solve",
                "linalg_pinv.atol_rtol_tensor", "linalg_matrix_exp"], ["RESTRICTED"],
     {"validator": "linalg", "capability": "linalg", "cost": "cubic"}),
    ("fft", ["_fft_c2c"], [], {"capability": "complex"}),
    ("fp8", ["_scaled_mm"], ["RESTRICTED"], {"validator": "scaled_mm", "capability": "fp8", "cost": "matmul"}),
    ("rnn", ["_thnn_fused_lstm_cell", "_thnn_fused_lstm_cell_backward_impl", "_thnn_fused_gru_cell",
             "_thnn_fused_gru_cell_backward"], [], {"capability": "rnn_fused"}),
    ("rnn", ["_cudnn_rnn", "_cudnn_rnn_backward"], ["RESTRICTED"],
     {"validator": "cudnn_rnn", "capability": "rnn_fused"}),
    ("ctc", ["_ctc_loss", "_ctc_loss_backward", "_ctc_loss.Tensor", "_ctc_loss_backward.Tensor"],
     ["IDX", "RESTRICTED"], {"validator": "ctc", "capability": "ctc"}),
    ("cdist", ["_cdist_backward", "_pdist_backward"], [], {"capability": "cdist_backward"}),
]

# reflop-native operators (§8.5). Schemas are written by hand; their
# implementations live in server/worker/reflop_worker/natives.py.
NATIVE = [
    {"name": "reflop.dataset_gather", "category": "native", "flags": ["IDX"], "validator": "dataset_gather",
     "schema": "reflop::dataset_gather(Tensor dataset, Tensor indices) -> Tensor",
     "index_inputs": [{"arg": "indices", "into": "size(dataset, 0)"}]},
    {"name": "reflop.bounds_check", "category": "native", "flags": ["IDX"], "validator": "bounds_check",
     "schema": "reflop::bounds_check(Tensor index, int size, int? sentinel=None) -> Tensor"},
    {"name": "reflop.nonzero_count", "category": "native", "flags": [],
     "schema": "reflop::nonzero_count(Tensor self) -> Tensor"},
    {"name": "reflop.mask_count", "category": "native", "flags": [],
     "schema": "reflop::mask_count(Tensor mask) -> Tensor"},
    {"name": "reflop.randperm_seeded", "category": "native", "flags": ["ALLOC"], "validator": "alloc",
     "schema": "reflop::randperm_seeded(int n, int seed, ScalarType dtype=long) -> Tensor"},
    {"name": "reflop.fused_adam", "category": "optimizer", "flags": ["FUNC", "RESTRICTED"], "validator": "fused_optim",
     "schema": "reflop::fused_adam(Tensor[] params, Tensor[] grads, Tensor[] exp_avgs, Tensor[] exp_avg_sqs, "
               "Tensor[] max_exp_avg_sqs, Tensor[] state_steps, float lr, float beta1, float beta2, "
               "float weight_decay, float eps, bool amsgrad, bool maximize, Tensor? grad_scale=None, "
               "Tensor? found_inf=None) -> (Tensor[], Tensor[], Tensor[], Tensor[], Tensor[])"},
    {"name": "reflop.fused_adamw", "category": "optimizer", "flags": ["FUNC", "RESTRICTED"], "validator": "fused_optim",
     "schema": "reflop::fused_adamw(Tensor[] params, Tensor[] grads, Tensor[] exp_avgs, Tensor[] exp_avg_sqs, "
               "Tensor[] max_exp_avg_sqs, Tensor[] state_steps, float lr, float beta1, float beta2, "
               "float weight_decay, float eps, bool amsgrad, bool maximize, Tensor? grad_scale=None, "
               "Tensor? found_inf=None) -> (Tensor[], Tensor[], Tensor[], Tensor[], Tensor[])"},
    {"name": "reflop.fused_sgd", "category": "optimizer", "flags": ["FUNC", "RESTRICTED"], "validator": "fused_optim",
     "schema": "reflop::fused_sgd(Tensor[] params, Tensor[] grads, Tensor[] momentum_buffers, float weight_decay, "
               "float momentum, float lr, float dampening, bool nesterov, bool maximize, bool is_first_step, "
               "Tensor? grad_scale=None, Tensor? found_inf=None) -> (Tensor[], Tensor[])"},
    {"name": "reflop.fused_adagrad", "category": "optimizer", "flags": ["FUNC", "RESTRICTED"], "validator": "fused_optim",
     "schema": "reflop::fused_adagrad(Tensor[] params, Tensor[] grads, Tensor[] state_sums, Tensor[] state_steps, "
               "float lr, float lr_decay, float weight_decay, float eps, bool maximize, Tensor? grad_scale=None, "
               "Tensor? found_inf=None) -> (Tensor[], Tensor[], Tensor[])"},
    {"name": "reflop.amp_unscale_and_check", "category": "amp", "flags": ["FUNC", "RESTRICTED"], "validator": "foreach_lengths",
     "schema": "reflop::amp_unscale_and_check(Tensor[] grads, Tensor found_inf, Tensor inv_scale) -> (Tensor[], Tensor)"},
    {"name": "reflop.amp_update_scale", "category": "amp", "flags": ["FUNC"],
     "schema": "reflop::amp_update_scale(Tensor scale, Tensor growth_tracker, Tensor found_inf, "
               "float growth_factor, float backoff_factor, int growth_interval) -> (Tensor, Tensor)"},
]

EXCLUDED_PREFIXES = ("_foreach_", "_fused_", "_amp_")


def overloads(packet) -> list[str]:
    return list(packet.overloads())


def resolve(name: str) -> list[str]:
    """Map a §8 name to fully qualified overload names that exist."""
    base, _, ov = name.partition(".")
    packet = getattr(torch.ops.aten, base, None)
    if packet is None:
        return []
    if ov == "*":
        return [f"aten.{base}.{o}" for o in overloads(packet) if not o.startswith("out") and o != "out"]
    ov = ov or "default"
    return [f"aten.{base}.{ov}"] if ov in overloads(packet) else []


def lookup(qual: str):
    _, base, ov = qual.split(".", 2)
    return getattr(getattr(torch.ops.aten, base), ov)


def type_str(t) -> str:
    return str(t)


def default_repr(v):
    if v is None:
        return None
    if isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        return [default_repr(x) for x in v]
    if isinstance(v, torch.dtype):
        return str(v).replace("torch.", "")
    if isinstance(v, torch.layout):
        return str(v).replace("torch.", "")
    if isinstance(v, torch.memory_format):
        return str(v).replace("torch.", "")
    if isinstance(v, torch.device):
        return str(v)
    return str(v)


def native_types(text: str) -> tuple[list[str], list[str]]:
    """Exact native types ("ScalarType?", "SymInt[]", "Tensor(a!)") via torchgen."""
    from torchgen.model import FunctionSchema

    body = re.sub(r"^[a-z_]+::", "", text)
    # torchgen splits arguments on ", "; list defaults like [1, 1] must not contain it.
    body = re.sub(r"=\[([^\]]*)\]", lambda m: "=[" + m.group(1).replace(" ", "") + "]", body)
    fs = FunctionSchema.parse(body)
    return [str(a.type) for a in fs.arguments.flat_all], [str(r.type) for r in fs.returns]


def schema_of(op) -> dict:
    s = op._schema
    args = []
    atypes, rtypes = native_types(str(s))
    for a, t in zip(s.arguments, atypes, strict=True):
        d = {"name": a.name, "type": t}
        if a.kwarg_only:
            d["kwarg_only"] = True
        if a.has_default_value():
            d["default"] = default_repr(a.default_value)
        if a.alias_info is not None and a.alias_info.is_write:
            d["mutable"] = True
        args.append(d)
    rets = [{"type": t} for t in rtypes]
    return {"args": args, "returns": rets, "text": str(s)}


def is_mutating(op) -> bool:
    return any(a.alias_info is not None and a.alias_info.is_write for a in op._schema.arguments)


def main() -> None:
    entries: list[dict] = []
    seen: set[str] = set()
    census = {"torch": str(torch.__version__), "core_not_in_plan": [], "plan_not_found": [],
              "plan_not_core_tier0": [], "mutating_dropped": []}
    core = sorted(
        f"aten.{n}.{o}"
        for n in dir(torch.ops.aten)
        if not n.startswith("__") or n in ("__lshift__", "__rshift__")
        for p in [getattr(torch.ops.aten, n, None)]
        if isinstance(p, torch._ops.OpOverloadPacket)
        for o in p.overloads()
        if torch.Tag.core in getattr(p, o).tags
    )
    next_id = [1]

    def add(tier: int, group) -> None:
        cat, names, flags, extra = group
        for nm in names:
            quals = resolve(nm)
            if not quals:
                census["plan_not_found"].append(f"tier{tier}:{nm}")
                continue
            for q in quals:
                if q in seen:
                    continue
                op = lookup(q)
                fl = list(flags)
                if is_mutating(op) and "BANNED" not in fl and "FUNC" not in fl and "CLIENT" not in fl:
                    # The wire is functional (§11.2). Mutating overloads are
                    # never sent; the client maps them to functional forms.
                    census["mutating_dropped"].append(q)
                    continue
                seen.add(q)
                e = {"id": next_id[0], "name": q, "tier": tier, "category": cat}
                next_id[0] += 1
                if fl:
                    e["flags"] = fl
                for k, v in extra.items():
                    e[k] = v
                e["devices"] = ["cuda", "cpu"]
                e["schema"] = schema_of(op)
                if tier == 0 and q not in core:
                    census["plan_not_core_tier0"].append(q)
                entries.append(e)

    for g in TIER0:
        add(0, g)
    for q in core:
        if q not in seen:
            census["core_not_in_plan"].append(q)
    # Core ATen entries the plan did not list explicitly are still Tier 0:
    # "any PyTorch program that decomposes to Core ATen can run" (§8.1).
    add(0, ("core_extra", [q.split(".", 1)[1] for q in census["core_not_in_plan"]], [], {}))
    for g in TIER1:
        add(1, g)
    for g in TIER2:
        add(2, g)
    for n in NATIVE:
        e = {"id": next_id[0], "name": n["name"] + ".default", "tier": 1, "category": n["category"]}
        next_id[0] += 1
        for k in ("flags", "validator", "index_inputs"):
            if k in n and n[k]:
                e[k] = n[k]
        e["devices"] = ["cuda", "cpu"]
        e["schema"] = parse_native_schema(n["schema"])
        e["impl"] = "reflop"
        entries.append(e)

    # Every str-typed schema argument is an enum on the wire (§7.1).
    str_enums = {"side": ["left", "right"], "mode": ["reduced", "complete", "r"],
                 "driver": ["gesvd", "gesvdj", "gesvda"], "UPLO": ["L", "U"]}
    for e in entries:
        for a in e["schema"]["args"]:
            if a["type"].rstrip("?") == "str":
                ea = e.setdefault("enum_args", {})
                if a["name"] not in ea:
                    ea[a["name"]] = str_enums[a["name"]]
                if "ENUM" not in e.setdefault("flags", []):
                    e["flags"].append("ENUM")
    for e in entries:
        e.setdefault("cost", "elementwise")
        e["since"] = 1
    doc = {
        "opset_version": 1,
        "pinned_torch": str(torch.__version__).split("+")[0],
        "flags": ["IDX", "RNG", "DDS", "ALLOC", "ENUM", "UNINIT", "FUNC", "RESTRICTED", "CLIENT", "BANNED"],
        "ops": entries,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    header = ("# SPDX-License-Identifier: Apache-2.0\n"
              "# opset v1: the single source of truth for reFLOP's operator whitelist (invariant I10).\n"
              "# Bootstrapped by codegen/bootstrap_opset.py; edit by hand (procedure: PLAN.md §8.8).\n"
              "# Op ids are permanent and never reused.\n")
    OUT.write_text(header + yaml.safe_dump(doc, sort_keys=False, width=120))
    census["counts"] = {f"tier{t}": sum(1 for e in entries if e["tier"] == t) for t in (0, 1, 2)}
    census["core_total"] = len(core)
    CENSUS.write_text(json.dumps(census, indent=1) + "\n")
    print(f"opset: {len(entries)} entries, core={len(core)}", file=sys.stderr)
    for k in ("plan_not_found", "core_not_in_plan", "plan_not_core_tier0", "mutating_dropped"):
        print(f"  {k}: {len(census[k])}", file=sys.stderr)


def parse_native_schema(text: str) -> dict:
    """Parse a reflop:: schema with torch's own schema parser."""
    s = torch._C.parse_schema(text)
    atypes, rtypes = native_types(text)
    args = []
    for a, t in zip(s.arguments, atypes, strict=True):
        d = {"name": a.name, "type": t}
        if a.kwarg_only:
            d["kwarg_only"] = True
        if a.has_default_value():
            d["default"] = default_repr(a.default_value)
        args.append(d)
    return {"args": args, "returns": [{"type": t} for t in rtypes], "text": text}


if __name__ == "__main__":
    main()
