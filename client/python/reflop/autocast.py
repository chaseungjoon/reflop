# SPDX-License-Identifier: Apache-2.0
"""torch.autocast("reflop") support.

PyTorch only ships autocast kernels for its built-in backends, so reflop
registers AutocastPrivateUse1 kernels that mirror the CUDA policy: matmul-
like ops run in the autocast dtype, numerically sensitive ops in float32,
everything else falls through unchanged.
"""

from __future__ import annotations

import torch

# Mirrors aten/src/ATen/autocast_mode.cpp (CUDA lower_precision_fp list).
LOWER = [
    "_convolution.deprecated", "_convolution", "conv1d", "conv2d", "conv3d", "conv_tbc", "conv_transpose1d",
    "conv_transpose2d.input", "conv_transpose3d.input", "convolution", "prelu", "addmm", "addmv", "addr",
    "matmul", "einsum", "mm", "mv", "linalg_vecdot", "linear", "addbmm", "baddbmm", "bmm", "chain_matmul",
    "linalg_multi_dot", "_thnn_fused_lstm_cell", "_thnn_fused_gru_cell", "lstm_cell", "gru_cell", "rnn_tanh_cell",
    "rnn_relu_cell", "_scaled_dot_product_flash_attention", "scaled_dot_product_attention",
]
# Mirrors the CUDA fp32 list.
FP32 = [
    "acos", "asin", "cosh", "erfinv", "exp", "expm1", "log", "log10", "log2", "log1p", "reciprocal", "rsqrt",
    "sinh", "tan", "pow.Tensor_Scalar", "pow.Tensor_Tensor", "pow.Scalar", "softplus", "layer_norm",
    "native_layer_norm", "group_norm", "frobenius_norm.dim", "nuclear_norm", "nuclear_norm.dim", "cosine_similarity",
    "poisson_nll_loss", "cosine_embedding_loss", "nll_loss", "nll_loss2d", "hinge_embedding_loss", "kl_div",
    "l1_loss", "smooth_l1_loss", "huber_loss", "mse_loss", "margin_ranking_loss", "multilabel_margin_loss",
    "soft_margin_loss", "triplet_margin_loss", "multi_margin_loss", "binary_cross_entropy_with_logits", "dist",
    "pdist", "cdist", "renorm", "logsumexp", "upsample_nearest1d", "_upsample_nearest_exact1d",
    "upsample_nearest2d", "_upsample_nearest_exact2d", "upsample_nearest3d", "_upsample_nearest_exact3d",
    "upsample_linear1d", "upsample_bilinear2d", "_upsample_bilinear2d_aa", "upsample_trilinear3d",
    "upsample_bicubic2d", "_upsample_bicubic2d_aa", "cross_entropy_loss", "norm.Scalar", "norm.ScalarOpt_dim",
    "norm.ScalarOpt_dtype", "norm.names_ScalarOpt_dim", "softmax.int", "log_softmax.int",
]
_lib = None
_fb = None
KEY = "AutocastPrivateUse1"
_EXCLUDE = None


def _cast(x, dtype):
    if isinstance(x, torch.Tensor) and x.is_floating_point() and x.dtype != torch.float64:
        return x.to(dtype)
    if isinstance(x, (list, tuple)):
        return type(x)(_cast(y, dtype) for y in x)
    return x


def _kernel(op, mode):
    def k(*args, **kwargs):
        dtype = torch.get_autocast_dtype("reflop") if mode == "lower" else torch.float32
        with torch._C._ExcludeDispatchKeyGuard(_EXCLUDE):
            return op(*_cast(args, dtype), **{n: _cast(v, dtype) for n, v in kwargs.items()})
    return k


def register() -> None:
    global _lib, _fb, _EXCLUDE
    if _lib is not None:
        return
    _EXCLUDE = torch._C.DispatchKeySet(torch._C.DispatchKey.AutocastPrivateUse1)
    _fb = torch.library.Library("_", "IMPL")
    _fb.fallback(torch.library.fallthrough_kernel, dispatch_key=KEY)
    _lib = torch.library.Library("aten", "IMPL")
    for names, mode in ((LOWER, "lower"), (FP32, "fp32")):
        for n in names:
            base, _, ov = n.partition(".")
            pkt = getattr(torch.ops.aten, base, None)
            if pkt is None:
                continue
            ov = ov or "default"
            if ov not in pkt.overloads():
                continue
            try:
                _lib.impl(n if "." in n else base, _kernel(getattr(pkt, ov), mode), KEY)
            except RuntimeError:
                pass  # already registered by another backend (not expected)
