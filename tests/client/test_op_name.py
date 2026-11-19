# SPDX-License-Identifier: Apache-2.0
"""The dispatch path's cached operator names match str(func)."""

import torch
from reflop.tensor import _OP_NAMES, _op_name


def test_op_name_matches_str_and_is_cached():
    ops = [torch.ops.aten.add.Tensor, torch.ops.aten.mul_.Tensor, torch.ops.aten.view.default,
           torch.ops.aten._to_copy.default]
    for op in ops:
        assert _op_name(op) == str(op)
        assert _OP_NAMES[op] == str(op)
    assert _op_name(torch.ops.aten.add.Tensor) is _op_name(torch.ops.aten.add.Tensor)
