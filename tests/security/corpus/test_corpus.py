# SPDX-License-Identifier: Apache-2.0
"""Security corpus (PLAN.md §17.3): every case must be rejected with a
structured error, and the worker must keep serving afterwards."""

from __future__ import annotations

import pytest
import torch
from helpers import (
    INTS,
    NAME,
    NONE,
    B,
    F,
    I,
    call,
    decode,
    graph,
    make_session,
    meta,
    register,
    run,
    tensor_binding,
    tin,
    tnode,
)
from reflop_worker._pb import compute_pb2 as cpb
from reflop_worker._pb import graph_pb2 as gpb

E = cpb  # error codes


@pytest.fixture(scope="module", params=["cpu", "cuda"])
def sess(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA")
    return make_session(request.param)


def ok_after(sess):
    """The worker still works after a rejection."""
    g = graph(sess, [(torch.float32, (2,))], [("aten.neg.default", [tin(0)])], [(torch.float32, (2,))])
    r = run(sess, g, [torch.ones(2)])
    assert isinstance(r, cpb.ExecuteResult), r
    assert decode(r.returned[0]).tolist() == [-1.0, -1.0]


def expect(err, code, stage=None):
    assert isinstance(err, cpb.Error), f"expected rejection, got {type(err).__name__}"
    assert err.code == code, f"code {cpb.ErrorCode.Name(err.code)}: {err.message}"
    if stage is not None:
        assert err.stage == stage, cpb.ValidationStage.Name(err.stage)


# ---------------------------------------------------------------- structure


def test_unknown_op_id(sess):
    g = graph(sess, [(torch.float32, (2,))], [("aten.neg.default", [tin(0)])], [(torch.float32, (2,))])
    g.nodes[0].op_id = 999999
    expect(register(sess, g).error, E.ERROR_INVALID_GRAPH, E.STAGE_V2_STRUCTURE)
    ok_after(sess)


@pytest.mark.parametrize("name", ["aten.resize_.default", "aten._local_scalar_dense.default", "aten.sym_size.int"])
def test_banned_and_client_ops(sess, name):
    if name not in NAME:
        pytest.skip(f"{name} not in opset")
    g = graph(sess, [(torch.float32, (2,))], [("aten.neg.default", [tin(0)])], [(torch.float32, (2,))])
    g.nodes[0].op_id = NAME[name]
    expect(register(sess, g).error, E.ERROR_INVALID_GRAPH)


def test_forward_reference(sess):
    g = graph(sess, [(torch.float32, (2,))], [("aten.neg.default", [tnode(0)])], [(torch.float32, (2,))])
    expect(register(sess, g).error, E.ERROR_INVALID_GRAPH, E.STAGE_V2_STRUCTURE)


def test_cycle_impossible(sess):
    g = graph(sess, [(torch.float32, (2,))], [("aten.neg.default", [tin(0)]), ("aten.neg.default", [tnode(1)])],
              [(torch.float32, (2,))])
    expect(register(sess, g).error, E.ERROR_INVALID_GRAPH)


def test_out_of_range_output_index(sess):
    g = graph(sess, [(torch.float32, (2,))], [("aten.neg.default", [tin(0)])], [(torch.float32, (2,))], outputs=[(0, 5)])
    expect(register(sess, g).error, E.ERROR_INVALID_GRAPH)


def test_missing_input(sess):
    g = graph(sess, [(torch.float32, (2,))], [("aten.neg.default", [tin(7)])], [(torch.float32, (2,))])
    expect(register(sess, g).error, E.ERROR_INVALID_GRAPH)


def test_node_limit():
    s = make_session("cpu", max_nodes=10)
    nodes = [("aten.neg.default", [tin(0)])] + [("aten.neg.default", [tnode(i)]) for i in range(10)]
    g = graph(s, [(torch.float32, (2,))], nodes, [(torch.float32, (2,))])
    expect(register(s, g).error, E.ERROR_RESOURCE_LIMIT)


def test_wrong_arity(sess):
    g = graph(sess, [(torch.float32, (2,))], [("aten.add.Tensor", [tin(0), tin(0)])], [(torch.float32, (2,))])
    expect(register(sess, g).error, E.ERROR_INVALID_GRAPH)


def test_wrong_device(sess):
    g = graph(sess, [(torch.float32, (2,))], [("aten.neg.default", [tin(0)])], [(torch.float32, (2,))])
    g.device.kind = gpb.DEVICE_CUDA if g.device.kind == gpb.DEVICE_CPU else gpb.DEVICE_CPU
    expect(register(sess, g).error, E.ERROR_INVALID_GRAPH)


def test_unsupported_opset(sess):
    g = graph(sess, [(torch.float32, (2,))], [("aten.neg.default", [tin(0)])], [(torch.float32, (2,))])
    g.opset_version = 2
    expect(register(sess, g).error, E.ERROR_UNSUPPORTED_VERSION)


# ---------------------------------------------------------------- schema


def test_bad_dtype(sess):
    g = graph(sess, [(torch.float32, (2,))], [("aten.neg.default", [tin(0)])], [(torch.float32, (2,))])
    g.inputs[0].meta.dtype = 99
    expect(register(sess, g).error, E.ERROR_INVALID_GRAPH)


def test_numel_overflow(sess):
    g = graph(sess, [(torch.float32, (1 << 40, 1 << 40))], [("aten.neg.default", [tin(0)])],
              [(torch.float32, (1 << 40, 1 << 40))])
    expect(register(sess, g).error, E.ERROR_RESOURCE_LIMIT)


def test_negative_size(sess):
    g = graph(sess, [(torch.float32, (2,))], [("aten.neg.default", [tin(0)])], [(torch.float32, (2,))])
    g.inputs[0].meta.shape[0] = -3
    expect(register(sess, g).error, E.ERROR_RESOURCE_LIMIT)


def test_too_many_dims(sess):
    shape = (1,) * 40
    g = graph(sess, [(torch.float32, shape)], [("aten.neg.default", [tin(0)])], [(torch.float32, shape)])
    expect(register(sess, g).error, E.ERROR_RESOURCE_LIMIT)


def test_dim_out_of_range(sess):
    g = graph(sess, [(torch.float32, (2, 3))], [("aten._softmax.default", [tin(0), I(99), B(False)])],
              [(torch.float32, (2, 3))])
    err = register(sess, g).error
    assert err.code in (E.ERROR_SHAPE_MISMATCH, E.ERROR_INVALID_GRAPH)


def test_unknown_enum_code(sess):
    g = graph(sess, [(torch.float32, (2,))], [("aten.gelu.default", [tin(0), gpb.Arg(enum_code=7)])],
              [(torch.float32, (2,))])
    expect(register(sess, g).error, E.ERROR_INVALID_GRAPH, E.STAGE_V3_SCHEMA)


def test_string_like_payload_in_numeric_field(sess):
    g = graph(sess, [(torch.float32, (2,))], [("aten.leaky_relu.default", [tin(0), gpb.Arg(ints=gpb.Int64List(values=[1]))])],
              [(torch.float32, (2,))])
    expect(register(sess, g).error, E.ERROR_INVALID_GRAPH, E.STAGE_V3_SCHEMA)


def test_declared_output_mismatch(sess):
    g = graph(sess, [(torch.float32, (2,))], [("aten.neg.default", [tin(0)])], [(torch.float32, (3,))])
    expect(register(sess, g).error, E.ERROR_SHAPE_MISMATCH, E.STAGE_V4_META)


def test_generator_must_be_none(sess):
    if "aten.rand.default" not in NAME:
        pytest.skip()
    g = graph(sess, [], [("aten.rand.default", [INTS(2), NONE, NONE, gpb.Arg(device=sess.gdevice), NONE])],
              [(torch.float32, (2,))])
    r = register(sess, g)
    assert r.WhichOneof("body") == "graph_registered"


# ---------------------------------------------------------------- index (IDX)

BAD_INDEX = [-1, 3, (1 << 63) - 1, -(1 << 63)]


@pytest.mark.parametrize("bad", BAD_INDEX)
def test_index_select_oob(sess, bad):
    g = graph(sess, [(torch.float32, (3, 2)), (torch.int64, (1,))],
              [("aten.index_select.default", [tin(0), I(0), tin(1)])], [(torch.float32, (1, 2))])
    err = run(sess, g, [torch.ones(3, 2), torch.tensor([bad])])
    expect(err, E.ERROR_INDEX_OUT_OF_BOUNDS, E.STAGE_E2_RUNTIME)
    ok_after(sess)


@pytest.mark.parametrize("bad", BAD_INDEX)
def test_embedding_oob(sess, bad):
    g = graph(sess, [(torch.float32, (3, 4)), (torch.int64, (2,))],
              [("aten.embedding.default", [tin(0), tin(1), I(-1), B(False), B(False)])], [(torch.float32, (2, 4))])
    expect(run(sess, g, [torch.ones(3, 4), torch.tensor([0, bad])]), E.ERROR_INDEX_OUT_OF_BOUNDS)


@pytest.mark.parametrize("bad", BAD_INDEX)
def test_gather_oob(sess, bad):
    g = graph(sess, [(torch.float32, (3,)), (torch.int64, (2,))],
              [("aten.gather.default", [tin(0), I(0), tin(1), B(False)])], [(torch.float32, (2,))])
    expect(run(sess, g, [torch.ones(3), torch.tensor([0, bad])]), E.ERROR_INDEX_OUT_OF_BOUNDS)


@pytest.mark.parametrize("bad", BAD_INDEX)
def test_scatter_add_oob(sess, bad):
    g = graph(sess, [(torch.float32, (3,)), (torch.int64, (2,)), (torch.float32, (2,))],
              [("aten.scatter_add.default", [tin(0), I(0), tin(1), tin(2)])], [(torch.float32, (3,))])
    expect(run(sess, g, [torch.ones(3), torch.tensor([0, bad]), torch.ones(2)]), E.ERROR_INDEX_OUT_OF_BOUNDS)


@pytest.mark.parametrize("bad", [3, -4, (1 << 63) - 1])
def test_advanced_index_oob(sess, bad):
    g = graph(sess, [(torch.float32, (3,)), (torch.int64, (2,))],
              [("aten.index.Tensor", [tin(0), gpb.Arg(tensors=gpb.TensorList(items=[gpb.ValueRef(graph_input=1)]))])],
              [(torch.float32, (2,))])
    expect(run(sess, g, [torch.ones(3), torch.tensor([0, bad])]), E.ERROR_INDEX_OUT_OF_BOUNDS)


def test_advanced_index_negative_in_range_ok(sess):
    g = graph(sess, [(torch.float32, (3,)), (torch.int64, (1,))],
              [("aten.index.Tensor", [tin(0), gpb.Arg(tensors=gpb.TensorList(items=[gpb.ValueRef(graph_input=1)]))])],
              [(torch.float32, (1,))])
    r = run(sess, g, [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([-1])])
    assert decode(r.returned[0]).tolist() == [3.0]


def test_bool_mask_index_rejected(sess):
    g = graph(sess, [(torch.float32, (3,)), (torch.bool, (3,))],
              [("aten.index.Tensor", [tin(0), gpb.Arg(tensors=gpb.TensorList(items=[gpb.ValueRef(graph_input=1)]))])],
              [(torch.float32, (2,))])
    err = register(sess, g).error
    assert err.code in (E.ERROR_INVALID_GRAPH, E.ERROR_SHAPE_MISMATCH)


def test_wrong_index_dtype(sess):
    g = graph(sess, [(torch.float32, (3, 2)), (torch.float32, (1,))],
              [("aten.index_select.default", [tin(0), I(0), tin(1)])], [(torch.float32, (1, 2))])
    err = register(sess, g).error
    assert err.code in (E.ERROR_INVALID_GRAPH, E.ERROR_SHAPE_MISMATCH)


def test_empty_index_ok(sess):
    g = graph(sess, [(torch.float32, (3, 2)), (torch.int64, (0,))],
              [("aten.index_select.default", [tin(0), I(0), tin(1)])], [(torch.float32, (0, 2))])
    assert isinstance(run(sess, g, [torch.ones(3, 2), torch.zeros(0, dtype=torch.int64)]), cpb.ExecuteResult)


@pytest.mark.parametrize("target,ok", [([0, 4], True), ([0, -100], True), ([0, 5], False), ([0, -1], False),
                                       ([0, -101], False)])
def test_nll_loss_sentinel(sess, target, ok):
    g = graph(sess, [(torch.float32, (2, 5)), (torch.int64, (2,))],
              [("aten.nll_loss_forward.default", [tin(0), tin(1), NONE, I(1), I(-100)])],
              [(torch.float32, ()), (torch.float32, ())], outputs=[(0, 0), (0, 1)])
    r = run(sess, g, [torch.randn(2, 5).log_softmax(-1), torch.tensor(target)])
    if ok:
        assert isinstance(r, cpb.ExecuteResult), r
    else:
        expect(r, E.ERROR_INDEX_OUT_OF_BOUNDS)


def test_embedding_padding_idx_does_not_widen_range(sess):
    g = graph(sess, [(torch.float32, (3, 4)), (torch.int64, (1,))],
              [("aten.embedding.default", [tin(0), tin(1), I(5), B(False), B(False)])], [(torch.float32, (1, 4))])
    expect(run(sess, g, [torch.ones(3, 4), torch.tensor([5])]), E.ERROR_INDEX_OUT_OF_BOUNDS)


def test_max_pool_backward_indices(sess):
    x = torch.randn(1, 1, 4, 4)
    out, idx = torch.ops.aten.max_pool2d_with_indices(x, [2, 2], [2, 2])
    g = graph(sess, [(torch.float32, (1, 1, 2, 2)), (torch.float32, (1, 1, 4, 4)), (torch.int64, (1, 1, 2, 2))],
              [("aten.max_pool2d_with_indices_backward.default",
                [tin(0), tin(1), INTS(2, 2), INTS(2, 2), INTS(0, 0), INTS(1, 1), B(False), tin(2)])],
              [(torch.float32, (1, 1, 4, 4))])
    assert isinstance(run(sess, g, [torch.ones_like(out), x, idx]), cpb.ExecuteResult)
    bad = idx.clone()
    bad[0, 0, 0, 0] = 16
    expect(run(sess, g, [torch.ones_like(out), x, bad]), E.ERROR_INDEX_OUT_OF_BOUNDS)


def test_take_oob(sess):
    g = graph(sess, [(torch.float32, (2, 2)), (torch.int64, (1,))], [("aten.take.default", [tin(0), tin(1)])],
              [(torch.float32, (1,))])
    expect(run(sess, g, [torch.ones(2, 2), torch.tensor([4])]), E.ERROR_INDEX_OUT_OF_BOUNDS)


def test_unsafe_index_is_checked(sess):
    g = graph(sess, [(torch.float32, (3,)), (torch.int64, (1,))],
              [("aten._unsafe_index.Tensor", [tin(0), gpb.Arg(tensors=gpb.TensorList(items=[gpb.ValueRef(graph_input=1)]))])],
              [(torch.float32, (1,))])
    expect(run(sess, g, [torch.ones(3), torch.tensor([3])]), E.ERROR_INDEX_OUT_OF_BOUNDS)


# ---------------------------------------------------------------- restricted


def test_as_strided_past_storage(sess):
    g = graph(sess, [(torch.float32, (4,))], [("aten.as_strided.default", [tin(0), INTS(2, 2), INTS(2, 1), I(1)])],
              [(torch.float32, (2, 2))])
    expect(register(sess, g).error, E.ERROR_INVALID_GRAPH)


def test_as_strided_negative_stride(sess):
    g = graph(sess, [(torch.float32, (4,))], [("aten.as_strided.default", [tin(0), INTS(2), INTS(-1), I(1)])],
              [(torch.float32, (2,))])
    expect(register(sess, g).error, E.ERROR_INVALID_GRAPH)


def test_as_strided_overflowing_stride_product(sess):
    g = graph(sess, [(torch.float32, (4,))],
              [("aten.as_strided.default", [tin(0), INTS(1 << 31, 1 << 31), INTS(1 << 40, 1 << 40), I(0)])],
              [(torch.float32, (1 << 31, 1 << 31))])
    err = register(sess, g).error
    assert err.code in (E.ERROR_INVALID_GRAPH, E.ERROR_RESOURCE_LIMIT)


def test_as_strided_in_bounds_ok(sess):
    g = graph(sess, [(torch.float32, (4,))], [("aten.as_strided.default", [tin(0), INTS(2), INTS(2), I(0)])],
              [(torch.float32, (2,))])
    r = run(sess, g, [torch.arange(4.0)])
    assert decode(r.returned[0]).tolist() == [0.0, 2.0]


def test_empty_strided_overlap(sess):
    g = graph(sess, [], [("aten.empty_strided.default", [INTS(2, 2), INTS(1, 1), NONE, NONE, gpb.Arg(device=sess.gdevice),
                                                         NONE])], [(torch.float32, (2, 2))])
    expect(register(sess, g).error, E.ERROR_INVALID_GRAPH)


def test_embedding_bag_offsets(sess):
    def g_(n_off):
        return graph(sess, [(torch.float32, (5, 3)), (torch.int64, (4,)), (torch.int64, (n_off,))],
                     [("aten._embedding_bag.default", [tin(0), tin(1), tin(2), B(False), I(0), B(False), NONE, B(False),
                                                       I(-1)])],
                     [(torch.float32, (n_off, 3))])
    w, idx = torch.ones(5, 3), torch.tensor([0, 1, 2, 3])
    assert isinstance(run(sess, g_(2), [w, idx, torch.tensor([0, 2])]), cpb.ExecuteResult)
    expect(run(sess, g_(2), [w, idx, torch.tensor([2, 0])]), E.ERROR_INDEX_OUT_OF_BOUNDS)
    expect(run(sess, g_(2), [w, idx, torch.tensor([1, 2])]), E.ERROR_INDEX_OUT_OF_BOUNDS)
    expect(run(sess, g_(2), [w, idx, torch.tensor([0, 9])]), E.ERROR_INDEX_OUT_OF_BOUNDS)


def test_masked_scatter_short_source(sess):
    g = graph(sess, [(torch.float32, (4,)), (torch.bool, (4,)), (torch.float32, (1,))],
              [("aten.masked_scatter.default", [tin(0), tin(1), tin(2)])], [(torch.float32, (4,))])
    expect(run(sess, g, [torch.zeros(4), torch.tensor([True, True, False, True]), torch.ones(1)]), E.ERROR_RUNTIME_CHECK)


@pytest.mark.parametrize("op", ["aten.remainder.Tensor", "aten.fmod.Tensor", "aten.floor_divide.default"])
def test_integer_division_by_zero(sess, op):
    g = graph(sess, [(torch.int64, (2,)), (torch.int64, (2,))], [(op, [tin(0), tin(1)])], [(torch.int64, (2,))])
    expect(run(sess, g, [torch.tensor([4, 5]), torch.tensor([2, 0])]), E.ERROR_RUNTIME_CHECK)


def test_div_mode_by_zero(sess):
    g = graph(sess, [(torch.int64, (2,)), (torch.int64, (2,))],
              [("aten.div.Tensor_mode", [tin(0), tin(1), gpb.Arg(enum_code=1)])], [(torch.int64, (2,))])
    expect(run(sess, g, [torch.tensor([4, 5]), torch.tensor([2, 0])]), E.ERROR_RUNTIME_CHECK)


def test_float_division_by_zero_is_fine(sess):
    g = graph(sess, [(torch.float32, (2,)), (torch.float32, (2,))], [("aten.div.Tensor", [tin(0), tin(1)])],
              [(torch.float32, (2,))])
    assert isinstance(run(sess, g, [torch.ones(2), torch.zeros(2)]), cpb.ExecuteResult)


@pytest.mark.parametrize("amount", [-1, 64, 1000])
def test_shift_amount(sess, amount):
    g = graph(sess, [(torch.int64, (1,)), (torch.int64, (1,))],
              [("aten.bitwise_left_shift.Tensor", [tin(0), tin(1)])], [(torch.int64, (1,))])
    expect(run(sess, g, [torch.tensor([1]), torch.tensor([amount])]), E.ERROR_RUNTIME_CHECK)


def test_multinomial_negative_probs(sess):
    g = graph(sess, [(torch.float32, (3,))], [("aten.multinomial.default", [tin(0), I(1), B(True), NONE])],
              [(torch.int64, (1,))])
    expect(run(sess, g, [torch.tensor([0.5, -1.0, 0.5])]), E.ERROR_RUNTIME_CHECK)


# ---------------------------------------------------------------- uninitialized memory (I13)


@pytest.mark.parametrize("op,args,out", [
    ("aten.empty.memory_format", lambda d: [INTS(64, 64), NONE, NONE, gpb.Arg(device=d), NONE, NONE], (64, 64)),
    ("aten.empty_strided.default", lambda d: [INTS(64, 64), INTS(64, 1), NONE, NONE, gpb.Arg(device=d), NONE], (64, 64)),
])
def test_empty_is_zero(sess, op, args, out):
    # Dirty the allocator first so stale memory would show.
    junk = graph(sess, [(torch.float32, out)], [("aten.mul.Tensor", [tin(0), F(7.0)])], [(torch.float32, out)])
    run(sess, junk, [torch.full(out, 3.0)])
    g = graph(sess, [], [(op, args(sess.gdevice))], [(torch.float32, out)])
    r = run(sess, g, [])
    assert decode(r.returned[0]).abs().sum().item() == 0.0


def test_empty_like_is_zero(sess):
    g = graph(sess, [(torch.float32, (32,))], [("aten.empty_like.default", [tin(0), NONE, NONE, NONE, NONE, NONE])],
              [(torch.float32, (32,))])
    r = run(sess, g, [torch.full((32,), 5.0)])
    assert decode(r.returned[0]).abs().sum().item() == 0.0


# ---------------------------------------------------------------- resources


def test_peak_larger_than_device():
    s = make_session("cpu")
    s.dev_info.total_memory_bytes = 1 << 20  # pretend the device has 1 MiB
    g = graph(s, [], [("aten.full.default", [INTS(1024, 1024), F(1.0), NONE, NONE, gpb.Arg(device=s.gdevice), NONE])],
              [(torch.float32, (1024, 1024))])
    expect(register(s, g).error, E.ERROR_OUT_OF_MEMORY, E.STAGE_V5_SAFETY)


def test_alloc_beyond_numel_limit(sess):
    g = graph(sess, [], [("aten.full.default", [INTS(1 << 21, 1 << 21), F(1.0), NONE, NONE, gpb.Arg(device=sess.gdevice),
                                               NONE])], [(torch.float32, (1 << 21, 1 << 21))])
    err = register(sess, g).error
    assert err.code == E.ERROR_RESOURCE_LIMIT


def test_inline_tensor_limit():
    s = make_session("cpu", max_inline_tensor_bytes=16)
    g = graph(s, [(torch.float32, (8,))], [("aten.neg.default", [tin(0)])], [(torch.float32, (8,))])
    expect(run(s, g, [torch.ones(8)]), E.ERROR_RESOURCE_LIMIT)


# ---------------------------------------------------------------- handles (I8)


def test_unknown_and_freed_handles(sess):
    g = graph(sess, [(torch.float32, (2,))], [("aten.neg.default", [tin(0)])], [(torch.float32, (2,))])
    r = register(sess, g)
    h = b"\x01" * 16
    ex = cpb.Execute(graph_hash=r.graph_registered.graph_hash, bindings=[cpb.Binding(handle=h)],
                     outputs=[cpb.OutputSpec(kind=cpb.OutputSpec.KIND_RETURN)])
    expect(call(sess, cpb.Envelope(request_id=3, execute=ex))[0].error, E.ERROR_UNKNOWN_HANDLE)
    up = cpb.UploadTensor(handle=h, meta=meta(torch.float32, (2,)), data=b"\0" * 8, final=True)
    assert call(sess, cpb.Envelope(request_id=4, upload_tensor=up))[0].WhichOneof("body") == "uploaded"
    assert call(sess, cpb.Envelope(request_id=5, execute=ex))[0].WhichOneof("body") == "execute_result"
    call(sess, cpb.Envelope(request_id=6, free=cpb.Free(handles=[h])))
    expect(call(sess, cpb.Envelope(request_id=7, execute=ex))[0].error, E.ERROR_UNKNOWN_HANDLE)


@pytest.mark.parametrize("h", [bytes(16), b"\xff" + bytes(15), b"short"])
def test_reserved_handles(sess, h):
    up = cpb.UploadTensor(handle=h, meta=meta(torch.float32, (1,)), data=b"\0" * 4, final=True)
    assert call(sess, cpb.Envelope(request_id=8, upload_tensor=up))[0].WhichOneof("body") == "error"


def test_cross_session_handles_do_not_exist():
    a, b = make_session("cpu"), make_session("cpu")
    h = b"\x02" * 16
    up = cpb.UploadTensor(handle=h, meta=meta(torch.float32, (1,)), data=b"\0" * 4, final=True)
    call(a, cpb.Envelope(request_id=1, upload_tensor=up))
    f = call(b, cpb.Envelope(request_id=2, fetch=cpb.Fetch(handles=[h])))
    expect(f[0].error, E.ERROR_UNKNOWN_HANDLE)


def test_rebind_cannot_change_meta(sess):
    h = b"\x03" * 16
    up = cpb.UploadTensor(handle=h, meta=meta(torch.float32, (2,)), data=b"\0" * 8, final=True)
    call(sess, cpb.Envelope(request_id=1, upload_tensor=up))
    g = graph(sess, [(torch.float32, (3,))], [("aten.neg.default", [tin(0)])], [(torch.float32, (3,))])
    r = register(sess, g)
    ex = cpb.Execute(graph_hash=r.graph_registered.graph_hash, bindings=[tensor_binding(torch.ones(3))],
                     outputs=[cpb.OutputSpec(kind=cpb.OutputSpec.KIND_REBIND, handle=h)])
    expect(call(sess, cpb.Envelope(request_id=2, execute=ex))[0].error, E.ERROR_BINDING)


def test_upload_checksum_and_order(sess):
    h = b"\x04" * 16
    m = meta(torch.float32, (4,))
    e = call(sess, cpb.Envelope(request_id=9, upload_tensor=cpb.UploadTensor(handle=h, meta=m, offset=8, data=b"\0" * 8)))
    assert e[0].WhichOneof("body") == "error"
    e = call(sess, cpb.Envelope(request_id=10, upload_tensor=cpb.UploadTensor(handle=h, meta=m, data=b"\0" * 16,
                                                                            final=True, sha256=b"\0" * 32)))
    assert e[0].WhichOneof("body") == "error"


def test_unknown_graph_hash(sess):
    ex = cpb.Execute(graph_hash=b"\x00" * 32)
    expect(call(sess, cpb.Envelope(request_id=1, execute=ex))[0].error, E.ERROR_UNKNOWN_GRAPH)


# ---------------------------------------------------------------- decode


def test_unknown_fields_rejected():
    from reflop_worker.wire import ProtocolError, check_no_unknown

    env = cpb.Envelope(request_id=1)
    raw = env.SerializeToString() + b"\xa8\x1f\x01"  # field 501, varint 1
    parsed = cpb.Envelope()
    parsed.ParseFromString(raw)
    with pytest.raises(ProtocolError):
        check_no_unknown(parsed)


def test_unknown_enum_rejected():
    from reflop_worker.wire import ProtocolError, check_enums

    g = gpb.Graph(device=gpb.Device(kind=77))
    with pytest.raises(ProtocolError):
        check_enums(g)


def test_server_hash_is_recomputed(sess):
    g = graph(sess, [(torch.float32, (2,))], [("aten.neg.default", [tin(0)])], [(torch.float32, (2,))])
    r = register(sess, g)
    from reflop_worker.execute import graph_hash

    assert r.graph_registered.graph_hash == graph_hash(g)
