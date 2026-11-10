# SPDX-License-Identifier: Apache-2.0
"""Build raw FlopIR graphs and drive a worker Session directly."""

from __future__ import annotations

import torch

from reflop_worker._pb import compute_pb2 as cpb
from reflop_worker._pb import graph_pb2 as gpb
from reflop_worker._pb import worker_pb2 as wpb
from reflop_worker.core import Session
from reflop_worker.dtypes import TORCH_TO_PB
from reflop_worker.ops_generated import OPS

NAME = {o.name: i for i, o in OPS.items()}


def make_session(device: str = "cpu", **limit_overrides) -> Session:
    from reflop.connection import default_limits
    from reflop_worker._pb import capabilities_pb2 as kpb

    init = wpb.WorkerInit(seed=0)
    init.device.kind = gpb.DEVICE_CPU if device == "cpu" else gpb.DEVICE_CUDA
    lim = default_limits(kpb)
    for k, v in limit_overrides.items():
        setattr(lim, k, v)
    init.limits.CopyFrom(lim)
    return Session(init)


def meta(dt, shape):
    return gpb.TensorMeta(dtype=TORCH_TO_PB[dt], shape=list(shape))


def tin(i):
    return gpb.Arg(tensor=gpb.ValueRef(graph_input=i))


def tnode(n, k=0):
    return gpb.Arg(tensor=gpb.ValueRef(node_output=gpb.NodeOutput(node=n, output=k)))


def I(v):  # noqa: E743
    return gpb.Arg(i=v)


def F(v):
    return gpb.Arg(f=v)


def B(v):
    return gpb.Arg(b=v)


def INTS(*v):
    return gpb.Arg(ints=gpb.Int64List(values=list(v)))


NONE = gpb.Arg(none=True)


def graph(sess, inputs, nodes, out_meta, outputs=None):
    g = gpb.Graph(opset_version=1, device=sess.gdevice)
    for dt, shape in inputs:
        g.inputs.append(gpb.GraphInput(kind=gpb.GraphInput.KIND_TENSOR, meta=meta(dt, shape)))
    for name, args in nodes:
        g.nodes.append(gpb.Node(op_id=NAME[name], args=args))
    outs = outputs if outputs is not None else [(len(nodes) - 1, 0)]
    for n, k in outs:
        g.outputs.append(gpb.ValueRef(node_output=gpb.NodeOutput(node=n, output=k)))
    for dt, shape in out_meta:
        g.output_meta.append(meta(dt, shape))
    return g


def tensor_binding(t: torch.Tensor) -> cpb.Binding:
    from reflop_worker.safetensors import tensor_bytes

    b = cpb.Binding()
    b.inline.meta.CopyFrom(meta(t.dtype, t.shape))
    b.inline.data = tensor_bytes(t)
    return b


def call(sess, env: cpb.Envelope) -> list[cpb.Envelope]:
    out = []
    sess.handle(env, out.append)
    return out


def register(sess, g):
    r = call(sess, cpb.Envelope(request_id=1, register_graph=cpb.RegisterGraph(graph=g)))[0]
    return r


def run(sess, g, inputs: list[torch.Tensor], ret=True):
    r = register(sess, g)
    if r.WhichOneof("body") == "error":
        return r.error
    ex = cpb.Execute(graph_hash=r.graph_registered.graph_hash)
    for t in inputs:
        ex.bindings.append(tensor_binding(t))
    for _ in g.outputs:
        ex.outputs.append(cpb.OutputSpec(kind=cpb.OutputSpec.KIND_RETURN if ret else cpb.OutputSpec.KIND_DISCARD))
    e = call(sess, cpb.Envelope(request_id=2, execute=ex))[0]
    if e.WhichOneof("body") == "error":
        return e.error
    return e.execute_result


def decode(td):
    from reflop_worker.execute import tensor_from_bytes
    from reflop_worker.dtypes import PB_TO_TORCH

    return tensor_from_bytes(td.data, PB_TO_TORCH[td.meta.dtype], tuple(td.meta.shape), "cpu")
