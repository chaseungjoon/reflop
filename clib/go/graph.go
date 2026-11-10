// SPDX-License-Identifier: Apache-2.0

package main

import (
	"fmt"
	"strings"

	"github.com/chaseungjoon/reflop/pkg/opset"
	pb "github.com/chaseungjoon/reflop/pkg/reflopv1"
)

type value struct {
	node  int32
	index uint32
}

type builder struct {
	g       *pb.Graph
	op      *opset.Op
	args    []*pb.Arg
	outputs []value
}

func newBuilder() *builder { return &builder{g: &pb.Graph{OpsetVersion: opset.Version}} }

func (b *builder) input(meta *pb.TensorMeta) value {
	b.g.Inputs = append(b.g.Inputs, &pb.GraphInput{Kind: pb.GraphInput_KIND_TENSOR, Meta: meta})
	return value{-1, uint32(len(b.g.Inputs) - 1)}
}

func ref(v value) *pb.ValueRef {
	if v.node < 0 {
		return &pb.ValueRef{Ref: &pb.ValueRef_GraphInput{GraphInput: v.index}}
	}
	return &pb.ValueRef{Ref: &pb.ValueRef_NodeOutput{NodeOutput: &pb.NodeOutput{Node: uint32(v.node), Output: v.index}}}
}

func (b *builder) begin(name string) error {
	if b.op != nil {
		return fmt.Errorf("reflop_op_begin(%s) while %s is still open", name, b.op.Name)
	}
	op, ok := opset.ByName(name)
	if !ok {
		return fmt.Errorf("%q is not in opset v1 (names look like aten.mm.default)", name)
	}
	if op.Flags&(opset.CLIENT|opset.BANNED) != 0 {
		return fmt.Errorf("%s cannot be sent to a server", name)
	}
	b.op, b.args = op, nil
	return nil
}

func (b *builder) push(a *pb.Arg) error {
	if b.op == nil {
		return fmt.Errorf("no operator open: call reflop_op_begin first")
	}
	if len(b.args) >= len(b.op.Args) {
		return fmt.Errorf("%s takes %d arguments", b.op.Name, len(b.op.Args))
	}
	b.args = append(b.args, a)
	return nil
}

func (b *builder) enum(v string) error {
	if b.op == nil || len(b.args) >= len(b.op.Args) {
		return b.push(nil)
	}
	name := b.op.Args[len(b.args)].Name
	codes := b.op.Enums[name]
	for i, c := range codes {
		if c == v {
			return b.push(&pb.Arg{Value: &pb.Arg_EnumCode{EnumCode: uint32(i)}})
		}
	}
	return fmt.Errorf("%s: %q is not a valid value for %s (one of %s)", b.op.Name, v, name, strings.Join(codes, ", "))
}

func (b *builder) end() (value, error) {
	if b.op == nil {
		return value{}, fmt.Errorf("no operator open")
	}
	if len(b.args) != len(b.op.Args) {
		names := make([]string, len(b.op.Args))
		for i, a := range b.op.Args {
			names[i] = a.Type + " " + a.Name
		}
		err := fmt.Errorf("%s takes %d arguments (%s), got %d", b.op.Name, len(b.op.Args), strings.Join(names, ", "), len(b.args))
		b.op = nil
		return value{}, err
	}
	b.g.Nodes = append(b.g.Nodes, &pb.Node{OpId: b.op.ID, Args: b.args})
	b.op = nil
	return value{int32(len(b.g.Nodes) - 1), 0}, nil
}

func (b *builder) output(v value, meta *pb.TensorMeta) {
	b.g.Outputs = append(b.g.Outputs, ref(v))
	b.g.OutputMeta = append(b.g.OutputMeta, meta)
}
