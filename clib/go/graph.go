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

// pending is a node whose arguments are still being pushed.
type pending struct {
	op   *opset.Op
	args []*pb.Arg
}

// builder keeps a stack of open nodes, so an argument can itself be built
// inline (C++17 evaluates g.op("outer") before its argument
// g.op("inner")...end()). Arguments go to the innermost open node; a node is
// appended when it ends, so inner nodes precede the nodes that use them.
type builder struct {
	g       *pb.Graph
	open    []*pending
	outputs []value
}

func (b *builder) top() *pending {
	if len(b.open) == 0 {
		return nil
	}
	return b.open[len(b.open)-1]
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
	op, ok := opset.ByName(name)
	if !ok {
		return fmt.Errorf("%q is not in opset v1 (names look like aten.mm.default)", name)
	}
	if op.Flags&(opset.CLIENT|opset.BANNED) != 0 {
		return fmt.Errorf("%s cannot be sent to a server", name)
	}
	b.open = append(b.open, &pending{op: op})
	return nil
}

func (b *builder) push(a *pb.Arg) error {
	p := b.top()
	if p == nil {
		return fmt.Errorf("no operator open: call reflop_op_begin first")
	}
	if len(p.args) >= len(p.op.Args) {
		return fmt.Errorf("%s takes %d arguments", p.op.Name, len(p.op.Args))
	}
	p.args = append(p.args, a)
	return nil
}

func (b *builder) enum(v string) error {
	p := b.top()
	if p == nil || len(p.args) >= len(p.op.Args) {
		return b.push(nil)
	}
	name := p.op.Args[len(p.args)].Name
	codes := p.op.Enums[name]
	for i, c := range codes {
		if c == v {
			return b.push(&pb.Arg{Value: &pb.Arg_EnumCode{EnumCode: uint32(i)}})
		}
	}
	return fmt.Errorf("%s: %q is not a valid value for %s (one of %s)", p.op.Name, v, name, strings.Join(codes, ", "))
}

func (b *builder) end() (value, error) {
	p := b.top()
	if p == nil {
		return value{}, fmt.Errorf("no operator open")
	}
	b.open = b.open[:len(b.open)-1]
	if len(p.args) != len(p.op.Args) {
		names := make([]string, len(p.op.Args))
		for i, a := range p.op.Args {
			names[i] = a.Type + " " + a.Name
		}
		return value{}, fmt.Errorf("%s takes %d arguments (%s), got %d", p.op.Name, len(p.op.Args), strings.Join(names, ", "), len(p.args))
	}
	b.g.Nodes = append(b.g.Nodes, &pb.Node{OpId: p.op.ID, Args: p.args})
	return value{int32(len(b.g.Nodes) - 1), 0}, nil
}

func (b *builder) output(v value, meta *pb.TensorMeta) {
	b.g.Outputs = append(b.g.Outputs, ref(v))
	b.g.OutputMeta = append(b.g.OutputMeta, meta)
}
