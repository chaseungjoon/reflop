// SPDX-License-Identifier: Apache-2.0

package main

import (
	"strings"
	"testing"

	pb "github.com/chaseungjoon/reflop/pkg/reflopv1"
)

func tensorArg(v value) *pb.Arg { return &pb.Arg{Value: &pb.Arg_Tensor{Tensor: ref(v)}} }

func mustNoErr(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}

// The C++ README pattern: addmm opens first, then t(W) is built inline as its
// third argument (C++17 evaluates g.op("addmm") before the argument).
func TestNestedOps(t *testing.T) {
	b := newBuilder()
	meta := &pb.TensorMeta{}
	x, w, bias := b.input(meta), b.input(meta), b.input(meta)

	mustNoErr(t, b.begin("aten.addmm.default"))
	mustNoErr(t, b.push(tensorArg(bias)))
	mustNoErr(t, b.push(tensorArg(x)))
	mustNoErr(t, b.begin("aten.t.default"))
	mustNoErr(t, b.push(tensorArg(w)))
	wt, err := b.end()
	mustNoErr(t, err)
	mustNoErr(t, b.push(tensorArg(wt)))
	mustNoErr(t, b.push(&pb.Arg{Value: &pb.Arg_I{I: 1}}))
	mustNoErr(t, b.push(&pb.Arg{Value: &pb.Arg_I{I: 1}}))
	h, err := b.end()
	mustNoErr(t, err)

	if len(b.g.Nodes) != 2 || wt.node != 0 || h.node != 1 {
		t.Fatalf("nodes %d, t at %d, addmm at %d; want t (0) before addmm (1)", len(b.g.Nodes), wt.node, h.node)
	}
	if got := b.g.Nodes[1].Args[2].GetTensor().GetNodeOutput().GetNode(); got != 0 {
		t.Fatalf("addmm's mat2 refers to node %d, want 0", got)
	}
	if len(b.g.Nodes[0].Args) != 1 || len(b.g.Nodes[1].Args) != 5 {
		t.Fatalf("args: t %d, addmm %d", len(b.g.Nodes[0].Args), len(b.g.Nodes[1].Args))
	}
	if b.top() != nil {
		t.Fatal("an op is still open")
	}
}

func TestEndWrongCountPopsOp(t *testing.T) {
	b := newBuilder()
	x := b.input(&pb.TensorMeta{})
	mustNoErr(t, b.begin("aten.relu.default"))
	mustNoErr(t, b.begin("aten.mm.default"))
	mustNoErr(t, b.push(tensorArg(x)))
	if _, err := b.end(); err == nil || !strings.Contains(err.Error(), "aten.mm.default takes 2 arguments") {
		t.Fatalf("err = %v", err)
	}
	// relu is still open and takes the next argument.
	mustNoErr(t, b.push(tensorArg(x)))
	if _, err := b.end(); err != nil {
		t.Fatal(err)
	}
	if _, err := b.end(); err == nil {
		t.Fatal("end with nothing open succeeded")
	}
}
