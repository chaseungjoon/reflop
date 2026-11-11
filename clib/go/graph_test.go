// SPDX-License-Identifier: Apache-2.0

package main

import (
	"os"
	"path/filepath"
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

// SEC_SWEEP L2/M3: sizes come from shape x dtype, with overflow checks.
func TestTensorBytes(t *testing.T) {
	for _, c := range []struct {
		m    *pb.TensorMeta
		want uint64
		ok   bool
	}{
		{&pb.TensorMeta{Dtype: pb.DType_DTYPE_FLOAT32, Shape: []int64{2, 3}}, 24, true},
		{&pb.TensorMeta{Dtype: pb.DType_DTYPE_COMPLEX128, Shape: []int64{4}}, 64, true},
		{&pb.TensorMeta{Dtype: pb.DType_DTYPE_BOOL}, 1, true},
		{&pb.TensorMeta{Dtype: pb.DType_DTYPE_FLOAT32, Shape: []int64{0, 1 << 40}}, 0, true},
		{&pb.TensorMeta{Dtype: pb.DType_DTYPE_FLOAT32, Shape: []int64{-1}}, 0, false},
		{&pb.TensorMeta{Dtype: pb.DType_DTYPE_FLOAT64, Shape: []int64{1 << 31, 1 << 31, 4}}, 0, false},
		{&pb.TensorMeta{Dtype: pb.DType(99), Shape: []int64{1}}, 0, false},
	} {
		got, err := tensorBytes(c.m)
		if (err == nil) != c.ok || (c.ok && got != c.want) {
			t.Errorf("%v: got %d, %v", c.m, got, err)
		}
	}
}

// SEC_SWEEP L6: pinned host keys become a reflop-owned known_hosts file and
// options that override the user's ssh_config.
func TestPinnedHostArgs(t *testing.T) {
	t.Setenv("REFLOP_CLIENT_HOME", t.TempDir()+"/a b%c")
	fp := strings.Repeat("ab", 32)
	key := "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAILs7graug4rm/Iz4hvKqazO5MLF8HcSzVedtFmkYjbTH"
	args, err := pinnedHostArgs(&serverEntry{Fingerprint: fp, SSHHostKeys: []string{key}})
	if err != nil {
		t.Fatal(err)
	}
	joined := strings.Join(args, " ")
	for _, want := range []string{"HostKeyAlias=reflop-" + fp, "GlobalKnownHostsFile=/dev/null", `/a b%%c/ssh_known_hosts/` + fp + `"`} {
		if !strings.Contains(joined, want) {
			t.Errorf("args %q lack %q", joined, want)
		}
	}
	b, err := os.ReadFile(filepath.Join(configDir(), "ssh_known_hosts", fp))
	if err != nil || string(b) != "reflop-"+fp+" "+key+"\n" {
		t.Fatalf("known_hosts %q, %v", b, err)
	}
	if a, err := pinnedHostArgs(&serverEntry{Fingerprint: fp}); err != nil || a != nil {
		t.Fatalf("no pinned keys: %v %v", a, err)
	}
	if _, err := pinnedHostArgs(&serverEntry{Fingerprint: fp, SSHHostKeys: []string{"ssh-ed25519 AAAAB3NzaC1yc2EAAAADAQAB"}}); err == nil {
		t.Fatal("key whose type does not match its data accepted")
	}
}
