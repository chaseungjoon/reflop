// SPDX-License-Identifier: Apache-2.0

package server

import (
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"
	"strings"
	"time"

	pb "github.com/chaseungjoon/reflop/server/reflopd/internal/pb/reflopv1"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/supervisor"
)

func opID(name string) uint32 {
	for id, o := range opTable {
		if o.name == name {
			return id
		}
	}
	return 0
}

func f32bytes(v []float32) []byte {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(x))
	}
	return b
}

// SelfTest runs a functional check (and, with security, external isolation
// checks) on a fresh worker per device.
func (s *Server) SelfTest(ctx context.Context, out io.Writer, security bool) error {
	for _, slot := range s.slots {
		sess := &Session{Device: slot, srv: s, numerics: s.allowedNumerics(nil), routes: map[uint64]*route{}}
		copy(sess.ID[:], []byte(fmt.Sprintf("selftest-%d-%d", slot.PB.Kind, time.Now().UnixNano()%1e6)))
		t0 := time.Now()
		w, err := s.spawnWorker(ctx, sess)
		if err != nil {
			return fmt.Errorf("%s: %w", slot.Name, err)
		}
		fmt.Fprintf(out, "  %s: worker ready in %s (%s)\n", slot.Name, time.Since(t0).Truncate(time.Millisecond), describeSandbox(w))
		err = s.matmulCheck(w, slot)
		if err == nil && security {
			err = isolationChecks(out, w)
		}
		w.Kill("self-test done")
		<-w.Done()
		os.RemoveAll(s.layout.SessionScratch(sess.hex()))
		if err != nil {
			return fmt.Errorf("%s: %w", slot.Name, err)
		}
		fmt.Fprintf(out, "  %s: matmul OK\n", slot.Name)
	}
	return nil
}

func (s *Server) matmulCheck(w *supervisor.Worker, slot *DeviceSlot) error {
	mm := opID("aten.mm.default")
	meta := &pb.TensorMeta{Dtype: pb.DType_DTYPE_FLOAT32, Shape: []int64{2, 2}}
	g := &pb.Graph{OpsetVersion: 1, Device: slot.PB,
		Inputs: []*pb.GraphInput{{Kind: pb.GraphInput_KIND_TENSOR, Meta: meta}, {Kind: pb.GraphInput_KIND_TENSOR, Meta: meta}},
		Nodes: []*pb.Node{{OpId: mm, Args: []*pb.Arg{
			{Value: &pb.Arg_Tensor{Tensor: &pb.ValueRef{Ref: &pb.ValueRef_GraphInput{GraphInput: 0}}}},
			{Value: &pb.Arg_Tensor{Tensor: &pb.ValueRef{Ref: &pb.ValueRef_GraphInput{GraphInput: 1}}}},
		}}},
		Outputs:    []*pb.ValueRef{{Ref: &pb.ValueRef_NodeOutput{NodeOutput: &pb.NodeOutput{Node: 0, Output: 0}}}},
		OutputMeta: []*pb.TensorMeta{meta},
	}
	call := func(env *pb.Envelope) (*pb.Envelope, error) {
		if err := w.Send(&pb.WorkerEnvelope{Body: &pb.WorkerEnvelope_Client{Client: env}}); err != nil {
			return nil, err
		}
		for {
			r, err := w.Recv()
			if err != nil {
				return nil, err
			}
			if c := r.GetClient(); c != nil && c.RequestId == env.RequestId {
				if e := c.GetError(); e != nil {
					return nil, fmt.Errorf("%s: %s", e.Code, e.Message)
				}
				return c, nil
			}
		}
	}
	reg, err := call(&pb.Envelope{RequestId: 1, Body: &pb.Envelope_RegisterGraph{RegisterGraph: &pb.RegisterGraph{Graph: g}}})
	if err != nil {
		return fmt.Errorf("register: %w", err)
	}
	a := &pb.SmallTensor{Meta: meta, Data: f32bytes([]float32{1, 2, 3, 4})}
	b := &pb.SmallTensor{Meta: meta, Data: f32bytes([]float32{5, 6, 7, 8})}
	res, err := call(&pb.Envelope{RequestId: 2, Body: &pb.Envelope_Execute{Execute: &pb.Execute{
		GraphHash: reg.GetGraphRegistered().GraphHash,
		Bindings:  []*pb.Binding{{Value: &pb.Binding_Inline{Inline: a}}, {Value: &pb.Binding_Inline{Inline: b}}},
		Outputs:   []*pb.OutputSpec{{Kind: pb.OutputSpec_KIND_RETURN}},
	}}})
	if err != nil {
		return fmt.Errorf("execute: %w", err)
	}
	got := res.GetExecuteResult().Returned[0].Data
	want := f32bytes([]float32{19, 22, 43, 50})
	if string(got) != string(want) {
		return fmt.Errorf("matmul returned wrong values")
	}
	return nil
}

// isolationChecks verifies the sandbox from the outside (release builds
// carry no probe code inside the worker; tests/security/isolation runs the
// in-worker probes in test builds).
func isolationChecks(out io.Writer, w *supervisor.Worker) error {
	pid := strconv.Itoa(w.PID())
	st, err := os.ReadFile("/proc/" + pid + "/status")
	if err != nil {
		return err
	}
	field := func(k string) string {
		for _, l := range strings.Split(string(st), "\n") {
			if v, ok := strings.CutPrefix(l, k+":"); ok {
				return strings.TrimSpace(v)
			}
		}
		return ""
	}
	var fails []string
	check := func(ok bool, what string) {
		mark := "ok  "
		if !ok {
			mark = "FAIL"
			fails = append(fails, what)
		}
		fmt.Fprintf(out, "    [%s] %s\n", mark, what)
	}
	check(field("NoNewPrivs") == "1", "no_new_privs is set")
	check(field("Seccomp") == "2", "seccomp filter mode is active")
	check(field("CapEff") == "0000000000000000", "no effective capabilities")
	env, _ := os.ReadFile("/proc/" + pid + "/environ")
	leaked := []string{}
	for _, kv := range strings.Split(string(env), "\x00") {
		k, _, _ := strings.Cut(kv, "=")
		switch k {
		case "", "CUBLAS_WORKSPACE_CONFIG", "LANG", "HOME", "TMPDIR", "XDG_CACHE_HOME", "CUDA_CACHE_PATH", "CUDA_MODULE_LOADING", "CUDA_DEVICE_ORDER",
			"TORCHINDUCTOR_DISABLE", "PYTORCH_NVML_BASED_CUDA_CHECK", "CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH":
		default:
			leaked = append(leaked, k)
		}
	}
	check(len(leaked) == 0, "environment contains only the allowlist")
	fds, _ := os.ReadDir("/proc/" + pid + "/fd")
	tcp := 0
	for _, fd := range fds {
		l, _ := os.Readlink("/proc/" + pid + "/fd/" + fd.Name())
		if strings.HasPrefix(l, "socket:") {
			tcp++
		}
	}
	check(tcp <= 2, "only the two inherited sockets are open")
	check(w.Report.Landlock || w.Report.LandlockABI == 0, fmt.Sprintf("landlock domain applied (abi %d)", w.Report.LandlockABI))
	if len(fails) > 0 {
		return fmt.Errorf("isolation checks failed: %s", strings.Join(fails, "; "))
	}
	return nil
}
