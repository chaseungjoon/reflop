// SPDX-License-Identifier: Apache-2.0

package server

import (
	"testing"
	"time"

	pb "github.com/chaseungjoon/reflop/pkg/reflopv1"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/auth"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/gpu"
)

// evictionFixture is a GPU slot with Guest A attached and Guest B detached.
func evictionFixture() (srv *Server, a, b *Session) {
	srv = testServer(8)
	slot := &DeviceSlot{Name: "cuda:0", GPU: &gpu.GPU{}}
	a = &Session{ID: [16]byte{1}, Identity: auth.Identity{Name: "alice", ID: "a"}, Device: slot, srv: srv,
		routes: map[uint64]*route{}, conn: &clientConn{}}
	b = &Session{ID: [16]byte{2}, Identity: auth.Identity{Name: "bob", ID: "b"}, Device: slot, srv: srv,
		routes: map[uint64]*route{}, detached: time.Now().Add(-time.Minute)}
	srv.sessions[a.hex()], srv.sessions[b.hex()] = a, b
	return srv, a, b
}

func oom(stage pb.ValidationStage) *pb.Error {
	return &pb.Error{Code: pb.ErrorCode_ERROR_OUT_OF_MEMORY, Stage: stage, Message: "out of memory"}
}

// A graph larger than the whole device fails validation (OUT_OF_MEMORY at
// V5) without allocating anything. It must not end another Guest's detached
// session: before 0.1.8 any Guest could end all of them by registering such
// graphs in a loop.
func TestOOMFromValidationDoesNotEvict(t *testing.T) {
	_, a, b := evictionFixture()
	for st := range pb.ValidationStage_name {
		stage := pb.ValidationStage(st)
		if stage < pb.ValidationStage_STAGE_V0_TRANSPORT || stage > pb.ValidationStage_STAGE_V5_SAFETY {
			continue
		}
		if v := a.evictionVictim(oom(stage)); v != nil {
			t.Fatalf("OUT_OF_MEMORY at %s evicted %q", stage, v.Identity.Name)
		}
	}
	if v := a.evictionVictim(oom(pb.ValidationStage(99))); v != nil {
		t.Fatalf("an unknown stage evicted %q", v.Identity.Name)
	}
	if v := a.evictionVictim(&pb.Error{Code: pb.ErrorCode_ERROR_RESOURCE_LIMIT, Stage: pb.ValidationStage_STAGE_E3_EXECUTE}); v != nil {
		t.Fatalf("a non-OOM error evicted %q", v.Identity.Name)
	}

	// Through deliver, the path a worker response takes. b has no worker, so
	// evicting it would panic. The route's gen differs from a.conn's, so the
	// response is dropped instead of written to the fake connection.
	a.routes[7] = &route{rid: 7, kind: "register", gen: 1}
	a.deliver(&pb.Envelope{RequestId: 7, End: true, Body: &pb.Envelope_Error{
		Error: oom(pb.ValidationStage_STAGE_V5_SAFETY)}})
	if b.closed {
		t.Fatal("bob's detached session was ended by alice's oversized graph")
	}
}

// An allocation that failed while running can be helped by freeing a
// detached session's memory, as before.
func TestOOMWhileRunningEvictsOldestDetached(t *testing.T) {
	srv, a, b := evictionFixture()
	for _, stage := range []pb.ValidationStage{pb.ValidationStage_STAGE_NONE, pb.ValidationStage_STAGE_E1_BINDINGS,
		pb.ValidationStage_STAGE_E2_RUNTIME, pb.ValidationStage_STAGE_E3_EXECUTE} {
		if v := a.evictionVictim(oom(stage)); v != b {
			t.Fatalf("OUT_OF_MEMORY at %s: victim %v, want bob", stage, v)
		}
	}
	// Never the failing session itself, and never on a CPU slot.
	if v := b.evictionVictim(oom(pb.ValidationStage_STAGE_E3_EXECUTE)); v != nil {
		t.Fatalf("bob's own OOM picked %q", v.Identity.Name)
	}
	cpu := &Session{Device: &DeviceSlot{Name: "cpu"}, srv: srv}
	if v := cpu.evictionVictim(oom(pb.ValidationStage_STAGE_E3_EXECUTE)); v != nil {
		t.Fatalf("a CPU OOM picked %q", v.Identity.Name)
	}
}
