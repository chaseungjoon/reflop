// SPDX-License-Identifier: Apache-2.0

package server

import (
	"bytes"
	"testing"

	pb "github.com/reflopai/reflop/pkg/reflopv1"
	"github.com/reflopai/reflop/pkg/wire"
)

func heldExecute(rid uint64, payload int) *pb.Envelope {
	return &pb.Envelope{RequestId: rid, Body: &pb.Envelope_Execute{Execute: &pb.Execute{GraphHash: make([]byte, payload)}}}
}

// errorsByCode reads every response written to out.
func errorsByCode(t *testing.T, out *bytes.Buffer) map[pb.ErrorCode]int {
	t.Helper()
	got := map[pb.ErrorCode]int{}
	r := wire.NewReader(bytes.NewReader(out.Bytes()), 64<<20)
	for {
		env := &pb.Envelope{}
		if err := r.Read(env); err != nil {
			return got
		}
		if e := env.GetError(); e != nil {
			got[e.Code]++
		}
	}
}

// Issue #15: while paused, a session holds at most maxHeldRequests requests
// and 2 x max_frame_bytes; the rest get ERROR_PAUSED instead of growing
// reflopd until its cgroup OOM-kills it.
func TestHeldRequestsAreBounded(t *testing.T) {
	srv, sess, c, out := snapFixture(t)
	srv.paused.Store(true)
	for i := 0; i < maxHeldRequests+10; i++ {
		if err := sess.forward(c, heldExecute(uint64(i+1), 8)); err != nil {
			t.Fatal(err)
		}
	}
	if len(sess.held) != maxHeldRequests {
		t.Fatalf("held %d requests, want %d", len(sess.held), maxHeldRequests)
	}
	if n := errorsByCode(t, out)[pb.ErrorCode_ERROR_PAUSED]; n != 10 {
		t.Fatalf("%d ERROR_PAUSED answers, want 10", n)
	}

	// Bytes: 2 x max_frame_bytes per session.
	srv2, sess2, c2, out2 := snapFixture(t)
	srv2.paused.Store(true)
	cfg := srv2.cfg()
	cfg.Limits.MaxFrameBytes = 1 << 20
	srv2.config = cfg
	for i := 0; i < 8; i++ {
		sess2.forward(c2, heldExecute(uint64(i+1), 512<<10))
	}
	if sess2.heldBytes > 2<<20 || len(sess2.held) != 3 {
		t.Fatalf("held %d requests, %d bytes; want 3 within 2 MiB", len(sess2.held), sess2.heldBytes)
	}
	if n := errorsByCode(t, out2)[pb.ErrorCode_ERROR_PAUSED]; n != 5 {
		t.Fatalf("%d ERROR_PAUSED answers, want 5", n)
	}
	if srv2.heldBytes.Load() != sess2.heldBytes {
		t.Fatalf("server total %d, session %d", srv2.heldBytes.Load(), sess2.heldBytes)
	}
}

// The cap across all sessions holds even when each session is under its own.
func TestHeldRequestsTotalCap(t *testing.T) {
	old := maxHeldBytesTotal
	maxHeldBytesTotal = 3<<20 + 4096 // three 1 MiB requests plus envelope overhead
	defer func() { maxHeldBytesTotal = old }()
	srv, a, ca, outA := snapFixture(t)
	srv.paused.Store(true)
	b := &Session{ID: [16]byte{0xb2}, srv: srv, routes: map[uint64]*route{}}
	outB := &bytes.Buffer{}
	cb := &clientConn{srv: srv, w: wire.NewWriter(outB, 64<<20), gen: 1}
	b.conn = cb
	for i := 0; i < 2; i++ {
		a.forward(ca, heldExecute(uint64(i+1), 1<<20))
		b.forward(cb, heldExecute(uint64(i+1), 1<<20))
	}
	if got := srv.heldBytes.Load(); got > maxHeldBytesTotal {
		t.Fatalf("held %d bytes in total, cap is %d", got, maxHeldBytesTotal)
	}
	if errorsByCode(t, outA)[pb.ErrorCode_ERROR_PAUSED]+errorsByCode(t, outB)[pb.ErrorCode_ERROR_PAUSED] != 1 {
		t.Fatal("want exactly one request refused by the total cap")
	}
}

// A session that ends during a pause answers its held requests and gives
// their bytes back; before, they were never answered.
func TestLostSessionAnswersHeldRequests(t *testing.T) {
	srv, sess, c, out := snapFixture(t)
	srv.paused.Store(true)
	for i := 0; i < 5; i++ {
		sess.forward(c, heldExecute(uint64(i+1), 64))
	}
	sess.lost("test")
	if n := errorsByCode(t, out)[pb.ErrorCode_ERROR_SESSION_LOST]; n != 5 {
		t.Fatalf("%d SESSION_LOST answers, want 5", n)
	}
	if len(sess.held) != 0 || sess.heldBytes != 0 || srv.heldBytes.Load() != 0 {
		t.Fatalf("held queue not emptied: %d requests, %d bytes, server %d", len(sess.held), sess.heldBytes,
			srv.heldBytes.Load())
	}
	sess.release() // a resume after the end must not dispatch anything (no worker here)
}

// While release forwards held requests, a new one must queue behind them,
// not overtake them. This session has no worker, so a direct dispatch would
// panic.
func TestNewRequestsQueueBehindRelease(t *testing.T) {
	_, sess, c, _ := snapFixture(t)
	sess.releasing = true
	sess.forward(c, heldExecute(1, 8))
	if len(sess.held) != 1 {
		t.Fatalf("held %d requests, want the new one queued", len(sess.held))
	}
}
