// SPDX-License-Identifier: Apache-2.0

package server

import (
	"bytes"
	"encoding/hex"
	"errors"
	"io"
	"log"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	pb "github.com/reflopai/reflop/pkg/reflopv1"
	"github.com/reflopai/reflop/pkg/wire"
	"github.com/reflopai/reflop/server/reflopd/internal/audit"
	"github.com/reflopai/reflop/server/reflopd/internal/auth"
	"github.com/reflopai/reflop/server/reflopd/internal/config"
	"github.com/reflopai/reflop/server/reflopd/internal/paths"
	"github.com/reflopai/reflop/server/reflopd/internal/store"
)

// snapFixture is a server with a temporary state directory and one session
// of identity "a" whose responses go to out.
func snapFixture(t *testing.T) (*Server, *Session, *clientConn, *bytes.Buffer) {
	t.Helper()
	dir := t.TempDir()
	l := paths.Layout{Config: filepath.Join(dir, "config"), State: filepath.Join(dir, "state"), Runtime: filepath.Join(dir, "runtime"),
		Run: filepath.Join(dir, "run"), Scratch: filepath.Join(dir, "scratch")}
	if err := l.Ensure(); err != nil {
		t.Fatal(err)
	}
	al, err := audit.Open(l.Audit(), 30)
	if err != nil {
		t.Fatal(err)
	}
	cfg := config.Defaults()
	cfg.Rails.DiskFreeFloorBytes, cfg.Rails.DiskFreeFloorPercent = config.MinDiskFreeFloorBytes, 1
	srv := &Server{config: cfg, layout: l, audit: al, logger: log.New(io.Discard, "", 0), sessions: map[string]*Session{},
		conns: map[*clientConn]bool{}, preauth: map[string]int{}, idReserved: map[string]uint64{}}
	srv.store = store.New(l.Store(), 4<<20, 1<<40, func() config.Rails { return srv.cfg().Rails }, func(string) uint64 { return 0 })
	out := &bytes.Buffer{}
	c := &clientConn{srv: srv, w: wire.NewWriter(out, 64<<20), gen: 1}
	sess := &Session{ID: [16]byte{0xa1}, Identity: auth.Identity{Name: "alice", ID: "aa"}, srv: srv,
		routes: map[uint64]*route{}, conn: c, Device: &DeviceSlot{Name: "cpu"}}
	srv.sessions[sess.hex()] = sess
	return srv, sess, c, out
}

func lastError(t *testing.T, out *bytes.Buffer) *pb.Error {
	t.Helper()
	r := wire.NewReader(bytes.NewReader(out.Bytes()), 64<<20)
	var last *pb.Envelope
	for {
		env := &pb.Envelope{}
		if err := r.Read(env); err != nil {
			break
		}
		last = env
	}
	if last == nil || last.GetError() == nil {
		t.Fatalf("no error response (got %v)", last)
	}
	return last.GetError()
}

func ckEnv() *pb.Envelope {
	return &pb.Envelope{RequestId: 5, Body: &pb.Envelope_Checkpoint{Checkpoint: &pb.Checkpoint{}}}
}

// Checkpoints are off unless the Owner turns them on (ADR-029).
func TestCheckpointNeedsTheOwner(t *testing.T) {
	_, sess, c, out := snapFixture(t)
	env := ckEnv()
	if err := sess.checkpoint(c, env, env.GetCheckpoint()); err != nil {
		t.Fatal(err)
	}
	if e := lastError(t, out); e.Code != pb.ErrorCode_ERROR_PERMISSION_DENIED {
		t.Fatalf("got %v, want PERMISSION_DENIED", e)
	}
}

// The server, not the client, enforces the Owner's interval.
func TestCheckpointIntervalEnforced(t *testing.T) {
	srv, sess, c, out := snapFixture(t)
	srv.config.Checkpoints = config.Checkpoints{Enabled: true, Interval: config.Duration(config.MinCheckpointInterval)}
	sess.lastCheckpoint = time.Now().Add(-4 * time.Minute)
	env := ckEnv()
	if err := sess.checkpoint(c, env, env.GetCheckpoint()); err != nil {
		t.Fatal(err)
	}
	if e := lastError(t, out); e.Code != pb.ErrorCode_ERROR_RESOURCE_LIMIT || !strings.Contains(e.Message, "one checkpoint every") {
		t.Fatalf("got %v, want RESOURCE_LIMIT", e)
	}
	out.Reset()
	env = &pb.Envelope{RequestId: 6, Body: &pb.Envelope_Checkpoint{Checkpoint: &pb.Checkpoint{Tag: make([]byte, maxTagBytes+1)}}}
	if err := sess.checkpoint(c, env, env.GetCheckpoint()); err != nil {
		t.Fatal(err)
	}
	if e := lastError(t, out); e.Code != pb.ErrorCode_ERROR_RESOURCE_LIMIT {
		t.Fatalf("oversized tag: got %v", e)
	}
}

// A file whose size is not what the worker announced is never published,
// and the previous snapshot stays.
func TestPublishKeepsPreviousOnFailure(t *testing.T) {
	_, sess, _, _ := snapFixture(t)
	dir := sess.snapshotDir()
	os.MkdirAll(dir, 0o700)
	final := filepath.Join(dir, "run.safetensors")
	tmp := filepath.Join(dir, ".run.tmp")
	os.WriteFile(final, []byte("previous"), 0o600)
	os.WriteFile(tmp, []byte("short"), 0o600)
	pw := &pendingWrite{tmp: tmp, final: final, expect: 100}
	if err := sess.publish(pw); err == nil {
		t.Fatal("a file of the wrong size was published")
	}
	if b, _ := os.ReadFile(final); string(b) != "previous" {
		t.Fatalf("previous snapshot changed: %q", b)
	}
	// The right size publishes, durably, and replaces the previous one.
	os.WriteFile(tmp, bytes.Repeat([]byte{1}, 100), 0o600)
	if err := sess.publish(pw); err != nil {
		t.Fatal(err)
	}
	if st, _ := os.Stat(final); st.Size() != 100 {
		t.Fatal("not replaced")
	}
	if _, err := os.Stat(tmp); !errors.Is(err, os.ErrNotExist) {
		t.Fatal("temporary file left behind")
	}
}

// A session ended on purpose never gets a checkpoint back, even when its
// write finishes after the end, and it cannot be recovered.
func TestEndedOnPurposeIsNeverRevived(t *testing.T) {
	_, sess, c, out := snapFixture(t)
	dir := sess.checkpointDir()
	os.MkdirAll(dir, 0o700)
	final := checkpointPath(dir, sess.ID[:])
	os.WriteFile(final, []byte("old checkpoint"), 0o600)
	sess.final = true
	sess.deleteCheckpoint()
	tmp := final + ".tmp"
	os.WriteFile(tmp, bytes.Repeat([]byte{1}, 10), 0o600)
	if err := sess.publish(&pendingWrite{tmp: tmp, final: final, expect: 10, checkpoint: true}); !errors.Is(err, errSessionFinal) {
		t.Fatalf("late publish after a final close: %v", err)
	}
	if _, err := os.Stat(final); !errors.Is(err, os.ErrNotExist) {
		t.Fatal("the checkpoint of a session ended on purpose exists")
	}
	// A new session of the same identity cannot recover it.
	next := &Session{ID: [16]byte{0xb2}, Identity: sess.Identity, srv: sess.srv, routes: map[uint64]*route{}, conn: c}
	delete(sess.srv.sessions, sess.hex())
	env := &pb.Envelope{RequestId: 9, Body: &pb.Envelope_RecoverSession{RecoverSession: &pb.RecoverSession{FromSessionId: sess.ID[:]}}}
	if err := next.recoverSession(c, env, env.GetRecoverSession()); err != nil {
		t.Fatal(err)
	}
	if e := lastError(t, out); e.Code != pb.ErrorCode_ERROR_UNKNOWN_SESSION {
		t.Fatalf("got %v, want UNKNOWN_SESSION", e)
	}
}

// RecoverSession finds checkpoints only in the caller's own identity
// directory (I7, I8), and never while the old session still runs.
func TestRecoverIsIdentityScoped(t *testing.T) {
	srv, sess, c, out := snapFixture(t)
	// Bob's checkpoint of session b0...
	bob := srv.layout.IdentitySnapshots("bb")
	os.MkdirAll(filepath.Join(bob, checkpointsDir), 0o700)
	bobSession := [16]byte{0xb0}
	os.WriteFile(checkpointPath(filepath.Join(bob, checkpointsDir), bobSession[:]), []byte("x"), 0o600)
	env := &pb.Envelope{RequestId: 3, Body: &pb.Envelope_RecoverSession{RecoverSession: &pb.RecoverSession{FromSessionId: bobSession[:]}}}
	if err := sess.recoverSession(c, env, env.GetRecoverSession()); err != nil {
		t.Fatal(err)
	}
	if e := lastError(t, out); e.Code != pb.ErrorCode_ERROR_UNKNOWN_SESSION {
		t.Fatalf("alice reached bob's checkpoint: %v", e)
	}
	// A running session of alice is resumed, not recovered.
	out.Reset()
	other := &Session{ID: [16]byte{0xa2}, Identity: sess.Identity, srv: srv, routes: map[uint64]*route{}}
	srv.sessions[other.hex()] = other
	env = &pb.Envelope{RequestId: 4, Body: &pb.Envelope_RecoverSession{RecoverSession: &pb.RecoverSession{FromSessionId: other.ID[:]}}}
	if err := sess.recoverSession(c, env, env.GetRecoverSession()); err != nil {
		t.Fatal(err)
	}
	if e := lastError(t, out); !strings.Contains(e.Message, "still running") {
		t.Fatalf("got %v", e)
	}
}

// A corrupt checkpoint fails closed before anything reaches the worker.
func TestRecoverCorruptCheckpointFailsClosed(t *testing.T) {
	_, sess, c, out := snapFixture(t)
	dir := sess.checkpointDir()
	os.MkdirAll(dir, 0o700)
	old := [16]byte{0xa0}
	for _, body := range [][]byte{nil, []byte("\x10\x00\x00\x00\x00\x00\x00\x00{\"a\":1}"), bytes.Repeat([]byte{0xff}, 64)} {
		os.WriteFile(checkpointPath(dir, old[:]), body, 0o600)
		out.Reset()
		env := &pb.Envelope{RequestId: 3, Body: &pb.Envelope_RecoverSession{RecoverSession: &pb.RecoverSession{FromSessionId: old[:]}}}
		if err := sess.recoverSession(c, env, env.GetRecoverSession()); err != nil {
			t.Fatal(err)
		}
		if e := lastError(t, out); e.Code != pb.ErrorCode_ERROR_PROTOCOL {
			t.Fatalf("corrupt checkpoint %q: got %v", body, e)
		}
	}
}

// Writes in progress count against the storage quota together, so two
// writes cannot each fit and together exceed it.
func TestReserveWriteQuota(t *testing.T) {
	srv, _, _, _ := snapFixture(t)
	if _, err := srv.reserveWriteQuota("aa", 1000, 600, 0); err != nil {
		t.Fatal(err)
	}
	if code, err := srv.reserveWriteQuota("aa", 1000, 600, 0); err == nil || code != pb.ErrorCode_ERROR_QUOTA {
		t.Fatalf("second write past the quota allowed: %v %v", code, err)
	}
	// Replacing a file counts its size as freed.
	os.MkdirAll(srv.layout.IdentitySnapshots("aa"), 0o700)
	os.WriteFile(filepath.Join(srv.layout.IdentitySnapshots("aa"), "x.safetensors"), make([]byte, 300), 0o600)
	srv.releaseWrite("aa", 600)
	if _, err := srv.reserveWriteQuota("aa", 1000, 900, 300); err != nil {
		t.Fatalf("replacing a 300-byte file with 900 bytes under a 1000-byte quota: %v", err)
	}
	srv.releaseWrite("aa", 900)
	if srv.diskReserved != 0 || len(srv.idReserved) != 0 {
		t.Fatalf("reservations leaked: %d %v", srv.diskReserved, srv.idReserved)
	}
}

// Keys the worker refuses are refused here too (writer/reader agreement).
func TestSnapshotKeyRules(t *testing.T) {
	for _, k := range []string{"", "__metadata__", "__reflop__/meta", "__reflop__/rng/cpu", strings.Repeat("k", 257)} {
		if keyError(k) == "" {
			t.Errorf("key %q accepted", k)
		}
	}
	for _, k := range []string{"0.weight", "layer.1.bias", "__other__", strings.Repeat("k", 256)} {
		if why := keyError(k); why != "" {
			t.Errorf("key %q refused: %s", k, why)
		}
	}
}

func TestSweepCheckpoints(t *testing.T) {
	srv, sess, _, _ := snapFixture(t)
	dir := sess.checkpointDir()
	os.MkdirAll(dir, 0o700)
	old := time.Now().Add(-2 * checkpointKeep)
	write := func(name string, mt time.Time) string {
		p := filepath.Join(dir, name)
		os.WriteFile(p, []byte("x"), 0o600)
		os.Chtimes(p, mt, mt)
		return p
	}
	live := write(sess.hex()+".safetensors", old)
	orphan := write(hex.EncodeToString([]byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16})+".safetensors", old)
	recent := write(hex.EncodeToString(bytes.Repeat([]byte{7}, 16))+".safetensors", time.Now())
	tmp := write(".x.safetensors.tmp-1234", old)
	srv.sweepCheckpoints()
	for p, want := range map[string]bool{live: true, orphan: false, recent: true, tmp: false} {
		_, err := os.Stat(p)
		if (err == nil) != want {
			t.Errorf("%s: exists=%v, want %v", filepath.Base(p), err == nil, want)
		}
	}
}
