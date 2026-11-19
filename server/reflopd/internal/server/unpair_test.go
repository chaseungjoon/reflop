// SPDX-License-Identifier: Apache-2.0

package server

import (
	"bytes"
	"crypto/ed25519"
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"log"
	"net"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	pb "github.com/reflopai/reflop/pkg/reflopv1"
	"github.com/reflopai/reflop/pkg/wire"
	"github.com/reflopai/reflop/server/reflopd/internal/audit"
	"github.com/reflopai/reflop/server/reflopd/internal/auth"
)

type syncBuf struct {
	mu sync.Mutex
	b  bytes.Buffer
}

func (s *syncBuf) Write(p []byte) (int, error) { s.mu.Lock(); defer s.mu.Unlock(); return s.b.Write(p) }
func (s *syncBuf) String() string              { s.mu.Lock(); defer s.mu.Unlock(); return s.b.String() }

func unpairServer(t *testing.T) (*Server, *syncBuf, string) {
	t.Helper()
	dir := t.TempDir()
	s := testServer(8)
	s.events = newBroadcaster()
	logs := &syncBuf{}
	s.logger = log.New(logs, "", 0)
	var err error
	if s.key, err = auth.LoadOrCreateServerKey(filepath.Join(dir, "server.key"), filepath.Join(dir, "server.crt")); err != nil {
		t.Fatal(err)
	}
	if s.ids, err = auth.Open(dir); err != nil {
		t.Fatal(err)
	}
	auditDir := filepath.Join(dir, "audit")
	if s.audit, err = audit.Open(auditDir, 30); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.audit.Close() })
	return s, logs, auditDir
}

// dialPipe starts handleConn and does Hello; it returns the server nonce.
func dialPipe(t *testing.T, s *Server) (*wire.Writer, *wire.Reader, []byte, chan struct{}) {
	t.Helper()
	client, srvSide := net.Pipe()
	t.Cleanup(func() { client.Close() })
	done := make(chan struct{})
	go func() {
		s.handleConn(srvSide, "tls", "10.0.0.2:1", "", nil)
		close(done)
	}()
	client.SetDeadline(time.Now().Add(5 * time.Second))
	w, r := wire.NewWriter(client, 1<<20), wire.NewReader(client, 1<<20)
	if err := w.Write(&pb.Envelope{RequestId: 1, Body: &pb.Envelope_Hello{Hello: &pb.Hello{
		ProtocolVersions: []uint32{ProtocolVersion}, OpsetVersions: []uint32{1}}}}); err != nil {
		t.Fatal(err)
	}
	var env pb.Envelope
	if err := r.Read(&env); err != nil || env.GetHelloResponse() == nil {
		t.Fatalf("hello: %v %v", err, &env)
	}
	return w, r, env.GetHelloResponse().ServerNonce, done
}

// ADR-025: an authenticated Guest's Unpair revokes its own identity, tells
// the Owner (log, audit, `reflopd users`) and ends the connection.
func TestGuestUnpairRevokesOwnIdentity(t *testing.T) {
	s, logs, auditDir := unpairServer(t)
	pub, priv, _ := ed25519.GenerateKey(rand.Reader)
	if _, err := s.ids.AddKey("alice", pub, "tls", ""); err != nil {
		t.Fatal(err)
	}
	// Another active Guest is not touched.
	otherPub, _, _ := ed25519.GenerateKey(rand.Reader)
	if _, err := s.ids.AddKey("bob", otherPub, "tls", ""); err != nil {
		t.Fatal(err)
	}
	w, r, nonce, done := dialPipe(t, s)
	fp := s.key.Fingerprint[:]
	if err := w.Write(&pb.Envelope{RequestId: 3, Body: &pb.Envelope_Authenticate{Authenticate: &pb.Authenticate{
		PublicKey: pub, Signature: ed25519.Sign(priv, auth.AuthMessage(nonce, fp, pub))}}}); err != nil {
		t.Fatal(err)
	}
	var env pb.Envelope
	if err := r.Read(&env); err != nil || env.GetAuthResult() == nil {
		t.Fatalf("auth: %v %v", err, &env)
	}
	if err := w.Write(&pb.Envelope{RequestId: 4, Body: &pb.Envelope_Unpair{Unpair: &pb.Unpair{}}}); err != nil {
		t.Fatal(err)
	}
	env.Reset()
	if err := r.Read(&env); err != nil || env.GetUnpaired().GetIdentityName() != "alice" {
		t.Fatalf("unpair reply: %v %v", err, &env)
	}
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("connection not closed after Unpair")
	}

	alice, _ := s.ids.Get("alice")
	if !alice.Revoked || alice.GuestUnpaired == nil || len(alice.Keys) != 0 {
		t.Fatalf("alice after Unpair: %+v", alice)
	}
	if bob, _ := s.ids.Get("bob"); bob.Revoked || len(bob.Keys) != 1 {
		t.Fatalf("Unpair touched another identity: %+v", bob)
	}
	if !strings.Contains(logs.String(), "reflopd unpair alice") {
		t.Fatalf("the daemon log does not tell the Owner what to run:\n%s", logs.String())
	}
	recs, err := audit.Query(auditDir, "alice", time.Now().Add(-time.Hour), 0)
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, rec := range recs {
		found = found || rec.Event == "guest_unpaired"
	}
	if !found {
		t.Fatalf("no guest_unpaired audit record: %+v", recs)
	}

	// The key no longer authenticates.
	w, r, nonce, done = dialPipe(t, s)
	if err := w.Write(&pb.Envelope{RequestId: 3, Body: &pb.Envelope_Authenticate{Authenticate: &pb.Authenticate{
		PublicKey: pub, Signature: ed25519.Sign(priv, auth.AuthMessage(nonce, fp, pub))}}}); err != nil {
		t.Fatal(err)
	}
	env.Reset()
	if err := r.Read(&env); err != nil || env.GetError().GetCode() != pb.ErrorCode_ERROR_UNAUTHENTICATED {
		t.Fatalf("auth after Unpair: %v %v", err, &env)
	}
	<-done
}

// A pairing attempt with a key that is still registered tells the Owner
// what to run, in the log and the audit trail (issue #6).
func TestPairWithRegisteredKeyPromptsOwner(t *testing.T) {
	s, logs, auditDir := unpairServer(t)
	pub, priv, _ := ed25519.GenerateKey(rand.Reader)
	if _, err := s.ids.AddKey("carol", pub, "tls", ""); err != nil {
		t.Fatal(err)
	}
	inv, err := s.ids.CreateInvite("carol", time.Hour)
	if err != nil {
		t.Fatal(err)
	}
	w, r, nonce, done := dialPipe(t, s)
	fp := s.key.Fingerprint[:]
	if err := w.Write(&pb.Envelope{RequestId: 2, Body: &pb.Envelope_Pair{Pair: pairMsg(t, inv, priv, nonce, fp)}}); err != nil {
		t.Fatal(err)
	}
	var env pb.Envelope
	if err := r.Read(&env); err != nil {
		t.Fatal(err)
	}
	if msg := env.GetError().GetMessage(); !strings.Contains(msg, "reflopd unpair carol") {
		t.Fatalf("the Guest was told %q", msg)
	}
	<-done
	if !strings.Contains(logs.String(), "run: reflopd unpair carol") {
		t.Fatalf("daemon log:\n%s", logs.String())
	}
	recs, _ := audit.Query(auditDir, "", time.Now().Add(-time.Hour), 0)
	for _, rec := range recs {
		if rec.Event == "pair_failed" && strings.Contains(rec.Fields["reason"].(string), "already registered to carol") {
			return
		}
	}
	t.Fatalf("no pair_failed record with the reason: %+v", recs)
}

// Unpair before Authenticate is refused like any other pre-auth request:
// it cannot touch any identity.
func TestUnpairBeforeAuthRefused(t *testing.T) {
	s, _, _ := unpairServer(t)
	pub, _, _ := ed25519.GenerateKey(rand.Reader)
	if _, err := s.ids.AddKey("dora", pub, "tls", ""); err != nil {
		t.Fatal(err)
	}
	w, r, _, done := dialPipe(t, s)
	if err := w.Write(&pb.Envelope{RequestId: 2, Body: &pb.Envelope_Unpair{Unpair: &pb.Unpair{}}}); err != nil {
		t.Fatal(err)
	}
	var env pb.Envelope
	if err := r.Read(&env); err != nil || env.GetError().GetCode() != pb.ErrorCode_ERROR_UNAUTHENTICATED {
		t.Fatalf("pre-auth Unpair: %v %v", err, &env)
	}
	<-done
	if id, _ := s.ids.Get("dora"); id.Revoked || len(id.Keys) != 1 {
		t.Fatalf("pre-auth Unpair changed an identity: %+v", id)
	}
}

func pairMsg(t *testing.T, inv *auth.Invite, priv ed25519.PrivateKey, nonce, fp []byte) *pb.Pair {
	t.Helper()
	id, _ := hex.DecodeString(inv.ID)
	pub := priv.Public().(ed25519.PublicKey)
	msg := auth.PairMessage(nonce, fp, id, pub)
	m := hmac.New(sha256.New, inv.Secret)
	m.Write(msg)
	return &pb.Pair{InviteId: id, PublicKey: pub, Mac: m.Sum(nil), Signature: ed25519.Sign(priv, msg)}
}
