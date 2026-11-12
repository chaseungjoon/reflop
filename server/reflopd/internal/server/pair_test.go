// SPDX-License-Identifier: Apache-2.0

package server

import (
	"bytes"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"log"
	"net"
	"path/filepath"
	"strings"
	"testing"
	"time"

	pb "github.com/chaseungjoon/reflop/pkg/reflopv1"
	"github.com/chaseungjoon/reflop/pkg/wire"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/audit"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/auth"
)

// Issue #5: a failed pairing tells the Guest only ErrInvite's text, while
// the daemon log and the audit trail say why it failed.
func TestPairFailureReasonStaysWithOwner(t *testing.T) {
	dir := t.TempDir()
	s := testServer(8)
	var logs bytes.Buffer
	s.logger = log.New(&logs, "", 0)
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
	defer s.audit.Close()
	inv, err := s.ids.CreateInvite("dave", -time.Minute)
	if err != nil {
		t.Fatal(err)
	}

	client, srvSide := net.Pipe()
	defer client.Close()
	done := make(chan struct{})
	go func() {
		s.handleConn(srvSide, "tls", "10.0.0.1:1", "", nil)
		close(done)
	}()
	client.SetDeadline(time.Now().Add(5 * time.Second))
	w, r := wire.NewWriter(client, 1<<16), wire.NewReader(client, 1<<16)
	if err := w.Write(&pb.Envelope{RequestId: 1, Body: &pb.Envelope_Hello{Hello: &pb.Hello{
		ProtocolVersions: []uint32{ProtocolVersion}, OpsetVersions: []uint32{1}}}}); err != nil {
		t.Fatal(err)
	}
	var env pb.Envelope
	if err := r.Read(&env); err != nil || env.GetHelloResponse() == nil {
		t.Fatalf("hello: %v %v", err, &env)
	}
	pub, _, _ := ed25519.GenerateKey(rand.Reader)
	id, _ := hex.DecodeString(inv.ID)
	if err := w.Write(&pb.Envelope{RequestId: 2, Body: &pb.Envelope_Pair{Pair: &pb.Pair{
		InviteId: id, PublicKey: pub, Mac: make([]byte, 32), Signature: make([]byte, ed25519.SignatureSize)}}}); err != nil {
		t.Fatal(err)
	}
	env.Reset()
	if err := r.Read(&env); err != nil {
		t.Fatal(err)
	}
	if got := env.GetError().GetMessage(); got != auth.ErrInvite.Error() {
		t.Fatalf("the Guest was told %q, want only %q", got, auth.ErrInvite.Error())
	}
	<-done

	if !strings.Contains(logs.String(), `the invite for "dave" expired at`) {
		t.Fatalf("daemon log lacks the reason:\n%s", logs.String())
	}
	recs, err := audit.Query(auditDir, "", time.Now().Add(-time.Hour), 0)
	if err != nil {
		t.Fatal(err)
	}
	for _, rec := range recs {
		if rec.Event == "pair_failed" && strings.Contains(fmt.Sprint(rec.Fields["reason"]), `"dave" expired at`) {
			return
		}
	}
	t.Fatalf("no pair_failed audit record with the reason: %+v", recs)
}
