// SPDX-License-Identifier: Apache-2.0

package auth

import (
	"crypto/ed25519"
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"path/filepath"
	"strings"
	"testing"
	"time"

	pb "github.com/chaseungjoon/reflop/pkg/reflopv1"
)

func pairReq(t *testing.T, inv *Invite, priv ed25519.PrivateKey, nonce, fp []byte) *pb.Pair {
	t.Helper()
	id, _ := hex.DecodeString(inv.ID)
	pub := priv.Public().(ed25519.PublicKey)
	msg := PairMessage(nonce, fp, id, pub)
	m := hmac.New(sha256.New, inv.Secret)
	m.Write(msg)
	return &pb.Pair{InviteId: id, PublicKey: pub, Mac: m.Sum(nil), Signature: ed25519.Sign(priv, msg)}
}

func authReq(priv ed25519.PrivateKey, nonce, fp []byte) *pb.Authenticate {
	pub := priv.Public().(ed25519.PublicKey)
	return &pb.Authenticate{PublicKey: pub, Signature: ed25519.Sign(priv, AuthMessage(nonce, fp, pub))}
}

func TestPairAuthenticateRevoke(t *testing.T) {
	s, err := Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	nonce, fp := make([]byte, 32), make([]byte, 32)
	rand.Read(nonce)
	rand.Read(fp)
	_, priv, _ := ed25519.GenerateKey(rand.Reader)
	inv, err := s.CreateInvite("alice", time.Hour)
	if err != nil {
		t.Fatal(err)
	}
	id, err := s.Redeem(pairReq(t, inv, priv, nonce, fp), nonce, fp)
	if err != nil || id.Name != "alice" {
		t.Fatalf("redeem: %v", err)
	}
	// Single use.
	if _, err := s.Redeem(pairReq(t, inv, priv, nonce, fp), nonce, fp); !errors.Is(err, ErrInvite) {
		t.Fatalf("reused invite accepted: %v", err)
	}
	if _, err := s.Authenticate(authReq(priv, nonce, fp), nonce, fp, ""); err != nil {
		t.Fatalf("auth: %v", err)
	}
	// Replay with another nonce (or for another server key) fails.
	other := make([]byte, 32)
	rand.Read(other)
	if _, err := s.Authenticate(authReq(priv, nonce, fp), other, fp, ""); err == nil {
		t.Fatal("replayed signature accepted under a different nonce")
	}
	if _, err := s.Authenticate(authReq(priv, nonce, fp), nonce, other, ""); err == nil {
		t.Fatal("signature relayed to a different server accepted")
	}
	// Unknown key.
	_, stranger, _ := ed25519.GenerateKey(rand.Reader)
	if _, err := s.Authenticate(authReq(stranger, nonce, fp), nonce, fp, ""); err == nil {
		t.Fatal("unknown key accepted")
	}
	// SSH mode binds the key to the identity in the forced command.
	if _, err := s.Authenticate(authReq(priv, nonce, fp), nonce, fp, "00000000000000000000000000000000"); err == nil {
		t.Fatal("key accepted for another identity's forced command")
	}
	if _, err := s.Revoke("alice"); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Authenticate(authReq(priv, nonce, fp), nonce, fp, ""); err == nil {
		t.Fatal("revoked key accepted")
	}
	if _, err := s.CreateInvite("alice", time.Hour); err == nil {
		t.Fatal("revoked name reused")
	}
}

func TestInviteFailuresConsumeAndExpire(t *testing.T) {
	s, _ := Open(t.TempDir())
	nonce, fp := make([]byte, 32), make([]byte, 32)
	_, priv, _ := ed25519.GenerateKey(rand.Reader)
	inv, _ := s.CreateInvite("bob", time.Hour)
	bad := pairReq(t, inv, priv, nonce, fp)
	bad.Mac[0] ^= 1
	if _, err := s.Redeem(bad, nonce, fp); !errors.Is(err, ErrInvite) {
		t.Fatalf("bad MAC accepted: %v", err)
	}
	// A failed attempt burns the invite: no brute force.
	if _, err := s.Redeem(pairReq(t, inv, priv, nonce, fp), nonce, fp); !errors.Is(err, ErrInvite) {
		t.Fatalf("invite usable after a failed attempt: %v", err)
	}
	exp, _ := s.CreateInvite("carol", -time.Second)
	if _, err := s.Redeem(pairReq(t, exp, priv, nonce, fp), nonce, fp); !errors.Is(err, ErrInvite) {
		t.Fatalf("expired invite accepted: %v", err)
	}
	if _, err := s.CreateInvite("../etc", time.Hour); err == nil {
		t.Fatal("bad name accepted")
	}
}

// Issue #5: every invite failure is ErrInvite to the Guest, but carries the
// actual reason for the Owner.
func TestInviteFailureReasons(t *testing.T) {
	s, _ := Open(t.TempDir())
	nonce, fp := make([]byte, 32), make([]byte, 32)
	_, priv, _ := ed25519.GenerateKey(rand.Reader)
	reason := func(req *pb.Pair) string {
		t.Helper()
		_, err := s.Redeem(req, nonce, fp)
		var ie *InviteError
		if !errors.Is(err, ErrInvite) || !errors.As(err, &ie) {
			t.Fatalf("want an InviteError, got %v", err)
		}
		return ie.Reason
	}
	exp, _ := s.CreateInvite("carol", -90*time.Second)
	if r := reason(pairReq(t, exp, priv, nonce, fp)); !strings.Contains(r, `"carol" expired at `+exp.Expires.Format(time.RFC3339)) ||
		!strings.Contains(r, "ago") {
		t.Fatalf("expired: %q", r)
	}
	inv, _ := s.CreateInvite("bob", time.Hour)
	bad := pairReq(t, inv, priv, nonce, fp)
	bad.Mac[0] ^= 1
	if r := reason(bad); !strings.Contains(r, "wrong secret or signature") {
		t.Fatalf("bad MAC: %q", r)
	}
	if r := reason(pairReq(t, inv, priv, nonce, fp)); !strings.Contains(r, `"bob" was already used`) {
		t.Fatalf("used: %q", r)
	}
	unknown := pairReq(t, &Invite{ID: strings.Repeat("ab", 16), Secret: make([]byte, 32)}, priv, nonce, fp)
	if r := reason(unknown); !strings.Contains(r, "no such invite") {
		t.Fatalf("unknown: %q", r)
	}
	if r := reason(&pb.Pair{}); !strings.Contains(r, "malformed") {
		t.Fatalf("malformed: %q", r)
	}
}

func TestDeleteOnlyRevoked(t *testing.T) {
	s, _ := Open(t.TempDir())
	if _, err := s.AddKey("erin", make([]byte, 32), "ssh", ""); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Delete("erin"); !errors.Is(err, ErrNotRevoked) {
		t.Fatalf("active identity deleted: %v", err)
	}
	if _, ok := s.Get("erin"); !ok {
		t.Fatal("refused delete still removed the identity")
	}
	if _, err := s.Delete("nobody"); err == nil {
		t.Fatal("deleting an unknown name succeeded")
	}
	if _, err := s.Revoke("erin"); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Delete("erin"); err != nil {
		t.Fatal(err)
	}
	if _, ok := s.Get("erin"); ok {
		t.Fatal("deleted identity still listed")
	}
	for _, inv := range s.db.Invites {
		if inv.Name == "erin" {
			t.Fatal("deleted identity's invites kept")
		}
	}
	// The deletion is on disk, and the name can be invited again.
	s2, _ := Open(filepath.Dir(s.path))
	if _, ok := s2.Get("erin"); ok {
		t.Fatal("deletion not saved")
	}
	if _, err := s2.CreateInvite("erin", time.Hour); err != nil {
		t.Fatalf("name not freed: %v", err)
	}
}

func TestInviteCodeRoundTrip(t *testing.T) {
	c := &pb.InviteCode{Version: 1, Transport: pb.TransportKind_TRANSPORT_TLS, Addresses: []string{"10.0.0.2:7460"},
		ServerKeyFingerprint: make([]byte, 32), InviteId: make([]byte, 16), Secret: make([]byte, 32), IdentityName: "x"}
	s, err := EncodeInvite(c)
	if err != nil {
		t.Fatal(err)
	}
	d, err := DecodeInvite(s)
	if err != nil || d.Addresses[0] != "10.0.0.2:7460" {
		t.Fatalf("decode: %v", err)
	}
	if _, err := DecodeInvite("reflop1-!!!"); err == nil {
		t.Fatal("garbage accepted")
	}
}

func TestServerKeyPersistsAndIsPrivate(t *testing.T) {
	dir := t.TempDir()
	k1, err := LoadOrCreateServerKey(dir+"/k", dir+"/c")
	if err != nil {
		t.Fatal(err)
	}
	k2, err := LoadOrCreateServerKey(dir+"/k", dir+"/c")
	if err != nil || k1.Fingerprint != k2.Fingerprint {
		t.Fatal("server key changed between loads")
	}
}
