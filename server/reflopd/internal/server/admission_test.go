// SPDX-License-Identifier: Apache-2.0

package server

import (
	"encoding/binary"
	"io"
	"log"
	"net"
	"testing"
	"time"

	"github.com/chaseungjoon/reflop/server/reflopd/internal/config"
)

func testServer(maxConns uint32) *Server {
	cfg := config.Defaults()
	cfg.Limits.MaxConnections = maxConns
	return &Server{config: cfg, logger: log.New(io.Discard, "", 0), sessions: map[string]*Session{},
		conns: map[*clientConn]bool{}, preauth: map[string]int{}}
}

// SEC_SWEEP M1: connections count from accept, and one source cannot hold
// more than maxPreAuthPerSource unauthenticated connections.
func TestAdmission(t *testing.T) {
	s := testServer(maxPreAuthPerSource + 4)
	var held []*admission
	for i := 0; i < maxPreAuthPerSource; i++ {
		a := s.admit("10.0.0.1:1234", true)
		if a == nil {
			t.Fatalf("connection %d from one source refused early", i)
		}
		held = append(held, a)
	}
	if s.admit("10.0.0.1:9999", true) != nil {
		t.Fatal("per-source pre-auth limit not enforced")
	}
	held[0].authenticated() // an authenticated connection frees its pre-auth slot
	if a := s.admit("10.0.0.1:1", true); a == nil {
		t.Fatal("slot not freed by authentication")
	} else {
		held = append(held, a)
	}
	for i := 0; i < 3; i++ {
		if a := s.admit("10.0.0.2:1", true); a == nil {
			t.Fatalf("other source refused at %d", i)
		} else {
			held = append(held, a)
		}
	}
	if s.admit("10.0.0.3:1", true) != nil || s.admit("ssh", false) != nil {
		t.Fatal("max_connections not enforced for connections still handshaking")
	}
	for _, a := range held {
		a.release()
		a.release() // idempotent
	}
	if s.nconns != 0 || len(s.preauth) != 0 {
		t.Fatalf("leaked slots: %d conns, preauth %v", s.nconns, s.preauth)
	}
}

// SEC_SWEEP M1: before authentication a peer cannot declare a frame larger
// than preAuthMaxFrame; the connection ends without reading it.
func TestPreAuthFrameCap(t *testing.T) {
	s := testServer(8)
	client, server := net.Pipe()
	done := make(chan struct{})
	go func() {
		s.handleConn(server, "tls", "10.0.0.1:1", "", nil)
		close(done)
	}()
	var hdr [4]byte
	binary.BigEndian.PutUint32(hdr[:], preAuthMaxFrame+1)
	client.SetDeadline(time.Now().Add(5 * time.Second))
	if _, err := client.Write(hdr[:]); err != nil {
		t.Fatal(err)
	}
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("reflopd kept the connection open after an oversized pre-auth frame header")
	}
	client.Close()
}

// SEC_SWEEP L8: failed-handshake audit records are rate-limited per source
// and globally; suppressed records are counted in the next one written.
func TestFailLimiter(t *testing.T) {
	var f failLimiter
	now := time.Unix(1_700_000_000, 0)
	for i := 0; i < failBurst; i++ {
		if ok, _ := f.allow("10.0.0.1:5", now); !ok {
			t.Fatalf("record %d refused within the burst", i)
		}
	}
	for i := 0; i < 50; i++ {
		if ok, _ := f.allow("10.0.0.1:6", now); ok {
			t.Fatal("per-source limit not enforced")
		}
	}
	ok, suppressed := f.allow("10.0.0.1:7", now.Add(failRefill))
	if !ok || suppressed != 50 {
		t.Fatalf("after refill: ok=%v suppressed=%d", ok, suppressed)
	}
	// Many sources: the global bucket caps the total.
	written := 0
	for i := 0; i < 10*failGlobalBurst; i++ {
		ip := net.IPv4(10, 1, byte(i>>8), byte(i)).String()
		if ok, _ := f.allow(ip+":1", now.Add(failRefill)); ok {
			written++
		}
	}
	if written > failGlobalBurst {
		t.Fatalf("%d records written in one instant across sources; global cap is %d", written, failGlobalBurst)
	}
}
