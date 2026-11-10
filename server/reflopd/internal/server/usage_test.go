// SPDX-License-Identifier: Apache-2.0

package server

import (
	"path/filepath"
	"testing"
	"time"

	"github.com/chaseungjoon/reflop/server/reflopd/internal/auth"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/usage"
)

func TestGuestUsageIsOwnAndLive(t *testing.T) {
	dir := t.TempDir()
	l, err := usage.Open(filepath.Join(dir, "usage.json"), filepath.Join(dir, "audit"))
	if err != nil {
		t.Fatal(err)
	}
	l.SessionOpen("alice")
	l.Execute("alice", "cuda:0", 4000, 2e12, 77)
	l.Execute("alice", "cpu", 1000, 1e9, 5)
	l.Execute("bob", "cuda:0", 9e6, 9e15, 9)

	s := &Server{usage: l, sessions: map[string]*Session{}, conns: map[*clientConn]bool{}}
	live := &Session{Identity: auth.Identity{Name: "alice"}, created: time.Now().Add(-2 * time.Second)}
	s.sessions["a"] = live
	s.sessions["b"] = &Session{Identity: auth.Identity{Name: "bob"}, created: time.Now().Add(-time.Hour)}
	name, other := "alice", "bob"
	ca, cb := &clientConn{}, &clientConn{}
	ca.account.Store(&name)
	ca.bytesUp.Store(100)
	ca.bytesDown.Store(200)
	cb.account.Store(&other)
	cb.bytesUp.Store(1 << 30)
	s.conns[ca], s.conns[cb], s.conns[&clientConn{}] = true, true, true // the last is unauthenticated

	u := s.guestUsage("alice")
	if u.IdentityName != "alice" || u.Executes != 2 || u.DeviceUs != 5000 || u.Flops != 2e12+1e9 ||
		u.PeakMemoryBytes != 77 || u.Sessions != 1 || u.BytesUp != 100 || u.BytesDown != 200 {
		t.Fatalf("alice: %+v", u)
	}
	if u.SessionUs < 2e6 || u.SessionUs > 60e6 {
		t.Fatalf("live session time %d µs", u.SessionUs)
	}
	if len(u.ByDevice) != 2 || u.ByDevice[0].Device != "cpu" || u.ByDevice[0].DeviceUs != 1000 ||
		u.ByDevice[1].Device != "cuda:0" || u.ByDevice[1].DeviceUs != 4000 {
		t.Fatalf("by device: %+v", u.ByDevice)
	}
	if u.FirstUnix == 0 || u.LastUnix < u.FirstUnix || u.SinceUnix == 0 {
		t.Fatalf("times: %+v", u)
	}

	if n := s.guestUsage("carol"); n.IdentityName != "carol" || n.Executes != 0 || n.DeviceUs != 0 || n.FirstUnix != 0 {
		t.Fatalf("an identity with no usage: %+v", n)
	}
	// The Owner's view still sees everyone.
	if all := s.usageNow(""); len(all.Identities) != 2 || all.Identities["bob"].BytesUp != 1<<30 {
		t.Fatalf("owner view: %+v", all.Identities)
	}
}
