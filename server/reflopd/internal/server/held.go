// SPDX-License-Identifier: Apache-2.0

package server

import (
	"fmt"

	"google.golang.org/protobuf/proto"

	pb "github.com/reflopai/reflop/pkg/reflopv1"
	"github.com/reflopai/reflop/server/reflopd/internal/audit"
)

// Requests a session sends while the Owner has paused execution wait in
// reflopd until resume (issue #15). Without a bound, a Guest that keeps
// sending during a pause grows reflopd until its cgroup OOM-kills it, which
// ends every session. An honest client sends little while paused, because
// it soon waits on a result.
const maxHeldRequests = 4096 // per session

// maxHeldBytesTotal caps all sessions together (a variable for tests).
var maxHeldBytesTotal int64 = 1 << 30

type heldReq struct {
	env  *pb.Envelope
	conn *clientConn
	size int64
}

// maxHeldBytes is the per-session cap: two of the largest frames, so one
// maximum-size request always fits, but never more than the total cap.
func (s *Session) maxHeldBytes() int64 {
	return min(2*int64(s.srv.cfg().Limits.MaxFrameBytes), maxHeldBytesTotal)
}

// holdLocked queues env behind earlier held requests. When the session or
// reflopd already holds too much it returns the message for an ERROR_PAUSED
// answer instead, which the caller sends after unlocking s.mu: a Guest that
// stops reading must not be able to block the session lock with it.
func (s *Session) holdLocked(env *pb.Envelope, c *clientConn) (reject string) {
	n := int64(proto.Size(env))
	full := ""
	switch {
	case len(s.held) >= maxHeldRequests:
		full = fmt.Sprintf("%d requests are already waiting", len(s.held))
	case s.heldBytes+n > s.maxHeldBytes():
		full = fmt.Sprintf("%d MiB of requests are already waiting", s.heldBytes>>20)
	case s.srv.heldBytes.Add(n) > maxHeldBytesTotal:
		s.srv.heldBytes.Add(-n)
		full = "the server is holding too many requests from all Guests"
	}
	if full != "" {
		if !s.heldFullLogged {
			s.heldFullLogged = true
			s.srv.audit.Write(audit.Record{Identity: s.Identity.Name, ID: s.Identity.ID, Session: s.hex(),
				Event: "held_queue_full", Fields: map[string]any{"requests": len(s.held), "bytes": s.heldBytes}})
		}
		return "the Owner has paused execution and " + full + "; send this request again after execution resumes"
	}
	s.held = append(s.held, heldReq{env: env, conn: c, size: n})
	s.heldBytes += n
	return ""
}

// release forwards the held requests in order. Requests that arrive while it
// runs are queued behind them (s.releasing), so nothing overtakes a held one.
func (s *Session) release() {
	s.mu.Lock()
	if len(s.held) == 0 {
		s.heldFullLogged = false
		s.mu.Unlock()
		return
	}
	s.releasing = true
	s.mu.Unlock()
	for {
		s.mu.Lock()
		if len(s.held) == 0 || s.closed {
			s.releasing = false
			s.heldFullLogged = false
			s.mu.Unlock()
			return
		}
		h := s.held[0]
		s.held[0] = heldReq{}
		s.held = s.held[1:]
		s.heldBytes -= h.size
		s.mu.Unlock()
		s.srv.heldBytes.Add(-h.size)
		_ = s.dispatch(h.conn, h.env)
	}
}

// dropHeldLocked empties the queue of a session that ended and returns what
// was in it, so the caller can answer each request. s.mu is held.
func (s *Session) dropHeldLocked() []heldReq {
	held := s.held
	s.held, s.releasing = nil, false
	s.srv.heldBytes.Add(-s.heldBytes)
	s.heldBytes = 0
	return held
}
