// SPDX-License-Identifier: Apache-2.0

package server

import (
	"net"
	"sync"
	"time"
)

// Limits for peers that have not authenticated yet (SEC_SWEEP M1, L8).
const (
	// preAuthMaxFrame bounds every frame before AuthResult. Hello, Pair and
	// Authenticate are a few hundred bytes.
	preAuthMaxFrame = 64 << 10
	// maxPreAuthPerSource is how many connections one source address may
	// hold open while still in the TLS or reflop handshake.
	maxPreAuthPerSource = 16
	// Failed Pair/Authenticate audit records: a burst per source address,
	// then one per failRefill; a global cap bounds the total across sources.
	failBurst        = 10
	failRefill       = 30 * time.Second
	failGlobalBurst  = 120
	failGlobalRefill = time.Second
)

// admission tracks one accepted connection from accept to close, so
// connections count against limits.max_connections while they handshake.
type admission struct {
	s       *Server
	host    string // "" when the per-source limit does not apply (SSH bridge)
	preauth sync.Once
	done    sync.Once
}

// admit reserves a connection slot, or returns nil if the server is full or
// the source already has too many unauthenticated connections.
func (s *Server) admit(remote string, perSource bool) *admission {
	host := ""
	if perSource {
		host = remote
		if h, _, err := net.SplitHostPort(remote); err == nil {
			host = h
		}
	}
	max := int(s.cfg().Limits.MaxConnections)
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.nconns >= max {
		return nil
	}
	if host != "" {
		if s.preauth[host] >= maxPreAuthPerSource {
			return nil
		}
		s.preauth[host]++
	}
	s.nconns++
	return &admission{s: s, host: host}
}

// authenticated releases the pre-authentication slot.
func (a *admission) authenticated() {
	if a == nil {
		return
	}
	a.preauth.Do(func() {
		if a.host == "" {
			return
		}
		a.s.mu.Lock()
		if a.s.preauth[a.host]--; a.s.preauth[a.host] <= 0 {
			delete(a.s.preauth, a.host)
		}
		a.s.mu.Unlock()
	})
}

// release frees the connection slot.
func (a *admission) release() {
	if a == nil {
		return
	}
	a.authenticated()
	a.done.Do(func() {
		a.s.mu.Lock()
		a.s.nconns--
		a.s.mu.Unlock()
	})
}

// failLimiter rate-limits audit records written for unauthenticated peers,
// so failed handshakes cannot grow the audit log without bound (L8).
type failLimiter struct {
	mu         sync.Mutex
	perSource  map[string]*tokens
	global     tokens
	suppressed uint64
}

type tokens struct {
	n    float64
	last time.Time
}

func (t *tokens) take(now time.Time, burst float64, refill time.Duration) bool {
	if t.last.IsZero() {
		t.n = burst
	} else {
		t.n += float64(now.Sub(t.last)) / float64(refill)
		if t.n > burst {
			t.n = burst
		}
	}
	t.last = now
	if t.n < 1 {
		return false
	}
	t.n--
	return true
}

// allow reports whether a failure record for remote may be written, and how
// many earlier records were suppressed (to note in the record).
func (f *failLimiter) allow(remote string, now time.Time) (bool, uint64) {
	host := remote
	if h, _, err := net.SplitHostPort(remote); err == nil {
		host = h
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.perSource == nil {
		f.perSource = map[string]*tokens{}
	}
	if len(f.perSource) > 4096 {
		// Forget sources whose bucket has refilled; they lose nothing.
		for k, t := range f.perSource {
			if now.Sub(t.last) > failBurst*failRefill {
				delete(f.perSource, k)
			}
		}
	}
	t := f.perSource[host]
	if t == nil {
		t = &tokens{}
		f.perSource[host] = t
	}
	if !t.take(now, failBurst, failRefill) || !f.global.take(now, failGlobalBurst, failGlobalRefill) {
		f.suppressed++
		return false, 0
	}
	n := f.suppressed
	f.suppressed = 0
	return true, n
}
