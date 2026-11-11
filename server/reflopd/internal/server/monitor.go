// SPDX-License-Identifier: Apache-2.0

package server

import (
	"context"
	"os/exec"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	pb "github.com/chaseungjoon/reflop/pkg/reflopv1"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/audit"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/rails"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/store"
)

// monitor enforces the host-RAM rail, session TTLs, the optional watchdog
// and the optional auto-yield.
func (s *Server) monitor(ctx context.Context) {
	tick := time.NewTicker(500 * time.Millisecond)
	defer tick.Stop()
	var lastYield, lastSweep time.Time
	for {
		select {
		case <-ctx.Done():
			return
		case <-tick.C:
		}
		if time.Since(lastSweep) > time.Hour {
			lastSweep = time.Now()
			go s.store.SweepStale(store.UploadTTL) // abandoned uploads (L4)
		}
		s.ramRail()
		s.reapDetached()
		s.watchdog()
		cfg := s.cfg()
		if cfg.Yield.Auto && time.Since(lastYield) >= time.Duration(cfg.Yield.Interval) {
			lastYield = time.Now()
			s.autoYield()
		} else if !cfg.Yield.Auto && s.yielding.Load() {
			s.yielding.Store(false)
			s.Resume("auto-yield off")
		}
	}
}

// ramRail keeps the host RAM floor available for the Owner (§9.6).
func (s *Server) ramRail() {
	m, err := rails.Memory()
	if err != nil {
		// Fail closed (L5): with no reading, workers may not grow, and
		// spawnWorker refuses new ones.
		if s.ramErrLogged.CompareAndSwap(false, true) {
			s.logf("host RAM rail: %v; freezing worker memory and refusing new sessions", err)
		}
		if s.cg != nil {
			s.cg.SetMemoryMax(max(s.workersMemory(), 64<<20))
		}
		return
	}
	s.ramErrLogged.Store(false)
	r := s.cfg().Rails
	using := s.workersMemory()
	budget := rails.WorkerBudget(r, m, using)
	if s.cg != nil {
		if ceiling := m.Total - rails.RAMFloor(r, m.Total); ceiling > 512<<20 {
			s.cg.SetDaemonMemoryMax(ceiling)
		}
		// The kernel enforces it: an over-budget worker is OOM-killed inside
		// its own cgroup, never the Owner's processes.
		if budget < 64<<20 {
			budget = 64 << 20
		}
		s.cg.SetMemoryMax(budget)
		return
	}
	if m.Available >= rails.RAMFloor(r, m.Total) {
		return
	}
	// No cgroup: kill the largest worker.
	sessions := s.allSessions()
	sort.Slice(sessions, func(i, j int) bool { return sessions[i].worker.Memory() > sessions[j].worker.Memory() })
	if len(sessions) > 0 && sessions[0].worker.Memory() > 0 {
		victim := sessions[0]
		s.audit.Write(audit.Record{Identity: victim.Identity.Name, ID: victim.Identity.ID, Session: victim.hex(),
			Event: "rail_hit", Fields: map[string]any{"rail": "host_ram", "available": m.Available}})
		s.logf("host RAM floor reached (%d MiB available): killing session %s of %q", m.Available>>20, victim.hex()[:8], victim.Identity.Name)
		go victim.close("host memory safety rail")
	}
}

func (s *Server) reapDetached() {
	ttl := time.Duration(s.cfg().Session.TTL)
	for _, sess := range s.allSessions() {
		sess.mu.Lock()
		expired := sess.conn == nil && !sess.detached.IsZero() && time.Since(sess.detached) > ttl
		sess.mu.Unlock()
		if expired {
			go sess.close("session expired after disconnect")
		}
	}
}

func (s *Server) watchdog() {
	for _, sess := range s.allSessions() {
		q := s.quota(sess.Identity.Name)
		if q.ExecuteWallMS == 0 {
			continue
		}
		limit := time.Duration(q.ExecuteWallMS) * time.Millisecond
		sess.mu.Lock()
		over := false
		for _, r := range sess.routes {
			if r.kind == "execute" && time.Since(r.start) > limit {
				over = true
			}
		}
		sess.mu.Unlock()
		if over {
			s.audit.Write(audit.Record{Identity: sess.Identity.Name, ID: sess.Identity.ID, Session: sess.hex(),
				Event: "watchdog", Fields: map[string]any{"limit_ms": q.ExecuteWallMS}})
			go sess.close("execute exceeded the Owner's wall-clock quota (CUDA kernels cannot be preempted)")
		}
	}
}

// autoYield pauses Guests while non-reflop processes use the GPU (§9.7).
func (s *Server) autoYield() {
	busy := s.foreignGPUProcesses()
	switch {
	case busy && !s.yielding.Load():
		s.yielding.Store(true)
		s.paused.Store(true)
		s.events.publish(pb.Event_KIND_YIELDING, "the Owner is using the GPU; Guest work is paused")
		s.audit.Write(audit.Record{Identity: "owner", Event: "yield", Fields: map[string]any{"busy": true}})
	case !busy && s.yielding.Load():
		s.yielding.Store(false)
		s.Resume("auto-yield")
	}
}

func (s *Server) foreignGPUProcesses() bool {
	out, err := exec.Command("nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader").Output()
	if err != nil {
		return false
	}
	ours := map[int]bool{}
	for _, sess := range s.allSessions() {
		ours[sess.worker.PID()] = true
	}
	for _, f := range strings.Fields(string(out)) {
		pid, err := strconv.Atoi(strings.TrimSpace(f))
		if err == nil && !ours[pid] {
			return true
		}
	}
	return false
}

// bucket is a simple token bucket for the optional upload-bandwidth quota.
type bucket struct {
	mu   sync.Mutex
	last time.Time
	debt float64
}

func (s *Server) throttle(id string, n, rate uint64) {
	s.mu.Lock()
	b := s.throttles[id]
	if b == nil {
		b = &bucket{last: time.Now()}
		s.throttles[id] = b
	}
	s.mu.Unlock()
	b.mu.Lock()
	now := time.Now()
	b.debt -= now.Sub(b.last).Seconds() * float64(rate)
	if b.debt < 0 {
		b.debt = 0
	}
	b.last = now
	b.debt += float64(n)
	wait := time.Duration(b.debt / float64(rate) * float64(time.Second))
	b.mu.Unlock()
	if wait > 0 {
		time.Sleep(wait)
	}
}
