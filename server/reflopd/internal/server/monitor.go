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

	pb "github.com/reflopai/reflop/pkg/reflopv1"
	"github.com/reflopai/reflop/server/reflopd/internal/audit"
	"github.com/reflopai/reflop/server/reflopd/internal/rails"
	"github.com/reflopai/reflop/server/reflopd/internal/store"
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
			go s.sweepCheckpoints()
		}
		s.ramRail()
		s.retryPending(ctx)
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

// minWorkerMemory is the smallest memory limit reflopd sets (shared or per
// worker), so a nearly full machine never gets a limit of zero.
const minWorkerMemory = 64 << 20

// workerHeadroom is the memory all workers together may still add.
func workerHeadroom(budget, using uint64) uint64 {
	if budget > using {
		return budget - using
	}
	return 0
}

// workerLimit is a worker's own memory.max: what it uses plus the pool's
// free headroom, less a margin. A worker that grows into the margin hits its
// own limit, so the kernel kills that worker, not whichever worker is
// biggest when the shared limit is reached (issue #16). Without the margin
// the two limits coincide, and anything another worker adds before the next
// tick (500 ms) makes the shared limit fail first. One worker alone can still
// take most of the pool in one allocation, and all of it step by step, since
// the limit follows its use every tick. A suspended worker has no limit of
// its own (0): suspending moves its GPU memory into its host RAM, which
// suspendAll has already checked against the headroom.
func workerLimit(own, headroom uint64, suspended bool) uint64 {
	if suspended {
		return 0
	}
	margin := min(max(headroom/8, workerLimitMargin), headroom)
	return max(own+headroom-margin, minWorkerMemory)
}

// workerLimitMargin is the least of the pool's free headroom kept out of
// every worker's own limit (see workerLimit).
const workerLimitMargin = 32 << 20

// setWorkerLimits gives every worker its own limit (see workerLimit).
func (s *Server) setWorkerLimits(headroom uint64) {
	for _, sess := range s.allSessions() {
		sess.mu.Lock()
		w, suspended := sess.worker, sess.suspended
		sess.mu.Unlock()
		if w == nil {
			continue
		}
		_ = w.SetMemoryMax(workerLimit(w.Memory(), headroom, suspended))
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
			s.cg.SetMemoryMax(max(s.workersMemory(), minWorkerMemory))
			s.setWorkerLimits(0)
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
		s.cg.SetMemoryMax(max(budget, minWorkerMemory))
		s.setWorkerLimits(workerHeadroom(budget, using))
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
		if sess.suspended {
			sess.mu.Unlock()
			continue // the Owner suspended it: its executes are not running
		}
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
		if s.cfg().Yield.Suspend {
			// Owner option (ADR-030): also give the GPU memory back. A
			// session that cannot be suspended stays paused, not killed.
			go s.suspendAll(false, "auto-yield")
		}
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
