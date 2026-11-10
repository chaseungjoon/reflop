// SPDX-License-Identifier: Apache-2.0

// Package usage accounts how much compute each Guest identity has used.
//
// The primary measure is device time: the time a worker spent running the
// identity's executes, as measured on the worker. It is what Guests actually
// consume (like cloud GPU-hours) and, unlike FLOPs, it is measured rather
// than estimated and charges memory-bound work fairly. Estimated FLOPs,
// executes, sessions, session (held) time and bytes transferred are kept
// alongside it.
package usage

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/chaseungjoon/reflop/server/reflopd/internal/audit"
)

// Totals is the cumulative usage of one identity.
type Totals struct {
	Executes   uint64            `json:"executes"`
	DeviceUs   uint64            `json:"device_us"`
	ByDevice   map[string]uint64 `json:"device_us_by_device,omitempty"`
	FLOPs      float64           `json:"flops"`
	PeakMemory uint64            `json:"peak_memory_bytes"`
	Sessions   uint64            `json:"sessions"`
	SessionUs  uint64            `json:"session_us"`
	BytesUp    uint64            `json:"bytes_up"`
	BytesDown  uint64            `json:"bytes_down"`
	First      time.Time         `json:"first,omitempty"`
	Last       time.Time         `json:"last,omitempty"`
}

func (t *Totals) seen(at time.Time) {
	if t.First.IsZero() || at.Before(t.First) {
		t.First = at
	}
	if at.After(t.Last) {
		t.Last = at
	}
}

func (t *Totals) execute(device string, us uint64, flops float64, peak uint64, at time.Time) {
	t.Executes++
	t.DeviceUs += us
	if device != "" {
		if t.ByDevice == nil {
			t.ByDevice = map[string]uint64{}
		}
		t.ByDevice[device] += us
	}
	t.FLOPs += flops
	t.PeakMemory = max(t.PeakMemory, peak)
	t.seen(at)
}

func (t *Totals) clone() *Totals {
	c := *t
	if t.ByDevice != nil {
		c.ByDevice = make(map[string]uint64, len(t.ByDevice))
		for k, v := range t.ByDevice {
			c.ByDevice[k] = v
		}
	}
	return &c
}

// Report is a usage snapshot.
type Report struct {
	Since      time.Time          `json:"since"`
	Until      time.Time          `json:"until"`
	Source     string             `json:"source"`
	Identities map[string]*Totals `json:"identities"`
}

// Get returns (creating) the totals of one identity.
func (r *Report) Get(name string) *Totals {
	if r.Identities == nil {
		r.Identities = map[string]*Totals{}
	}
	t := r.Identities[name]
	if t == nil {
		t = &Totals{}
		r.Identities[name] = t
	}
	return t
}

// Ledger is the daemon's persistent, cumulative per-identity account.
type Ledger struct {
	mu    sync.Mutex
	path  string
	rep   Report
	dirty bool
}

type ledgerFile struct {
	Version    int                `json:"version"`
	Since      time.Time          `json:"since"`
	Identities map[string]*Totals `json:"identities"`
}

// Load reads a ledger file.
func Load(path string) (Report, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return Report{}, err
	}
	var f ledgerFile
	if err := json.Unmarshal(b, &f); err != nil {
		return Report{}, err
	}
	if f.Identities == nil {
		f.Identities = map[string]*Totals{}
	}
	return Report{Since: f.Since, Source: "ledger", Identities: f.Identities}, nil
}

// Open loads the ledger at path. The first time (no file yet), it starts from
// what the audit log still remembers so usage before the ledger existed is
// not lost.
func Open(path, auditDir string) (*Ledger, error) {
	l := &Ledger{path: path}
	rep, err := Load(path)
	switch {
	case err == nil:
		l.rep = rep
	case errors.Is(err, os.ErrNotExist):
		// Sessions the log never saw end belonged to a daemon that died;
		// they no longer hold anything.
		l.rep, _ = fromAudit(auditDir, time.Time{}, time.Now(), false)
		if l.rep.Since.IsZero() {
			l.rep.Since = time.Now().UTC()
		}
		l.dirty = true
	default:
		return nil, err
	}
	l.rep.Source = "ledger"
	if l.rep.Identities == nil {
		l.rep.Identities = map[string]*Totals{}
	}
	return l, nil
}

// Execute records one finished execute.
func (l *Ledger) Execute(name, device string, us, flops, peak uint64) {
	if l == nil {
		return
	}
	l.mu.Lock()
	l.rep.Get(name).execute(device, us, float64(flops), peak, time.Now().UTC())
	l.dirty = true
	l.mu.Unlock()
}

// SessionOpen counts a new session.
func (l *Ledger) SessionOpen(name string) {
	if l == nil {
		return
	}
	l.mu.Lock()
	t := l.rep.Get(name)
	t.Sessions++
	t.seen(time.Now().UTC())
	l.dirty = true
	l.mu.Unlock()
}

// SessionEnd adds an ended session's lifetime (time it held a worker).
func (l *Ledger) SessionEnd(name string, held time.Duration) {
	if l == nil {
		return
	}
	l.mu.Lock()
	t := l.rep.Get(name)
	t.SessionUs += uint64(held.Microseconds())
	t.seen(time.Now().UTC())
	l.dirty = true
	l.mu.Unlock()
}

// Transfer adds a finished connection's traffic.
func (l *Ledger) Transfer(name string, up, down uint64) {
	if l == nil || up+down == 0 {
		return
	}
	l.mu.Lock()
	t := l.rep.Get(name)
	t.BytesUp += up
	t.BytesDown += down
	l.dirty = true
	l.mu.Unlock()
}

// Snapshot returns a deep copy of the current totals.
func (l *Ledger) Snapshot() Report {
	l.mu.Lock()
	defer l.mu.Unlock()
	r := Report{Since: l.rep.Since, Until: time.Now().UTC(), Source: l.rep.Source, Identities: map[string]*Totals{}}
	for k, v := range l.rep.Identities {
		r.Identities[k] = v.clone()
	}
	return r
}

// Save writes the ledger if it changed (atomically, mode 0600).
func (l *Ledger) Save() error {
	if l == nil {
		return nil
	}
	l.mu.Lock()
	if !l.dirty {
		l.mu.Unlock()
		return nil
	}
	b, err := json.MarshalIndent(ledgerFile{Version: 1, Since: l.rep.Since, Identities: l.rep.Identities}, "", " ")
	l.dirty = false
	l.mu.Unlock()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(l.path), 0o700); err != nil {
		return err
	}
	tmp := l.path + ".tmp"
	if err := os.WriteFile(tmp, b, 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, l.path)
}

// Run saves the ledger every interval until ctx is done, then once more.
func (l *Ledger) Run(ctx context.Context, every time.Duration) {
	t := time.NewTicker(every)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			l.Save()
			return
		case <-t.C:
			l.Save()
		}
	}
}

func num(v any) float64 {
	f, _ := v.(float64)
	return f
}

// FromAudit totals the audit log between since and until. It covers only
// what the log still retains; sessions still open at until count up to until.
func FromAudit(dir string, since, until time.Time) (Report, error) {
	return fromAudit(dir, since, until, true)
}

func fromAudit(dir string, since, until time.Time, countOpen bool) (Report, error) {
	rep := Report{Since: since.UTC(), Until: until.UTC(), Source: "audit log", Identities: map[string]*Totals{}}
	type open struct {
		name   string
		device string
		start  time.Time
	}
	sessions := map[string]open{}
	var first time.Time
	err := audit.Scan(dir, since, func(r audit.Record) {
		if r.Time.After(until) || r.Identity == "" {
			return
		}
		if first.IsZero() {
			first = r.Time
		}
		switch r.Event {
		case "session_open":
			dev, _ := r.Fields["device"].(string)
			sessions[r.Session] = open{r.Identity, dev, r.Time}
			t := rep.Get(r.Identity)
			t.Sessions++
			t.seen(r.Time)
		case "session_end":
			start := since
			if o, ok := sessions[r.Session]; ok {
				start = o.start
				delete(sessions, r.Session)
			}
			if !start.IsZero() && r.Time.After(start) {
				rep.Get(r.Identity).SessionUs += uint64(r.Time.Sub(start).Microseconds())
			}
		case "execute":
			rep.Get(r.Identity).execute(sessions[r.Session].device, uint64(num(r.Fields["us"])),
				num(r.Fields["flops"]), uint64(num(r.Fields["peak"])), r.Time)
		case "disconnect":
			t := rep.Get(r.Identity)
			t.BytesUp += uint64(num(r.Fields["bytes_up"]))
			t.BytesDown += uint64(num(r.Fields["bytes_down"]))
		}
	})
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return rep, err
	}
	for _, o := range sessions {
		if countOpen && until.After(o.start) {
			rep.Get(o.name).SessionUs += uint64(until.Sub(o.start).Microseconds())
		}
	}
	for name, t := range rep.Identities {
		if t.Executes == 0 && t.Sessions == 0 && t.BytesUp+t.BytesDown == 0 {
			delete(rep.Identities, name)
		}
	}
	if rep.Since.IsZero() && !first.IsZero() {
		rep.Since = first.UTC()
	}
	return rep, nil
}
