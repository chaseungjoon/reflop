// SPDX-License-Identifier: Apache-2.0

// Package audit keeps an append-only, per-identity log of what Guests did
// (PLAN.md §9.9): registrations, executes, transfers, validation failures,
// rail hits and kills. It never records tensor contents.
package audit

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

// Record is one audit entry.
type Record struct {
	Time     time.Time      `json:"time"`
	Identity string         `json:"identity"` // name
	ID       string         `json:"id,omitempty"`
	Session  string         `json:"session,omitempty"`
	Event    string         `json:"event"`
	Fields   map[string]any `json:"fields,omitempty"`
}

// Log writes records to <dir>/<YYYY-MM-DD>.jsonl (mode 0600).
type Log struct {
	mu        sync.Mutex
	dir       string
	day       string
	f         *os.File
	retention int
}

// Open opens the audit directory.
func Open(dir string, retentionDays int) (*Log, error) {
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return nil, err
	}
	l := &Log{dir: dir, retention: retentionDays}
	l.prune()
	return l, nil
}

// Write appends a record; errors are swallowed (auditing must not stop
// service) but reported via the returned error for tests.
func (l *Log) Write(r Record) error {
	if l == nil {
		return nil
	}
	if r.Time.IsZero() {
		r.Time = time.Now().UTC()
	}
	b, err := json.Marshal(r)
	if err != nil {
		return err
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	day := r.Time.Format("2006-01-02")
	if l.f == nil || day != l.day {
		if l.f != nil {
			l.f.Close()
		}
		f, err := os.OpenFile(filepath.Join(l.dir, day+".jsonl"), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
		if err != nil {
			return err
		}
		l.f, l.day = f, day
		go l.prune()
	}
	_, err = l.f.Write(append(b, '\n'))
	return err
}

// Close closes the current file.
func (l *Log) Close() {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.f != nil {
		l.f.Close()
		l.f = nil
	}
}

func (l *Log) prune() {
	if l.retention <= 0 {
		return
	}
	cutoff := time.Now().UTC().AddDate(0, 0, -l.retention).Format("2006-01-02")
	ents, _ := os.ReadDir(l.dir)
	for _, e := range ents {
		n := e.Name()
		if strings.HasSuffix(n, ".jsonl") && strings.TrimSuffix(n, ".jsonl") < cutoff {
			os.Remove(filepath.Join(l.dir, n))
		}
	}
}

// Query reads records, newest last, optionally for one identity name.
func Query(dir, identity string, since time.Time, limit int) ([]Record, error) {
	ents, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var files []string
	for _, e := range ents {
		if strings.HasSuffix(e.Name(), ".jsonl") && e.Name() >= since.UTC().Format("2006-01-02") {
			files = append(files, e.Name())
		}
	}
	sort.Strings(files)
	var out []Record
	for _, fn := range files {
		f, err := os.Open(filepath.Join(dir, fn))
		if err != nil {
			continue
		}
		sc := bufio.NewScanner(f)
		sc.Buffer(make([]byte, 64<<10), 4<<20)
		for sc.Scan() {
			var r Record
			if json.Unmarshal(sc.Bytes(), &r) != nil {
				continue
			}
			if identity != "" && r.Identity != identity {
				continue
			}
			if r.Time.Before(since) {
				continue
			}
			out = append(out, r)
		}
		f.Close()
	}
	if limit > 0 && len(out) > limit {
		out = out[len(out)-limit:]
	}
	return out, nil
}
