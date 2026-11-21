// SPDX-License-Identifier: Apache-2.0

package cgroup

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"golang.org/x/sys/unix"
)

// unixClose closes the directory fd (Remove does it otherwise).
func unixClose(w *Worker) { unix.Close(w.FD) }

func read(t *testing.T, p string) string {
	t.Helper()
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	return strings.TrimSpace(string(b))
}

// The files a worker leaf gets (a plain directory stands in for cgroupfs).
func TestNewWorkerWritesItsOwnMemoryLimit(t *testing.T) {
	root := t.TempDir()
	if err := os.Mkdir(filepath.Join(root, "workers"), 0o755); err != nil {
		t.Fatal(err)
	}
	m := &Manager{Root: root}
	w, err := m.NewWorker("a", 64, 3<<30)
	if err != nil {
		t.Fatal(err)
	}
	if got := read(t, filepath.Join(w.Dir, "memory.max")); got != "3221225472" {
		t.Fatalf("memory.max = %q", got)
	}
	if got := read(t, filepath.Join(w.Dir, "pids.max")); got != "64" {
		t.Fatalf("pids.max = %q", got)
	}
	if err := w.SetMemoryMax(0); err != nil {
		t.Fatal(err)
	}
	if got := read(t, filepath.Join(w.Dir, "memory.max")); got != "max" {
		t.Fatalf("memory.max after SetMemoryMax(0) = %q, want max", got)
	}
	if err := w.SetMemoryMax(123 << 20); err != nil {
		t.Fatal(err)
	}
	if got := read(t, filepath.Join(w.Dir, "memory.max")); got != "128974848" {
		t.Fatalf("memory.max = %q", got)
	}
	unixClose(w)

	w2, err := m.NewWorker("b", 64, 0)
	if err != nil {
		t.Fatal(err)
	}
	if got := read(t, filepath.Join(w2.Dir, "memory.max")); got != "max" {
		t.Fatalf("no limit: memory.max = %q, want max", got)
	}
	unixClose(w2)
}
