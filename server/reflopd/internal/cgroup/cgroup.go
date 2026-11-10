// SPDX-License-Identifier: Apache-2.0

// Package cgroup manages worker cgroups inside a delegated cgroup v2
// subtree (systemd user service or scope with Delegate=yes, or the Level 1
// system unit). It is used only for safety rails: host memory and pids.
package cgroup

import (
	"bufio"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"golang.org/x/sys/unix"
)

const fsRoot = "/sys/fs/cgroup"

// Manager owns <delegated>/workers/*.
type Manager struct {
	Root string // delegated cgroup directory
}

// Self returns this process's cgroup v2 directory.
func Self() (string, error) {
	f, err := os.Open("/proc/self/cgroup")
	if err != nil {
		return "", err
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		if p, ok := strings.CutPrefix(sc.Text(), "0::"); ok {
			return filepath.Join(fsRoot, p), nil
		}
	}
	return "", errors.New("not on a cgroup v2 unified hierarchy")
}

// Delegated reports whether dir is writable by us with memory and pids
// controllers available.
func Delegated(dir string) error {
	if err := unix.Access(dir, unix.W_OK); err != nil {
		return fmt.Errorf("%s is not delegated to this user", dir)
	}
	for _, f := range []string{"cgroup.procs", "cgroup.subtree_control"} {
		if err := unix.Access(filepath.Join(dir, f), unix.W_OK); err != nil {
			return fmt.Errorf("%s is not delegated to this user", dir)
		}
	}
	b, err := os.ReadFile(filepath.Join(dir, "cgroup.controllers"))
	if err != nil {
		return err
	}
	have := strings.Fields(string(b))
	for _, want := range []string{"memory", "pids"} {
		found := false
		for _, h := range have {
			found = found || h == want
		}
		if !found {
			return fmt.Errorf("cgroup controller %q is not delegated", want)
		}
	}
	return nil
}

// Setup prepares the delegated subtree: reflopd moves into <root>/daemon
// (cgroup v2 forbids processes in inner nodes), and workers get
// <root>/workers/<name> leaves.
func Setup() (*Manager, error) {
	root, err := Self()
	if err != nil {
		return nil, err
	}
	if filepath.Base(root) == "daemon" {
		root = filepath.Dir(root) // restarted inside an existing layout
	}
	if err := Delegated(root); err != nil {
		return nil, err
	}
	daemon := filepath.Join(root, "daemon")
	if err := os.MkdirAll(daemon, 0o755); err != nil {
		return nil, err
	}
	// Move every process currently in root (normally just us) into daemon.
	if b, err := os.ReadFile(filepath.Join(root, "cgroup.procs")); err == nil {
		for _, pid := range strings.Fields(string(b)) {
			_ = os.WriteFile(filepath.Join(daemon, "cgroup.procs"), []byte(pid), 0)
		}
	}
	if err := write(root, "cgroup.subtree_control", "+memory +pids"); err != nil {
		return nil, err
	}
	workers := filepath.Join(root, "workers")
	if err := os.MkdirAll(workers, 0o755); err != nil {
		return nil, err
	}
	if err := write(workers, "cgroup.subtree_control", "+memory +pids"); err != nil {
		return nil, err
	}
	return &Manager{Root: root}, nil
}

func write(dir, file, val string) error {
	if err := os.WriteFile(filepath.Join(dir, file), []byte(val), 0); err != nil {
		return fmt.Errorf("write %s/%s: %w", dir, file, err)
	}
	return nil
}

func (m *Manager) workers() string { return filepath.Join(m.Root, "workers") }

// Worker is one worker cgroup.
type Worker struct {
	Dir string
	FD  int // open directory fd, for CLONE_INTO_CGROUP
}

// NewWorker creates a leaf cgroup for one worker.
func (m *Manager) NewWorker(name string, maxPIDs uint32) (*Worker, error) {
	dir := filepath.Join(m.workers(), name)
	if err := os.Mkdir(dir, 0o755); err != nil {
		return nil, err
	}
	if err := write(dir, "pids.max", strconv.FormatUint(uint64(maxPIDs), 10)); err != nil {
		os.Remove(dir)
		return nil, err
	}
	// Kill the whole worker on OOM, never just one thread of it.
	_ = write(dir, "memory.oom.group", "1")
	_ = write(dir, "memory.swap.max", "0")
	fd, err := unix.Open(dir, unix.O_RDONLY|unix.O_DIRECTORY|unix.O_CLOEXEC, 0)
	if err != nil {
		os.Remove(dir)
		return nil, err
	}
	return &Worker{Dir: dir, FD: fd}, nil
}

// SetMemoryMax caps all workers together.
func (m *Manager) SetMemoryMax(bytes uint64) error {
	v := "max"
	if bytes > 0 {
		v = strconv.FormatUint(bytes, 10)
	}
	return write(m.workers(), "memory.max", v)
}

// MemoryCurrent returns the memory used by all workers.
func (m *Manager) MemoryCurrent() uint64 {
	return readUint(filepath.Join(m.workers(), "memory.current"))
}

// MemoryCurrent returns the worker's memory use.
func (w *Worker) MemoryCurrent() uint64 { return readUint(filepath.Join(w.Dir, "memory.current")) }

// Events returns the oom_kill count.
func (w *Worker) OOMKills() uint64 {
	b, err := os.ReadFile(filepath.Join(w.Dir, "memory.events"))
	if err != nil {
		return 0
	}
	for _, l := range strings.Split(string(b), "\n") {
		if v, ok := strings.CutPrefix(l, "oom_kill "); ok {
			n, _ := strconv.ParseUint(strings.TrimSpace(v), 10, 64)
			return n
		}
	}
	return 0
}

// Kill kills every process in the worker cgroup.
func (w *Worker) Kill() { _ = write(w.Dir, "cgroup.kill", "1") }

// Remove deletes the (empty) cgroup.
func (w *Worker) Remove() {
	unix.Close(w.FD)
	for i := 0; i < 50; i++ {
		if os.Remove(w.Dir) == nil || errors.Is(os.Remove(w.Dir), os.ErrNotExist) {
			return
		}
		w.Kill()
		unix.Nanosleep(&unix.Timespec{Nsec: 20e6}, nil)
	}
}

func readUint(p string) uint64 {
	b, err := os.ReadFile(p)
	if err != nil {
		return 0
	}
	n, _ := strconv.ParseUint(strings.TrimSpace(string(b)), 10, 64)
	return n
}
