// SPDX-License-Identifier: Apache-2.0

package sandbox

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"runtime"
	"strings"

	"golang.org/x/sys/unix"
)

// Worker file descriptor layout (shared with server/worker/reflop_worker).
const (
	FDControl = 3 // SOCK_STREAM socketpair: framed WorkerEnvelope messages
	FDPass    = 4 // SOCK_SEQPACKET socketpair: dataset/snapshot fds
	FDConfig  = 5 // pipe: ExecConfig JSON, read and closed by the helper
)

// HelperArg is the hidden reflopd subcommand that applies pre-exec layers.
const HelperArg = "__sandbox-exec"

// ExecConfig is what reflopd hands the sandbox helper.
type ExecConfig struct {
	Argv  []string   `json:"argv"`
	Env   []string   `json:"env"`
	Dir   string     `json:"dir"`
	Rules []PathRule `json:"rules"`
	// AllowProcSelf adds read and write access to the worker's own
	// /proc/<pid>: the CUDA driver names its threads through
	// /proc/self/task/<tid>/comm and fails to initialize without it.
	AllowProcSelf   bool   `json:"allow_proc_self"`
	RequireLandlock bool   `json:"require_landlock"`
	NoFile          uint64 `json:"nofile"`
	NProc           uint64 `json:"nproc"` // 0 = leave unchanged
	FSize           uint64 `json:"fsize"` // 0 = leave unchanged
}

// HelperReport is printed by the helper on stderr as one line prefixed
// with ReportPrefix before exec, so reflopd can show what was applied.
type HelperReport struct {
	LandlockABI int      `json:"landlock_abi"`
	Landlock    bool     `json:"landlock"`
	LandlockNet bool     `json:"landlock_net"`
	Scoped      bool     `json:"landlock_scoped"`
	NoNewPrivs  bool     `json:"no_new_privs"`
	Skipped     []string `json:"skipped,omitempty"`
	Error       string   `json:"error,omitempty"`
}

// ReportPrefix marks the helper's report line on stderr.
const ReportPrefix = "reflop-sandbox-report "

// RunHelper is the body of `reflopd __sandbox-exec`. It never returns on
// success: it execs the worker. It runs in a fresh process forked by
// reflopd, possibly inside new user and network namespaces.
func RunHelper() {
	runtime.LockOSThread()
	rep := HelperReport{}
	fail := func(err error) {
		rep.Error = err.Error()
		writeReport(rep)
		os.Exit(111)
	}
	f := os.NewFile(FDConfig, "config")
	raw, err := io.ReadAll(io.LimitReader(f, 1<<20))
	f.Close()
	if err != nil {
		fail(fmt.Errorf("read config: %w", err))
	}
	var cfg ExecConfig
	if err := json.Unmarshal(raw, &cfg); err != nil {
		fail(fmt.Errorf("parse config: %w", err))
	}
	if len(cfg.Argv) == 0 {
		fail(errors.New("empty argv"))
	}
	// Close every inherited descriptor except stdio, control and fd-pass.
	if err := unix.CloseRange(FDConfig, ^uint(0), 0); err != nil {
		for fd := FDConfig; fd < 4096; fd++ {
			unix.Close(fd)
		}
	}
	setLimit := func(res int, v uint64) {
		if v == 0 {
			return
		}
		var cur unix.Rlimit
		_ = unix.Getrlimit(res, &cur)
		lim := unix.Rlimit{Cur: v, Max: v}
		if cur.Max < v {
			lim = unix.Rlimit{Cur: cur.Max, Max: cur.Max}
		}
		_ = unix.Setrlimit(res, &lim)
	}
	_ = unix.Setrlimit(unix.RLIMIT_CORE, &unix.Rlimit{})
	setLimit(unix.RLIMIT_NOFILE, cfg.NoFile)
	setLimit(unix.RLIMIT_NPROC, cfg.NProc)
	setLimit(unix.RLIMIT_FSIZE, cfg.FSize)
	// Never RLIMIT_AS: CUDA reserves huge virtual ranges (PLAN.md §4.3).

	rules := cfg.Rules
	if cfg.AllowProcSelf {
		rules = append(rules, PathRule{Path: fmt.Sprintf("/proc/%d", os.Getpid()), Access: AccessRead | AccessWrite})
	}
	if err := unix.Prctl(unix.PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0); err != nil {
		fail(fmt.Errorf("no_new_privs: %w", err))
	}
	rep.NoNewPrivs = true
	ll, err := RestrictSelf(rules)
	rep.LandlockABI = ll.ABI
	rep.Landlock = ll.FS
	rep.LandlockNet = ll.Net
	rep.Scoped = ll.Scoped
	rep.Skipped = ll.Skipped
	if err != nil {
		if cfg.RequireLandlock {
			fail(err)
		}
		rep.Error = "landlock: " + err.Error()
	}
	writeReport(rep)
	if cfg.Dir != "" {
		if err := unix.Chdir(cfg.Dir); err != nil {
			fail(fmt.Errorf("chdir: %w", err))
		}
	}
	err = unix.Exec(cfg.Argv[0], cfg.Argv, cfg.Env)
	rep.Error = "exec: " + err.Error()
	writeReport(rep)
	os.Exit(112)
}

func writeReport(r HelperReport) {
	b, _ := json.Marshal(r)
	fmt.Fprintf(os.Stderr, "%s%s\n", ReportPrefix, b)
}

// ParseReport extracts a HelperReport from a stderr line.
func ParseReport(line string) (HelperReport, bool) {
	var r HelperReport
	if !strings.HasPrefix(line, ReportPrefix) {
		return r, false
	}
	if json.Unmarshal([]byte(strings.TrimPrefix(line, ReportPrefix)), &r) != nil {
		return r, false
	}
	return r, true
}
