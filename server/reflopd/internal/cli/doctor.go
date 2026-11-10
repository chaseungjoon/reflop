// SPDX-License-Identifier: Apache-2.0

package cli

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"

	"github.com/spf13/cobra"

	"github.com/chaseungjoon/reflop/server/reflopd/internal/cgroup"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/config"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/gpu"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/rails"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/runtime"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/sandbox"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/server"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/sshpipe"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/supervisor"
)

func doctorCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "doctor",
		Short: "Explain which protections are active on this machine",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			return runDoctor(cmd.OutOrStdout())
		},
	}
}

type report struct {
	out      io.Writer
	critical int
	warn     int
}

func (r *report) ok(f string, a ...any)   { fmt.Fprintf(r.out, "  ok    "+f+"\n", a...) }
func (r *report) info(f string, a ...any) { fmt.Fprintf(r.out, "  info  "+f+"\n", a...) }
func (r *report) w(f string, a ...any)    { r.warn++; fmt.Fprintf(r.out, "  warn  "+f+"\n", a...) }
func (r *report) bad(f string, a ...any)  { r.critical++; fmt.Fprintf(r.out, "  FAIL  "+f+"\n", a...) }

func runDoctor(out io.Writer) error {
	r := &report{out: out}
	l, err := layout()
	if err != nil {
		return err
	}
	cfg, err := config.Load(l.ConfigFile(), settingsPath(l))
	if err != nil {
		r.bad("config: %v", err)
	}
	level := 0
	if l.System {
		level = 1
	}
	fmt.Fprintf(out, "deployment\n")
	if level == 0 {
		r.info("level 0 (rootless): reflopd and workers run as your user; a sandbox escape would land in your account. `sudo reflopd install --system` moves them to a dedicated account")
	} else {
		r.ok("level 1: dedicated system user, hardened systemd unit")
	}
	if os.Geteuid() == 0 && !l.System {
		r.bad("running as root: run reflopd as a normal user (level 0) or install the system service (level 1)")
	}

	fmt.Fprintf(out, "sandbox layers (workers)\n")
	abi := sandbox.LandlockABI()
	switch {
	case abi >= 6:
		r.ok("Landlock ABI %d: filesystem confined, TCP denied, abstract sockets and signals scoped", abi)
	case abi >= 4:
		r.ok("Landlock ABI %d: filesystem confined, TCP denied", abi)
	case abi >= 1:
		r.w("Landlock ABI %d: filesystem confined; no network rules (kernel < 6.7)", abi)
	default:
		r.w("Landlock unavailable (kernel < 5.13, or disabled: add landlock to the lsm= boot parameter)")
	}
	if sandbox.SeccompAvailable() {
		r.ok("seccomp-BPF: no exec, no fork, no new sockets, no ptrace/mount/bpf after warm-up")
	} else {
		r.w("seccomp unavailable")
	}
	if abi == 0 && !sandbox.SeccompAvailable() {
		r.bad("neither Landlock nor seccomp works: reflopd refuses Guest sessions unless started with --accept-weak-isolation")
	}
	if supervisor.UserNSAvailable() {
		r.ok("unprivileged user namespaces: workers also get an empty network namespace")
	} else {
		r.info("unprivileged user namespaces are restricted (Ubuntu 24.04 default); seccomp and Landlock cover networking instead")
	}
	if self, err := cgroup.Self(); err == nil {
		if err := cgroup.Delegated(self); err == nil {
			r.ok("cgroup v2 delegated (%s): memory and pids rails are kernel-enforced", self)
		} else if systemdUser() {
			r.info("this shell's cgroup is not delegated; `reflopd serve` asks systemd for a delegated scope, and the user service has Delegate=yes")
		} else {
			r.w("no delegated cgroup and no systemd user manager: rails fall back to RLIMIT_NPROC and memory polling")
		}
	}

	fmt.Fprintf(out, "safety rails\n")
	if u, err := rails.Disk(l.State); err == nil {
		floor := rails.DiskFloor(cfg.Rails, u.Total)
		if u.Free < floor {
			r.w("disk: %s free on the store filesystem, below the %s floor; uploads are refused", humanBytes(u.Free), humanBytes(floor))
		} else {
			r.ok("disk: %s free, floor %s (max of %d%% and %s)", humanBytes(u.Free), humanBytes(floor), cfg.Rails.DiskFreeFloorPercent, cfg.Rails.DiskFreeFloorBytes)
		}
	}
	if m, err := rails.Memory(); err == nil {
		r.ok("host RAM: %s total, floor %s kept available for you", humanBytes(m.Total), humanBytes(rails.RAMFloor(cfg.Rails, m.Total)))
	}

	fmt.Fprintf(out, "hardware and runtime\n")
	inf := gpu.Detect()
	for _, n := range inf.Notes {
		r.info("%s", n)
	}
	for _, g := range inf.GPUs {
		r.ok("GPU %d: %s (cc %d.%d, %d MiB), driver %s", g.Index, g.Name, g.CCMajor, g.CCMinor, g.MemoryMiB, inf.Driver)
	}
	lock, err := runtime.Load()
	if err == nil {
		ch, err := lock.Choose(inf, "")
		if err == nil {
			arch, _ := runtime.Arch()
			if inst, ok := runtime.Find(l.Runtime, lock.ID(ch.Variant, arch)); ok {
				r.ok("runtime %s installed (torch %s, CUDA %s)", inst.Marker.ID, inst.Marker.Torch, orNone(inst.Marker.CUDA))
			} else {
				r.w("runtime %s is not installed: run `reflopd up`", ch.Variant)
			}
			for _, w := range ch.Warn {
				r.info("%s", w)
			}
		}
	}
	if glibc := glibcVersion(); glibc != "" {
		r.info("glibc %s (the runtime needs ≥ 2.28)", glibc)
	}

	fmt.Fprintf(out, "network\n")
	for _, a := range cfg.Listen {
		if strings.HasPrefix(a, "127.") || strings.HasPrefix(a, "localhost") || strings.HasPrefix(a, "[::1]") {
			r.ok("listens on %s (loopback only)", a)
		} else {
			r.info("listens on %s: reachable from the network; prefer a Tailscale/WireGuard address", a)
		}
	}
	if _, err := server.ControlClient(l, server.ControlRequest{Cmd: "status"}, 0); err == nil {
		r.ok("the daemon is running")
	} else {
		r.info("the daemon is not running (`reflopd up`)")
	}

	fmt.Fprintf(out, "ssh pipe mode\n")
	if ak, err := sshpipe.AuthorizedKeysPath(); err == nil {
		fs, err := sshpipe.Scan(ak)
		if err != nil {
			r.w("cannot read %s: %v", ak, err)
		}
		if len(fs) == 0 {
			r.ok("%s: no unsafe reflop keys", ak)
		}
		for _, f := range fs {
			r.bad("%s line %d: %s", ak, f.Line, f.Msg)
		}
	}
	fmt.Fprintf(out, "\n%d critical, %d warnings\n", r.critical, r.warn)
	if r.critical > 0 {
		return exitWith(ExitUnsafe, fmt.Errorf("%d critical problem(s)", r.critical))
	}
	return nil
}

func orNone(s string) string {
	if s == "" {
		return "none"
	}
	return s
}

func glibcVersion() string {
	out, err := exec.Command("ldd", "--version").Output()
	if err != nil {
		return ""
	}
	l, _, _ := strings.Cut(string(out), "\n")
	f := strings.Fields(l)
	if len(f) == 0 {
		return ""
	}
	return f[len(f)-1]
}
