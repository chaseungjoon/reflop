// SPDX-License-Identifier: Apache-2.0

package cli

import (
	"context"
	"errors"
	"fmt"
	"log"
	"net"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/spf13/cobra"

	"github.com/chaseungjoon/reflop/server/reflopd/internal/auth"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/cgroup"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/config"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/gpu"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/paths"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/runtime"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/sandbox"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/server"
)

const unitName = "reflopd.service"

type serveFlags struct {
	listen     []string
	weak       bool
	stdio      bool
	identity   string
	foreground bool
	noSandbox  bool
}

func serveCmd() *cobra.Command {
	var f serveFlags
	cmd := &cobra.Command{
		Use:   "serve",
		Short: "Run the daemon in the foreground (used by the service unit)",
		Long: `Run reflopd in the foreground. ` + "`reflopd up`" + ` runs this as a systemd user
service; use it directly to watch logs or in containers without systemd.

  reflopd serve --listen 100.64.0.1:7460      bind a Tailscale address
  reflopd serve --stdio --identity <id>       SSH pipe mode (forced command only)`,
		Args: cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error { return runServe(f) },
	}
	cmd.Flags().StringSliceVar(&f.listen, "listen", nil, "address to listen on (repeatable); default 127.0.0.1:7460")
	cmd.Flags().BoolVar(&f.weak, "accept-weak-isolation", false, "serve Guests even if neither seccomp nor Landlock works (dangerous)")
	cmd.Flags().BoolVar(&f.stdio, "stdio", false, "SSH pipe mode: speak the protocol on stdin/stdout")
	cmd.Flags().StringVar(&f.identity, "identity", "", "SSH pipe mode: the identity id this key belongs to")
	cmd.Flags().BoolVar(&f.noSandbox, "no-self-sandbox", false, "do not apply Landlock/seccomp to reflopd itself (debugging)")
	return cmd
}

// serveArgv is the command line that re-executes this process as
// `reflopd serve` with the same settings. `reflopd up --foreground` (and
// --no-service) also ends in runServe; re-executing os.Args there ran all of
// `up` again (detection, runtime check, self-test, invite) once per re-exec.
func serveArgv(f serveFlags) []string {
	argv := []string{os.Args[0], "serve"}
	for _, a := range f.listen {
		argv = append(argv, "--listen="+a)
	}
	if f.weak {
		argv = append(argv, "--accept-weak-isolation")
	}
	if f.noSandbox {
		argv = append(argv, "--no-self-sandbox")
	}
	return argv
}

func runServe(f serveFlags) error {
	l, err := layout()
	if err != nil {
		return err
	}
	if f.stdio {
		return runStdio(l, f.identity)
	}
	cfg, err := config.Load(l.ConfigFile(), filepath.Join(l.State, "settings.json"))
	if err != nil {
		return err
	}
	listen := cfg.Listen
	if len(f.listen) > 0 {
		listen = f.listen
	}
	// Get a delegated cgroup for the worker safety rails when running from
	// a terminal (the systemd unit already has Delegate=yes).
	if os.Getenv("REFLOP_SCOPED") == "" && os.Getenv("REFLOP_NO_SCOPE") == "" && systemdUser() {
		if path, err := exec.LookPath("systemd-run"); err == nil {
			args := append([]string{path, "--user", "--scope", "--quiet", "--collect", "-p", "Delegate=yes", "--"}, serveArgv(f)...)
			env := append(os.Environ(), "REFLOP_SCOPED=1")
			if err := syscall.Exec(path, args, env); err != nil {
				fmt.Fprintf(os.Stderr, "reflopd: systemd-run: %v (continuing without a delegated cgroup)\n", err)
			}
		}
	}
	if err := l.Ensure(); err != nil {
		return err
	}
	// Unpack the worker package before the daemon confines itself (its
	// Landlock domain keeps the runtime tree read-only).
	if _, err := runtime.EnsureWorker(l.Runtime); err != nil {
		return fmt.Errorf("worker package: %w", err)
	}
	if os.Getenv(sandbox.DaemonSandboxEnv) == "" && !f.noSandbox {
		if err := sandbox.ReexecSandboxed(daemonRules(l), listenPorts(listen), serveArgv(f)); err != nil {
			fmt.Fprintf(os.Stderr, "reflopd: could not sandbox the daemon itself: %v\n", err)
		}
	}
	logger := log.New(os.Stderr, "reflopd: ", log.LstdFlags)
	srv, err := server.New(server.Options{Layout: l, AcceptWeakIsolation: f.weak, Log: logger, Listen: f.listen})
	if err != nil {
		return err
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if err := srv.Prepare(ctx); err != nil {
		return err
	}
	if err := srv.Listen(); err != nil {
		return err
	}
	os.WriteFile(l.PIDFile(), []byte(strconv.Itoa(os.Getpid())+"\n"), 0o600)
	defer os.Remove(l.PIDFile())
	hup := make(chan os.Signal, 1)
	signal.Notify(hup, syscall.SIGHUP)
	go func() {
		for range hup {
			if err := srv.Reload(); err != nil {
				logger.Printf("reload: %v", err)
			}
		}
	}()
	fp := srv.Fingerprint()
	logger.Printf("ready; server key %x", fp[:8])
	return srv.Serve(ctx)
}

// daemonRules is reflopd's own Landlock policy: system files read-only,
// its own directories read-write, the Owner's home otherwise denied.
func daemonRules(l paths.Layout) []sandbox.PathRule {
	var ro uint64 = sandbox.AccessRead
	var rx uint64 = sandbox.AccessRead | sandbox.AccessExec
	var rw uint64 = sandbox.AccessAll
	rules := []sandbox.PathRule{
		{Path: "/usr", Access: rx, Optional: true}, {Path: "/lib", Access: rx, Optional: true},
		{Path: "/lib64", Access: rx, Optional: true}, {Path: "/bin", Access: rx, Optional: true},
		{Path: "/sbin", Access: rx, Optional: true}, {Path: "/opt", Access: rx, Optional: true},
		// /proc is writable in the daemon's outer domain only so workers can
		// write their own /proc/<pid> (their inner domain allows nothing else).
		{Path: "/etc", Access: ro, Optional: true}, {Path: "/proc", Access: ro | sandbox.AccessWrite},
		{Path: "/sys", Access: ro, Optional: true},
		{Path: "/dev", Access: sandbox.AccessDevice | sandbox.AccessRead, Optional: true},
		{Path: l.Runtime, Access: rx}, {Path: l.Config, Access: rw}, {Path: l.State, Access: rw},
		{Path: l.Run, Access: rw}, {Path: l.Scratch, Access: rw},
	}
	if exe, err := os.Executable(); err == nil {
		rules = append(rules, sandbox.PathRule{Path: exe, Access: rx})
	}
	if self, err := cgroup.Self(); err == nil && os.Getenv("REFLOP_SCOPED") != "" {
		rules = append(rules, sandbox.PathRule{Path: self, Access: rw, Optional: true})
	}
	if l.System {
		rules = append(rules, sandbox.PathRule{Path: "/var/lib/reflop", Access: rw, Optional: true})
	}
	return rules
}

func listenPorts(addrs []string) []uint16 {
	var out []uint16
	for _, a := range addrs {
		_, p, err := net.SplitHostPort(a)
		if err != nil {
			continue
		}
		n, err := strconv.Atoi(p)
		if err == nil && n > 0 && n < 65536 {
			out = append(out, uint16(n))
		}
	}
	return out
}

func delegated() bool {
	self, err := cgroup.Self()
	if err != nil {
		return false
	}
	if filepath.Base(self) == "daemon" {
		self = filepath.Dir(self)
	}
	return cgroup.Delegated(self) == nil
}

func systemdUser() bool {
	if _, err := exec.LookPath("systemctl"); err != nil {
		return false
	}
	return exec.Command("systemctl", "--user", "show-environment").Run() == nil
}

// userService reports whether the reflopd user unit manages this daemon. The
// unit always serves the default home, so a REFLOP_HOME daemon never touches it.
func userService() bool {
	return os.Getenv("REFLOP_HOME") == "" && systemdUser()
}

// runStdio is SSH pipe mode (PLAN.md §15.4).
func runStdio(l paths.Layout, identity string) error {
	if len(identity) != 32 {
		return exitWith(ExitUsage, errors.New("--stdio requires --identity <32 hex chars>"))
	}
	// The forced command ignores what the client asked to run, except the
	// one value the reflop client sends.
	if oc, ok := os.LookupEnv("SSH_ORIGINAL_COMMAND"); ok && oc != "" && oc != "reflop-stdio" {
		return exitWith(ExitUsage, errors.New("this key can only be used by the reflop client"))
	}
	return server.ServeStdio(l, identity, os.Stdin, os.Stdout)
}

func upCmd() *cobra.Command {
	var variant, invite string
	var foreground, noService, weak bool
	var listen []string
	cmd := &cobra.Command{
		Use:   "up",
		Short: "Set up and start reflopd (first run: fetch the runtime, self-test, print an invite)",
		Long: `Bring reflopd up without root:

  1. detect the GPU, driver and kernel sandbox features
  2. fetch the pinned compute runtime (CPython + PyTorch, SHA-256 verified)
  3. run the self-test (a sandboxed worker on every device)
  4. start serving as a systemd user service (or in the foreground)
  5. on first run, print a one-time invite code

Everything lives under your user (~/.local/share/reflop, ~/.config/reflop).`,
		Args: cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			out := cmd.OutOrStdout()
			l, err := layout()
			if err != nil {
				return err
			}
			if err := l.Ensure(); err != nil {
				return err
			}
			inf := gpu.Detect()
			describeHost(out, inf)
			lock, err := runtime.Load()
			if err != nil {
				return err
			}
			choice, err := lock.Choose(inf, variant)
			if err != nil {
				return err
			}
			for _, w := range choice.Warn {
				fmt.Fprintf(out, "  note: %s\n", w)
			}
			fmt.Fprintf(out, "runtime: %s (%s)\n", choice.Variant, choice.Reason)
			ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
			defer stop()
			inst, err := lock.Install(ctx, l.Runtime, choice.Variant, func(f string, a ...any) {
				fmt.Fprintf(out, "  "+f+"\n", a...)
			})
			if err != nil {
				return fmt.Errorf("runtime install: %w", err)
			}
			fmt.Fprintf(out, "runtime ready: %s\n", inst.Marker.ID)
			if !weak && sandbox.LandlockABI() == 0 && !sandbox.SeccompAvailable() {
				return exitWith(ExitUnsafe, errors.New("neither Landlock nor seccomp is available on this kernel; reflopd will not serve Guests (see `reflopd doctor`, or --accept-weak-isolation)"))
			}
			fmt.Fprintln(out, "self-test:")
			skipped, err := runSelftest(ctx, out, l, false, weak)
			if err != nil {
				return fmt.Errorf("self-test failed: %w", err)
			}
			if len(skipped) > 0 {
				// A full GPU is the Owner's own use, not a fault: serve the
				// devices that passed, and the GPU once memory frees up.
				fmt.Fprintln(out, "  note: starting anyway; run `reflopd selftest` once GPU memory is free")
			}
			first := false
			if ids, err := auth.Open(l.Identities()); err == nil && len(ids.List()) == 0 {
				first = true
			}
			useService := !foreground && !noService && userService()
			if useService {
				if err := installUserService(l, listen, weak); err != nil {
					return err
				}
				fmt.Fprintf(out, "started the %s user service (logs: journalctl --user -u reflopd -f)\n", unitName)
				if err := waitRunning(l, 60*time.Second); err != nil {
					return err
				}
				lingerHint(out)
			}
			if first || invite != "" {
				name := invite
				if name == "" {
					name = "guest"
				}
				code, err := makeInvite(l, name, 24*time.Hour, listen)
				if err != nil {
					return err
				}
				fmt.Fprintf(out, "\nInvite for %q (one-time, expires in 24h):\n\n  %s\n\nOn the Guest machine: reflop pair <code>\n", name, code)
				if len(listen) == 0 {
					fmt.Fprintln(out, "\nreflopd listens on 127.0.0.1 only. To serve other machines, restart with\n  reflopd up --listen <tailscale-or-lan-ip>:7460")
				}
			}
			if !useService {
				fmt.Fprintln(out, "serving in the foreground (Ctrl-C to stop)")
				return runServe(serveFlags{listen: listen, weak: weak})
			}
			return nil
		},
	}
	cmd.Flags().StringVar(&variant, "variant", "", "force a runtime variant (cu132, cu126, cpu)")
	cmd.Flags().StringVar(&invite, "invite", "", "also create an invite for this name")
	cmd.Flags().BoolVar(&foreground, "foreground", false, "serve in this terminal instead of a user service")
	cmd.Flags().BoolVar(&noService, "no-service", false, "never install a systemd user service")
	cmd.Flags().BoolVar(&weak, "accept-weak-isolation", false, "allow serving without seccomp and Landlock (dangerous)")
	cmd.Flags().StringSliceVar(&listen, "listen", nil, "address(es) to listen on; default 127.0.0.1:7460")
	return cmd
}

func describeHost(out interface{ Write([]byte) (int, error) }, inf gpu.Info) {
	if len(inf.GPUs) == 0 {
		fmt.Fprintln(out, "GPU: none detected (the server CPU will be served)")
	}
	for _, g := range inf.GPUs {
		fmt.Fprintf(out, "GPU %d: %s, %d MiB, compute capability %d.%d\n", g.Index, g.Name, g.MemoryMiB, g.CCMajor, g.CCMinor)
	}
	if inf.Driver != "" {
		kind := "proprietary"
		if inf.Open {
			kind = "open"
		}
		if inf.WSL {
			kind = "WSL2"
		}
		fmt.Fprintf(out, "driver: %s (%s)\n", inf.Driver, kind)
	}
	fmt.Fprintf(out, "sandbox: landlock abi %d, seccomp %v\n", sandbox.LandlockABI(), sandbox.SeccompAvailable())
}

func installUserService(l paths.Layout, listen []string, weak bool) error {
	exe, err := os.Executable()
	if err != nil {
		return err
	}
	exe, _ = filepath.EvalSymlinks(exe)
	args := []string{exe, "serve"}
	for _, a := range listen {
		args = append(args, "--listen", a)
	}
	if weak {
		args = append(args, "--accept-weak-isolation")
	}
	home, _ := os.UserHomeDir()
	dir := filepath.Join(home, ".config", "systemd", "user")
	if v := os.Getenv("XDG_CONFIG_HOME"); v != "" {
		dir = filepath.Join(v, "systemd", "user")
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	unit := fmt.Sprintf(`# Generated by reflopd up. Edit with care: reflopd up rewrites this file.
[Unit]
Description=reFLOP daemon (reflopd)
After=network-online.target

[Service]
Type=simple
ExecStart=%s
Restart=on-failure
RestartSec=3
# Workers get their own cgroups below this unit (memory and pids rails).
Delegate=yes
NoNewPrivileges=yes
Environment=REFLOP_SCOPED=1

[Install]
WantedBy=default.target
`, quoteArgs(args))
	if err := os.WriteFile(filepath.Join(dir, unitName), []byte(unit), 0o644); err != nil {
		return err
	}
	for _, a := range [][]string{{"daemon-reload"}, {"enable", unitName}, {"restart", unitName}} {
		if out, err := exec.Command("systemctl", append([]string{"--user"}, a...)...).CombinedOutput(); err != nil {
			return fmt.Errorf("systemctl --user %s: %v: %s", strings.Join(a, " "), err, strings.TrimSpace(string(out)))
		}
	}
	return nil
}

func quoteArgs(args []string) string {
	out := make([]string, len(args))
	for i, a := range args {
		if strings.ContainsAny(a, " \t\"'\\") {
			a = strconv.Quote(a)
		}
		out[i] = a
	}
	return strings.Join(out, " ")
}

func waitRunning(l paths.Layout, d time.Duration) error {
	deadline := time.Now().Add(d)
	for time.Now().Before(deadline) {
		if _, err := server.ControlClient(l, server.ControlRequest{Cmd: "status"}, 2*time.Second); err == nil {
			return nil
		}
		time.Sleep(500 * time.Millisecond)
	}
	return errors.New("reflopd did not start; see: journalctl --user -u reflopd -n 50")
}

func lingerHint(out interface{ Write([]byte) (int, error) }) {
	u := os.Getenv("USER")
	b, err := exec.Command("loginctl", "show-user", u, "--property=Linger").Output()
	if err == nil && strings.TrimSpace(string(b)) == "Linger=no" {
		fmt.Fprintf(out, "note: reflopd stops when you log out. To keep it running: loginctl enable-linger %s\n", u)
	}
}

func stopCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "stop",
		Short: "Stop the daemon (all sessions end)",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			if userService() && exec.Command("systemctl", "--user", "is-active", "--quiet", unitName).Run() == nil {
				if out, err := exec.Command("systemctl", "--user", "stop", unitName).CombinedOutput(); err != nil {
					return fmt.Errorf("systemctl: %v: %s", err, out)
				}
				fmt.Fprintln(cmd.OutOrStdout(), "stopped")
				return nil
			}
			if err := control("stop", nil, nil); err != nil {
				return err
			}
			fmt.Fprintln(cmd.OutOrStdout(), "stopping")
			return nil
		},
	}
}
