// SPDX-License-Identifier: Apache-2.0

//go:build reflop_probes

package server

// Isolation tests (PLAN.md §17.5). They need a test build and an installed
// runtime:
//
//	REFLOP_TEST_HOME=<REFLOP_HOME with a runtime> go test -tags reflop_probes ./server/reflopd/internal/server -run Isolation -v
//
// Each probe attempts something the sandbox must prevent inside a fully
// sandboxed worker (Landlock before exec, seccomp after warm-up) and exits
// 0 only if the attempt failed.
//
// REFLOP_TEST_STRICT=1 (set by CI, .github/workflows/isolation.yml) turns
// every "cannot test this here" into a failure instead of a skip: no runtime,
// no Landlock, no seccomp, a missing read target (a probe that gets ENOENT
// proves nothing), or a user-namespace mode other than REFLOP_TEST_USERNS
// ("1" available, "0" blocked). It also makes each probe worker require
// Landlock and seccomp, so a sandbox layer that fails to apply stops the
// worker instead of being skipped.

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	pb "github.com/reflopai/reflop/pkg/reflopv1"
	"github.com/reflopai/reflop/server/reflopd/internal/config"
	"github.com/reflopai/reflop/server/reflopd/internal/gpu"
	"github.com/reflopai/reflop/server/reflopd/internal/paths"
	"github.com/reflopai/reflop/server/reflopd/internal/runtime"
	"github.com/reflopai/reflop/server/reflopd/internal/sandbox"
	"github.com/reflopai/reflop/server/reflopd/internal/supervisor"
)

// The supervisor re-executes the current binary as the sandbox helper; in
// tests that binary is the test binary.
func TestMain(m *testing.M) {
	if len(os.Args) > 1 && os.Args[1] == sandbox.HelperArg {
		sandbox.RunHelper()
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "__userns-probe" {
		os.Exit(0)
	}
	os.Exit(m.Run())
}

func strictMode() bool { return os.Getenv("REFLOP_TEST_STRICT") == "1" }

// unavailable skips the test, or fails it in strict mode.
func unavailable(t *testing.T, msg string) {
	t.Helper()
	if strictMode() {
		t.Fatalf("REFLOP_TEST_STRICT=1: %s", msg)
	}
	t.Skip(msg)
}

// sandboxProblems lists what keeps this host from running the probes under
// the full sandbox. landlockABI, seccomp and userNS describe the host;
// wantUserNS is REFLOP_TEST_USERNS ("" means either mode is fine).
func sandboxProblems(landlockABI int, seccomp, userNS bool, wantUserNS string) []string {
	var p []string
	if landlockABI <= 0 {
		p = append(p, "Landlock is not available on this kernel")
	}
	if !seccomp {
		p = append(p, "seccomp is not available on this kernel")
	}
	switch wantUserNS {
	case "", "any":
	case "1":
		if !userNS {
			p = append(p, "REFLOP_TEST_USERNS=1 but unprivileged user namespaces are blocked")
		}
	case "0":
		if userNS {
			p = append(p, "REFLOP_TEST_USERNS=0 but unprivileged user namespaces are available")
		}
	default:
		p = append(p, "REFLOP_TEST_USERNS must be 0, 1 or empty, not "+wantUserNS)
	}
	return p
}

func probeEnv(t *testing.T) (runtime.Installed, string, paths.Layout) {
	home := os.Getenv("REFLOP_TEST_HOME")
	if home == "" {
		unavailable(t, "set REFLOP_TEST_HOME to a REFLOP_HOME with an installed runtime")
	}
	os.Setenv("REFLOP_HOME", home)
	l, err := paths.Resolve()
	if err != nil {
		t.Fatal(err)
	}
	rts := runtime.List(l.Runtime)
	if len(rts) == 0 {
		unavailable(t, "no runtime installed in "+l.Runtime)
	}
	if strictMode() {
		if p := sandboxProblems(sandbox.LandlockABI(), sandbox.SeccompAvailable(), supervisor.UserNSAvailable(),
			os.Getenv("REFLOP_TEST_USERNS")); len(p) > 0 {
			t.Fatalf("REFLOP_TEST_STRICT=1: %s", strings.Join(p, "; "))
		}
	}
	t.Logf("sandbox: landlock abi %d, seccomp %v, user namespaces %v, strict %v",
		sandbox.LandlockABI(), sandbox.SeccompAvailable(), supervisor.UserNSAvailable(), strictMode())
	wd, err := runtime.EnsureWorker(l.Runtime)
	if err != nil {
		t.Fatal(err)
	}
	return rts[0], wd, l
}

func runProbe(t *testing.T, rt runtime.Installed, wd string, l paths.Layout, probe string, cuda bool) int {
	t.Helper()
	scratch := filepath.Join(l.Scratch, "probe-"+filepath.Base(t.Name()))
	defer os.RemoveAll(scratch)
	init := &pb.WorkerInit{Device: &pb.Device{Kind: pb.DeviceKind_DEVICE_CPU}, Limits: &pb.Limits{
		MaxFrameBytes: 64 << 20, MaxNodes: 1000, MaxGraphInputs: 100, MaxDims: 16, MaxTensorNumel: 1 << 30,
		MaxInlineTensorBytes: 1 << 20, MaxFetchBytes: 1 << 30, MaxListLength: 1000, UploadChunkBytes: 4 << 20}}
	strict := strictMode()
	if strict {
		// Fail closed, as reflopd does without --accept-weak-isolation.
		init.RequireSeccomp = true
	}
	spec := supervisor.Spec{Name: "probe", Python: rt.PythonBin(), RuntimeDir: rt.Dir, WorkerDir: wd, Scratch: scratch,
		Init: init, MaxPIDs: 4096, RequireLandlock: strict || sandbox.LandlockABI() > 0, MaxFrame: 64 << 20, TestProbe: probe,
		UserNS: supervisor.UserNSAvailable()}
	if cuda {
		if rt.Marker.CUDA == "" {
			t.Skipf("runtime %s has no CUDA", rt.Marker.ID)
		}
		inf := gpu.Detect()
		if len(inf.GPUs) == 0 {
			t.Skip("no GPU")
		}
		init.Device = &pb.Device{Kind: pb.DeviceKind_DEVICE_CUDA}
		spec.GPUUUIDs = []string{inf.GPUs[0].UUID}
		spec.DeviceNodes = gpu.DeviceNodes(inf.GPUs[:1], inf.WSL)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	w, err := supervisor.SpawnWithFilter(ctx, spec, func(pid int) ([]byte, error) {
		f, err := sandbox.Policy{SelfPID: pid}.Filter()
		return sandbox.Encode(f), err
	})
	if err != nil {
		t.Fatalf("spawn: %v", err)
	}
	select {
	case <-w.Done():
	case <-ctx.Done():
		w.Kill("timeout")
		t.Fatal("probe timed out")
	}
	c := w.ExitCode()
	out := w.OutputTail()
	// A probe that never ran is a test failure, not a pass.
	ran := false
	for _, l := range out {
		if strings.Contains(l, "probe "+probe+":") {
			ran = true
		}
	}
	if !ran {
		t.Logf("worker output: %v", out)
		return 99
	}
	if strict && !w.Report.Landlock {
		t.Logf("sandbox report: %+v", w.Report)
		t.Error("REFLOP_TEST_STRICT=1: the Landlock domain was not applied")
	}
	if c != 0 {
		t.Logf("worker output: %v", out)
	}
	return c
}

func TestIsolationProbes(t *testing.T) {
	rt, wd, l := probeEnv(t)
	home, _ := os.UserHomeDir()
	other := filepath.Join(l.Scratch, "other-session")
	os.MkdirAll(other, 0o700)
	os.WriteFile(filepath.Join(other, "secret"), []byte("x"), 0o600)
	defer os.RemoveAll(other)
	probes := []string{
		"tcp_socket", "udp_socket", "unix_socket", "connect_loopback", "read_etc_shadow", "read_etc_passwd",
		"list_root", "write_tmp", "write_var_tmp", "exec", "fork", "ptrace_parent", "signal_parent",
		"signal_init", "mount", "unshare", "setuid0", "open_dev_mem", "open_dev_sda", "open_dev_input",
		"read:" + filepath.Join(home, ".ssh", "id_ed25519"), "list:" + filepath.Join(home, ".ssh"),
		"list:" + home, "write:" + filepath.Join(home, "reflop-probe"), "read:" + filepath.Join(other, "secret"),
		"read:" + l.ServerKey(), "read:" + filepath.Join(l.Identities(), "identities.json"),
		"write:" + filepath.Join(rt.Dir, "python", "evil"), "list:" + l.Store(),
	}
	if os.Getenv("XDG_RUNTIME_DIR") != "" {
		probes = append(probes, "list:"+os.Getenv("XDG_RUNTIME_DIR"))
	}
	for _, p := range probes {
		p := p
		t.Run(p, func(t *testing.T) {
			// A read or list target that does not exist fails with ENOENT,
			// which the probe counts as blocked: it tested nothing.
			if target, ok := probeTarget(p); ok && strictMode() {
				if _, err := os.Stat(target); err != nil {
					t.Fatalf("REFLOP_TEST_STRICT=1: probe target must exist so the sandbox, not ENOENT, blocks it: %v", err)
				}
			}
			if code := runProbe(t, rt, wd, l, p, false); code != 0 {
				t.Errorf("probe %s: exit %d (1 = sandbox hole)", p, code)
			}
		})
	}
	t.Run("cuda/tcp_socket", func(t *testing.T) {
		if code := runProbe(t, rt, wd, l, "tcp_socket", true); code != 0 {
			t.Errorf("exit %d", code)
		}
	})
	t.Run("cuda/read_ssh", func(t *testing.T) {
		if code := runProbe(t, rt, wd, l, "list:"+filepath.Join(home, ".ssh"), true); code != 0 {
			t.Errorf("exit %d", code)
		}
	})
	_ = config.Defaults
}

// probeTarget returns the path a read or list probe touches.
func probeTarget(probe string) (string, bool) {
	switch {
	case probe == "read_etc_shadow":
		return "/etc/shadow", true
	case probe == "read_etc_passwd":
		return "/etc/passwd", true
	case strings.HasPrefix(probe, "read:"), strings.HasPrefix(probe, "list:"):
		return probe[5:], true
	}
	return "", false
}

// The strict-mode checks must catch every way the CI job could turn into a
// silent no-op (issue #3).
func TestIsolationStrictChecks(t *testing.T) {
	if p := sandboxProblems(6, true, false, "0"); len(p) != 0 {
		t.Errorf("full sandbox, namespaces blocked as expected: %v", p)
	}
	if p := sandboxProblems(6, true, true, ""); len(p) != 0 {
		t.Errorf("full sandbox, any namespace mode: %v", p)
	}
	for _, c := range []struct {
		abi             int
		seccomp, userNS bool
		want            string
	}{
		{0, true, true, "1"},   // no Landlock
		{6, false, true, "1"},  // no seccomp
		{6, true, false, "1"},  // namespaces expected but blocked
		{6, true, true, "0"},   // namespaces expected blocked but available
		{6, true, true, "yes"}, // typo in REFLOP_TEST_USERNS
	} {
		if p := sandboxProblems(c.abi, c.seccomp, c.userNS, c.want); len(p) == 0 {
			t.Errorf("%+v: no problem reported", c)
		}
	}
	for probe, want := range map[string]string{
		"read_etc_shadow": "/etc/shadow", "read:/a/b": "/a/b", "list:/c": "/c",
	} {
		if got, ok := probeTarget(probe); !ok || got != want {
			t.Errorf("probeTarget(%q) = %q, %v", probe, got, ok)
		}
	}
	for _, probe := range []string{"write:/x", "tcp_socket", "open_dev_sda"} {
		if _, ok := probeTarget(probe); ok {
			t.Errorf("probeTarget(%q) should not need an existing target", probe)
		}
	}
}
