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

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	pb "github.com/chaseungjoon/reflop/pkg/reflopv1"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/config"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/gpu"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/paths"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/runtime"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/sandbox"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/supervisor"
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

func probeEnv(t *testing.T) (runtime.Installed, string, paths.Layout) {
	home := os.Getenv("REFLOP_TEST_HOME")
	if home == "" {
		t.Skip("set REFLOP_TEST_HOME to a REFLOP_HOME with an installed runtime")
	}
	os.Setenv("REFLOP_HOME", home)
	l, err := paths.Resolve()
	if err != nil {
		t.Fatal(err)
	}
	rts := runtime.List(l.Runtime)
	if len(rts) == 0 {
		t.Skip("no runtime installed")
	}
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
	spec := supervisor.Spec{Name: "probe", Python: rt.PythonBin(), RuntimeDir: rt.Dir, WorkerDir: wd, Scratch: scratch,
		Init: init, MaxPIDs: 4096, RequireLandlock: sandbox.LandlockABI() > 0, MaxFrame: 64 << 20, TestProbe: probe,
		UserNS: supervisor.UserNSAvailable()}
	if cuda {
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
	w, err := supervisor.SpawnWithFilter(ctx, spec, func(pid int) []byte {
		f, _ := sandbox.Policy{SelfPID: pid}.Filter()
		return sandbox.Encode(f)
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
