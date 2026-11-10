// SPDX-License-Identifier: Apache-2.0

// Package supervisor spawns, watches and kills sandboxed workers: one
// process per (identity, session), invariant I5.
package supervisor

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"golang.org/x/sys/unix"

	"github.com/chaseungjoon/reflop/server/reflopd/internal/cgroup"
	pb "github.com/chaseungjoon/reflop/server/reflopd/internal/pb/reflopv1"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/sandbox"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/wire"
)

// Spec describes one worker.
type Spec struct {
	Name        string // cgroup / log name (session hex)
	Python      string // runtime interpreter
	RuntimeDir  string
	Scratch     string
	GPUUUIDs    []string // exposed GPUs; empty = CPU only
	DeviceNodes []string
	WSL         bool
	Init        *pb.WorkerInit

	UserNS          bool
	CG              *cgroup.Manager
	MaxPIDs         uint32
	RequireLandlock bool
	FSize           uint64
	ExtraRules      []sandbox.PathRule
	MaxFrame        uint32
	ReadyTimeout    time.Duration
	Log             func(format string, args ...any)
	// TestProbe, when non-empty, runs reflop_worker's isolation probe mode
	// instead of the server loop (test builds only; see tests/security/isolation).
	TestProbe string
}

// Worker is a running worker process.
type Worker struct {
	Spec   Spec
	cmd    *exec.Cmd
	ctrl   *net.UnixConn
	fdpass *net.UnixConn
	cg     *cgroup.Worker
	r      *wire.Reader
	w      *wire.Writer

	Ready  *pb.WorkerReady
	Report sandbox.HelperReport

	done    chan struct{}
	mu      sync.Mutex
	exitErr error
	killed  string
	stderr  []string
}

// WorkerRules is the Landlock policy for workers (PLAN.md §4.3, §9.4).
// Everything not listed, the Owner's home included, is denied.
func WorkerRules(runtimeDir, scratch string, devices []string, wsl bool) []sandbox.PathRule {
	var ro uint64 = sandbox.AccessRead
	var rx uint64 = sandbox.AccessRead | sandbox.AccessExec
	rules := []sandbox.PathRule{
		{Path: runtimeDir, Access: rx},
		{Path: scratch, Access: sandbox.AccessScratch},
		// Shared libraries: glibc and the NVIDIA driver's user-space libs.
		{Path: "/usr/lib", Access: rx, Optional: true},
		{Path: "/usr/lib64", Access: rx, Optional: true},
		{Path: "/lib", Access: rx, Optional: true},
		{Path: "/lib64", Access: rx, Optional: true},
		{Path: "/usr/local/lib", Access: rx, Optional: true},
		{Path: "/etc/ld.so.cache", Access: ro, Optional: true},
		{Path: "/etc/localtime", Access: ro, Optional: true},
		// Hardware topology read by glibc, OpenMP and the CUDA driver.
		{Path: "/proc/cpuinfo", Access: ro},
		{Path: "/proc/meminfo", Access: ro},
		{Path: "/proc/stat", Access: ro, Optional: true},
		{Path: "/proc/sys/kernel/osrelease", Access: ro, Optional: true},
		{Path: "/proc/sys/vm/overcommit_memory", Access: ro, Optional: true},
		{Path: "/proc/sys/kernel/numa_balancing", Access: ro, Optional: true},
		{Path: "/proc/driver/nvidia", Access: ro, Optional: true},
		{Path: "/sys/devices/system/cpu", Access: ro, Optional: true},
		{Path: "/sys/devices/system/node", Access: ro, Optional: true},
		{Path: "/sys/fs/cgroup", Access: ro, Optional: true},
		{Path: "/sys/bus/pci", Access: ro, Optional: true},
		{Path: "/sys/devices", Access: ro, Optional: true},
		{Path: "/sys/module/nvidia", Access: ro, Optional: true},
		{Path: "/dev/null", Access: sandbox.AccessRead | sandbox.AccessWrite},
		{Path: "/dev/zero", Access: ro},
		{Path: "/dev/urandom", Access: ro},
	}
	if wsl {
		rules = append(rules, sandbox.PathRule{Path: "/usr/lib/wsl", Access: rx, Optional: true})
	}
	for _, d := range devices {
		rules = append(rules, sandbox.PathRule{Path: d, Access: sandbox.AccessDevice, Optional: true})
	}
	return rules
}

// Env is the worker's complete environment: nothing is inherited from the
// Owner (PLAN.md §4.3 step 1).
func Env(spec Spec) []string {
	env := []string{
		"LANG=C.UTF-8",
		"HOME=" + spec.Scratch,
		"TMPDIR=" + filepath.Join(spec.Scratch, "tmp"),
		"XDG_CACHE_HOME=" + filepath.Join(spec.Scratch, "cache"),
		"CUDA_CACHE_PATH=" + filepath.Join(spec.Scratch, "nvcache"),
		"CUDA_MODULE_LOADING=LAZY",
		"CUDA_DEVICE_ORDER=PCI_BUS_ID",
		"TORCHINDUCTOR_DISABLE=1",
		"PYTORCH_NVML_BASED_CUDA_CHECK=0",
	}
	env = append(env, "CUDA_VISIBLE_DEVICES="+strings.Join(spec.GPUUUIDs, ","))
	if spec.WSL {
		env = append(env, "LD_LIBRARY_PATH=/usr/lib/wsl/lib")
	}
	return env
}

// Spawn starts a worker and waits for WorkerReady.
func Spawn(ctx context.Context, spec Spec) (*Worker, error) {
	if spec.ReadyTimeout == 0 {
		spec.ReadyTimeout = 3 * time.Minute
	}
	if spec.Log == nil {
		spec.Log = func(string, ...any) {}
	}
	for _, d := range []string{spec.Scratch, filepath.Join(spec.Scratch, "tmp"), filepath.Join(spec.Scratch, "cache")} {
		if err := os.MkdirAll(d, 0o700); err != nil {
			return nil, err
		}
	}
	ctrlPair, err := unix.Socketpair(unix.AF_UNIX, unix.SOCK_STREAM|unix.SOCK_CLOEXEC, 0)
	if err != nil {
		return nil, err
	}
	passPair, err := unix.Socketpair(unix.AF_UNIX, unix.SOCK_SEQPACKET|unix.SOCK_CLOEXEC, 0)
	if err != nil {
		unix.Close(ctrlPair[0])
		unix.Close(ctrlPair[1])
		return nil, err
	}
	cfgR, cfgW, err := os.Pipe()
	if err != nil {
		return nil, err
	}
	childCtrl := os.NewFile(uintptr(ctrlPair[1]), "ctrl")
	childPass := os.NewFile(uintptr(passPair[1]), "pass")
	defer childCtrl.Close()
	defer childPass.Close()
	defer cfgR.Close()

	self, err := os.Executable()
	if err != nil {
		return nil, err
	}
	argv := []string{spec.Python, "-I", "-B", "-m", "reflop_worker"}
	if spec.TestProbe != "" {
		argv = append(argv, "--isolation-probe", spec.TestProbe)
	}
	cfg := sandbox.ExecConfig{
		Argv:            argv,
		Env:             Env(spec),
		Dir:             spec.Scratch,
		Rules:           append(WorkerRules(spec.RuntimeDir, spec.Scratch, spec.DeviceNodes, spec.WSL), spec.ExtraRules...),
		AllowProcSelf:   true,
		RequireLandlock: spec.RequireLandlock,
		NoFile:          8192,
		FSize:           spec.FSize,
	}
	if spec.CG == nil {
		// No cgroup: fall back to RLIMIT_NPROC (counts all of the Owner's
		// processes, so leave headroom above the current count).
		cfg.NProc = uint64(countUserTasks()) + uint64(spec.MaxPIDs)
	}
	stderrR, stderrW, err := os.Pipe()
	if err != nil {
		return nil, err
	}
	devnull, _ := os.Open(os.DevNull)
	defer devnull.Close()
	cmd := &exec.Cmd{
		Path:       self,
		Args:       []string{"reflop-worker", sandbox.HelperArg},
		Env:        []string{},
		Stdin:      devnull,
		Stdout:     devnull,
		Stderr:     stderrW,
		ExtraFiles: []*os.File{childCtrl, childPass, cfgR}, // fds 3, 4, 5
	}
	attr := &syscall.SysProcAttr{Setpgid: true, Pdeathsig: syscall.SIGKILL}
	var cgw *cgroup.Worker
	if spec.CG != nil {
		cgw, err = spec.CG.NewWorker(spec.Name, spec.MaxPIDs)
		if err != nil {
			return nil, fmt.Errorf("worker cgroup: %w", err)
		}
		attr.UseCgroupFD = true
		attr.CgroupFD = cgw.FD
	}
	if spec.UserNS {
		uid, gid := os.Getuid(), os.Getgid()
		attr.Cloneflags = syscall.CLONE_NEWUSER | syscall.CLONE_NEWNET
		attr.UidMappings = []syscall.SysProcIDMap{{ContainerID: uid, HostID: uid, Size: 1}}
		attr.GidMappings = []syscall.SysProcIDMap{{ContainerID: gid, HostID: gid, Size: 1}}
		attr.GidMappingsEnableSetgroups = false
	}
	cmd.SysProcAttr = attr
	if err := cmd.Start(); err != nil {
		stderrR.Close()
		stderrW.Close()
		if cgw != nil {
			cgw.Remove()
		}
		return nil, fmt.Errorf("start worker: %w", err)
	}
	stderrW.Close()
	b, _ := json.Marshal(cfg)
	cfgW.Write(b)
	cfgW.Close()

	ctrlF := os.NewFile(uintptr(ctrlPair[0]), "ctrl-parent")
	passF := os.NewFile(uintptr(passPair[0]), "pass-parent")
	ctrlC, err := net.FileConn(ctrlF)
	ctrlF.Close()
	if err != nil {
		cmd.Process.Kill()
		return nil, err
	}
	passC, err := net.FileConn(passF)
	passF.Close()
	if err != nil {
		cmd.Process.Kill()
		return nil, err
	}
	w := &Worker{
		Spec: spec, cmd: cmd, ctrl: ctrlC.(*net.UnixConn), fdpass: passC.(*net.UnixConn), cg: cgw,
		done: make(chan struct{}),
	}
	w.r = wire.NewReader(w.ctrl, spec.MaxFrame)
	w.w = wire.NewWriter(w.ctrl, spec.MaxFrame)
	reportCh := make(chan sandbox.HelperReport, 1)
	go w.readStderr(stderrR, reportCh)
	go func() {
		err := cmd.Wait()
		w.mu.Lock()
		w.exitErr = err
		w.mu.Unlock()
		close(w.done)
		w.ctrl.Close()
		w.fdpass.Close()
		if cgw != nil {
			cgw.Remove()
		}
	}()

	if spec.TestProbe != "" {
		return w, nil
	}
	if err := w.Send(&pb.WorkerEnvelope{Body: &pb.WorkerEnvelope_Init{Init: spec.Init}}); err != nil {
		w.Kill("init failed")
		return nil, w.failure(fmt.Errorf("send init: %w", err))
	}
	type res struct {
		env *pb.WorkerEnvelope
		err error
	}
	ch := make(chan res, 1)
	go func() { e, err := w.Recv(); ch <- res{e, err} }()
	select {
	case r := <-ch:
		if r.err != nil {
			w.Kill("startup failed")
			return nil, w.failure(r.err)
		}
		switch b := r.env.Body.(type) {
		case *pb.WorkerEnvelope_Ready:
			w.Ready = b.Ready
		case *pb.WorkerEnvelope_Fatal:
			w.Kill("startup failed")
			return nil, w.failure(errors.New(b.Fatal.Message))
		default:
			w.Kill("protocol error")
			return nil, errors.New("worker sent an unexpected message before ready")
		}
	case <-time.After(spec.ReadyTimeout):
		w.Kill("startup timeout")
		return nil, w.failure(errors.New("worker did not become ready in time"))
	case <-ctx.Done():
		w.Kill("cancelled")
		return nil, ctx.Err()
	}
	select {
	case w.Report = <-reportCh:
	default:
	}
	return w, nil
}

func (w *Worker) failure(err error) error {
	<-w.done
	w.mu.Lock()
	defer w.mu.Unlock()
	tail := w.stderr
	if len(tail) > 20 {
		tail = tail[len(tail)-20:]
	}
	if len(tail) == 0 {
		return err
	}
	return fmt.Errorf("%w\nworker output:\n  %s", err, strings.Join(tail, "\n  "))
}

func (w *Worker) readStderr(r *os.File, report chan<- sandbox.HelperReport) {
	defer r.Close()
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 64<<10), 1<<20)
	for sc.Scan() {
		line := sc.Text()
		if rep, ok := sandbox.ParseReport(line); ok {
			select {
			case report <- rep:
			default:
			}
			w.mu.Lock()
			w.Report = rep
			w.mu.Unlock()
			if rep.Error != "" {
				w.Spec.Log("worker %s sandbox: %s", w.Spec.Name, rep.Error)
			}
			continue
		}
		w.mu.Lock()
		w.stderr = append(w.stderr, line)
		if len(w.stderr) > 200 {
			w.stderr = w.stderr[100:]
		}
		w.mu.Unlock()
		w.Spec.Log("worker %s: %s", w.Spec.Name, line)
	}
}

// Send writes one envelope.
func (w *Worker) Send(e *pb.WorkerEnvelope) error { return w.w.Write(e) }

// Recv reads one envelope.
func (w *Worker) Recv() (*pb.WorkerEnvelope, error) {
	e := &pb.WorkerEnvelope{}
	if err := w.r.Read(e); err != nil {
		return nil, err
	}
	return e, nil
}

// SendFDs passes file descriptors on the fd channel with a tag.
func (w *Worker) SendFDs(tag []byte, fds []int) error {
	rights := unix.UnixRights(fds...)
	_, _, err := w.fdpass.WriteMsgUnix(tag, rights, nil)
	return err
}

// Kill terminates the worker (and its cgroup) with a reason.
func (w *Worker) Kill(reason string) {
	w.mu.Lock()
	if w.killed == "" {
		w.killed = reason
	}
	w.mu.Unlock()
	if w.cg != nil {
		w.cg.Kill()
	}
	if w.cmd.Process != nil {
		_ = syscall.Kill(-w.cmd.Process.Pid, syscall.SIGKILL)
		_ = w.cmd.Process.Kill()
	}
}

// Done is closed when the process has exited.
func (w *Worker) Done() <-chan struct{} { return w.done }

// ExitReason describes why the worker ended.
func (w *Worker) ExitReason() string {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.killed != "" {
		return w.killed
	}
	if w.cg != nil && w.cg.OOMKills() > 0 {
		return "host memory safety rail (OOM in worker cgroup)"
	}
	if w.exitErr != nil {
		return w.exitErr.Error()
	}
	return "exited"
}

// PID returns the worker pid.
func (w *Worker) PID() int { return w.cmd.Process.Pid }

// Memory returns host memory used by the worker.
func (w *Worker) Memory() uint64 {
	if w.cg != nil {
		return w.cg.MemoryCurrent()
	}
	return rss(w.cmd.Process.Pid)
}

// OutputTail returns recent worker stderr lines.
func (w *Worker) OutputTail() []string {
	w.mu.Lock()
	defer w.mu.Unlock()
	return append([]string(nil), w.stderr...)
}

func rss(pid int) uint64 {
	b, err := os.ReadFile("/proc/" + strconv.Itoa(pid) + "/smaps_rollup")
	if err != nil {
		b, err = os.ReadFile("/proc/" + strconv.Itoa(pid) + "/status")
		if err != nil {
			return 0
		}
	}
	for _, l := range strings.Split(string(b), "\n") {
		if v, ok := strings.CutPrefix(l, "Rss:"); ok {
			return parseKB(v)
		}
		if v, ok := strings.CutPrefix(l, "VmRSS:"); ok {
			return parseKB(v)
		}
	}
	return 0
}

func parseKB(v string) uint64 {
	f := strings.Fields(v)
	if len(f) == 0 {
		return 0
	}
	n, _ := strconv.ParseUint(f[0], 10, 64)
	return n * 1024
}

func countUserTasks() int {
	uid := strconv.Itoa(os.Getuid())
	ents, _ := os.ReadDir("/proc")
	n := 0
	for _, e := range ents {
		if _, err := strconv.Atoi(e.Name()); err != nil {
			continue
		}
		b, err := os.ReadFile("/proc/" + e.Name() + "/status")
		if err != nil {
			continue
		}
		var owner, threads string
		for _, l := range strings.Split(string(b), "\n") {
			if v, ok := strings.CutPrefix(l, "Uid:"); ok {
				if f := strings.Fields(v); len(f) > 0 {
					owner = f[0]
				}
			}
			if v, ok := strings.CutPrefix(l, "Threads:"); ok {
				threads = strings.TrimSpace(v)
			}
		}
		if owner == uid {
			t, _ := strconv.Atoi(threads)
			n += t
		}
	}
	return n
}

// UserNSAvailable probes whether unprivileged user+network namespaces can
// be created (false on Ubuntu 24.04 defaults, where AppArmor restricts them).
func UserNSAvailable() bool {
	cmd := exec.Command("/proc/self/exe", "__userns-probe")
	cmd.Env = []string{}
	uid, gid := os.Getuid(), os.Getgid()
	cmd.SysProcAttr = &syscall.SysProcAttr{
		Cloneflags:                 syscall.CLONE_NEWUSER | syscall.CLONE_NEWNET,
		UidMappings:                []syscall.SysProcIDMap{{ContainerID: uid, HostID: uid, Size: 1}},
		GidMappings:                []syscall.SysProcIDMap{{ContainerID: gid, HostID: gid, Size: 1}},
		GidMappingsEnableSetgroups: false,
	}
	return cmd.Run() == nil
}
