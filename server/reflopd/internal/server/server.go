// SPDX-License-Identifier: Apache-2.0

// Package server is reflopd's daemon: listeners, authentication, sessions,
// workers, rails, Owner controls and audit (PLAN.md §9).
package server

import (
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/chaseungjoon/reflop/pkg/opset"
	pb "github.com/chaseungjoon/reflop/pkg/reflopv1"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/audit"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/auth"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/buildinfo"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/cgroup"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/config"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/gpu"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/paths"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/rails"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/runtime"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/sandbox"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/store"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/supervisor"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/usage"
)

// DeviceSlot is one device this server exposes.
type DeviceSlot struct {
	PB      *pb.Device
	Name    string
	GPU     *gpu.GPU // nil for CPU
	Nodes   []string
	Info    *pb.DeviceInfo
	Enabled []uint32
	Caps    []string
	Runtime *pb.RuntimeInfo
}

// Options configure a Server.
type Options struct {
	Layout              paths.Layout
	AcceptWeakIsolation bool
	Log                 *log.Logger
	Listen              []string // overrides config
	SkipProbe           bool     // tests: do not spawn probe workers
	QuietWorkers        bool     // self-test: worker stderr goes only into the error of a worker that fails
}

// Server is the reflopd daemon.
type Server struct {
	layout    paths.Layout
	opts      Options
	cfgMu     sync.RWMutex
	config    config.Config
	key       *auth.ServerKey
	ids       *auth.Store
	audit     *audit.Log
	usage     *usage.Ledger
	store     *store.Store
	runtime   runtime.Installed
	workerDir string
	gpuInfo   gpu.Info
	cg        *cgroup.Manager
	userNS    bool
	slots     []*DeviceSlot
	logger    *log.Logger

	mu           sync.Mutex
	sessions     map[string]*Session
	conns        map[*clientConn]bool
	nconns       int            // accepted and not yet closed, authenticated or not
	preauth      map[string]int // unauthenticated connections per source address
	fails        failLimiter
	paused       atomic.Bool
	ramErrLogged atomic.Bool
	yielding     atomic.Bool
	events       *broadcaster
	throttles    map[string]*bucket

	listeners []net.Listener
	started   time.Time
}

// New prepares a server (no listening yet).
func New(opts Options) (*Server, error) {
	l := opts.Layout
	if err := l.Ensure(); err != nil {
		return nil, err
	}
	cfg, err := config.Load(l.ConfigFile(), filepath.Join(l.State, "settings.json"))
	if err != nil {
		return nil, err
	}
	if opts.AcceptWeakIsolation {
		cfg.Sandbox.AcceptWeakIsolation = true
	}
	key, err := auth.LoadOrCreateServerKey(l.ServerKey(), l.ServerCert())
	if err != nil {
		return nil, err
	}
	ids, err := auth.Open(l.Identities())
	if err != nil {
		return nil, err
	}
	al, err := audit.Open(l.Audit(), cfg.Audit.RetentionDays)
	if err != nil {
		return nil, err
	}
	ul, err := usage.Open(filepath.Join(l.State, "usage.json"), l.Audit())
	if err != nil {
		return nil, err
	}
	logger := opts.Log
	if logger == nil {
		logger = log.New(os.Stderr, "reflopd: ", log.LstdFlags)
	}
	s := &Server{layout: l, opts: opts, config: cfg, key: key, ids: ids, audit: al, usage: ul, logger: logger,
		sessions: map[string]*Session{}, conns: map[*clientConn]bool{}, preauth: map[string]int{}, events: newBroadcaster(),
		throttles: map[string]*bucket{}, started: time.Now()}
	s.store = store.New(l.Store(), int64(cfg.Limits.UploadChunkBytes), cfg.Limits.MaxTensorNumel,
		func() config.Rails { return s.cfg().Rails },
		func(name string) uint64 { return uint64(s.quotaByID(name).StorageBytes) })
	return s, nil
}

func (s *Server) cfg() config.Config {
	s.cfgMu.RLock()
	defer s.cfgMu.RUnlock()
	return s.config
}

func (s *Server) logf(format string, args ...any) { s.logger.Printf(format, args...) }

// workerLog is where workers' stderr lines go. A worker that fails to start
// carries the last lines in its error, so the self-test does not also print
// them as they arrive.
func (s *Server) workerLog() func(string, ...any) {
	if s.opts.QuietWorkers {
		return nil
	}
	return s.logf
}

// Reload re-reads identities and runtime-changeable settings.
func (s *Server) Reload() error {
	cfg, err := config.Load(s.layout.ConfigFile(), filepath.Join(s.layout.State, "settings.json"))
	if err != nil {
		return err
	}
	if s.opts.AcceptWeakIsolation {
		cfg.Sandbox.AcceptWeakIsolation = true
	}
	s.cfgMu.Lock()
	s.config = cfg
	s.cfgMu.Unlock()
	if err := s.ids.Reload(); err != nil {
		return err
	}
	// Revoked identities lose their sessions immediately (§15.1).
	s.mu.Lock()
	var kill []*Session
	for _, sess := range s.sessions {
		if !s.ids.IsActive(sess.Identity.ID) {
			kill = append(kill, sess)
		}
	}
	var drop []*clientConn
	for c := range s.conns {
		if c.identity != nil && !s.ids.IsActive(c.identity.ID) {
			drop = append(drop, c)
		}
	}
	s.mu.Unlock()
	for _, sess := range kill {
		go sess.close("identity revoked")
	}
	for _, c := range drop {
		c.close()
	}
	return nil
}

// Prepare detects hardware, picks the installed runtime and probes devices.
func (s *Server) Prepare(ctx context.Context) error {
	s.gpuInfo = gpu.Detect()
	lock, err := runtime.Load()
	if err != nil {
		return err
	}
	choice, err := lock.Choose(s.gpuInfo, os.Getenv("REFLOP_RUNTIME_VARIANT"))
	if err != nil {
		return err
	}
	arch, err := runtime.Arch()
	if err != nil {
		return err
	}
	inst, ok := runtime.Find(s.layout.Runtime, lock.ID(choice.Variant, arch))
	if !ok {
		return fmt.Errorf("the %s compute runtime is not installed; run `reflopd up` (or `reflopd runtime install`)", choice.Variant)
	}
	s.runtime = inst
	wd, err := runtime.EnsureWorker(s.layout.Runtime)
	if err != nil {
		return fmt.Errorf("worker package: %w", err)
	}
	s.workerDir = wd
	// Only manage cgroups inside a scope or unit that belongs to reflopd
	// (never the shell's own scope, which other processes share).
	if os.Getenv("REFLOP_SCOPED") == "" {
		s.logf("cgroup rails: not running in a reflopd scope or unit (falling back to RLIMIT_NPROC and memory polling)")
	} else if cg, err := cgroup.Setup(); err == nil {
		s.cg = cg
	} else {
		s.logf("cgroup rails: %v (falling back to RLIMIT_NPROC and memory polling)", err)
	}
	if s.cfg().Sandbox.TryUserNamespaces {
		s.userNS = supervisor.UserNSAvailable()
	}
	s.buildSlots(choice.Variant)
	if s.opts.SkipProbe {
		return nil
	}
	for _, slot := range s.slots {
		if err := s.probe(ctx, slot); err != nil {
			return fmt.Errorf("probe %s: %w", slot.Name, err)
		}
	}
	return nil
}

func (s *Server) buildSlots(variant string) {
	cfg := s.cfg()
	s.slots = nil
	if variant != "cpu" {
		want := map[int]bool{}
		for _, i := range cfg.Devices.CUDA {
			want[i] = true
		}
		idx := 0
		for i := range s.gpuInfo.GPUs {
			g := s.gpuInfo.GPUs[i]
			if len(want) > 0 && !want[g.Index] {
				continue
			}
			s.slots = append(s.slots, &DeviceSlot{
				PB:    &pb.Device{Kind: pb.DeviceKind_DEVICE_CUDA, Index: uint32(idx)},
				Name:  fmt.Sprintf("cuda:%d (%s)", idx, g.Name),
				GPU:   &g,
				Nodes: gpu.DeviceNodes([]gpu.GPU{g}, s.gpuInfo.WSL),
			})
			idx++
		}
	}
	if cfg.Devices.CPU || len(s.slots) == 0 {
		s.slots = append(s.slots, &DeviceSlot{PB: &pb.Device{Kind: pb.DeviceKind_DEVICE_CPU}, Name: "cpu"})
	}
}

func (s *Server) slotFor(d *pb.Device) *DeviceSlot {
	for _, sl := range s.slots {
		if d == nil && sl.PB.Kind == pb.DeviceKind_DEVICE_CUDA {
			return sl
		}
		if d != nil && sl.PB.Kind == d.Kind && sl.PB.Index == d.Index {
			return sl
		}
	}
	if d == nil && len(s.slots) > 0 {
		return s.slots[0]
	}
	return nil
}

// enabledFor filters a device's op table by Owner settings.
func (s *Server) enabledFor(slot *DeviceSlot) []uint32 {
	return slot.Enabled
}

// probe starts one worker to learn the device's capabilities.
func (s *Server) probe(ctx context.Context, slot *DeviceSlot) error {
	sess := &Session{Device: slot, srv: s, seed: 0, numerics: s.allowedNumerics(nil)}
	copy(sess.ID[:], []byte("probe-"+slot.PB.Kind.String()))
	w, err := s.spawnWorker(ctx, sess)
	if err != nil {
		return err
	}
	defer func() { w.Kill("probe done"); <-w.Done() }()
	slot.Info = w.Ready.Devices[0]
	slot.Caps = w.Ready.Capabilities
	slot.Runtime = w.Ready.RuntimeInfo
	slot.Enabled = w.Ready.EnabledOpIds
	s.logf("device %s: %d operators, capabilities %v, sandbox %v", slot.Name, len(slot.Enabled), slot.Caps,
		describeSandbox(w))
	return nil
}

func describeSandbox(w *supervisor.Worker) string {
	r := w.Ready.GetSandbox()
	parts := []string{}
	if r.GetNoNewPrivs() {
		parts = append(parts, "no_new_privs")
	}
	if w.Report.Landlock {
		parts = append(parts, fmt.Sprintf("landlock(abi %d)", w.Report.LandlockABI))
	}
	if r.GetSeccomp() {
		parts = append(parts, "seccomp")
	}
	if w.Spec.UserNS {
		parts = append(parts, "userns+netns")
	}
	if w.Spec.CG != nil {
		parts = append(parts, "cgroup")
	}
	return strings.Join(parts, ", ")
}

func (s *Server) isolationOK() bool {
	if s.cfg().Sandbox.AcceptWeakIsolation {
		return true
	}
	return sandbox.SeccompAvailable() || sandbox.LandlockABI() > 0
}

// spawnWorker starts a sandboxed worker for a session.
func (s *Server) spawnWorker(ctx context.Context, sess *Session) (*supervisor.Worker, error) {
	cfg := s.cfg()
	slot := sess.Device
	init := &pb.WorkerInit{
		SessionId: sess.ID[:], Device: slot.PB, Seed: sess.seed, Numerics: sess.numerics,
		Limits: s.limits(), RequireSeccomp: !cfg.Sandbox.AcceptWeakIsolation && sandbox.SeccompAvailable(),
	}
	if slot.Enabled != nil {
		init.EnabledOpIds = s.filterOps(slot)
	}
	if sess.Identity.Name != "" {
		q := s.quota(sess.Identity.Name)
		init.DeviceMemoryQuotaBytes = uint64(q.DeviceMemoryBytes)
		init.ExecuteFlopsQuota = q.ExecuteFLOPs
	}
	m, err := rails.Memory()
	if err != nil {
		return nil, fmt.Errorf("the host RAM safety rail cannot read memory use (%v); not starting a worker", err)
	}
	init.HostMemoryBudgetBytes = rails.WorkerBudget(cfg.Rails, m, s.workersMemory())
	var uuids []string
	if slot.GPU != nil {
		uuids = []string{slot.GPU.UUID}
		if slot.GPU.UUID == "" {
			uuids = []string{fmt.Sprint(slot.GPU.Index)}
		}
	}
	name := sess.hex()
	scratch := s.layout.SessionScratch(name)
	spec := supervisor.Spec{
		Name: name, Python: s.runtime.PythonBin(), RuntimeDir: s.runtime.Dir, WorkerDir: s.workerDir, Scratch: scratch,
		GPUUUIDs: uuids, DeviceNodes: slot.Nodes, WSL: s.gpuInfo.WSL, Init: init,
		UserNS: s.userNS, CG: s.cg, MaxPIDs: cfg.Rails.MaxPIDs,
		RequireLandlock: !cfg.Sandbox.AcceptWeakIsolation && sandbox.LandlockABI() > 0,
		FSize:           rails.DiskBudget(cfg.Rails, s.layout.State),
		MaxFrame:        uint32(cfg.Limits.MaxFrameBytes) + 1<<20,
		Log:             s.workerLog(),
		TestProbe:       os.Getenv("REFLOP_TEST_PROBE"),
	}
	if spec.FSize == 0 {
		spec.FSize = 1 << 20
	}
	// The filter needs the worker's pid, so it is built right after fork.
	w, err := supervisor.SpawnWithFilter(ctx, spec, func(pid int) ([]byte, error) {
		if !sandbox.SeccompAvailable() {
			return nil, nil
		}
		f, err := sandbox.Policy{SelfPID: pid}.Filter()
		if err != nil {
			if init.RequireSeccomp {
				// Fail closed (SEC_SWEEP M2): never start a worker without the filter.
				return nil, fmt.Errorf("seccomp filter: %w", err)
			}
			s.logf("seccomp filter: %v (accept_weak_isolation is set; starting without it)", err)
			return nil, nil
		}
		return sandbox.Encode(f), nil
	})
	if err != nil {
		os.RemoveAll(scratch)
		return nil, err
	}
	return w, nil
}

func (s *Server) filterOps(slot *DeviceSlot) []uint32 {
	cfg := s.cfg()
	disabled := map[string]bool{}
	for _, n := range cfg.Ops.Disabled {
		disabled[n] = true
	}
	var out []uint32
	for _, id := range slot.Enabled {
		o, _ := opset.ByID(id)
		name := o.Name
		if disabled[name] {
			continue
		}
		if c := o.Capability; c != "" {
			if on, ok := cfg.Ops.Capabilities[c]; ok && !on {
				continue
			}
		}
		out = append(out, id)
	}
	return out
}

func (s *Server) limits() *pb.Limits {
	cfg := s.cfg()
	l := cfg.Limits
	m, _ := rails.Memory()
	return &pb.Limits{
		MaxFrameBytes: uint64(l.MaxFrameBytes), MaxNodes: l.MaxNodes, MaxGraphInputs: l.MaxGraphInputs,
		MaxDims: l.MaxDims, MaxTensorNumel: l.MaxTensorNumel, MaxInlineTensorBytes: uint64(l.MaxInlineTensorBytes),
		MaxConstantBytes: uint64(l.MaxConstantBytes), MaxFetchBytes: uint64(l.MaxFetchBytes),
		MaxListLength: l.MaxListLength, UploadChunkBytes: uint64(l.UploadChunkBytes),
		MaxHandlesPerSession: l.MaxHandlesPerSession, MaxGraphsPerSession: l.MaxGraphsPerSession,
		MaxManifestBytes:   uint64(l.MaxManifestBytes),
		DiskFreeFloorBytes: uint64(cfg.Rails.DiskFreeFloorBytes), DiskFreeFloorPercent: cfg.Rails.DiskFreeFloorPercent,
		HostRamFloorBytes: rails.RAMFloor(cfg.Rails, m.Total),
	}
}

func (s *Server) capabilitiesFor(id *auth.Identity) *pb.ServerCapabilities {
	c := &pb.ServerCapabilities{ProtocolVersions: []uint32{ProtocolVersion}, OpsetVersions: []uint32{1}}
	seen := map[string]bool{}
	for _, slot := range s.slots {
		if slot.Info != nil {
			c.Devices = append(c.Devices, slot.Info)
		} else {
			c.Devices = append(c.Devices, &pb.DeviceInfo{Device: slot.PB, Name: slot.Name})
		}
		for _, k := range slot.Caps {
			if !seen[k] {
				seen[k] = true
				c.Capabilities = append(c.Capabilities, k)
			}
		}
	}
	c.Capabilities = append(c.Capabilities, "usage")
	if len(s.slots) > 0 {
		c.EnabledOpIds = s.filterOps(s.slots[0])
		c.RuntimeInfo = s.slots[0].Runtime
	}
	if c.RuntimeInfo == nil {
		c.RuntimeInfo = &pb.RuntimeInfo{}
	}
	c.RuntimeInfo.ReflopdVersion = buildinfo.Version
	c.RuntimeInfo.DriverVersion = s.gpuInfo.Driver
	c.RuntimeInfo.RuntimeVariant = s.runtime.Marker.Variant
	c.Numerics = s.allowedNumerics(nil)
	c.Limits = s.limits()
	q := s.quota(id.Name)
	c.Limits.QuotaDeviceMemoryBytes = uint64(q.DeviceMemoryBytes)
	c.Limits.QuotaMaxSessions = q.MaxSessions
	c.Limits.QuotaExecuteWallMs = q.ExecuteWallMS
	c.Limits.QuotaExecuteFlops = q.ExecuteFLOPs
	c.Limits.QuotaStorageBytes = uint64(q.StorageBytes)
	c.Limits.QuotaUploadBytesPerSec = uint64(q.UploadBytesPerSec)
	return c
}

// enabledOpsFor returns the op ids enabled on a given device for clients.
func (s *Server) enabledOpsFor(d *pb.Device) []uint32 {
	if slot := s.slotFor(d); slot != nil {
		return s.filterOps(slot)
	}
	return nil
}

// allowedNumerics applies the Owner's allowlist to a client request.
func (s *Server) allowedNumerics(req *pb.Numerics) *pb.Numerics {
	n := s.cfg().Numerics
	out := &pb.Numerics{Tf32Matmul: n.TF32Matmul, Tf32Cudnn: n.TF32CuDNN, Deterministic: n.Deterministic, CudnnBenchmark: n.CuDNNBenchmark}
	if req != nil {
		// All four switches only trade speed for precision or reproducibility
		// within the Guest's own session, so clients may set them.
		out.Tf32Matmul = req.Tf32Matmul
		out.Tf32Cudnn = req.Tf32Cudnn
		out.Deterministic = req.Deterministic
		out.CudnnBenchmark = req.CudnnBenchmark
	}
	return out
}

func (s *Server) quota(name string) config.Quota { return s.cfg().Quotas[name] }

func (s *Server) quotaByID(idHex string) config.Quota {
	for _, id := range s.ids.List() {
		if id.ID == idHex {
			return s.quota(id.Name)
		}
	}
	return config.Quota{}
}

// ---------------------------------------------------------------- registry

func (s *Server) addSession(sess *Session) {
	s.mu.Lock()
	s.sessions[sess.hex()] = sess
	s.mu.Unlock()
}

func (s *Server) removeSession(sess *Session) {
	s.mu.Lock()
	delete(s.sessions, sess.hex())
	s.mu.Unlock()
}

func (s *Server) countSessions(idHex string) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	n := 0
	for _, sess := range s.sessions {
		if sess.Identity.ID == idHex {
			n++
		}
	}
	return n
}

func (s *Server) addConn(c *clientConn) {
	s.mu.Lock()
	s.conns[c] = true
	s.mu.Unlock()
}

func (s *Server) removeConn(c *clientConn) {
	s.mu.Lock()
	delete(s.conns, c)
	s.mu.Unlock()
	s.events.unsubscribe(c)
}

func (s *Server) allSessions() []*Session {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]*Session, 0, len(s.sessions))
	for _, x := range s.sessions {
		out = append(out, x)
	}
	return out
}

func (s *Server) workersMemory() uint64 {
	if s.cg != nil {
		return s.cg.MemoryCurrent()
	}
	var n uint64
	for _, sess := range s.allSessions() {
		if sess.worker != nil {
			n += sess.worker.Memory()
		}
	}
	return n
}

// ---------------------------------------------------------------- serving

// TLSConfig is the daemon-mode TLS configuration.
func (s *Server) TLSConfig() *tls.Config {
	return &tls.Config{
		Certificates: []tls.Certificate{s.key.Cert},
		MinVersion:   tls.VersionTLS13,
		NextProtos:   []string{"reflop/1"},
		ClientAuth:   tls.NoClientCert, // clients authenticate inside the channel (ADR-006)
	}
}

// Listen opens the TCP listeners and the internal sockets.
func (s *Server) Listen() error {
	addrs := s.cfg().Listen
	if len(s.opts.Listen) > 0 {
		addrs = s.opts.Listen
	}
	for _, a := range addrs {
		ln, err := tls.Listen("tcp", a, s.TLSConfig())
		if err != nil {
			s.closeListeners()
			return fmt.Errorf("listen %s: %w", a, err)
		}
		s.listeners = append(s.listeners, ln)
		s.logf("listening on %s (TLS 1.3)", ln.Addr())
		if !isLoopback(a) {
			s.logf("warning: %s is reachable from the network; prefer a Tailscale/WireGuard address", a)
		}
	}
	return nil
}

func isLoopback(addr string) bool {
	h, _, err := net.SplitHostPort(addr)
	if err != nil {
		return false
	}
	ip := net.ParseIP(h)
	return h == "localhost" || (ip != nil && ip.IsLoopback())
}

func (s *Server) closeListeners() {
	for _, ln := range s.listeners {
		ln.Close()
	}
}

// Addrs returns the bound listener addresses.
func (s *Server) Addrs() []string {
	var out []string
	for _, ln := range s.listeners {
		out = append(out, ln.Addr().String())
	}
	return out
}

// Fingerprint is the server key fingerprint.
func (s *Server) Fingerprint() [32]byte { return s.key.Fingerprint }

// Serve accepts connections until ctx is done.
func (s *Server) Serve(ctx context.Context) error {
	ctl, err := s.listenControl()
	if err != nil {
		return err
	}
	bridge, err := s.listenBridge()
	if err != nil {
		ctl.Close()
		return err
	}
	go s.monitor(ctx)
	usageCtx, stopUsage := context.WithCancel(context.Background())
	usageDone := make(chan struct{})
	go func() {
		s.usage.Run(usageCtx, 30*time.Second)
		close(usageDone)
	}()
	go s.acceptControl(ctx, ctl)
	go s.acceptBridge(ctx, bridge)
	var wg sync.WaitGroup
	for _, ln := range s.listeners {
		wg.Add(1)
		go func(ln net.Listener) {
			defer wg.Done()
			s.acceptTLS(ctx, ln)
		}(ln)
	}
	<-ctx.Done()
	s.events.publish(pb.Event_KIND_SHUTDOWN, "reflopd is shutting down")
	s.closeListeners()
	ctl.Close()
	bridge.Close()
	os.Remove(s.layout.ControlSocket())
	os.Remove(s.bridgePath())
	for _, sess := range s.allSessions() {
		sess.close("reflopd stopped")
	}
	s.mu.Lock()
	for c := range s.conns {
		c.rw.Close()
	}
	s.mu.Unlock()
	wg.Wait()
	stopUsage()
	<-usageDone
	s.audit.Close()
	return nil
}

func (s *Server) acceptTLS(ctx context.Context, ln net.Listener) {
	for {
		conn, err := ln.Accept()
		if err != nil {
			if errors.Is(err, net.ErrClosed) || ctx.Err() != nil {
				return
			}
			time.Sleep(50 * time.Millisecond)
			continue
		}
		// The slot is taken before the TLS handshake, so connections that
		// never authenticate still count against max_connections (M1).
		adm := s.admit(conn.RemoteAddr().String(), true)
		if adm == nil {
			conn.Close()
			continue
		}
		go func() {
			defer adm.release()
			tc := conn.(*tls.Conn)
			tc.SetDeadline(time.Now().Add(15 * time.Second))
			if err := tc.HandshakeContext(ctx); err != nil {
				conn.Close()
				return
			}
			tc.SetDeadline(time.Time{})
			s.handleConn(tc, "tls", conn.RemoteAddr().String(), "", adm)
		}()
	}
}

// ServeStdio handles one SSH pipe connection in this process's stdio by
// bridging it to the running daemon (PLAN.md §15.4).
func ServeStdio(layout paths.Layout, identityID string, stdin io.Reader, stdout io.Writer) error {
	return bridgeStdio(layout, identityID, stdin, stdout)
}

func firstLine(s string) string { l, _, _ := strings.Cut(s, "\n"); return l }
