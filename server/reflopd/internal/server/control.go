// SPDX-License-Identifier: Apache-2.0

package server

import (
	"bufio"
	"context"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	pb "github.com/chaseungjoon/reflop/pkg/reflopv1"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/audit"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/buildinfo"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/config"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/gpu"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/paths"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/rails"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/sandbox"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/suspend"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/usage"
)

// ControlRequest is one Owner command on the control socket.
type ControlRequest struct {
	Cmd  string          `json:"cmd"`
	Args json.RawMessage `json:"args,omitempty"`
}

// ControlResponse answers a ControlRequest.
type ControlResponse struct {
	OK    bool            `json:"ok"`
	Error string          `json:"error,omitempty"`
	Data  json.RawMessage `json:"data,omitempty"`
}

func listenUnix(path string) (net.Listener, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return nil, err
	}
	if c, err := net.DialTimeout("unix", path, 200*time.Millisecond); err == nil {
		c.Close()
		return nil, fmt.Errorf("another reflopd is already running (%s answers)", path)
	}
	os.Remove(path)
	old := umask(0o177)
	ln, err := net.Listen("unix", path)
	umask(old)
	if err != nil {
		return nil, err
	}
	os.Chmod(path, 0o600)
	return ln, nil
}

func (s *Server) listenControl() (net.Listener, error) { return listenUnix(s.layout.ControlSocket()) }

func (s *Server) bridgePath() string { return filepath.Join(s.layout.Run, "bridge.sock") }

func (s *Server) listenBridge() (net.Listener, error) { return listenUnix(s.bridgePath()) }

// peerIsOwner checks SO_PEERCRED: only the Owner's uid (or root) may use the
// internal sockets, even if file permissions were loosened by mistake.
func peerIsOwner(c net.Conn) bool {
	uc, ok := c.(*net.UnixConn)
	if !ok {
		return false
	}
	uid, ok := peerUID(uc)
	return ok && (uid == os.Getuid() || uid == 0)
}

func (s *Server) acceptControl(ctx context.Context, ln net.Listener) {
	for {
		c, err := ln.Accept()
		if err != nil {
			if ctx.Err() != nil || errors.Is(err, net.ErrClosed) {
				return
			}
			continue
		}
		if !peerIsOwner(c) {
			c.Close()
			continue
		}
		go s.serveControl(ctx, c)
	}
}

func (s *Server) acceptBridge(ctx context.Context, ln net.Listener) {
	for {
		c, err := ln.Accept()
		if err != nil {
			if ctx.Err() != nil || errors.Is(err, net.ErrClosed) {
				return
			}
			continue
		}
		if !peerIsOwner(c) {
			c.Close()
			continue
		}
		// sshd already authenticated the Guest; the slot still counts.
		adm := s.admit("ssh", false)
		if adm == nil {
			c.Close()
			continue
		}
		go func() {
			defer adm.release()
			c.SetReadDeadline(time.Now().Add(10 * time.Second))
			br := bufio.NewReader(c)
			line, err := br.ReadString('\n')
			if err != nil {
				c.Close()
				return
			}
			c.SetReadDeadline(time.Time{})
			f := strings.Fields(line)
			if len(f) != 3 || f[0] != "reflop-bridge" || f[1] != "1" || len(f[2]) != 32 {
				c.Close()
				return
			}
			if _, err := hex.DecodeString(f[2]); err != nil {
				c.Close()
				return
			}
			s.handleConn(&bufferedConn{Conn: c, r: br}, "ssh", "ssh", f[2], adm)
		}()
	}
}

type bufferedConn struct {
	net.Conn
	r *bufio.Reader
}

func (b *bufferedConn) Read(p []byte) (int, error) { return b.r.Read(p) }

func (s *Server) serveControl(ctx context.Context, c net.Conn) {
	defer c.Close()
	sc := bufio.NewScanner(c)
	sc.Buffer(make([]byte, 64<<10), 1<<20)
	enc := json.NewEncoder(c)
	for sc.Scan() {
		var req ControlRequest
		if err := json.Unmarshal(sc.Bytes(), &req); err != nil {
			enc.Encode(ControlResponse{Error: "bad request"})
			continue
		}
		data, err := s.control(ctx, req)
		resp := ControlResponse{OK: err == nil}
		if err != nil {
			resp.Error = err.Error()
		}
		if data != nil {
			resp.Data, _ = json.Marshal(data)
		}
		enc.Encode(resp)
	}
}

// SessionView is what `reflopd top` shows per session.
type SessionView struct {
	ID          string `json:"id"`
	Identity    string `json:"identity"`
	Device      string `json:"device"`
	Attached    bool   `json:"attached"`
	Age         string `json:"age"`
	Executes    uint64 `json:"executes"`
	FLOPs       uint64 `json:"flops"`
	HostMemory  uint64 `json:"host_memory"`
	DeviceAlloc uint64 `json:"device_memory"`
	Handles     uint32 `json:"handles"`
	Graphs      uint32 `json:"graphs"`
	PID         int    `json:"pid"`
	Suspended   bool   `json:"suspended,omitempty"`
}

// StatusView is `reflopd status`.
type StatusView struct {
	Version     string        `json:"version"`
	Uptime      string        `json:"uptime"`
	Listen      []string      `json:"listen"`
	Fingerprint string        `json:"fingerprint"`
	Runtime     string        `json:"runtime"`
	Devices     []string      `json:"devices"`
	Paused      bool          `json:"paused"`
	Yielding    bool          `json:"yielding"`
	Connections int           `json:"connections"`
	Sessions    []SessionView `json:"sessions"`
	Sandbox     []string      `json:"sandbox"`
	Level       int           `json:"level"`
	Checkpoints string        `json:"checkpoints"` // "off" or the interval
}

func (s *Server) sandboxLayers() []string {
	var out []string
	if abi := sandbox.LandlockABI(); abi > 0 {
		out = append(out, fmt.Sprintf("landlock abi %d", abi))
	}
	if sandbox.SeccompAvailable() {
		out = append(out, "seccomp")
	}
	if s.userNS {
		out = append(out, "user+network namespaces")
	}
	if s.cg != nil {
		out = append(out, "cgroup v2 (memory, pids)")
	} else {
		out = append(out, "RLIMIT_NPROC + memory polling (no delegated cgroup)")
	}
	return out
}

func (s *Server) status(withStats bool) StatusView {
	fp := s.key.Fingerprint
	v := StatusView{
		Version: buildinfo.Version, Uptime: time.Since(s.started).Truncate(time.Second).String(),
		Listen: s.Addrs(), Fingerprint: hex.EncodeToString(fp[:]),
		Runtime: s.runtime.Marker.ID, Paused: s.paused.Load(), Yielding: s.yielding.Load(), Sandbox: s.sandboxLayers(),
		Checkpoints: "off",
	}
	if ck := s.cfg().Checkpoints; ck.Enabled {
		v.Checkpoints = "every " + time.Duration(ck.Interval).String()
	}
	if s.layout.System {
		v.Level = 1
	}
	for _, sl := range s.devices() {
		v.Devices = append(v.Devices, sl.Name)
	}
	for _, name := range s.pendingDevices() {
		v.Devices = append(v.Devices, name+" (not served: no free GPU memory; retried every "+probeRetryInterval.String()+")")
	}
	s.mu.Lock()
	v.Connections = len(s.conns)
	s.mu.Unlock()
	for _, sess := range s.allSessions() {
		sess.mu.Lock()
		sv := SessionView{ID: sess.hex(), Identity: sess.Identity.Name, Device: sess.Device.Name,
			Attached: sess.conn != nil, Age: time.Since(sess.created).Truncate(time.Second).String(),
			Executes: sess.executes, FLOPs: sess.flops, Suspended: sess.suspended}
		sess.mu.Unlock()
		if sess.worker != nil {
			sv.PID = sess.worker.PID()
			sv.HostMemory = sess.worker.Memory()
		}
		if withStats && !sv.Suspended {
			if st := sess.stats(500 * time.Millisecond); st != nil {
				sv.DeviceAlloc = st.DeviceMemoryAllocated
				sv.Handles = st.Handles
				sv.Graphs = st.Graphs
			}
		}
		v.Sessions = append(v.Sessions, sv)
	}
	return v
}

// usageNow is the ledger plus what live sessions and connections have used
// so far (held time and traffic are only booked when they end). A non-empty
// name restricts it to that identity.
func (s *Server) usageNow(name string) usage.Report {
	rep := s.usage.Snapshot()
	if name != "" {
		t := rep.Identities[name]
		rep.Identities = map[string]*usage.Totals{}
		if t != nil {
			rep.Identities[name] = t
		}
	}
	for _, sess := range s.allSessions() {
		if name == "" || sess.Identity.Name == name {
			rep.Get(sess.Identity.Name).SessionUs += uint64(time.Since(sess.created).Microseconds())
		}
	}
	s.mu.Lock()
	for c := range s.conns {
		if n := c.account.Load(); n != nil && (name == "" || *n == name) {
			t := rep.Get(*n)
			t.BytesUp += c.bytesUp.Load()
			t.BytesDown += c.bytesDown.Load()
		}
	}
	s.mu.Unlock()
	return rep
}

// guestUsage is what a Guest may see: its own totals only.
func (s *Server) guestUsage(name string) *pb.UsageReport {
	rep := s.usageNow(name)
	t := rep.Get(name)
	u := &pb.UsageReport{IdentityName: name, SinceUnix: rep.Since.Unix(), Executes: t.Executes, DeviceUs: t.DeviceUs,
		Flops: t.FLOPs, PeakMemoryBytes: t.PeakMemory, Sessions: t.Sessions, SessionUs: t.SessionUs,
		BytesUp: t.BytesUp, BytesDown: t.BytesDown}
	if !t.First.IsZero() {
		u.FirstUnix, u.LastUnix = t.First.Unix(), t.Last.Unix()
	}
	devs := make([]string, 0, len(t.ByDevice))
	for d := range t.ByDevice {
		devs = append(devs, d)
	}
	sort.Strings(devs)
	for _, d := range devs {
		u.ByDevice = append(u.ByDevice, &pb.DeviceUsage{Device: d, DeviceUs: t.ByDevice[d]})
	}
	return u
}

func (s *Server) control(ctx context.Context, req ControlRequest) (any, error) {
	switch req.Cmd {
	case "usage":
		return s.usageNow(""), nil
	case "status":
		return s.status(false), nil
	case "top":
		return s.status(true), nil
	case "pause":
		var a struct {
			Now bool `json:"now"`
		}
		json.Unmarshal(req.Args, &a)
		return s.Pause(a.Now, "Owner"), nil
	case "resume":
		s.Resume("Owner")
		return nil, nil
	case "say":
		var a struct {
			Text string `json:"text"`
		}
		json.Unmarshal(req.Args, &a)
		s.events.publish(pb.Event_KIND_OWNER_MESSAGE, a.Text)
		return nil, nil
	case "kill":
		var a struct {
			Target string `json:"target"`
		}
		json.Unmarshal(req.Args, &a)
		n := s.Kill(a.Target, "killed by the Owner")
		if n == 0 {
			return nil, fmt.Errorf("no session or identity matches %q", a.Target)
		}
		return map[string]int{"killed": n}, nil
	case "reload":
		return nil, s.Reload()
	case "yield":
		var a struct {
			Auto    bool  `json:"auto"`
			Suspend *bool `json:"suspend"`
		}
		json.Unmarshal(req.Args, &a)
		s.cfgMu.Lock()
		s.config.Yield.Auto = a.Auto
		if a.Suspend != nil {
			s.config.Yield.Suspend = *a.Suspend
		}
		s.cfgMu.Unlock()
		return nil, nil
	case "stop":
		go func() {
			time.Sleep(100 * time.Millisecond)
			if p, err := os.FindProcess(os.Getpid()); err == nil {
				p.Signal(os.Interrupt)
			}
		}()
		return nil, nil
	}
	return nil, fmt.Errorf("unknown command %q", req.Cmd)
}

// Pause queues new executes. With now, GPU workers are also suspended:
// their GPU memory moves to host RAM and comes back on Resume (ADR-030). A
// worker that cannot be suspended (not enough free host RAM, a kernel that
// does not finish within suspend.LockTimeout, a driver without the API, or a
// CPU worker) is killed, as before; its last checkpoint stays recoverable.
func (s *Server) Pause(now bool, by string) PauseResult {
	s.paused.Store(true)
	s.events.publish(pb.Event_KIND_PAUSED, "the Owner paused GPU sharing; queued work resumes later")
	s.audit.Write(audit.Record{Identity: "owner", Event: "pause", Fields: map[string]any{"now": now, "by": by}})
	if !now {
		return PauseResult{}
	}
	return s.suspendAll(true, "the Owner paused GPU sharing now (pause --now)")
}

// PauseResult is what `reflopd pause --now` reports.
type PauseResult struct {
	Suspended int      `json:"suspended"`
	Killed    int      `json:"killed"`
	Notes     []string `json:"notes,omitempty"`
}

// suspendAll suspends every running GPU session. With kill, a session that
// cannot be suspended is ended (pause --now); without, it keeps running
// paused (auto-yield).
func (s *Server) suspendAll(kill bool, why string) PauseResult {
	var res PauseResult
	sessions := s.allSessions()
	procMem, smiOK := gpu.ProcessMemory()
	headroom := uint64(0)
	if m, err := rails.Memory(); err == nil {
		using := s.workersMemory()
		if b := rails.WorkerBudget(s.cfg().Rails, m, using); b > using {
			headroom = b - using
		}
	}
	type job struct {
		sess *Session
		pid  int
	}
	var jobs []job
	end := func(sess *Session, reason string) {
		if !kill {
			return
		}
		res.Killed++
		res.Notes = append(res.Notes, fmt.Sprintf("session %s of %q: killed (%s)", sess.hex()[:8], sess.Identity.Name, reason))
		s.audit.Write(audit.Record{Identity: sess.Identity.Name, ID: sess.Identity.ID, Session: sess.hex(), Event: "killed",
			Fields: map[string]any{"reason": why + ": " + reason}})
		go sess.close("killed by the Owner (pause --now): " + reason)
	}
	for _, sess := range sessions {
		sess.mu.Lock()
		done := sess.suspended || sess.closed
		sess.mu.Unlock()
		if done || sess.worker == nil {
			continue
		}
		if sess.Device.GPU == nil {
			end(sess, "a CPU session cannot be suspended")
			continue
		}
		pid := sess.worker.PID()
		need := procMem[pid]
		switch {
		case !smiOK:
			end(sess, "nvidia-smi could not report its GPU memory")
		case need+need/8 > headroom:
			end(sess, fmt.Sprintf("suspending needs %s of host RAM; %s is free above the RAM floor",
				config.Bytes(need), config.Bytes(headroom)))
		default:
			headroom -= need + need/8
			jobs = append(jobs, job{sess, pid})
		}
	}
	type outcome struct {
		sess *Session
		err  error
	}
	results := make(chan outcome, len(jobs))
	for _, j := range jobs {
		go func(j job) {
			sess := j.sess
			// Mark first: the watchdog must not count the time the lock
			// waits for a kernel, and no request should start meanwhile.
			sess.mu.Lock()
			sess.suspended = true
			sess.mu.Unlock()
			err := s.suspendHelper(sess).Suspend(j.pid)
			results <- outcome{sess, err}
		}(j)
	}
	for range jobs {
		o := <-results
		if o.err == nil {
			res.Suspended++
			s.audit.Write(audit.Record{Identity: o.sess.Identity.Name, ID: o.sess.Identity.ID, Session: o.sess.hex(),
				Event: "suspended", Fields: map[string]any{"by": why}})
			s.logf("session %s of %q suspended: its GPU memory is in host RAM until resume", o.sess.hex()[:8], o.sess.Identity.Name)
			continue
		}
		o.sess.mu.Lock()
		o.sess.suspended = false
		o.sess.mu.Unlock()
		s.logf("session %s: suspend failed: %v", o.sess.hex()[:8], o.err)
		if errors.Is(o.err, suspend.ErrLockTimeout) || errors.Is(o.err, suspend.ErrUnsupported) {
			end(o.sess, o.err.Error())
			continue
		}
		// Unknown driver state: the worker may be locked. End it either way.
		res.Killed++
		res.Notes = append(res.Notes, fmt.Sprintf("session %s of %q: killed (suspend failed: %v)", o.sess.hex()[:8], o.sess.Identity.Name, o.err))
		go o.sess.close("suspend failed: " + firstLine(o.err.Error()))
	}
	if res.Suspended > 0 {
		s.events.publish(pb.Event_KIND_SUSPENDED, "the Owner needs the GPU now: your session's GPU memory was moved to "+
			"host RAM and comes back when sharing resumes")
	}
	return res
}

func (s *Server) suspendHelper(sess *Session) suspend.Helper {
	h := suspend.Helper{Python: s.runtime.PythonBin()}
	if sess.Device.GPU != nil {
		h.GPUUUID = sess.Device.GPU.UUID
	}
	return h
}

// resumeSuspended brings suspended workers back before held work is released.
func (s *Server) resumeSuspended() {
	var wg sync.WaitGroup
	for _, sess := range s.allSessions() {
		sess.mu.Lock()
		sus := sess.suspended && !sess.closed
		sess.mu.Unlock()
		if !sus {
			continue
		}
		wg.Add(1)
		go func(sess *Session) {
			defer wg.Done()
			err := s.suspendHelper(sess).Resume(sess.worker.PID())
			sess.mu.Lock()
			sess.suspended = false
			now := time.Now()
			for _, r := range sess.routes {
				r.start = now // the watchdog does not count the suspension
			}
			sess.mu.Unlock()
			if err != nil {
				s.logf("session %s: resume failed: %v", sess.hex()[:8], err)
				sess.close("could not resume after a suspend: " + firstLine(err.Error()))
				return
			}
			s.audit.Write(audit.Record{Identity: sess.Identity.Name, ID: sess.Identity.ID, Session: sess.hex(), Event: "resumed"})
		}(sess)
	}
	wg.Wait()
}

// Resume brings suspended workers back and releases queued work.
func (s *Server) Resume(by string) {
	s.resumeSuspended()
	s.paused.Store(false)
	s.events.publish(pb.Event_KIND_RESUMED, "GPU sharing resumed")
	s.audit.Write(audit.Record{Identity: "owner", Event: "resume", Fields: map[string]any{"by": by}})
	for _, sess := range s.allSessions() {
		sess.release()
	}
}

// Kill terminates sessions matching an identity name or session id prefix.
func (s *Server) Kill(target, why string) int {
	n := 0
	for _, sess := range s.allSessions() {
		if target != "" && (sess.Identity.Name == target || strings.HasPrefix(sess.hex(), target)) {
			s.audit.Write(audit.Record{Identity: sess.Identity.Name, ID: sess.Identity.ID, Session: sess.hex(), Event: "killed",
				Fields: map[string]any{"reason": why}})
			go sess.closeFinal(why)
			n++
		}
	}
	return n
}

// ControlClient talks to a running daemon.
func ControlClient(layout paths.Layout, req ControlRequest, timeout time.Duration) (ControlResponse, error) {
	c, err := net.DialTimeout("unix", layout.ControlSocket(), 2*time.Second)
	if err != nil {
		return ControlResponse{}, fmt.Errorf("reflopd is not running (%v)", err)
	}
	defer c.Close()
	c.SetDeadline(time.Now().Add(timeout))
	if err := json.NewEncoder(c).Encode(req); err != nil {
		return ControlResponse{}, err
	}
	var resp ControlResponse
	if err := json.NewDecoder(c).Decode(&resp); err != nil {
		return resp, err
	}
	if !resp.OK {
		return resp, errors.New(resp.Error)
	}
	return resp, nil
}

func bridgeStdio(layout paths.Layout, identityID string, stdin io.Reader, stdout io.Writer) error {
	c, err := net.DialTimeout("unix", filepath.Join(layout.Run, "bridge.sock"), 5*time.Second)
	if err != nil {
		return fmt.Errorf("reflopd is not running on this host: %v", err)
	}
	defer c.Close()
	if _, err := fmt.Fprintf(c, "reflop-bridge 1 %s\n", identityID); err != nil {
		return err
	}
	done := make(chan struct{}, 2)
	go func() { io.Copy(c, stdin); c.(*net.UnixConn).CloseWrite(); done <- struct{}{} }()
	go func() { io.Copy(stdout, c); done <- struct{}{} }()
	<-done
	return nil
}
