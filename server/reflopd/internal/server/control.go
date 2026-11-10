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
	"strings"
	"time"

	pb "github.com/chaseungjoon/reflop/pkg/reflopv1"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/audit"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/buildinfo"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/paths"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/sandbox"
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
		go func() {
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
			s.handleConn(&bufferedConn{Conn: c, r: br}, "ssh", "ssh", f[2])
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
	}
	if s.layout.System {
		v.Level = 1
	}
	for _, sl := range s.slots {
		v.Devices = append(v.Devices, sl.Name)
	}
	s.mu.Lock()
	v.Connections = len(s.conns)
	s.mu.Unlock()
	for _, sess := range s.allSessions() {
		sess.mu.Lock()
		sv := SessionView{ID: sess.hex(), Identity: sess.Identity.Name, Device: sess.Device.Name,
			Attached: sess.conn != nil, Age: time.Since(sess.created).Truncate(time.Second).String(),
			Executes: sess.executes, FLOPs: sess.flops}
		sess.mu.Unlock()
		if sess.worker != nil {
			sv.PID = sess.worker.PID()
			sv.HostMemory = sess.worker.Memory()
		}
		if withStats {
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
// so far (held time and traffic are only booked when they end).
func (s *Server) usageNow() usage.Report {
	rep := s.usage.Snapshot()
	for _, sess := range s.allSessions() {
		rep.Get(sess.Identity.Name).SessionUs += uint64(time.Since(sess.created).Microseconds())
	}
	s.mu.Lock()
	for c := range s.conns {
		if name := c.account.Load(); name != nil {
			t := rep.Get(*name)
			t.BytesUp += c.bytesUp.Load()
			t.BytesDown += c.bytesDown.Load()
		}
	}
	s.mu.Unlock()
	return rep
}

func (s *Server) control(ctx context.Context, req ControlRequest) (any, error) {
	switch req.Cmd {
	case "usage":
		return s.usageNow(), nil
	case "status":
		return s.status(false), nil
	case "top":
		return s.status(true), nil
	case "pause":
		var a struct {
			Now bool `json:"now"`
		}
		json.Unmarshal(req.Args, &a)
		s.Pause(a.Now, "Owner")
		return nil, nil
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
			Auto bool `json:"auto"`
		}
		json.Unmarshal(req.Args, &a)
		s.cfgMu.Lock()
		s.config.Yield.Auto = a.Auto
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

// Pause queues new executes; with now, running workers are killed too.
func (s *Server) Pause(now bool, by string) {
	s.paused.Store(true)
	s.events.publish(pb.Event_KIND_PAUSED, "the Owner paused GPU sharing; queued work resumes later")
	s.audit.Write(audit.Record{Identity: "owner", Event: "pause", Fields: map[string]any{"now": now, "by": by}})
	if now {
		for _, sess := range s.allSessions() {
			go sess.close("killed by the Owner (pause --now)")
		}
	}
}

// Resume releases queued work.
func (s *Server) Resume(by string) {
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
			go sess.close(why)
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
