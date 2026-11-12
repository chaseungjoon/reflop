// SPDX-License-Identifier: Apache-2.0

package server

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"golang.org/x/sys/unix"

	pb "github.com/chaseungjoon/reflop/pkg/reflopv1"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/audit"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/auth"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/rails"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/store"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/supervisor"
)

type route struct {
	gen    uint64
	rid    uint64
	kind   string
	start  time.Time
	nodes  int
	hash   []byte
	onDone func(*pb.Envelope)
}

// Session is one worker-backed unit of work (PLAN.md §6.1).
type Session struct {
	ID       [16]byte
	Identity auth.Identity
	Device   *DeviceSlot
	srv      *Server
	worker   *supervisor.Worker
	numerics *pb.Numerics
	seed     uint64
	created  time.Time
	quiet    bool // probe or self-test: worker stderr is not logged (see workerLog)

	mu        sync.Mutex
	conn      *clientConn
	detached  time.Time
	nextID    uint64
	routes    map[uint64]*route
	streams   map[[2]uint64]uint64 // (gen, client rid) → internal id, for upload streams
	held      []*pb.Envelope       // queued while the Owner has paused execution
	heldConn  []*clientConn
	datasets  map[string]bool
	closed    bool
	lostWhy   string
	executes  uint64
	flops     uint64
	statsWait []chan *pb.WorkerStats
	lastStats *pb.WorkerStats
}

func (s *Server) openSession(c *clientConn, req *pb.OpenSession) (*Session, error) {
	slot := s.slotFor(req.Device)
	if slot == nil {
		return nil, statusf(pb.ErrorCode_ERROR_INVALID_GRAPH, "this server has no such device")
	}
	if q := s.quota(c.identity.Name); q.MaxSessions > 0 && s.countSessions(c.identity.ID) >= int(q.MaxSessions) {
		return nil, statusf(pb.ErrorCode_ERROR_BUSY, "identity %q already has %d sessions (Owner quota)", c.identity.Name, q.MaxSessions)
	}
	if !s.isolationOK() {
		return nil, statusf(pb.ErrorCode_ERROR_PERMISSION_DENIED, "the server cannot sandbox workers (no seccomp and no Landlock); the Owner must fix this or pass --accept-weak-isolation")
	}
	sess := &Session{Identity: *c.identity, Device: slot, srv: s, routes: map[uint64]*route{},
		streams: map[[2]uint64]uint64{}, datasets: map[string]bool{}, created: time.Now()}
	if _, err := rand.Read(sess.ID[:]); err != nil {
		return nil, err
	}
	sess.seed = req.Seed
	if !req.HasSeed {
		var b [8]byte
		rand.Read(b[:])
		sess.seed = uint64(b[0]) | uint64(b[1])<<8 | uint64(b[2])<<16 | uint64(b[3])<<24 | uint64(b[4])<<32 |
			uint64(b[5])<<40 | uint64(b[6])<<48 | uint64(b[7]&0x7f)<<56
	}
	sess.numerics = s.allowedNumerics(req.Numerics)
	w, err := s.spawnWorker(context.Background(), sess)
	// A detached session belongs to a client that went away; its worker still
	// holds GPU memory until the session TTL. Reclaim those before refusing.
	for err != nil && isOOM(err) && slot.GPU != nil {
		victim := s.oldestDetached(slot)
		if victim == nil {
			break
		}
		s.logf("GPU full while starting a session for %q: ending detached session %s of %q", c.identity.Name, victim.hex()[:8], victim.Identity.Name)
		s.audit.Write(audit.Record{Identity: victim.Identity.Name, ID: victim.Identity.ID, Session: victim.hex(), Event: "session_evicted",
			Fields: map[string]any{"reason": "gpu_full", "for": c.identity.Name}})
		victim.close("ended while detached: the GPU was needed for a new session")
		w, err = s.spawnWorker(context.Background(), sess)
	}
	if err != nil {
		s.logf("session for %q: %v", c.identity.Name, err)
		if isOOM(err) {
			return nil, statusf(pb.ErrorCode_ERROR_OUT_OF_MEMORY, "the GPU has no free memory for a new session (other sessions or the Owner are using it): %v", firstLine(err.Error()))
		}
		return nil, statusf(pb.ErrorCode_ERROR_INTERNAL, "could not start a worker: %v", firstLine(err.Error()))
	}
	sess.worker = w
	sess.conn = c
	s.addSession(sess)
	go sess.readWorker()
	go func() {
		<-w.Done()
		sess.lost(w.ExitReason())
	}()
	s.usage.SessionOpen(sess.Identity.Name)
	s.audit.Write(audit.Record{Identity: sess.Identity.Name, ID: sess.Identity.ID, Session: sess.hex(), Event: "session_open",
		Fields: map[string]any{"device": slot.Name, "sandbox": w.Ready.GetSandbox(), "landlock_abi": w.Report.LandlockABI}})
	return sess, nil
}

func (s *Session) hex() string { return hex.EncodeToString(s.ID[:]) }

func isOOM(err error) bool {
	return strings.Contains(strings.ToLower(err.Error()), "out of memory")
}

// freeableOOM reports whether e is a GPU allocation that failed while the
// worker ran, the only kind of OOM that ending a detached session can help.
// An OOM from validation (STAGE_V*, e.g. a graph larger than the whole
// device) allocated nothing: evicting for it would let any Guest end other
// Guests' detached sessions at no cost. Unknown stages do not evict.
func freeableOOM(e *pb.Error) bool {
	if e.GetCode() != pb.ErrorCode_ERROR_OUT_OF_MEMORY {
		return false
	}
	switch e.GetStage() {
	case pb.ValidationStage_STAGE_NONE, // torch.OutOfMemoryError outside a graph (uploads)
		pb.ValidationStage_STAGE_E1_BINDINGS,
		pb.ValidationStage_STAGE_E2_RUNTIME,
		pb.ValidationStage_STAGE_E3_EXECUTE:
		return true
	}
	return false
}

// evictionVictim returns the detached session to end so that the request
// that failed with e may succeed on retry, or nil.
func (s *Session) evictionVictim(e *pb.Error) *Session {
	if s.Device.GPU == nil || !freeableOOM(e) {
		return nil
	}
	if victim := s.srv.oldestDetached(s.Device); victim != s {
		return victim
	}
	return nil
}

// oldestDetached returns the session on slot that has been detached longest.
func (s *Server) oldestDetached(slot *DeviceSlot) *Session {
	var best *Session
	var since time.Time
	for _, sess := range s.allSessions() {
		if sess.Device != slot {
			continue
		}
		sess.mu.Lock()
		d, ok := sess.detached, sess.conn == nil && !sess.closed && !sess.detached.IsZero()
		sess.mu.Unlock()
		if ok && (best == nil || d.Before(since)) {
			best, since = sess, d
		}
	}
	return best
}

func (s *Server) resume(c *clientConn, id []byte) (*Session, error) {
	if len(id) != 16 {
		return nil, errors.New("bad session id")
	}
	s.mu.Lock()
	sess := s.sessions[hex.EncodeToString(id)]
	s.mu.Unlock()
	if sess == nil || sess.Identity.ID != c.identity.ID {
		return nil, errors.New("no such session (it may have expired)")
	}
	sess.mu.Lock()
	defer sess.mu.Unlock()
	if sess.closed {
		return nil, errors.New("session ended: " + sess.lostWhy)
	}
	if sess.conn != nil {
		return nil, errors.New("session is attached to another connection")
	}
	sess.conn = c
	sess.detached = time.Time{}
	return sess, nil
}

func (s *Session) detach(c *clientConn) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.conn == c {
		s.conn = nil
		s.detached = time.Now()
	}
}

// forward sends a client request to the worker with a session-internal id.
func (s *Session) forward(c *clientConn, env *pb.Envelope) error {
	s.mu.Lock()
	if s.closed {
		why := s.lostWhy
		s.mu.Unlock()
		c.sendErr(env.RequestId, pb.ErrorCode_ERROR_SESSION_LOST, "session lost: "+why)
		return nil
	}
	if s.srv.paused.Load() {
		if _, isUpload := env.Body.(*pb.Envelope_UploadTensor); !isUpload {
			s.held = append(s.held, env)
			s.heldConn = append(s.heldConn, c)
			s.mu.Unlock()
			return nil
		}
	}
	if len(s.held) > 0 {
		// Keep order behind held requests.
		s.held = append(s.held, env)
		s.heldConn = append(s.heldConn, c)
		s.mu.Unlock()
		return nil
	}
	s.mu.Unlock()
	return s.dispatch(c, env)
}

func (s *Session) dispatch(c *clientConn, env *pb.Envelope) error {
	switch b := env.Body.(type) {
	case *pb.Envelope_Execute:
		if err := s.ensureDatasets(b.Execute); err != nil {
			var se *statusError
			if errors.As(err, &se) {
				c.sendErr(env.RequestId, se.code, se.msg)
				return nil
			}
			c.sendErr(env.RequestId, pb.ErrorCode_ERROR_DATASET, err.Error())
			return nil
		}
	case *pb.Envelope_Snapshot:
		return s.snapshot(c, env, b.Snapshot)
	case *pb.Envelope_Restore:
		return s.restore(c, env, b.Restore)
	}
	s.mu.Lock()
	key := [2]uint64{c.gen, env.RequestId}
	id, streaming := s.streams[key]
	if !streaming {
		s.nextID++
		id = s.nextID
		r := &route{gen: c.gen, rid: env.RequestId, start: time.Now()}
		switch b := env.Body.(type) {
		case *pb.Envelope_RegisterGraph:
			r.kind, r.nodes = "register", len(b.RegisterGraph.GetGraph().GetNodes())
		case *pb.Envelope_Execute:
			r.kind, r.hash = "execute", b.Execute.GraphHash
		case *pb.Envelope_UploadTensor:
			r.kind = "upload"
			if !b.UploadTensor.Final {
				s.streams[key] = id
			}
		default:
			r.kind = "other"
		}
		s.routes[id] = r
	} else if u := env.GetUploadTensor(); u != nil && u.Final {
		delete(s.streams, key)
	}
	s.mu.Unlock()
	if rg := env.GetRegisterGraph(); rg != nil {
		s.auditRegister(rg.Graph)
	}
	fwd := &pb.Envelope{RequestId: id, End: env.End, Body: env.Body}
	if err := s.worker.Send(&pb.WorkerEnvelope{Body: &pb.WorkerEnvelope_Client{Client: fwd}}); err != nil {
		c.sendErr(env.RequestId, pb.ErrorCode_ERROR_SESSION_LOST, "worker is gone")
	}
	return nil
}

func (s *Session) auditRegister(g *pb.Graph) {
	hist := map[uint32]int{}
	for _, n := range g.GetNodes() {
		hist[n.OpId]++
	}
	type kv struct {
		Op    uint32 `json:"op"`
		Count int    `json:"n"`
	}
	var top []kv
	for k, v := range hist {
		top = append(top, kv{k, v})
	}
	sort.Slice(top, func(i, j int) bool { return top[i].Count > top[j].Count })
	if len(top) > 20 {
		top = top[:20]
	}
	s.srv.audit.Write(audit.Record{Identity: s.Identity.Name, ID: s.Identity.ID, Session: s.hex(), Event: "register",
		Fields: map[string]any{"nodes": len(g.GetNodes()), "inputs": len(g.GetInputs()), "ops": top}})
}

// readWorker routes worker responses back to the attached connection.
func (s *Session) readWorker() {
	for {
		env, err := s.worker.Recv()
		if err != nil {
			return
		}
		switch b := env.Body.(type) {
		case *pb.WorkerEnvelope_Client:
			s.deliver(b.Client)
		case *pb.WorkerEnvelope_Stats:
			s.mu.Lock()
			s.lastStats = b.Stats
			waiters := s.statsWait
			s.statsWait = nil
			s.mu.Unlock()
			for _, ch := range waiters {
				ch <- b.Stats
			}
		case *pb.WorkerEnvelope_Fatal:
			s.srv.logf("session %s worker fatal: %s", s.hex()[:8], b.Fatal.Message)
			s.srv.audit.Write(audit.Record{Identity: s.Identity.Name, ID: s.Identity.ID, Session: s.hex(),
				Event: "worker_fatal", Fields: map[string]any{"message": b.Fatal.Message, "bug": true}})
		}
	}
}

func (s *Session) deliver(env *pb.Envelope) {
	s.mu.Lock()
	r := s.routes[env.RequestId]
	if r == nil {
		s.mu.Unlock()
		return
	}
	if env.End {
		delete(s.routes, env.RequestId)
	}
	conn := s.conn
	if r.kind == "execute" && env.End {
		s.executes++
		if er := env.GetExecuteResult(); er != nil {
			s.flops += er.Flops
		}
	}
	s.mu.Unlock()
	if r.onDone != nil && env.End {
		r.onDone(env)
		return
	}
	if e := env.GetError(); e != nil {
		if victim := s.evictionVictim(e); victim != nil {
			s.srv.logf("GPU out of memory in session %s of %q: ending detached session %s of %q", s.hex()[:8], s.Identity.Name, victim.hex()[:8], victim.Identity.Name)
			s.srv.audit.Write(audit.Record{Identity: victim.Identity.Name, ID: victim.Identity.ID, Session: victim.hex(), Event: "session_evicted",
				Fields: map[string]any{"reason": "gpu_full", "for": s.Identity.Name}})
			victim.close("ended while detached: the GPU was needed by another session")
			e.Message += " (a disconnected session was holding GPU memory and has been ended; retrying may succeed)"
		}
		s.srv.audit.Write(audit.Record{Identity: s.Identity.Name, ID: s.Identity.ID, Session: s.hex(), Event: "request_failed",
			Fields: map[string]any{"kind": r.kind, "code": e.Code.String(), "stage": e.Stage.String(), "message": trunc(e.Message, 300)}})
	} else if er := env.GetExecuteResult(); er != nil && env.End {
		s.srv.usage.Execute(s.Identity.Name, s.Device.Name, er.DurationUs, er.Flops, er.PeakMemoryBytes)
		s.srv.audit.Write(audit.Record{Identity: s.Identity.Name, ID: s.Identity.ID, Session: s.hex(), Event: "execute",
			Fields: map[string]any{"us": er.DurationUs, "flops": er.Flops, "peak": er.PeakMemoryBytes}})
	}
	if conn == nil || conn.gen != r.gen {
		return // detached: the response is dropped
	}
	out := &pb.Envelope{RequestId: r.rid, End: env.End, Body: env.Body}
	if err := conn.send(out); err != nil {
		conn.close()
	}
}

func trunc(s string, n int) string {
	if len(s) > n {
		return s[:n]
	}
	return s
}

// release forwards requests held during a pause, in order.
func (s *Session) release() {
	s.mu.Lock()
	held, conns := s.held, s.heldConn
	s.held, s.heldConn = nil, nil
	s.mu.Unlock()
	for i, env := range held {
		_ = s.dispatch(conns[i], env)
	}
}

func (s *Session) lost(why string) {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return
	}
	s.closed = true
	s.lostWhy = why
	conn := s.conn
	routes := s.routes
	s.routes = map[uint64]*route{}
	s.mu.Unlock()
	s.srv.removeSession(s)
	if conn != nil {
		for _, r := range routes {
			if r.gen == conn.gen {
				conn.sendErr(r.rid, pb.ErrorCode_ERROR_SESSION_LOST, "session lost: "+why)
			}
		}
	}
	os.RemoveAll(s.srv.layout.SessionScratch(s.hex()))
	s.srv.usage.SessionEnd(s.Identity.Name, time.Since(s.created))
	s.srv.audit.Write(audit.Record{Identity: s.Identity.Name, ID: s.Identity.ID, Session: s.hex(), Event: "session_end",
		Fields: map[string]any{"reason": why, "executes": s.executes, "flops": s.flops}})
}

func (s *Session) close(why string) {
	s.worker.Kill(why)
	<-s.worker.Done()
	s.lost(why)
}

// stats asks the worker for live statistics.
func (s *Session) stats(timeout time.Duration) *pb.WorkerStats {
	ch := make(chan *pb.WorkerStats, 1)
	s.mu.Lock()
	s.statsWait = append(s.statsWait, ch)
	s.mu.Unlock()
	if err := s.worker.Send(&pb.WorkerEnvelope{Body: &pb.WorkerEnvelope_StatsRequest{StatsRequest: &pb.StatsRequest{}}}); err != nil {
		return nil
	}
	select {
	case st := <-ch:
		return st
	case <-time.After(timeout):
		s.mu.Lock()
		defer s.mu.Unlock()
		return s.lastStats
	}
}

// ---------------------------------------------------------------- datasets

func (s *Session) ensureDatasets(ex *pb.Execute) error {
	for _, b := range ex.Bindings {
		ds := b.GetDataset()
		if ds == nil {
			continue
		}
		key := hex.EncodeToString(ds.DatasetId)
		s.mu.Lock()
		open := s.datasets[key]
		s.mu.Unlock()
		if open {
			continue
		}
		rec, err := s.srv.store.Get(s.Identity.ID, ds.DatasetId)
		if err != nil {
			return statusf(pb.ErrorCode_ERROR_UNKNOWN_DATASET, "unknown dataset")
		}
		files, err := s.srv.store.OpenShards(s.Identity.ID, rec)
		if err != nil {
			return err
		}
		msg := &pb.DatasetOpen{DatasetId: ds.DatasetId}
		for _, t := range rec.Manifest.Tensors {
			msg.Tensors = append(msg.Tensors, &pb.DatasetTensor{Name: t.Name,
				Meta: &pb.TensorMeta{Dtype: store.ManifestDTypeToPB[t.DType], Shape: t.Shape}})
		}
		for i, sh := range rec.Manifest.Shards {
			msg.Shards = append(msg.Shards, &pb.DatasetShard{Index: uint32(i), NumSamples: sh.NumSamples})
		}
		fds := make([]int, len(files))
		for i, f := range files {
			fds[i] = int(f.Fd())
		}
		err = s.worker.SendFDs(ds.DatasetId, fds)
		for _, f := range files {
			f.Close()
		}
		if err != nil {
			return err
		}
		if err := s.worker.Send(&pb.WorkerEnvelope{Body: &pb.WorkerEnvelope_DatasetOpen{DatasetOpen: msg}}); err != nil {
			return err
		}
		s.mu.Lock()
		s.datasets[key] = true
		s.mu.Unlock()
	}
	return nil
}

// ---------------------------------------------------------------- snapshots

func (s *Session) snapshotDir() string { return s.srv.layout.IdentitySnapshots(s.Identity.ID) }

func (s *Session) snapshot(c *clientConn, env *pb.Envelope, req *pb.Snapshot) error {
	if !store.NameRe.MatchString(req.Name) {
		c.sendErr(env.RequestId, pb.ErrorCode_ERROR_PROTOCOL, "snapshot name must match [a-z0-9][a-z0-9_-]{0,63}")
		return nil
	}
	if len(req.Handles) != len(req.Keys) || len(req.Handles) == 0 {
		c.sendErr(env.RequestId, pb.ErrorCode_ERROR_PROTOCOL, "handles and keys must be non-empty and equal in length")
		return nil
	}
	if rails.DiskBudget(s.srv.cfg().Rails, s.srv.layout.State) == 0 {
		c.sendErr(env.RequestId, pb.ErrorCode_ERROR_SAFETY_RAIL, "the disk safety rail leaves no room for snapshots")
		return nil
	}
	dir := s.snapshotDir()
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}
	final := filepath.Join(dir, req.Name+".safetensors")
	tmp := filepath.Join(dir, "."+req.Name+".tmp-"+s.hex()[:8])
	f, err := os.OpenFile(tmp, os.O_CREATE|os.O_EXCL|os.O_WRONLY|unix.O_CLOEXEC, 0o600)
	if err != nil {
		c.sendErr(env.RequestId, pb.ErrorCode_ERROR_INTERNAL, "could not create the snapshot file")
		return nil
	}
	id := s.newRoute(c, env.RequestId, "snapshot", func(resp *pb.Envelope) {
		ok := resp.GetSnapshotSaved() != nil
		if ok {
			if st, err := os.Stat(tmp); err == nil {
				if err := rails.CheckDiskWrite(s.srv.cfg().Rails, s.srv.layout.State, 0); err != nil {
					ok = false
					resp = &pb.Envelope{End: true, Body: &pb.Envelope_Error{Error: &pb.Error{
						Code: pb.ErrorCode_ERROR_SAFETY_RAIL, Message: err.Error()}}}
				} else {
					resp.GetSnapshotSaved().Name = req.Name
					resp.GetSnapshotSaved().Bytes = uint64(st.Size())
				}
			}
		}
		if ok {
			os.Rename(tmp, final)
		} else {
			os.Remove(tmp)
		}
		s.reply(c, env.RequestId, resp)
	})
	err = s.worker.SendFDs([]byte("snapshot"), []int{int(f.Fd())})
	f.Close()
	if err != nil {
		os.Remove(tmp)
		return err
	}
	return s.worker.Send(&pb.WorkerEnvelope{Body: &pb.WorkerEnvelope_SnapshotWrite{SnapshotWrite: &pb.SnapshotWrite{
		RequestId: id, Handles: req.Handles, Keys: req.Keys}}})
}

func (s *Session) restore(c *clientConn, env *pb.Envelope, req *pb.Restore) error {
	if !store.NameRe.MatchString(req.Name) || len(req.Keys) != len(req.Handles) {
		c.sendErr(env.RequestId, pb.ErrorCode_ERROR_PROTOCOL, "invalid restore request")
		return nil
	}
	f, err := os.Open(filepath.Join(s.snapshotDir(), req.Name+".safetensors"))
	if err != nil {
		c.sendErr(env.RequestId, pb.ErrorCode_ERROR_PROTOCOL, "no such snapshot")
		return nil
	}
	id := s.newRoute(c, env.RequestId, "restore", func(resp *pb.Envelope) { s.reply(c, env.RequestId, resp) })
	err = s.worker.SendFDs([]byte("restore"), []int{int(f.Fd())})
	f.Close()
	if err != nil {
		return err
	}
	return s.worker.Send(&pb.WorkerEnvelope{Body: &pb.WorkerEnvelope_SnapshotRead{SnapshotRead: &pb.SnapshotRead{
		RequestId: id, Keys: req.Keys, Handles: req.Handles}}})
}

func (s *Session) newRoute(c *clientConn, rid uint64, kind string, done func(*pb.Envelope)) uint64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.nextID++
	s.routes[s.nextID] = &route{gen: c.gen, rid: rid, kind: kind, start: time.Now(), onDone: done}
	return s.nextID
}

func (s *Session) reply(c *clientConn, rid uint64, env *pb.Envelope) {
	out := &pb.Envelope{RequestId: rid, End: true, Body: env.Body}
	_ = c.send(out)
}

func (c *clientConn) snapshotAdmin(env *pb.Envelope) error {
	dir := c.srv.layout.IdentitySnapshots(c.identity.ID)
	switch b := env.Body.(type) {
	case *pb.Envelope_ListSnapshots:
		ents, _ := os.ReadDir(dir)
		out := &pb.SnapshotList{}
		for _, e := range ents {
			n := e.Name()
			if filepath.Ext(n) != ".safetensors" || n[0] == '.' {
				continue
			}
			fi, err := e.Info()
			if err != nil {
				continue
			}
			out.Snapshots = append(out.Snapshots, &pb.SnapshotInfo{Name: n[:len(n)-len(".safetensors")],
				Bytes: uint64(fi.Size()), CreatedUnix: fi.ModTime().Unix()})
		}
		return c.send(&pb.Envelope{RequestId: env.RequestId, End: true, Body: &pb.Envelope_SnapshotList{SnapshotList: out}})
	case *pb.Envelope_DeleteSnapshot:
		if !store.NameRe.MatchString(b.DeleteSnapshot.Name) {
			c.sendErr(env.RequestId, pb.ErrorCode_ERROR_PROTOCOL, "invalid snapshot name")
			return nil
		}
		if err := os.Remove(filepath.Join(dir, b.DeleteSnapshot.Name+".safetensors")); err != nil {
			c.sendErr(env.RequestId, pb.ErrorCode_ERROR_PROTOCOL, "no such snapshot")
			return nil
		}
		return c.send(&pb.Envelope{RequestId: env.RequestId, End: true, Body: &pb.Envelope_Freed{Freed: &pb.Freed{}}})
	}
	return nil
}

func (c *clientConn) dataRequest(env *pb.Envelope) error {
	st := c.srv.store
	id := c.identity.ID
	rid := env.RequestId
	fail := func(err error) error {
		var se *store.Error
		if errors.As(err, &se) {
			c.sendErr(rid, se.Code, se.Msg)
		} else {
			c.sendErr(rid, pb.ErrorCode_ERROR_INTERNAL, err.Error())
		}
		c.srv.audit.Write(audit.Record{Identity: c.identity.Name, ID: id, Event: "data_failed", Fields: map[string]any{"error": err.Error()}})
		return nil
	}
	switch b := env.Body.(type) {
	case *pb.Envelope_PackBegin:
		res, err := st.Begin(id, b.PackBegin, int(c.srv.cfg().Limits.MaxManifestBytes))
		if err != nil {
			return fail(err)
		}
		return c.send(&pb.Envelope{RequestId: rid, End: true, Body: &pb.Envelope_PackStatus{PackStatus: res}})
	case *pb.Envelope_PackChunk:
		if q := c.srv.quota(c.identity.Name); q.UploadBytesPerSec > 0 {
			c.srv.throttle(id, uint64(len(b.PackChunk.Data)), uint64(q.UploadBytesPerSec))
		}
		if err := st.PutChunk(id, b.PackChunk); err != nil {
			return fail(err)
		}
		return c.send(&pb.Envelope{RequestId: rid, End: true, Body: &pb.Envelope_Freed{Freed: &pb.Freed{}}})
	case *pb.Envelope_PackCommit:
		info, err := st.Commit(id, b.PackCommit.PackId)
		if err != nil {
			return fail(err)
		}
		c.srv.audit.Write(audit.Record{Identity: c.identity.Name, ID: id, Event: "dataset_added",
			Fields: map[string]any{"name": info.Name, "bytes": info.Bytes, "samples": info.NumSamples}})
		return c.send(&pb.Envelope{RequestId: rid, End: true, Body: &pb.Envelope_DatasetInfo{DatasetInfo: info}})
	case *pb.Envelope_ListDatasets:
		out := &pb.DatasetList{}
		for _, r := range st.List(id) {
			out.Datasets = append(out.Datasets, r.Info())
		}
		return c.send(&pb.Envelope{RequestId: rid, End: true, Body: &pb.Envelope_DatasetList{DatasetList: out}})
	case *pb.Envelope_DeleteDataset:
		if err := st.Delete(id, b.DeleteDataset.DatasetId); err != nil {
			return fail(err)
		}
		return c.send(&pb.Envelope{RequestId: rid, End: true, Body: &pb.Envelope_DatasetDeleted{DatasetDeleted: &pb.DatasetDeleted{}}})
	}
	return fmt.Errorf("unexpected data request")
}
