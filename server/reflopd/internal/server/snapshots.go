// SPDX-License-Identifier: Apache-2.0

package server

import (
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"golang.org/x/sys/unix"

	pb "github.com/reflopai/reflop/pkg/reflopv1"
	"github.com/reflopai/reflop/server/reflopd/internal/audit"
	"github.com/reflopai/reflop/server/reflopd/internal/config"
	"github.com/reflopai/reflop/server/reflopd/internal/rails"
	"github.com/reflopai/reflop/server/reflopd/internal/store"
)

// Snapshots and checkpoints (ADR-028, ADR-029).
//
// A write has two phases. reflopd creates a temporary file in the
// identity's directory and passes its descriptor to the worker. The worker
// takes the cut and reports the exact file size (SnapshotCut); reflopd
// reserves that much against the disk rail and the storage quota and
// answers SnapshotGo. The worker writes in the background. When it reports
// success, reflopd syncs the file, checks its size, renames it into place
// and syncs the directory; only then is the client told it is saved. A
// failure at any step removes the temporary file and keeps the previous one.

const (
	// reservedPrefix marks keys only checkpoints write (worker: RESERVED).
	reservedPrefix = "__reflop__/"
	metaKey        = reservedPrefix + "meta"
	maxKeyBytes    = 256
	maxTagBytes    = 4096
	maxMetaBytes   = 16 << 10
	// checkpointKeep is how long a checkpoint whose session has ended stays
	// on disk for RecoverSession.
	checkpointKeep = 24 * time.Hour
	checkpointsDir = ".checkpoints"
)

// pendingWrite is the one snapshot or checkpoint a session is writing.
type pendingWrite struct {
	rid        uint64 // session-internal request id
	tmp, final string
	checkpoint bool
	generation uint64
	replaced   uint64 // size of the file it replaces
	reserved   uint64
	expect     uint64
}

func (s *Session) snapshotDir() string { return s.srv.layout.IdentitySnapshots(s.Identity.ID) }

func (s *Session) checkpointDir() string { return filepath.Join(s.snapshotDir(), checkpointsDir) }

func checkpointPath(dir string, sessionID []byte) string {
	return filepath.Join(dir, hex.EncodeToString(sessionID)+".safetensors")
}

// keyError says why k may not name a tensor in a snapshot (the worker's
// check_key, so both refuse the same keys).
func keyError(k string) string {
	switch {
	case k == "" || len(k) > maxKeyBytes:
		return "tensor names must be 1 to 256 bytes"
	case k == "__metadata__":
		return "'__metadata__' is reserved by the safetensors format"
	case strings.HasPrefix(k, reservedPrefix):
		return "names starting with " + reservedPrefix + " are reserved"
	}
	return ""
}

func fileSize(p string) uint64 {
	if st, err := os.Stat(p); err == nil {
		return uint64(st.Size())
	}
	return 0
}

// ---------------------------------------------------------------- writes

func (s *Session) snapshot(c *clientConn, env *pb.Envelope, req *pb.Snapshot) error {
	if !store.NameRe.MatchString(req.Name) {
		c.sendErr(env.RequestId, pb.ErrorCode_ERROR_PROTOCOL, "snapshot name must match [a-z0-9][a-z0-9_-]{0,63}")
		return nil
	}
	if len(req.Handles) != len(req.Keys) || len(req.Handles) == 0 {
		c.sendErr(env.RequestId, pb.ErrorCode_ERROR_PROTOCOL, "handles and keys must be non-empty and equal in length")
		return nil
	}
	for _, k := range req.Keys {
		if why := keyError(k); why != "" {
			c.sendErr(env.RequestId, pb.ErrorCode_ERROR_PROTOCOL, fmt.Sprintf("snapshot key %q: %s", trunc(k, 64), why))
			return nil
		}
	}
	dir := s.snapshotDir()
	pw := &pendingWrite{final: filepath.Join(dir, req.Name+".safetensors")}
	return s.startWrite(c, env, dir, pw, &pb.SnapshotWrite{Handles: req.Handles, Keys: req.Keys}, req.Name)
}

func (s *Session) checkpoint(c *clientConn, env *pb.Envelope, req *pb.Checkpoint) error {
	cfg := s.srv.cfg().Checkpoints
	if !cfg.Enabled {
		c.sendErr(env.RequestId, pb.ErrorCode_ERROR_PERMISSION_DENIED, "the Owner has not enabled checkpoints on this server")
		return nil
	}
	if len(req.Tag) > maxTagBytes {
		c.sendErr(env.RequestId, pb.ErrorCode_ERROR_RESOURCE_LIMIT, fmt.Sprintf("checkpoint tags are at most %d bytes", maxTagBytes))
		return nil
	}
	iv := time.Duration(cfg.Interval)
	s.mu.Lock()
	last, gen := s.lastCheckpoint, s.checkpointGen+1
	s.mu.Unlock()
	if !last.IsZero() && time.Since(last) < iv {
		c.sendErr(env.RequestId, pb.ErrorCode_ERROR_RESOURCE_LIMIT, fmt.Sprintf(
			"the Owner allows one checkpoint every %s; the last one was %s ago", iv, time.Since(last).Truncate(time.Second)))
		return nil
	}
	dir := s.checkpointDir()
	pw := &pendingWrite{final: checkpointPath(dir, s.ID[:]), checkpoint: true, generation: gen}
	return s.startWrite(c, env, dir, pw, &pb.SnapshotWrite{Checkpoint: true, Tag: req.Tag, Generation: gen}, "")
}

func (s *Session) startWrite(c *clientConn, env *pb.Envelope, dir string, pw *pendingWrite, msg *pb.SnapshotWrite, name string) error {
	if rails.DiskBudget(s.srv.cfg().Rails, s.srv.layout.State) == 0 {
		c.sendErr(env.RequestId, pb.ErrorCode_ERROR_SAFETY_RAIL, "the disk safety rail leaves no room for snapshots")
		return nil
	}
	if err := os.MkdirAll(dir, 0o700); err != nil {
		c.sendErr(env.RequestId, pb.ErrorCode_ERROR_INTERNAL, "could not create the snapshot directory")
		return nil
	}
	s.mu.Lock()
	if s.writing != nil {
		s.mu.Unlock()
		c.sendErr(env.RequestId, pb.ErrorCode_ERROR_BUSY, "a snapshot or checkpoint of this session is still being written")
		return nil
	}
	if s.final {
		s.mu.Unlock()
		c.sendErr(env.RequestId, pb.ErrorCode_ERROR_SESSION_LOST, "session ended")
		return nil
	}
	s.writing = pw
	s.mu.Unlock()
	pw.tmp = filepath.Join(dir, "."+filepath.Base(pw.final)+".tmp-"+s.hex()[:8])
	pw.replaced = fileSize(pw.final)
	f, err := os.OpenFile(pw.tmp, os.O_CREATE|os.O_EXCL|os.O_WRONLY|unix.O_CLOEXEC, 0o600)
	if err != nil {
		s.clearWrite(pw)
		c.sendErr(env.RequestId, pb.ErrorCode_ERROR_INTERNAL, "could not create the snapshot file")
		return nil
	}
	pw.rid = s.newRoute(c, env.RequestId, map[bool]string{true: "checkpoint", false: "snapshot"}[pw.checkpoint],
		func(resp *pb.Envelope) { s.finishWrite(c, env.RequestId, pw, name, resp) })
	msg.RequestId = pw.rid
	err = s.worker.SendFDs([]byte("snapshot"), []int{int(f.Fd())})
	f.Close()
	if err == nil {
		err = s.worker.Send(&pb.WorkerEnvelope{Body: &pb.WorkerEnvelope_SnapshotWrite{SnapshotWrite: msg}})
	}
	if err != nil {
		s.dropRoute(pw.rid)
		s.clearWrite(pw)
		os.Remove(pw.tmp)
		c.sendErr(env.RequestId, pb.ErrorCode_ERROR_SESSION_LOST, "worker is gone")
	}
	return nil
}

// onCut answers the worker's SnapshotCut: reserve the bytes or refuse.
func (s *Session) onCut(cut *pb.SnapshotCut) {
	s.mu.Lock()
	pw := s.writing
	s.mu.Unlock()
	if pw == nil || pw.rid != cut.RequestId {
		return
	}
	g := &pb.SnapshotGo{RequestId: cut.RequestId, Proceed: true}
	if code, err := s.srv.reserveWrite(s.Identity.ID, cut.Bytes, pw.replaced); err != nil {
		g.Proceed, g.Code, g.Reason = false, code, err.Error()
	} else {
		s.mu.Lock()
		pw.reserved, pw.expect = cut.Bytes, cut.Bytes
		s.mu.Unlock()
	}
	_ = s.worker.Send(&pb.WorkerEnvelope{Body: &pb.WorkerEnvelope_SnapshotGo{SnapshotGo: g}})
}

func (s *Session) clearWrite(pw *pendingWrite) {
	s.mu.Lock()
	if s.writing == pw {
		s.writing = nil
	}
	n := pw.reserved
	pw.reserved = 0
	s.mu.Unlock()
	s.srv.releaseWrite(s.Identity.ID, n)
}

func (s *Session) finishWrite(c *clientConn, rid uint64, pw *pendingWrite, name string, resp *pb.Envelope) {
	defer s.clearWrite(pw)
	what := map[bool]string{true: "checkpoint", false: "snapshot"}[pw.checkpoint]
	ok := false
	switch {
	case resp.GetSnapshotSaved() != nil:
		ok = true
		resp.GetSnapshotSaved().Name = name
	case resp.GetCheckpointSaved() != nil && !resp.GetCheckpointSaved().Skipped:
		ok = true
	}
	if !ok {
		os.Remove(pw.tmp)
		if cs := resp.GetCheckpointSaved(); cs != nil && cs.Skipped {
			s.srv.audit.Write(audit.Record{Identity: s.Identity.Name, ID: s.Identity.ID, Session: s.hex(), Event: "checkpoint_skipped",
				Fields: map[string]any{"reason": trunc(cs.Reason, 200)}})
		}
		s.reply(c, rid, resp)
		return
	}
	if err := s.publish(pw); err != nil {
		os.Remove(pw.tmp)
		s.srv.logf("session %s: %s not saved: %v", s.hex()[:8], what, err)
		code := pb.ErrorCode_ERROR_INTERNAL
		if errors.Is(err, errSessionFinal) {
			code = pb.ErrorCode_ERROR_SESSION_LOST
		}
		s.reply(c, rid, &pb.Envelope{End: true, Body: &pb.Envelope_Error{Error: &pb.Error{Code: code,
			Message: fmt.Sprintf("the %s was not saved: %v", what, err)}}})
		return
	}
	if pw.checkpoint {
		s.mu.Lock()
		s.lastCheckpoint, s.checkpointGen = time.Now(), pw.generation
		s.mu.Unlock()
		resp.GetCheckpointSaved().Bytes = pw.expect
	} else {
		resp.GetSnapshotSaved().Bytes = pw.expect
	}
	s.srv.audit.Write(audit.Record{Identity: s.Identity.Name, ID: s.Identity.ID, Session: s.hex(), Event: what + "_saved",
		Fields: auditWriteFields(pw, name)})
	s.reply(c, rid, resp)
}

var errSessionFinal = errors.New("the session was ended on purpose")

func auditWriteFields(pw *pendingWrite, name string) map[string]any {
	f := map[string]any{"bytes": pw.expect}
	if pw.checkpoint {
		f["generation"] = pw.generation
	} else {
		f["name"] = name
	}
	return f
}

// publish makes a finished temporary file the snapshot: sync, size check,
// rename, directory sync. The previous file stays until the rename.
func (s *Session) publish(pw *pendingWrite) error {
	f, err := os.Open(pw.tmp)
	if err != nil {
		return err
	}
	err = f.Sync()
	st, serr := f.Stat()
	f.Close()
	if err != nil {
		return fmt.Errorf("sync: %w", err)
	}
	if serr != nil {
		return serr
	}
	if uint64(st.Size()) != pw.expect || pw.expect == 0 {
		return fmt.Errorf("the file has %d bytes, the worker announced %d", st.Size(), pw.expect)
	}
	// Under the lock, so that a session ended on purpose (which deletes its
	// checkpoint) never gets one back from a late rename.
	s.mu.Lock()
	if s.final && pw.checkpoint {
		s.mu.Unlock()
		return errSessionFinal
	}
	err = os.Rename(pw.tmp, pw.final)
	s.mu.Unlock()
	if err != nil {
		return fmt.Errorf("rename: %w", err)
	}
	return syncDir(filepath.Dir(pw.final))
}

func syncDir(dir string) error {
	d, err := os.Open(dir)
	if err != nil {
		return err
	}
	defer d.Close()
	if err := d.Sync(); err != nil {
		return fmt.Errorf("sync directory: %w", err)
	}
	return nil
}

// reserveWrite accounts n bytes of a write in progress against the disk rail
// and the identity's storage quota (datasets, snapshots and checkpoints, plus
// every other write in progress). replaced is the file the write replaces.
func (s *Server) reserveWrite(identity string, n, replaced uint64) (pb.ErrorCode, error) {
	return s.reserveWriteQuota(identity, uint64(s.quotaByID(identity).StorageBytes), n, replaced)
}

func (s *Server) reserveWriteQuota(identity string, q, n, replaced uint64) (pb.ErrorCode, error) {
	s.diskMu.Lock()
	defer s.diskMu.Unlock()
	if err := rails.CheckDiskWrite(s.cfg().Rails, s.layout.State, n+s.diskReserved); err != nil {
		return pb.ErrorCode_ERROR_SAFETY_RAIL, err
	}
	if q > 0 {
		used := s.store.Usage(identity) + dirUsage(s.layout.IdentitySnapshots(identity)) + s.idReserved[identity]
		if used >= replaced {
			used -= replaced
		}
		if used+n > q {
			return pb.ErrorCode_ERROR_QUOTA, fmt.Errorf("storage quota of %s would be exceeded (%s in use, %s more needed)",
				byteStr(q), byteStr(used), byteStr(n))
		}
	}
	s.diskReserved += n
	s.idReserved[identity] += n
	return 0, nil
}

func (s *Server) releaseWrite(identity string, n uint64) {
	if n == 0 {
		return
	}
	s.diskMu.Lock()
	defer s.diskMu.Unlock()
	s.diskReserved -= min(n, s.diskReserved)
	s.idReserved[identity] -= min(n, s.idReserved[identity])
	if s.idReserved[identity] == 0 {
		delete(s.idReserved, identity)
	}
}

func dirUsage(dir string) uint64 {
	var n uint64
	_ = filepath.WalkDir(dir, func(_ string, d fs.DirEntry, err error) error {
		if err == nil && d.Type().IsRegular() {
			if fi, err := d.Info(); err == nil {
				n += uint64(fi.Size())
			}
		}
		return nil
	})
	return n
}

func byteStr(n uint64) string { return config.Bytes(n).String() }

// ---------------------------------------------------------------- reads

func (s *Session) restore(c *clientConn, env *pb.Envelope, req *pb.Restore) error {
	if !store.NameRe.MatchString(req.Name) || len(req.Keys) != len(req.Handles) || len(req.Keys) == 0 {
		c.sendErr(env.RequestId, pb.ErrorCode_ERROR_PROTOCOL, "invalid restore request")
		return nil
	}
	for _, k := range req.Keys {
		if why := keyError(k); why != "" {
			c.sendErr(env.RequestId, pb.ErrorCode_ERROR_PROTOCOL, fmt.Sprintf("snapshot key %q: %s", trunc(k, 64), why))
			return nil
		}
	}
	f, err := os.Open(filepath.Join(s.snapshotDir(), req.Name+".safetensors"))
	if err != nil {
		c.sendErr(env.RequestId, pb.ErrorCode_ERROR_PROTOCOL, "no such snapshot")
		return nil
	}
	id := s.newRoute(c, env.RequestId, "restore", func(resp *pb.Envelope) { s.reply(c, env.RequestId, resp) })
	err = s.worker.SendFDs([]byte("restore"), []int{int(f.Fd())})
	f.Close()
	if err == nil {
		err = s.worker.Send(&pb.WorkerEnvelope{Body: &pb.WorkerEnvelope_SnapshotRead{SnapshotRead: &pb.SnapshotRead{
			RequestId: id, Keys: req.Keys, Handles: req.Handles}}})
	}
	if err != nil {
		s.dropRoute(id)
		c.sendErr(env.RequestId, pb.ErrorCode_ERROR_SESSION_LOST, "worker is gone")
	}
	return nil
}

type checkpointMeta struct {
	Generation  uint64 `json:"generation"`
	CreatedUnix int64  `json:"created_unix"`
	Tag         string `json:"tag"`
}

// recoverSession restores another session's last checkpoint into s.
func (s *Session) recoverSession(c *clientConn, env *pb.Envelope, req *pb.RecoverSession) error {
	rid := env.RequestId
	from := req.FromSessionId
	if len(from) != 16 || hex.EncodeToString(from) == s.hex() {
		c.sendErr(rid, pb.ErrorCode_ERROR_PROTOCOL, "bad session id")
		return nil
	}
	s.srv.mu.Lock()
	old := s.srv.sessions[hex.EncodeToString(from)]
	s.srv.mu.Unlock()
	if old != nil && old.Identity.ID == s.Identity.ID {
		c.sendErr(rid, pb.ErrorCode_ERROR_PROTOCOL, "that session is still running: resume it instead")
		return nil
	}
	// The path comes from the caller's own identity: another identity's
	// checkpoints are not reachable (I7, I8).
	path := checkpointPath(s.checkpointDir(), from)
	f, err := os.Open(path)
	if err != nil {
		c.sendErr(rid, pb.ErrorCode_ERROR_UNKNOWN_SESSION, "no checkpoint of that session: none was saved, or the "+
			"session was ended on purpose (reflopd kill, revoke, unpair or CloseSession), which deletes it")
		return nil
	}
	fail := func(msg string) error {
		f.Close()
		c.sendErr(rid, pb.ErrorCode_ERROR_PROTOCOL, "checkpoint file: "+msg)
		return nil
	}
	st, err := f.Stat()
	if err != nil {
		return fail(err.Error())
	}
	hdr, err := store.ParseSnapshot(f, st.Size(), s.srv.cfg().Limits.MaxTensorNumel)
	if err != nil {
		return fail(err.Error())
	}
	m, ok := hdr[metaKey]
	if !ok || m.DType != "U8" || m.End-m.Begin > maxMetaBytes {
		return fail("missing or invalid metadata")
	}
	raw := make([]byte, m.End-m.Begin)
	hl := make([]byte, 8)
	if _, err := f.ReadAt(hl, 0); err != nil {
		return fail(err.Error())
	}
	start := int64(8) + int64(leU64(hl))
	if _, err := f.ReadAt(raw, start+m.Begin); err != nil {
		return fail(err.Error())
	}
	var meta checkpointMeta
	if err := json.Unmarshal(raw, &meta); err != nil {
		return fail("invalid metadata")
	}
	tag, err := base64.StdEncoding.DecodeString(meta.Tag)
	if err != nil || len(tag) > maxTagBytes {
		return fail("invalid tag")
	}
	type ent struct {
		key string
		h   []byte
		t   store.STTensor
	}
	var ents []ent
	for k, t := range hdr {
		if strings.HasPrefix(k, reservedPrefix) {
			continue
		}
		h, err := hex.DecodeString(k)
		if err != nil || len(h) != 16 || hex.EncodeToString(h) != k {
			return fail("a tensor name is not a handle")
		}
		ents = append(ents, ent{k, h, t})
	}
	sort.Slice(ents, func(i, j int) bool { return ents[i].key < ents[j].key })
	out := &pb.SessionRecovered{Tag: tag, Generation: meta.Generation, CreatedUnix: meta.CreatedUnix}
	read := &pb.SnapshotRead{RestoreRng: true}
	for _, e := range ents {
		read.Keys = append(read.Keys, e.key)
		read.Handles = append(read.Handles, e.h)
		out.Handles = append(out.Handles, e.h)
		out.Metas = append(out.Metas, &pb.TensorMeta{Dtype: store.SnapshotDTypeToPB[e.t.DType], Shape: e.t.Shape})
	}
	id := s.newRoute(c, rid, "recover", func(resp *pb.Envelope) {
		if resp.GetRestored() == nil {
			s.reply(c, rid, resp)
			return
		}
		// The checkpoint now belongs to this session: a second failure
		// before this session's own first checkpoint can recover again.
		s.mu.Lock()
		if !s.final {
			if err := os.Rename(path, checkpointPath(s.checkpointDir(), s.ID[:])); err == nil {
				s.checkpointGen = meta.Generation
			}
		}
		s.mu.Unlock()
		s.srv.audit.Write(audit.Record{Identity: s.Identity.Name, ID: s.Identity.ID, Session: s.hex(), Event: "session_recovered",
			Fields: map[string]any{"from": hex.EncodeToString(from), "generation": meta.Generation, "handles": len(out.Handles)}})
		s.reply(c, rid, &pb.Envelope{End: true, Body: &pb.Envelope_SessionRecovered{SessionRecovered: out}})
	})
	read.RequestId = id
	err = s.worker.SendFDs([]byte("restore"), []int{int(f.Fd())})
	f.Close()
	if err == nil {
		err = s.worker.Send(&pb.WorkerEnvelope{Body: &pb.WorkerEnvelope_SnapshotRead{SnapshotRead: read}})
	}
	if err != nil {
		s.dropRoute(id)
		c.sendErr(rid, pb.ErrorCode_ERROR_SESSION_LOST, "worker is gone")
	}
	return nil
}

func leU64(b []byte) uint64 {
	var v uint64
	for i := 7; i >= 0; i-- {
		v = v<<8 | uint64(b[i])
	}
	return v
}

// ---------------------------------------------------------------- lifecycle

// deleteCheckpoint removes the session's checkpoint: it was ended on purpose.
func (s *Session) deleteCheckpoint() {
	p := checkpointPath(s.checkpointDir(), s.ID[:])
	if err := os.Remove(p); err == nil {
		s.srv.audit.Write(audit.Record{Identity: s.Identity.Name, ID: s.Identity.ID, Session: s.hex(), Event: "checkpoint_deleted"})
	}
}

// closeFinal ends a session on purpose (Owner kill, revoke, unpair,
// CloseSession): its checkpoint is deleted and can never be published
// again, so the session cannot be revived.
func (s *Session) closeFinal(why string) {
	s.mu.Lock()
	s.final = true
	s.mu.Unlock()
	s.close(why)
	s.deleteCheckpoint()
}

// sweepCheckpoints deletes checkpoints of ended sessions after
// checkpointKeep, and temporary files left by a crash.
func (s *Server) sweepCheckpoints() {
	live := map[string]bool{}
	for _, sess := range s.allSessions() {
		live[sess.hex()+".safetensors"] = true
	}
	ids, _ := os.ReadDir(s.layout.Snapshots())
	for _, id := range ids {
		for _, dir := range []string{filepath.Join(s.layout.Snapshots(), id.Name()), filepath.Join(s.layout.Snapshots(), id.Name(), checkpointsDir)} {
			ents, _ := os.ReadDir(dir)
			for _, e := range ents {
				fi, err := e.Info()
				if err != nil || !fi.Mode().IsRegular() {
					continue
				}
				n := e.Name()
				age := time.Since(fi.ModTime())
				stale := strings.HasPrefix(n, ".") && strings.Contains(n, ".tmp-") && age > time.Hour
				if filepath.Base(dir) == checkpointsDir && !strings.HasPrefix(n, ".") && !live[n] && age > checkpointKeep {
					stale = true
				}
				if stale {
					os.Remove(filepath.Join(dir, n))
				}
			}
		}
	}
}

// ---------------------------------------------------------------- routes

func (s *Session) newRoute(c *clientConn, rid uint64, kind string, done func(*pb.Envelope)) uint64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.nextID++
	s.routes[s.nextID] = &route{gen: c.gen, rid: rid, kind: kind, start: time.Now(), onDone: done}
	return s.nextID
}

func (s *Session) dropRoute(id uint64) {
	s.mu.Lock()
	delete(s.routes, id)
	s.mu.Unlock()
}

func (s *Session) reply(c *clientConn, rid uint64, env *pb.Envelope) {
	out := &pb.Envelope{RequestId: rid, End: true, Body: env.Body}
	_ = c.send(out)
}

// ---------------------------------------------------------------- admin

func (c *clientConn) snapshotAdmin(env *pb.Envelope) error {
	dir := c.srv.layout.IdentitySnapshots(c.identity.ID)
	switch b := env.Body.(type) {
	case *pb.Envelope_ListSnapshots:
		ents, _ := os.ReadDir(dir)
		out := &pb.SnapshotList{}
		for _, e := range ents {
			n := e.Name()
			if filepath.Ext(n) != ".safetensors" || n[0] == '.' || !e.Type().IsRegular() {
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
	case *pb.Envelope_DescribeSnapshot:
		name := b.DescribeSnapshot.Name
		if !store.NameRe.MatchString(name) {
			c.sendErr(env.RequestId, pb.ErrorCode_ERROR_PROTOCOL, "invalid snapshot name")
			return nil
		}
		f, err := os.Open(filepath.Join(dir, name+".safetensors"))
		if err != nil {
			c.sendErr(env.RequestId, pb.ErrorCode_ERROR_PROTOCOL, "no such snapshot")
			return nil
		}
		defer f.Close()
		st, err := f.Stat()
		if err != nil {
			c.sendErr(env.RequestId, pb.ErrorCode_ERROR_INTERNAL, err.Error())
			return nil
		}
		hdr, err := store.ParseSnapshot(f, st.Size(), c.srv.cfg().Limits.MaxTensorNumel)
		if err != nil {
			c.sendErr(env.RequestId, pb.ErrorCode_ERROR_PROTOCOL, "snapshot file: "+err.Error())
			return nil
		}
		keys := make([]string, 0, len(hdr))
		for k := range hdr {
			if keyError(k) == "" {
				keys = append(keys, k)
			}
		}
		sort.Strings(keys)
		out := &pb.SnapshotDescription{Name: name, Bytes: uint64(st.Size()), Keys: keys}
		for _, k := range keys {
			out.Metas = append(out.Metas, &pb.TensorMeta{Dtype: store.SnapshotDTypeToPB[hdr[k].DType], Shape: hdr[k].Shape})
		}
		return c.send(&pb.Envelope{RequestId: env.RequestId, End: true, Body: &pb.Envelope_SnapshotDescription{SnapshotDescription: out}})
	}
	return nil
}
