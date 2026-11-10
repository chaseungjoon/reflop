// SPDX-License-Identifier: Apache-2.0

// Package store keeps each identity's TensorPacks content-addressed under
// a server-chosen path: <root>/<identity>/blobs/<sha256> (invariant I7).
// Client-provided names are metadata only. Uploads are chunked, verified
// per chunk, resumable, and checked against the disk safety rail
// continuously (PLAN.md §10.3).
package store

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"sync"
	"time"

	pb "github.com/chaseungjoon/reflop/pkg/reflopv1"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/config"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/rails"
)

// NameRe constrains dataset and snapshot names.
var NameRe = regexp.MustCompile(`^[a-z0-9][a-z0-9_-]{0,63}$`)

// Error carries a protocol error code.
type Error struct {
	Code pb.ErrorCode
	Msg  string
}

func (e *Error) Error() string { return e.Msg }

func errf(code pb.ErrorCode, f string, a ...any) error {
	return &Error{Code: code, Msg: fmt.Sprintf(f, a...)}
}

// Store is the dataset store.
type Store struct {
	Root     string
	Chunk    int64
	MaxNumel uint64
	Rails    func() config.Rails
	Quota    func(identity string) uint64 // storage quota in bytes, 0 = none
	mu       sync.Mutex
	locks    map[string]*sync.Mutex
}

// New returns a store rooted at root.
func New(root string, chunk int64, maxNumel uint64, railsFn func() config.Rails, quota func(string) uint64) *Store {
	return &Store{Root: root, Chunk: chunk, MaxNumel: maxNumel, Rails: railsFn, Quota: quota, locks: map[string]*sync.Mutex{}}
}

func (s *Store) lock(identity string) func() {
	s.mu.Lock()
	l := s.locks[identity]
	if l == nil {
		l = &sync.Mutex{}
		s.locks[identity] = l
	}
	s.mu.Unlock()
	l.Lock()
	return l.Unlock
}

func (s *Store) dir(identity string, parts ...string) string {
	return filepath.Join(append([]string{s.Root, identity}, parts...)...)
}

// PackRecord is a committed dataset.
type PackRecord struct {
	ID       string    `json:"id"`
	Name     string    `json:"name"`
	Manifest Manifest  `json:"manifest"`
	Created  time.Time `json:"created"`
}

func (s *Store) chunks(size int64) int { return int((size + s.Chunk - 1) / s.Chunk) }

func (s *Store) ensure(identity string) error {
	for _, d := range []string{"blobs", "packs", "uploads"} {
		if err := os.MkdirAll(s.dir(identity, d), 0o700); err != nil {
			return err
		}
	}
	return nil
}

// Begin starts or resumes an upload and reports missing chunks.
func (s *Store) Begin(identity string, req *pb.PackBegin, maxManifest int) (*pb.PackStatus, error) {
	if !NameRe.MatchString(req.Name) {
		return nil, errf(pb.ErrorCode_ERROR_DATASET, "dataset name must match [a-z0-9][a-z0-9_-]{0,63}")
	}
	if len(req.Manifest) > maxManifest {
		return nil, errf(pb.ErrorCode_ERROR_RESOURCE_LIMIT, "manifest larger than %d bytes", maxManifest)
	}
	m, err := ParseManifest(req.Manifest, s.MaxNumel)
	if err != nil {
		return nil, errf(pb.ErrorCode_ERROR_DATASET, "%v", err)
	}
	unlock := s.lock(identity)
	defer unlock()
	if err := s.ensure(identity); err != nil {
		return nil, err
	}
	sum := sha256.Sum256(req.Manifest)
	id := hex.EncodeToString(sum[:])
	st := &pb.PackStatus{PackId: sum[:], ChunkBytes: uint64(s.Chunk)}
	var missing uint64
	for i, sh := range m.Shards {
		if _, err := os.Stat(s.dir(identity, "blobs", sh.SHA256)); err == nil {
			continue
		}
		have := s.bitmap(identity, sh.SHA256, s.chunks(sh.SizeBytes))
		ss := &pb.ShardStatus{Shard: uint32(i)}
		for c, ok := range have {
			if !ok {
				ss.MissingChunks = append(ss.MissingChunks, uint32(c))
				missing += uint64(s.chunkLen(sh.SizeBytes, c))
			}
		}
		if len(ss.MissingChunks) > 0 {
			st.Shards = append(st.Shards, ss)
		}
	}
	if missing > 0 {
		if err := rails.CheckDiskWrite(s.Rails(), s.Root, missing); err != nil {
			return nil, errf(pb.ErrorCode_ERROR_SAFETY_RAIL, "%v", err)
		}
		if q := s.Quota(identity); q > 0 && s.usage(identity)+missing > q {
			return nil, errf(pb.ErrorCode_ERROR_QUOTA, "storage quota of %s would be exceeded", config.Bytes(q))
		}
	}
	rec := struct {
		Name     string `json:"name"`
		Manifest []byte `json:"manifest"`
	}{req.Name, req.Manifest}
	b, _ := json.Marshal(rec)
	if err := config.WriteFileAtomic(s.dir(identity, "uploads", id+".begin"), b, 0o600); err != nil {
		return nil, err
	}
	return st, nil
}

func (s *Store) chunkLen(size int64, c int) int64 {
	off := int64(c) * s.Chunk
	if size-off < s.Chunk {
		return size - off
	}
	return s.Chunk
}

func (s *Store) bitmap(identity, sha string, n int) []bool {
	b, _ := os.ReadFile(s.dir(identity, "uploads", sha+".chunks"))
	out := make([]bool, n)
	for i := range out {
		out[i] = i < len(b) && b[i] == 1
	}
	return out
}

func (s *Store) pending(identity string, packID []byte) (string, *Manifest, error) {
	if len(packID) != 32 {
		return "", nil, errf(pb.ErrorCode_ERROR_DATASET, "bad pack id")
	}
	id := hex.EncodeToString(packID)
	b, err := os.ReadFile(s.dir(identity, "uploads", id+".begin"))
	if err != nil {
		return "", nil, errf(pb.ErrorCode_ERROR_DATASET, "no upload in progress for this pack; send PackBegin first")
	}
	var rec struct {
		Name     string `json:"name"`
		Manifest []byte `json:"manifest"`
	}
	if err := json.Unmarshal(b, &rec); err != nil {
		return "", nil, err
	}
	m, err := ParseManifest(rec.Manifest, s.MaxNumel)
	if err != nil {
		return "", nil, err
	}
	return rec.Name, m, nil
}

// Chunk stores one verified chunk.
func (s *Store) PutChunk(identity string, c *pb.PackChunk) error {
	unlock := s.lock(identity)
	defer unlock()
	_, m, err := s.pending(identity, c.PackId)
	if err != nil {
		return err
	}
	if int(c.Shard) >= len(m.Shards) {
		return errf(pb.ErrorCode_ERROR_DATASET, "shard index out of range")
	}
	sh := m.Shards[c.Shard]
	n := s.chunks(sh.SizeBytes)
	if int(c.Chunk) >= n {
		return errf(pb.ErrorCode_ERROR_DATASET, "chunk index out of range")
	}
	if int64(len(c.Data)) != s.chunkLen(sh.SizeBytes, int(c.Chunk)) {
		return errf(pb.ErrorCode_ERROR_DATASET, "chunk has the wrong length")
	}
	sum := sha256.Sum256(c.Data)
	if len(c.Sha256) != 32 || string(sum[:]) != string(c.Sha256) {
		return errf(pb.ErrorCode_ERROR_DATASET, "chunk checksum mismatch")
	}
	if _, err := os.Stat(s.dir(identity, "blobs", sh.SHA256)); err == nil {
		return nil // already have it
	}
	if err := rails.CheckDiskWrite(s.Rails(), s.Root, uint64(len(c.Data))); err != nil {
		s.abortLocked(identity, m)
		return errf(pb.ErrorCode_ERROR_SAFETY_RAIL, "%v; the partial upload was removed", err)
	}
	part := s.dir(identity, "uploads", sh.SHA256+".part")
	f, err := os.OpenFile(part, os.O_CREATE|os.O_WRONLY, 0o600)
	if err != nil {
		return err
	}
	if _, err := f.WriteAt(c.Data, int64(c.Chunk)*s.Chunk); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	bm := s.dir(identity, "uploads", sh.SHA256+".chunks")
	bf, err := os.OpenFile(bm, os.O_CREATE|os.O_WRONLY, 0o600)
	if err != nil {
		return err
	}
	defer bf.Close()
	if fi, _ := bf.Stat(); fi != nil && fi.Size() < int64(n) {
		if err := bf.Truncate(int64(n)); err != nil {
			return err
		}
	}
	_, err = bf.WriteAt([]byte{1}, int64(c.Chunk))
	return err
}

func (s *Store) abortLocked(identity string, m *Manifest) {
	for _, sh := range m.Shards {
		os.Remove(s.dir(identity, "uploads", sh.SHA256+".part"))
		os.Remove(s.dir(identity, "uploads", sh.SHA256+".chunks"))
	}
}

// Commit validates every shard and issues the dataset (§10.3: a dataset
// handle exists only after full validation).
func (s *Store) Commit(identity string, packID []byte) (*pb.DatasetInfo, error) {
	unlock := s.lock(identity)
	defer unlock()
	name, m, err := s.pending(identity, packID)
	if err != nil {
		return nil, err
	}
	for i, sh := range m.Shards {
		blob := s.dir(identity, "blobs", sh.SHA256)
		if _, err := os.Stat(blob); err == nil {
			if err := s.validateShard(blob, m, i); err != nil {
				return nil, err
			}
			continue
		}
		for c, ok := range s.bitmap(identity, sh.SHA256, s.chunks(sh.SizeBytes)) {
			if !ok {
				return nil, errf(pb.ErrorCode_ERROR_DATASET, "shard %d is missing chunk %d", i, c)
			}
		}
		part := s.dir(identity, "uploads", sh.SHA256+".part")
		if err := verifyFile(part, sh.SHA256, sh.SizeBytes); err != nil {
			s.abortLocked(identity, m)
			return nil, errf(pb.ErrorCode_ERROR_DATASET, "shard %d: %v", i, err)
		}
		if err := s.validateShard(part, m, i); err != nil {
			s.abortLocked(identity, m)
			return nil, err
		}
		if err := os.Chmod(part, 0o400); err != nil {
			return nil, err
		}
		if err := os.Rename(part, blob); err != nil {
			return nil, err
		}
		os.Remove(s.dir(identity, "uploads", sh.SHA256+".chunks"))
	}
	id := hex.EncodeToString(packID)
	// One dataset per name: re-pushing a name replaces the old pack.
	for _, r := range s.listLocked(identity) {
		if r.Name == name && r.ID != id {
			os.Remove(s.dir(identity, "packs", r.ID+".json"))
		}
	}
	rec := PackRecord{ID: id, Name: name, Manifest: *m, Created: time.Now().UTC().Truncate(time.Second)}
	b, _ := json.MarshalIndent(rec, "", " ")
	if err := config.WriteFileAtomic(s.dir(identity, "packs", id+".json"), b, 0o600); err != nil {
		return nil, err
	}
	os.Remove(s.dir(identity, "uploads", id+".begin"))
	s.gcLocked(identity)
	return rec.Info(), nil
}

func (s *Store) validateShard(path string, m *Manifest, i int) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return err
	}
	ts, err := ParseSafetensors(f, st.Size(), s.MaxNumel)
	if err != nil {
		return errf(pb.ErrorCode_ERROR_DATASET, "shard %d: %v", i, err)
	}
	if len(ts) != len(m.Tensors) {
		return errf(pb.ErrorCode_ERROR_DATASET, "shard %d: tensors do not match the manifest", i)
	}
	for _, mt := range m.Tensors {
		t, ok := ts[mt.Name]
		if !ok || t.DType != DTypeToST[mt.DType] || len(t.Shape) != len(mt.Shape) ||
			uint64(t.Shape[0]) != m.Shards[i].NumSamples {
			return errf(pb.ErrorCode_ERROR_DATASET, "shard %d: tensor %q does not match the manifest", i, mt.Name)
		}
		for d := 1; d < len(mt.Shape); d++ {
			if t.Shape[d] != mt.Shape[d] {
				return errf(pb.ErrorCode_ERROR_DATASET, "shard %d: tensor %q has the wrong shape", i, mt.Name)
			}
		}
	}
	return nil
}

func verifyFile(path, want string, size int64) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return err
	}
	if st.Size() != size {
		return fmt.Errorf("size %d, expected %d", st.Size(), size)
	}
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return err
	}
	if hex.EncodeToString(h.Sum(nil)) != want {
		return errors.New("checksum mismatch")
	}
	return nil
}

// Info converts a record for the protocol.
func (r *PackRecord) Info() *pb.DatasetInfo {
	id, _ := hex.DecodeString(r.ID)
	inf := &pb.DatasetInfo{DatasetId: id, Name: r.Name, NumSamples: r.Manifest.NumSamples,
		Bytes: r.Manifest.TotalBytes(), CreatedUnix: r.Created.Unix()}
	for _, t := range r.Manifest.Tensors {
		inf.Tensors = append(inf.Tensors, &pb.DatasetTensorInfo{Name: t.Name, Meta: &pb.TensorMeta{
			Dtype: ManifestDTypeToPB[t.DType], Shape: t.Shape}})
	}
	return inf
}

// ManifestDTypeToPB maps manifest dtype names to wire dtypes.
var ManifestDTypeToPB = map[string]pb.DType{
	"bool": pb.DType_DTYPE_BOOL, "uint8": pb.DType_DTYPE_UINT8, "int8": pb.DType_DTYPE_INT8,
	"int16": pb.DType_DTYPE_INT16, "int32": pb.DType_DTYPE_INT32, "int64": pb.DType_DTYPE_INT64,
	"float16": pb.DType_DTYPE_FLOAT16, "bfloat16": pb.DType_DTYPE_BFLOAT16, "float32": pb.DType_DTYPE_FLOAT32,
	"float64": pb.DType_DTYPE_FLOAT64,
}

func (s *Store) listLocked(identity string) []*PackRecord {
	ents, _ := os.ReadDir(s.dir(identity, "packs"))
	var out []*PackRecord
	for _, e := range ents {
		b, err := os.ReadFile(s.dir(identity, "packs", e.Name()))
		if err != nil {
			continue
		}
		var r PackRecord
		if json.Unmarshal(b, &r) == nil {
			out = append(out, &r)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// List returns an identity's datasets.
func (s *Store) List(identity string) []*PackRecord {
	unlock := s.lock(identity)
	defer unlock()
	return s.listLocked(identity)
}

// Get returns one dataset by id.
func (s *Store) Get(identity string, id []byte) (*PackRecord, error) {
	if len(id) != 32 {
		return nil, errf(pb.ErrorCode_ERROR_UNKNOWN_DATASET, "unknown dataset")
	}
	b, err := os.ReadFile(s.dir(identity, "packs", hex.EncodeToString(id)+".json"))
	if err != nil {
		return nil, errf(pb.ErrorCode_ERROR_UNKNOWN_DATASET, "unknown dataset")
	}
	var r PackRecord
	if err := json.Unmarshal(b, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

// Delete removes a dataset and unreferenced blobs.
func (s *Store) Delete(identity string, id []byte) error {
	unlock := s.lock(identity)
	defer unlock()
	if len(id) != 32 {
		return errf(pb.ErrorCode_ERROR_UNKNOWN_DATASET, "unknown dataset")
	}
	if err := os.Remove(s.dir(identity, "packs", hex.EncodeToString(id)+".json")); err != nil {
		return errf(pb.ErrorCode_ERROR_UNKNOWN_DATASET, "unknown dataset")
	}
	s.gcLocked(identity)
	return nil
}

func (s *Store) gcLocked(identity string) {
	keep := map[string]bool{}
	for _, r := range s.listLocked(identity) {
		for _, sh := range r.Manifest.Shards {
			keep[sh.SHA256] = true
		}
	}
	ents, _ := os.ReadDir(s.dir(identity, "blobs"))
	for _, e := range ents {
		if !keep[e.Name()] {
			p := s.dir(identity, "blobs", e.Name())
			os.Chmod(p, 0o600)
			os.Remove(p)
		}
	}
}

// OpenShards opens a dataset's shard files read-only, for fd passing.
func (s *Store) OpenShards(identity string, r *PackRecord) ([]*os.File, error) {
	var fs []*os.File
	for _, sh := range r.Manifest.Shards {
		f, err := os.Open(s.dir(identity, "blobs", sh.SHA256))
		if err != nil {
			for _, x := range fs {
				x.Close()
			}
			return nil, err
		}
		fs = append(fs, f)
	}
	return fs, nil
}

func (s *Store) usage(identity string) uint64 {
	var n uint64
	_ = filepath.Walk(s.dir(identity), func(_ string, fi os.FileInfo, err error) error {
		if err == nil && fi.Mode().IsRegular() {
			n += uint64(fi.Size())
		}
		return nil
	})
	return n
}

// Usage reports an identity's storage use in bytes.
func (s *Store) Usage(identity string) uint64 { return s.usage(identity) }
