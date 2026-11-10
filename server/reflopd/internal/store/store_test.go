// SPDX-License-Identifier: Apache-2.0

package store

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	pb "github.com/chaseungjoon/reflop/pkg/reflopv1"
	"github.com/chaseungjoon/reflop/server/reflopd/internal/config"
)

func stFile(hdr string, data []byte) []byte {
	var b bytes.Buffer
	binary.Write(&b, binary.LittleEndian, uint64(len(hdr)))
	b.WriteString(hdr)
	b.Write(data)
	return b.Bytes()
}

func parse(t *testing.T, f []byte) error {
	t.Helper()
	_, err := ParseSafetensors(bytes.NewReader(f), int64(len(f)), 1<<40)
	return err
}

func TestSafetensorsValid(t *testing.T) {
	f := stFile(`{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"__metadata__":{"a":"b"}}`, make([]byte, 8))
	if err := parse(t, f); err != nil {
		t.Fatal(err)
	}
}

func TestSafetensorsCorpus(t *testing.T) {
	cases := map[string][]byte{
		"header bomb":     append([]byte{0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x0f}, []byte("{}")...),
		"past end":        stFile(`{"x":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}`, make([]byte, 8)),
		"size mismatch":   stFile(`{"x":{"dtype":"F32","shape":[3],"data_offsets":[0,8]}}`, make([]byte, 8)),
		"overlap":         stFile(`{"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},"b":{"dtype":"U8","shape":[4],"data_offsets":[2,6]}}`, make([]byte, 6)),
		"trailing":        stFile(`{"x":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}`, make([]byte, 5)),
		"bad dtype":       stFile(`{"x":{"dtype":"PY_OBJECT","shape":[1],"data_offsets":[0,1]}}`, make([]byte, 1)),
		"negative dim":    stFile(`{"x":{"dtype":"U8","shape":[-1],"data_offsets":[0,0]}}`, nil),
		"numel overflow":  stFile(`{"x":{"dtype":"U8","shape":[1099511627776,1099511627776],"data_offsets":[0,0]}}`, nil),
		"extra field":     stFile(`{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1],"evil":1}}`, make([]byte, 1)),
		"duplicate key":   stFile(`{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}`, make([]byte, 1)),
		"not object":      stFile(`[1,2,3]`, nil),
		"truncated":       []byte{1, 2, 3},
		"negative offset": stFile(`{"x":{"dtype":"U8","shape":[1],"data_offsets":[-1,0]}}`, make([]byte, 1)),
	}
	for name, f := range cases {
		if err := parse(t, f); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

func manifest(t *testing.T, shards ...ManifestShard) []byte {
	var n uint64
	for _, s := range shards {
		n += s.NumSamples
	}
	b, _ := json.Marshal(Manifest{FormatVersion: 1, NumSamples: n,
		Tensors: []ManifestTensor{{Name: "x", DType: "uint8", Shape: []int64{int64(n), 2}}}, Shards: shards})
	return b
}

func TestManifestCorpus(t *testing.T) {
	good := ManifestShard{File: "shard-00000.safetensors", SHA256: strings.Repeat("a", 64), SizeBytes: 100, NumSamples: 2}
	if _, err := ParseManifest(manifest(t, good), 1<<40); err != nil {
		t.Fatal(err)
	}
	bad := map[string][]byte{
		"path traversal file": manifest(t, ManifestShard{File: "../../etc/passwd", SHA256: good.SHA256, SizeBytes: 100, NumSamples: 2}),
		"slash in file":       manifest(t, ManifestShard{File: "a/shard-00000.safetensors", SHA256: good.SHA256, SizeBytes: 100, NumSamples: 2}),
		"bad sha":             manifest(t, ManifestShard{File: good.File, SHA256: "xyz", SizeBytes: 100, NumSamples: 2}),
		"unknown field":       []byte(`{"format_version":1,"num_samples":0,"tensors":[],"shards":[],"evil":1}`),
		"bad tensor name":     []byte(`{"format_version":1,"num_samples":1,"tensors":[{"name":"../x","dtype":"uint8","shape":[1]}],"shards":[{"file":"shard-00000.safetensors","sha256":"` + good.SHA256 + `","size_bytes":100,"num_samples":1}]}`),
		"count mismatch":      []byte(`{"format_version":1,"num_samples":5,"tensors":[{"name":"x","dtype":"uint8","shape":[5]}],"shards":[{"file":"shard-00000.safetensors","sha256":"` + good.SHA256 + `","size_bytes":100,"num_samples":1}]}`),
	}
	for name, m := range bad {
		if _, err := ParseManifest(m, 1<<40); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

func TestUploadResumeAndValidate(t *testing.T) {
	dir := t.TempDir()
	s := New(dir, 4, 1<<40, func() config.Rails { return config.Rails{} }, func(string) uint64 { return 0 })
	data := make([]byte, 6)
	shard := stFile(`{"x":{"dtype":"U8","shape":[3,2],"data_offsets":[0,6]}}  `, data)
	sum := sha256.Sum256(shard)
	m := manifest(t, ManifestShard{File: "shard-00000.safetensors", SHA256: hex.EncodeToString(sum[:]), SizeBytes: int64(len(shard)), NumSamples: 3})
	st, err := s.Begin("id1", &pb.PackBegin{Name: "toy", Manifest: m}, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	n := len(st.Shards[0].MissingChunks)
	put := func(c int) error {
		end := (c + 1) * 4
		if end > len(shard) {
			end = len(shard)
		}
		d := shard[c*4 : end]
		h := sha256.Sum256(d)
		return s.PutChunk("id1", &pb.PackChunk{PackId: st.PackId, Shard: 0, Chunk: uint32(c), Data: d, Sha256: h[:]})
	}
	if err := put(0); err != nil {
		t.Fatal(err)
	}
	// "Reconnect": Begin again reports only the remaining chunks.
	st2, err := s.Begin("id1", &pb.PackBegin{Name: "toy", Manifest: m}, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	if got := len(st2.Shards[0].MissingChunks); got != n-1 {
		t.Fatalf("resume: %d missing, want %d", got, n-1)
	}
	if _, err := s.Commit("id1", st.PackId); err == nil {
		t.Fatal("commit with missing chunks succeeded")
	}
	for c := 1; c < n; c++ {
		if err := put(c); err != nil {
			t.Fatal(err)
		}
	}
	bad := sha256.Sum256([]byte("nope"))
	if err := s.PutChunk("id1", &pb.PackChunk{PackId: st.PackId, Chunk: 0, Data: shard[:4], Sha256: bad[:]}); err == nil {
		t.Fatal("bad chunk checksum accepted")
	}
	info, err := s.Commit("id1", st.PackId)
	if err != nil {
		t.Fatal(err)
	}
	if info.NumSamples != 3 || info.Name != "toy" {
		t.Fatalf("info %+v", info)
	}
	if _, err := os.Stat(filepath.Join(dir, "id1", "blobs", hex.EncodeToString(sum[:]))); err != nil {
		t.Fatal("blob not stored content-addressed")
	}
	if len(s.List("id2")) != 0 {
		t.Fatal("dataset visible to another identity")
	}
	if _, err := s.Get("id2", st.PackId); err == nil {
		t.Fatal("cross-identity Get succeeded")
	}
}

func TestDiskRailRefusesUpload(t *testing.T) {
	dir := t.TempDir()
	s := New(dir, 4, 1<<40, func() config.Rails { return config.Rails{DiskFreeFloorPercent: 90, DiskFreeFloorBytes: 1 << 62} },
		func(string) uint64 { return 0 })
	m := manifest(t, ManifestShard{File: "shard-00000.safetensors", SHA256: strings.Repeat("a", 64), SizeBytes: 100, NumSamples: 2})
	_, err := s.Begin("id1", &pb.PackBegin{Name: "big", Manifest: m}, 1<<20)
	var se *Error
	if err == nil || !errorsAs(err, &se) || se.Code != pb.ErrorCode_ERROR_SAFETY_RAIL {
		t.Fatalf("want SAFETY_RAIL, got %v", err)
	}
}

func errorsAs(err error, target **Error) bool {
	e, ok := err.(*Error)
	if ok {
		*target = e
	}
	return ok
}

func FuzzSafetensors(f *testing.F) {
	f.Add(stFile(`{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}`, make([]byte, 8)))
	f.Fuzz(func(t *testing.T, data []byte) {
		ParseSafetensors(bytes.NewReader(data), int64(len(data)), 1<<40)
	})
}

func FuzzManifest(f *testing.F) {
	f.Add([]byte(`{"format_version":1}`))
	f.Fuzz(func(t *testing.T, data []byte) { ParseManifest(data, 1<<40) })
}
