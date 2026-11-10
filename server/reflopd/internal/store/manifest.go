// SPDX-License-Identifier: Apache-2.0

package store

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
)

// Manifest is a TensorPack's manifest.json (PLAN.md §10.2).
//
//	{"format_version": 1, "num_samples": 50000,
//	 "tensors": [{"name": "images", "dtype": "uint8", "shape": [50000, 3, 32, 32]}],
//	 "shards":  [{"file": "shard-00000.safetensors", "sha256": "…", "size_bytes": 123, "num_samples": 25000}]}
//
// Every shard holds every tensor, sharded along the sample axis.
type Manifest struct {
	FormatVersion int              `json:"format_version"`
	NumSamples    uint64           `json:"num_samples"`
	Tensors       []ManifestTensor `json:"tensors"`
	Shards        []ManifestShard  `json:"shards"`
}

type ManifestTensor struct {
	Name  string  `json:"name"`
	DType string  `json:"dtype"`
	Shape []int64 `json:"shape"`
}

type ManifestShard struct {
	File       string `json:"file"`
	SHA256     string `json:"sha256"`
	SizeBytes  int64  `json:"size_bytes"`
	NumSamples uint64 `json:"num_samples"`
}

var (
	tensorNameRe = regexp.MustCompile(`^[a-z0-9_]{1,64}$`)
	shardFileRe  = regexp.MustCompile(`^shard-[0-9]{5}\.safetensors$`)
	sha256Re     = regexp.MustCompile(`^[0-9a-f]{64}$`)
)

// DTypeToST maps manifest dtypes to safetensors dtypes.
var DTypeToST = map[string]string{
	"bool": "BOOL", "uint8": "U8", "int8": "I8", "int16": "I16", "int32": "I32", "int64": "I64",
	"float16": "F16", "bfloat16": "BF16", "float32": "F32", "float64": "F64",
}

// MaxShards bounds a pack.
const MaxShards = 100000

// ParseManifest decodes and validates a manifest strictly (unknown fields
// rejected, names constrained, counts consistent).
func ParseManifest(b []byte, maxNumel uint64) (*Manifest, error) {
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	var m Manifest
	if err := dec.Decode(&m); err != nil {
		return nil, fmt.Errorf("manifest: %w", err)
	}
	if dec.More() {
		return nil, errors.New("manifest: trailing data")
	}
	if m.FormatVersion != 1 {
		return nil, fmt.Errorf("manifest: unsupported format_version %d", m.FormatVersion)
	}
	if len(m.Tensors) == 0 || len(m.Tensors) > 64 {
		return nil, errors.New("manifest: 1 to 64 tensors required")
	}
	if len(m.Shards) == 0 || len(m.Shards) > MaxShards {
		return nil, errors.New("manifest: bad shard count")
	}
	seen := map[string]bool{}
	for _, t := range m.Tensors {
		if !tensorNameRe.MatchString(t.Name) {
			return nil, fmt.Errorf("manifest: tensor name %q must match [a-z0-9_]{1,64}", t.Name)
		}
		if seen[t.Name] {
			return nil, fmt.Errorf("manifest: duplicate tensor %q", t.Name)
		}
		seen[t.Name] = true
		if _, ok := DTypeToST[t.DType]; !ok {
			return nil, fmt.Errorf("manifest: tensor %q: dtype %q not allowed", t.Name, t.DType)
		}
		if len(t.Shape) == 0 || len(t.Shape) > 16 || t.Shape[0] < 0 || uint64(t.Shape[0]) != m.NumSamples {
			return nil, fmt.Errorf("manifest: tensor %q: shape must start with num_samples", t.Name)
		}
		n := uint64(1)
		for _, d := range t.Shape {
			if d < 0 || (d != 0 && n > maxNumel/uint64(d)) {
				return nil, fmt.Errorf("manifest: tensor %q: bad shape", t.Name)
			}
			n *= uint64(d)
		}
	}
	files := map[string]bool{}
	var total uint64
	for i, s := range m.Shards {
		if !shardFileRe.MatchString(s.File) || s.File != fmt.Sprintf("shard-%05d.safetensors", i) {
			return nil, fmt.Errorf("manifest: shard %d must be named shard-%05d.safetensors", i, i)
		}
		if files[s.File] {
			return nil, fmt.Errorf("manifest: duplicate shard %q", s.File)
		}
		files[s.File] = true
		if !sha256Re.MatchString(s.SHA256) {
			return nil, fmt.Errorf("manifest: shard %d: bad sha256", i)
		}
		if s.SizeBytes <= 8 || s.SizeBytes > 1<<40 {
			return nil, fmt.Errorf("manifest: shard %d: bad size", i)
		}
		total += s.NumSamples
	}
	if total != m.NumSamples {
		return nil, errors.New("manifest: shard num_samples do not add up to num_samples")
	}
	return &m, nil
}

// TotalBytes sums shard sizes.
func (m *Manifest) TotalBytes() uint64 {
	var n uint64
	for _, s := range m.Shards {
		n += uint64(s.SizeBytes)
	}
	return n
}

// ShardDigest decodes a shard's sha256.
func (s ManifestShard) Digest() []byte { b, _ := hex.DecodeString(s.SHA256); return b }
