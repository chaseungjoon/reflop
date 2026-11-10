// SPDX-License-Identifier: Apache-2.0

package store

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sort"
)

// MaxHeader bounds the safetensors JSON header (§10.2).
const MaxHeader = 1 << 20

var stDTypes = map[string]int{
	"BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "I32": 4, "I64": 8, "F16": 2, "BF16": 2, "F32": 4, "F64": 8,
}

// STTensor is one parsed tensor entry.
type STTensor struct {
	DType string
	Shape []int64
	Begin int64
	End   int64
}

// ParseSafetensors validates a safetensors file strictly: header length
// and JSON shape, dtype whitelist, in-bounds and non-overlapping offsets
// consistent with dtype × shape, and no trailing bytes.
func ParseSafetensors(r io.ReaderAt, size int64, maxNumel uint64) (map[string]STTensor, error) {
	if size < 8 {
		return nil, errors.New("safetensors: file too short")
	}
	var hl [8]byte
	if _, err := r.ReadAt(hl[:], 0); err != nil {
		return nil, err
	}
	n := binary.LittleEndian.Uint64(hl[:])
	if n == 0 || n > MaxHeader || int64(n)+8 > size {
		return nil, errors.New("safetensors: bad header length")
	}
	hdr := make([]byte, n)
	if _, err := r.ReadAt(hdr, 8); err != nil {
		return nil, err
	}
	dec := json.NewDecoder(bytes.NewReader(hdr))
	dec.UseNumber()
	tok, err := dec.Token()
	if err != nil || tok != json.Delim('{') {
		return nil, errors.New("safetensors: header must be a JSON object")
	}
	out := map[string]STTensor{}
	dataLen := size - 8 - int64(n)
	for dec.More() {
		kt, err := dec.Token()
		if err != nil {
			return nil, fmt.Errorf("safetensors: %w", err)
		}
		key, ok := kt.(string)
		if !ok {
			return nil, errors.New("safetensors: bad key")
		}
		if _, dup := out[key]; dup {
			return nil, fmt.Errorf("safetensors: duplicate key %q", key)
		}
		if key == "__metadata__" {
			var skip json.RawMessage
			if err := dec.Decode(&skip); err != nil {
				return nil, err
			}
			continue
		}
		var raw map[string]json.RawMessage
		if err := dec.Decode(&raw); err != nil {
			return nil, fmt.Errorf("safetensors: tensor %q: %w", key, err)
		}
		if len(raw) != 3 || raw["dtype"] == nil || raw["shape"] == nil || raw["data_offsets"] == nil {
			return nil, fmt.Errorf("safetensors: tensor %q: unexpected fields", key)
		}
		var t STTensor
		var offs []int64
		if json.Unmarshal(raw["dtype"], &t.DType) != nil || json.Unmarshal(raw["shape"], &t.Shape) != nil ||
			json.Unmarshal(raw["data_offsets"], &offs) != nil || len(offs) != 2 {
			return nil, fmt.Errorf("safetensors: tensor %q: malformed entry", key)
		}
		item, ok := stDTypes[t.DType]
		if !ok {
			return nil, fmt.Errorf("safetensors: tensor %q: dtype %q not allowed", key, t.DType)
		}
		if len(t.Shape) > 16 {
			return nil, fmt.Errorf("safetensors: tensor %q: too many dims", key)
		}
		numel := uint64(1)
		for _, d := range t.Shape {
			if d < 0 {
				return nil, fmt.Errorf("safetensors: tensor %q: negative dim", key)
			}
			if d != 0 && numel > maxNumel/uint64(d) {
				return nil, fmt.Errorf("safetensors: tensor %q: too many elements", key)
			}
			numel *= uint64(d)
		}
		if numel > maxNumel {
			return nil, fmt.Errorf("safetensors: tensor %q: too many elements", key)
		}
		t.Begin, t.End = offs[0], offs[1]
		if t.Begin < 0 || t.Begin > t.End || t.End > dataLen || uint64(t.End-t.Begin) != numel*uint64(item) {
			return nil, fmt.Errorf("safetensors: tensor %q: data_offsets out of bounds or inconsistent with shape", key)
		}
		out[key] = t
	}
	if _, err := dec.Token(); err != nil {
		return nil, errors.New("safetensors: malformed header")
	}
	rest := bytes.TrimRight(hdr[dec.InputOffset():], " ")
	if len(rest) != 0 {
		return nil, errors.New("safetensors: trailing bytes in header")
	}
	type span struct{ b, e int64 }
	var spans []span
	for _, t := range out {
		spans = append(spans, span{t.Begin, t.End})
	}
	sort.Slice(spans, func(i, j int) bool { return spans[i].b < spans[j].b })
	var end int64
	for _, s := range spans {
		if s.b < end {
			return nil, errors.New("safetensors: overlapping tensors")
		}
		end = s.e
	}
	if end != dataLen {
		return nil, errors.New("safetensors: trailing or unreferenced bytes after the tensor data")
	}
	return out, nil
}
