// SPDX-License-Identifier: Apache-2.0

package wire

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"runtime"
	"testing"

	"google.golang.org/protobuf/encoding/protowire"

	pb "github.com/chaseungjoon/reflop/pkg/reflopv1"
)

func TestRoundTrip(t *testing.T) {
	var buf bytes.Buffer
	w := NewWriter(&buf, 1<<20)
	if err := w.Write(&pb.Envelope{RequestId: 7, Body: &pb.Envelope_Ping{Ping: &pb.Ping{Nonce: 9}}}); err != nil {
		t.Fatal(err)
	}
	r := NewReader(&buf, 1<<20)
	env := &pb.Envelope{}
	if err := r.Read(env); err != nil {
		t.Fatal(err)
	}
	if env.RequestId != 7 || env.GetPing().Nonce != 9 {
		t.Fatalf("got %v", env)
	}
}

func TestRejectsUnknownFieldsAtDepth(t *testing.T) {
	inner := protowire.AppendTag(nil, 99, protowire.VarintType)
	inner = protowire.AppendVarint(inner, 1)
	ping := protowire.AppendTag(nil, 28, protowire.BytesType) // Envelope.ping
	ping = protowire.AppendBytes(ping, inner)
	if err := Unmarshal(ping, &pb.Envelope{}); !errors.Is(err, ErrUnknownField) {
		t.Fatalf("want ErrUnknownField, got %v", err)
	}
	top := protowire.AppendTag(nil, 500, protowire.VarintType)
	top = protowire.AppendVarint(top, 3)
	if err := Unmarshal(top, &pb.Envelope{}); !errors.Is(err, ErrUnknownField) {
		t.Fatalf("want ErrUnknownField at top level, got %v", err)
	}
}

func TestRejectsUnknownEnum(t *testing.T) {
	g := &pb.Graph{Device: &pb.Device{Kind: pb.DeviceKind(77)}}
	if err := CheckEnums(g.ProtoReflect()); err == nil {
		t.Fatal("unknown enum accepted")
	}
}

func TestFrameLimit(t *testing.T) {
	var buf bytes.Buffer
	buf.Write([]byte{0x7f, 0xff, 0xff, 0xff})
	r := NewReader(&buf, 1<<20)
	if _, err := r.ReadFrame(); !errors.Is(err, ErrFrameTooLarge) {
		t.Fatalf("want ErrFrameTooLarge, got %v", err)
	}
}

func FuzzUnmarshalEnvelope(f *testing.F) {
	b, _ := Marshal(&pb.Envelope{RequestId: 1, Body: &pb.Envelope_Hello{Hello: &pb.Hello{ProtocolVersions: []uint32{1}}}})
	f.Add(b)
	f.Fuzz(func(t *testing.T, data []byte) {
		env := &pb.Envelope{}
		if Unmarshal(data, env) == nil {
			_ = CheckEnums(env.ProtoReflect())
		}
	})
}

// SEC_SWEEP M1: a declared frame length is not allocated up front, so a
// peer that sends only a header costs almost nothing.
func TestLargeDeclaredFrameNotPreallocated(t *testing.T) {
	var hdr [4]byte
	binary.BigEndian.PutUint32(hdr[:], 512<<20)
	r := NewReader(io.MultiReader(bytes.NewReader(hdr[:]), bytes.NewReader([]byte("short"))), 1<<30)
	var before, after runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&before)
	_, err := r.ReadFrame()
	runtime.ReadMemStats(&after)
	if !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("got %v, want ErrUnexpectedEOF", err)
	}
	if d := after.TotalAlloc - before.TotalAlloc; d > 8<<20 {
		t.Fatalf("allocated %d bytes for a 5-byte body of a 512 MiB frame", d)
	}
	// Frames still round-trip across the eager-allocation threshold.
	for _, n := range []int{0, 1, eagerAlloc, eagerAlloc + 1, 3*eagerAlloc + 7} {
		var b bytes.Buffer
		w := NewWriter(&b, 1<<30)
		payload := bytes.Repeat([]byte{0xab}, n)
		if err := w.WriteFrame(payload); err != nil {
			t.Fatal(err)
		}
		got, err := NewReader(&b, 1<<30).ReadFrame()
		if err != nil || !bytes.Equal(got, payload) {
			t.Fatalf("n=%d: %v", n, err)
		}
	}
	r2 := NewReader(bytes.NewReader([]byte{0, 0, 1, 0}), 1<<20)
	r2.SetMax(16)
	if _, err := r2.ReadFrame(); !errors.Is(err, ErrFrameTooLarge) {
		t.Fatalf("SetMax not applied: %v", err)
	}
}
