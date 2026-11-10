// SPDX-License-Identifier: Apache-2.0

package wire

import (
	"bytes"
	"errors"
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
