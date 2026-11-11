// SPDX-License-Identifier: Apache-2.0

// Package wire implements reflop framing: a uint32 big-endian length prefix
// followed by a protobuf message. Decoding is strict: unknown fields
// anywhere in the message tree are rejected (invariant I11).
package wire

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"sync"
	"sync/atomic"

	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"
)

// ErrFrameTooLarge is returned when a frame exceeds the configured limit.
var ErrFrameTooLarge = errors.New("frame exceeds size limit")

// ErrUnknownField is returned when a decoded message carries unknown fields.
var ErrUnknownField = errors.New("message contains unknown fields")

// Reader reads frames.
type Reader struct {
	r   *bufio.Reader
	max atomic.Uint32
	hdr [4]byte
}

// NewReader returns a frame reader that rejects frames larger than max bytes.
func NewReader(r io.Reader, max uint32) *Reader {
	fr := &Reader{r: bufio.NewReaderSize(r, 64<<10)}
	fr.max.Store(max)
	return fr
}

// SetMax changes the frame limit for the frames that follow (reflopd raises
// it from the small pre-authentication limit once a peer has authenticated).
func (r *Reader) SetMax(max uint32) { r.max.Store(max) }

// eagerAlloc is the largest buffer allocated up front from a declared frame
// length. Longer frames grow as their bytes arrive, so a peer that only
// declares a large frame costs nothing (SEC_SWEEP M1).
const eagerAlloc = 1 << 20

// ReadFrame returns the next frame payload.
func (r *Reader) ReadFrame() ([]byte, error) {
	if _, err := io.ReadFull(r.r, r.hdr[:]); err != nil {
		return nil, err
	}
	n := binary.BigEndian.Uint32(r.hdr[:])
	if max := r.max.Load(); n > max {
		return nil, fmt.Errorf("%w: %d > %d", ErrFrameTooLarge, n, max)
	}
	var buf []byte
	var err error
	if n <= eagerAlloc {
		buf = make([]byte, n)
		_, err = io.ReadFull(r.r, buf)
	} else {
		buf, err = readGrowing(r.r, int(n))
	}
	if err != nil {
		if errors.Is(err, io.EOF) {
			err = io.ErrUnexpectedEOF
		}
		return nil, err
	}
	return buf, nil
}

// readGrowing reads exactly n bytes, doubling the buffer only as data
// arrives.
func readGrowing(r io.Reader, n int) ([]byte, error) {
	buf := make([]byte, 0, eagerAlloc)
	for len(buf) < n {
		if len(buf) == cap(buf) {
			c := 2 * cap(buf)
			if c > n {
				c = n
			}
			nb := make([]byte, len(buf), c)
			copy(nb, buf)
			buf = nb
		}
		m, err := r.Read(buf[len(buf):cap(buf)])
		buf = buf[:len(buf)+m]
		if err != nil {
			if len(buf) == n {
				break
			}
			if errors.Is(err, io.EOF) {
				return nil, io.ErrUnexpectedEOF
			}
			return nil, err
		}
	}
	return buf, nil
}

// Read decodes the next frame into m strictly.
func (r *Reader) Read(m proto.Message) error {
	b, err := r.ReadFrame()
	if err != nil {
		return err
	}
	return Unmarshal(b, m)
}

// Writer writes frames. It is safe for concurrent use.
type Writer struct {
	mu  sync.Mutex
	w   io.Writer
	max uint32
}

// NewWriter returns a frame writer.
func NewWriter(w io.Writer, max uint32) *Writer { return &Writer{w: w, max: max} }

// WriteFrame writes one payload as a frame.
func (w *Writer) WriteFrame(b []byte) error {
	if uint64(len(b)) > uint64(w.max) {
		return fmt.Errorf("%w: %d > %d", ErrFrameTooLarge, len(b), w.max)
	}
	buf := make([]byte, 4+len(b))
	binary.BigEndian.PutUint32(buf, uint32(len(b)))
	copy(buf[4:], b)
	w.mu.Lock()
	defer w.mu.Unlock()
	_, err := w.w.Write(buf)
	return err
}

// Write encodes m deterministically and writes it as one frame.
func (w *Writer) Write(m proto.Message) error {
	b, err := Marshal(m)
	if err != nil {
		return err
	}
	return w.WriteFrame(b)
}

// Marshal encodes deterministically.
func Marshal(m proto.Message) ([]byte, error) {
	return proto.MarshalOptions{Deterministic: true}.Marshal(m)
}

// Unmarshal decodes b into m and rejects unknown fields at any depth.
func Unmarshal(b []byte, m proto.Message) error {
	if err := (proto.UnmarshalOptions{RecursionLimit: 64}).Unmarshal(b, m); err != nil {
		return err
	}
	return CheckNoUnknown(m.ProtoReflect())
}

// CheckNoUnknown walks the message tree and fails if any message holds
// unknown fields. Proto3 enums are open, so unknown enum numbers survive
// decoding as plain values; callers validate enum ranges separately.
func CheckNoUnknown(m protoreflect.Message) error {
	if len(m.GetUnknown()) > 0 {
		return fmt.Errorf("%w in %s", ErrUnknownField, m.Descriptor().FullName())
	}
	var err error
	m.Range(func(fd protoreflect.FieldDescriptor, v protoreflect.Value) bool {
		switch {
		case fd.IsList() && fd.Message() != nil:
			l := v.List()
			for i := 0; i < l.Len() && err == nil; i++ {
				err = CheckNoUnknown(l.Get(i).Message())
			}
		case fd.IsMap():
			if fd.MapValue().Message() != nil {
				v.Map().Range(func(_ protoreflect.MapKey, mv protoreflect.Value) bool {
					err = CheckNoUnknown(mv.Message())
					return err == nil
				})
			}
		case fd.Message() != nil:
			err = CheckNoUnknown(v.Message())
		}
		return err == nil
	})
	return err
}

// CheckEnums fails if any enum field holds a value not declared in its enum
// (invariant I11: unknown enum values are rejected).
func CheckEnums(m protoreflect.Message) error {
	var err error
	m.Range(func(fd protoreflect.FieldDescriptor, v protoreflect.Value) bool {
		switch {
		case fd.Enum() != nil && fd.IsList():
			l := v.List()
			for i := 0; i < l.Len() && err == nil; i++ {
				err = checkEnumValue(fd, l.Get(i).Enum())
			}
		case fd.Enum() != nil:
			err = checkEnumValue(fd, v.Enum())
		case fd.IsList() && fd.Message() != nil:
			l := v.List()
			for i := 0; i < l.Len() && err == nil; i++ {
				err = CheckEnums(l.Get(i).Message())
			}
		case fd.Message() != nil && !fd.IsMap():
			err = CheckEnums(v.Message())
		}
		return err == nil
	})
	return err
}

func checkEnumValue(fd protoreflect.FieldDescriptor, n protoreflect.EnumNumber) error {
	if fd.Enum().Values().ByNumber(n) == nil {
		return fmt.Errorf("unknown enum value %d for %s", n, fd.FullName())
	}
	return nil
}
