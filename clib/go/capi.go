// SPDX-License-Identifier: Apache-2.0

// Command libreflop builds the reFLOP C API as a shared library:
//
//	go build -buildmode=c-shared -o libreflop.so ./clib/go
package main

/*
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
// Same layout as include/reflop.h (which adds const qualifiers; the ABI is
// identical). The public header is not included here because cgo writes its
// own prototypes without const.
typedef int reflop_dtype;
typedef struct { uint8_t bytes[16]; } reflop_handle;
typedef struct { uint8_t bytes[32]; } reflop_graph_id;
typedef struct { int32_t node; uint32_t index; } reflop_value;
typedef uint64_t reflop_session;
typedef uint64_t reflop_graph;
*/
import "C"

import (
	"sync"
	"unsafe"

	"google.golang.org/protobuf/proto"

	pb "github.com/chaseungjoon/reflop/pkg/reflopv1"
)

const version = "0.1.2"

var (
	mu       sync.Mutex
	nextObj  uint64
	sessions = map[uint64]*conn{}
	graphs   = map[uint64]*builder{}
	lastErr  = map[uint64]string{} // per OS thread would need cgo TLS; per-object store plus a global is enough
	errMsg   string
	verC     = C.CString(version)
	errC     *C.char
)

func fail(err error) C.int {
	mu.Lock()
	errMsg = sanitize(err.Error())
	mu.Unlock()
	return -1
}

func ok() C.int { return 0 }

//export reflop_last_error
func reflop_last_error() *C.char {
	mu.Lock()
	defer mu.Unlock()
	if errC != nil {
		C.free(unsafe.Pointer(errC))
	}
	errC = C.CString(errMsg)
	return errC
}

//export reflop_version
func reflop_version() *C.char { return verC }

func sess(s C.reflop_session) (*conn, bool) {
	mu.Lock()
	defer mu.Unlock()
	c, ok := sessions[uint64(s)]
	return c, ok
}

func graph(g C.reflop_graph) (*builder, bool) {
	mu.Lock()
	defer mu.Unlock()
	b, ok := graphs[uint64(g)]
	return b, ok
}

func newID() uint64 {
	nextObj++
	return nextObj
}

//export reflop_connect
func reflop_connect(server *C.char, device *C.char, out *C.reflop_session) C.int {
	srv, dev := "", "auto"
	if server != nil {
		srv = C.GoString(server)
	}
	if device != nil {
		dev = C.GoString(device)
	}
	c, err := connect(srv, dev)
	if err != nil {
		return fail(err)
	}
	mu.Lock()
	id := newID()
	sessions[id] = c
	mu.Unlock()
	*out = C.reflop_session(id)
	return ok()
}

//export reflop_close
func reflop_close(s C.reflop_session) C.int {
	c, found := sess(s)
	if !found {
		return fail(errBad("session"))
	}
	c.call(&pb.Envelope{Body: &pb.Envelope_CloseSession{CloseSession: &pb.CloseSession{}}})
	c.rw.Close()
	mu.Lock()
	delete(sessions, uint64(s))
	mu.Unlock()
	return ok()
}

type badErr string

func (e badErr) Error() string { return "invalid " + string(e) + " handle" }
func errBad(what string) error { return badErr(what) }

func meta(dtype C.reflop_dtype, ndim C.int, shape *C.int64_t) *pb.TensorMeta {
	m := &pb.TensorMeta{Dtype: pb.DType(dtype)}
	if ndim > 0 {
		sh := unsafe.Slice((*int64)(unsafe.Pointer(shape)), int(ndim))
		m.Shape = append([]int64(nil), sh...)
	}
	return m
}

//export reflop_upload
func reflop_upload(s C.reflop_session, dtype C.reflop_dtype, ndim C.int, shape *C.int64_t, data unsafe.Pointer,
	nbytes C.size_t, out *C.reflop_handle) C.int {
	c, found := sess(s)
	if !found {
		return fail(errBad("session"))
	}
	buf := C.GoBytes(data, C.int(nbytes))
	h, err := c.upload(meta(dtype, ndim, shape), buf)
	if err != nil {
		return fail(err)
	}
	C.memcpy(unsafe.Pointer(&out.bytes[0]), unsafe.Pointer(&h[0]), 16)
	return ok()
}

func handle(h C.reflop_handle) []byte { return C.GoBytes(unsafe.Pointer(&h.bytes[0]), 16) }

//export reflop_fetch
func reflop_fetch(s C.reflop_session, h C.reflop_handle, buf unsafe.Pointer, capacity C.size_t, nbytes *C.size_t) C.int {
	c, found := sess(s)
	if !found {
		return fail(errBad("session"))
	}
	frames, err := c.stream(&pb.Envelope{Body: &pb.Envelope_Fetch{Fetch: &pb.Fetch{Handles: [][]byte{handle(h)}}}})
	if err != nil {
		return fail(err)
	}
	var data []byte
	for _, f := range frames {
		data = append(data, f.GetFetchChunk().GetData()...)
	}
	if nbytes != nil {
		*nbytes = C.size_t(len(data))
	}
	n := len(data)
	if n > int(capacity) {
		n = int(capacity)
	}
	if n > 0 {
		C.memcpy(buf, unsafe.Pointer(&data[0]), C.size_t(n))
	}
	return ok()
}

//export reflop_free
func reflop_free(s C.reflop_session, h C.reflop_handle) C.int {
	c, found := sess(s)
	if !found {
		return fail(errBad("session"))
	}
	if _, err := c.call(&pb.Envelope{Body: &pb.Envelope_Free{Free: &pb.Free{Handles: [][]byte{handle(h)}}}}); err != nil {
		return fail(err)
	}
	return ok()
}

//export reflop_graph_new
func reflop_graph_new() C.reflop_graph {
	mu.Lock()
	defer mu.Unlock()
	id := newID()
	graphs[id] = newBuilder()
	return C.reflop_graph(id)
}

//export reflop_graph_free
func reflop_graph_free(g C.reflop_graph) {
	mu.Lock()
	delete(graphs, uint64(g))
	mu.Unlock()
}

//export reflop_graph_input
func reflop_graph_input(g C.reflop_graph, dtype C.reflop_dtype, ndim C.int, shape *C.int64_t) C.reflop_value {
	b, found := graph(g)
	if !found {
		fail(errBad("graph"))
		return C.reflop_value{node: -2}
	}
	v := b.input(meta(dtype, ndim, shape))
	return C.reflop_value{node: C.int32_t(v.node), index: C.uint32_t(v.index)}
}

func withGraph(g C.reflop_graph, f func(*builder) error) C.int {
	b, found := graph(g)
	if !found {
		return fail(errBad("graph"))
	}
	if err := f(b); err != nil {
		return fail(err)
	}
	return ok()
}

func gv(v C.reflop_value) value { return value{int32(v.node), uint32(v.index)} }

//export reflop_op_begin
func reflop_op_begin(g C.reflop_graph, op *C.char) C.int {
	return withGraph(g, func(b *builder) error { return b.begin(C.GoString(op)) })
}

//export reflop_arg_tensor
func reflop_arg_tensor(g C.reflop_graph, v C.reflop_value) C.int {
	return withGraph(g, func(b *builder) error { return b.push(&pb.Arg{Value: &pb.Arg_Tensor{Tensor: ref(gv(v))}}) })
}

//export reflop_arg_tensors
func reflop_arg_tensors(g C.reflop_graph, vs *C.reflop_value, n C.int) C.int {
	return withGraph(g, func(b *builder) error {
		l := &pb.TensorList{}
		for _, v := range unsafe.Slice(vs, int(n)) {
			l.Items = append(l.Items, ref(gv(v)))
		}
		return b.push(&pb.Arg{Value: &pb.Arg_Tensors{Tensors: l}})
	})
}

//export reflop_arg_int
func reflop_arg_int(g C.reflop_graph, v C.int64_t) C.int {
	return withGraph(g, func(b *builder) error { return b.push(&pb.Arg{Value: &pb.Arg_I{I: int64(v)}}) })
}

//export reflop_arg_ints
func reflop_arg_ints(g C.reflop_graph, v *C.int64_t, n C.int) C.int {
	return withGraph(g, func(b *builder) error {
		vals := append([]int64(nil), unsafe.Slice((*int64)(unsafe.Pointer(v)), int(n))...)
		return b.push(&pb.Arg{Value: &pb.Arg_Ints{Ints: &pb.Int64List{Values: vals}}})
	})
}

//export reflop_arg_float
func reflop_arg_float(g C.reflop_graph, v C.double) C.int {
	return withGraph(g, func(b *builder) error { return b.push(&pb.Arg{Value: &pb.Arg_F{F: float64(v)}}) })
}

//export reflop_arg_bool
func reflop_arg_bool(g C.reflop_graph, v C.int) C.int {
	return withGraph(g, func(b *builder) error { return b.push(&pb.Arg{Value: &pb.Arg_B{B: v != 0}}) })
}

//export reflop_arg_dtype
func reflop_arg_dtype(g C.reflop_graph, v C.reflop_dtype) C.int {
	return withGraph(g, func(b *builder) error { return b.push(&pb.Arg{Value: &pb.Arg_Dtype{Dtype: pb.DType(v)}}) })
}

//export reflop_arg_enum
func reflop_arg_enum(g C.reflop_graph, v *C.char) C.int {
	return withGraph(g, func(b *builder) error { return b.enum(C.GoString(v)) })
}

//export reflop_arg_device
func reflop_arg_device(g C.reflop_graph) C.int {
	return withGraph(g, func(b *builder) error {
		return b.push(&pb.Arg{Value: &pb.Arg_Device{Device: &pb.Device{}}}) // filled at register time
	})
}

//export reflop_arg_none
func reflop_arg_none(g C.reflop_graph) C.int {
	return withGraph(g, func(b *builder) error { return b.push(&pb.Arg{Value: &pb.Arg_None{None: true}}) })
}

//export reflop_op_end
func reflop_op_end(g C.reflop_graph, out *C.reflop_value) C.int {
	return withGraph(g, func(b *builder) error {
		v, err := b.end()
		if err == nil && out != nil {
			out.node, out.index = C.int32_t(v.node), C.uint32_t(v.index)
		}
		return err
	})
}

//export reflop_graph_output
func reflop_graph_output(g C.reflop_graph, v C.reflop_value, dtype C.reflop_dtype, ndim C.int, shape *C.int64_t) C.int {
	return withGraph(g, func(b *builder) error { b.output(gv(v), meta(dtype, ndim, shape)); return nil })
}

//export reflop_register
func reflop_register(s C.reflop_session, g C.reflop_graph, out *C.reflop_graph_id) C.int {
	c, found := sess(s)
	if !found {
		return fail(errBad("session"))
	}
	b, found := graph(g)
	if !found {
		return fail(errBad("graph"))
	}
	gr := proto.Clone(b.g).(*pb.Graph)
	gr.Device = c.device
	for _, n := range gr.Nodes {
		for _, a := range n.Args {
			if d := a.GetDevice(); d != nil {
				a.Value = &pb.Arg_Device{Device: c.device}
			}
		}
	}
	r, err := c.call(&pb.Envelope{Body: &pb.Envelope_RegisterGraph{RegisterGraph: &pb.RegisterGraph{Graph: gr}}})
	if err != nil {
		return fail(err)
	}
	h := r.GetGraphRegistered().GetGraphHash()
	C.memcpy(unsafe.Pointer(&out.bytes[0]), unsafe.Pointer(&h[0]), 32)
	return ok()
}

//export reflop_execute
func reflop_execute(s C.reflop_session, id C.reflop_graph_id, inputs *C.reflop_handle, nIn C.int,
	outputs *C.reflop_handle, nOut C.int) C.int {
	c, found := sess(s)
	if !found {
		return fail(errBad("session"))
	}
	ex := &pb.Execute{GraphHash: C.GoBytes(unsafe.Pointer(&id.bytes[0]), 32)}
	for _, h := range unsafe.Slice(inputs, int(nIn)) {
		ex.Bindings = append(ex.Bindings, &pb.Binding{Value: &pb.Binding_Handle{Handle: handle(h)}})
	}
	outs := unsafe.Slice(outputs, int(nOut))
	newH := make([][16]byte, nOut)
	for i := range outs {
		newH[i] = c.newHandle()
		ex.Outputs = append(ex.Outputs, &pb.OutputSpec{Kind: pb.OutputSpec_KIND_STORE, Handle: newH[i][:]})
	}
	if _, err := c.call(&pb.Envelope{Body: &pb.Envelope_Execute{Execute: ex}}); err != nil {
		return fail(err)
	}
	for i := range outs {
		C.memcpy(unsafe.Pointer(&outs[i].bytes[0]), unsafe.Pointer(&newH[i][0]), 16)
	}
	return ok()
}

func main() {}

var _ = lastErr
