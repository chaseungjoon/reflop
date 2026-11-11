/* SPDX-License-Identifier: Apache-2.0
 *
 * libreflop: the reFLOP C API (PLAN.md §18, M7).
 *
 * Build operator graphs over opset v1, upload tensors, execute graphs on a
 * reflop server, and fetch results. The program runs on this machine; only
 * graphs and tensors are sent. Connections use the servers paired with the
 * Python client (`reflop pair`); the server key is pinned.
 *
 * Every function returns 0 on success and a negative value on error; the
 * message is available from reflop_last_error(): per thread, sanitized, and
 * valid until the next failing call on the same thread.
 */
#ifndef REFLOP_H
#define REFLOP_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Wire dtypes (graph.proto DType). */
typedef enum {
  REFLOP_BOOL = 1, REFLOP_U8 = 2, REFLOP_I8 = 3, REFLOP_I16 = 4, REFLOP_I32 = 5, REFLOP_I64 = 6,
  REFLOP_F16 = 7, REFLOP_BF16 = 8, REFLOP_F32 = 9, REFLOP_F64 = 10
} reflop_dtype;

typedef struct { uint8_t bytes[16]; } reflop_handle; /* a server-resident tensor */
typedef struct { uint8_t bytes[32]; } reflop_graph_id; /* registered graph hash */

/* A value inside a graph under construction: a graph input or a node output. */
typedef struct {
  int32_t node;   /* -1 for graph inputs */
  uint32_t index; /* input index, or output index of the node */
} reflop_value;

/* Error codes (other failures return -1). */
#define REFLOP_ERANGE (-34) /* reflop_fetch: the buffer is smaller than the tensor */

typedef uint64_t reflop_session; /* 0 is never a valid session */
typedef uint64_t reflop_graph;   /* 0 is never a valid graph builder */

const char *reflop_last_error(void);
const char *reflop_version(void);

/* server: paired server name, or NULL for the default.
 * device: "auto", "cuda" or "cpu" (the server's CPU). */
int reflop_connect(const char *server, const char *device, reflop_session *out);
int reflop_close(reflop_session s);

/* nbytes must equal the product of shape times the dtype size. */
int reflop_upload(reflop_session s, reflop_dtype dtype, int ndim, const int64_t *shape, const void *data,
                  size_t nbytes, reflop_handle *out);
/* *nbytes (required) receives the tensor size. With buf == NULL and cap == 0
 * only the size is returned. If the tensor does not fit in cap bytes,
 * nothing is copied and REFLOP_ERANGE is returned. */
int reflop_fetch(reflop_session s, reflop_handle h, void *buf, size_t cap, size_t *nbytes);
int reflop_free(reflop_session s, reflop_handle h);

/* Graph builder. */
reflop_graph reflop_graph_new(void);
void reflop_graph_free(reflop_graph g);
reflop_value reflop_graph_input(reflop_graph g, reflop_dtype dtype, int ndim, const int64_t *shape);

/* Add one node: op name is the fully qualified overload ("aten.mm.default").
 * Push every schema argument in order (keyword-only ones included), then
 * reflop_op_end. Its first output is returned in *out (others: node + k).
 * Nodes nest: reflop_op_begin while another node is open starts an inner node,
 * arguments go to the innermost open node, and reflop_op_end closes it. */
int reflop_op_begin(reflop_graph g, const char *op);
int reflop_arg_tensor(reflop_graph g, reflop_value v);
int reflop_arg_tensors(reflop_graph g, const reflop_value *v, int n);
int reflop_arg_int(reflop_graph g, int64_t v);
int reflop_arg_ints(reflop_graph g, const int64_t *v, int n);
int reflop_arg_float(reflop_graph g, double v);
int reflop_arg_bool(reflop_graph g, int v);
int reflop_arg_dtype(reflop_graph g, reflop_dtype v);
int reflop_arg_enum(reflop_graph g, const char *value);
int reflop_arg_device(reflop_graph g);
int reflop_arg_none(reflop_graph g);
int reflop_op_end(reflop_graph g, reflop_value *out);

/* Declare a graph output and the dtype/shape you expect (checked by the
 * server's meta inference). */
int reflop_graph_output(reflop_graph g, reflop_value v, reflop_dtype dtype, int ndim, const int64_t *shape);

int reflop_register(reflop_session s, reflop_graph g, reflop_graph_id *out);
/* Bind one handle per graph input; every output is stored in a new handle. */
int reflop_execute(reflop_session s, reflop_graph_id id, const reflop_handle *inputs, int n_inputs,
                   reflop_handle *outputs, int n_outputs);

#ifdef __cplusplus
}
#endif
#endif /* REFLOP_H */
