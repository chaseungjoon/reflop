# libreflop: the reFLOP C and C++ API

Build operator graphs from C or C++, run them on a reflop server, and fetch the
results. Your program runs locally; only graphs and tensors are sent.

```c
#include "reflop.h"

reflop_session s;
reflop_connect(NULL, "auto", &s);              /* default paired server */

int64_t sa[2] = {64, 48}, sb[2] = {48, 32}, sc[2] = {64, 32};
reflop_handle ha, hb;
reflop_upload(s, REFLOP_F32, 2, sa, a, sizeof a, &ha);
reflop_upload(s, REFLOP_F32, 2, sb, b, sizeof b, &hb);

reflop_graph g = reflop_graph_new();
reflop_value x = reflop_graph_input(g, REFLOP_F32, 2, sa);
reflop_value y = reflop_graph_input(g, REFLOP_F32, 2, sb);
reflop_value z;
reflop_op_begin(g, "aten.mm.default");
reflop_arg_tensor(g, x);
reflop_arg_tensor(g, y);
reflop_op_end(g, &z);
reflop_graph_output(g, z, REFLOP_F32, 2, sc);

reflop_graph_id id;
reflop_register(s, g, &id);
reflop_handle in[2] = {ha, hb}, out[1];
reflop_execute(s, id, in, 2, out, 1);
reflop_fetch(s, out[0], c, sizeof c, NULL);
```

The C++ header wraps the same calls:

```cpp
#include "reflop.hpp"

reflop::Session s;
reflop::Graph g;
auto x = g.input(REFLOP_F32, {4, 8});
auto w = g.input(REFLOP_F32, {16, 8});
auto b = g.input(REFLOP_F32, {16});
auto h = g.op("aten.addmm.default").tensor(b).tensor(x)
          .tensor(g.op("aten.t.default").tensor(w).end()).i(1).i(1).end();
g.output(g.op("aten.relu.default").tensor(h).end(), REFLOP_F32, {4, 16});
auto out = s.run(g, {s.upload(xs, {4, 8}), s.upload(ws, {16, 8}), s.upload(bs, {16})});
std::vector<float> r = s.fetch<float>(out[0]);
```

## Build

```sh
make -C clib                # build/libreflop.so, build/matmul, build/mlp_forward
make -C clib check          # run both examples against your default server
cc -Iclib/include prog.c -Lclib/build -lreflop -Wl,-rpath,clib/build
```

The same `cc` line works on Linux and macOS: on macOS the library's install
name is `@rpath/libreflop.so`, so the `-rpath` you pass decides where it is
found (`@loader_path` for "next to the program").

Building needs Go (cgo) and a C compiler. The library has no runtime
dependencies beyond libc.

## Connecting

`reflop_connect` reads the servers you paired with the Python client
(`reflop pair`), from `~/.config/reflop/servers.json` and the key in
`~/.config/reflop/id_ed25519`. On macOS the Python client stores the key in the
Keychain; run `reflop key --write-file` once to also store it as that file.
`REFLOP_SERVER` selects a server by name; `device` is `"auto"`, `"cuda"` or
`"cpu"` (the server's CPU). The server key is pinned as in the Python client.

## Building nodes

A node is an opset entry ([docs/OPSET.md](../docs/OPSET.md)) named by its
fully qualified overload, for example `aten.addmm.default`. Push every schema
argument in order, keyword-only ones included:

| Schema type | Call |
|---|---|
| `Tensor` | `reflop_arg_tensor` |
| `Tensor[]`, `Tensor?[]` | `reflop_arg_tensors` |
| `int`, `SymInt`, `Scalar` (integer) | `reflop_arg_int` |
| `int[]`, `SymInt[]` | `reflop_arg_ints` |
| `float`, `Scalar` (float) | `reflop_arg_float` |
| `bool` | `reflop_arg_bool` |
| `ScalarType` | `reflop_arg_dtype` |
| `str` (enum) | `reflop_arg_enum`, e.g. `"tanh"` for `gelu`'s `approximate` |
| `Device` | `reflop_arg_device` (the session's device) |
| an optional argument left as `None` | `reflop_arg_none` |

`reflop_op_end` rejects a wrong argument count and names the schema. Declare
each output with `reflop_graph_output`; the server's meta inference checks the
dtype and shape you give and rejects the graph on a mismatch.

Every function returns 0 on success; `reflop_last_error()` describes a failure.
The server validates the graph exactly as it does for the Python client.
