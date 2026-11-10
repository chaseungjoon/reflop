/* SPDX-License-Identifier: Apache-2.0
 * Remote matmul with the reFLOP C API; checks the result locally. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "reflop.h"

#define CHECK(x) do { if ((x) != 0) { fprintf(stderr, "%s: %s\n", #x, reflop_last_error()); exit(1); } } while (0)

int main(void) {
  enum { M = 64, K = 48, N = 32 };
  static float a[M * K], b[K * N], c[M * N], ref[M * N];
  for (int i = 0; i < M * K; i++) a[i] = (float)((i * 7) % 13) / 13.0f - 0.5f;
  for (int i = 0; i < K * N; i++) b[i] = (float)((i * 5) % 11) / 11.0f - 0.5f;
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) {
      float s = 0;
      for (int k = 0; k < K; k++) s += a[i * K + k] * b[k * N + j];
      ref[i * N + j] = s;
    }

  reflop_session s;
  CHECK(reflop_connect(NULL, "auto", &s));
  int64_t sa[2] = {M, K}, sb[2] = {K, N}, sc[2] = {M, N};
  reflop_handle ha, hb;
  CHECK(reflop_upload(s, REFLOP_F32, 2, sa, a, sizeof a, &ha));
  CHECK(reflop_upload(s, REFLOP_F32, 2, sb, b, sizeof b, &hb));

  reflop_graph g = reflop_graph_new();
  reflop_value x = reflop_graph_input(g, REFLOP_F32, 2, sa);
  reflop_value y = reflop_graph_input(g, REFLOP_F32, 2, sb);
  reflop_value z;
  CHECK(reflop_op_begin(g, "aten.mm.default"));
  CHECK(reflop_arg_tensor(g, x));
  CHECK(reflop_arg_tensor(g, y));
  CHECK(reflop_op_end(g, &z));
  CHECK(reflop_graph_output(g, z, REFLOP_F32, 2, sc));

  reflop_graph_id id;
  CHECK(reflop_register(s, g, &id));
  reflop_handle in[2] = {ha, hb}, out[1];
  CHECK(reflop_execute(s, id, in, 2, out, 1));
  size_t n = 0;
  CHECK(reflop_fetch(s, out[0], c, sizeof c, &n));

  double err = 0;
  for (int i = 0; i < M * N; i++) err = fmax(err, fabs(c[i] - ref[i]));
  printf("matmul %dx%d @ %dx%d on the server: %zu bytes back, max abs error %.2e\n", M, K, K, N, n, err);
  reflop_graph_free(g);
  CHECK(reflop_close(s));
  return err < 1e-3 ? 0 : 2;
}
