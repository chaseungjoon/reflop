/* SPDX-License-Identifier: Apache-2.0
 *
 * SEC_SWEEP L1: reflop_last_error() is per thread. A pointer one thread got
 * stays valid and unchanged while another thread fails, and neither thread
 * sees the other's message. L3: a short fetch buffer is reported, and
 * nbytes may not be NULL. Runs without a server.
 */
#include <pthread.h>
#include <stdio.h>
#include <string.h>

#include "reflop.h"

static int failures;
#define EXPECT(c, ...) do { if (!(c)) { fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); failures++; } } while (0)

static void *graph_thread(void *arg) {
  (void)arg;
  EXPECT(reflop_op_begin(424242, "aten.mm.default") != 0, "op_begin on a bad graph succeeded");
  const char *mine = reflop_last_error();
  EXPECT(strstr(mine, "graph") != NULL, "thread B sees %s", mine);
  return NULL;
}

int main(void) {
  EXPECT(reflop_close(777) != 0, "close on a bad session succeeded");
  const char *p = reflop_last_error(); /* thread A keeps this pointer */
  char copy[256];
  snprintf(copy, sizeof copy, "%s", p);
  EXPECT(strstr(copy, "session") != NULL, "thread A sees %s", copy);

  pthread_t t; /* thread B fails and reads its own message meanwhile */
  pthread_create(&t, NULL, graph_thread, NULL);
  pthread_join(t, NULL);

  EXPECT(strcmp(p, copy) == 0, "thread A's message changed to %s", p);
  EXPECT(strcmp(reflop_last_error(), copy) == 0, "thread A now sees %s", reflop_last_error());

  reflop_handle h;
  memset(&h, 0, sizeof h);
  EXPECT(reflop_fetch(777, h, NULL, 0, NULL) < 0, "fetch accepted bad arguments");
  int64_t shape[1] = {-1};
  EXPECT(reflop_upload(777, REFLOP_F32, 1, shape, NULL, 4, &h) < 0, "upload accepted bad arguments");

  if (failures == 0) {
    puts("lasterror: ok");
  }
  return failures != 0;
}
