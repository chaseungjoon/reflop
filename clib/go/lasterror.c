// SPDX-License-Identifier: Apache-2.0
//
// Thread-local storage for reflop_last_error() (SEC_SWEEP L1). Exported Go
// functions run on the C thread that called them, so a failing call stores
// its message for that thread only; no other thread can see or free it.

#include <stdlib.h>
#include <string.h>

static __thread char *reflop_tls_error;

// reflop_set_error replaces this thread's message with a copy of msg[0:n].
void reflop_set_error(const char *msg, size_t n) {
  char *p = malloc(n + 1);
  if (p == NULL) {
    return; // keep the previous message rather than lose the error
  }
  memcpy(p, msg, n);
  p[n] = '\0';
  free(reflop_tls_error);
  reflop_tls_error = p;
}

// reflop_get_error returns this thread's message ("" if none). The pointer
// stays valid until the next failing libreflop call on the same thread.
const char *reflop_get_error(void) {
  return reflop_tls_error != NULL ? reflop_tls_error : "";
}
