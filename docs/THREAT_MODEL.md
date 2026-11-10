# reFLOP threat model

reFLOP lets people the Owner does not fully trust (Guests) use the Owner's GPU.
This document states what it defends, against whom, how each defense is
enforced, how it is tested, and what remains possible. It expands
[PLAN.md §2–§3](../PLAN.md#2-threat-model).

## Actors and trust

| Actor | Trust | Notes |
|---|---|---|
| Owner | trusted | owns the GPU machine, runs `reflopd` |
| `reflopd` and `reflop-worker` | trusted code, assumed to have bugs | defense in depth around both |
| Guest (client) | **untrusted** | may be malicious, careless or compromised; may skip the reflop client and write raw protocol messages |
| Network | untrusted | eavesdropping, tampering, replay, impersonation |
| Server machine | compromise considered unlikely | the client defends itself anyway |

## Assets

| Asset | Threats |
|---|---|
| Owner's files, processes, account and OS | code execution, file read/write, privilege escalation, persistence |
| Machine availability (disk, host RAM, process table) | disk fill, RAM exhaustion, fork bombs |
| GPU time for the Owner | a Guest monopolizing the GPU |
| Guest-to-Guest isolation | one Guest reading or changing another's tensors or datasets |
| Guest machines | a malicious or impersonated server attacking the client |
| The connection | MITM, replay, impersonation |

## Attack surface

A Guest can influence the server only through:

1. the handshake (Hello, Pair, Authenticate),
2. protobuf requests (register graph, execute, upload, fetch, free, datasets, snapshots),
3. raw tensor bytes with an explicit dtype and shape,
4. TensorPack uploads (safetensors shards and a JSON manifest).

There is no request that carries code, a file path, a shell command, a package,
a kernel, or a pickled object. Anything else reaching the server is a defect.

## Invariants and how they hold

| ID | Invariant | Enforcement | Verified by |
|---|---|---|---|
| I1 | the server never runs client-supplied bytes as code | the worker has no code path that loads code; the op table refers to PyTorch kernels by literal name in generated source | `tests/security/static/check_worker.py` (CI) bans `eval`, `exec`, pickle family, `subprocess`, `importlib`, `torch.load`, `torch.compile`, `torch.library`, new sockets and dynamic attribute access in worker code |
| I2 | dispatch is integer id → static table | `ops_generated.py` is generated from `opset.yaml`; no `getattr` with a runtime name | the same static check |
| I3 | full validation before any device allocation | V1 strict decode, V2 structure (DAG order, whitelist, capability, arity), V3 schema (types, enums, dim ranges, sizes with checked arithmetic), V4 meta inference, V5 peak memory vs device size and quotas | 184 cases in `tests/security/corpus` |
| I4 | data-dependent indices are bounds-checked before the kernel | per-op validators (`validators.py`) run before each IDX op; value ranges of immutable server tensors are proven on the host where possible, otherwise measured on the device | an out-of-bounds case (−1, size, INT64_MAX, INT64_MIN) per IDX op in the corpus; sentinel rules (`ignore_index`, `padding_idx`) |
| I5 | one sandboxed worker per (identity, session), no network | Landlock applied by `reflopd __sandbox-exec` before `exec`; seccomp allowlist installed after warm-up on all threads; user+network namespaces when allowed; cgroup per worker | `tests/security/isolation` (32 probes inside real workers) and `TestWorkerSeccompFilter` |
| I6 | client formats are data only | strict protobuf (unknown fields and enum values rejected), raw bytes with checked sizes, our own safetensors parsers in Go and Python, schema-checked JSON manifests | corpus decode and data cases; Go fuzzing of framing, safetensors and manifests |
| I7 | no shell, no OS accounts per Guest, no Guest-chosen paths | daemon mode has no shell at all; SSH mode uses `restrict,command=…`; storage paths are content hashes under reflopd's directories | `reflopd doctor` scan of `authorized_keys`; manifest path-traversal cases |
| I8 | handles and datasets are per identity | a worker serves one session, so other sessions' handles do not exist in it; datasets live under the identity's id and are opened by reflopd, which passes file descriptors | cross-session and cross-identity tests (corpus, store tests) |
| I9 | the client deserializes no executable formats, writes nowhere the server chooses, strips control sequences | protobuf and raw bytes only; checkpoints are safetensors; `sanitize()` on every server string | `test_client_protection.py` |
| I10 | `opset.yaml` is the single source of truth | `codegen/gen.py` generates the worker table, the client mapping, the Go table and `docs/OPSET.md` | `gen.py --check` in CI |
| I11 | fail closed | unknown fields, ops, dtypes, versions, devices, attributes and enum values are rejected | corpus decode, structure and schema cases |
| I12 | the client sends no environment, paths, hostnames, secrets or source | the encoder serializes only graphs, tensors and protocol fields | `test_client_never_sends_environment` |
| I13 | no uninitialized memory | `empty*` and `new_empty*` outputs are zero-filled | UNINIT cases in the corpus, which dirty the allocator first |

## Sandbox layers

| Layer | Level 0 (rootless) | Level 1 (`install --system`) |
|---|---|---|
| Account | the Owner | system user `reflop` (no shell, no home of its own) |
| no_new_privs | yes | yes |
| Landlock | worker: read-only runtime, system libraries and device topology; read-write only its scratch directory and its own `/proc/<pid>`; device nodes of its GPU; TCP (and UDP on newer kernels) denied; abstract Unix sockets and signals scoped (ABI ≥ 6) | same |
| seccomp | worker allowlist: no `execve`, no process creation, no new sockets of any family, no `connect`/`bind`, no `ptrace`, mount, namespaces, eBPF, perf, keyrings, `userfaultfd`, `io_uring`; signals only to itself | same |
| Namespaces | user + empty network namespace when unprivileged ones are allowed | same |
| cgroup | per-worker leaf under a delegated scope: `pids.max`, OOM kills the whole worker, shared `memory.max` that tracks the host-RAM floor | delegated from the system unit |
| reflopd itself | Landlock (system read-only, its own directories read-write, TCP bind only on its ports) and a seccomp denylist, applied by re-exec | plus systemd hardening |
| Environment | an explicit allowlist (no Owner variables) | same |

Two details found on real hardware shaped the policy (ADR-004): the CUDA driver
writes `/proc/self/task/<tid>/comm` during initialization, so a worker may write
its own `/proc/<pid>`; and Landlock rules apply to inodes, so workers are
started through `/proc/self/exe`, which keeps working after the binary on disk
is replaced.

If neither Landlock nor seccomp is available, reflopd refuses Guest sessions
unless the Owner passes `--accept-weak-isolation`.

## Network and identity

- Daemon mode: TLS 1.3. The client pins the SHA-256 of the server's
  SubjectPublicKeyInfo, learned from the invite, and checks it before sending a
  byte, then again inside the protocol.
- Client authentication: Ed25519 over `"reflop-auth-v1\0" ‖ server nonce ‖
  server key fingerprint ‖ public key`. The nonce prevents replay; the
  fingerprint prevents relaying a signature to another server.
- Pairing: an HMAC with the invite's one-time secret over the same binding,
  plus a signature by the new key. An invite is consumed by its first use,
  successful or not.
- SSH mode: `restrict` and a forced command; inside the pipe the same
  signature check runs, and the key must belong to the identity in the forced
  command. `SSH_ORIGINAL_COMMAND` is ignored except for the one value the client sends.
- Revocation takes effect on the next request: keys are dropped and sessions killed.

## Safety rails

Rails protect the machine, not fairness, and are always on: disk free floor
max(10%, 20 GiB), host RAM floor 4 GiB, 4096 processes/threads per worker,
protocol sanity limits, and a peak-memory estimate against the GPU's total
memory. Quotas (device memory, sessions, wall clock per execute, FLOPs per
execute, storage, upload bandwidth) are optional and off by default.

## Client protection

| Threat | Mitigation |
|---|---|
| server impersonation, MITM | pinned key; a mismatch is a hard failure |
| deserialization attacks | protobuf and raw tensor bytes only; safetensors for checkpoints |
| size bombs | frame size caps on every response |
| path injection | the client never builds a path from server data |
| terminal injection | ANSI, OSC, C0/C1 and bidi controls removed from every server string |
| server-initiated actions | events are fixed-schema and only printed |
| information leaks | no environment, paths, hostnames, source or telemetry are sent |
| key theft | macOS Keychain, or a 0600 file the client refuses to use if others can read it |

## Residual risks

| Risk | Status |
|---|---|
| Bugs in whitelisted PyTorch, CUDA or driver code | reachable by design. Validation and bounds checks narrow the inputs; the sandbox limits what a successful exploit gains. Cannot be reduced to zero |
| Meta kernels disagreeing with real kernels | outputs that later operators consume are checked against the validated shape and dtype at run time; a mismatch is an error. Unused auxiliary outputs may differ (seen with `_embedding_bag`) |
| Timing side channels between Guests on one GPU | out of scope |
| A Guest exhausting GPU memory for others | by design without quotas; others get `OUT_OF_MEMORY`, nothing crashes. PyTorch's OOM text is trimmed so it does not reveal other tenants' processes |
| Escape at Level 0 lands in the Owner's account | Level 1 moves it to an empty account |
| The Owner sees Guests' data | by design (non-goal N1) |

## Known limitations

1. The server sees Guest data and models in plaintext. reFLOP protects the
   Owner's machine from Guests; it does not protect Guests' data from the Owner.
2. Bugs in whitelisted kernels (PyTorch, CUDA libraries, the GPU driver) remain
   an attack surface. The sandbox limits the blast radius; it cannot make it zero.
3. No arbitrary CPU programs, no custom CUDA or Triton kernels, no server-side
   media decoding.
4. Results can differ bitwise from local runs (TF32, non-deterministic GPU kernels).
5. GPU timing side channels between Guests are out of scope.
6. Killing a worker (watchdog, rail, or the Owner) loses that session's state.
   Long jobs should fetch or snapshot periodically.
7. At Level 0 a sandbox escape would land in the Owner's account. Level 1 is stronger.
8. Without quotas, Guests can starve each other of GPU memory. That is by
   design; contention produces errors, not damage.
9. `MemoryDenyWriteExecute` cannot be set on the Level 1 unit: it is inherited
   by workers, and CUDA and PyTorch need writable and executable mappings.
