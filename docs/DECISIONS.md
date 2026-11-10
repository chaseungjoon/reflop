# reFLOP design decisions

Architecture decision records (PLAN.md rule R3). Each record states the
context, the decision, the alternatives considered, and the consequences.
Deviations from PLAN.md are marked **Deviation** and explain why the plan's
reason did not hold.

| ADR | Decision |
|---|---|
| [001](#adr-001-flopir-as-the-wire-format) | FlopIR: a loop-free DAG over opset entries, protobuf-encoded |
| [002](#adr-002-capture-strategy) | Lazy dispatch subclass; `reflop.capture` instead of `torch.compile` |
| [003](#adr-003-opset-v1-adjustments) | Opset v1 as generated from PyTorch 2.15.0 |
| [004](#adr-004-sandbox-implementation) | Landlock before exec, seccomp after warm-up, fds for data |
| [005](#adr-005-framed-protobuf-instead-of-grpc) | Framed protobuf over TLS or stdio, not gRPC |
| [006](#adr-006-authentication-and-pairing) | Pinned SPKI, Ed25519 challenge-response, HMAC pairing |
| [007](#adr-007-reuse-of-recom-code) | Reuse of recom code, relicensed by its sole author |
| [008](#adr-008-the-pinned-runtime) | python-build-standalone plus pinned wheels, no triton |
| [009](#adr-009-safety-rail-defaults) | Rail defaults: disk max(10%, 20 GiB), host RAM 4 GiB |
| [010](#adr-010-distribution-and-updates) | One `install.sh`; `--server` for reflopd |
| [011](#adr-011-node-argument-encoding) | Positional, fully explicit node arguments |
| [012](#adr-012-data-dependent-shapes) | Two-phase DDS via `nonzero_count` + `nonzero_static` |
| [013](#adr-013-level-1-unit-without-memorydenywriteexecute) | No `MemoryDenyWriteExecute` on the Level 1 unit |
| [014](#adr-014-performance-without-weakening-checks) | Meta cache, capture, proven index ranges, no per-execute sync |
| [015](#adr-015-one-autograd-visible-device) | All reflop tensors report `reflop:0` |
| [016](#adr-016-cgroups-only-in-reflopd-owned-units) | cgroups only inside a reflopd-owned scope or unit |
| [017](#adr-017-libreflop-in-go) | The C API is a Go `c-shared` library |
| [018](#adr-018-reflopd-sandboxes-itself-by-re-exec) | reflopd applies its own sandbox by re-exec |
| [019](#adr-019-file-backed-state-with-change-detection) | File-backed identity state, reloaded on change |
| [020](#adr-020-oom-messages-are-trimmed) | OOM messages are trimmed |

## ADR-001: FlopIR as the wire format

**Context.** The server must accept computation without accepting code (I1),
estimate its cost statically, and later accept frontends other than PyTorch.

**Decision.** A Graph is a list of inputs, a list of nodes in topological order
(each `ValueRef` points strictly backward, so cycles cannot be expressed), and
outputs with declared metadata. Static shapes only. The server computes the
graph hash itself: SHA-256 over `"reflop-graph-v1\0"`, the opset version and the
deterministic protobuf encoding.

**Alternatives.** TorchScript or FX pickles (code, rejected by I1); StableHLO
directly (no PyTorch-native kernel mapping, more work for no v1 benefit).

**Consequences.** Validation is linear in the node count. A StableHLO frontend
can later lower to the same opset. Data-dependent control flow stays on the
client.

## ADR-002: Capture strategy

**Context.** PLAN.md §11.1 chose a lazy `__torch_dispatch__` wrapper subclass
(A) as the primary path and a `torch.compile` backend (B) as an accelerator for M4.

**Decision.** A is implemented: `RemoteTensor` is a wrapper subclass on the
PrivateUse1 device renamed to `reflop`, set up with PyTorch's Python-backend
helper so that `model.to()`, `nn.Parameter`, autograd, foreach and fused
optimizers, and autocast work. Views share a Storage and record a view chain;
in-place updates are functionalized with explicit inverse scatters.
**Deviation:** B is replaced by `reflop.capture`, a static capture-and-replay
of a whole step, verified over two eager calls.

**Alternatives.** Spike S2 (`spikes/s2_compile.py`) shows that TorchDynamo
cannot fake-ify `RemoteTensor` (it is not a traceable subclass: it has no inner
tensor to flatten). Making it traceable would require pretending its inner meta
tensor is the data, which AOTAutograd would then compile against.

**Consequences.** Eager mode costs about 20–30 µs per operation on the client.
`reflop.capture` brings ResNet-18 at CIFAR shapes to 74% of local-GPU
throughput (the §17.6 target is 70%). Steps whose Python scalars drift are
detected and refused, so capture cannot silently freeze values.

## ADR-003: Opset v1 adjustments

**Context.** PLAN.md §8 listed operators from the PyTorch 2.15 documentation and
asked M0 to verify every name (R9). Spike S5 is `codegen/bootstrap_opset.py`;
its census is `docs/opset_census.json`.

**Decision.**

- The pinned PyTorch 2.15.0 tags 189 overloads `core` (the plan expected 193).
  Every §8.4 name resolved. `avg_pool1d`, `adaptive_avg_pool1d`,
  `sym_is_contiguous` and `resize_` are listed in §8.4 but not tagged core; they
  keep their §8.4 disposition.
- 12 Tier 1 names do not exist in 2.15 (the `.vec` variants of the upsample
  backward ops, and `masked_fill_`) and were dropped.
- 39 `_foreach_*_out` overloads that a wildcard matched mutate their arguments;
  the wire is functional, so they were dropped.
- **Deviation:** each opset entry is one fully qualified overload, so the node
  carries `op_id` only; the plan's separate `overload_id` would be redundant.
- `nonzero` is marked CLIENT and DDS (see ADR-012); `nonzero_static` was added.
- Every `str` schema argument became an enum: `side`, `mode`, `driver` and
  `UPLO` were added to the plan's list.
- `int[N]` in a schema is PyTorch's broadcast hint, not a fixed length; V3 caps
  list length and meta inference applies each op's real rule.
- V3 range-checks every dim-like argument (`dim`, `dim0`, `start_dim`, …)
  against the first input's rank, because some meta kernels (`_softmax`) do not.
- Workers advertise only operators that have a kernel (or a composite) for their
  device: `_thnn_fused_lstm_cell` exists only for CUDA. The CPU device enables
  569 operators, CUDA 580.

**Consequences.** 582 entries: Tier 0 193, Tier 1 360 (including 11
reflop-native ops), Tier 2 29. `docs/OPSET.md` is generated.

## ADR-004: Sandbox implementation

**Context.** Spike S6: a warm worker must end up under Landlock and seccomp on
Ubuntu 24.04 defaults (unprivileged user namespaces restricted by AppArmor)
and still run CUDA. Landlock before ABI 8 restricts only the calling thread,
and PyTorch starts threads during initialization.

**Decision.**

- `reflopd __sandbox-exec` (a re-exec of reflopd) sets `no_new_privs`, applies
  the worker's Landlock domain on a locked thread, closes inherited fds, sets
  rlimits and `exec`s the interpreter, so every worker thread starts confined.
  The plan applied Landlock after warm-up; applying it before `exec` covers
  every thread on every ABI and leaves no window.
- After warm-up the worker installs a seccomp allowlist with
  `SECCOMP_FILTER_FLAG_TSYNC`. reflopd builds the BPF program (it needs the
  worker's pid to restrict signals) and sends it in `WorkerInit`. The worker
  calls `seccomp(2)` through `ctypes` in `sandbox.py`, the only worker module
  the static check allows to import ctypes.
- New sockets of every family are denied, not just non-`AF_UNIX`: a worker
  could otherwise `connect` to the Owner's D-Bus or ssh-agent socket, which
  Landlock does not cover before ABI 9 (`RESOLVE_UNIX`); this kernel reports ABI 8.
- Datasets and snapshots reach the worker as file descriptors over a second
  `SOCK_SEQPACKET` socketpair. The worker has no filesystem access to the store
  at all, which is tighter than the plan's read-only store rule.
- Found on the RTX 5070: the CUDA driver fails to initialize (error 304) unless
  it can write `/proc/self/task/<tid>/comm`. Workers may read and write their
  own `/proc/<pid>` and nothing else in `/proc` beyond a few read-only files.
- Workers are started via `/proc/self/exe`: reflopd's Landlock rule covers the
  inode it started from, which survives replacing the binary on disk.
- User and network namespaces are added when the kernel allows them.
- Isolation probes are compiled in only with `-tags reflop_probes`; release
  builds contain no probe code.

**Alternatives.** bubblewrap (depends on user namespaces, often restricted);
libseccomp (a C dependency); Landlock inside the worker after warm-up
(misses threads before ABI 8).

**Consequences.** 32 isolation probes pass on Ubuntu 24.04, kernel 7.0,
Landlock ABI 8. The seccomp filter is a linear allowlist with hot syscalls
first.

## ADR-005: Framed protobuf instead of gRPC

**Context.** PLAN.md §5 names gRPC over TLS 1.3, plus an SSH stdio mode.

**Decision.** **Deviation:** every message is a `uint32` big-endian length and a
protobuf `Envelope`, over TLS 1.3 in daemon mode and over the SSH pipe in SSH
mode. Requests carry a client-chosen id; streams send several frames with the
same id and mark the last one.

**Alternatives.** gRPC: Python's grpcio cannot run over an arbitrary byte
stream such as an `ssh` subprocess, so SSH mode would need a second protocol
stack; it would also add HTTP/2 to the attack surface.

**Consequences.** One protocol implementation per side, the same for both
transports. Pipelining comes from request ids. Strict decoding (unknown fields
and enum values rejected at every depth) is ours in both Go and Python.

## ADR-006: Authentication and pairing

**Context.** PLAN.md §15.2 left open plain HMAC over the pinned channel versus a
PAKE such as SPAKE2.

**Decision.** The invite carries the server's SPKI fingerprint. The client
pins it at the TLS layer and again inside the protocol. Pairing proves the
invite secret with an HMAC over `"reflop-pair-v1\0" ‖ nonce ‖ fingerprint ‖
invite id ‖ public key` and proves key possession with a signature over the
same bytes. Afterwards, clients authenticate with an Ed25519 signature over the
server nonce and fingerprint. The same check runs inside SSH pipes.

**Alternatives.** SPAKE2: its benefit is resisting offline guessing of a
low-entropy password; the invite secret is 256 random bits and the channel is
already authenticated by the pinned key. mTLS client certificates: Python's
`ssl` cannot verify against an application key store without a CA, and SSH
mode would need a separate scheme.

**Consequences.** An invite is consumed by its first attempt, successful or
not. A signature cannot be replayed (nonce) or relayed to another server
(fingerprint).

## ADR-007: Reuse of recom code

**Context.** PLAN.md §24 forbids copying code from recom (GPL-3.0) unless the
human confirms sole authorship.

**Decision.** The author confirmed being recom's sole author and that recom
code may be used under Apache-2.0. Adapted: `install.sh` (download, checksum,
install logic), `internal/selfupdate`, `internal/buildinfo`, and the
`update` command.

**Consequences.** `NOTICE` records the relicensing.

## ADR-008: The pinned runtime

**Context.** The worker needs PyTorch, but the Owner should not install Python
packages (G4), and Guests must never influence what is installed.

**Decision.** `codegen/lock_runtime.py` resolves each variant with `uv`, picks
the exact wheel per architecture, and writes URLs and SHA-256 sums to
`lock.json`, which is embedded in reflopd. `reflopd up` downloads a
python-build-standalone CPython (3.12.16) and the wheels, verifies each, and
unpacks them with a Go wheel installer into a read-only tree. Variants:

| Variant | PyTorch | Selected when |
|---|---|---|
| `cu132` | 2.15.0 + CUDA 13.2 | driver ≥ 580.65 and every exposed GPU in compute capability 7.5–12.1 |
| `cu126` | 2.14.1 + CUDA 12.6 | driver ≥ 525.60 and compute capability 5.0–9.0 |
| `cpu` | 2.15.0 CPU | otherwise |

`triton` is left out of every variant: the worker never compiles kernels, so
the JIT compiler is not shipped. `numpy` is not installed either; the worker
avoids it. The worker package is unpacked separately under
`runtime/workers/<hash>/`, so a reflopd upgrade does not re-extract the runtime.

**Alternatives.** Conda or a system Python (Owner-managed, unpinned);
containers (heavier setup, PLAN.md §4.1 option D).

**Consequences.** About 3.5 GB downloaded for CUDA variants, cached and
resumable. Missing overloads in the legacy 2.14.1 runtime are disabled, not
looked up dynamically.

## ADR-009: Safety rail defaults

**Context.** PLAN.md O6 asked to confirm rail defaults.

**Decision.** The Owner confirmed: disk free floor max(10%, 20 GiB); host RAM
floor 4 GiB (capped at half of physical RAM on small machines); 4096 processes
per worker.

**Consequences.** `reflopd rails set` changes them; quotas remain off by default.

## ADR-010: Distribution and updates

**Context.** The Owner wants installation by `curl … | sh` and an `update`
command. The client is a Python package that must be importable from the
Guest's own training environment.

**Decision.** One `install.sh`. By default it installs the client wheel from the
GitHub release into the active virtual environment, else as a `uv` tool, else
into a private venv with a `reflop` symlink, and prints how to add the library
to a project. `--server` installs the `reflopd` binary (Linux only). Both
verify `checksums.txt`. `reflopd update` and `reflop update` fetch and verify
the same release assets.

**Alternatives.** PyPI (planned later; not published without the Owner's
go-ahead); auto-detecting the role (surprising on Linux laptops with GPUs).

**Consequences.** Nothing installs until a release is published.

## ADR-011: Node argument encoding

**Context.** PLAN.md §7.1 sketched inputs plus keyed attributes.

**Decision.** **Deviation:** a node's `args` are the schema's arguments in
schema order, keyword-only ones included, with defaults filled in by the
client. Every argument type has one encoding. Python floats in `float` or
`Scalar` positions become scalar graph inputs (lifted) unless the op allocates
from them, so learning rates and bias corrections do not change the graph hash.
Python numbers in `Tensor` positions (PyTorch's wrapped numbers) stay numbers,
so the server applies the same weak-scalar type promotion. CPU tensors become
graph inputs bound inline (≤ 1 MiB) or by a temporary upload, never constants.

**Consequences.** Equal programs produce byte-identical graphs. Validation
needs no keyword matching.

## ADR-012: Data-dependent shapes

**Context.** `nonzero` and boolean-mask indexing have data-dependent output
shapes; v1 graphs are static.

**Decision.** The client asks the server for `reflop.nonzero_count`, fetches the
count, then records `aten.nonzero_static` with that size. Boolean indices are
expanded this way into integer indices. Other DDS operators (`unique`) fall
back to the client in `"auto"` placement.

**Consequences.** Two round trips per data-dependent op. The server only ever
executes static shapes, and boolean index tensors are rejected server-side.

## ADR-013: Level 1 unit without MemoryDenyWriteExecute

**Context.** PLAN.md §9.8 asks for `MemoryDenyWriteExecute=yes` for reflopd only.

**Decision.** **Deviation:** not set. systemd implements it as a seccomp filter
inherited by every child, and workers (CUDA, PyTorch) need writable and
executable mappings.

**Consequences.** reflopd keeps its own seccomp denylist (ADR-018); the
remaining systemd hardening applies.

## ADR-014: Performance without weakening checks

**Context.** The §17.6 target is ≥ 70% of local-GPU throughput for ResNet-18 on
CIFAR-10. The first measurement was 5.8%. Rule R7: per-execute checks are never
skipped.

**Decision.**

- Client: meta inference results are cached per (op, input metadata, int and
  bool arguments) for non-view, non-ALLOC, non-RNG ops (2.5× in eager mode).
- `reflop.capture` (ADR-002).
- Worker: validated programs are compiled into flat per-node plans once; no
  device synchronization at the end of an execute.
- Bounds checks (I4) still run before every IDX kernel, but server tensors are
  immutable (functional semantics), so a tensor's value range, once known,
  stays valid. Ranges come from the host copy at upload, a one-time scan of
  dataset tensors, propagation through value-preserving ops (gathers, views,
  same-dtype copies) and earlier measurements. Only unknown ranges need a device sync.

**Consequences.** ResNet-18 at batch 128: 4.2 ms locally, 5.6 ms through reflopd
with capture (74%), 23 ms eager.

## ADR-015: One autograd-visible device

**Context.** PyTorch's Python-backend device guard reports one device, and the
autograd engine asserts when a tensor's device index exceeds it.

**Decision.** Every `RemoteTensor` reports `reflop:0`. Each tensor's server
session (GPU, server CPU) is tracked in its Storage. Factories inside a
dispatched op use the dispatching session; `reflop.device(i)` selects session
i for new tensors. An op mixing two sessions raises.

**Consequences.** `tensor.device` does not distinguish sessions. Moving between
sessions goes through the client.

## ADR-016: cgroups only in reflopd-owned units

**Context.** Worker rails need a delegated cgroup v2 subtree. A shell's own
scope is writable by the user but holds other processes; enabling controllers
there fails, and changing it would affect the terminal.

**Decision.** reflopd manages cgroups only when it runs in a unit or scope it
owns (`REFLOP_SCOPED`, set by its user unit and by its `systemd-run --scope
-p Delegate=yes` re-exec). Elsewhere it uses `RLIMIT_NPROC` and memory polling.

**Consequences.** `reflopd serve` from a terminal still gets kernel-enforced
rails when a systemd user manager is available.

## ADR-017: libreflop in Go

**Context.** M7 asks for a C ABI and a C++ wrapper.

**Decision.** `clib/go` builds with `-buildmode=c-shared` and reuses
`pkg/reflopv1`, `pkg/wire` and `pkg/opset`. Objects cross the ABI as integer
handles, not pointers. `reflop.h` is hand-written and const-correct;
`reflop.hpp` wraps it with RAII.

**Alternatives.** A C implementation would need its own protobuf encoder,
strict decoder, TLS pinning and key parsing, all in a memory-unsafe language.

**Consequences.** One shared library (about 13 MB) with no runtime dependencies
beyond libc. It reads the Python client's config and key file.

## ADR-018: reflopd sandboxes itself by re-exec

**Context.** PLAN.md §9.8: at Level 0, reflopd applies Landlock and seccomp to
itself. Go programs are multi-threaded from the start.

**Decision.** `reflopd serve` applies a Landlock domain (system paths
read-only, its directories read-write, TCP bind only on its listening ports,
no outbound TCP) and a seccomp denylist on a locked thread, then re-executes
itself. Workers inherit this outer domain and add their own.

**Consequences.** The daemon's outer domain must allow what workers need,
including writes to `/proc` (each worker's inner domain then limits this to its
own `/proc/<pid>`).

## ADR-019: File-backed state with change detection

**Context.** Owner commands (`invite`, `revoke`, `quota set`) are separate
processes; the daemon must see their changes.

**Decision.** Identities and invites live in one 0600 JSON file written
atomically. The daemon reloads it whenever its modification time or size
changes; `revoke` also tells the running daemon to kill sessions at once.

## ADR-020: OOM messages are trimmed

**Context.** PyTorch's out-of-memory message lists other processes on the GPU
and their memory use, which would tell one Guest about other tenants.

**Decision.** Workers return only the failed allocation size and a note that
the device is shared.
