# reFLOP design decisions

Architecture decision records ([PLAN.md](PLAN.md) rule R3). Each record states the
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
| [015](#adr-015-one-autograd-visible-device) | All reFLOP tensors report `reflop:0` |
| [016](#adr-016-cgroups-only-in-reflopd-owned-units) | cgroups only inside a reflopd-owned scope or unit |
| [017](#adr-017-libreflop-in-go) | The C API is a Go `c-shared` library |
| [018](#adr-018-reflopd-sandboxes-itself-by-re-exec) | reflopd applies its own sandbox by re-exec |
| [019](#adr-019-file-backed-state-with-change-detection) | File-backed identity state, reloaded on change |
| [020](#adr-020-oom-messages-are-trimmed) | OOM messages are trimmed |
| [021](#adr-021-the-client-on-pypi-uploaded-from-the-signed-release) | The client on PyPI, uploaded from the signed release |
| [022](#adr-022-only-an-oom-while-running-ends-a-detached-session) | Only an OOM while running ends a detached session |
| [023](#adr-023-a-gpu-that-is-full-at-startup-is-served-once-a-probe-succeeds) | A GPU that is full at startup is served once a probe succeeds |
| [024](#adr-024-deleting-a-revoked-identity) | Deleting a revoked identity |
| [025](#adr-025-unpairing-on-both-sides) | Unpairing on both sides |
| [026](#adr-026-showing-sshs-stderr-in-connection-errors) | Showing ssh's stderr in connection errors |
| [027](#adr-027-reflop-run-keeps-openmp-on-one-thread-on-macos) | `reflop run` keeps OpenMP on one thread on macOS |
| [028](#adr-028-snapshots-are-written-durably-and-restored-all-or-nothing) | Snapshots are written durably and restored all or nothing |
| [029](#adr-029-guest-checkpoints-owner-opt-in-a-5-minute-floor-recovery-without-a-redo-log) | Guest checkpoints: Owner opt-in, a 5-minute floor, recovery without a redo log |
| [030](#adr-030-pause---now-suspends-gpu-workers) | `pause --now` suspends GPU workers |

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
reFLOP-native ops), Tier 2 29. `docs/OPSET.md` is generated.

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

**Alternatives.** PyPI (adopted for the client in ADR-021); auto-detecting the
role (surprising on Linux laptops with GPUs).

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

## ADR-021: The client on PyPI, uploaded from the signed release

**Context.** The client is a library the Guest imports from their own
project. `install.sh` usually put it in a separate environment (a `uv` tool
or a private venv), so the Guest needed a second install, from a release URL,
to `import reflop`, and `reflop update` updated only one of the two copies.

**Decision.** Publish the `reflop` package on PyPI, so `uv add reflop` or
`pip install reflop` puts the library and the `reflop` command in the
project's environment.
The release builds a wheel and an sdist; both are in the signed
`checksums.txt`. Publishing the draft (`scripts/sign-release.sh`) starts
`.github/workflows/pypi.yml`, which downloads those files from the release,
checks the signature against the key in `install.sh` and each SHA-256, and
uploads the same bytes through PyPI Trusted Publishing (no token on GitHub),
from the `pypi` environment. `reflopd` stays off PyPI: it is a Linux binary
that needs no Python and updates itself.

**Alternatives.** Building on upload (the PyPI file would differ from the
signed one); a PyPI API token as a repository secret; a separate package that
ships `reflopd` in platform wheels.

**Consequences.** `pip` does not check the offline signature, so for PyPI
installs the trust root is the PyPI project and this repository's Actions.
`install.sh` and `reflop update` still check the signature locally. In a uv
project `reflop update` upgrades `uv.lock` and refuses locked files whose
hashes are not in the signed `checksums.txt`; installing a wheel there would
be undone by the next `uv sync`. `reflopd update` replaces only the `reflopd`
binary, so the two updaters never touch each other's files.

## ADR-022: Only an OOM while running ends a detached session

**Context.** A session whose client went away keeps its worker, and that
worker's GPU memory, for `[session] ttl` so the Guest can resume. When a GPU
session fails with `OUT_OF_MEMORY`, reflopd ends the oldest detached session
on that GPU, which may belong to another Guest. Since 0.1.5 a graph larger
than the whole device is also reported as `OUT_OF_MEMORY` (at
`STAGE_V5_SAFETY`), so clients raise `torch.OutOfMemoryError`. That graph is
rejected before anything is allocated, so any Guest could end every detached
session on the GPU by registering such graphs in a loop, at no cost.

**Decision.** reflopd ends a detached session only for an `OUT_OF_MEMORY`
error raised while the worker ran: at `STAGE_E1_BINDINGS`,
`STAGE_E2_RUNTIME`, `STAGE_E3_EXECUTE`, or with no stage (an allocation
outside a graph, such as an upload). An OOM at a validation stage, or at a
stage reflopd does not know, never evicts. Starting a session still evicts
when the worker cannot get GPU memory.

**Alternatives.** Report the oversized graph as `RESOURCE_LIMIT` again (breaks
batch-size fallbacks that catch `torch.OutOfMemoryError`); a separate error
code for "larger than the device" (a protocol change for the same result).

**Consequences.** A Guest who really fills the GPU can still end detached
sessions, one per failed allocation; that is the trade the eviction was made
for. The worker's error stage is now part of the eviction decision, and a
corpus test pins the stage of the oversized-graph error.

## ADR-023: A GPU that is full at startup is served once a probe succeeds

**Context.** At startup reflopd probes every device with a sandboxed worker
to learn its operators and capabilities. When other programs held all of a
GPU's memory (on the reference machine, a leaking login-screen
`gnome-shell` held 11.4 of 12 GiB), the probe worker could not create a
CUDA context, and the self-test and the probe failed. reflopd then did not
start at all, not even for the CPU.

**Decision.** A GPU whose self-test or probe worker fails only with a CUDA
out-of-memory error is skipped, not fatal, as long as another device passes
the self-test. Until a probe succeeds the GPU is not served or advertised:
an unprobed slot has no operator table, and workers started without one
would not get the Owner's operator filter. The monitor probes it again every
30 seconds and publishes it once a probe works. The set of served slots is
replaced as a whole (an atomic pointer), so connections never see a slot
whose fields are still being filled in. `reflopd status` lists the GPU as
not served, and `reflopd selftest` exits 1 when it skipped a device. Any
other probe or self-test error is still fatal.

**Alternatives.** Keep failing and ask the Owner to free the GPU first (the
daemon would not come back on its own after a reboot with the desktop
running); serve the unprobed GPU with the worker's default operator table
(fails open); probe lazily when a Guest opens a session (a Guest request
would start probes).

**Consequences.** A Guest that connects while the GPU is full sees only the
other devices and has to reconnect to see the GPU later. Probe workers and
self-test workers no longer stream their stderr to the log; a worker that
fails carries its last lines in the error.

## ADR-024: Deleting a revoked identity

**Context.** `reflopd revoke` keeps the identity in `identities.json`, marked
revoked and without keys, so its name cannot be invited again. Owners had no
way to remove revoked Guests from `reflopd users`, and their datasets and
snapshots stayed on disk where nobody could reach them.

**Decision.** `reflopd users delete <name>` deletes a revoked identity. It
removes the identity and its invites from the store, its directories under
`store/` and `snapshots/` (paths built from the identity's 32-hex id, which is
checked first), its quotas in `settings.json`, and any SSH forced-command
lines that still name it. The name can then be invited again. An active
identity is refused, in the CLI and in `auth.Store.Delete`, with a message to
revoke it first. The audit log is not changed.

**Alternatives.** Let `revoke --delete` do both steps at once (one typo
deletes an active Guest's data); keep the identity but hide revoked ones in
`reflopd users` (the data and the name stay taken); also strip the identity
from the audit log (the audit log is append-only).

**Consequences.** A new Guest invited under a deleted name gets a new
identity id and none of the old data, but shares the old name's history in
`reflopd audit --name` and `reflopd usage`. Quotas set for the name in
`reflopd.toml` are not edited; the command prints a note if one remains.

## ADR-025: Unpairing on both sides

**Context.** A machine has one client key, and reflopd registers a key to one
identity. `reflop servers remove` only deleted the local entry, so the key
stayed registered and every later `reflop pair` with that server failed with
"this key is already registered" (issue #6). In the other direction, after
`reflopd revoke` and `users delete` the Guest saw only "authentication
failed" and kept a stale entry, and pairing again added a second one.

**Decision.** Unpairing has a Guest half and an Owner half, done in either
order; after both, the two machines have forgotten each other and a new
invite pairs them as the first time did.

- `reflop unpair [server]` connects, authenticates and sends the new
  `Unpair` message, then removes the server from `servers.json` and its
  pinned SSH host keys. reflopd answers by revoking the calling identity
  (`auth.Store.GuestUnpair`: keys dropped, invites used up, sessions ended,
  the same effect as `reflopd revoke`), records `guest_unpaired` in the audit
  log, marks the identity, and closes the connection. `Unpair` is accepted only
  after authentication and acts only on the connection's own identity, so a
  Guest can give up its own access and nothing else.
- The local entry is removed whatever happens on the server: unreachable,
  already revoked, a different server key (nothing is sent then), or a
  reflopd too old to know `Unpair`. `--local` skips the connection. Each
  outcome says what the Owner still has to run.
- `reflopd unpair <name>` is `revoke` plus `users delete`, behind a
  confirmation prompt (or `--yes`; without a terminal it refuses).
- Each side is told when the other has gone first. `reflopd users` shows
  "unpaired by guest" and `reflopd status` and the daemon log name the
  command to run. A client whose pinned server rejects its key raises
  `NotPairedError` with "run `reflop unpair <server>`". `reflop pair` refuses
  a server this machine is already paired with, before the invite is used. A
  Pair whose key is still registered says, to the Guest and in the Owner's
  log and audit record, to run `reflopd unpair <name>`.

**Alternatives.** Leave the key valid after the Guest's `Unpair` and only
flag it for the Owner (an unused key stays live until the Owner acts); let
`reflop unpair` delete the identity's datasets on the server as well
(deleting data stays the Owner's decision); unpair locally only and let the
Owner find out at the next failed pairing (the Owner learns late, and the
invite is wasted); give the client a new key on unpair (one key serves every
paired server, so the others would break).

**Consequences.** Protocol messages `Unpair` (envelope field 33) and
`Unpaired` (122) are added. Identities gain an optional `guest_unpaired`
timestamp; a reflopd older than this change refuses to load an
`identities.json` that contains one. In SSH pipe mode the `authorized_keys`
line stays until the Owner deletes the identity, but the key no longer
authenticates inside the pipe. ADR-024 rejected `revoke --delete` because one
typo could delete an active Guest's data; `reflopd unpair` asks before it
deletes for that reason.

## ADR-026: Showing ssh's stderr in connection errors

**Context.** In SSH pipe mode the client read ssh's stderr and threw every
line away, because nothing from that stream is trusted. ssh puts the reason it
failed there: a changed host key, `Permission denied (publickey)`, a refused
connection, an unknown host name. reflopd's own errors on the server
(`this key can only be used by the reflop client`, a daemon that is not
running) arrive on the same stream. With all of it dropped, the Guest saw
"closed before the reflop handshake" or "connection to the server lost" and
nothing about why (issue #8).

**Decision.** The client keeps the last 8 lines ssh writes to stderr, each cut
to 300 bytes, decoded with replacement characters and passed through
`errors.sanitize`, the same filter as every other server string (I9). Memory
stays bounded however much the stream sends or however long a line is. The kept
text is appended, as `(ssh: ...)`, to the error raised when the connection
closes before the handshake, when it drops later (unless the client closed it),
and to the detail `reflop unpair` shows when it cannot reach the server. With
`REFLOP_DEBUG=1` each kept line is also logged as it arrives. Nothing parses the
text, branches on it or acts on it: it is display only. TLS mode has no such
stream and is unchanged.

**Alternatives.** Keep dropping it and tell users to run `ssh` by hand (the
failure is often intermittent or environment-specific, and users do not know
the options reflop passes); show it only with `REFLOP_DEBUG` (the first,
unreproduced failure is the one worth explaining); match known ssh messages
and print our own wording (parsing untrusted text, and ssh's wording varies
by version and locale).

**Consequences.** A malicious server, or anyone on its side of the ssh
connection, can choose up to about 2.4 KB of plain text that appears in a
Guest's error message, after the same sanitizing as error messages the server
already sends inside the protocol. It cannot move the cursor, rewrite earlier
output or hide text with bidirectional overrides. The text is labelled as ssh's,
so it does not read as reflop's own advice. A test corpus case
(`tests/security/corpus/test_ssh_stderr.py`) checks the sanitizing and the
bounds.

## ADR-027: `reflop run` keeps OpenMP on one thread on macOS

**Context.** On macOS the client crashed or hung unless the Guest set
`OMP_NUM_THREADS=1` (issue #7). The macOS PyTorch wheel ships its own
`libomp`. When a second OpenMP runtime is in the same process (scikit-learn,
xgboost, LightGBM and faiss wheels bundle one; conda and Homebrew install
their own), the runtimes fight over threads once both start a pool: `OMP:
Error #15`, a segfault, or a hang. With one thread no pool starts, so they
never collide. On Linux torch uses libgomp, several copies of which coexist,
and the Owner workstation and Linux CI never saw the failure. Until now the
workaround was documented, and a crash under `reflop run` on macOS printed the
workaround.

**Decision.** On macOS, `reflop run` sets `OMP_NUM_THREADS=1` in the
command's environment when the variable is not set, and says so on stderr when
it starts. A value the user set, including an empty one, is passed through.
Other platforms are unchanged. `import reflop` does not touch the variable:
a program started without `reflop run` keeps its threads, and the Guest guide
tells macOS users to set the variable themselves. `reflop doctor` lists the
OpenMP runtimes the installed packages ship (from their RECORD files) and the
ones loaded in its own process after `import torch` (`_dyld_get_image_name` on
macOS, `/proc/self/maps` on Linux), and names the package each came from. On
macOS a second loaded runtime is a failure; a second installed one is a
warning, since it is only loaded when a program imports that package. On
other platforms the runtimes are listed but not judged. A crash under
`reflop run` on macOS now points at `reflop doctor` when OpenMP was already on
one thread, and at `OMP_NUM_THREADS=1` when the user chose more.

**Alternatives.** Keep documenting the variable (Guests on macOS kept hitting
the failure before reading the docs). Set it on every platform (Linux Guests
never had the problem and would lose local threads for nothing). Set it, or
call `torch.set_num_threads(1)`, at `import reflop` (a library that changes
process-wide threading on import is surprising, `set_num_threads` covers only
torch's pool and not the second runtime, and the variable is read too late
once torch has started its pool). Set `KMP_DUPLICATE_LIB_OK=TRUE` (silences
Error #15 but keeps two runtimes running, which Intel documents as unsafe and
which can give wrong results).

**Consequences.** On macOS, operators that run on the Guest's CPU under
`reflop run` (local fallbacks, data loading and preprocessing in torch,
scikit-learn) run on one OpenMP thread. The math sent to the server is not
affected. Guests who want more threads set the variable. Embedded use without
`reflop run` still needs the variable on a Mac with a second runtime;
`reflop doctor` says so. The macOS CI job installs scikit-learn and runs a
program that starts both runtimes' pools around reflop tensors under
`reflop run` (`tests/client/test_openmp_clash.py`).

## ADR-028: Snapshots are written durably and restored all or nothing

**Context.** The Snapshot and Restore requests existed but no client used them
and no test covered them (Issue #2 review). A snapshot was reported as saved
without syncing the file or its directory, a failed rename was ignored, and
the disk rail was checked only after the whole file was written; the storage
quota was not checked at all. The worker built the whole file in memory to
write it and read the whole file to restore it. A restore created its tensors
one at a time, so a failure halfway left some of them behind; two equal
destination handles silently kept the last tensor; the writer accepted a key
named `__metadata__` that the readers then skipped.

**Decision.** A write has two phases. The worker takes references to the
tensors (the cut), computes the exact file size and sends it to reflopd
(`SnapshotCut`). reflopd reserves that size against the disk rail and the
identity's storage quota, counting every other write in progress, and answers
`SnapshotGo`, or refuses with the reason. The worker writes in a background
thread through two 64 MiB staging buffers. reflopd then syncs the file,
checks its size, renames it into place and syncs the directory; only then
does the client hear that it is saved, and any failure keeps the previous
file. A restore validates everything first: lengths, distinct handles and
keys, handle capacity, every key present, the free GPU memory. It allocates
and fills all tensors from bounded reads, and only then adds all handles at
once. Writer and readers (Go and Python) share one key rule: 1 to 256 bytes,
not `__metadata__`, not starting with `__reflop__/`, and `__metadata__` in a
file must map strings to strings. `DescribeSnapshot` returns a snapshot's keys
and metas, parsed by reflopd, so that `reflop.load` can choose handles. The
client gains `reflop.save`, `reflop.load`, `reflop.snapshots` and
`reflop.delete_snapshot`, behind the `snapshots` capability.

**Alternatives.** Keep the synchronous write (it blocks the worker for about
1.3 ms per MiB of state: 4.4 seconds for 3.3 GiB). Reserve an estimate before
the cut (the size is only known at the cut; an overestimate refuses a write
that fits). Let the worker sync the file (reflopd owns publication and the
directory, so it syncs both).

**Consequences.** A saved snapshot survives a crash or power loss, and a
failed write or restore leaves nothing behind. One write runs per session at
a time. The tensors of a snapshot need free GPU memory twice for a moment
while they are copied (ADR-029). The header limit (1 MiB, about 10,000
tensors) applies to the writer too. Tests: `TestPublishKeepsPreviousOnFailure`,
`TestReserveWriteQuota`, `TestSnapshotKeyRules`, and
`tests/security/corpus/test_snapshots.py`.

## ADR-029: Guest checkpoints: Owner opt-in, a 5-minute floor, recovery without a redo log

**Context.** Issue #2: a session's state lives only in its worker, so a crash,
the watchdog, a rail, a restart or `pause --now` loses all of it. The issue
suggested asynchronous checkpoints or replicated state. Measurements on an
RTX 5070 (`docs/fault_tolerance.md`) showed that a background checkpoint does
not slow training measurably (under 5% even back to back), that it holds up
to one extra copy of the state in GPU memory while it runs, and that its real
cost is disk writes: a Guest whose state fills a 12 GiB GPU writes 72 GiB per
hour at a 10-minute interval. A client redo log replayed after a restore gave
bit-identical parameters in every test, but only with deterministic numerics,
and it needs a client recovery state machine that reconnect does not have yet.

**Decision.** Checkpoints are off until the Owner runs `reflopd checkpoints on`.
The interval defaults to 10 minutes and has a hard floor of 5 minutes: config
validation, the CLI and the server's per-session rate check all enforce it.
Turning them on prints a red warning with the worst case for the machine's
GPUs. A Guest opts in with `reflop.device(checkpoint=True)`, gets the Owner's
interval or a longer one (a shorter one raises), and is warned in yellow that
the state's GPU memory doubles during a checkpoint. The client sends
`Checkpoint` at step boundaries when due; the server writes every live handle
under its hex id, the CPU and CUDA RNG state, and a small metadata entry (a
Guest tag of at most 4 KiB, a generation number, the time) under reserved
`__reflop__/` keys. The cut is exact because the worker serves requests in
order. The worker refuses the cut, with a "skipped" answer, when the GPU lacks
free memory for another copy of the state, and drops each reference as soon as
its tensor is copied. Each session keeps one checkpoint, replaced atomically.
`reflop.recover()` opens a new session and sends `RecoverSession`: the server
restores every handle under its old id and the RNG state, all or nothing, and
the client marks tensors created after the checkpoint as lost and registers
its graphs again on use. Ending a session on purpose (`reflopd kill`, revoke,
unpair, `CloseSession`) deletes its checkpoint, and a write that finishes
after that is never published, so such a session cannot come back; other ends
keep the checkpoint for 24 hours.

**Alternatives.** Replicated state: a standby worker doubles GPU memory and
compute and covers only a crash; streaming the state to the Guest breaks the
few-KB-per-step design; keeping it in reflopd puts Guest tensors in the daemon.
A client redo log in this release: deferred until deterministic numerics are
strict (`warn_only` today) and the client can reconnect with requests in
flight; without it a Guest loses at most one interval. Checkpoints on by
default: they write to the Owner's disk without the Owner choosing it. A
configurable checkpoint directory, for example on a hard disk: dropped, it adds
a path outside reflopd's own directories to its sandbox and slows checkpoints,
which then hold the extra GPU memory longer.

**Consequences.** A Guest that uses checkpoints loses at most one interval of
server-side work, in the same process. What it keeps on its own machine (a
plain Adam's step count, the data loader position, Python variables) is not
rewound; `capturable=True` optimizers keep their step count on the server.
Checkpoints count against the disk rail and the storage quota. The worker
needs its 128 MiB of pinned staging memory once it writes, and its write
channel takes a lock because the writer thread answers from the background.
Tests: `TestCheckpointNeedsTheOwner`, `TestCheckpointIntervalEnforced`,
`TestEndedOnPurposeIsNeverRevived`, `TestRecoverIsIdentityScoped`,
`TestCheckpointIntervalFloor`, `TestCheckpointsIntervalFloor`,
`tests/client/test_checkpoints.py` (including a recovery that is bit-identical
to an uninterrupted run) and the corpus cases.

## ADR-030: `pause --now` suspends GPU workers

**Context.** `pause --now` killed every worker, losing every Guest's session,
and auto-yield paused Guests without freeing their GPU memory. NVIDIA drivers
from 550 have a process checkpoint API (`cuCheckpointProcessLock`,
`cuCheckpointProcessCheckpoint`, `cuCheckpointProcessRestore`,
`cuCheckpointProcessUnlock`) that moves a process's GPU memory into its own
host memory and back. Tested on an RTX 5070 with driver 595: a fully
sandboxed worker (Landlock, seccomp, user and network namespaces), driven by a
helper inside reflopd's own sandbox, gave the GPU back in 0.31 s for 1.3 GiB
and resumed in 0.25 s with its data intact; an unsandboxed 8 GiB process took
1.4 s and 0.7 s.

**Decision.** `pause --now` suspends each GPU worker through a small helper:
an embedded Python script that reflopd runs with the pinned runtime's
interpreter, as a child that inherits reflopd's sandbox. It takes only an
operation and a pid, calls the four driver functions with a 10-second lock
timeout, and refuses any process whose parent is not its own parent or whose
command line is not the worker's; reflopd passes only its own workers' pids.
Before suspending, reflopd checks that the worker's GPU memory (from
nvidia-smi), plus an eighth, fits in host RAM above the RAM floor. A worker
that cannot be suspended (not enough RAM, a kernel still running at the
timeout, a driver without the API, a CPU session) is killed as before, and its
checkpoint stays recoverable. `reflopd resume` resumes workers before it
releases queued work. The watchdog does not count suspended time, and eviction
skips suspended sessions, which hold no GPU memory. `yield --auto --suspend`,
off by default, also suspends during auto-yield; there a worker that cannot be
suspended stays paused.

**Alternatives.** Checkpoint every session to disk and kill it (seconds to
minutes, and the disk writes of ADR-029 without the Owner choosing them).
Link libcuda into reflopd (reflopd is a static Go binary without cgo; a
helper keeps the daemon unchanged). Run the helper inside the worker (the
API restores from another process, and the worker's seccomp filter must not
grow).

**Consequences.** The Owner gets the GPU back within seconds without ending
sessions. Suspended memory sits in the worker's own host memory and counts
against the RAM rail. The helper is new code that can freeze processes, which
is why it accepts only reflop workers. The driver API needs no ptrace, so
reflopd's seccomp filter does not limit it. Guests see a "suspended" event and
their requests wait. Tests: `TestHelperRefusesOtherProcesses`,
`test_suspend_helper_refuses_non_workers`.

## ADR-031: The isolation probes run in CI and fail closed

**Context.** The isolation probes (PLAN.md §17.5) are the only test that a
sandboxed worker really cannot open sockets, read the Owner's files, exec,
fork, ptrace, mount or unshare. They need a test build and an installed
runtime, so they ran only by hand; CI only compiled them (issue #3). A
change to the sandbox, the seccomp filter or the supervisor could break
isolation with CI still green. A CI job that skips when the runner lacks
Landlock or seccomp would have the same problem after a runner kernel change.

**Decision.** `.github/workflows/isolation.yml` builds reflopd, installs the
pinned CPU runtime into a temporary `REFLOP_HOME` and runs the probes with
`REFLOP_TEST_STRICT=1`. Strict mode fails instead of skipping when there is
no runtime, no Landlock or no seccomp, or when the user namespace mode is not
the one the job asked for (`REFLOP_TEST_USERNS`). It makes each probe worker
require Landlock and seccomp, checks that the Landlock domain was applied,
and fails if a read or list target does not exist, because a probe that gets
ENOENT counts as blocked without testing the sandbox. The job sets the
namespace mode itself (`user.max_user_namespaces`) and runs twice, with user
namespaces off and on, since runner defaults vary and reflopd behaves
differently in each mode. It runs on pull requests that touch the sandbox,
supervisor, runtime, worker or the job itself, on every push to `main`, and
nightly. The probes stay in `server/worker/testprobes/`, because `go:embed`
cannot reach files outside the embedding package's directory;
`tests/security/isolation/README.md` points there.

**Alternatives.** Run on every pull request (a few minutes added to every
PR, also those that cannot affect isolation). Test only the runner's default
namespace mode (one of reflopd's two sandbox setups would never run). Move
the probes to `tests/security/isolation/` (reflopd would have to import a
package from `tests/`).

**Consequences.** A sandbox break is caught on the pull request when it
touches the sandbox paths, and otherwise on the push to `main` or by the next
nightly run. The job takes a few minutes per mode; the runtime download is
cached by `lock.json` hash and verified on every install. If GitHub's runners
ever lose Landlock or seccomp, the job turns red rather than passing. Tests:
`TestIsolationProbes`, `TestIsolationStrictChecks`.

## ADR-032: The worker's own meta kernel for `_ctc_loss_backward`, with exact input shapes

**Context.** Stage V4 runs every node on meta tensors and rejects an op that
has no meta kernel (I11). PyTorch 2.15 has none for `_ctc_loss_backward`, so
every graph with a CTC backward was rejected, and the client could not
compute its output shape either: training with `F.ctc_loss` never worked on
reFLOP, and the Guest's process aborted in `.backward()`. A meta kernel that
only returns the output shape would not be enough. PyTorch's backward
kernels read `grad`, `neg_log_likelihood` and `log_alpha` without checking
their shapes: on the CPU, a `log_alpha` shorter than the forward's crashes
the worker, and a short `grad` or `neg_log_likelihood` is read out of bounds
without an error. A graph can pass any tensors there, not only the forward's
outputs.

**Decision.** `meta.OVERRIDES` maps `aten._ctc_loss_backward.default` to a
meta function in the worker. The function returns an empty tensor with
`log_probs`' shape and dtype, and before that runs `ctc_backward_static`,
which the `ctc` validator also runs:

- `log_probs` is `(T, N, C)`, float32 or float64;
- the lengths are int lists of `N` entries, `0 <= input_length <= T`;
- 2-D targets are `(N, S)` with every `target_length <= S`; 1-D targets hold
  at least `sum(target_lengths)` elements;
- `grad` and `neg_log_likelihood` are exactly `[N]`, and `log_alpha` exactly
  `[N, T, 2 * max(target_lengths) + 1]`, all with `log_probs`' dtype.

The `.Tensor` overload (tensor lengths) stays rejected: the client turns tensor
lengths into int lists and sends `.default`. The client has its own meta
function, which only computes the output shape.

**Alternatives.** Register a Meta kernel through `torch.library` (it would
also change PyTorch for everything else in the process, and the check would be
less visible in an audit). Run the backward on the Guest's CPU (no server
change, but every step moves `log_alpha`, `N × T × (2S + 1)` values, both
ways). Leave CTC training unsupported.

**Consequences.** CTC training works. Since the shapes are part of the
registered graph and bindings must match them (E1), checking them once at
validation is enough; target values are still bounds-checked at every run by
`ctc_runtime`. Each later meta kernel the worker supplies goes into
`meta.OVERRIDES` with a check of the inputs its kernel reads without
checking. Tests: `test_ctc_backward_*` in `tests/security/corpus/test_corpus.py`
(every malformed graph is rejected at V4 and is accepted without the check),
`test_ctc_loss_with_tensor_lengths` in `tests/conformance/test_ops.py`.

## ADR-033: A read never returns data while an earlier request's failure is pending

**Context.** The client sends executes, uploads and frees without waiting,
and a failure is stored when its answer arrives and raised at the
next read. `Session.flush` checked for a stored failure only when it
started. A read waits for its answer, and by then the answer to an earlier
request that failed has arrived, but nobody looked again: the read returned
data that the failed request should have changed, with no error. In the
benchmark, a captured step whose replay failed returned the previous batch's
loss exactly, and the error came one call later. The in-process worker
answers synchronously, so no test saw it.

**Decision.** After every wait that hands data or state back to the program
(`flush` with returned values, `fetch_handles`, `reflop.save`,
`reflop.load`), the client raises a stored failure before returning. This
relies on the server answering a session's requests in order: the worker
handles them one at a time, reflopd forwards them in order, and its own
rejections (paused, quota) are sent at once, so they arrive earlier, never
later. Uploads that pass a pause go ahead of held requests but return no
data. The worker commits a graph's outputs only after every node has run,
so a failed execute leaves the handles as they were. A read therefore never
needs to wait for anything beyond its own answer. `repr` must not raise, so when a read inside it fails, the error is
kept (`Session.keep_unread`) and raised by the next read, or printed at exit.
The C API waits for every answer and was never affected.

**Alternatives.** Wait for every request in flight before returning a value
(no extra cost when answers are in order, but checkpoint answers come late
by design and would stall reads). Sequence numbers in answers so the client
can tell which failures precede a read (a protocol change for a guarantee
the transport already gives). Make executes synchronous (one round trip per
step; most of the throughput is gone).

**Consequences.** Reads after a failure raise it; each failure is raised
once, and the failed request leaves the server's tensors as they were. If
reflopd ever answers out of order, a read could again return data while an
earlier failure is still on the way; the ordering argument has to be checked
whenever the forwarding path changes (held requests, data plane). Tests:
`tests/e2e/test_async_errors.py` delivers the in-process worker's answers
from another thread after a delay, and runs against a real reflopd with
`REFLOP_E2E=1`.

## ADR-034: Placement never runs a writing or aliasing operation on copies

**Context.** `"local"` placement, and `"auto"` when its cost model prefers
the client, ran an operation with `_run_locally`: it fetched copies of the
reFLOP inputs and ran the operation on them. An operation that writes to an
argument (in place, `out=`) changed the copy, and a view of a reFLOP tensor
became an independent CPU copy, so writes through it went nowhere. Every one
of these silently lost the update. Under `"local"` that included the
optimizer step, so parameters never changed; under `"auto"`, the default, a
small reFLOP tensor updated from large client tensors (`scatter_add_`).

**Decision.** `_dispatch` takes the local shortcut only for operations that
neither write to an argument nor return a view of one (schema alias
annotations, `is_view`, the `inplace_view` tag; cached per operator). The
others keep their usual path: `_mutating` dispatches the functional form,
which may itself run locally under the same rules, and `_assign` writes the
result into the target (a reFLOP tensor receives the CPU result as a graph
input; a CPU target is copied into). Views stay lazy views of the reFLOP
tensor in every mode, so they alias it and a read fetches only the view.

**Alternatives.** Keep views as CPU copies under `"local"` (what that mode
promised for every operation) and only write back for in-place and `out=`
operations: writes through views would still be lost, and reading a slice
would keep fetching the whole base tensor. Upload every local result of a
writing operation from `_run_locally` itself: duplicates `_mutating`'s
functional-form and write-back logic, including its special cases.

**Consequences.** Under `"local"`, views of a reFLOP tensor are reFLOP tensors
(operations on them still return CPU tensors); this is the one visible change
for code that relied on the old behavior. In-place operations under
`"local"` run on the client and send their result back, so a writing loop
under `"local"` uploads each result. Graphs outside placement are unchanged
(ResNet-18 training registers the same graph before and after). Tests:
`tests/e2e/test_placement.py` (writes in every mode against PyTorch, the
cost model still choosing the client for the large scatter, views reading
only their part, training and gradient accumulation under `"local"`).

## ADR-035: Each worker gets its own memory limit below the shared one

**Context.** Workers shared one cgroup `memory.max` (the host-RAM pool above
the floor), and each worker leaf only had `pids.max` and `memory.oom.group`.
When the shared limit was reached, the kernel killed the largest worker, not
necessarily the one growing: one Guest could get another Guest's larger
session killed (issue #16). A worker was also told at spawn that its budget
was the whole pool, although other workers already used part of it, and its
own checks had gaps: each upload stream was checked alone while allocating
its whole buffer up front, finishing an upload copied the data twice more,
and a fetch held about four copies of the data.

**Decision.** The RAM rail (every 500 ms) sets each worker leaf's
`memory.max` to its own use plus the pool's free headroom, less a margin of
1/8 of the headroom (at least 32 MiB, never more than the headroom). The
margin is what makes it work: without it the worker's own limit and the
shared one coincide, and anything another worker adds before the next tick
makes the shared limit fail first; then the kernel kills the largest worker
again. A new worker starts with a limit and a budget of the free headroom.
A suspended worker has no limit of its own while suspend moves its GPU
memory into its host RAM; `suspendAll` already checks that against the
headroom. Inside the worker, uploads in progress are counted together
against the budget, the upload buffer becomes the tensor (CPU) or is copied
to the device once, and a fetch makes one host copy per tensor (`.cpu()`)
and builds its frames from it before sending.

**Alternatives.** A static per-worker limit equal to the whole pool (what a
worker is told at spawn): the pool can still be filled by one worker and the
kernel still picks the largest. A static share of the pool per worker: one
Guest on an idle machine could not use most of it, and lowering the share
below a worker's use when another worker starts would OOM-kill it. A larger
margin (half the headroom): a single large allocation would fail while the
pool has room. `memory.high` throttling: slows the worker instead of
bounding it, and the shared limit can still be reached.

**Consequences.** A worker that grows past what is free is OOM-killed
alone; the kernel test (`TestWorkerLimitsKillTheGrowerNotTheBiggest`, needs
a delegated scope) shows the old setup killing the innocent larger worker
and the new one killing the grower. Two workers that grow within the same
tick can still reach the shared limit together. One worker alone can take
7/8 of the free pool in one allocation and all of it step by step. Without a
delegated cgroup nothing changes (the rail still kills the largest worker
below the floor). Streaming a fetch from a host buffer filled with
`torch.frombuffer(...).copy_(gpu_tensor)` returned zeros now and then
through the in-process connection in the full e2e suite, while `.cpu()`
never did; the cause is not known (BUGS.md), so the fetch uses `.cpu()`.
Tests: `tests/security/corpus/test_worker_memory.py`, `cgroup_test.go`,
`workermem_test.go`.

## ADR-036: Keep the explicit default and seed every session

**Context.** Opening a CPU session through `placement("remote-cpu")` made it
the default. `reflop.load(name)` then restored into the CPU session, and
`reflop.seed(n)` left the GPU RNG untouched.

**Decision.** Only explicitly opened sessions enter the default session stack.
`reflop.seed(n)` reseeds all open sessions and remembers the seed for new
sessions, even when called before opening a session. The PyTorch seed hook
continues to reseed open sessions.

**Alternatives.** Adding a session argument to `reflop.seed` would let callers
reseed one session at a time. It would leave existing calls dependent on which
session opened last.

**Consequences.** Snapshot and checkpoint calls without a device index keep
using the explicit default when placement opens a CPU session. Opening a
CPU session explicitly still makes it the default. The server uses the
existing reseed request.

## ADR-037: Fetch large execute results in chunks

**Context.** Reading an unflushed tensor could put its entire contents in
an execute response. A tensor of 64 MiB already exceeds the default frame
limit once the response includes its metadata.

**Decision.** Give the combined inline results a budget of one quarter of
the server's advertised frame limit, capped by the client's frame ceiling.
When a result would exceed that budget, store it on the worker and use the
existing chunked fetch. Reuse the output's handle when the execute already
stores it. Otherwise allocate a temporary handle and free it when the read
finishes or fails. Wait for the execute response before sending the fetch
so validation and execution errors reach the caller.

**Alternatives.** Raising the maximum frame size would increase the maximum
memory consumed by a single response and leave a cutoff for larger tensors.
Fetching every result in chunks would add a round trip to scalar reads and
other small results.

**Consequences.** Small results still travel in the execute response.
Larger reads use the existing fetch size limits and chunk validation.
The change works with the existing worker and wire protocol.
