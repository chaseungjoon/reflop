# reFLOP design plan (remote FLOP)

> Every design decision comes with its reason. Do not "simplify" a decision unless you understand and can refute the reason. The invariants in §3 must never be weakened, not even temporarily.
>
> Names: project **reFLOP** in prose; server binary `reflopd`; server compute process `reflop-worker`; client Python package and CLI `reflop`.
> License: **Apache-2.0** (§24).

**Status.** This is the design reFLOP was built from. Milestones M0–M7 (§18) are implemented; M8 is optional and not started. Where the code differs from this plan, an ADR in [DECISIONS.md](DECISIONS.md) says why and is marked **Deviation**; the ADR wins. The main deviations:

| Plan | Implementation | ADR |
|---|---|---|
| `torch.compile` backend as the accelerator path (§11.1 B) | `reflop.capture`: static capture and replay of a whole step | [ADR-002](DECISIONS.md#adr-002-capture-strategy) |
| One opset entry per op name, overloads by `overload_id` (§7.1, §8.2) | One entry per fully qualified overload | [ADR-003](DECISIONS.md#adr-003-opset-v1-adjustments) |
| gRPC over TLS 1.3 (§4.4, §5, §6.2) | Length-framed protobuf over TLS 1.3 or the SSH pipe | [ADR-005](DECISIONS.md#adr-005-framed-protobuf-instead-of-grpc) |
| Node inputs plus keyed attributes (§7.1) | Positional, fully explicit node arguments | [ADR-011](DECISIONS.md#adr-011-node-argument-encoding) |
| `MemoryDenyWriteExecute=yes` on the Level 1 unit (§9.8) | Not set | [ADR-013](DECISIONS.md#adr-013-level-1-unit-without-memorydenywriteexecute) |
| `libreflop` C ABI (§18 M7), language unspecified | A Go `c-shared` library | [ADR-017](DECISIONS.md#adr-017-libreflop-in-go) |

---

## Table of contents

0. Agent working rules
1. Context, purpose, goals, non-goals
2. Threat model
3. Security invariants
4. Deployment model — what must run on the server, and why
5. Architecture overview
6. Execution model
7. FlopIR (the wire graph format)
8. Opset v1 (the operator whitelist)
9. Server design
10. Data plane (bootstrapped data)
11. Client design
12. Client protection
13. CPU work and dynamic placement
14. Compatibility: capability negotiation
15. Network, identity and pairing
16. CLI and Python API
17. Testing strategy
18. Milestones and acceptance criteria
19. Reference environment
20. Open questions
21. Known limitations
22. Glossary
23. Repository layout
24. Licensing and trademarks

---

## 0. Agent working rules

| # | Rule |
|---|---|
| R1 | Work milestone by milestone (§18). Do not write code for milestone N+1 until every acceptance criterion of milestone N passes. |
| R2 | Never violate an invariant in §3, even "temporarily" or "behind a flag". If a feature conflicts with an invariant, drop the feature, record it in `docs/DECISIONS.md`, and ask the human. |
| R3 | Record every design decision in `docs/DECISIONS.md` as an ADR (Context / Decision / Alternatives / Consequences). ADR numbers referenced in this plan (ADR-001 …) must exist once the decision is made. |
| R4 | If something is ambiguous, check §20 Open Questions first. If it is not there, add it and ask the human. Do not guess on anything security-relevant. |
| R5 | Adding an operator to the opset requires, in the same change: (1) `opset.yaml` entry, (2) validator, (3) cost function, (4) conformance test, (5) security test. See §8.8. |
| R6 | Server-side code is guarded by static analysis rules (§17.4) that enforce I1/I2. Never disable or add exceptions to those rules. |
| R7 | Never trade security for speed. Caching *validation results of an identical graph hash* is allowed (same hash ⇒ same structural validation result). Skipping *per-execution* checks (bindings, bounds, sizes) is never allowed. |
| R8 | All code, comments, commit messages and docs are in English. |
| R9 | Operator names in this plan were taken from the PyTorch 2.15 documentation and from PyTorch's operator registry as known at planning time. In M0, verify every name and overload against the pinned PyTorch version by introspection (`torch.ops.aten.*`, `torch.Tag.core`) and report mismatches in `docs/SPIKE_REPORT.md`. Do not invent operators. |

---

## 1. Context, purpose, goals, non-goals

### 1.1 Background

The author's earlier project `recom` (github.com/chaseungjoon/recom, GPL-3.0) rsyncs a local project to a remote Linux host and runs commands there over SSH. That is excellent when you trust the remote host *and* the remote host trusts you, but it hands the client a **shell** on the server: effectively remote code execution (RCE) by design.

### 1.2 Problem

Typical scenario: in a student or research team, **only one member (the Owner) has a GPU workstation.** Other members (Guests) need GPU time. With `recom` or plain SSH, a Guest, malicious or simply careless, can damage the Owner's machine with one bad command: deleting files, filling the disk, fork bombs, installing a malicious pip package, reading the Owner's private files, leaving background processes, and so on.

### 1.3 Purpose

reFLOP lets Guests use the Owner's GPU **as a calculator and nothing more.**

- The server executes **only whitelisted tensor operators** from a closed set, over data the Guest uploaded. Guest code never runs on the server.
- The Guest's program (Python training loop, control flow, preprocessing, logging) runs **on the Guest's own machine.** Only tensor math is delegated.
- The server does not know or care whether the client is Python, C or C++. It only sees an operator graph (IR) and tensors.
- No SSH shell and no OS accounts for Guests. An optional SSH transport mode exists, but it never yields a shell (§15.4).

### 1.4 Goals

| ID | Goal |
|---|---|
| G1 | Run PyTorch training and inference workloads on a remote GPU (first target: PyTorch; second: a C/C++ API). |
| G2 | The server executes only whitelisted numeric tensor operators. Client-supplied code can never execute (RCE is impossible by construction, not by policy). |
| G3 | The client is untrusted: assume it is malicious, buggy, or compromised. |
| G4 | **Zero per-project and per-client setup on the server.** No venv, pip, conda, compilers, or project files. The Owner performs a one-command, root-free bootstrap once (§4). |
| G5 | Data is uploaded once ("bootstrapped"). Per-training-step traffic after warm-up is in the kilobyte range. |
| G6 | The client needs no CUDA (e.g. an Apple Silicon MacBook works). |
| G7 | Client protection: the client is defended against a malicious or impersonated server (no reverse RCE, no path injection, no terminal injection). |
| G8 | No usage quotas by default. Owner and Guests negotiate usage among themselves. The software provides optional quota knobs and Owner controls (pause, kill), but never enables them by default. |
| G9 | Tensor-shaped CPU work can run on the server CPU. Everything else runs on the client, with automatic placement. |

### 1.5 Non-goals (v1)

| ID | Non-goal | Reason |
|---|---|---|
| N1 | Confidentiality of Guest data **from the server** | Threat model trusts the server (§2). The server sees tensors in plaintext. This is a scoping decision, not a bug. |
| N2 | Executing arbitrary CPU programs remotely | Sending an arbitrary program is, by definition, sending code (§13). Conflicts with G2. |
| N3 | Multi-GPU or multi-node training | v1 is single server, single GPU (multi-GPU servers may expose one device per session). |
| N4 | TensorFlow / JAX | v1 is PyTorch-first. The IR is designed to allow a StableHLO frontend later (ADR-001). |
| N5 | Running client-supplied CUDA/Triton/PTX kernels | A kernel is code. Conflicts with G2. |
| N6 | Production inference serving | This is a team training/experimentation tool. |
| N7 | Decoding media on the server (JPEG, PNG, MP4, audio, text tokenization) | Media decoders are a classic source of memory-safety bugs. Decoding happens on the client (§10). |

### 1.6 Deliberate deviations from the original idea (do not revert these)

| Original idea | What this plan does instead | Why |
|---|---|---|
| "Only floating-point operations" | "Only **whitelisted numeric tensor operators**", including integer and bool dtypes | Real training needs integer tensors (token ids, indices, labels) and bool masks. The security property comes from "data, not code, plus a closed operator set", not from the dtype being float. |
| "On init, pull the server's CUDA version and build payloads to match it" | **Capability negotiation** (§14) | The payload is an operator graph, not CUDA code, so the server's CUDA version is irrelevant to the client. Building CUDA-version-specific binaries on the client and sending them would itself be code shipping, which violates G2. |
| "CPU work should run on the server too" | Tensor-shaped CPU work: yes, on the server CPU via the same IR. Arbitrary CPU programs: no, they run on the client (§13). | There is no general way to verify that an arbitrary program is safe (Rice's theorem). A general-purpose CPU instruction set *is* code. |
| "Nothing should be installed on the server" | Nothing per project or per client. The Owner runs a **one-command, rootless bootstrap** of a single binary once (§4). | The enforcement point must live where the attacker cannot modify it: on the server, installed by the Owner. Anything enforced only on the client can be bypassed by the client. |
| "No guest limits" | No **quotas** by default. **Safety rails** that prevent machine damage stay on (§9.6). | Quotas are about fairness, and fairness is negotiated between people. Safety rails are about the machine surviving (e.g. not letting the disk reach 100% or host RAM trigger the OOM killer against the Owner's desktop session), which is the whole point of this project. |

---

## 2. Threat model

### 2.1 Actors and trust

| Actor | Trust | Notes |
|---|---|---|
| Owner | Trusted | Owns the GPU machine. Runs `reflopd`. |
| `reflopd` + `reflop-worker` | Trusted, but assumed to contain bugs | We write it. Defense in depth around it. |
| Server machine | Compromise considered **very unlikely** | Client protections exist anyway (§12). |
| Guest (client) | **Untrusted** | May be malicious, careless, or compromised. Can craft arbitrary protocol messages. |
| Network | Untrusted | Eavesdropping, tampering, replay, impersonation. |

### 2.2 Assets

| Asset | Threats | Addressed in |
|---|---|---|
| Owner machine integrity (files, processes, OS, Owner account) | Guest RCE, file read/write, privilege escalation, persistence | §3, §4, §9.4 |
| Owner machine availability (disk, host RAM, process table) | Disk fill, RAM exhaustion, fork/process bombs | §9.6 safety rails |
| GPU availability for the Owner | Guest monopolizes GPU | Owner controls (§9.7). Usage itself is negotiated (G8). |
| Guest-to-Guest isolation | Guest A reads/modifies Guest B's tensors or datasets | §9.4, I8 |
| Guest machine integrity | Reverse attacks from a malicious or impersonated server | §12 |
| Channel confidentiality/integrity | MITM, replay, impersonation | §15 |

### 2.3 Attack surface (Guest → Server)

In v1 the **only** ways a Guest can influence the server are:

1. Authentication/handshake messages.
2. Protobuf control messages (register graph, execute, upload, fetch, free, …).
3. Raw tensor bytes with explicit dtype and shape.
4. TensorPack uploads (safetensors shards + JSON manifest).

Any other path is a design defect. Residual risks:

| Residual surface | Example | Mitigation |
|---|---|---|
| Decoder bugs | Malformed protobuf, huge length prefixes | Decode in memory-safe languages (Go, Python), size limits, fuzzing (§17.3) |
| Memory safety of whitelisted kernels | Out-of-bounds indices into `index_select`, `embedding`, `scatter`, pooling-backward indices | Mandatory runtime bounds checks (I4). Restricted ops get custom validators (§8). |
| Integer overflow | Shape `[2^40, 2^40]` overflowing `numel * itemsize` | Checked arithmetic in validation |
| Uninitialized memory disclosure | `empty`/`empty_strided` returning stale GPU memory | Server implements them as zero-filled (§8.3 `UNINIT`) |
| Resource exhaustion | Huge allocations, giant matmuls, million-node graphs | Protocol sanity limits + safety rails (§9.6) |
| CUDA context poisoning | Device-side assert kills the CUDA context | Per-identity worker processes; pre-checks so asserts never fire |
| GPU driver bugs | Valid op sequences triggering driver bugs | Defense in depth (sandbox). Cannot be reduced to zero; documented (§21). |
| Side channels | Timing inference across Guests | Out of scope for v1; documented |

---

## 3. Security invariants (never violate)

| ID | Invariant | How it is verified |
|---|---|---|
| **I1** | The server never executes any client-supplied bytes as code. Forbidden on the server, anywhere reachable from request handling: Python `eval`/`exec`/`compile`; `pickle`/`marshal`/`dill`/`cloudpickle`/`shelve`; `torch.load` (any `weights_only` value); `torch.jit.load`/`torch.export.load`/`torch.package`; `dlopen`/`ctypes`/`cffi`/`importlib` with dynamic names; loading PTX/cubin/SASS/fatbin; Triton/NVRTC/inductor JIT from client input; `subprocess`/`os.system`/shells; `torch.library` registration at runtime. | Static analysis (§17.4), code review, security tests |
| **I2** | Operator dispatch is **integer enum id → static whitelist table** only. Never `getattr(torch.ops, <client string>)`, never dynamic import, never reflection driven by client data. | Static analysis rule banning dynamic `getattr` on `torch.ops`/modules in worker code |
| **I3** | Every graph passes full validation (structure → schema → meta shape inference → resource/safety check) **before any device memory is allocated** for it. | Security corpus (§17.3) |
| **I4** | Every operator input that carries data-dependent indices (§8 flag `IDX`) is bounds-checked at runtime before the kernel launches. On failure, the kernel is not launched and a structured error is returned. | Per-op OOB security tests |
| **I5** | Workers run one process per (identity, session), inside the sandbox of §9.4, with all layers available on the host enabled. A worker never has network access other than its inherited Unix socket to `reflopd`. | Isolation tests (§17.5) |
| **I6** | Client→server formats are data-only: protobuf, raw tensor bytes with explicit dtype/shape, strictly validated safetensors, schema-validated JSON manifests. | Parser fuzzing |
| **I7** | No shell for Guests, ever. No OS accounts are created per Guest. Guests never choose server filesystem paths: all storage locations are server-decided. If SSH transport is used, it is a forced-command pipe with `restrict` (§15.4). | Design review, path-traversal tests, SSH-mode tests |
| **I8** | Tensor handles and dataset handles are namespaced per identity. One identity cannot name, guess, enumerate or access another identity's handles. | Cross-tenant tests |
| **I9** | The client never deserializes executable formats from the server, never writes to server-chosen paths, and strips terminal control sequences from any server-provided string before printing. | Client security tests (§12) |
| **I10** | `opset/v1/opset.yaml` is the single source of truth for the operator set. Server dispatch tables and client mappings are code-generated from it. | CI check: generated files match |
| **I11** | **Fail closed.** Unknown fields, ops, dtypes, versions, devices, attributes or enum values are rejected. | Security tests |
| **I12** | The client never sends environment variables, local file paths, hostnames, secrets, or source code. Only IR, tensors, TensorPacks, and protocol metadata. | Client message audit test |
| **I13** | Worker returns no uninitialized memory: allocation-only operators produce zero-filled tensors. | `UNINIT` tests |

---

## 4. Deployment model — what must run on the server, and why

### 4.1 Why "install nothing on the server" cannot protect the server

The client is untrusted (G3). Any check that runs only on the client runs on a machine the attacker controls and can be skipped. For example, the Guest can bypass the `reflop` client and open a raw SSH connection or speak the protocol directly. Therefore **something on the server, installed by the Owner and not modifiable by Guests, must decide what is allowed.** That component is the enforcement point.

| Option | Server-side install | Protects the server? | Verdict |
|---|---|---|---|
| A. Plain SSH (like `recom`) | Nothing | No. Guest gets a shell (full RCE). | Rejected |
| B. Client-side enforcement only | Nothing | No. Bypassable by a modified or alternative client. | Rejected |
| C. Guest pushes the server component on first use (like `recom` pushing helpers) | Nothing manual | No. The Guest controls the enforcer and can push a modified one. No pre-existing trusted component exists to verify a signature. | Rejected |
| D. Owner builds a GPU container and gives Guests SSH into it | Container runtime, image, NVIDIA container toolkit | Partially. Still RCE inside the container, shared GPU driver, larger setup, container-escape risk. | Rejected as default; mentioned in docs as an alternative |
| **E. Owner runs `reflopd` (single binary, rootless by default)** | One binary plus an auto-fetched pinned runtime | **Yes.** Guests can only speak the reFLOP protocol. | **Chosen** |

### 4.2 Make option E as close to "zero setup" as possible

**Level 0 — rootless, zero-config (default):**

- The Owner runs one bootstrap command (install script or downloaded release binary) and then `reflopd up`.
- Everything lives under the Owner's user, in XDG directories (`~/.local/share/reflop`, `~/.config/reflop`). **No root, no system user, no system packages, no edits to system files.**
- `reflopd up` on first run:
  1. Detects GPU, driver and kernel features (Landlock ABI, seccomp, user namespaces, systemd user session, cgroup delegation).
  2. Fetches the **pinned compute runtime** (a specific PyTorch build matching the detected GPU architecture, verified by SHA-256 hashes baked into the `reflopd` release) into `~/.local/share/reflop/runtime/<version>/`. Guests can never trigger, choose or modify this.
  3. Runs `reflopd selftest` (functional + isolation checks).
  4. Prints a one-time invite code (§15.2).
- No configuration file is required. Sane defaults apply. A config file is optional.

**Level 1 — system install (optional, recommended for stronger isolation, needs `sudo` once):**

- `sudo reflopd install --system` creates a dedicated system user `reflop` (no login shell, no home), a hardened systemd unit (§9.8), and cgroup delegation.
- Benefit: a hypothetical sandbox escape lands in an empty, unprivileged account instead of the Owner's account (which holds the Owner's files, SSH keys, browser profile, etc.).

`reflopd doctor` prints which level is active and which sandbox layers are effective, in plain language.

### 4.3 Rootless sandbox feasibility (design constraints the agent must respect)

| Mechanism | Needs root? | Notes and constraints |
|---|---|---|
| `no_new_privs` + seccomp-BPF | No | Works on all supported kernels. **Primary** mechanism for blocking `execve`, process creation, non-Unix sockets, `ptrace`, `mount`, and similar. |
| Landlock | No (any process can sandbox itself) | Filesystem rules since Linux 5.13 (ABI 1). TCP rules need Linux 6.7+ (ABI 4). Use best-effort ABI detection. Rules do not apply to file descriptors opened before the ruleset is enforced, so the worker must **enforce Landlock after warm-up** and must not keep unneeded fds open. |
| Unprivileged user namespaces (bubblewrap-style mount/net namespaces) | No, but often **restricted** | Ubuntu 24.04 restricts unprivileged user namespaces through AppArmor by default (`kernel.apparmor_restrict_unprivileged_userns=1`). **Do not depend on them.** Use them as an extra layer only when available. |
| cgroup v2 limits | Needs systemd user delegation (`systemd-run --user --scope`) or Level 1 | Used only for safety rails (host RAM, pids). If unavailable at Level 0, fall back to in-process accounting plus `RLIMIT_NPROC`. **Never use `RLIMIT_AS`**: CUDA reserves very large virtual address ranges and breaks under it. |
| Network isolation | No | seccomp: deny `socket()` for all families except `AF_UNIX` (and deny `connect`/`bind` on non-inherited sockets where feasible). Landlock TCP rules are an extra layer on 6.7+. Network namespace is an extra layer when user namespaces are available. |

**Worker start-up order (important):**
1. Start the process with a minimal environment (no inherited Owner env vars except an explicit allowlist such as `CUDA_VISIBLE_DEVICES`).
2. Import the runtime, initialize CUDA, preload kernels (`CUDA_MODULE_LOADING=EAGER` or warm-up run) and the op table.
3. Close every fd except the control socket and required device fds.
4. Apply `no_new_privs`, then Landlock (read-only: runtime dir, needed `/proc`/`/sys` entries discovered in M1; read-write: session scratch dir only; read-only: this identity's dataset store), then seccomp.
5. Only then accept requests.

### 4.4 Transport modes

| Mode | What it is | When to use |
|---|---|---|
| **Daemon mode** (default) | `reflopd` listens on a TCP port with TLS 1.3 and key-based client auth (§15). Default bind is `127.0.0.1`; exposing it requires an explicit flag. Recommended exposure is via a Tailscale/WireGuard address. | General use |
| **SSH pipe mode** (optional) | Reuses the existing `sshd`. The Guest's key is added to the Owner's (Level 0) or `reflop`'s (Level 1) `authorized_keys` with `restrict,command="reflopd serve --stdio --identity <id>"`. SSH becomes a pipe that can only run that command, with no PTY, no port forwarding, no agent forwarding, no X11. | Owner already runs `sshd` and does not want to open a new port. Natural migration path for `recom` users. |

In SSH pipe mode, `reflopd serve --stdio` must ignore `SSH_ORIGINAL_COMMAND` except for an exact allowlisted value, and must spawn the worker through the same supervisor and sandbox as daemon mode. `reflopd invite --via ssh` prints (and with Owner confirmation, appends) the exact `authorized_keys` line.

---

## 5. Architecture overview

```
┌────────────── Guest machine (e.g. Apple Silicon MacBook, no CUDA) ──────────────┐
│ user program (train.py)                                                          │
│    │  import reflop; dev = reflop.device(); model.to(dev)                        │
│    ▼                                                                             │
│ reflop (Python package)                                                          │
│  ├─ RemoteTensor  (__torch_dispatch__ wrapper subclass)  → intercepts aten ops    │
│  ├─ Tracer        (lazy trace, decomposition, functionalization)                 │
│  ├─ GraphCache    (canonicalize → hash → register once)                          │
│  ├─ Placement     (local / server-GPU / server-CPU)                              │
│  ├─ TensorPack    (dataset bootstrap: safetensors shards + manifest)            │
│  └─ Transport     (gRPC over TLS 1.3, pinned server key; or SSH pipe)            │
└───────────────────────────────────────┬──────────────────────────────────────────┘
                                        │
┌──────────────── Owner machine (Linux + NVIDIA GPU) ─────────────────────────────┐
│ reflopd (Go, single static binary, runs as Owner user or `reflop` system user)   │
│  ├─ auth        pairing, identities, revoke                                      │
│  ├─ transport   gRPC/TLS listener or stdio (SSH pipe), message size limits       │
│  ├─ sessions    per-identity sessions, TTL, garbage collection                   │
│  ├─ store       content-addressed TensorPack storage per identity                │
│  ├─ rails       safety rails (disk/RAM floors); optional quotas (off by default) │
│  ├─ supervisor  spawns/monitors/kills sandboxed workers                          │
│  ├─ runtime     pinned compute runtime fetch/verify (Owner-triggered only)       │
│  └─ audit       append-only audit log                                            │
│       │ inherited Unix socketpair (worker has no other network)                  │
│       ▼                                                                          │
│ reflop-worker (one process per identity-session, sandboxed after warm-up)        │
│  ├─ decode     protobuf → internal graph                                         │
│  ├─ validate   structure → schema → meta inference → safety                      │
│  ├─ execute    enum id → static op table → PyTorch kernels (CUDA or CPU)         │
│  └─ handles    identity-scoped tensor table                                      │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### 5.1 Core idea in one sentence

The client describes *what* to compute as a loop-free DAG over a closed operator set. The server validates the DAG and runs it with trusted kernels it already has. Only data crosses the wire, so the server's attack surface shrinks to "parser + validator + whitelisted kernels".

### 5.2 Relation to `recom`

| Aspect | `recom` | reFLOP |
|---|---|---|
| Sent to server | Source code, data files, dependency manifests | Operator graphs, tensors, TensorPacks |
| Runs on server | Arbitrary commands (shell) | Whitelisted operators only |
| Where the user program runs | Server | **Client** (`reflop run` also runs locally) |
| Auth | SSH | Own key-based auth, or SSH forced-command pipe |
| Server setup | Per-project venv auto-setup | None per project; one rootless bootstrap |
| Languages | 17+ | PyTorch (v1), C/C++ API (M7) |
| Non-tensor CPU work | On server | On client |

They serve different trust relationships. `recom` is for "my own server"; reFLOP is for "lending my GPU to people I don't fully trust". reFLOP does not replace `recom`.

---

## 6. Execution model

### 6.1 Entities

| Entity | Definition |
|---|---|
| Identity | One paired Guest (one or more public keys). The Owner can also pair themselves as an identity. |
| Session | One connection-scoped unit of work. Exactly one worker process per session. |
| Handle | 128-bit id referring to a server-resident tensor. **Client-proposed** within the session's namespace (so the client can pipeline requests that reference future outputs). Stored server-side under (identity, session, handle). Duplicates or reserved ranges are rejected. |
| Dataset handle | Created by TensorPack upload. Identity-scoped, shareable across that identity's sessions, never across identities. |
| Graph | A registered, validated DAG. Keyed by SHA-256 of its canonical encoding. |

### 6.2 RPCs (`reflop.v1.Compute`)

| RPC | Direction | Purpose |
|---|---|---|
| `Hello` | C→S | Version exchange. Returns `ServerCapabilities` (§14). |
| `OpenSession` / `CloseSession` | C→S | Create or destroy a session (spawn or kill a worker). |
| `RegisterGraph(graph)` | C→S | Full validation (§9.2). Returns the graph hash and an execution plan summary (peak memory estimate, FLOP estimate). Re-registering an identical hash is a no-op. |
| `Execute(graph_hash, bindings, output_spec)` | C→S | Run a registered graph. Bindings map each graph input to {handle, inline small tensor, dataset reference, scalar}. Output spec per output: {store as new handle, rebind an existing handle, return to client}. Asynchronous and pipelinable. |
| `Upload(stream)` | C→S | Upload a tensor (e.g. initial weights) into a handle. |
| `Fetch(handles)` | C→S | Download tensors (subject to protocol sanity limits). |
| `Free(handles)` | C→S | Release handles. |
| `PushTensorPack(stream)` | C→S | Chunked, resumable dataset upload (§10). |
| `ListDatasets` / `DeleteDataset` | C→S | Dataset management. |
| `Snapshot(handles)` / `Restore` | C→S | Persist or restore session tensors to or from server storage as safetensors (optional). |
| `Events` (server stream) | S→C | Progress, pause/resume notices, Owner messages. **Fixed schema, informational only. Never triggers client-side actions** (§12). |

There is no RPC by which the server asks the client to do anything. All actions are client-initiated.

### 6.3 One training step (example flow)

1. User code runs `loss = model(x); loss.backward(); opt.step()`.
2. `RemoteTensor` intercepts every aten op and records it in a lazy trace without executing. Shapes/dtypes are computed locally with meta tensors, so shape-dependent Python code works without round trips.
3. At a sync point (§11.4) or `reflop.step_boundary()`, the trace is flushed.
4. The trace is decomposed to opset v1, functionalized (no in-place, no aliasing on the wire), canonicalized and hashed.
5. If the hash is new: `RegisterGraph`. Then (or directly): `Execute(hash, bindings)`.
6. Parameters and optimizer state live in server handles. Updates use `rebind` output specs; the server may execute updates in place on donated buffers (§9.3).
7. Only values the user explicitly materializes (e.g. `loss.item()`) are downloaded.

**Traffic target after warm-up:** a few KB upload per step (graph hash, bindings, batch indices or seed), downloads only for requested values.

---

## 7. FlopIR (the wire graph format)

Defined in `proto/reflop/v1/*.proto`. Protobuf `proto3` with explicit presence where needed. **Unknown fields are rejected** (decode with a strict mode or verify that no unknown fields were preserved; I11).

### 7.1 Messages

| Message | Fields | Constraints |
|---|---|---|
| `Graph` | `opset_version`, `inputs[]`, `nodes[]`, `outputs[]`, `device` (enum: CUDA0…, CPU) | `nodes` ≤ `max_nodes` (protocol sanity default 1,000,000). `inputs` ≤ 65,536. |
| `GraphInput` | `index`, `kind` ∈ {TENSOR, SCALAR, DATASET_REF, GENERATOR}, `dtype`, `shape[]` | v1: **static shapes only**. Each dim ≥ 0. `numel × itemsize` computed with checked arithmetic. |
| `Node` | `op_id` (uint32), `overload_id` (uint32, part of the op key), `inputs[]` (`ValueRef`), `attrs[]` (`Attr`) | `(op_id)` must exist in the negotiated opset |
| `ValueRef` | oneof { `graph_input` (uint32), `node_output` (node index, output index), `const` (`Constant`), `none` } | `node_output.node` must be **strictly less than** the current node index: topological order is enforced, so cycles are impossible |
| `TensorList` | `items[]` (`ValueRef`) | For `cat`, `index`, `split_with_sizes` outputs and similar. Length ≤ 65,536. |
| `Constant` | oneof { int64, double, bool, int64 list, double list, small tensor (≤ 64 KiB) } | Larger constants must be uploaded as handles |
| `Attr` | `key_id` (enum), oneof { int64, double, bool, int64 list, double list, dtype enum, memory-format enum, generic enum value } | **No string attributes.** String-valued schema args (`rounding_mode`, `approximate`, `reduce`, …) are encoded as enums (`ENUM` flag, §8.3). |
| `GraphOutput` | `value` (`ValueRef`) | |
| `DType` enum | bool, uint8, int8, int16, int32, int64, float16, bfloat16, float32, float64, complex64, complex128, float8_e4m3fn, float8_e5m2 | fp8 and complex are capability-gated (§14). Unknown values are rejected. |

### 7.2 Forbidden structures (v1)

Control-flow higher-order ops (`cond`, `while_loop`, `scan`, `map`, `associative_scan`), subgraphs, callbacks, string attributes, symbolic/dynamic shapes, I/O ops, collective communication ops, sparse/nested/quantized layouts. A v1 graph is a **loop-free DAG with static shapes**, so its cost is statically estimable.

### 7.3 Canonical encoding and hashing

The client and server both define a canonical byte encoding (deterministic field order, values renumbered in topological order, no unknown fields). Hash = SHA-256 of the canonical encoding plus `opset_version`. The server **recomputes** the hash itself and never trusts a client-claimed hash.

### 7.4 Versioning

- `opset_version` increases monotonically. Op ids are permanent. A semantic change requires a new id.
- Protocol version (`reflop.v1`) and opset version are independent.

---

## 8. Opset v1 (the operator whitelist)

### 8.1 Design

Opset v1 has three tiers.

| Tier | Contents | Purpose | Default |
|---|---|---|---|
| **Tier 0 — Core** | PyTorch **Core ATen IR** (193 entries listed in the PyTorch 2.15 docs, counting overloads), minus banned/client-only entries, with security flags | Completeness: any PyTorch program that decomposes to Core ATen can run | Enabled |
| **Tier 1 — Performance & training extensions** | Non-core aten ops that PyTorch's own backward/optimizer/attention/loss paths emit and that have fused trusted kernels | Speed: avoid decomposing attention, norms, losses, optimizer updates into hundreds of primitive ops | Enabled |
| **Tier 2 — Capability-gated** | Linear algebra, complex FFT, fp8 matmul, fused RNN kernels, CTC loss | Useful, but more validation surface or hardware-specific | Enabled only if the server reports the capability (§14). Each op individually toggleable by the Owner. |

Anything not listed is rejected (I11). The client handles unlisted ops by (1) PyTorch decomposition into listed ops, (2) local fallback per placement policy (§13), or (3) a clear error.

**M0 task** (done: [ADR-003](DECISIONS.md#adr-003-opset-v1-adjustments), [OPSET.md](OPSET.md)): generate the actual Tier 0 list by introspecting the pinned PyTorch (`torch.Tag.core` on `torch.ops.aten` overloads), diff it against §8.4, and record differences in the spike report. Tier 1/2 names must be verified to exist in the pinned version; drop or rename with an ADR otherwise.

### 8.2 Opset file schema (`opset/v1/opset.yaml`)

| Field | Meaning |
|---|---|
| `id` | Stable uint32, never reused |
| `name` | Fully qualified overload, e.g. `aten.add.Tensor` |
| `tier` | 0 / 1 / 2 |
| `schema` | Positional/keyword args with types and allowed dtypes; outputs |
| `flags` | Subset of §8.3 flags |
| `validator` | Name of the server-side extra validator (required for `IDX`, `ALLOC`, `RESTRICTED`, `DDS`) |
| `cost` | Name of the FLOP/byte estimator (e.g. `mm: 2*M*N*K`) |
| `index_inputs` | For `IDX`: which inputs are indices and which dimension/size they index into, plus sentinel rules (e.g. `ignore_index`, `padding_idx`) |
| `enum_args` | For `ENUM`: allowed string values and their enum codes |
| `devices` | `cuda`, `cpu` |
| `capability` | For Tier 2: required capability key |
| `functional_of` | For `FUNC`: the in-place/out PyTorch op this functional op mirrors |
| `since`, `deprecated` | Versioning |

### 8.3 Security flags

| Flag | Meaning | Server obligation |
|---|---|---|
| `IDX` | Has data-dependent index inputs | Runtime bounds check before launch (I4), honoring declared sentinels |
| `RNG` | Consumes randomness | Uses the session generator. Seed is client-settable or server-random. |
| `DDS` | Data-dependent output shape | Two-phase execution: run to a sync point, return the shape, client continues (§11.4). M2 may start with client-side fallback. |
| `ALLOC` | Output size derived from scalar arguments (e.g. `full`, `arange`, `repeat`, `upsample`) | Output size is computed during validation and must pass protocol sanity limits and safety rails before allocation |
| `ENUM` | Schema has string arguments | Encoded as enum on the wire; unknown values rejected |
| `UNINIT` | PyTorch would return uninitialized memory | Server implements as zero-filled (I13) |
| `FUNC` | Mirrors an in-place/out PyTorch op | Wire form is functional. Server may run in place only on donated buffers it owns (§9.3). |
| `RESTRICTED` | Needs a custom validator beyond meta inference | See per-op notes |
| `CLIENT` | Never sent; resolved on the client (shape queries, scalar extraction) | Reject if received |
| `BANNED` | Never allowed | Reject |

### 8.4 Tier 0 — Core ATen (from the PyTorch 2.15 Core ATen IR table)

Grouped by category. Every overload listed in the source table appears here exactly once with a disposition. Unflagged ops need only standard meta validation.

| Category | Operators | Flags / disposition |
|---|---|---|
| Unary math | `abs`, `acos`, `acosh`, `asin`, `asinh`, `atan`, `atanh`, `ceil`, `cos`, `cosh`, `erf`, `exp`, `expm1`, `floor`, `log`, `log10`, `log1p`, `log2`, `neg`, `reciprocal`, `round`, `rsqrt`, `sign`, `sin`, `sinh`, `sqrt`, `tan`, `tanh`, `trunc`, `sigmoid` | — |
| Binary arithmetic | `add.Scalar`, `add.Tensor`, `sub.Scalar`, `sub.Tensor`, `mul.Scalar`, `mul.Tensor`, `div.Scalar`, `div.Tensor`, `fmod.Scalar`, `fmod.Tensor`, `remainder.Scalar`, `remainder.Tensor`, `pow.Scalar`, `pow.Tensor_Scalar`, `pow.Tensor_Tensor`, `atan2`, `maximum`, `minimum` | Integer division/modulo by zero: validator inserts a runtime zero-divisor check for integer dtypes (avoids undefined behavior on CPU) |
| | `div.Scalar_mode`, `div.Tensor_mode` | `ENUM` (`rounding_mode`: none/trunc/floor) + integer zero-divisor check |
| | `atan2.out` | `FUNC` → mapped to functional `atan2` (no `out=` on the wire) |
| Comparison / logical / bitwise | `eq.Scalar`, `eq.Tensor`, `ne.Scalar`, `ne.Tensor`, `lt.Scalar`, `lt.Tensor`, `le.Scalar`, `le.Tensor`, `gt.Scalar`, `gt.Tensor`, `ge.Scalar`, `ge.Tensor`, `isinf`, `isnan`, `logical_and`, `logical_or`, `logical_xor`, `logical_not`, `bitwise_and.Scalar`, `bitwise_and.Tensor`, `bitwise_or.Scalar`, `bitwise_or.Tensor`, `bitwise_xor.Scalar`, `bitwise_xor.Tensor`, `bitwise_not`, `where.self` | — |
| Clamping / activations | `clamp`, `clamp.Tensor`, `relu`, `leaky_relu`, `hardtanh`, `elu` | — |
| | `gelu` | `ENUM` (`approximate`: none/tanh) |
| Reductions | `sum.dim_IntList`, `mean`, `mean.dim`, `prod`, `prod.dim_int`, `amax`, `amin`, `any`, `any.dim`, `any.dims`, `argmax`, `argmin`, `max.dim`, `min.dim`, `var.correction`, `var.dim`, `cumsum` | — |
| Sorting / selection | `sort`, `topk` | `topk`: `k` validated ≤ dim size (static) |
| Matmul / conv | `mm`, `addmm`, `bmm`, `convolution`, `convolution_backward` | Cost functions mandatory (large FLOPs) |
| Normalization | `native_layer_norm`, `native_layer_norm_backward`, `native_group_norm`, `native_group_norm_backward`, `_native_batch_norm_legit.no_stats`, `_native_batch_norm_legit_no_training` | — |
| | `_native_batch_norm_legit` | `FUNC` (mutates `running_mean`/`running_var` in PyTorch). Wire form returns updated running stats as extra outputs; server rebinds. Prefer `_native_batch_norm_legit_functional` (Tier 1) when the client produces it. |
| Softmax | `_softmax`, `_log_softmax` | — |
| Dropout / RNG | `native_dropout`, `rand`, `randn`, `randperm` | `RNG`; `rand`/`randn`/`randperm` also `ALLOC` |
| Pooling | `avg_pool1d`, `avg_pool2d`, `avg_pool2d_backward`, `avg_pool3d`, `adaptive_avg_pool1d`, `_adaptive_avg_pool2d`, `_adaptive_avg_pool2d_backward`, `_adaptive_avg_pool3d`, `max_pool2d_with_indices`, `max_pool3d_with_indices` | — |
| | `max_pool2d_with_indices_backward` | `IDX` (the `indices` input may come from the client; check against the input spatial size) |
| Padding / resampling | `constant_pad_nd`, `reflection_pad1d`, `reflection_pad2d`, `reflection_pad3d`, `replication_pad2d`, `replication_pad3d` | `ALLOC`. Reflection pads require pad < input dim (meta enforces; validator double-checks). |
| | `upsample_bilinear2d.vec`, `upsample_nearest2d.vec`, `col2im`, `grid_sampler_2d` | `ALLOC` (output size). `grid_sampler_2d` modes are int-encoded and validated against allowed values. |
| Indexing / gather / scatter | `index.Tensor` | `IDX` (integer indices). Boolean-mask indices make it `DDS` → two-phase. |
| | `index_put` | `IDX` (+ `DDS` handling for bool masks) |
| | `index_select`, `gather`, `scatter.src`, `scatter.value`, `scatter_add`, `embedding`, `embedding_dense_backward` | `IDX` (`embedding`: honor `padding_idx`; `embedding_dense_backward`: indices < `num_weights`) |
| | `scatter_reduce.two` | `IDX` + `ENUM` (`reduce`: sum/prod/mean/amax/amin) |
| | `_embedding_bag` | `IDX` + `RESTRICTED`: `offsets` must be non-decreasing, within `[0, len(indices)]`, first offset 0; `per_sample_weights` shape check |
| | `masked_scatter` | `RESTRICTED`: runtime check that `source.numel() ≥ mask.sum()` |
| | `select.int`, `select_scatter`, `slice.Tensor`, `slice_scatter` | Static integer arguments validated by meta inference |
| | `nonzero` | `DDS` → two-phase |
| Shape / view ops | `view`, `expand`, `permute`, `squeeze.dim`, `squeeze.dims`, `unsqueeze`, `diagonal`, `alias`, `split_with_sizes`, `cat`, `flip`, `repeat`, `clone` | `repeat`: `ALLOC`. Views are executed by the server's real PyTorch on server-owned tensors. The client never controls strides or offsets directly. |
| | `as_strided` | **`RESTRICTED`** — avoid. The client tracer must first try to rewrite `as_strided` into `view`/`slice`/`permute`/`expand` chains. If it cannot, the server validator requires: all strides ≥ 0, sizes ≥ 0, checked-arithmetic maximum reachable element `offset + Σ (size_i − 1)·stride_i` < **the server-side storage numel of the actual input tensor** (never a client claim). Any doubt → reject. |
| Creation | `full`, `full_like`, `scalar_tensor`, `arange.start_step`, `fill.Scalar` | `ALLOC` (`arange`: numel computed with checked float→int arithmetic) |
| | `empty.memory_format`, `empty_strided` | `UNINIT` → zero-filled. `empty_strided`: `RESTRICTED`: strides must describe a non-overlapping dense layout; otherwise reject. |
| Type / device | `_to_copy`, `copy` | `RESTRICTED`: `device` must be one of this session's server devices; `layout` must be strided; `pin_memory` ignored; dtype must be capability-supported |
| Distance | `_cdist_forward`, `_pdist_forward` | `compute_mode` validated |
| FFT | `_fft_r2c`, `_fft_c2r` | Tier-2 capability `complex` required at runtime (complex outputs/inputs) |
| Client-only | `sym_size.int`, `sym_stride.int`, `sym_numel`, `sym_storage_offset`, `sym_is_contiguous` | `CLIENT` (resolved from static shapes; reject if received) |
| | `_local_scalar_dense` | `CLIENT` (implemented as `Fetch` of a 0-d tensor) |
| Banned | `resize_` | `BANNED` (in-place storage resize; incompatible with a functional wire format) |

### 8.5 Tier 1 — Performance and training extensions

These are emitted by AOTAutograd backward graphs, `torch.optim`, AMP, `F.scaled_dot_product_attention`, and common loss functions. Each has a trusted fused kernel in PyTorch. Without them, these would decompose into many Tier 0 ops: correct but slow.

| Category | Operators | Flags / notes |
|---|---|---|
| Activation backward & extra activations | `threshold_backward`, `gelu_backward`, `sigmoid_backward`, `tanh_backward`, `elu_backward`, `leaky_relu_backward`, `hardtanh_backward`, `silu`, `silu_backward`, `hardswish`, `hardswish_backward`, `hardsigmoid`, `hardsigmoid_backward`, `softplus`, `softplus_backward`, `mish`, `mish_backward`, `log_sigmoid_forward`, `log_sigmoid_backward`, `_prelu_kernel`, `_prelu_kernel_backward`, `glu`, `glu_backward` | `gelu_backward`: `ENUM` |
| Softmax backward & friends | `_softmax_backward_data`, `_log_softmax_backward_data`, `logsumexp` | — |
| Normalization | `native_batch_norm`, `native_batch_norm_backward`, `_native_batch_norm_legit_functional`, `_fused_rms_norm`, `_fused_rms_norm_backward`, `_weight_norm_interface`, `_weight_norm_interface_backward` | `native_batch_norm`: `FUNC` (running stats as outputs). `_fused_rms_norm*`: include only if present in the pinned PyTorch (verify in M0). |
| Attention (fused) | `_scaled_dot_product_flash_attention`, `_scaled_dot_product_flash_attention_backward`, `_scaled_dot_product_efficient_attention`, `_scaled_dot_product_efficient_attention_backward`, `_scaled_dot_product_cudnn_attention`, `_scaled_dot_product_cudnn_attention_backward`, `_scaled_dot_product_attention_math` | `RNG` when `dropout_p > 0` (Philox seed/offset are outputs, kept as server handles and consumed by backward). Attention-mask/bias shapes validated. Backend availability is capability-reported per GPU. |
| Losses | `nll_loss_forward`, `nll_loss_backward`, `nll_loss2d_forward`, `nll_loss2d_backward` | `IDX` on `target` with `ignore_index` sentinel honored; `reduction` int-validated |
| | `mse_loss`, `mse_loss_backward`, `smooth_l1_loss`, `smooth_l1_loss_backward`, `huber_loss`, `huber_loss_backward`, `binary_cross_entropy`, `binary_cross_entropy_backward`, `binary_cross_entropy_with_logits`, `soft_margin_loss`, `soft_margin_loss_backward` | `reduction` validated |
| Matmul variants | `baddbmm`, `addbmm`, `addmv`, `_addmm_activation`, `_int_mm`, `dot`, `vdot`, `mv`, `outer` | `_int_mm`: int8 dtype constraints per capability |
| Pooling / spatial backward | `avg_pool3d_backward`, `max_pool3d_with_indices_backward`, `_adaptive_avg_pool3d_backward`, `adaptive_max_pool2d`, `adaptive_max_pool2d_backward`, `adaptive_max_pool3d`, `adaptive_max_pool3d_backward`, `max_unpool2d`, `max_unpool3d` | `*_with_indices_backward`, `adaptive_max_pool*_backward`, `max_unpool*`: `IDX` |
| Upsampling | `upsample_nearest1d`, `upsample_nearest2d`, `upsample_nearest3d`, `_upsample_nearest_exact1d/2d/3d`, `upsample_linear1d`, `upsample_bilinear2d`, `upsample_bicubic2d`, `upsample_trilinear3d`, `_upsample_bilinear2d_aa`, and their `_backward` variants | `ALLOC` |
| Spatial transforms | `grid_sampler_2d_backward`, `grid_sampler_3d`, `grid_sampler_3d_backward`, `affine_grid_generator`, `im2col`, `pixel_shuffle`, `pixel_unshuffle`, `channel_shuffle`, `roll`, `rot90` | `ALLOC` where applicable |
| Indexing extras | `index_add`, `index_copy`, `index_fill.int_Scalar`, `index_fill.int_Tensor`, `masked_fill.Scalar`, `masked_fill.Tensor`, `take`, `_unsafe_index.Tensor`, `_unsafe_index_put`, `searchsorted.Tensor`, `bucketize.Tensor`, `_embedding_bag_backward`, `_embedding_bag_dense_backward`, `embedding_renorm` | `index_*`, `take`, `_embedding_bag_*`: `IDX`. **`_unsafe_index*` skip PyTorch's own checks, so they are mandatory `IDX` with no exceptions** (they appear in upsample/interpolate decompositions). `embedding_renorm`: `FUNC` + `IDX`. |
| Reductions extras | `sum` (full), `nansum`, `std.correction`, `var_mean.correction`, `std_mean.correction`, `cumprod`, `cummax`, `cummin`, `logcumsumexp`, `linalg_vector_norm`, `norm.ScalarOpt_dim`, `all`, `all.dim`, `aminmax`, `median.dim`, `nanmedian.dim`, `mode`, `kthvalue` | `kthvalue`: `k` validated |
| Elementwise extras | `clamp_min`, `clamp_max`, `exp2`, `erfc`, `erfinv`, `lgamma`, `digamma`, `polygamma`, `i0`, `special_i0e`, `special_i1`, `special_i1e`, `lerp.Scalar`, `lerp.Tensor`, `addcmul`, `addcdiv`, `nan_to_num`, `copysign`, `hypot`, `floor_divide`, `bitwise_left_shift`, `bitwise_right_shift`, `isfinite`, `signbit`, `fmax`, `fmin`, `xlogy.Tensor`, `logit`, `frac`, `heaviside`, `sgn`, `deg2rad`, `rad2deg`, `logaddexp`, `logaddexp2` | `floor_divide`: integer zero-divisor check. Shifts: shift amount validated per dtype width (avoid UB). |
| Creation / shape extras | `zeros`, `ones`, `zeros_like`, `ones_like`, `empty_like`, `eye`, `linspace`, `logspace`, `tril`, `triu`, `diag_embed`, `narrow`, `unbind.int`, `stack`, `transpose.int`, `t`, `reshape`, `_unsafe_view`, `unfold`, `view_as_real`, `view_as_complex`, `polar`, `complex`, `contiguous`, `lift_fresh_copy`, `detach`, `repeat_interleave.self_int`, `tile`, `movedim.int`, `expand_as`, `flatten.using_ints` | `ALLOC` for creation ops. `empty_like`: `UNINIT` → zeros. `_unsafe_view`: memory-safe (numel must match; "unsafe" refers only to autograd alias tracking), treated like `view`. `view_as_complex`/`polar`/`complex`: require capability `complex`. `repeat_interleave`: tensor `repeats` variant is `DDS`; only the static int form is Tier 1. |
| RNG extras | `bernoulli.p`, `bernoulli.default`, `uniform`, `normal_functional`, `normal.Tensor_float`, `normal.float_Tensor`, `normal.Tensor_Tensor`, `exponential`, `randint`, `randint.low`, `rand_like`, `randn_like`, `randint_like`, `multinomial`, `native_dropout_backward` | `RNG`, `ALLOC`. In-place PyTorch variants (`uniform_` etc.) map to functional forms (`FUNC`). `multinomial`: probabilities validated non-negative at runtime. |
| Optimizer (fused/foreach) | `_foreach_add.List`, `_foreach_add.Scalar`, `_foreach_add.ScalarList`, `_foreach_sub.*`, `_foreach_mul.*`, `_foreach_div.*`, `_foreach_addcmul.*`, `_foreach_addcdiv.*`, `_foreach_lerp.*`, `_foreach_sqrt`, `_foreach_neg`, `_foreach_maximum.*`, `_foreach_minimum.*`, `_foreach_norm.Scalar`, `_foreach_sign`, `_foreach_abs`, `_foreach_clamp_min.*`, `_foreach_clamp_max.*`, `_foreach_pow.*`, `_foreach_reciprocal`, `_foreach_exp`, `_foreach_zero` | Functional (non-`_`) forms on the wire. List lengths must match (validator). Server may run the in-place kernel on donated buffers. |
| | `reflop::fused_adam`, `reflop::fused_adamw`, `reflop::fused_sgd`, `reflop::fused_adagrad` | Functional wrappers defined by reFLOP over PyTorch's `_fused_adam_` / `_fused_adamw_` / `_fused_sgd_` / `_fused_adagrad_` kernels. Inputs: param, grad, state lists, step, hyperparameters (scalar inputs, so changing `lr` does not change the graph hash). Outputs: updated tensors, rebound by output spec. |
| AMP | `reflop::amp_unscale_and_check` (wraps `_amp_foreach_non_finite_check_and_unscale_`), `reflop::amp_update_scale` (wraps `_amp_update_scale_`) | Functional wrappers; GradScaler support |
| reflop-native | `reflop::dataset_gather(dataset_ref, tensor_id, indices)`, `reflop::randperm_seeded(n, seed)`, `reflop::bounds_check(idx, size, sentinel?)`, `reflop::nonzero_count(t)`, `reflop::mask_count(mask)` | Trusted server implementations. `dataset_gather`: `IDX`, integer index dtypes only, reads only this identity's datasets (I8). `bounds_check` is inserted automatically by the validator and may also be sent explicitly. |

### 8.6 Tier 2 — Capability-gated

| Capability key | Operators | Notes |
|---|---|---|
| `linalg` | `linalg_cholesky_ex`, `linalg_qr`, `_linalg_svd`, `_linalg_eigh`, `_linalg_solve_ex`, `linalg_solve_triangular`, `linalg_inv_ex`, `_linalg_det`, `_linalg_slogdet`, `linalg_lu_factor_ex`, `linalg_lu_solve`, `cholesky_solve`, `triangular_solve`, `linalg_pinv` (if not decomposed), `linalg_matrix_exp` | `info` outputs returned as tensors; no exceptions across the wire. Batch and matrix sizes capped by protocol sanity limits (cubic cost). |
| `complex` | `_fft_c2c`, plus Tier 0/1 ops on complex64/complex128 (`_fft_r2c`, `_fft_c2r`, `view_as_real`, `view_as_complex`, `polar`, `complex`) | Needed for FFT-based models and rotary embeddings implemented with complex numbers (e.g. LLaMA-style RoPE via `view_as_complex`). |
| `fp8` | `_scaled_mm` | Only on GPUs that report fp8 support; scale tensors validated |
| `rnn_fused` | `_thnn_fused_lstm_cell`, `_thnn_fused_lstm_cell_backward_impl`, `_thnn_fused_gru_cell`, `_thnn_fused_gru_cell_backward`, `_cudnn_rnn`, `_cudnn_rnn_backward` | `_cudnn_rnn*`: `RESTRICTED` (flat weight buffer layout validated against the declared RNN config). Without this capability, LSTM/GRU decompose to Tier 0 (correct, slower). |
| `ctc` | `_ctc_loss`, `_ctc_loss_backward` | `IDX` on targets; length tensors validated (`target_lengths` ≤ max, `input_lengths` ≤ T) |
| `cdist_backward` | `_cdist_backward`, `_pdist_backward` | — |

### 8.7 Explicitly excluded (never add without an ADR and human approval)

`resize_`, `set_`, any storage-level op (`_make_per_tensor_quantized_tensor`, `untyped_storage` manipulation), sparse layouts (`to_sparse*`, `_sparse_*`), nested tensors (`_nested_*`), quantized dtypes and ops (`quantize_per_*`, `q*`), distributed collectives (`c10d.*`, `_c10d_functional.*`), `_assert_async` and other device-side assertion ops (they poison the CUDA context; the client asserts after fetching instead), `record_stream`, CUDA graph capture ops, `_make_dep_token`/`with_effects` token ops, all custom ops (`torch.library` registrations), Triton/inductor-generated kernels, `torch.ops.load_library`, any op with file, network or process side effects.

### 8.8 Adding an operator (procedure, R5)

1. Add a `opset.yaml` entry with a fresh id, tier, flags, validator, cost, and enum/index metadata.
2. Implement the validator (if flagged) and the cost function.
3. Run codegen (server table + client mapping).
4. Add a conformance test (§17.2) and security tests for every flag (§17.3).
5. CI must pass, including static analysis.

---

## 9. Server design

### 9.1 Components and languages

| Component | Language | Reason |
|---|---|---|
| `reflopd` | Go | Memory-safe, single static binary (fits "zero setup"), good concurrency. Faces the network. Never touches the GPU. |
| `reflop-worker` | Python + PyTorch from the pinned runtime (**default, M2–M6**) | Fastest path to correctness. Python code is memory-safe. Executing no client code (I1/I2) means Python is not an RCE vector by itself. |
| `reflop-worker-native` | C++ + libtorch (**M8, optional**) | Removes the Python interpreter from the server entirely. Same protocol and validator semantics. |

### 9.2 Validation pipeline (inside the worker, fixed order)

Every stage fails closed with a structured error code. No device allocation happens before stage V5 succeeds (I3).

| Stage | What | Rejects, for example |
|---|---|---|
| V0 Transport (`reflopd`) | Authenticated session, message size ≤ protocol limit, rate sanity | 1 GiB control message |
| V1 Decode | Strict protobuf decode, unknown fields rejected, repeated-field lengths capped | 10^8 nodes |
| V2 Structure | Opset version supported, `op_id` whitelisted and enabled for this server's capabilities, DAG order (`ValueRef` only points backward), arity matches schema | Cycles, unknown ops, Tier 2 op without capability |
| V3 Schema / attributes | dtype allowed per position, attribute types and ranges (e.g. `dim` within rank), enum codes known, checked arithmetic on all sizes | `dim = 99`, negative sizes, `numel` overflow |
| V4 Meta inference | Execute the graph on PyTorch `meta` tensors (or FakeTensor) to obtain every intermediate shape/dtype. Compare with client-declared output shapes. | Shape mismatch, broadcasting errors |
| V5 Safety | Per-node cost functions → FLOP estimate; liveness analysis → peak memory estimate; check against **safety rails** (§9.6) and **optional quotas** (off by default) | Peak memory larger than physical device memory; output tensor that would breach the host-RAM floor |
| *(registered)* | Cache the validated graph and execution plan under its server-computed hash | |
| E1 Bindings (every execute) | Handles exist, belong to this identity/session, match declared shape/dtype; inline tensors within limits | Another identity's handle, shape mismatch |
| E2 Runtime checks (every execute) | `IDX` bounds checks, integer zero-divisor checks, `RESTRICTED` runtime predicates (e.g. `masked_scatter` source size) | Index −1 or ≥ size |
| E3 Execute | Static op table dispatch with watchdog (if enabled) | — |

### 9.3 Executor

- `op_table: id → callable` is **generated code** that references concrete PyTorch overloads statically (I2).
- Executes under `torch.no_grad()`. Autograd happened on the client side: backward ops arrive as ordinary nodes.
- RNG: one `torch.Generator` per session and device. The seed is client-settable for reproducibility; otherwise it is server-random. `RNG` ops consume it in graph order.
- Numerics: server defaults (e.g. TF32 on/off, deterministic algorithms) are reported in capabilities. The client may request changes within an allowlist.
- **Buffer donation:** when an output spec says "rebind handle H" and the graph input bound to H has no other use after the producing node, the server may execute the `FUNC` op in place on H's buffer. This is a server-internal optimization and never changes semantics.
- Data-dependent shape (`DDS`) ops: the graph is split at the `DDS` node. The prefix executes, the server returns the result shape (and only the shape, unless the client asks for values), and the client continues tracing with the now-static shape.

### 9.4 Worker sandbox (I5)

| Layer | Level 0 (rootless) | Level 1 (system install) |
|---|---|---|
| Account | Owner's user | Dedicated `reflop` system user, no shell, no home |
| Process | One worker per identity-session, spawned by `reflopd` | Same |
| Privileges | `no_new_privs`, no capabilities | Same + systemd hardening |
| Syscalls | seccomp allowlist. Deny `execve`/`execveat`, `fork`/`vfork`/`clone` without `CLONE_THREAD`, `ptrace`, `mount`, `unshare` (after setup), `socket` for non-`AF_UNIX`, `bpf`, `perf_event_open`, `keyctl`, `add_key`, `userfaultfd`. The initial allowlist is derived in M1 from tracing a warm worker, then minimized. | Same |
| Filesystem | Landlock (best-effort ABI): read-only runtime dir and required `/proc`/`/sys`/`/dev/nvidia*` paths; read-only this identity's dataset store; read-write session scratch dir only. Everything else denied (Owner's home included). | Same, plus mount namespace if available |
| Network | seccomp socket-family filter; Landlock TCP deny-all on ABI ≥ 4; network namespace if user namespaces are available | Same + `RestrictAddressFamilies` |
| Devices | Only the NVIDIA device nodes needed (`/dev/nvidia<N>`, `/dev/nvidiactl`, `/dev/nvidia-uvm`, `/dev/nvidia-uvm-tools` only if required) | `DeviceAllow=` in the unit |
| Resources (safety rails only) | cgroup via `systemd-run --user --scope` when available (`memory.max` derived from the host-RAM floor, `pids.max`); otherwise `RLIMIT_NPROC` + in-process accounting. **Never `RLIMIT_AS`.** | cgroup delegation from the system unit |
| Environment | Fresh minimal env; no Owner env vars beyond an allowlist | Same |

`reflopd doctor` reports each layer as active/inactive. If **neither** seccomp nor Landlock can be applied, `reflopd` refuses Guest sessions unless the Owner passes `--accept-weak-isolation` (documented as dangerous).

### 9.5 Failure handling

| Situation | Behavior |
|---|---|
| Validation failure | Structured error; worker keeps running |
| CUDA OOM | Free cached blocks, return `OUT_OF_MEMORY`; worker keeps running (OOM is recoverable) |
| Device-side assert / CUDA context error | Worker is terminated; session marked `SESSION_LOST`; audit log flags it as a **bug** (I4 should make this impossible) |
| Worker crash | `SESSION_LOST`; other sessions unaffected |
| Watchdog timeout (if the Owner enabled one) | Worker killed (CUDA kernels cannot be preempted); `SESSION_LOST` |
| `reflopd` restart | All workers terminated; only snapshotted state is recoverable |

### 9.6 Safety rails vs quotas

**Quotas are off by default** (G8). Owner and Guests agree on usage between themselves. Contention behaves predictably: allocation is first-come-first-served, and a Guest that cannot allocate receives `OUT_OF_MEMORY` instead of crashing anything.

**Safety rails are always on.** They do not ration fair use; they keep the machine alive. All are configurable by the Owner, and every default is chosen so that a normal training job never hits it.

| Rail | Default | Why it is a rail, not a quota |
|---|---|---|
| Disk free floor | Refuse uploads/snapshots that would leave < 10% or < 20 GiB free (whichever is larger) on the store filesystem | A 100%-full disk breaks the Owner's whole OS session |
| Host RAM floor | Worker host-memory use limited so that ≥ 4 GiB (or 10%) of system RAM stays free | Prevents the kernel OOM killer from hitting the Owner's desktop processes |
| Process/thread cap | `pids.max` sized for PyTorch's thread pools (e.g. 4096) | Prevents thread/process explosions from degrading the host |
| Protocol sanity limits | Max message size, max nodes per graph, max inline tensor size, max single tensor numel (e.g. 2^40 elements), max dims (e.g. 16) | Parser/validator robustness against absurd inputs; not usage limits |
| Physical feasibility | Reject graphs whose estimated peak exceeds the device's **total** memory | Would fail anyway; fail fast without allocating |

**Optional quotas** (per identity, all unset by default): device memory share, max concurrent sessions, wall-clock per execute, FLOPs per execute, storage per identity, upload bandwidth. Configured with `reflopd quota set`.

### 9.7 Owner controls

- `reflopd pause [--now]` / `resume`: queue new executes. `--now` kills running workers.
- `reflopd kill <identity|session>`.
- `reflopd yield --auto` (optional, off by default): NVML watches non-reflop GPU processes and pauses Guest execution while the Owner is using the GPU.
- `reflopd say "<message>"`: broadcast an informational `Event` to connected Guests (e.g. "pausing in 10 min"). Plain text, sanitized on the client.

### 9.8 `reflopd` hardening

- Level 0: `reflopd` itself applies Landlock to its own process (read/write only its XDG dirs and runtime dir) and a seccomp profile that still allows spawning workers. It never touches the GPU.
- Level 1 systemd unit: `NoNewPrivileges=yes`, `ProtectSystem=strict`, `ProtectHome=yes`, `PrivateTmp=yes`, `ReadWritePaths=/var/lib/reflop`, `CapabilityBoundingSet=` (empty), `RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6 AF_NETLINK`, `SystemCallFilter=@system-service`, `LockPersonality=yes`, `MemoryDenyWriteExecute=yes` for `reflopd` only (not for workers: CUDA/PyTorch may need W+X mappings; dropped by ADR-013), `Delegate=yes` for worker cgroups.

### 9.9 Audit log

Per identity: graph registrations (hash, op histogram, node count), executes (duration, FLOP estimate, peak memory), bytes up/down, validation failures (stage, code), rail hits, forced kills. Never tensor contents. Viewable via `reflopd audit` and `reflopd top`. Default retention: 30 days (§20).

---

## 10. Data plane (bootstrapped data)

### 10.1 Concept

1. The client converts its dataset into tensors **on the client** (decoding images, tokenizing text, resizing) — "bootstrapping".
2. The result, a **TensorPack**, is uploaded once.
3. Training steps sample batches server-side with `reflop::dataset_gather`. Per step, only indices or a seed cross the wire.

### 10.2 TensorPack format

```
mypack/
  manifest.json
  shard-00000.safetensors
  shard-00001.safetensors
  ...
```

| Manifest field | Constraint |
|---|---|
| `format_version` | Supported versions only |
| `tensors[]`: `name`, `dtype`, `shape` (sample axis first), `shards[]` | `name` matches `[a-z0-9_]{1,64}` (no path characters). Sharded along the sample axis. |
| `shards[]`: `file`, `sha256`, `size_bytes` | `file` matches `shard-[0-9]{5}\.safetensors` |
| `num_samples` | Equals the sum of first dimensions for every tensor |

Server-side safetensors validation (strict, our own checks even though the format is data-only): header length ≤ 1 MiB; JSON header schema; dtype whitelist; every tensor's `data_offsets` in-bounds, non-overlapping, and consistent with `dtype × shape`; no trailing garbage; `__metadata__` ignored.

### 10.3 Upload protocol

- 4 MiB chunks, each SHA-256 verified. Content-addressed storage enables dedup and resume.
- The server chooses the storage path: `<store>/<identity_id>/<sha256>` (I7). Client-provided names are metadata only.
- Disk free floor (§9.6) is checked continuously. On breach the upload is aborted and partial data removed.
- A dataset handle is issued only after full validation (§10.2).

### 10.4 Server-side loading

- Shards are memory-mapped. `dataset_gather` gathers rows into pinned host buffers and copies asynchronously to the device.
- GPU-resident mode: if the dataset fits on the device, it can be loaded once and gathered on the GPU.
- Augmentation is expressed as tensor ops (seeded crop via slicing, flips, normalization, mixup). The plan had the client library provide `reflop.augment` helpers that emit such ops; they were not built, and ordinary tensor code does the same job.

### 10.5 Client tooling

- `reflop data build --from <module:function> --out mypack/`: runs the user's local Python (e.g. iterates a torchvision dataset) **on the client** and writes a TensorPack.
- `reflop data push mypack/ --name cifar10`, `reflop data ls`, `reflop data rm <name>`.

---

## 11. Client design

### 11.1 Capture strategy (decided in M0, ADR-002)

| Option | Mechanism | Pros | Cons |
|---|---|---|---|
| **A. Lazy dispatch subclass (chosen)** | `RemoteTensor` as a `torch.Tensor` wrapper subclass (`_make_wrapper_subclass`) with `__torch_dispatch__`; every aten op is recorded lazily, like XLA's LazyTensor | Works with eager code; parameters live remotely naturally; autograd runs above dispatch so backward ops are captured as ordinary ops | Must handle functionalization, views, `nn.Parameter` wrapping, optimizer in-place updates; Python overhead per op |
| B. `torch.compile` backend | Custom backend receives FX graphs via AOTAutograd (already functionalized and decomposed) | Whole-graph capture, fewer round trips, very clean IR | Graph breaks; parameters must still be remote handles, so it is combined with A |

Plan: A is the primary path (M2). B is added as an accelerator path (M4) that reuses A's handle model. (ADR-002: A is implemented; B is replaced by `reflop.capture`, because TorchDynamo cannot trace `RemoteTensor`.)

### 11.2 Tracing pipeline

1. **Intercept:** any aten op with a `RemoteTensor` argument is recorded. Shapes/dtypes come from running the op on the `RemoteTensor`'s inner meta tensor.
2. **Decompose:** if the op is not in the negotiated opset, apply PyTorch's Core ATen decomposition table plus reFLOP's own decompositions, recursively, until all ops are in the opset. If impossible: placement fallback (§13) or `UnsupportedOperatorError` naming the op and suggesting alternatives.
3. **Functionalize:** convert in-place ops and view mutations to functional ops plus explicit "rebind" outputs (consider stacking PyTorch's functionalization mode under the lazy mode; evaluate in M0).
4. **Rewrite restricted patterns:** e.g. `as_strided` → view chains (§8.4).
5. **Lift varying scalars:** Python floats that change across steps (learning rate, step count, loss scale) become scalar graph inputs, not constants, so the graph hash stays stable.
6. **Canonicalize and hash** (§7.3).

### 11.3 Parameters, optimizers, AMP

- `model.to(reflop.device())` uploads parameters and buffers once and replaces them with `RemoteTensor`s bound to handles.
- `optimizer.step()`: in-place updates are captured and functionalized. Where the client can recognize a standard optimizer (Adam, AdamW, SGD, Adagrad), it emits `reflop::fused_*` (one node) instead of hundreds of `_foreach` ops.
- `torch.amp.autocast` works on the client (it acts above dispatch). `GradScaler` maps to `reflop::amp_*` ops.
- Checkpointing: `reflop.fetch(model.state_dict(), "ckpt.safetensors")` downloads and saves locally in safetensors format only.

### 11.4 Sync points

| Client action | Handling |
|---|---|
| `.item()`, `.tolist()`, `float()`, `int()`, `bool()` (e.g. Python `if` on a tensor) | Flush, execute, `Fetch` the value |
| `.cpu()`, `.numpy()`, `print(t)`, `repr(t)` | Flush, execute, `Fetch` the tensor (protocol limits apply; large fetches warn) |
| `DDS` ops (`nonzero`, boolean indexing, `masked_select`, `unique`, `repeat_interleave` with tensor repeats) | Flush up to the op, two-phase execution (§9.3) |
| `reflop.step_boundary()` | Flush (makes traces step-aligned → high cache hit rate) |
| Trace length threshold | Automatic flush |

### 11.5 Graph cache and pipelining

- The client keeps a hash → registered map per session. Steady-state steps send only `Execute`.
- Executes are pipelined: the client does not wait for step N before submitting step N+1 unless a sync point needs a value.

---

## 12. Client protection (I9)

| Threat | Mitigation |
|---|---|
| Server impersonation / MITM | Server public key pinned at pairing (§15). Mismatch is a hard failure, never a warning-and-continue. |
| Reverse deserialization RCE | Server responses are protobuf + raw tensor bytes only. No `pickle`, no `torch.load`, no `eval`. Checkpoints are safetensors only. |
| Size bombs | Hard caps on every response; decompressed-size caps if compression is used |
| Path injection | The client never builds file paths from server strings. Downloads go only to user-specified paths. |
| Terminal injection | ANSI/OSC/C0/C1 control sequences are stripped from all server strings (errors, events, Owner messages) before printing |
| Server-initiated actions | `Events` are fixed-schema and informational; nothing in them triggers client behavior |
| Information leakage | No env vars, paths, hostnames, source code or telemetry are sent (I12) |
| Key storage | Client private key in the OS keychain where available (macOS Keychain), otherwise a `0600` file |
| Wrong results | Out of scope (server is trusted), but `reflop verify` cross-checks small graphs against local CPU execution for diagnostics |

---

## 13. CPU work and dynamic placement

### 13.1 What can and cannot run on the server

| Work | On the server? | Why |
|---|---|---|
| Tensor ops on GPU | Yes | Primary path |
| **Tensor ops on CPU** | **Yes** | The same FlopIR, executed with PyTorch CPU kernels. Same security properties as GPU. |
| Scalar Python logic, branches, loops, string processing, file I/O | No → client | Expressing these requires a general-purpose program, i.e. code |
| Arbitrary C/C++ routines | No → client (the C API in M7 lets C/C++ express *tensor* work explicitly) | Same reason |
| Media decoding | No → client | N7 |

"A CPU is also a calculator, so why not?" The deciding factor is **representation, not hardware**. Computation expressed as a DAG over a closed operator set can run on the server CPU. Computation that needs branches, loops and arbitrary memory access *is* a program, and running untrusted programs contradicts "no RCE".

### 13.2 Long-term option: WASM sandbox (M8, research, off by default)

C/C++/Rust compiled to `wasm32`, executed in wasmtime with no WASI imports (no files, no network) and fuel/memory limits. This is **"code in a sandbox", not "FLOPs only"**: the security argument changes from I1 to sandbox isolation. It must be explicitly enabled by the Owner and labeled as a different security tier everywhere.

### 13.3 Placement policy (M6)

- Placement granularity is a **tensor or trace, never a single op** (per-op ping-pong destroys performance).
- Default: ops on `RemoteTensor`s run remotely; ops on ordinary tensors run locally. Mixed ops move the smaller side (small local tensors become inline constants or uploads).
- Cost model using values measured by `reflop init`: `t_local = F / P_local`, `t_remote = F / P_server + bytes / bandwidth + RTT`, with hysteresis.
- User override: `with reflop.placement("auto" | "remote" | "local" | "remote-cpu"):`.

---

## 14. Compatibility: capability negotiation

The client never builds CUDA code, so the server's CUDA, driver and PyTorch versions are internal details. The client needs to know only which operators, dtypes and devices are available.

### 14.1 `ServerCapabilities` (Hello response)

| Field | Purpose |
|---|---|
| `protocol_versions[]`, `opset_versions[]` | Compatibility decision |
| `enabled_op_ids[]` | Exact enabled set (the Owner may disable individual ops; Tier 2 depends on hardware) |
| `capabilities[]` | `linalg`, `complex`, `fp8`, `rnn_fused`, `ctc`, `flash_attention`, `efficient_attention`, `cudnn_attention`, … |
| `devices[]` | kind (cuda/cpu), name, total memory, compute capability, supported dtypes (bf16, fp16, fp8 variants, complex) |
| `numerics` | TF32 default, deterministic mode support |
| `limits` | Protocol sanity limits, active safety rails, and this identity's quotas (usually "none") |
| `runtime_info` | torch/CUDA/driver versions — **informational only**, never used in client logic |

### 14.2 Client behavior

- `reflop init` caches capabilities in `.reflop.toml`.
- On an unsupported op/dtype at trace time: (1) decompose, (2) placement fallback, (3) clear error naming the op and the missing capability.

---

## 15. Network, identity and pairing

### 15.1 What replaces "handing over SSH"

| `recom` (SSH) | reFLOP |
|---|---|
| Guest key in an `authorized_keys` that grants a shell | No shell. Daemon mode: key registered in reflopd's identity DB. SSH pipe mode: forced-command + `restrict` line. |
| Shell = anything goes | Protocol = whitelisted operators only |
| Revoking = manual file editing | `reflopd revoke <name>` (immediately kills sessions; removes the forced-command line in SSH mode) |
| No auditing | Per-identity audit log |

### 15.2 Pairing (daemon mode)

1. Owner: `reflopd invite --name alice [--expires 24h]` prints a one-time invite code encoding (server address(es), server public-key fingerprint, one-time secret, expiry).
2. Guest: `reflop pair <code>` generates an Ed25519 key pair, connects with the server key pinned, authenticates with the one-time secret (an HMAC over the pinned TLS channel; ADR-006 explains why not a PAKE such as SPAKE2), and registers its public key.
3. Afterwards: TLS 1.3 with the server key pinned and client key authentication (an Ed25519 challenge-response over TLS, ADR-006). Nonces prevent replay. Invite secrets are single-use.

### 15.3 Exposure

- Default bind `127.0.0.1`. Exposure requires an explicit `--listen` address.
- Recommended: bind only to the Tailscale/WireGuard address. reflopd's own authentication is still mandatory regardless (defense in depth).
- Public-internet exposure is allowed but documented with warnings.

### 15.4 SSH pipe mode

- `reflopd invite --name alice --via ssh` prints the exact line for the target `authorized_keys`: `restrict,command="<abs path>/reflopd serve --stdio --identity <id>" ssh-ed25519 AAAA…`. With Owner confirmation, `reflopd` appends it atomically.
- The client connects with `ssh -T` and speaks the same protocol over stdio, with an inner protocol-level authentication step as well (so a misconfigured `authorized_keys` line without the forced command does not silently grant access to reFLOP identities).
- `reflopd doctor` scans `authorized_keys` for reflop-managed keys **missing** `restrict` or `command=` and reports them as critical.

---

## 16. CLI and Python API

### 16.1 Server (`reflopd`, Owner)

| Command | Purpose |
|---|---|
| `reflopd up` | First run: detect, fetch pinned runtime, selftest, start serving, print an invite. Later runs: start serving. |
| `reflopd install --system` | Optional Level 1 install (sudo) |
| `reflopd status` / `reflopd stop` | Control |
| `reflopd invite --name … [--via ssh] [--expires …]` | Pairing |
| `reflopd users` / `reflopd revoke <name>` / `reflopd users delete <name>` | Identity management (ADR-024) |
| `reflopd quota set <name> …` / `reflopd quota clear <name>` | Optional quotas (off by default) |
| `reflopd rails show` / `reflopd rails set …` | Safety rail configuration |
| `reflopd pause [--now]` / `resume` / `kill …` / `yield --auto` / `say "…"` | Owner controls |
| `reflopd top` / `reflopd audit [--name …]` / `reflopd usage [name]` | Monitoring, compute used per Guest |
| `reflopd doctor` | Runtime, GPU, sandbox layers, SSH-mode configuration checks |
| `reflopd selftest [--security]` | Functional tests; security corpus and isolation tests |
| `reflopd runtime install` / `list` / `prune` | Owner-only runtime install or upgrade (pinned hashes from this release) |
| `reflopd update` / `reflopd version` | Self-update from a signed release (ADR-010) |
| `reflopd uninstall [--system]` | Remove the Level 0 user service or the Level 1 system service |

### 16.2 Client (`reflop`, Guest; `uv add reflop` or `pip install reflop`)

| Command / API | Purpose |
|---|---|
| `reflop pair <code>` / `reflop servers` / `reflop key` | Register with a server; list paired servers; print this machine's public key |
| `reflop init` | Hello, cache capabilities, measure bandwidth/RTT/local performance, write `.reflop.toml` |
| `reflop data build / push / ls / rm` | TensorPacks (§10) |
| `reflop status` / `reflop usage` / `reflop doctor` | Diagnostics, compute used on the server |
| `reflop update` / `reflop version` | Update the client from a signed release (ADR-010, ADR-021) |
| `reflop run python train.py` | Runs **locally** with reFLOP auto-enabled (shim mode, best effort, M6). Output and docs state clearly that the program runs on the client. |
| `reflop verify` | Cross-check small graphs against local CPU |
| `dev = reflop.device()` | Remote device handle (`model.to(dev)`, `x.to(dev)`) |
| `reflop.dataset("cifar10")` | Server-resident dataset object: `.batch(indices)`, `.loader(batch_size, shuffle=True, seed=…)` |
| `reflop.step_boundary()` | Explicit flush |
| `reflop.capture(fn)` | Capture a whole training step once and replay it (ADR-002) |
| `reflop.fetch(obj, path)` | Download to a local safetensors file |
| `reflop.placement(…)` | Placement override (§13) |
| `reflop.seed(n)` | Session RNG seed |

**Shim mode (M6, best effort):** maps `"cuda"` device requests to the `reflop` device so unmodified code (`model.cuda()`) works. Full emulation of `torch.cuda.*` is not guaranteed; the explicit API is the supported path.

---

## 17. Testing strategy

### 17.1 Principles

Security tests are first-class and mandatory in CI. Every bug fix includes a regression test. Every security bug adds a reproducer to `tests/security/corpus/`.

### 17.2 Conformance (correctness)

- Per op: random valid inputs (including edge shapes: zero-size dims, size-1 broadcast, non-contiguous inputs via views), server result vs local CPU reference, per-dtype tolerances.
- End-to-end: MLP/MNIST, ResNet-18/CIFAR-10, a small character-level GPT, a small U-Net (exercises upsampling/grid ops), an LSTM (Tier 0 decomposition and `rnn_fused`). Loss curves must match local runs within defined statistical tolerance at fixed seeds.

### 17.3 Security corpus (`tests/security/`) — expected result: rejected, server unaffected

| Category | Cases |
|---|---|
| Decode | Truncated protobuf, huge length prefixes, unknown fields, deeply nested messages |
| Structure | Unknown `op_id`, Tier 2 op without capability, forward/cyclic references, out-of-range output index, node count = limit + 1 |
| Schema | Bad dtype, out-of-range `dim`, negative sizes, `numel` overflow (`[2^40, 2^40]`), unknown enum codes, string-like payloads in numeric fields |
| Index (`IDX`) | For **every** `IDX` op: −1, size, INT64_MAX, INT64_MIN, empty index tensors, wrong index dtype; sentinel abuse (`ignore_index`, `padding_idx`) |
| Restricted | `as_strided` reaching past storage, negative strides, overflowing stride products; overlapping `empty_strided`; `_embedding_bag` decreasing offsets; `masked_scatter` short source; integer division by zero; out-of-range shift amounts |
| Uninitialized | `empty*` results must be all zeros |
| Resource / rails | Peak > device memory, upload breaching disk floor, host-RAM floor breach, graph/message size limits |
| Handles | Guessing/reusing another identity's handles, freed handles, reserved ids, cross-session access |
| Data | safetensors header bombs, overlapping/out-of-range offsets, manifest names with `../` or `/`, hash mismatches, shard count mismatch |
| Auth | Unknown key, revoked key, expired/reused invite, replay, wrong pinned server key |
| SSH mode | Missing `restrict`/`command=` detection, `SSH_ORIGINAL_COMMAND` abuse, attempts to get a PTY or port forward |
| Client protection | ANSI/OSC sequences in server strings, oversized responses, unexpected message types |

**Fuzzing:** atheris for the worker decoder/validator, Go native fuzzing for `reflopd` message handling and safetensors/manifest parsing. Short runs in CI, long runs nightly.

### 17.4 Static analysis

AST rules over server-side Python (`tests/security/static/check_worker.py`, authoritative; `semgrep.yml` next to it is an optional mirror): ban I1 APIs; ban `getattr`/`__getattribute__`/`importlib` with non-literal arguments in worker code; ban `subprocess`, `os.system`, `os.exec*`, `pty`; ban `pickle`-family imports; ban `torch.load`. CI fails on any match.

### 17.5 Isolation tests (I5)

A **test-only build hook** (excluded from release builds) executes probes inside a sandboxed worker. Each must fail: creating TCP/UDP sockets, connecting to loopback, reading `~/.ssh` / `/etc/shadow`, writing outside scratch, `execve`, `fork`, `ptrace` on `reflopd`, signaling another worker, opening other `/dev` nodes. Run at both Level 0 and Level 1, on a kernel with and without Landlock network support, and with user namespaces restricted (Ubuntu 24.04 default).

### 17.6 Performance (`tests/perf/`)

| Metric | Target (reference environment, §19) |
|---|---|
| Upload bytes per step after warm-up | ≤ 16 KiB (ResNet-18, index-based batching) |
| Per-step client overhead on cache hit (excluding RTT) | ≤ 5 ms |
| Throughput vs running locally on the server GPU (ResNet-18/CIFAR-10, LAN) | ≥ 70% |
| Validation time for a 10k-node graph | ≤ 1 s (once per registration) |

Targets are revisited after M0 and recorded in an ADR (revised in [SPIKE_REPORT.md](SPIKE_REPORT.md) and ADR-014).

---

## 18. Milestones and acceptance criteria

### M0 — Technical spikes (de-risk first)

- **S1:** `RemoteTensor` lazy dispatch capturing forward + `backward()` + `torch.optim.AdamW.step()` of a 2-layer MLP into a functional trace using only opset v1 ops. Explicitly test `nn.Parameter` wrapping a `RemoteTensor`, `foreach` optimizer paths, and in-place ops on views.
- **S2:** `torch.compile` custom backend + AOTAutograd capturing the same step.
- **S3:** Meta-tensor validation of the captured trace (V4) and peak-memory estimation (V5).
- **S4:** Op census: run S1 on MLP, ResNet-18, small GPT (with SDPA), U-Net, LSTM; list every op emitted; diff against §8; flag any `as_strided`/`DDS`/excluded ops.
- **S5:** Introspect the pinned PyTorch: verify every §8 name and overload exists; generate the Tier 0 list from `torch.Tag.core`.
- **S6:** Rootless sandbox prototype on Ubuntu 24.04 defaults: warm worker → Landlock + seccomp → run a CUDA matmul. Record the minimal syscall and path allowlists.

**Acceptance:** `docs/SPIKE_REPORT.md` with results, decisions (ADR-002 capture strategy, ADR-003 opset adjustments, ADR-004 sandbox implementation), and revised performance targets.

### M1 — Skeleton and security foundation

Protobuf definitions, `opset.yaml` schema and codegen, `reflopd` (identities, pairing, sessions, daemon mode), no-op worker with full sandbox (Level 0 and Level 1), audit log, static analysis in CI, `reflopd doctor`/`selftest`.
**Acceptance:** pairing works; unknown/revoked keys rejected; all isolation tests (§17.5) pass at both levels; static analysis enforced in CI.

### M2 — Minimal compute path

Tier 0 subset needed for MLP/MNIST, validation V1–V5 and E1–E3, handle table, Upload/Execute/Fetch/Free, client `RemoteTensor` + tracer + functionalization, `reflop.fetch`.
**Acceptance:** MNIST MLP trains remotely with inline data and matches local CPU within tolerance. Every included op has conformance and security tests. The relevant §17.3 categories are all rejected.

### M3 — Data plane

TensorPack build/push, strict safetensors/manifest validation, content-addressed store, resumable upload, `dataset_gather`, disk rail.
**Acceptance:** CIFAR-10 TensorPack upload (including interrupted-and-resumed upload); remote training with per-step upload at index level; all data-category security tests rejected.

### M4 — Full Tier 0 + Tier 1 and performance

Complete Tier 0 and Tier 1, graph cache, pipelining, compile path (S2), fused optimizers, SDPA, AMP.
**Acceptance:** §17.6 targets (as revised) met; ResNet-18/CIFAR-10, small GPT, U-Net train successfully.

### M5 — Multi-user, Owner controls, SSH pipe mode

Concurrent identities, pause/resume/kill/yield/say, optional quotas, `reflopd top`/`audit`, SSH pipe mode with `doctor` checks.
**Acceptance:** two Guests training concurrently plus the Owner's own GPU job; contention produces clean `OUT_OF_MEMORY` errors, never crashes; cross-tenant tests pass; SSH-mode security tests pass.

### M6 — CPU execution and placement

Server CPU device path, placement cost model, `reflop.placement`, shim mode.
**Acceptance:** correctness and performance tests for remote-CPU and auto placement scenarios.

### M7 — C/C++ API

`libreflop` C ABI (graph builder, sessions, upload/fetch) and C++ header wrapper. No server changes.
**Acceptance:** C examples (matmul, MLP forward) execute remotely and match references.

### M8 — Optional / research

`reflop-worker-native` (C++ libtorch), Tier 2 expansion, WASM sandbox for CPU programs (§13.2, off by default), full `DDS` support.

---

## 19. Reference environment

| Role | Spec |
|---|---|
| Server (Owner) | AMD Ryzen 9 9950X, 64 GB DDR5, NVIDIA RTX 5070 (Blackwell consumer GPU), 2 TB NVMe, Linux (test on Ubuntu 24.04 defaults, including restricted unprivileged user namespaces) |
| Client (Guest) | Apple M2 MacBook Air (8-core CPU, 8-core GPU, 16 GB), macOS, no CUDA |
| Network | Same LAN, and remote via Tailscale (measure both) |

Supported client OS in v1: macOS, Linux. Server: Linux + NVIDIA. The pinned runtime must include a PyTorch/CUDA build that supports the server GPU architecture; `reflopd up` verifies this and `reflopd doctor` reports it. The runtime variants and their driver and compute-capability ranges are in ADR-008.

---

## 20. Open questions

All of these are settled. New questions go here first (R4), with a default until the human decides.

| ID | Question | Outcome |
|---|---|---|
| O1 | Project name | `reflop` (written reFLOP in prose) |
| O2 | Worker language long-term: Python runtime vs native C++ | Python worker; a native worker stays an M8 option |
| O3 | License | Apache-2.0 |
| O4 | Pinned PyTorch version for the server runtime and the minimum client PyTorch version | Server: PyTorch 2.15.0 (2.14.1 for the CUDA 12.6 variant), ADR-008. Client: `torch>=2.6` |
| O5 | Should Level 1 (system install) be recommended in the README, or Level 0 presented as the main path? | Level 0 is the main path; Level 1 is recommended for public exposure (`docs/OWNER_GUIDE.md`) |
| O6 | Guest limits | No quotas by default; safety rails on (§9.6). Rail defaults confirmed in ADR-009 |
| O7 | SSH pipe mode in v1 (M5) or later | Shipped in M5 |
| O8 | Session state TTL after disconnect | 30 minutes (`session.ttl`) |
| O9 | Scope of `DDS` support in v1 | Two-phase execution, ADR-012 |
| O10 | Audit log retention | 30 days (`audit.retention_days`) |
| O11 | Ship Tier 2 enabled by default when the hardware supports it? | Yes, except `rnn_fused` (`_cudnn_rnn` validation complexity), which stays off |
| O12 | Should the Owner also use reFLOP for their own jobs (as an identity) or run PyTorch directly? | Direct use; reFLOP does not manage the Owner's own processes (`reflopd yield --auto` can pause Guests while the Owner uses the GPU) |

---|---|---|
| O1 | ~~Project name~~ | **Resolved: `reflop`** |
| O2 | Worker language long-term: Python runtime vs native C++ | Python for M2–M6; native evaluated in M8 |
| O3 | ~~License~~ | **Resolved: Apache-2.0** |
| O4 | Pinned PyTorch version for the server runtime and the minimum client PyTorch version | Latest stable at M0 (docs consulted at planning time were 2.15) |
| O5 | Should Level 1 (system install) be recommended in the README, or Level 0 presented as the main path? | Level 0 main path; Level 1 recommended for public exposure |
| O6 | ~~Guest limits~~ | **Resolved: no quotas by default; safety rails on** (§9.6). Confirm rail defaults. |
| O7 | SSH pipe mode in v1 (M5) or later | M5 |
| O8 | Session state TTL after disconnect | 30 minutes |
| O9 | Scope of `DDS` support in v1 | Two-phase for `nonzero`/boolean indexing in M4; client fallback in M2 |
| O10 | Audit log retention | 30 days |
| O11 | Ship Tier 2 enabled by default when the hardware supports it? | Yes, except `rnn_fused` (`_cudnn_rnn` validation complexity) until M4 tests exist |
| O12 | Should the Owner also use reflop for their own jobs (as an identity) or run PyTorch directly? | Direct use is assumed; reflop does not manage the Owner's own processes |

---

## 21. Known limitations (state these in README)

1. The server sees Guest data and models in plaintext. reFLOP protects the Owner's machine from Guests; it does not protect Guests' data from the Owner (N1).
2. Bugs in whitelisted kernels (PyTorch, CUDA libraries, GPU driver) remain an attack surface. The sandbox limits the blast radius; it cannot make it zero.
3. No arbitrary CPU programs, custom CUDA/Triton kernels, or server-side media decoding.
4. Results may differ bitwise from local runs because of non-deterministic GPU kernels and TF32 settings.
5. GPU timing side channels between Guests are out of scope.
6. Killing a worker (watchdog or Owner) loses session state. Long jobs should snapshot or fetch periodically.
7. At Level 0 a sandbox escape lands in the Owner's account. Level 1 is stronger.
8. Without quotas, Guests can starve each other of GPU memory. That is by design (G8); contention yields errors, not damage.

---

## 22. Glossary

| Term | Meaning |
|---|---|
| Owner | GPU machine owner, runs `reflopd` |
| Guest | Untrusted user borrowing the GPU |
| Identity | One paired Guest |
| Session | One worker-backed unit of work |
| Handle | Id of a server-resident tensor, identity-scoped |
| FlopIR | reFLOP's wire graph format |
| Opset | Whitelisted operator set (`opset.yaml`) |
| Tier 0/1/2 | Core ATen / performance extensions / capability-gated ops |
| Trace | Lazily recorded op sequence on the client |
| Sync point | Where the client must wait for server results |
| TensorPack | Client-bootstrapped dataset (safetensors shards + manifest) |
| Safety rail | Always-on limit that protects the machine, not fairness |
| Quota | Optional fairness limit, off by default |
| Capability negotiation | Exchanging supported ops/dtypes instead of CUDA versions |
| Level 0 / Level 1 | Rootless default deployment / optional system install |
| SSH pipe mode | SSH used only as a forced-command transport, never a shell |

---

## 23. Repository layout

The layout as built (the plan's original sketch had `PLAN.md` at the root and a separate `server/sandbox/`; sandboxing lives in `server/reflopd/internal/sandbox` instead).

```
reflop/
├── LICENSE                    # Apache-2.0
├── NOTICE
├── README.md, CONTRIBUTING.md, SECURITY.md, CHANGELOG.md
├── install.sh                 # client and server installer (ADR-010)
├── docs/
│   ├── PLAN.md                # this file
│   ├── DECISIONS.md           # ADRs
│   ├── SPIKE_REPORT.md        # M0 results
│   ├── THREAT_MODEL.md        # §2 expanded
│   ├── OPSET.md               # generated from opset.yaml
│   ├── OWNER_GUIDE.md
│   └── GUEST_GUIDE.md
├── proto/reflop/v1/           # graph, compute, auth, data, capabilities, worker .proto
├── opset/v1/opset.yaml        # single source of truth (I10)
├── codegen/                   # opset.yaml → server op table, client op mapping, OPSET.md; runtime lock
├── pkg/                       # Go shared by reflopd and libreflop: reflopv1, wire, opset, sshhost
├── server/
│   ├── reflopd/               # Go: cmd/reflopd, internal/{auth,sshpipe,server,store,rails,usage,supervisor,sandbox,cgroup,runtime,gpu,audit,config,selfupdate,cli,…}
│   └── worker/reflop_worker/  # Python: main, wire, graph, meta, cost, execute, handles, validators, natives, datasets, safetensors, sandbox, ops_generated
├── client/python/reflop/      # device, tensor (RemoteTensor), trace, capture, decomp, optim, natives, _placement, shim, client, data/, cli/, update
├── clib/                      # M7: libreflop (Go c-shared, ADR-017) + C and C++ headers
├── scripts/sign-release.sh    # offline release signing
├── spikes/                    # M0 spikes
└── tests/
    ├── conformance/
    ├── security/{corpus,fuzz,isolation,static}
    ├── client/
    ├── e2e/
    └── perf/
```

---

## 24. Licensing and trademarks

- reFLOP is licensed under **Apache-2.0**. Include `LICENSE` and `NOTICE`; add SPDX headers (`SPDX-License-Identifier: Apache-2.0`) to source files.
- `recom` is GPL-3.0. **Do not copy code from `recom` into reFLOP** unless the human confirms they are the sole copyright holder of that code (a sole author may relicense their own code; code contributed by others to `recom` may not be relicensed). Record any reuse in an ADR (done: ADR-007).
- Dependencies must be Apache-2.0-compatible (e.g. PyTorch BSD-style, gRPC Apache-2.0, safetensors Apache-2.0, protobuf BSD-3). Do not vendor GPL/AGPL code.
- "NVIDIA" and "CUDA" are trademarks of NVIDIA Corporation. Use them only descriptively in docs ("works with NVIDIA GPUs"). Never in project, package, binary or module names.
