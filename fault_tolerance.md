# Fault-tolerant session state (Issue #2)

Status: draft plan, 2026-11-15. Branch `enhance/fault-tolerant-state`.
Nothing here is implemented yet. Once the open questions at the end are
answered, the decisions go into `docs/DECISIONS.md` as ADRs.

## Problem

A session's state lives only in its worker's memory. When the worker ends,
the Guest loses every tensor on the server: parameters, optimizer state,
buffers. Issue #2 suggests asynchronous checkpoints or fully replicated state.

## What exists today

- **Snapshot and Restore are implemented on the server but unused.**
  `Snapshot`, `Restore`, `ListSnapshots` and `DeleteSnapshot` are in
  `proto/reflop/v1/compute.proto`. `Session.snapshot` in
  `server/reflopd/internal/server/session.go` writes a temp file in the
  identity's snapshot directory, checks the disk rail and renames the file into
  place. The worker writes safetensors to a file descriptor
  (`core.py: snapshot_write`) and checks the header again on restore. The
  Python client exposes none of it, and no test calls these RPCs.
- **A dropped connection already survives.** A detached session keeps its
  worker for `[session] ttl` (30 minutes by default), and
  `RemoteConnection._reconnect` re-attaches with `resume_session_id`.
- **Handle ids are chosen by the client** (PLAN.md §6.1). If a new session
  restores the old ids, every `RemoteTensor` the Guest holds stays valid, so
  recovery can be invisible to the Guest's code.
- **Server tensors are never modified in place.** `KIND_REBIND` replaces the
  table entry with a new tensor (`execute.py`), and ADR-014 already relies on
  this. Holding references to the current tensors therefore gives a consistent
  point-in-time view without blocking execution. Buffer donation (PLAN.md §9.3)
  is not implemented.

### Where state is lost today

| Event | Today |
|---|---|
| Worker crash, CUDA context error | `SESSION_LOST` |
| Watchdog timeout | worker killed, `SESSION_LOST` |
| `reflopd kill`, `reflopd pause --now` | worker killed |
| Detached session evicted for GPU memory (ADR-022) | worker killed |
| Detached session past its TTL | worker killed |
| `reflopd` restart or update, host reboot | all workers end (`Pdeathsig`) |
| Guest process dies | the server session stays until its TTL, but nothing can resume it |

### What a session holds besides tensors

- **RNG state.** The worker uses torch's global generators, seeded in
  `core.py` at init and on `Reseed`.
- **Registered graphs.** The client can register them again; it tracks them in
  `Session.registered`.
- **Opened datasets.** They are on disk per identity and survive.

## The two options in the issue

### Asynchronous checkpoints: adopt as the base

Tensors are immutable, so a checkpoint costs little:

1. The worker handles requests one at a time, so a checkpoint request falls
   exactly between two executes. At that point the worker records a reference
   to every live handle and the CPU and CUDA RNG state.
2. A background thread copies the tensors from GPU to host on a separate CUDA
   stream. The stream first waits on an event recorded at the cut. The thread
   streams safetensors to the file descriptor. The header can be written first
   because all shapes and dtypes are known.
3. Execution continues meanwhile. A rebind drops the table's reference, but the
   checkpoint's reference keeps the old version alive until it is copied.

Details:

- **GPU memory.** Old versions of rebound tensors stay allocated until they are
  copied. If an execute fails with an OOM while a checkpoint is running, the
  worker finishes the copy, drops those references and retries once before it
  reports `OUT_OF_MEMORY`.
- **Host memory.** Stage into pageable host memory up to the host-RAM rail;
  past that, stream through a small pinned ring buffer (for example two
  64 MiB buffers). Pinned memory cannot be swapped out, so it must stay small
  to protect the Owner's desktop.
- **Allocator.** Call `record_stream` for the copy stream, or synchronize it,
  before dropping references, so the caching allocator does not reuse memory
  that a copy is still reading.
- **Donation.** If buffer donation is implemented later, a tensor referenced by
  a running checkpoint counts as another use and must not be donated.

### Fully replicated state: reject as stated

| Replica | Why not |
|---|---|
| Standby worker on the same GPU | Doubles memory and compute, and covers only a worker crash |
| State streamed to the client | Breaks the target of a few KB of traffic per step |
| State kept in `reflopd` memory | Puts Guest data in `reflopd`, which must not touch tensors, and dies with it on restart |

### Replicate the inputs instead: a client redo log

The client already sends every request. It keeps a redo log of everything sent
since the last confirmed checkpoint: graph hashes and bindings, inline tensors,
uploads, frees and reseeds. Recovery:

1. Open a new session.
2. `Restore` the checkpoint into the same handle ids.
3. Register the graphs again.
4. Replay the log.

Recovery replays everything sent since the checkpoint, so it does not lose the
work done between checkpoints. With `Numerics.deterministic` set, the replay is
bit-identical. Without it, the result equals a nondeterministic rerun.

## Design

### Protocol

All additions sit behind a new capability string. A client talking to an older
server turns recovery off (I11).

- **`Checkpoint{}`** saves all live handles plus RNG state, asynchronously. The
  response `CheckpointSaved{bytes}` arrives once the file is durable.
- **A checkpoint is a barrier.** Requests sent after it, uploads included, stay
  behind it. Today uploads skip the queue that holds requests during a pause
  (`Session.forward`). Without the barrier, an upload could reach the worker
  before an earlier checkpoint, end up inside it, and fail with
  `HANDLE_EXISTS` when replayed.
- **`Restore{name, all: true}`** restores every tensor in the file. The key
  is the handle in hex and maps back to the same handle. Each handle still
  goes through `check_new`.
- **Reserved keys.** The RNG state is stored under the prefix `__reflop__/`.
  `Snapshot` rejects Guest keys with that prefix. Restore checks the sizes of
  the RNG tensors before it sets them.
- **Lost reason.** `SESSION_LOST` carries a reason that is either
  *recoverable* or *final*:
  - recoverable: crash, CUDA fault, eviction, TTL, `reflopd` restart, and
    probably `pause --now` (open question 3);
  - final: `reflopd kill`, revoke, unpair. The client must never bring back a
    session the Owner ended on purpose.

### Worker

- Checkpoint thread as described above. Threads are allowed by the seccomp
  filter (`clone` with `CLONE_THREAD`). The worker writes only to the file
  descriptor that `reflopd` passes, as `snapshot_write` does today.
- RNG state is saved as `uint8` tensors from `torch.get_rng_state()` and
  `torch.cuda.get_rng_state(device)`.
- One checkpoint at a time per session. A second request waits for the first.

### reflopd

- The file name is chosen by the server: `auto-<session id>.safetensors` in the
  identity's snapshot directory. It keeps one checkpoint and replaces it by
  atomic rename. The monitor sweeps orphaned auto checkpoints after
  `keep`.
- Disk rail and the optional storage quota apply as for snapshots, plus a
  per-identity `max_bytes` for auto checkpoints.
- I/O priority for workers (ioprio set before exec, or cgroup `io.max`) so a
  large checkpoint does not stall the Owner's disk. To be measured first.
- `reflopd` also uses checkpoints itself:
  - **Spill instead of kill.** Before it evicts a detached session (ADR-022) or
    ends one at its TTL, it checkpoints the session. A later resume falls back
    to restoring that checkpoint.
  - **Drain.** On SIGTERM, `reflopd update` and `pause --now`, it checkpoints
    every session within a deadline (for example 30 s), then stops the
    workers.
  - A hung kernel cannot be checkpointed. A watchdog kill falls back to the
    last periodic checkpoint.
- After a restart, `reflopd` does not need to remember sessions. The client
  presents the checkpoint name, and `reflopd` opens the file only from the
  calling identity's directory (I7, I8).

### Client

- `reflop.device(..., checkpoint="auto")` checkpoints at `step_boundary()`
  every N seconds or when the redo log passes a size limit. A large upload
  triggers an early checkpoint so the log does not have to keep its bytes.
- `reflop.checkpoint()` starts one explicitly and returns a future.
- The log is trimmed when `CheckpointSaved` arrives.
- During replay, results the Guest already received are discarded. Requests
  that were in flight when the session was lost get their futures resolved from
  the replay.
- One recovery attempt per failure, then the error is raised, so a graph that
  crashes the worker cannot loop.
- Named snapshots (`reflop.save(name, ...)` and `reflop.load(name, ...)`) cover
  a Guest process that dies, where the redo log is gone too. Keys follow
  `state_dict` paths. Non-tensor parts of an optimizer `state_dict` stay on the
  Guest machine.

### Owner configuration

```toml
[session.checkpoint]
enabled = true         # allow Guest auto checkpoints
max_bytes = 0          # per identity; 0 = only the disk rail
keep = "7d"            # how long an orphaned auto checkpoint is kept
spill_detached = true  # checkpoint a detached session before ending it
```

## Security

- **Unchanged invariants.** I1, I6 and I9 hold: checkpoints are safetensors
  written by the worker, and the worker checks the header again on restore.
  Replay goes through full validation; no per-execute check is cached. I13
  holds: restored tensors come from file bytes.
- **New risks and their limits:**
  - Owner disk space and I/O: disk rail, `max_bytes`, I/O priority.
  - Pinned host memory: a small fixed ring, counted against the host-RAM rail.
  - A Guest forcing other Guests' detached sessions to spill. This replaces
    ADR-022's eviction, which destroys state, so it is no worse than today.
  - Checkpoint files hold Guest data at rest: mode 0600, deleted on session
    `CloseSession` for auto checkpoints, and on identity deletion (ADR-024).
- **Tests under `tests/security/`:**
  - a Guest key with the `__reflop__/` prefix is rejected;
  - one identity cannot restore another identity's checkpoint;
  - a truncated or corrupt checkpoint fails closed;
  - a session ended by `reflopd kill` is never revived;
  - recovery stops after one attempt when replay crashes the worker again.

## Phases

0. **Client API for the existing Snapshot and Restore**, plus their first
   tests (Go and Python). This already covers a Guest script that crashes and
   starts over.
1. **Asynchronous checkpoints:** worker thread, `Checkpoint` RPC, barrier,
   reserved keys, rails, rotation, capability string.
2. **Client redo log** and transparent recovery, with recoverable and final
   reasons.
3. **`reflopd` uses checkpoints:** spill detached sessions, drain on shutdown,
   update and `pause --now`.
4. **Fault-injection end-to-end test:** `kill -9` the worker mid-training on
   CPU with deterministic numerics, and check that the parameters are
   bit-identical to an uninterrupted run.

Each phase adds its tests, an ADR or an ADR update, and a
`CHANGELOG.md` entry.

## Open questions

1. Auto checkpoint default: allowed by default on the Owner side but off in the
   client unless the Guest asks? It costs the Owner disk space.
2. Is the client redo log in scope for the first release, or are checkpoints
   alone, losing up to one interval, enough?
3. Is `pause --now` recoverable? Proposed: yes; `reflopd kill` stays final.
4. One ADR for the whole design, or one per phase?
