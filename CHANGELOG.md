# Changelog

## Unreleased

- `reflop.seed(n)` seeds all open sessions and applies the same seed to future
  sessions. A CPU session opened by `placement("remote-cpu")` leaves the
  default session in place, so calls such as `reflop.load(name)` still use it.
- Reading a large unflushed tensor with `.cpu()` downloads it in chunks.
  The read no longer fails when the result is too large for one execute
  response frame.
- `placement("remote-cpu")` uses the selected device's Owner and Guest
  identity when it opens or reuses a CPU session.

## 0.2.6 (2026-11-21)

A Guest can no longer get another Guest's session killed by using up host
RAM, and operations that run on the client no longer lose their writes to
reFLOP tensors. Owners: update reflopd for the memory fix; the placement fix
is in the Guest client.

Updating from 0.2.0 still needs the version named (see 0.2.1). From 0.2.1 or
later, plain `update` works.

Owner:

- One Guest can no longer get another Guest's session killed by using up
  host RAM (#16). Workers shared one cgroup memory limit, so when it was
  reached the kernel killed the largest worker, which was not always the one
  growing. Each worker now also gets its own limit, what it uses plus 7/8 of
  the pool's free headroom, which the RAM rail updates twice a second; a
  worker that keeps growing reaches its own limit first and is the one
  killed. Suspended workers have no limit of their own while their GPU memory
  is in host RAM. A new worker's budget is the free headroom, not the whole
  pool.
- Inside the worker, all uploads in progress count together against the
  host budget (each one allocates its whole buffer up front, so parallel
  uploads could each take a full budget), finishing an upload no longer
  copies the data twice more (about 3x the tensor at the peak), and a fetch
  holds about two copies of the data instead of four. Needs the Owner to
  update reflopd.

Guest:

- An operation that runs on the client and writes to a reFLOP tensor updates
  it. Running locally works on fetched copies of the inputs, and an in-place
  operation, an `out=` argument or a write through a view changed only the
  copy: the tensor silently kept its old value. In the default `"auto"`
  placement this happened when the cost model ran such an operation locally
  (`x.scatter_add_(0, index, src)` with a small `x` and large client
  tensors); under `placement("local")` it happened for every in-place
  operation, so training there never updated its parameters. Writing
  operations now run their functional form (locally when placement says so)
  and write the result back, and views of a reFLOP tensor stay views of it,
  also under `"local"` (ADR-034).

## 0.2.5 (2026-11-21)

A value read after a request that failed on the server is never returned
stale. Only the Guest client changes; reflopd is the same apart from its
version. The README and the Guest guide have the 21 November benchmark
numbers.

Updating from 0.2.0 still needs the version named (see 0.2.1). From 0.2.1 or
later, plain `update` works.

Guest:

- A value read after a request that failed on the server is never returned
  stale. Executes are sent without waiting; when one failed (an index out of
  range found at run time, an out-of-memory error, a binding error), the
  tensors it would have written kept their old contents, and the next
  `.item()`, `.cpu()` or similar returned that old data with no error. The
  error surfaced one read later. A `reflop.capture` step whose replay failed
  returned the previous call's loss exactly (benchmark B2). Now every read
  that waits for the server, and `reflop.save` and `reflop.load`, raises an
  earlier failure first. `print(t)` still never raises; it shows the error,
  and the next read raises it (ADR-033).

## 0.2.4 (2026-11-21)

CTC loss training works, and BatchNorm statistics are updated when the
forward's output is not used. Owners: update reflopd for CTC training; the
BatchNorm fixes are in the Guest client only.

Updating from 0.2.0 still needs the version named (see 0.2.1). From 0.2.1 or
later, plain `update` works.

Owner:

- reflopd validates and runs the CTC loss backward (`_ctc_loss_backward`).
  Before, it rejected every such graph, because PyTorch has no meta kernel for
  it. The worker now supplies one, and it first checks every input shape
  exactly: PyTorch's backward kernels read `grad`, `neg_log_likelihood` and
  `log_alpha` unchecked, and a short `log_alpha` crashes the CPU kernel
  (ADR-032).

Guest:

- Training with `F.ctc_loss` works: `.backward()` used to abort the process
  (`terminate called without an active exception`), with int lengths too. It
  needs this client and a reflopd from this release; an older reflopd rejects
  the backward with "cannot be validated (no meta kernel)".
- BatchNorm's `running_mean` and `running_var` are updated when the forward
  pass's output is not used, as in a BN recalibration loop. Before, only
  `num_batches_tracked` went up. `native_batch_norm` does not mark the
  statistics as written in its schema, so the client treated it as a pure
  function, and the statistics changed only through the kernel's in-place
  side effect on the server, which is skipped when the output is unused. The
  client now records the update itself. Eager training steps with many BN
  layers are a few percent slower (ResNet-18: about 3%).
- BatchNorm in eval mode with gradients (fine-tuning with frozen statistics)
  works in a server-CPU session. It failed with "output is torch.float32[0],
  validated torch.float32[8]": PyTorch's CPU kernel returns empty saved
  statistics in eval mode, its meta kernel full ones.

## 0.2.3 (2026-11-21)

Guest fixes: default-dtype `randint`/`randperm`, `reflop.capture` after a shape
change, `F.ctc_loss` with tensor lengths, and CUDA autocast under `reflop run`.
Only the Guest client changes; reflopd is the same apart from its version.

Updating from 0.2.0 still needs the version named (see 0.2.1). From 0.2.1 or
later, plain `update` works.

Guest:

- `torch.randint` and `torch.randperm` on the reFLOP device work with their
  default dtype (int64). Before, they failed with `KeyError: 4`; an explicit
  `dtype=torch.int32` worked.
- `reflop.capture` captures again when the arguments' shapes change. Before,
  the second call with a new batch shape (a smaller last batch, say) replayed
  the old graph and failed with a BINDING error ("bound tensor is
  [16,784], the graph expects [32,784]").
- `F.ctc_loss` with lengths given as tensors works. Before, it raised
  `NotImplementedError` (no meta kernel). The lengths are read on the client,
  so lengths on the reFLOP device cost one fetch. The CTC backward pass still
  does not work on reFLOP.
- Under `reflop run`, `torch.autocast("cuda", ...)` (also
  `torch.amp.autocast` and `torch.cuda.amp.autocast`) runs on the reFLOP
  device. Before, float16 did nothing and bfloat16 raised "Torch not compiled
  with CUDA enabled". `torch.cuda.is_bf16_supported()` asks the server, and
  `torch.is_autocast_enabled("cuda")` and `torch.get_autocast_dtype("cuda")`
  report the reFLOP device.
- `reflop run` with a local worker (`REFLOP_LOCAL=cpu`, for tests and
  diagnostics) no longer deadlocks on the first `.cuda()` after
  `torch.manual_seed`.

## 0.2.2 (2026-11-19)

reflopd bounds the requests it holds while execution is paused (#15), and the
Go module moved to github.com/reflopai/reflop.

Updating from 0.2.0 still needs the version named:
`reflopd update --version v0.2.2` and `reflop update --version v0.2.2` (see
0.2.1). From 0.2.1, plain `update` works.

Owner:

- Requests a Guest sends while execution is paused are now bounded: at most
  4096 requests and twice `max_frame_bytes` per session, and 1 GiB for all
  sessions together. Past that, a request is answered with `ERROR_PAUSED`.
  Before, the queue had no limit, and a Guest that kept sending during a pause
  (including one started by `yield --auto`) could grow reflopd until its
  cgroup OOM-killed it, ending every session (#15).
- A session that ends during a pause now answers its held requests with
  `SESSION_LOST`; before, they were never answered.
- Requests sent while `resume` is still forwarding held ones queue behind
  them instead of overtaking them.

Development:

- The Go module path is now `github.com/reflopai/reflop`, matching the
  repository's new home. Go code that imports reflop packages needs the new
  path; the generated protobuf code was regenerated with the same protoc
  29.5 and protoc-gen-go 1.36.12.
- The client caches operator names on the dispatch path instead of
  formatting them several times per traced op.

## 0.2.1 (2026-11-19)

The sandbox isolation probes now run in CI (#3, ADR-031), and every link
points at the repository's new home, github.com/reflopai/reflop.

Owner and Guest:

- `reflopd update`, `reflop update` and `install.sh` download releases from
  github.com/reflopai/reflop.
- reflopd and the client at 0.2.0 or older cannot find this release by
  themselves: they fail with "could not determine the latest release",
  because they read the version from GitHub's redirect and the old address
  now redirects to the new one first. Name the version instead:
  `reflopd update --version v0.2.1` and `reflop update --version v0.2.1`.
  Both still check the signature. From 0.2.1 on, plain `update` works again.

Development:

- CI runs the sandbox isolation probes inside real sandboxed workers, with
  user namespaces off and on, on sandbox pull requests, every push to `main`
  and nightly (#3, ADR-031). In strict mode (`REFLOP_TEST_STRICT=1`) the test
  fails instead of skipping when Landlock, seccomp or the runtime is missing,
  and when a probe target does not exist.
- The CUDA isolation probes skip when the installed runtime has no CUDA,
  instead of failing.

## 0.2.0 (2026-11-18)

Session state survives more failures (#2, ADR-028, ADR-029, ADR-030).

Owner:

- `reflopd pause --now` no longer ends sessions: it suspends each GPU worker,
  moving its GPU memory to host RAM, and `reflopd resume` brings it back. A
  worker is still killed when host RAM is short, a kernel does not finish
  within 10 seconds, the driver is older than 550, or it runs on the CPU.
  `pause --now` reports what it did. `reflopd yield --auto --suspend`
  (off by default) also suspends while auto-yield pauses Guests.
- `reflopd checkpoints on|off [--interval]` lets Guests checkpoint their
  sessions. Off by default; the interval defaults to 10 minutes and can never
  be shorter than 5. Checkpoints write the Guest's whole state to disk each
  time, which wears SSDs, and double that state's GPU memory while they are
  copied: turning them on prints the worst case in red.
- Snapshots are now synced to disk before they count as saved, never replace
  the previous file on a failed write, and count against the disk rail and
  the storage quota while they are written. A restore adds all of its tensors
  or none.

Client:

- `reflop.save(name, state_dict)`, `reflop.load(name)`, `reflop.snapshots()`
  and `reflop.delete_snapshot(name)` keep tensors on the server.
- `reflop.device(checkpoint=True)` checkpoints the session at step boundaries,
  at the Owner's interval or a longer `checkpoint_interval`, and warns in
  yellow that the state's GPU memory doubles during a checkpoint. After a
  `SessionLostError`, `reflop.recover()` continues from the last checkpoint
  and returns the value set with `reflop.checkpoint_tag()`.

## 0.1.14 (2026-11-17)

On macOS, `reflop run` keeps OpenMP on one thread (#7, ADR-027).

Client:

- On macOS, `reflop run` sets `OMP_NUM_THREADS=1` for the program unless the
  variable is already set, and says so when it starts. PyTorch on macOS ships
  its own OpenMP runtime; a second one in the same process (scikit-learn,
  xgboost, conda, Homebrew) can crash or hang the program. A value you set,
  even an empty one, is kept. Linux and `import reflop` are unchanged; for
  `python train.py` on a Mac, set the variable yourself.
- `reflop doctor` lists the OpenMP runtimes your packages install and the
  ones loaded with torch, and names the package each came from. On macOS two
  loaded runtimes fail the check; a second installed one is a new `WARN` line,
  which does not change the exit status.
- After a crash on macOS, `reflop run` points at `reflop doctor` when OpenMP
  was already on one thread, and at `OMP_NUM_THREADS=1` when you chose more.

## 0.1.13 (2026-11-17)

`reflop run` and the client no longer fail silently (#8).

Client:

- When a signal kills the program `reflop run` started, reflop names the
  signal and exits with 128 + n, as a shell would. Before, it printed nothing
  and exited 245 for a segfault. Only the program name is printed, not its
  arguments. On macOS a crash also points at the OpenMP workaround (#7).
- `reflop run` sets `PYTHONFAULTHANDLER=1` for the program unless it is
  already set, so a native crash prints the Python stack of every thread.
- `reflop run` no longer dies on Ctrl-C or `Ctrl-\` before the program does, so
  the program's traceback is not lost. A SIGTERM sent to reflop is passed on
  to the program.
- With `REFLOP_DEBUG=1`, `reflop run` prints the program's pid, and
  `kill -USR1 <pid>` or `Ctrl-\` prints the Python stack of every thread
  without stopping it. This finds where a hung program is stuck.
- `REFLOP_DEBUG=1` also logs each connection step to stderr: the ssh command
  or TLS address, handshake, authentication, session open, reconnects, and
  why a connection ended. `REFLOP_DEBUG=0` now turns it off. It is documented
  in the Guest guide (Troubleshooting) and in `reflop run --help`.
- In SSH pipe mode, connection errors now end with the last lines ssh wrote
  to stderr, such as `Permission denied (publickey)` or `Connection refused`,
  sanitized and bounded (ADR-026). Before, they were thrown away.
- At exit, the client waits up to 5 seconds for requests still in flight and
  prints any error from them that was never raised. Before, a script whose last
  steps failed on the server could end with status 0 and no message.

## 0.1.12 (2026-11-15)

Unpairing (#6, ADR-025). A Guest can now pair again with a server it was
paired with, once both sides have unpaired, in either order.

reflopd:

- New `reflopd unpair <name>`: `revoke` and `users delete` in one step. It
  asks first, because it deletes the Guest's datasets and snapshots; pass
  `--yes` in scripts.
- When a Guest runs `reflop unpair`, reflopd revokes that identity at once,
  logs it, writes a `guest_unpaired` audit record, and lists it in `reflopd
  users` as "unpaired by guest". `reflopd status` shows the command that
  finishes it.
- A pairing attempt with a key that is still registered now tells the Guest,
  the daemon log and the audit record to run `reflopd unpair <name>`.
  Inviting a revoked name says the same.
- Confirmation prompts no longer treat `/dev/null` on stdin as a terminal.

Client:

- New `reflop unpair [server]`: tells the server to revoke this machine's
  key, then forgets the server and its pinned SSH host keys. The server is
  forgotten even if it cannot be reached; the message then says what the Owner
  has to run. `--local` skips the connection.
- `reflop servers remove <name>` now unpairs. Before, it removed only the
  local entry, which left the key registered on the server, so pairing with
  that server again always failed.
- When a server rejects this machine's key, the error says to run `reflop
  unpair <server>`. `reflop pair` with a server this machine is already
  paired with says to unpair first, before it uses up the invite. libreflop
  gives the same hint.
- Protocol: new messages `Unpair` and `Unpaired`.

## 0.1.11 (2026-11-12)

reflopd:

- New `reflopd users delete <name>` removes a revoked identity: it leaves
  `reflopd users`, its datasets and snapshots are deleted from disk, its
  quotas are cleared, and the name can be invited again. The audit log keeps
  its history. It refuses an active identity and says to revoke it first
  (ADR-024).
- When pairing fails on the server, the Guest still sees only "invalid,
  expired or already used invite", but the daemon log and the `pair_failed`
  audit record now give the reason: expired (with the time), already used,
  unknown to this server, a wrong secret or signature, or a malformed
  request (#5).

Client:

- `reflop pair` with an expired invite now says when the invite expired and
  what this machine's clock says, both in UTC. An invite made minutes ago that
  shows as expired usually means the Guest's clock or date is wrong; before,
  the message gave no way to tell (#5).

## 0.1.10 (2026-11-12)

reflopd:

- reflopd starts when a GPU's memory is full. Before, if other programs held
  all of a GPU's memory (for example a leaking `gnome-shell`), the self-test
  and the startup probe failed with "CUDA error: out of memory" and reflopd
  did not start at all, not even for the CPU. Now the self-test skips that
  GPU with one line that says how much memory is in use, and reflopd serves
  the other devices. It checks the GPU every 30 seconds and serves it once
  its memory is free. `reflopd status` shows it as not served until then,
  and `reflopd selftest` exits 1 when it skipped a device (ADR-023).
- `reflopd up --foreground` and `--no-service` ran detection, the runtime
  check and the self-test three times, because the daemon restarted the
  whole `up` command when moving into its systemd scope and its sandbox.
  It now restarts as `reflopd serve` with the same flags. Plain `reflopd up`
  (the user service) was not affected.
- The self-test no longer prints a failing worker's output three times, and
  workers no longer warn that NumPy is missing (the runtime leaves it out on
  purpose).

## 0.1.9 (2026-11-12)

- `reflop update` for a client installed with `uv tool` could fail right
  after a release with `there is no version of reflop==X`. It checks PyPI
  through an API that lists a release as soon as it is uploaded, but uv
  installs from PyPI's package index page, which it caches for up to 10
  minutes. `reflop update` now tells uv to fetch that page again
  (`--refresh-package reflop`). Clients in a uv project were not affected.
  If 0.1.8 or older hits this, run
  `uv tool install --force --refresh-package reflop reflop==0.1.9`, or wait
  10 minutes.

## 0.1.8 (2026-11-12)

Security fix for reflopd.

- A Guest could end other Guests' disconnected sessions. A session whose
  client drops is kept for `[session] ttl` so it can be resumed, but the
  oldest such session on a GPU is ended when an allocation there fails for
  lack of memory. Since 0.1.5 a graph that needs more memory than the whole
  device also fails with `OUT_OF_MEMORY`, although it is rejected before
  anything is allocated, so registering such graphs in a loop ended every
  disconnected session on the GPU and lost their state. Now only an
  allocation that fails while running ends a disconnected session
  (ADR-022). Upgrade with `reflopd update`. The client is unchanged apart
  from its version.

## 0.1.7 (2026-11-12)

- The client is on PyPI as `reflop`: `uv add reflop` or `pip install reflop`
  puts the library and the `reflop` command in your project's environment.
  Each version is uploaded from the signed GitHub release after the signature
  and checksums are checked (ADR-021).
- Releases include the client's sdist (`reflop-X.Y.Z.tar.gz`) next to the wheel.
- The sdist ships real `LICENSE` and `NOTICE` files. They were symlinks, so a
  wheel could not be built from it.
- `install.sh` prints `uv add reflop==X.Y.Z` instead of a release URL.
- `reflop update` follows how the client was installed. In a uv project it
  upgrades `reflop` in `uv.lock`, checks the locked hashes against the signed
  release, and syncs, so `uv run` no longer reinstalls the old version. A uv
  tool installed from PyPI stays a PyPI install. Elsewhere it installs the
  release's verified wheel, as before. It never downgrades a newer local
  version, and checks the installed version afterwards.

## 0.1.6 (2026-11-11)

Security fixes from the 2026-11-11 sweep. Every fix has a reproducer test.

Releases and updates:

- Releases are signed. `checksums.txt` carries `checksums.txt.sig`, an
  OpenSSH signature made with an offline key that never touches GitHub (CI
  publishes a draft, and `scripts/sign-release.sh` signs and publishes it).
  `install.sh` (via `ssh-keygen -Y verify`, OpenSSH 8.1+), `reflopd update` and
  `reflop update` refuse a release without a valid signature. As a result they
  cannot install 0.1.5 or older.
- `REFLOP_UPDATE_URL` must be `https://` (plain `http://` only for localhost).
  `reflop update` caps downloads at 256 MiB, like reflopd.

reflopd:

- Before authentication, frames are capped at 64 KiB (was the full
  `max_frame_bytes`), and buffers grow only as bytes arrive. Connections count
  against `max_connections` from accept, not from authentication. One source
  address may hold at most 16 unauthenticated connections. reflopd caps its own
  cgroup at total RAM minus the RAM floor.
- Seccomp fails closed. If the filter cannot be built or is missing, or the
  worker reports it inactive while seccomp is required, the worker is not
  started. Before, a filter that failed to build meant workers ran without
  one. A test pins the filter well below the kernel's 4096-instruction limit.
- No setting turns a protection off. The rails have minimums (1% and 1 GiB of
  disk, 512 MiB of RAM, 64 pids), and going below the defaults needs
  `reflopd rails set --force`. Every `[limits]` value has a range that
  excludes 0, and `audit.retention_days` must be 1..3650. **reflopd refuses
  to start** with a configuration below these. Fix it with `reflopd rails set`.
- Quotas reject 0, which used to mean "no limit": use `unlimited` to remove
  one (`reflopd quota set alice --storage unlimited`). `device_memory = 0` in
  `reflopd.toml` is an error.
- The storage quota counts uploads in progress, so parallel uploads can no
  longer exceed it together. Uploads with no progress for 7 days are deleted.
- If `/proc/meminfo` cannot be read, worker memory is frozen and new workers
  are refused (was: the RAM rail silently did nothing).
- Audit records for failed pairing and authentication are rate-limited per
  source and in total. The next record that is written counts the suppressed ones.
- The SSH forced command shell-quotes the reflopd path and `REFLOP_HOME`.
- Fixed: sizes such as `4GiB` and `512MiB` were rejected everywhere
  (`reflopd rails set --ram 8GiB`, `quota set --device-memory 6GiB`, TOML).
  Decimal units (`20GB`) worked.

SSH pipe mode:

- `reflopd invite --via ssh` puts the machine's sshd host keys in the invite
  (`--ssh-host-keys <glob>` for an sshd elsewhere). The client keeps them in
  its own known_hosts file under a `HostKeyAlias` tied to the server's reFLOP
  key and runs ssh with `StrictHostKeyChecking=yes`. Servers paired before
  0.1.6 use your known_hosts, also strictly. SSH invites from 0.1.6 need a
  0.1.6 client (older clients reject the new field).
- libreflop passes the same ssh options as the Python client (no X11 or port
  forwarding, no LocalCommand), and both put `--` before the target.

Clients:

- A server can no longer exhaust a Guest's memory. Every fetch chunk is
  checked against its tensor's declared size, offset and index, and a
  response is capped at 16 GiB (and at the server's `max_fetch_bytes`). The
  frame cap the server advertises is limited to 1 GiB plus 1 MiB.
- macOS: the Keychain entry is written through `security -i` on stdin, so the
  key seed no longer appears in the process list.

libreflop (C API):

- `reflop_last_error()` is per thread, as documented. The pointer stays valid
  until the next failing call on the same thread. Before, it was one global
  that the next call from any thread freed.
- `reflop_upload` checks that `nbytes` equals shape × dtype size, so `size_t`
  is never truncated to `int`.
- `reflop_fetch` returns `REFLOP_ERANGE` when the buffer is too small (was
  success with a partial copy). `nbytes` must not be NULL, and
  `buf = NULL, cap = 0` asks for the size only.
- `make -C clib test` runs API tests that need no server.

## 0.1.5 (2026-11-11)

Fixes from issue #1.

- libreflop: nodes nest. `reflop_op_begin` while a node is open starts an
  inner node, so the C++ README pattern
  `g.op("aten.addmm.default")...tensor(g.op("aten.t.default")...end())...`
  works (before, it failed with "still open"). `reflop_register` rejects a
  graph with a node still open. `make -C clib check` now builds a node inline.
- A graph that needs more memory than the device has is now `OUT_OF_MEMORY`
  (was `RESOURCE_LIMIT`), and `reflop.OutOfMemoryError` is also a
  `torch.OutOfMemoryError`, so code that catches it to shrink the batch works.
- When the graph being run fails its checks, that error is raised directly. An
  error from a request sent without waiting keeps its class and names the
  request ("checking a graph failed: ...") instead of "an earlier asynchronous
  operation failed".
- `reflop.capture`: a step that modifies a CPU argument in place before
  uploading it (or uploads a slice of one) stays eager. Before, if two calls
  happened to get equal data, later replays reused that data even after the
  argument changed. A step that still runs eagerly after warm-up now gets a
  one-time warning with the reason.
- Docs: GUEST_GUIDE covers late async errors, the OOM case, what capture does
  not replay, and why an upload-bound step gains nothing from capture.
- README: the "How it works" picture.

## 0.1.4 (2026-11-11)

- Guests can see their own compute: `reflop usage [--json]`, `reflop.usage()`
  and a usage line in `reflop status` (device time per device, estimated
  FLOPs, executes, sessions, traffic, including the current session). They see
  only their own identity's totals. New protocol messages `GetUsage` and
  `UsageReport` behind the `usage` capability; against an older server the
  client says it needs reflopd 0.1.4.
- `reflopd stop` and `reflopd up` with `REFLOP_HOME` set no longer act on the
  `reflopd` user service, which serves the default home.
- `reflopd usage` shows sub-second device time in milliseconds.

## 0.1.3 (2026-11-11)

- `reflopd usage [NAME] [--since DUR] [--json]`: compute used per Guest.
  The main measure is device time (how long the device ran the Guest's
  executes, measured by the worker), with estimated FLOPs, executes, sessions,
  session time and traffic. Lifetime totals persist in `usage.json` in the
  state directory, starting from what the audit log holds; `--since` reads the
  audit log.

## 0.1.2 (2026-11-10)

- Client: operators that PyTorch builds out of others in C++ (such as the
  causal mask of `scaled_dot_product_attention(is_causal=True)`) create their
  temporary tensors in their inputs' session, so transformers train on the
  server-CPU session while a GPU session is open.
- Client: `reflop.capture` binds tensors inside list, tuple and dict arguments,
  so a step that takes a dataset batch (`step(batch)`) replays instead of
  running eagerly every call.

## 0.1.1 (2026-11-10)

- Client: `t.device` reports the tensor's session, so `device=t.device`,
  `t.to(other)`, `module.to(...)` and differentiable moves work across the GPU
  and server-CPU sessions; `tensor.data = other` works; `torch.tensor(...,
  device=dev)` works.
- Client: `reflop.capture` accepts steps that move CPU arguments with
  `x.to(dev)`; sessions close when the Python process exits; dataset batches
  send their indices once.
- `reflopd`: a session that cannot start, or an allocation that fails, for lack
  of GPU memory evicts the oldest disconnected session on that GPU; start-up
  OOM is reported as `OUT_OF_MEMORY`.
- macOS: separate Keychain keys per `REFLOP_CLIENT_HOME`; `libreflop` links
  with an `@rpath` install name.

## 0.1.0 (2026-11-10)

First implementation of PLAN.md milestones M0–M7.

- `reflopd`: rootless setup (`reflopd up`) with a pinned, checksum-verified
  PyTorch runtime (CUDA 13.2, CUDA 12.6 for older drivers and GPUs, or CPU);
  TLS 1.3 daemon mode and SSH pipe mode; pairing with one-time invites; sandboxed
  workers (Landlock, seccomp, namespaces when available, cgroups); safety rails;
  optional quotas; pause, resume, kill, auto-yield and messages; status, top and
  an audit log; `doctor` and `selftest`; Level 1 system install; self-update.
- Worker: validation stages V1–V5 and E1–E3 over opset v1 (582 operators),
  runtime bounds checks, zero-filled allocations, datasets, snapshots.
- Python client: the `reflop` device for PyTorch, lazy capture with views and
  in-place operations, `reflop.capture`, datasets, placement, shim mode for CUDA
  scripts, autocast and GradScaler, the `reflop` CLI.
- `libreflop`: C ABI and a C++ wrapper.
- `install.sh` for both sides, GoReleaser configuration, CI.
