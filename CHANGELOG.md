# Changelog

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
  its own known_hosts file under a `HostKeyAlias` tied to the server's reflop
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
