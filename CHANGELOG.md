# Changelog

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
