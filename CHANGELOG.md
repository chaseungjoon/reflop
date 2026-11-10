# Changelog

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
