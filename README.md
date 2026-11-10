<div align="center">

# reFLOP

[![CI](https://github.com/chaseungjoon/reflop/actions/workflows/ci.yml/badge.svg)](https://github.com/chaseungjoon/reflop/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/chaseungjoon/reflop?include_prereleases)](https://github.com/chaseungjoon/reflop/releases)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](docs/GUEST_GUIDE.md)
[![C](https://img.shields.io/badge/C-A8B9CC?logo=c&logoColor=black)](clib/README.md)
[![C++](https://img.shields.io/badge/C%2B%2B-00599C?logo=cplusplus&logoColor=white)](clib/README.md)
[![Go](https://img.shields.io/badge/Go-00ADD8?logo=go&logoColor=white)](server/reflopd)
[![Linux](https://img.shields.io/badge/server-Linux-FCC624?logo=linux&logoColor=black)](docs/OWNER_GUIDE.md)
[![macOS](https://img.shields.io/badge/client-macOS%20%7C%20Linux-000000?logo=apple&logoColor=white)](docs/GUEST_GUIDE.md)

**Lend your GPU as a calculator. They don't get shell access, they can only run floating point calculations.**

</div>

---

One person in the team has a powerful GPU workstation. Everyone else trains on it from
their laptop. 

**reFLOP** sends raw tensor math to the GPU directly, and ***only tensor math*** is sent: your
Python program keeps running on your machine.

```python
import torch, reflop

dev = reflop.device()          # the GPU on the Owner's workstation
model = MyModel().to(dev)      # weights live on the server
for x, y in loader:
    loss = loss_fn(model(x.to(dev)), y.to(dev))
    loss.backward()
    opt.step(); opt.zero_grad()
    print(loss.item())         # only this number comes back
```

Unmodified CUDA scripts work too:

```sh
# instead of
python train.py

# run
reflop run python train.py    # .cuda() and device="cuda" go to the reflop server
```

The server **never runs full code**. It accepts operator graphs over a closed set of
582 operators (571 PyTorch overloads and 11 reflop-native ones), validates them, and executes them in a sandboxed worker.

A Guest cannot delete the Owner's files, fill the disk, start processes, open
network connections, or read anyone else's data, because no request can express
any of that.

Only floating point calculations are sent to the server.

## Table of Contents

- [Install](#install)
- [Quick Start](#quick-start)
- [What a step does](#what-a-step-does)
- [What the Owner is protected from](#what-the-owner-is-protected-from)
- [Datasets](#datasets)
- [Fast steady-state training](#fast-steady-state-training)
- [More Information](#more-information)
- [Development](#development)
- [License](#license)

## Install

> The Owner (the machine with the GPU, Linux) installs `reflopd`. Guests (macOS or
> Linux laptops, no CUDA needed) install the `reflop` client.

### Requirements

- Server: Linux x86_64 or aarch64, glibc ≥ 2.28. An NVIDIA GPU with driver ≥ 525 is
  optional (without one, reflopd serves the CPU). No root, no Docker, no Python.
- Client: Python ≥ 3.10 with PyTorch ≥ 2.6. macOS (Apple Silicon or Intel) or Linux.

### Install Script (Recommended)

- Owner, on the GPU workstation:

```sh
curl -fsSL https://raw.githubusercontent.com/chaseungjoon/reflop/main/install.sh | sh -s -- --server
```

- Guest, on the laptop:

```sh
curl -fsSL https://raw.githubusercontent.com/chaseungjoon/reflop/main/install.sh | sh
```

`reflopd` goes to `~/.local/bin`. The client goes into the active virtual
environment if one is active, otherwise into its own environment managed by `uv`
(or a private venv). Both downloads are checked against the release's SHA-256 sums.

- Update

```sh
reflopd update     # Owner
reflop update      # Guest
```

### Releases tab

Download `reflopd_<version>_linux_<arch>.tar.gz` or
`reflop-<version>-py3-none-any.whl` from
[Releases](https://github.com/chaseungjoon/reflop/releases), verify it against
`checksums.txt`, and put `reflopd` on your `PATH` or `pip install` the wheel.

### Build from source

Requires Go (see `go.mod`) and, for the client, Python 3.10+:

```sh
git clone https://github.com/chaseungjoon/reflop.git && cd reflop
make build                                  # bin/reflopd, with the worker embedded
pip install ./client/python                 # the reflop client
```

## Quick Start

```sh
# 1) Owner: set up and start (rootless). Fetches the pinned PyTorch runtime,
#    runs a sandboxed self-test, starts a systemd user service, prints an invite.
reflopd up --listen 100.64.0.7:7460        # e.g. your Tailscale address

# 2) Owner: one invite per Guest
reflopd invite --name alice

# 3) Guest: pair once (the server key gets pinned), then check
reflop pair reflop1-baaraai2b4…
reflop status

# 4) Guest: train
python train.py                             # with reflop.device(), or:
reflop run python train.py                  # unmodified CUDA code
```

## What a step does

1. **Records** the step instead of running it.

   Every PyTorch operation on a `reflop` tensor is recorded on your machine.
   Shapes and dtypes are computed locally, so shape-dependent code works without
   round trips.

2. **Sends a graph** when a value is needed.

   `loss.item()`, `print(t)` or `.cpu()` turns the recorded operations into one
   graph. The graph is sent once; later steps send only its hash and the new inputs.

3. **Validates** it on the server.

   Strict decoding, the operator whitelist, schema and argument checks, shape
   inference on meta tensors, and a peak-memory estimate against the GPU's size,
   all before any GPU memory is allocated. Index arguments are bounds-checked
   before each kernel launches.

4. **Runs** it in a sandboxed worker.

   One worker process per session, with no network, no `exec`, no `fork`, and no
   access to the Owner's files. Weights and optimizer state stay on the server
   between steps.

5. **Returns** only what you asked for.

   Usually a few bytes: the loss.

## What the Owner is protected from

| A Guest tries to | What happens |
|---|---|
| run a shell, a script, a pip package, a CUDA kernel | there is no request type for it; only whitelisted operators exist |
| read `~/.ssh`, the browser profile, other Guests' data | the worker's Landlock domain covers only its own scratch directory |
| open a socket, connect to localhost, `exec`, `fork`, `ptrace` | seccomp denies it after warm-up |
| index out of bounds to read GPU memory | bounds are checked before the kernel launches |
| read stale GPU memory with `torch.empty` | allocation-only operators return zeros |
| fill the disk or exhaust host RAM | safety rails: disk keeps max(10%, 20 GiB) free, host RAM keeps 4 GiB |
| hog the GPU | `reflopd pause`, `kill`, `yield --auto`; optional quotas |
| impersonate the server | the client pins the server key at pairing and refuses a different one |

`reflopd doctor` shows which protections are active on your machine, and
`reflopd selftest --security` checks them on a live worker. Details:
[threat model](docs/THREAT_MODEL.md).

## Datasets

Decode and preprocess on your laptop, upload once, then send only batch indices:

```sh
reflop data build --from my_data:load --out ./cifar_pack   # runs your code locally
reflop data push ./cifar_pack --name cifar10               # resumable, verified per 4 MiB chunk
```

```python
ds = reflop.dataset("cifar10")
for batch in ds.loader(batch_size=128, shuffle=True, seed=0):
    x = batch["images"].float() / 255     # gathered on the server
    y = batch["labels"]
```

## Fast steady-state training

Eager mode records every operation each step, which costs tens of microseconds
per operation on the client. For steps whose graph does not change,
`reflop.capture` replays the step on the server without running your Python:

```python
step = reflop.capture(train_step)     # train_step(x, y) -> loss
for x, y in data:
    loss = step(x, y)                 # one request per step after two warm-up calls
```

ResNet-18 on CIFAR-10 shapes (batch 128, RTX 5070, client on the same machine
over TLS) runs at 74% of the throughput of plain PyTorch on the same GPU with
`reflop.capture`, and 19% in eager mode. Steps whose Python numbers change every
step (a non-capturable Adam's bias correction, a Python learning-rate schedule)
are detected and refused; use `capturable=True` or tensor learning rates.

## More Information

| | |
|---|---|
| `reflopd status` · `top` · `audit` | who is connected, live memory and FLOPs, the audit log |
| `reflopd pause` · `resume` · `kill alice` · `say "…"` | Owner controls |
| `reflopd yield --auto` | pause Guests while you use the GPU yourself |
| `reflopd quota set alice --device-memory 6GiB` | optional quotas (none by default) |
| `reflopd rails show` · `rails set --ram 8GiB` | safety rails (always on) |
| `reflopd invite --name bob --via ssh --key "ssh-ed25519 …"` | reuse `sshd`: a forced-command pipe, never a shell |
| `sudo reflopd install --system` | move reflopd into a dedicated system account |
| `reflop data build` · `push` · `ls` · `rm` | datasets |
| `reflop verify` | cross-check small graphs against your CPU |
| `with reflop.placement("local"):` | run part of the program on the client |

- [Owner guide](docs/OWNER_GUIDE.md): setup, exposure, Level 1, SSH mode, rails, quotas, troubleshooting
- [Guest guide](docs/GUEST_GUIDE.md): API, capture, datasets, placement, what works and what does not
- [C and C++ API](clib/README.md): `libreflop` for non-Python programs
- [Threat model](docs/THREAT_MODEL.md) · [Design decisions](docs/DECISIONS.md) · [Spike report](docs/SPIKE_REPORT.md)
- [Operator set](docs/OPSET.md) · [Design plan](PLAN.md)

**Limitations:** the server sees your data and models in plaintext; reFLOP
protects the Owner's machine from Guests, not Guests' data from the Owner.
Arbitrary CPU programs, custom CUDA or Triton kernels, and server-side image or
text decoding are not supported. GPU results can differ bitwise from local runs
(TF32, non-deterministic kernels). Killing a worker loses its session state.
[All limitations](docs/THREAT_MODEL.md#known-limitations).

## Relation to recom

[recom](https://github.com/chaseungjoon/recom) runs your commands on your own
remote machine over SSH: full access, by design. reFLOP is for lending a GPU to
people you don't fully trust. It runs only whitelisted tensor operators and gives
no shell. Use recom for your own server and reFLOP for a shared one.

## Development

```sh
make build              # bin/reflopd
make test               # Go tests (race) and Python tests
make lint               # gofmt, go vet, ruff, and the I1/I2 static check
make build-probes       # test build with isolation probes (never ship it)
make test-isolation     # run the probes inside real sandboxed workers
make clib               # libreflop and the C/C++ examples
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md).

## License

[Apache-2.0](LICENSE)
