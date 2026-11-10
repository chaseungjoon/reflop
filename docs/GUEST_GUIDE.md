# reFLOP Guest guide

You train on someone else's GPU from your own machine. Your Python program runs
here; reFLOP sends the tensor math to the server and brings back only the
values you ask for. Your machine needs no CUDA: a MacBook works.

- [Install and pair](#install-and-pair)
- [The device](#the-device)
- [When work is sent](#when-work-is-sent)
- [Sessions](#sessions)
- [Errors](#errors)
- [Datasets](#datasets)
- [reflop.capture](#reflopcapture)
- [Unmodified CUDA scripts](#unmodified-cuda-scripts)
- [Placement](#placement)
- [The server CPU](#the-server-cpu)
- [Saving checkpoints](#saving-checkpoints)
- [Mixed precision](#mixed-precision)
- [What works and what does not](#what-works-and-what-does-not)
- [CLI reference](#cli-reference)
- [Privacy](#privacy)

## Install and pair

```sh
curl -fsSL https://raw.githubusercontent.com/chaseungjoon/reflop/main/install.sh | sh
reflop pair reflop1-…          # the code the Owner gave you
reflop status
reflop verify                   # optional: compares a few results with your CPU
```

The installer puts the client into your active virtual environment if one is
active. Otherwise it installs the `reflop` command into its own environment and
prints the line to add the library to your project (`uv add …` or `pip install …`).

Pairing creates an Ed25519 key for this machine (macOS: stored in the login
Keychain; Linux: `~/.config/reflop/id_ed25519`, mode 0600) and pins the
server's key. If the server ever presents a different key, the client stops
before sending anything. With `REFLOP_CLIENT_HOME` set, the client keeps its
config there and uses a separate key (a separate Keychain entry on macOS), so
one machine can act as several Guests. The C API reads a key file; on macOS run
`reflop key --write-file` once.

## The device

```python
import torch, reflop

dev = reflop.device()                 # default paired server, its GPU
model = Net().to(dev)
x = torch.randn(64, 784).to(dev)
y = torch.zeros(64, dtype=torch.long, device=dev)
```

`reflop.device(server="lab", kind="cuda")` picks a paired server by name and a
device kind. Tensors on this device print as `device='reflop:0'`. `torch.manual_seed(n)`
also seeds the server's generator, so dropout and `torch.randn(..., device=dev)`
are reproducible.

## When work is sent

Operations on reflop tensors are recorded, not executed. The recording is sent
as one graph when a value has to leave the server:

| You call | What happens |
|---|---|
| `.item()`, `float(t)`, `if t > 0:`, `.tolist()` | runs everything recorded so far, fetches the value |
| `print(t)`, `.cpu()`, `.numpy()` | same, fetches the tensor |
| `reflop.step_boundary()` | runs the pending work, fetches nothing |
| `torch.nonzero`, boolean-mask indexing | counts on the server first, then continues (two round trips) |
| every 20,000 recorded operations | runs the pending work |

Execution is asynchronous: your loop keeps recording step N+1 while the server
runs step N. A server-side error from an earlier step is raised at the next point
where you wait for a value, the same way CUDA errors surface.

Identical steps produce identical graphs, so after the first step or two only
the graph's hash and the step's inputs are sent. Learning rates, step counts and
other Python floats are sent as inputs, not baked into the graph, so they do not
force a new graph each step. A step that never reads a value (no `.item()`)
merges with the next one into a larger graph until something forces a flush;
call `reflop.step_boundary()` at the end of each step to keep one graph per step.
`reflop.stats()` shows flushes, graph registrations, executes and tensor bytes
sent and received (tensor payloads only, not protocol framing).

## Sessions

The first `reflop.device()` call opens a session: a worker process on the
server that holds your tensors. `reflop.close()` ends it and frees its GPU
memory; the client also closes its sessions when your Python process exits.
If the connection drops instead (network loss, a killed process), the server
keeps the session for a while (`[session] ttl`, 30 minutes by default) so the
client can reconnect. A kept session whose client is gone gives up its GPU
memory as soon as a new session needs it.

## Errors

| Exception | Meaning |
|---|---|
| `reflop.UnsupportedOperatorError` | an operation has no form the server accepts; the message names it |
| `reflop.OutOfMemoryError` | the GPU is full (other people use it too); your session still works |
| `reflop.SessionLostError` | the worker ended: the Owner paused with `--now` or killed it, a rail was hit, or it crashed. State on the server is gone |
| `reflop.ReflopRemoteError` | the server rejected a request; the message names the validation stage and the operator |

Messages from the server are stripped of terminal control sequences before you
see them.

## Datasets

Decode images, tokenize text, and resize on your machine. Upload the resulting
tensors once:

```python
# my_data.py
import torch, torchvision

def load():
    ds = torchvision.datasets.CIFAR10("data", train=True, download=True)
    images = torch.tensor(ds.data).permute(0, 3, 1, 2).contiguous()   # uint8, N×3×32×32
    labels = torch.tensor(ds.targets)
    return {"images": images, "labels": labels}
```

```sh
reflop data build --from my_data:load --out ./cifar_pack
reflop data push ./cifar_pack --name cifar10
reflop data ls
```

`build` accepts a function that returns a dict of full tensors (sample axis
first) or an iterable of samples (dicts, `(x, y)` pairs, or tensors), such as a
torchvision dataset. A push is resumable: run it again after an interruption
and only the missing 4 MiB chunks are sent. The server verifies every chunk and
the whole pack before the dataset becomes usable.

```python
ds = reflop.dataset("cifar10")
mean = torch.tensor([125.3, 123.0, 113.9], device=dev).view(1, 3, 1, 1)
for batch in ds.loader(batch_size=128, shuffle=True, seed=0, epochs=10):
    x = (batch["images"].float() - mean) / 64.0
    loss = F.cross_entropy(model(x), batch["labels"])
```

Each step sends only the batch indices (8 bytes per sample, once per batch
whatever the number of tensors in the dataset). `ds.batch(indices)`
gathers specific rows. Augmentation written as tensor operations (slicing for
crops, `flip`, arithmetic) runs on the server too.

## reflop.capture

In eager mode your Python runs every step and every operation is recorded:
roughly 20–30 µs per operation on the client. For a training step whose graph
is the same every time, `reflop.capture` skips that:

```python
model = resnet18(num_classes=10).to(dev)
opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

def train_step(x, y):
    opt.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    opt.step()
    return loss

step = reflop.capture(train_step)
for x, y in data:
    loss = step(x, y)           # calls 1–2: eager and checked; then one request per call
    if i % 100 == 0:
        print(loss.item())
```

The first two calls run eagerly. If both produced the same graph with the same
constants, later calls send a single request that binds the new arguments, and
your Python function is not run. Rules (checked, not assumed):

- Tensor arguments, also inside lists, tuples and dicts (such as a dataset
  batch), must keep their shapes and dtypes; a change re-captures.
- Anything that changes every step must be a tensor. With a plain Adam or AdamW,
  the bias correction is a Python number that changes each step: pass
  `capturable=True` (or `fused=True`). A Python learning-rate schedule has the
  same problem: use a tensor learning rate. When this rule is broken,
  `reflop.CaptureError` explains it instead of silently freezing the value.
- The returned tensors are the same objects every call, with new contents (like
  CUDA graphs). Clone them if you need to keep a value.
- CPU tensor arguments may be moved inside the step (`x.to(dev)`, as above) or
  before the call (`step(x.to(dev), y.to(dev))`). Inside the step, each replay
  uploads the new argument.

`torch.compile` does not work with reflop tensors (TorchDynamo cannot trace this
kind of tensor subclass). `reflop.capture` gives the same benefit.

## Unmodified CUDA scripts

```sh
reflop run python train.py --epochs 10
```

`reflop run` starts the command on this machine with a shim: `torch.cuda.is_available()`
returns True, and `.cuda()`, `.to("cuda")`, `device="cuda"` and
`torch.cuda.manual_seed` go to the reflop device. This is best effort: code
that calls into `torch.cuda` in other ways (streams, memory statistics, CUDA
graphs) needs the explicit API.

## Placement

```python
with reflop.placement("remote"):     # never fall back to the client: unsupported ops raise
    ...
with reflop.placement("local"):      # operations on reflop tensors run here and return CPU tensors
    ...
with reflop.placement("remote-cpu"): # new tensors go to the server's CPU (see below)
    ...
```

The default, `"auto"`, runs operations on reflop tensors remotely. An operation
the server cannot run falls back to your CPU, with a one-time warning naming the
operation. An operation that mixes large local tensors with small reflop tensors
may run locally when a cost model (your CPU speed, the server's speed, bandwidth,
round-trip time; measured by `reflop init`) says that is faster.

## The server CPU

The server's CPU is a device too, useful for tensor work that does not suit a
GPU or when the GPU is busy:

```python
gpu = reflop.device(0)
cpu = reflop.device(1, kind="cpu")    # a second session on the server's CPU
a = torch.randn(4096, 4096).to(cpu)
b = (a @ a).to(gpu)                    # moving between sessions goes through your machine
```

`t.device` is the session's device (`reflop:0`, `reflop:1`), so
`torch.zeros(..., device=t.device)`, `torch.arange(n, device=idx.device)`,
`t.to(other.device)`, `t.to(other)` and `model.to(cpu)` follow the session.
(PyTorch's autograd internally sees a single device.) Moves between sessions are differentiable. An
operation mixing tensors from two sessions raises an error asking you to move
one.

## Saving checkpoints

```python
reflop.fetch(model.state_dict(), "ckpt.safetensors")    # download and save locally
state = reflop.fetch(model.state_dict())                 # or just download
```

Checkpoints are always safetensors. reflop tensors refuse to be pickled, so
`torch.save` on them raises; fetch them first. Loading works as usual: load on
the CPU, then `.to(dev)`.

## Mixed precision

`torch.autocast("reflop", dtype=torch.bfloat16)` follows the same policy as CUDA
autocast: matmuls and convolutions in the lower precision, reductions, norms and
losses in float32. `GradScaler` works through its regular API.

## What works and what does not

Works, tested against local training: MLPs, CNNs (ResNet-18), transformers
with `scaled_dot_product_attention`, U-Nets (upsampling, concatenation, batch
norm), LSTMs and GRUs (including `batch_first`, multiple layers), in-place
operations and views, `out=` arguments, integer and boolean tensors, float16 and
bfloat16, AdamW, Adam, SGD with momentum, autograd.

Does not work:

- `torch.compile` on reflop tensors (use `reflop.capture`)
- custom CUDA, Triton or C++ extension kernels, and `torch.library` custom ops
- operations whose output shape depends on data, beyond `nonzero` and boolean
  masks (`torch.unique`, for example, runs on your CPU in `"auto"` placement)
- sparse, nested or quantized tensors
- multi-GPU training, `torch.distributed`
- explicit `torch.Generator` objects on reflop tensors (use `reflop.seed(n)`)

GPU results can differ from your CPU's in the last bits (TF32 is on for
convolutions by default, and some GPU kernels are not deterministic).

## CLI reference

| Command | Purpose |
|---|---|
| `reflop pair <code> [--name lab]` | register with a server |
| `reflop servers [default\|remove <name>]` | list and manage paired servers |
| `reflop status` | devices, enabled operators, limits, your quotas, datasets |
| `reflop init` | measure round-trip time and local speed, write `.reflop.toml` |
| `reflop data build\|push\|ls\|rm` | datasets |
| `reflop run <command>` | run a command here with `cuda` mapped to reflop |
| `reflop verify` | compare small results with your CPU |
| `reflop doctor` | check torch, the key, the config and the connection |
| `reflop key [--ssh] [--write-file]` | print your public key; `--write-file` also stores the private key for the C API on macOS |
| `reflop update` | update the client |

`--server <name>` (or `REFLOP_SERVER`) selects a paired server for any command.

## Privacy

The client sends operator graphs, tensors, datasets you push, and protocol
metadata. It never sends environment variables, file paths, host names, or
source code. The server sees your data and models in plaintext: reFLOP protects
the Owner's machine from Guests, not Guests' data from the Owner. Do not train
on data you are not allowed to give to the Owner.
