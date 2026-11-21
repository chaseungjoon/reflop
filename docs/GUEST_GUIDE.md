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
uv add reflop                   # or: pip install reflop, in your project's environment
reflop pair reflop1-…          # the code the Owner gave you
reflop status
reflop verify                   # optional: compares a few results with your CPU
```

`reflop` on PyPI is the library you import and the `reflop` command. Install it
where your training code runs, so `import reflop` works there. Every version on
PyPI is uploaded from the signed GitHub release, after its signature is checked.

The install script does the same from the GitHub release and checks the
signature on your machine:

```sh
curl -fsSL https://raw.githubusercontent.com/reflopai/reflop/main/install.sh | sh
```

It installs into the active virtual environment if one is active. Otherwise it
installs the `reflop` command into its own environment and prints the line to
add the library to your project.

An invite works once, until it expires (24 hours unless the Owner chose
otherwise). If `reflop pair` says it expired, the message shows the expiry time
and this machine's clock, both in UTC. If the Owner made the invite recently,
this machine's date or time is probably wrong: compare `date -u` here and on
the Owner's machine.

Pairing creates an Ed25519 key for this machine (macOS: stored in the login
Keychain; Linux: `~/.config/reflop/id_ed25519`, mode 0600) and pins the
server's key. If the server ever presents a different key, the client stops
before sending anything. With `REFLOP_CLIENT_HOME` set, the client keeps its
config there and uses a separate key (a separate Keychain entry on macOS), so
one machine can act as several Guests. The C API reads a key file; on macOS run
`reflop key --write-file` once. SSH pipe mode also needs the key as a file, so
on macOS the first SSH connection writes `~/.config/reflop/id_ed25519` (mode
0600) next to the Keychain entry.

With an SSH invite, the client also pins the server's SSH host keys. It keeps
them in its own `known_hosts` file and runs ssh with
`StrictHostKeyChecking=yes`, so a host that presents any other key is refused,
whatever your `~/.ssh/config` says. Servers paired with an older client have no
pinned host keys; ssh then checks your `~/.ssh/known_hosts`, and never accepts
an unknown key. Pair again with a new invite to pin them.

### Unpairing

```sh
reflop unpair lab          # or just `reflop unpair` for the default server
```

This tells the server to stop accepting this machine's key, then forgets the
server here. The Owner's half is `reflopd unpair <your-name>`, which also
deletes your datasets and snapshots on the server. Either side may go first;
once both have, `reflop pair` with a new invite works as it did the first time.

If the server cannot be reached, `reflop unpair` still forgets it and tells
you to ask the Owner to run `reflopd unpair <your-name>`. `--local` skips the
connection. If the Owner unpaired you first, commands fail with "no longer
accepts this machine's key: run `reflop unpair lab`". `reflop pair` with a
server you are still paired with tells you to unpair first.

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

Operations on reFLOP tensors are recorded, not executed. The recording is sent
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
where you wait for a value, the same way CUDA errors surface. A value you read
is never stale: if any earlier request failed, the read raises that error
instead of returning data (see [Errors](#errors)).

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

## Your usage

`reflop usage` shows the compute you have used on a server: device time (how
long the server's devices ran your executes, per device), estimated FLOPs,
executes, sessions and session time, peak memory of your largest execute and
traffic. `reflop status` prints a one-line summary.

```
$ reflop usage
compute used by 'alice' on lab since 2026-11-10 21:54
device time:  31m01s (cuda:0 30m58s, cpu 3.1s)
session time: 12h24m (devices busy 4% of it)
executes:     265447, 7.0 ms average
est. FLOPs:   5050.9 TFLOP (2.7 TFLOP/s while running)
...
```

`reflop usage --json` and `reflop.usage()` return the same numbers as data.
You see only your own totals (counted per identity name, so all keys paired
under one name add up). This needs reflopd 0.1.4 or newer.

## Errors

| Exception | Meaning |
|---|---|
| `reflop.UnsupportedOperatorError` | an operation has no form the server accepts; the message names it |
| `reflop.OutOfMemoryError` | the GPU is full (other people use it too), or one graph needs more memory than the GPU has. It is also a `torch.OutOfMemoryError`, so code that catches that to shrink the batch works as is. Your session still works |
| `reflop.SessionLostError` | the worker ended: the Owner killed it, a rail was hit, or it crashed. State on the server is gone, unless you use checkpoints: then `reflop.recover()` continues from the last one (see [Saving state](#saving-state)) |
| `reflop.ReflopRemoteError` | the server rejected a request; the message names the validation stage and the operator |

Requests are sent without waiting for the answer, so an error can surface a few
lines after the operation that caused it, at the next value you read
(`.item()`, `.cpu()`, `.tolist()`, `.numpy()`, `if t:`, `reflop.fetch`) or at
`reflop.synchronize()`, `reflop.save` or `reflop.load`. The message then says
which request failed ("running a graph failed: ...").

- The read that raises returns nothing, even when the value it asked for does
  not depend on the failed request. The server answers in order, so by the
  time a value arrives every earlier failure is known; a value is never
  handed back while one is pending.
- The failed request changed nothing on the server: tensors it would have
  written keep their values from before it. After you catch the error (for
  example an out-of-memory error you recover from by shrinking the batch),
  the session works as usual.
- Each error is raised once. `print(t)` and `repr(t)` never raise (debuggers
  and tracebacks call them); they show `<unavailable: ...>` with the error,
  and the next read still raises it.
- `reflop.save` raises an earlier failure after writing the snapshot, which
  holds the values from before the failed request.
- `reflop.capture` replays follow the same rules: a replay that fails makes
  the next read of its outputs raise, never return the previous call's values.

In reFLOP 0.2.4 and earlier, a read right after a failed request could
return the tensor's old contents without an error, and the error surfaced one
read later.

If the program ends before an error like that surfaces, reFLOP prints it to
stderr at exit ("an error was not reported before the program ended: ...").
It first waits up to 5 seconds for requests still running on the server.

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
  batch), must keep their shapes and dtypes; a change re-captures. The next two
  calls with the new shapes (a smaller last batch, say) run eagerly, and later
  ones replay again.
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
- CPU work inside the step is not replayed. A step that changes a CPU argument
  in place before uploading it (`x.sub_(mean)` then `x.to(dev)`), or uploads a
  slice of one, stays eager: results are correct, but there is no speed-up.
  Normalize on the device after `.to(dev)`, or before calling the step.

When a step still runs eagerly after warm-up, `reflop.capture` warns once with
the reason. `step.replays` counts the calls that were replayed.

### When capture does not help

Capture removes the client's per-operation cost. It does not make uploads
faster. A step that uploads its batch every call is bound by the upload: in a
live test, ResNet-18 at batch 128 took 143 ms per step both eagerly and with
capture, because the batch dominated. With the data already on the server, the
same step took 19 ms eagerly and 5.7 ms with capture. Keep the data on the
server: use a [dataset](#datasets), or upload the tensors once before the loop
and index them inside the step.

`torch.compile` does not work with reFLOP tensors (TorchDynamo cannot trace this
kind of tensor subclass). `reflop.capture` gives the same benefit.

## Measuring throughput

Use [`benchmark/`](../benchmark/README.md) to compare reFLOP with native eager
PyTorch on the same Owner GPU. The Owner collects `native_cuda.sh`; the Guest
collects `inventory.sh`, `correctness.sh`, and `throughput.sh`. Use the same
benchmark source, timing configuration, and numerical policy on both sides.
`compare_cuda.sh` then reports absolute samples/second, milliseconds/step, and
the percentage of native throughput retained.

Compare resident and uploaded inputs separately. Resident cases reuse a batch
already on the device; uploaded cases include the Guest-to-Owner transfer each
step. Capture/eager speedup alone does not tell you how much native performance
you retain. Check correctness failures and missing cases before selecting a
workload, and record your connection conditions with the results. The
[saved report](../benchmark/report/README.md) compares a MacBook Pro Guest over
Tailscale TLS with native execution on the Owner's RTX 5070. At batch 128,
resident capture retains 35.6–70.1% of native throughput for eight workloads
and 4.0–4.6% for LSTM/GRU; uploading each step retains 0.26–10.52% with capture.
These are synthetic steady-state measurements. The available ResNet-18
correctness failures remain unresolved.

## Unmodified CUDA scripts

```sh
reflop run python train.py --epochs 10
```

`reflop run` starts the command on this machine with a shim: `torch.cuda.is_available()`
returns True, and `.cuda()`, `.to("cuda")`, `device="cuda"`,
`torch.cuda.manual_seed` and `torch.autocast("cuda")` go to the reFLOP device;
`torch.cuda.is_bf16_supported()` asks the server. This is best effort: code
that calls into `torch.cuda` in other ways (streams, memory statistics, CUDA
graphs) needs the explicit API.

The exit status is the command's own. If a signal kills the command, reFLOP
says so (`reflop: python was killed by SIGSEGV (signal 11)`) and exits with
128 + the signal number, as a shell would. See [Troubleshooting](#troubleshooting)
when a run crashes or hangs.

On macOS, `reflop run` sets `OMP_NUM_THREADS=1` for the command unless you set
the variable yourself, and says so when it starts. PyTorch on macOS ships its
own OpenMP runtime (`libomp`). A second one in the same process, from
scikit-learn, xgboost, LightGBM, faiss, conda or Homebrew, crashes or hangs the
program once both start their thread pools
([#7](https://github.com/reflopai/reflop/issues/7)). With one thread
neither does. Your math runs on the server, so only operators that fall back
to your CPU are slower. To keep more threads, set the variable yourself
(`OMP_NUM_THREADS=8 reflop run ...`). A program you start without `reflop run`
(`python train.py` with `import reflop`) gets no default; on macOS run it as
`OMP_NUM_THREADS=1 python train.py`.

## Placement

```python
with reflop.placement("remote"):     # never fall back to the client: unsupported ops raise
    ...
with reflop.placement("local"):      # operations on reflop tensors run here and return CPU tensors
    ...
with reflop.placement("remote-cpu"): # new tensors go to the server's CPU (see below)
    ...
```

The default, `"auto"`, runs operations on reFLOP tensors remotely. An operation
the server cannot run falls back to your CPU, with a one-time warning naming the
operation. An operation that mixes large local tensors with small reFLOP tensors
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

## Saving state

### On your machine

```python
reflop.fetch(model.state_dict(), "ckpt.safetensors")    # download and save locally
state = reflop.fetch(model.state_dict())                 # or just download
```

Files are always safetensors. reFLOP tensors refuse to be pickled, so
`torch.save` on them raises; fetch them first. Loading works as usual: load on
the CPU, then `.to(dev)`.

### On the server

Snapshots keep tensors on the Owner's machine, so nothing is downloaded:

```python
reflop.save("run1", model.state_dict())        # a dict of reflop tensors; nested dicts are fine
model.load_state_dict(reflop.load("run1"))     # later, also from a new process
reflop.snapshots()                             # [{"name", "bytes", "created"}]
reflop.delete_snapshot("run1")
```

Only tensors are stored. Keep step counts and other Python values on your
machine. A save replaces an older snapshot of the same name only once the new
one is fully written. Snapshots count against your storage quota, if the Owner
set one. This needs reflopd 0.2.0 or newer.

### Automatic checkpoints

If the Owner has turned checkpoints on (`reflopd checkpoints on`), the server
can save your whole session at a step boundary: every server tensor and the
server's random number generator state. This needs reflopd 0.2.0 or newer.

```python
dev = reflop.device(checkpoint=True)                    # the Owner's interval
dev = reflop.device(checkpoint=True, checkpoint_interval=1800)   # or longer, in seconds
```

The interval defaults to the Owner's, which is at least 5 minutes. You can
make it longer, never shorter: a shorter one raises `ReflopError` and names
the Owner's interval. Checkpoints are taken at `reflop.step_boundary()` and
after each call of a `reflop.capture` step. They run in the background on the
server and did not slow training measurably in our tests.

> [!WARNING]
> **During a checkpoint, the GPU memory for your session's state doubles.**
> Turning checkpoints on prints this in yellow. A tensor that your next steps
> change keeps its old version until the checkpoint has copied it, so for a
> moment parameters, optimizer state and buffers can need twice their
> memory. Leave that much GPU memory free when you pick a batch size. If it
> is not free, the checkpoint is skipped: reflop prints why and tries again
> later, and training goes on.

When the worker is lost, the next request raises `reflop.SessionLostError`.
`reflop.recover()` then opens a new session from the last checkpoint, in the
same process:

```python
dev = reflop.device(checkpoint=True)
step = 0
while step < total_steps:
    try:
        train_step(next_batch(step))
        step += 1
        reflop.checkpoint_tag({"step": step})   # saved with the next checkpoint
        reflop.step_boundary()
    except reflop.SessionLostError:
        step = reflop.recover()["step"]          # values and RNG as at that checkpoint
```

After `recover()`:

- every tensor that existed at the checkpoint has its value from then, and
  your model and optimizer objects keep working;
- tensors created after the checkpoint raise `SessionLostError` when used;
  create them again;
- the server's RNG is back where it was, so dropout repeats exactly;
- whatever your program keeps on your machine is not rewound: Python
  variables, the data loader's position, the CPU RNG. A plain Adam or AdamW
  also counts its steps on your machine; use `capturable=True` (or
  `fused=True`) so the step count is on the server and in the checkpoint;
- `reflop.recover()` returns the `checkpoint_tag` value of that checkpoint,
  and raises `ReflopError` if there is none.

A session the Owner ended on purpose (`reflopd kill`, revoke, unpair) cannot
be recovered: that deletes its checkpoint. Closing the session yourself does
too. When your process ends, use `reflop.save` for what you want to keep.

If the Owner runs `reflopd pause --now`, your session is usually suspended
rather than ended: your program waits until the Owner resumes, and nothing is
lost.

While execution is paused, the server holds your requests until it resumes.
It holds up to 4096 of them per session (and a limit in bytes); a request
past that fails with "the Owner has paused execution and ... send this
request again after execution resumes". A normal training loop never gets
near it, because it soon waits on a result.

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

- `torch.compile` on reFLOP tensors (use `reflop.capture`)
- custom CUDA, Triton or C++ extension kernels, and `torch.library` custom ops
- operations whose output shape depends on data, beyond `nonzero` and boolean
  masks (`torch.unique`, for example, runs on your CPU in `"auto"` placement)
- sparse, nested or quantized tensors
- multi-GPU training, `torch.distributed`
- explicit `torch.Generator` objects on reFLOP tensors (use `reflop.seed(n)`)

GPU results can differ from your CPU's in the last bits (TF32 is on for
convolutions by default, and some GPU kernels are not deterministic).

## Troubleshooting

**The program crashed.** `reflop run` names the signal that killed it. Python
programs it starts get `PYTHONFAULTHANDLER=1` (unless you set the variable
yourself), so a crash in native code (torch, OpenMP, another C extension) also
prints the Python stack of every thread.

**macOS: the program crashes or hangs, or prints `OMP: Error #15`.** This is
usually two OpenMP runtimes in one process
([#7](https://github.com/reflopai/reflop/issues/7)). `reflop doctor` lists
the OpenMP runtimes your packages install and the ones torch loads, and names
the package each one came from. `reflop run` already sets `OMP_NUM_THREADS=1`;
if you set it to more, or run without `reflop run`, try
`OMP_NUM_THREADS=1`. If it still fails, add the output of
`DYLD_PRINT_LIBRARIES=1 python train.py 2>&1 | grep -i omp` to the issue.

**The program hangs.** Run it with `REFLOP_DEBUG=1`. reFLOP prints the
program's pid. `kill -USR1 <pid>` from another terminal, or `Ctrl-\` in this one,
prints the Python stack of every thread and lets the program go on. Without
`REFLOP_DEBUG`, SIGUSR1 is left to your program. Ctrl-C reaches the program,
which prints its traceback before reFLOP exits.

**The connection fails.** Errors in SSH pipe mode end with what ssh said, for
example `(ssh: me@gpu: Permission denied (publickey).)`. That text comes
from ssh and from the server, so reFLOP strips control sequences from it and
keeps only the last few lines. `reflop doctor` checks the setup.

**`REFLOP_DEBUG=1`** works for every `reflop` command and for programs started
with `reflop run`:

- `reflop` commands print the full traceback instead of one line;
- each connection step goes to stderr: the ssh command or TLS address,
  handshake, authentication, session open, reconnects, and the reason a
  connection ended. These lines carry no keys, secrets or tensor data;
- `reflop run` prints the pid and enables the stack dump above.

`REFLOP_DEBUG=0` or an empty value turns it off.

## CLI reference

| Command | Purpose |
|---|---|
| `reflop pair <code> [--name lab]` | register with a server |
| `reflop unpair [<name>] [--local]` | forget a server and give up this machine's access to it |
| `reflop servers [default\|remove <name>]` | list and manage paired servers (`remove` is `unpair`) |
| `reflop status` | devices, enabled operators, limits, your quotas, datasets |
| `reflop usage [--json]` | compute you have used on the server |
| `reflop init` | measure round-trip time and local speed, write `.reflop.toml` |
| `reflop data build\|push\|ls\|rm` | datasets |
| `reflop run <command>` | run a command here with `cuda` mapped to reFLOP |
| `reflop verify` | compare small results with your CPU |
| `reflop doctor` | check torch, its OpenMP runtimes, the key, the config and the connection |
| `reflop key [--ssh] [--write-file]` | print your public key; `--write-file` also stores the private key for the C API on macOS |
| `reflop update` | update the client to the newest signed release. In a uv project it upgrades `uv.lock` and syncs; elsewhere it installs the release's wheel |

`--server <name>` (or `REFLOP_SERVER`) selects a paired server for any command.
`REFLOP_DEBUG=1` adds tracebacks and a log of each connection step
([Troubleshooting](#troubleshooting)).

## Privacy

The client sends operator graphs, tensors, datasets you push, and protocol
metadata. It never sends environment variables, file paths, host names, or
source code. The server sees your data and models in plaintext: reFLOP protects
the Owner's machine from Guests, not Guests' data from the Owner. Do not train
on data you are not allowed to give to the Owner.
