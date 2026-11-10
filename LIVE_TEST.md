# LIVE_TEST.md: build a live test project for reFLOP

**Audience: an LLM coding agent.** Create a separate project, `reflop-livetest`,
that uses reFLOP the way real Guests do: realistic training scripts, datasets,
long runs, several users at once, the C API, and the Owner's controls. Run it
against a live server and write a report. Do not modify reFLOP itself. When
something fails, record it with a minimal reproducer instead of working around it.

Read the whole document before writing code.

---

## 1. What reFLOP is

reFLOP lets a Guest train on an Owner's GPU from another machine. The Guest's
Python program runs on the Guest's machine. Tensor operations on the `reflop`
device are recorded locally and sent to the server as operator graphs; the
daemon (`reflopd`) validates them and runs them in a sandboxed worker. Only the
values the program asks for come back.

| Piece | Where | Commands |
|---|---|---|
| `reflopd` | Owner machine (Linux; NVIDIA GPU or CPU only) | `up`, `invite`, `users`, `revoke`, `status`, `top`, `audit`, `pause`, `resume`, `kill`, `say`, `yield`, `quota`, `rails`, `doctor`, `selftest` |
| `reflop` | Guest machine (macOS or Linux, no CUDA needed) | `pair`, `servers`, `status`, `init`, `data build/push/ls/rm`, `run`, `verify`, `doctor`, `key` |
| `libreflop` | Guest machine | C API `clib/include/reflop.h`, C++ wrapper `clib/include/reflop.hpp` |

Python API:

```python
import torch, reflop

dev = reflop.device()                        # GPU session on the default paired server
cpu = reflop.device(1, kind="cpu")           # second session, on the server's CPU
x = torch.randn(64, 32).to(dev)              # upload
y = (x @ x.T).relu().sum()                   # recorded, not run yet
print(y.item())                              # runs on the server, fetches one number

reflop.seed(0)                               # server RNG (torch.manual_seed also seeds it)
reflop.step_boundary()                       # run pending work, fetch nothing
reflop.synchronize()                         # wait; raises deferred server errors
reflop.stats()                               # {index: {flushes, registers, executes, bytes_up, bytes_down}}
reflop.fetch(model.state_dict(), "ckpt.safetensors")
step = reflop.capture(train_step)            # static replay after two eager calls
ds = reflop.dataset("name")                  # server-side dataset: ds.loader(...), ds.batch(idx)
with reflop.placement("auto"): ...           # also "remote", "local", "remote-cpu"
reflop.close()

# exceptions
reflop.UnsupportedOperatorError, reflop.OutOfMemoryError, reflop.SessionLostError,
reflop.ReflopRemoteError, reflop.CaptureError
```

Reference documentation in the reFLOP repository: `README.md`,
`docs/GUEST_GUIDE.md`, `docs/OWNER_GUIDE.md`, `docs/OPSET.md`, `clib/README.md`.
Read the Guest guide's "What works and what does not" section before choosing
models, so expected failures are not reported as bugs.

## 2. Environment

The human provides:

- an Owner machine with `reflopd` running (`reflopd status` works there), its
  listen address, and whether it has a GPU;
- one or more invite codes (`reflopd invite --name <guest>`), one per simulated
  Guest;
- a Guest machine (it may be the Owner machine itself) with Python 3.10+.

Ask the human for anything missing. Never guess an address or reuse an invite.

Install the client into the project's own environment:

```sh
uv venv --python 3.12 .venv && source .venv/bin/activate
curl -fsSL https://raw.githubusercontent.com/chaseungjoon/reflop/main/install.sh | sh
#   or, before a release exists:  pip install /path/to/reflop/client/python
uv pip install torchvision pytest
```

To simulate several Guests on one machine, give each its own client home and
pair each separately:

```sh
REFLOP_CLIENT_HOME=$PWD/guests/alice reflop pair <alice-code> --name lab
REFLOP_CLIENT_HOME=$PWD/guests/bob   reflop pair <bob-code>   --name lab
```

Every script in the project must accept `--guest <name>` and set
`REFLOP_CLIENT_HOME` accordingly before importing `reflop`.

## 3. Project layout to create

```
reflop-livetest/
├── README.md                  how to run everything, which Guest homes exist
├── pyproject.toml             depends on torch, torchvision, reflop, pytest
├── livetest/
│   ├── common.py              guest selection, timing, a reference-vs-remote comparer, result logging
│   ├── models.py              the model zoo (§4)
│   ├── data/
│   │   ├── synth_images.py    deterministic image-classification data (uint8, N×3×32×32, learnable labels)
│   │   ├── synth_text.py      deterministic character-level corpus and tokenizer
│   │   └── packs.py           `module:function` loaders for `reflop data build`
│   ├── train/
│   │   ├── train_classifier.py   ResNet/CNN on a server dataset, eager and with reflop.capture
│   │   ├── train_charlm.py       small GPT with scaled_dot_product_attention
│   │   ├── train_unet.py         segmentation on synthetic shapes (upsampling, concat, batch norm)
│   │   ├── train_seq.py          LSTM and GRU sequence classifier (batch_first, 2 layers)
│   │   └── train_amp.py          autocast bf16/fp16 with GradScaler
│   ├── cuda_script/
│   │   └── plain_cuda_train.py   an unmodified CUDA script, run via `reflop run`
│   ├── capi/
│   │   ├── Makefile
│   │   ├── matmul.c              uses libreflop directly
│   │   └── mlp.cpp               uses reflop.hpp
│   └── scenarios/            one runnable file per scenario in §5
├── tests/                    pytest wrappers that run each scenario and assert its criteria
└── report/
    └── REPORT.md             produced at the end (§7)
```

## 4. Model zoo and reference comparison

For every model, `common.py` must be able to train the same architecture twice
from the same seed and the same data: once on the CPU (or locally on CUDA if
the Guest machine has a GPU) as the reference, and once on the reflop device.
Compare per-step losses and, after the run, parameters.

| Model | Size | Notes |
|---|---|---|
| MLP | 784→256→10 | SGD and AdamW |
| CNN | 3 conv layers, batch norm, max pool | CIFAR-shaped input |
| ResNet-18 | torchvision, `num_classes=10`, 32×32 input | the throughput benchmark |
| Char GPT | 4 layers, 4 heads, d=128, context 128, `is_causal=True` | `nn.TransformerEncoderLayer(norm_first=True, batch_first=True)` or a hand-written block using `F.scaled_dot_product_attention` |
| U-Net | 2 down, 2 up, bilinear upsampling | per-pixel cross-entropy |
| LSTM / GRU | 2-layer LSTM then a GRU, `batch_first=True` | sequence classification |

Tolerances: on the server CPU, losses should match the CPU reference to about
1e-5. On a GPU, the first step's loss should match within about 1e-3 relative
(TF32 is on for convolutions by default); later steps may drift because GPU
reductions are not bit-identical. Report the drift; do not call it a bug unless
training diverges or the first step is off by more than 1e-2 relative.

## 5. Scenarios

Each scenario is a script in `livetest/scenarios/` that prints a JSON line with
its measurements and a pass/fail verdict, and a pytest wrapper in `tests/`.

### S1. Pairing and status

- `reflop status` for each Guest: record devices, the number of enabled
  operators, limits, quotas.
- `reflop verify`: all checks must pass.
- `reflop doctor`: must report a pinned key and a working connection.

### S2. Correctness across the model zoo

Train every model in §4 for 50 steps on `reflop.device()` and on the reference.
Record per-step losses for both, the maximum loss difference, and the final
parameter difference. Repeat on `reflop.device(1, kind="cpu")`.

### S3. Datasets

1. Build a 50,000-sample synthetic image pack with `reflop data build --from
   livetest.data.packs:images --out packs/images` and push it with
   `reflop data push packs/images --name livetest-images`.
2. Interrupt a second push of a larger pack (200,000 samples) partway, for
   example by killing the process after a few seconds, then run the same push
   again. Record how many chunks the second run sent (it should resume, not
   restart).
3. Train the CNN from `reflop.dataset("livetest-images").loader(...)` for one
   epoch. Record `bytes_up` per step from `reflop.stats()`; it should be about
   8 bytes per sample in the batch, not image-sized.
4. `reflop data ls`, then `reflop data rm` both packs.

### S4. Eager versus capture throughput

Run ResNet-18 at batch 128 for 100 steps:

- plain PyTorch on the Owner's GPU directly (only possible if you run on the
  Owner machine; otherwise skip this baseline and say so);
- reflop eager;
- reflop with `reflop.capture` and SGD with momentum;
- reflop with `reflop.capture` and `torch.optim.AdamW(..., capturable=True)`.

Record milliseconds per step, images per second, registrations, executes, and
bytes up and down per step. Verify that `reflop.capture` with a plain
(non-capturable) Adam raises `reflop.CaptureError` instead of running.

### S5. Unmodified CUDA script

Write `plain_cuda_train.py` as a normal CUDA training script (`.cuda()`,
`device="cuda"`, `torch.cuda.is_available()`, `torch.manual_seed`). Run it with
`reflop run python livetest/cuda_script/plain_cuda_train.py`. It must train,
and its weights must report `device='reflop:0'`.

### S6. Mixed precision

`train_amp.py`: CNN and GPT under `torch.autocast("reflop", dtype=torch.bfloat16)`,
then under float16 with `torch.amp.GradScaler("reflop")`. Losses must decrease
and the scaler's scale must stay positive.

### S7. Placement and fallbacks

- Call an operator the server does not support (for example `torch.unique` on
  a reflop tensor). Under `reflop.placement("auto")` it must run on the client
  with a warning; under `"remote"` it must raise
  `reflop.UnsupportedOperatorError` naming the operator.
- Under `"local"`, operations on reflop tensors must return CPU tensors.
- Moving a tensor between the GPU session and the server-CPU session with
  `.to(...)` must work. An operation mixing tensors from the two sessions must
  raise a clear error.

### S8. Checkpoints and resumption

Train the char GPT for 200 steps, `reflop.fetch(model.state_dict(), path)`,
load the file into a fresh CPU model, and check that it produces the same
outputs as the remote model on a fixed batch. Then load the checkpoint, move
it back to the reflop device, and continue training for 50 steps.

### S9. Several Guests at once

Run two or three Guests concurrently (separate processes, separate client
homes), each training a different model for a few minutes. Record per-Guest
step times alone and together. Then make the Guests deliberately fill GPU
memory (each allocating ever larger tensors until one fails): the failing
allocation must raise `reflop.OutOfMemoryError`, and that Guest's session must
keep working for small operations afterwards. Nobody's process may crash.

### S10. Owner controls

Coordinate with the human, or run these yourself if you are on the Owner
machine, while a Guest trains:

| Owner runs | Expected on the Guest |
|---|---|
| `reflopd say "hello from the Owner"` | the message is printed by the client |
| `reflopd pause`, wait 10 s, `reflopd resume` | the Guest's steps stall for about 10 s, then continue; no error |
| `reflopd top` | the Guest's session appears with growing executes |
| `reflopd kill <guest>` | the Guest gets `reflop.SessionLostError` |
| `reflopd quota set <guest> --max-sessions 1`, then a second session | the second session is refused with a clear message; afterwards `reflopd quota clear <guest>` |
| `reflopd audit --name <guest>` | shows the Guest's sessions and executes, and no tensor contents |

### S11. Disconnects

During a training loop, interrupt the network (for example, stop and restart
Tailscale, or put the laptop to sleep briefly). Record what the Guest sees:
either the loop continues after an idle reconnect, or it raises
`reflop.SessionLostError` with an explanation. A hang without an error is a bug.

### S12. C and C++ API

Build `capi/` against `libreflop` (`make -C <reflop>/clib` produces
`clib/build/libreflop.so`). `matmul.c` multiplies two matrices on the server
and compares with a C reference; `mlp.cpp` runs a two-layer MLP forward with
`reflop.hpp`. Both must match their local references to about 1e-4.

### S13. Long run

Train the char GPT with `reflop.capture` for at least 30 minutes on a
200,000-sample text pack. Every 1,000 steps record step time, `reflop.stats()`,
and the client process's memory use. Step time and client memory should stay
flat. Report any growth.

## 6. Rules for the test project

- Use only reFLOP's public API and CLI.
- Seed everything; reruns on the server CPU must give identical results.
- Keep each scenario under 15 minutes except S13.
- Never put secrets, invite codes or keys in the repository; read them from
  the environment or from the human at run time.
- Do not use `torch.compile`, custom CUDA kernels or `torch.library` ops on
  reflop tensors; the Guest guide lists them as unsupported.

## 7. The report

`report/REPORT.md`, written for the reFLOP author:

1. Environment: Owner machine (GPU, driver, `reflopd status` output), Guest
   machine(s), network (same machine, LAN, Tailscale), reflop and torch versions.
2. A table with one row per scenario: pass/fail, the key numbers, and a one-line
   note.
3. Correctness: per model and device, the first-step and final loss
   differences and the parameter difference.
4. Performance: the S4 table, S3 bytes per step, S9 contention numbers, S13
   stability numbers.
5. Bugs: for each, the exact error text, a minimal standalone reproducer
   (≤ 30 lines of Python or a shell command sequence), and which document
   promised different behavior.
6. Documentation gaps: places where the guides were wrong, unclear or silent.
