# M0 spike report

Results of the M0 technical spikes (PLAN.md §18) and the measurements that
revised the performance targets. Decisions are recorded in
[DECISIONS.md](DECISIONS.md) (ADR-002, ADR-003, ADR-004).

## Environment

| | |
|---|---|
| Server | AMD Ryzen 9 9950X (32 threads), 60 GiB RAM, NVIDIA GeForce RTX 5070 (12 GiB, compute capability 12.0), driver 595.71.05 (open kernel module) |
| OS | Ubuntu 24.04.5 LTS, kernel 7.0.0, Landlock ABI 8, seccomp, AppArmor restricting unprivileged user namespaces (`kernel.apparmor_restrict_unprivileged_userns=1`), systemd user manager with `cpu memory pids` delegated |
| Runtime | CPython 3.12.16 (python-build-standalone 20261026), PyTorch 2.15.0+cu132, CUDA 13.2 |
| Client | the same machine (loopback TLS) for the numbers below; the Apple M2 client of §19 was not available |

## S1: lazy dispatch capture

`spikes` and `client/python/reflop/tensor.py`. A `torch.Tensor` wrapper subclass
on the PrivateUse1 device renamed to `reflop`, set up with
`torch.utils.backend_registration._setup_privateuseone_for_python_backend`.

| Check | Result |
|---|---|
| `model.to(dev)`, `nn.Parameter` wrapping a RemoteTensor | works: parameters stay RemoteTensors and pass `isinstance(p, nn.Parameter)` |
| forward, `backward()`, AdamW (foreach), SGD, fused AdamW, capturable AdamW | works; backward ops arrive as ordinary aten ops |
| in-place ops on views (`a[1:3] *= 2`, `a.t()[2].add_(1)`) | works through view chains and inverse scatters |
| in-place view ops (`transpose_`, used by `nn.LSTM(batch_first=True)`) | needed metadata updates on the wrapper itself; done the way `return_and_correct_aliasing` does it |
| autocast | needs `AutocastPrivateUse1` kernels; reflop registers the CUDA policy |
| factories (`torch.arange(..., device=dev)`) | their composite forms resize an `out=` tensor; reflop registers direct kernels |
| autograd with device index > 0 | the Python device guard reports one device; all reflop tensors report `reflop:0` (ADR-015) |

An MLP trained with AdamW through the in-process worker on the CPU matches
local training bit for bit.

## S2: torch.compile backend

`spikes/s2_compile.py`. Both a custom backend and `aot_eager` fail before the
backend is called: TorchDynamo cannot create fake tensors for `RemoteTensor`
(it has no inner tensor to flatten). Replaced by `reflop.capture` (ADR-002).

## S3: meta validation and peak memory

`server/worker/reflop_worker/meta.py` and `cost.py`. Meta inference runs every
node on `meta` tensors; operators without a meta kernel are rejected (fail
closed). The peak-memory estimate is a liveness pass over the meta sizes with
views counted as aliases. Two findings:

- some meta kernels skip argument checks (`_softmax` accepts `dim=99`), so V3
  range-checks dim-like arguments for every operator;
- some kernels give unused auxiliary outputs a different shape than their meta
  kernel (`_embedding_bag`'s `offset2bag` on the CPU). Run-time checks compare
  every consumed output with its validated shape and dtype.

## S4: operator census

`spikes/s4_op_census.py` trains each model for three AdamW steps on the reflop
device and on the local CPU.

| Model | Distinct ops | Local fallbacks | Loss, reflop vs local (3 steps) |
|---|---:|---|---|
| MLP | 28 | none | 2.3232/2.3232, 2.2881/2.2881, 2.2534/2.2534 |
| ResNet-18 (32×32) | 38 | none | 2.4712/2.4699, 0.6903/0.5641, 0.0879/0.0917 |
| GPT (2 layers, SDPA, causal) | 53 | none | 4.3401/4.3401, 4.1694/4.1694, 4.0095/4.0095 |
| U-Net (bilinear upsampling, batch norm) | 37 | none | 0.7034/0.7034, 0.7000/0.7000, 0.6975/0.6975 |
| LSTM | 33 | none | 1.5253/1.5253, 1.5160/1.5160, 1.5067/1.5067 |

The union is 70 operators. None of the models emitted `as_strided` or an
excluded operator. ResNet-18's first-step difference comes from TF32 in cuDNN
convolutions: with `tf32_cudnn` off the first loss matches exactly (2.4699), and
later steps differ in the third decimal (batch of 4, random labels, GPU
reduction order). With `rnn_fused` disabled (the reflopd default), LSTM and GRU
use client-side decompositions of the fused cells and match local training to
five decimals.

Operators seen across all models: `_local_scalar_dense`, `_log_softmax`,
`_log_softmax_backward_data`, `_safe_softmax`, `_softmax`,
`_softmax_backward_data`, `_thnn_fused_lstm_cell`,
`_thnn_fused_lstm_cell_backward_impl`, `_unsafe_view`, `add.Tensor`, `add_`,
`addcdiv`, `addcdiv_`, `addcmul`, `addcmul_`, `addmm`, `alias`, `all.dim`, `bmm`,
`cat`, `clone`, `convolution`, `convolution_backward`, `copy_`, `detach`,
`div.Scalar`, `div.Tensor`, `embedding`, `embedding_dense_backward`, `eq.Scalar`,
`expand`, `lerp.Scalar`, `lerp_`, `max_pool2d_with_indices`,
`max_pool2d_with_indices_backward`, `mean.dim`, `mm`, `mul.Scalar`,
`mul.Tensor`, `mul_`, `native_batch_norm`, `native_batch_norm_backward`,
`native_layer_norm`, `native_layer_norm_backward`, `nll_loss_backward`,
`nll_loss_forward`, `ones_like`, `permute`, `relu`, `relu_`, `select.int`,
`select_backward`, `slice.Tensor`, `softmax.int`, `sqrt`, `squeeze.dim`,
`stack`, `sum.dim_IntList`, `t`, `threshold_backward`, `transpose.int`, `tril`,
`triu`, `unbind.int`, `unsqueeze`, `upsample_bilinear2d`,
`upsample_bilinear2d_backward`, `view`, `where.self`, `zeros_like`.

## S5: opset introspection

`codegen/bootstrap_opset.py`, census in `docs/opset_census.json`.

| | |
|---|---|
| Overloads tagged `core` in PyTorch 2.15.0 | 189 (the plan expected 193) |
| §8.4 names not found | 0 |
| §8.4 names not tagged core | `avg_pool1d`, `adaptive_avg_pool1d`, `sym_is_contiguous`, `resize_` (kept with their §8.4 disposition) |
| Tier 1/2 names not found | 12: the `.vec` overloads of the upsample backward ops, and `masked_fill_` |
| Mutating overloads dropped | 39 `_foreach_*.*_out` overloads |
| Result | 582 entries: Tier 0 193, Tier 1 360 (11 reflop-native), Tier 2 29 |

## S6: rootless sandbox

Ubuntu 24.04 defaults, the RTX 5070, kernel 7.0.

| Finding | Consequence |
|---|---|
| Unprivileged `unshare -Urn` fails (AppArmor), but a Go parent writing the uid map for a `CLONE_NEWUSER\|CLONE_NEWNET` child succeeds | workers also get an empty network namespace here; reflopd probes this at start |
| Landlock ABI 8 (includes `TSYNC`, scoping, `IOCTL_DEV`, TCP rules) | applied before `exec` anyway, which works on every ABI |
| CUDA init fails with error 304 unless the process can write `/proc/self/task/<tid>/comm` | workers may write their own `/proc/<pid>` |
| libcuda reads `/proc/devices`, `/proc/driver/nvidia`, `/proc/sys/vm/mmap_min_addr`, `/sys` topology; tries and fails to create `/dev/char/*` links and `/dev/nvidia-caps` (as it does without a sandbox) | read-only rules for the former |
| Device nodes needed: `/dev/nvidiactl`, `/dev/nvidia-uvm`, `/dev/nvidia<minor>` | read, write and `IOCTL_DEV` on exactly those |
| A worker's syscall set after warm-up fits an allowlist of about 190 syscalls (a few allowed only with argument checks: `clone` with `CLONE_THREAD`, signals to itself) | `server/reflopd/internal/sandbox/syscalls_*` |
| Replacing the reflopd binary while it runs breaks `exec` of the new file under its Landlock domain | workers start through `/proc/self/exe` |

32 isolation probes run inside fully sandboxed workers (CPU and CUDA) and are
all blocked: TCP, UDP and Unix sockets, connecting to loopback, reading
`/etc/shadow` and `/etc/passwd`, listing `/`, writing `/tmp` and `/var/tmp`,
`exec`, `fork`, `ptrace` of the parent, signals to the parent and to init,
`mount`, `unshare`, `setuid(0)`, opening `/dev/mem`, `/dev/sda` and input
devices, reading the Owner's `~/.ssh/id_ed25519`, listing `~/.ssh`, `~` and
`$XDG_RUNTIME_DIR`, writing to `~`, reading another session's scratch file,
the server key and the identity database, writing into the runtime, and
listing the dataset store.

## Revised performance targets (§17.6)

ResNet-18, CIFAR-10 shapes (batch 128, 3×32×32), SGD with momentum, client and
server on the same machine over loopback TLS.

| Metric | Target | Measured |
|---|---|---|
| Throughput vs plain PyTorch on the same GPU | ≥ 70% | 74% with `reflop.capture` (5.6 ms vs 4.2 ms per step); 19% eager (23 ms) |
| Upload per step after warm-up | ≤ 16 KiB | 0 KiB of tensor data with capture (inputs already on the server); about 1 KiB per step of batch indices with a server-side dataset |
| Client overhead per step on a cache hit | ≤ 5 ms | well under 1 ms with capture (one request per step); eager mode spends most of its 23 ms per step recording operations |
| Validation of a 10k-node graph | ≤ 1 s | 0.72 s (V1–V5, CPU and CUDA sessions); ResNet-18's step graph has about 360 nodes after dead-code elimination |

Revised targets (ADR-014): the ≥ 70% throughput target applies to steady-state
steps under `reflop.capture`; eager mode is a convenience path. The M2 MacBook
over Tailscale (§19) remains to be measured.
