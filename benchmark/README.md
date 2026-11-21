# reFLOP benchmark

Measure reFLOP throughput against native PyTorch on the same GPU, then check correctness, operator support, data movement, concurrency, checkpoints, and failure behavior. The suite uses synthetic workloads and saves each case separately, including failures and missing prerequisites.

The shell launchers are executable, **take no command-line arguments**, and work from any working directory. Edit `config.json` for settings. Each benchmark invocation creates a fresh directory under `benchmark/results/`. The measured launchers produce raw evidence. See the [results analysis and graphs](report/README.md) for the 15 November 2026 (UTC) native GPU comparison from a MacBook Pro over Tailscale TLS, the historical 14 November analysis, and plot reproduction instructions.

## Start here

```bash
./benchmark/scripts/bootstrap.sh
./benchmark/scripts/catalog.sh
# Once the owner has started reflopd and this client has been paired:
./benchmark/scripts/smoke.sh
./benchmark/scripts/inventory.sh
./benchmark/scripts/correctness.sh
./benchmark/scripts/throughput.sh
```

`bootstrap.sh` creates `benchmark/.venv`, installs `requirements.txt` and this checkout’s Python client in editable mode, records installed versions in `cache/installed-requirements.txt`, and runs offline self-checks. It does **not** import reFLOP, pair a client, start a daemon, contact a GPU, change server settings, or run a live benchmark. Installation needs package-index connectivity. If `uv` is installed it is used; otherwise Python's `venv` and `pip` are used. Python 3.10+ is required.

An existing paired client configuration is reused. Pairing is deliberately separate: redeem your owner-issued invite with the `reflop` executable from the Python environment you choose. On macOS, pairing may use Keychain. Never put an invite or private key in this directory.

Launchers choose `BENCH_PYTHON`, then `benchmark/.venv/bin/python`, then the repository's `.venv/bin/python`, then `python3`. You can use the already installed environment immediately without installing another copy:

```bash
BENCH_PYTHON="$PWD/.venv/bin/python" ./benchmark/scripts/selfcheck.sh
BENCH_PYTHON="$PWD/.venv/bin/python" ./benchmark/scripts/smoke.sh
```

Bootstrap defaults to this repository's `client/python`. Set `BENCH_REFLOP_SOURCE` to another checkout's absolute root to use it for the Python client and C/C++ builds. If you supply an existing `BENCH_PYTHON`, check its installed reFLOP version before measuring: launchers do not reinstall packages. Results record package versions, source hashes, and the selected checkout's Git commit and working-tree status.

## Compare native GPU and reFLOP throughput

The main metric is **retained throughput**:

```text
remote samples/second / native samples/second × 100%
```

100% means equal throughput. Also report absolute samples/second and milliseconds/step, model, batch size, data policy, network conditions, and correctness status. A ratio between reFLOP capture and reFLOP eager answers a different question.

The [saved report](report/README.md) compares 156 remote cases with 78 native eager cases on the Owner's RTX 5070. Native ran locally on the Linux Owner workstation; the Guest ran on a MacBook Pro over Tailscale TLS. At batch 128, resident capture retains 35.6–70.1% of native throughput for eight workloads and 4.0–4.6% for LSTM/GRU. ResNet-18 correctness remains unresolved in the available evidence. The root README's numbers come from a later run (21 November 2026) whose raw data is not saved here.

### 1. Collect the native baseline on the Owner GPU

Use the same checkout and configuration on the Linux machine that serves the GPU. A native run needs local CUDA, Torch, and torchvision; it does not need a paired reFLOP client or a running daemon. Set `native_cuda_label` in `config.json` to identify the GPU and location. Use `CUDA_VISIBLE_DEVICES` if needed to select the physical GPU used by reFLOP; native cases use visible device 0.

```bash
./benchmark/scripts/bootstrap.sh
./benchmark/scripts/native_cuda.sh
```

Save the printed run directory. With the default batches, this runs 78 native eager cases: ten workloads × four batches × two input policies, excluding ResNet-18 batch 1. No local CUDA means `skipped`, never a CPU substitute. Stop other GPU work during measurement. Do not run native and remote benchmarks at the same time.

### 2. Measure reFLOP from the Guest

Use the same benchmark source and timing configuration. The Owner runtime's PyTorch/CUDA versions and numerical policy must match the native environment. Record the driver version, GPU identity, any clocks or power limits, and the connection's RTT and bandwidth alongside the runs.

```bash
# Owner has started reflopd; Guest has paired separately.
./benchmark/scripts/inventory.sh
./benchmark/scripts/correctness.sh
./benchmark/scripts/throughput.sh
```

The default throughput selection contains 156 workload cases plus six synchronization cases. `resident` uploads one synthetic batch before timing and reuses it. `upload` includes a CPU-to-device copy each step; native CUDA copies over the Owner's local bus, while reFLOP copies across the Guest-to-Owner connection. This difference is part of the user-visible overhead. Neither policy includes a real data loader or the initial model transfer.

### 3. Generate the comparison

Copy the complete native run directory to the machine with the remote results. Install the optional plot dependencies, then supply both absolute paths. Set `BENCH_SAME_GPU=1` only after verifying they used the same physical GPU; matching device names alone cannot prove this.

```bash
uv pip install --python benchmark/.venv/bin/python -r benchmark/requirements-plots.txt
BENCH_REMOTE_RUN=/absolute/path/to/throughput-run \
BENCH_NATIVE_RUN=/absolute/path/to/native-cuda-run \
BENCH_SAME_GPU=1 ./benchmark/scripts/compare_cuda.sh
```

The command creates a new ignored `results/<id>-cuda-comparison/` directory with a Markdown table, CSV, JSON, and PNG/SVG graph. It performs no GPU work. It matches model, batch, optimizer, input residency, and synchronization policy; checks seed, CPU threads, warmup/measurement/repeat counts, model/training source hashes, GPU name/memory, Owner/native Torch and CUDA versions, and all four negotiated numerical flags. Mismatches stop the comparison. Failed, skipped, and missing pairs remain `N/A`, never zero. The separate synchronization sweep is excluded.

Driver, power state, competing load, physical GPU identity, and network conditions still require operator verification. The comparator does not infer numerical correctness from a timing pass or compute confidence intervals. Preserve repeated runs to quantify noise before using a threshold. Its baseline is eager PyTorch, not native CUDA graphs or `torch.compile`.

## Layout

```text
benchmark/
  README.md                  benchmark guide
  report/                    analysis, PNG/SVG figures, reviewed data snapshots
  requirements-plots.txt     optional graph-rendering dependencies
  config.json                editable sizes, durations, endpoints, hooks
  requirements.txt           benchmark dependencies
  scripts/*.sh               individual, no-argument launchers
  lib/launch.sh              common interpreter and working-directory setup
  bench/                     catalog, isolated runner, probes, references
  native/                    standalone C/C++ correctness/performance clients
  tests/                     offline harness and native-PyTorch verification
  .venv/                     generated by bootstrap; ignored
  cache/                     generated CPU operator recipes; ignored
  results/<run-id>/           immutable-per-run evidence; ignored
```

Generated environments, caches, and raw results are ignored by `benchmark/.gitignore`. Reviewed report data and figures are kept under `report/`.

## Launchers and coverage

Run `operators.sh` before replaying individual operator cases so the CPU recipes are available. Other suite launchers can run independently. `run_all.sh` includes all 29 benchmark suites, including long runs and configured fault hooks; it continues after individual case failures. With the 582-entry registry and default configuration, the catalog contains **1,388 cases**. Counts change when the installed client's registry or configuration changes. `catalog.sh` records the exact planned cases without importing or executing reFLOP.

| Script in `scripts/` | Measurements and checks |
| --- | --- |
| `bootstrap.sh` | Dependency setup and offline self-checks; no daemon required. |
| `selfcheck.sh` | Blocks real reFLOP imports and socket connections; checks catalog completeness, launchers, result handling, crashes/timeouts, CPU models, gradients, operator recipe replay, checkpoint integrity. Uses temporary files, not live benchmark results. |
| `catalog.sh` | Exact machine-readable inventory of every configured case. No GPU activity. |
| `smoke.sh` | Small diagnostic, MLP eager/capture correctness, tensor/view, transfer, and checkpoint selection. |
| `inventory.sh` | Client/dependency inventory; negotiated devices, enabled ops, numerics, limits, quotas; CLI status/verify/doctor; actual TLS version/cipher or SSH transport. |
| `operators.sh` | One case for **every entry** in the installed client's operator registry: forward/backward ops, overloads, RNG, allocation, client-side metadata, banned ops, and reFLOP-native primitives. |
| `operators_cpu.sh` | The same full registry census against the **server CPU**. This overrides `remote_kind` for that invocation. |
| `tensors.sh` | Arithmetic, special functions, broadcasts, reductions, scans, sort/top-k, matmul, 1D/2D/3D convolution/transposed convolution, depthwise, pooling, norms, losses, CTC, embeddings, indexing/scatter/masks, factories, alias mutation, out variants, scalar/empty/zero/singleton/high-rank/noncontiguous/channels-last shapes, dtype conversions, half/BF16/FP64/complex/FP8, interpolation/padding/grid sampling, attention, FFT, linalg, distances, RNG, sparse and quantized compatibility boundaries, NaN/Inf/signed-zero/subnormal behavior. |
| `autograd.sh` | First/second derivatives for smooth functions, broadcasting, matmul, convolution, embeddings, normalization, attention, views, and loss. |
| `correctness.sh` | Ten model families × eager/capture × remote CUDA/server CPU. Matches loss trajectories, final parameters/buffers, and gradients against identical native CPU references. |
| `optimizers.sh` | SGD, momentum, Nesterov, Adam, AdamW, capturable/fused variants, Adagrad, RMSprop, Adadelta, Adamax, Rprop, ASGD, LBFGS; eager/capture and both zero-grad policies. Unsupported capture paths remain visible as limitations. |
| `precision.sh` | FP64/FP32/FP16/BF16 storage, BF16 autocast, FP16 autocast with GradScaler, eager/capture, with explicit numerical tolerances. |
| `capture.sh` | Changing input values/shapes/dtypes, mutated CPU inputs, changing Python scalars/branches, stale views, RNG, nested outputs, gradient accumulation, learning-rate schedules. Requires correct results or explicit refusal for dynamic Python schedules/branches. |
| `throughput.sh` | Ten workloads × configured batch sizes × eager/capture × client upload/resident input; separate synchronization-frequency sweep. Warmups, repeated synchronized windows, samples/sec, and wire/graph counters. |
| `native_cpu.sh` | Identical workload/batch/input-residency matrix on native CPU. A reference and overhead comparison, **not** a GPU throughput baseline. |
| `compare_cuda.sh` | Offline native-versus-reFLOP table and graphs; requires completed runs and the plot dependencies. |
| `native_cuda.sh` | Identical matrix on native local CUDA. Run this on the **owner GPU** using a copy of this benchmark. Without local CUDA, cases are skipped, never silently run on CPU. |
| `transfer.sh` | Zero-byte through 64-MiB payloads by default; upload, fetch, roundtrip; synchronized timing and content verification. |
| `graphs.sh` | Matrix-size sweep, arithmetic intensity, tiny-op RPC/flush amplification, batched chains, graph cache reuse, shape/scalar churn, multiple outputs, dead code, alias mutation. |
| `datasets.sh` | Synthetic TensorPack creation/push/batching, upload interruption/resume, dedup, checksum/manifest rejection, remote loader training, captured loader training, byte counters. Only uniquely named benchmark datasets are deleted. |
| `sessions.sh` | Open/close latency, simultaneous GPU/server-CPU sessions, cross-session moves and differentiable moves, implicit mixing rejection, closed tensors, deterministic seed, threaded calls. |
| `placement.sh` | `auto`, strict `remote`, forced `local`, and `remote-cpu`; checks values and placement. |
| `concurrency.sh` | 1/2/4/8 client processes by default, synchronized start barriers, repeated alone-versus-together trials with alternating order, aggregate scaling, per-client latency, Jain fairness. |
| `isolation.sh` | Foreign session handles; cross-identity handles and dataset visibility/deletion. Distinct identity cases require two paired guest homes. |
| `checkpoint.sh` | Safe tensor weights, duplicate storage, complete optimizer/scheduler/sampler/step/client RNG serialization, integrity rejection, manual new-session continuation, server snapshot/restore/delete. |
| `migration.sh` | Manual full-state restoration to a different paired server; restore timing, rollback steps, and continuation equality. Requires `migration_target`. |
| `disconnect.sh` | Idle transport reconnect preserving handles, transport interruption immediately after an execute submission, unknown resume ID rejection. No owner access required. |
| `faults.sh` | Owner pause/resume, session kill, configured quota, network partition, worker kill; bounded observation and restore hooks. Skips before changing anything unless the required hooks exist. |
| `limits.sh` | Bounded allocations, handle churn/client RSS, concurrent-session ceiling, long graphs, invalid indexing/shapes/integer division, post-error health. Does not allocate until an unbounded OOM. |
| `soak.sh` | **30 minutes per mode** by default: GPT eager/capture, step progress, finite loss, RSS, thread count, counters, replay count. Emits raw trends without declaring every RSS increase a leak. |
| `wan.sh` | Matched synchronized workloads at three explicitly configured real network distances; resident/upload × eager/capture; authenticated protocol-ping RTT. Never relabels one default endpoint as three distances. |
| `shim.sh` | Standalone CUDA-style training through `reflop run`; forward/backward, AMP, CUDA API compatibility. The child asserts tensors really use the reFLOP backend. |
| `c_api.sh` | Builds C/C++ clients into that case's result directory, checks numerical results, records execute/fetch timing. Requires Go, C/C++ compilers, a source checkout, and a file-backed paired key. |
| `protocol.sh` | Unknown ops, malformed graph structure, foreign/unknown handles, oversized response framing, terminal escape sanitization, wrong TLS server pin rejection. Bounded probes against your own paired server. |
| `compatibility.sh` | `torch.compile`, export, TorchScript, activation checkpointing, custom autograd, gradient hooks/clipping, shared parameters, scheduler, single/spawn-worker DataLoaders, and GradScaler nonfinite-update skipping. Each is an isolated compatibility boundary, not a universal feature claim. |
| `run_all.sh` | One directory containing the entire configured catalog, its case outcomes, and artifacts. |

The ten workloads are: dense MLP, CNN with batch normalization, ResNet-18, small U-Net, two-layer LSTM, two-layer GRU, attention classifier, causal GPT, frozen-base LoRA-style adapter, and autoencoder. All use deterministic **synthetic** data and random initial weights; no pretrained models, external datasets, or downloads. ResNet-18 batch 1 is excluded from performance matrices because training BatchNorm receives only one value per channel. These workloads measure runtime behavior; they do not measure convergence or large-model quality.

The throughput matrix can take hours, especially on native CPU. The full run also includes two 30-minute soak cases and hundreds of independent operator processes. Start with smoke, then select suites. Reduce `measured_steps`, `repeats`, or matrix lists when iterating; the effective configuration is saved with every run. Capture warmup must remain at least three steps.

## Configuration

Edit `benchmark/config.json` once, then launch scripts without flags. Alternatively set `BENCH_CONFIG` to an absolute path containing another complete configuration. There is no implicit command-line configuration merge.

Important fields:

- `server`: paired server name, or `null` to use `REFLOP_SERVER`/the paired default.
- `client_home`: paired client directory, or `null` to retain the normal client configuration. No keys are copied.
- `remote_kind`: `cuda` by default; requests CUDA explicitly so a missing GPU cannot masquerade as a GPU result. Correctness deliberately tests both remote kinds; throughput deliberately targets remote CUDA.
- `seed`, `cpu_threads`, `warmup_steps`, `measured_steps`, `repeats`, `correctness_steps`: reproducible input and measurement controls.
- `rtol`/`atol`, `low_precision_rtol`/`low_precision_atol`: saved acceptance tolerances. Reduced-precision/TF32 policy is also available in negotiated capability artifacts. A failed comparison is evidence to investigate, not a reason to loosen thresholds silently.
- `transfer_bytes`, `matrix_sizes`, `batch_sizes`, `concurrency`: explicit matrix dimensions.
- `timeout_seconds`: hard per-case wall-clock ceiling, including connection, setup, execution, and cleanup. Slow cases have dedicated catalog timeouts. A timeout terminates the process group; a grace period lets Python cleanup run.
- `memory_budget_bytes`, `memory_cycles`, `max_sessions`, `max_graph_chain`: bounded stress ceilings. A successful bounded run is a **lower bound**, not a claim that the server has no limit.
- `soak_seconds`, `soak_sample_seconds`: duration and raw trend sampling.
- `guest_homes`: absolute paths to separately paired client homes. Concurrency cycles these homes when provided; otherwise it measures multiple sessions of one identity. Cross-identity isolation requires two homes that authenticate as different identities.
- `migration_target`: another paired server name and optional client home. Target and source names must differ; confirm they actually correspond to different GPUs/machines.

`REFLOP_LOCAL` is removed from child environments: live cases cannot silently use an in-process worker. Normal compatibility/performance cases disable local fallback. `placement.sh` explicitly tests the placement policies separately. Capability and protocol versions are recorded rather than assumed from a package version.

### Native, LAN, and WAN measurements

For the Phase 0 gate, collect the baseline on the same owner GPU, with the same framework/runtime version, batch/model/precision configuration and comparable numerics. Copy `benchmark/` to the owner inside a project directory, run bootstrap there, then:

```bash
./benchmark/scripts/native_cuda.sh
```

Copy the **whole resulting run directory** back under this client's `benchmark/results/`, retaining its unique name. Set `native_cuda_label` to describe the actual GPU/region before collecting it. The benchmark does not assume SSH access to the owner or transfer credentials automatically.

On the client, fill all three `network_targets` entries with paired server names and accurate location/path descriptions, then run:

```bash
./benchmark/scripts/wan.sh
```

Use distinct client vantage points against the same GPU when possible. Different GPUs at different distances confound hardware and network effects; label that experiment accurately. The suite measures authenticated application RTT, not ICMP latency, and does not install traffic-shaping rules. Missing targets are skipped. If collecting from three different client machines, keep the same configuration and retain all three run directories.

### Owner fault hooks

Default hooks are empty. Configuring an argv array authorizes that specific command when `faults.sh` or `run_all.sh` reaches its case. Commands are executed directly, without `shell=True`. The suite does not invent SSH hosts, install owner agents, change global network rules, reboot a machine, or modify quotas on its own.

Hook pairs: `pause`/`resume`, `quota_apply`/`quota_restore`, `network_down`/`network_up`, `worker_kill`/`worker_restore`. `kill_session` needs no inverse because it targets a disposable benchmark session. Keep owner commands narrowly scoped and make restore hooks idempotent. For example, existing owner CLI commands can be configured as:

```json
{
  "pause": ["ssh", "your-owner", "reflopd", "pause"],
  "resume": ["ssh", "your-owner", "reflopd", "resume"],
  "kill_session": ["ssh", "your-owner", "reflopd", "kill", "{session_id}"]
}
```

This is an example for fields inside `owner_hooks`, not a replacement for the whole configuration. `ssh` also interprets its remote command according to SSH's normal behavior; use trusted fixed arguments. `{session_id}`, `{identity}`, and `{server}` placeholders are substituted per argument. The same values are exported to hook processes as `BENCH_SESSION_ID`, `BENCH_IDENTITY`, and `BENCH_SERVER`. Inspect `hooks.log` for output and exit codes. Use `fault_wait_seconds` to control the interval before the inverse hook runs.

Restore hooks execute in `finally`, with a second path after errors. As with any process, SIGKILL, machine failure, or loss of the management network can prevent cleanup. Keep an independent owner-management path available. A fault run is not proof of cloud recovery: worker/session kills may correctly surface `SessionLostError`. Automatic cross-provider failover is not implemented here or asserted to exist.

For C/C++, the native library cannot read macOS Keychain. Export the already paired identity once with `reflop key --write-file` in the intended client home before `c_api.sh`. The launcher never exports a private key automatically. Native library artifacts stay under the case's `build/` directory.

## Evidence format

Every measured invocation creates:

```text
results/20261113T120000Z-throughput-1a2b3c4d/
  run.json                       configuration, lifecycle, counts, exit code
  environment.json               packages, platform, source hashes
  catalog.json                   exact case IDs, parameters, timeouts, plan refs
  results.jsonl                  one self-contained result per completed case
  cases/<suite>/<case-path>/
    request.json                 reproducible case/config snapshot
    result.json                  durable parent result (also in results.jsonl)
    outcome.json                 child outcome, when the child completed
    stdout.log / stderr.log      unfiltered child diagnostics
    capabilities-<index>-<n>.json negotiated capabilities per connection
    execute_metrics.json         server-reported duration/flops/peak memory
    events.jsonl                 per-repeat, correctness, or soak observations
    ...                          checkpoints, packs, native builds, hook logs
```

Some case-specific files are absent when a case stops before that phase. Results are flushed after each case; later failures do not erase earlier evidence. Ctrl-C records the interrupted case/run and stops scheduling new cases. A killed process may leave only `request.json` and logs; the parent records the timeout or crash when it remains alive. An abruptly killed parent leaves `run.json` in `running`, which must be treated as incomplete.

`catalog.sh` creates the run metadata and catalog only. `bootstrap.sh` and `selfcheck.sh` are setup/validation commands, not measured benchmark runs, so they do not create `results/` directories. CPU recipe preparation is included as `operators/prepare` when executing the operator suite.

Statuses:

| Status | Meaning |
| --- | --- |
| `pass` | This case's explicit assertions passed; a negative test can pass by observing the specified rejection. |
| `fail` | Wrong values, unexpected error, missing expected behavior, or cleanup failure. |
| `unsupported` | Explicit unsupported operator/capture/backend response; useful compatibility evidence, not success. |
| `skipped` | A required native GPU, identity, endpoint, hook, key, tool, or prepared fixture was unavailable. |
| `coverage_gap` | Registry entry has no safe validated input recipe in this environment. It was inventoried but not executed. |
| `fixture_error` | Native CPU reference/fixture failed; cannot attribute it to reFLOP. |
| `timeout` | Child exceeded its hard wall-clock deadline. |
| `error` | Worker crash, malformed/missing outcome, or harness failure. |
| `interrupted` | User/process interruption. |

Exit code 1 indicates any fail/error/timeout/fixture error; 130 indicates interruption. Unsupported, skipped, and coverage-gap outcomes do not make a launcher exit nonzero. **Exit 0 is not a viability verdict or proof of complete coverage.** Inspect the status counts and individual case records.

Remote errors retain their code, validation stage, node, and operator name when the client provides them. Failures are labeled by phase: connection, native reference, remote execution, correctness, or cleanup. All cases have stable IDs and plan-step references; registry IDs can change between client versions, so compare operator names as well as versions.

### Timing interpretation

- Throughput windows include forward, loss, zero-grad, backward, optimizer, the selected input upload policy, and a final synchronization. Input generation/model construction and warmup are excluded. Each repeat records total elapsed time and derives end-to-end seconds/step and samples/sec.
- `sync_every=1` measures synchronized step latency. `sync_every=0` measures throughput; individual step samples are **submission times**, so use `repeat_step_average` for throughput comparisons. Periodic synchronization is labeled explicitly.
- Warmup and repeat samples are retained. p90/p95/p99 use linear interpolation over recorded samples; default sample counts are small and do not establish a production tail-latency SLO. Repeat averages are not independent per-step latency samples.
- Native CUDA uses eager PyTorch with the same workload and data policy; remote capture is compared to that baseline. No native CUDA graph or `torch.compile` optimization is implied.
- Input-resident runs still execute model training remotely. Client-upload runs include uploads; TensorPack loaders have their own residency/traffic measurements. Read their counters before making egress claims.
- Server-reported execution duration, FLOPs, and memory are stored separately from client wall time. Zero/absent server metrics mean unavailable instrumentation, not zero resource use. These fields and client counters are runtime telemetry, not trusted billing records or NVML utilization measurements.
- Connection/setup cost is excluded from steady-state timing but retained in case wall time and explicit open/close/reconnect measurements. Each case runs in a fresh process to isolate failed traces and resource cleanup; this intentionally sacrifices connection reuse across cases.

### Operator coverage and numerical limits

The installed client's generated registry is parsed as data without importing reFLOP. Native CPU forwards/backwards/optimizers produce concrete low-level argument recipes; schema-generated supplements must execute repeatably on CPU before use. Recipes use JSON and safetensors, not pickle. With the development environment used here, discovery generated **555 native recipes** for a 582-entry registry. Eleven reFLOP-native primitives and banned `resize_` have dedicated runtime checks. Remaining device-specific kernels/workspaces are explicit coverage gaps, rather than fabricated inputs or successful compatibility claims.

This is systematic finite coverage of the exposed framework, not a mathematical promise to test every tensor shape, Python program, hardware combination, or possible failure. CUDA-only workspace kernels may need owner-generated fixtures or dedicated recipes before direct conformance is possible. High-level model/attention/RNN probes still exercise the backend's selected execution paths. Add native recipes in `fixtures.py` and deterministic expected behavior in `worker.py` when expanding that coverage. The registry census automatically includes new names and makes missing recipes visible.

Random algorithms are tested for same-backend seed reproducibility; CUDA and CPU random sequences need not match. Empty remote allocations are checked for zero initialization, rather than compared with uninitialized CPU memory. Eigenvector/SVD sign conventions, tied singular values, floating-point reduction order, subnormal flushing, TF32 and mixed precision can explain discrepancies; inspect raw failures and negotiated numerics before attributing a bug. Direct decomposition-output comparisons can expose harmless sign differences and need residual-based follow-up. The benchmark intentionally retains those discrepancies.

Full training checkpoints contain model state, optimizer, scheduler, sampler/permutation/cursor, step, client Torch/Python RNG, versions, and hashes. The controlled continuation test reseeds the remote generator by global step. **Arbitrary remote RNG state, native CUDA RNG, NumPy/data-worker RNG, and third-party data pipelines are not snapshotted.** Manual migration reports restored progress and timing; it does not claim transparent process continuation or production RTO/RPO guarantees. Duplicate-weight export checks values, not preservation of alias identity.

## Review and publication

Run `selfcheck.sh` before submitting changes to the harness. Keep routine CI offline; select live suites on a GPU runner or run them manually against a paired test server. `run_all.sh` includes long runs and any configured Owner hooks, so inspect the configuration before using it.

Publish a native CUDA comparison with the remote run, effective configuration, versions, workload hashes, hardware and network description, repeat counts, and correctness outcomes. Keep failed and unavailable cases visible. The saved `plan_refs` fields are legacy identifiers from the suite's original evaluation project; they are not dependencies or acceptance criteria for reFLOP.

Before sharing raw results, review configurations, logs, addresses, identities, local paths, and hook output. Never publish keys or pairing invites. Keep full runs in CI or release artifacts; put only reviewed snapshots and figures in `report/`. Runtime probes do not replace the security corpus or establish production reliability.
