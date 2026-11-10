# Contributing to reFLOP

Read [PLAN.md](PLAN.md) first: it is the design, and its invariants (§3) are
never weakened, not even temporarily or behind a flag. Decisions go into
[docs/DECISIONS.md](docs/DECISIONS.md) as ADRs.

## Setup

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch torchvision protobuf pytest hypothesis pyyaml \
    cryptography safetensors numpy ruff
make build test lint
```

On a Linux machine with an NVIDIA GPU, the tests also run on CUDA. To run the
sandbox probes you need an installed runtime:

```sh
REFLOP_HOME=/tmp/rf ./bin/reflopd runtime install
REFLOP_TEST_HOME=/tmp/rf make test-isolation
```

## Rules

- Server-side Python passes `tests/security/static/check_worker.py`. Do not add
  exceptions to it.
- Adding an operator (PLAN.md §8.8) takes, in one change: the `opset.yaml`
  entry with a new id, a validator for any `IDX`, `ALLOC`, `RESTRICTED` or `DDS`
  flag, a cost function, a conformance test, a security test for each flag,
  and `python codegen/gen.py`. CI checks that generated files are current.
- Op ids are permanent; never reuse one.
- Never cache a per-execute check (rule R7). Caching structural validation by
  graph hash is fine.
- Every bug fix adds a regression test; every security fix adds a case to
  `tests/security/`.
- Commit messages, comments and docs are in English. Code is Apache-2.0; do not
  vendor GPL or AGPL code.

## Layout

| Path | What |
|---|---|
| `proto/reflop/v1/` | wire protocol; `codegen/gen_proto.sh` regenerates Go and Python code |
| `opset/v1/opset.yaml` | the operator whitelist; `codegen/gen.py` generates the tables |
| `server/reflopd/` | the Go daemon |
| `server/worker/reflop_worker/` | the Python worker: validation and execution |
| `client/python/reflop/` | the Python client |
| `clib/` | the C/C++ API |
| `pkg/` | Go packages shared by reflopd and libreflop |
| `tests/` | conformance, security corpus, isolation, e2e, perf |
| `spikes/` | M0 spikes |
