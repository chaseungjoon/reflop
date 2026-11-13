# Contributing to reFLOP

Read [docs/PLAN.md](docs/PLAN.md) first: it is the design, and its
[invariants (§3)](docs/PLAN.md#3-security-invariants-never-violate) are never
weakened, not even temporarily or behind a flag. Where the code differs from
the plan, [docs/DECISIONS.md](docs/DECISIONS.md) says why; new decisions go
there as ADRs.

Everyone taking part in the project follows the
[Code of Conduct](CODE_OF_CONDUCT.md).

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
- Adding an operator
  ([PLAN.md §8.8](docs/PLAN.md#88-adding-an-operator-procedure-r5)) takes, in
  one change: the `opset.yaml` entry with a new id, a validator for any `IDX`,
  `ALLOC`, `RESTRICTED` or `DDS` flag, a cost function, a conformance test, a
  security test for each flag, and `python codegen/gen.py`. CI checks that
  generated files are current.
- Op ids are permanent; never reuse one.
- Never cache a per-execute check
  ([rule R7](docs/PLAN.md#0-agent-working-rules)). Caching structural
  validation by graph hash is fine.
- Every bug fix adds a regression test; every security fix adds a case to
  `tests/security/`.
- Commit messages, comments and docs are in English. Code is Apache-2.0; do not
  vendor GPL or AGPL code.

## Branches

Branch from `main` and name the branch after the kind of change:

| Change | Branch |
|---|---|
| a security issue or hardening | `security/<branch-name>` |
| a bug fix | `hotfix/<branch-name>` |
| an enhancement or new feature | `enhance/<branch-name>` |

Use a short kebab-case name, e.g. `hotfix/uv-lock-update`, and open a pull
request into `main`. Do not report an exploitable vulnerability in a public
issue or pull request; follow [SECURITY.md](SECURITY.md) first.

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
| `docs/` | the design (`PLAN.md`), ADRs, threat model, Owner and Guest guides |
| `tests/` | conformance, security corpus, isolation, e2e, perf |
| `spikes/` | M0 spikes |
