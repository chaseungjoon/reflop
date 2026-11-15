#!/usr/bin/env bash
# Sourced by the no-argument entry points in scripts/.
set -euo pipefail
BENCH_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_REPO="$(dirname -- "$BENCH_ROOT")"
if [[ -n "${BENCH_PYTHON:-}" ]]; then
  BENCH_INTERPRETER="$BENCH_PYTHON"
elif [[ -x "$BENCH_ROOT/.venv/bin/python" ]]; then
  BENCH_INTERPRETER="$BENCH_ROOT/.venv/bin/python"
elif [[ -x "$BENCH_REPO/.venv/bin/python" ]]; then
  BENCH_INTERPRETER="$BENCH_REPO/.venv/bin/python"
else
  BENCH_INTERPRETER="$(command -v python3)"
fi
if [[ "$#" -ne 0 ]]; then
  echo "These launchers take no arguments. Edit benchmark/config.json or use BENCH_CONFIG." >&2
  exit 2
fi
cd -- "$BENCH_REPO"
export PYTHONDONTWRITEBYTECODE=1

bench_run() {
  exec "$BENCH_INTERPRETER" -u -m benchmark.bench.runner "$1"
}
