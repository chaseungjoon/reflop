#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/lib/launch.sh"
VENV="$BENCH_ROOT/.venv"
BENCH_SOURCE="${BENCH_REFLOP_SOURCE:-$BENCH_REPO}"
if [[ ! -f "$BENCH_SOURCE/client/python/pyproject.toml" ]]; then
  echo "BENCH_REFLOP_SOURCE must contain client/python/pyproject.toml" >&2
  exit 2
fi
if [[ ! -x "$VENV/bin/python" ]]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv --python "$BENCH_INTERPRETER" "$VENV"
  else
    "$BENCH_INTERPRETER" -m venv "$VENV"
  fi
fi
if command -v uv >/dev/null 2>&1; then
  uv pip install --python "$VENV/bin/python" -r "$BENCH_ROOT/requirements.txt"
  uv pip install --python "$VENV/bin/python" -e "$BENCH_SOURCE/client/python"
else
  "$VENV/bin/python" -m pip install -r "$BENCH_ROOT/requirements.txt"
  "$VENV/bin/python" -m pip install -e "$BENCH_SOURCE/client/python"
fi
mkdir -p "$BENCH_ROOT/cache"
"$VENV/bin/python" -m pip freeze > "$BENCH_ROOT/cache/installed-requirements.txt" 2>/dev/null || \
  "$VENV/bin/python" -c 'import importlib.metadata as m; print("\n".join(sorted("{}=={}".format(d.metadata["Name"],d.version) for d in m.distributions())))' > "$BENCH_ROOT/cache/installed-requirements.txt"
BENCH_PYTHON="$VENV/bin/python" "$BENCH_ROOT/scripts/selfcheck.sh"
echo "Bootstrap complete. No reflop runtime was imported or contacted. Pair separately, then run smoke.sh."
