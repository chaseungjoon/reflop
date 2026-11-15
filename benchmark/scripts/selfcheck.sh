#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/lib/launch.sh"
exec "$BENCH_INTERPRETER" -B -m benchmark.tests.selfcheck
