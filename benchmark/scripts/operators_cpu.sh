#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/lib/launch.sh"
export BENCH_REMOTE_KIND=cpu
bench_run operators
