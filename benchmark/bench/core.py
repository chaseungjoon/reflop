from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent


class MissingPrerequisite(Exception):
    pass


class CoverageGap(Exception):
    pass


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_config():
    path = Path(os.environ.get("BENCH_CONFIG", ROOT / "config.json")).resolve()
    c = json.loads(path.read_text())
    if os.environ.get("BENCH_REMOTE_KIND"):
        c["remote_kind"] = os.environ["BENCH_REMOTE_KIND"]
    if c.get("schema_version") != 1:
        raise ValueError("config.schema_version must be 1")
    for key in ("timeout_seconds", "measured_steps", "repeats", "correctness_steps", "cpu_threads",
                "soak_seconds", "soak_sample_seconds", "memory_budget_bytes", "max_sessions",
                "memory_cycles", "dataset_samples", "dataset_shard_bytes", "concurrency_rounds",
                "recovery_deadline_seconds", "max_graph_chain"):
        if not isinstance(c[key], int) or isinstance(c[key], bool) or c[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if c["warmup_steps"] < 3:
        raise ValueError("warmup_steps must be >= 3 so capture can stabilize")
    if c["remote_kind"] not in ("cuda", "cpu"):
        raise ValueError("remote_kind must be cuda or cpu (auto would hide missing GPUs)")
    for key in ("transfer_bytes", "matrix_sizes", "batch_sizes", "concurrency"):
        if not c[key] or any(not isinstance(n, int) or n < (0 if key == "transfer_bytes" else 1) for n in c[key]):
            raise ValueError(f"invalid {key}")
    for key, cmd in c["owner_hooks"].items():
        if not isinstance(cmd, list) or any(not isinstance(v, str) for v in cmd):
            raise ValueError(f"owner_hooks.{key} must be an argv array, never a shell string")
    return c


def opset_source():
    try:
        d = importlib.metadata.distribution("reflop")
        path = Path(d.locate_file("reflop/_opset_generated.py"))
        if path.is_file():
            return path
        # Editable distributions can point at a source tree.
        direct = json.loads(d.read_text("direct_url.json") or "{}")
        if direct.get("url", "").startswith("file://"):
            from urllib.parse import unquote, urlparse
            path = Path(unquote(urlparse(direct["url"]).path)) / "reflop/_opset_generated.py"
            if path.is_file():
                return path
    except importlib.metadata.PackageNotFoundError:
        pass
    return Path(os.environ.get("BENCH_REFLOP_SOURCE", REPO)) / "client/python/reflop/_opset_generated.py"


def read_opset():
    path = opset_source()
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "OPS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"no literal OPS dictionary in {path}")


def fingerprint():
    h = hashlib.sha256()
    for p in sorted(ROOT.rglob("*.py")):
        if not any(v in p.parts for v in (".venv", "results", "cache")):
            h.update(str(p.relative_to(ROOT)).encode())
            h.update(p.read_bytes())
    h.update(opset_source().read_bytes())
    return h.hexdigest()


def environment():
    packages = {}
    for d in importlib.metadata.distributions():
        packages[d.metadata["Name"]] = d.version
    sources = {}
    for p in sorted(ROOT.rglob("*")):
        if p.is_file() and p.suffix in (".py",".sh",".c",".cpp",".json",".txt") and not any(
                s in p.relative_to(ROOT).parts for s in (".venv","results","cache","__pycache__")):
            sources[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "machine": platform.machine(), "cpu_count": os.cpu_count(), "packages": packages,
            "opset_source": str(opset_source()), "benchmark_sha256": fingerprint(),"source_hashes":sources,
            "git": source_revision(),
            "environment_controls": {k: os.environ[k] for k in
                ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES") if k in os.environ}}


def source_revision():
    source = Path(os.environ.get("BENCH_REFLOP_SOURCE", REPO)).resolve()
    def git(*args):
        result = subprocess.run(["git", "-C", str(source), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"source_root": str(source), "commit": git("rev-parse", "HEAD"),
            "status": git("status", "--porcelain")}


def distribution(samples, unit="seconds"):
    if not samples or any(not math.isfinite(x) or x < 0 for x in samples):
        raise ValueError("measurements must be finite nonnegative numbers")
    values = sorted(samples)

    def percentile(p):
        x = (len(values) - 1) * p
        a, b = math.floor(x), math.ceil(x)
        return values[a] + (values[b] - values[a]) * (x - a)

    return {"unit": unit, "count": len(values), "samples": samples, "mean": statistics.mean(values),
            "median": statistics.median(values), "min": values[0], "max": values[-1],
            "p90": percentile(.90), "p95": percentile(.95), "p99": percentile(.99),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0}
