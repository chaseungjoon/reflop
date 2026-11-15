"""Compare matched remote and native CUDA runs without contacting either device."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
import time
import uuid
from pathlib import Path

from .core import ROOT, write_json

CONFIG_KEYS = ("seed", "cpu_threads", "warmup_steps", "measured_steps", "repeats")
PARAM_KEYS = ("model", "batch", "residency", "sync_every", "optimizer")


def load_run(path, selection):
    path = Path(path).resolve()
    meta = json.loads((path / "run.json").read_text())
    if meta["selection"] != selection or meta["state"] != "completed":
        raise ValueError(f"{path}: expected a completed {selection} run")
    raw = (path / "results.jsonl").read_bytes()
    rows = [json.loads(line) for line in raw.splitlines()]
    if len(rows) != meta["planned_cases"] or len({r["case"]["id"] for r in rows}) != len(rows):
        raise ValueError(f"{path}: missing or duplicate case results")
    return path, meta, rows, json.loads((path / "environment.json").read_text()), hashlib.sha256(raw).hexdigest()


def seconds(row, config):
    values = row["details"]["repeat_step_average"]["samples"]
    if len(values) != config["repeats"] or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError(f"{row['case']['id']}: invalid synchronized repeat timings")
    return statistics.mean(values)


def compare_runs(remote_path, native_path):
    rp, rm, remote, re, rh = load_run(remote_path, "throughput")
    _, nm, native, ne, nh = load_run(native_path, "native_cuda")
    for key in CONFIG_KEYS:
        if rm["config"][key] != nm["config"][key]:
            raise ValueError(f"Configuration mismatch: {key}")
    for source in ("bench/models.py", "bench/worker.py"):
        a, b = re["source_hashes"].get(source), ne["source_hashes"].get(source)
        if not a or a != b:
            raise ValueError(f"Workload implementation mismatch: {source}; collect both runs from the same source")
    native_index = {}
    for row in native:
        p = row["case"]["params"]
        if p.get("target") != "native_cuda" or p.get("mode") != "eager":
            raise ValueError("Native baseline must use eager CUDA, never CPU")
        key = tuple(p[k] for k in PARAM_KEYS)
        if key in native_index:
            raise ValueError(f"Duplicate native comparison key: {key}")
        native_index[key] = row
    compared = []
    for row in remote:
        if "/sync/" in row["case"]["id"]:
            continue
        p = row["case"]["params"]
        if p.get("target") != "remote_cuda":
            raise ValueError("Remote case must target CUDA")
        base = native_index.get(tuple(p[k] for k in PARAM_KEYS))
        entry = {k: p[k] for k in (*PARAM_KEYS, "mode")}
        entry.update(
            case_id=row["case"]["id"],
            remote_status=row["status"],
            native_status=base["status"] if base else "missing",
            status="unavailable",
            remote_samples_s=None,
            native_samples_s=None,
            retained_percent=None,
            remote_ms_step=None,
            native_ms_step=None,
        )
        if row["status"] == "pass" and base and base["status"] == "pass":
            cap = json.loads(
                (
                    rp / row["artifact_directory"] / row["details"]["connection_history"][0]["capabilities_file"]
                ).read_text()
            )
            gpu = next(d for d in cap["devices"] if d["device"]["kind"] == "DEVICE_CUDA")
            ng = base["details"].get("native_gpu", {})
            if (gpu["name"], int(gpu["total_memory_bytes"])) != (ng.get("name"), ng.get("total_memory")):
                raise ValueError("GPU name or memory mismatch")
            runtime = cap["runtime_info"]
            if runtime["torch_version"] != ng.get("torch_version") or runtime["cuda_version"] != ng.get("torch_cuda"):
                raise ValueError("Owner/native PyTorch or CUDA version mismatch")
            # Protobuf omits false values. All recorded policies must agree with native.
            for bit in ("tf32_cudnn", "tf32_matmul", "deterministic", "cudnn_benchmark"):
                if cap.get("numerics", {}).get(bit, False) != ng.get("numerics", {}).get(bit):
                    raise ValueError(f"Numerical policy mismatch or missing metadata: {bit}")
            rt, nt = seconds(row, rm["config"]), seconds(base, nm["config"])
            entry.update(
                status="compared",
                remote_samples_s=p["batch"] / rt,
                native_samples_s=p["batch"] / nt,
                retained_percent=100 * nt / rt,
                remote_ms_step=1000 * rt,
                native_ms_step=1000 * nt,
            )
        compared.append(entry)
    if not any(r["status"] == "compared" for r in compared):
        raise ValueError("No successful matched CUDA cases; inspect native and remote outcomes")
    return {
        "schema_version": 1,
        "remote_run": rm["run_id"],
        "native_run": nm["run_id"],
        "remote_results_sha256": rh,
        "native_results_sha256": nh,
        "same_physical_gpu_confirmed_by_operator": True,
        "records": compared,
    }


def render(data, output):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "cache/matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = data["records"]
    output.mkdir(parents=True)
    write_json(output / "comparison.json", data)
    with (output / "comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    models = list(dict.fromkeys(r["model"] for r in rows))
    batches = sorted({r["batch"] for r in rows})
    modes = [("eager", "resident"), ("capture", "resident"), ("eager", "upload"), ("capture", "upload")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), layout="constrained")
    for ax, (mode, residency) in zip(axes.flat, modes):
        points = {
            (r["model"], r["batch"]): r["retained_percent"]
            for r in rows
            if r["mode"] == mode and r["residency"] == residency
        }
        values = [[points.get((m, b)) for b in batches] for m in models]
        import numpy as np

        matrix = np.array([[np.nan if v is None else v for v in line] for line in values])
        im = ax.imshow(
            matrix,
            cmap="Blues",
            vmin=0,
            vmax=max(100, np.nanmax(matrix)) if np.isfinite(matrix).any() else 100,
            aspect="auto",
        )
        ax.set(
            xticks=range(len(batches)),
            xticklabels=batches,
            yticks=range(len(models)),
            yticklabels=models,
            xlabel="Batch size",
            title=f"Reflop {mode} / {residency}",
        )
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(
                j,
                i,
                "N/A" if np.isnan(value) else f"{value:.1f}%",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > im.norm.vmax * 0.65 else "black",
            )
        fig.colorbar(im, ax=ax, label="% of native eager CUDA throughput", shrink=0.8)
    fig.suptitle(
        "Reflop throughput relative to native PyTorch on the same GPU\n100% = equal throughput; N/A = missing or nonpassing case"
    )
    for ext in ("png", "svg"):
        fig.savefig(output / f"native-cuda-comparison.{ext}", dpi=180)
    plt.close(fig)
    lines = [
        "# Native CUDA comparison",
        "",
        f"Remote run: `{data['remote_run']}`. Native run: `{data['native_run']}`.",
        "",
        "Retained throughput = remote samples/s / native samples/s × 100%. The operator confirmed the same physical GPU.",
        "",
        "The baseline is native eager PyTorch. Results above 100% are retained as measured and need investigation, including run-to-run noise.",
        "",
        "![Native GPU comparison](native-cuda-comparison.png)",
        "",
        "| Workload | Batch | Mode | Inputs | Remote ms/step | Native ms/step | Retained throughput |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for r in rows:
        result = (
            f"{r['remote_ms_step']:.3f} | {r['native_ms_step']:.3f} | {r['retained_percent']:.1f}%"
            if r["status"] == "compared"
            else f"— | — | N/A ({r['remote_status']}/{r['native_status']})"
        )
        lines.append(f"| {r['model']} | {r['batch']} | {r['mode']} | {r['residency']} | {result} |")
    lines += [
        "",
        "A timing pass checks finite loss, not full numerical equivalence. Read the separate correctness results before treating a workload as validated.",
        "See comparison.csv for absolute sample rates and all matched/unavailable cases. Repeat samples remain in the source runs. These ratios have no confidence intervals.",
        "",
    ]
    (output / "README.md").write_text("\n".join(lines))


def main():
    if os.environ.get("BENCH_SAME_GPU") != "1":
        raise SystemExit("Set BENCH_SAME_GPU=1 only after confirming both runs used the same physical GPU.")
    try:
        remote, native = os.environ["BENCH_REMOTE_RUN"], os.environ["BENCH_NATIVE_RUN"]
    except KeyError as e:
        raise SystemExit(f"Set {e.args[0]} to a completed run directory") from e
    data = compare_runs(remote, native)
    output = (
        ROOT / "results" / (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-cuda-comparison-" + uuid.uuid4().hex[:8])
    )
    render(data, output)
    print(f"CUDA comparison: {output}")


if __name__ == "__main__":
    main()
