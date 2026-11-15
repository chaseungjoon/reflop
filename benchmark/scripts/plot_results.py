#!/usr/bin/env python3
"""Render saved benchmark evidence; BENCH_PLOT_RUN optionally exports a new run.

No Reflop imports, connections, or benchmark execution. Default: bundled snapshot.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "cache/matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
SNAPSHOT = REPORT / "data/throughput.json"
MODELS = ["mlp", "cnn", "resnet18", "unet", "lstm", "gru", "transformer", "gpt", "lora", "autoencoder"]
LABELS = ["MLP", "CNN", "ResNet-18*", "U-Net", "LSTM", "GRU", "Attention", "GPT", "LoRA-style", "Autoencoder"]
STYLES = [
    ("eager", "resident", "#3176ae", "-"),
    ("capture", "resident", "#159580", "-"),
    ("eager", "upload", "#d28d27", "--"),
    ("capture", "upload", "#b66086", "--"),
]


def export_run(path):
    run = json.loads((path / "run.json").read_text())
    if run["state"] != "completed" or run["selection"] != "throughput":
        raise ValueError("Select a completed throughput run")
    raw = (path / "results.jsonl").read_bytes()
    records = [json.loads(line) for line in raw.splitlines()]
    if len(records) != run["planned_cases"]:
        raise ValueError("Incomplete results")
    rows = []
    for r in records:
        d = r.get("details", {})
        rows.append(
            dict(
                case_id=r["case"]["id"],
                params=r["case"]["params"],
                status=r["status"],
                message=r.get("message"),
                seconds_per_step=d.get("repeat_step_average", {}).get("samples", []),
                stats_deltas=d.get("stats_deltas", []),
            )
        )
    env = json.loads((path / "environment.json").read_text())
    cap_path = next(path.glob("cases/**/capabilities-*.json"))
    cap = json.loads(cap_path.read_text())
    return dict(
        schema_version=1,
        run_id=run["run_id"],
        counts=run["counts"],
        config=run["config"],
        source_sha256=hashlib.sha256(raw).hexdigest(),
        environment={k: env[k] for k in ("platform", "python", "packages", "benchmark_sha256")},
        devices=cap["devices"],
        runtime_info=cap["runtime_info"],
        numerics=cap.get("numerics", {}),
        records=rows,
    )


def throughput(row):
    values = row["seconds_per_step"]
    assert row["status"] == "pass" and values and all(np.isfinite(v) and v > 0 for v in values)
    return row["params"]["batch"] / np.mean(values)


def main():
    REPORT.joinpath("figures").mkdir(parents=True, exist_ok=True)
    source = os.environ.get("BENCH_PLOT_RUN")
    if source:
        data = export_run(Path(source).resolve())
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(json.dumps(data, indent=2) + "\n")
    else:
        data = json.loads(SNAPSHOT.read_text())
    rows = data["records"]
    index = {
        (r["params"]["model"], r["params"]["batch"], r["params"]["mode"], r["params"]["residency"]): r
        for r in rows
        if "/sync/" not in r["case_id"]
    }
    batches = data["config"]["batch_sizes"]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "reflop-benchmark",
        }
    )

    def save(fig, name, note):
        fig.text(
            0.02,
            0.015,
            note
            + f"\nReflop {data['runtime_info']['reflopd_version']} | {data['devices'][0]['name']} | Run {data['run_id']} | No native CUDA baseline",
            fontsize=9,
            color="#475569",
            va="bottom",
        )
        for ext in ("png", "svg"):
            fig.savefig(
                REPORT / f"figures/{name}.{ext}",
                dpi=180,
                facecolor="white",
                metadata={"Date": None} if ext == "svg" else {},
            )
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 6.8))
    x = np.arange(len(MODELS))
    width = 0.2
    for i, (mode, residency, color, _) in enumerate(STYLES):
        selected = [index[m, 128, mode, residency] for m in MODELS]
        rates = np.array([throughput(r) for r in selected])
        ranges = np.array([[128 / max(r["seconds_per_step"]), 128 / min(r["seconds_per_step"])] for r in selected])
        ax.bar(
            x + (i - 1.5) * width,
            rates,
            width,
            color=color,
            label=f"{mode.title()} / {residency}",
            yerr=np.array([rates - ranges[:, 0], ranges[:, 1] - rates]),
            capsize=2,
            error_kw={"lw": 0.8},
        )
    ax.set(
        yscale="log",
        ylabel="Training samples / second (log scale)",
        xticks=x,
        xticklabels=LABELS,
        title="Keeping inputs on the GPU dominates throughput at batch 128",
    )
    ax.grid(axis="y", alpha=0.15)
    ax.set_axisbelow(True)
    ax.legend(ncol=2, loc="upper left")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.18)
    save(
        fig,
        "throughput-b128",
        "Rate = batch / mean synchronized seconds per step; whiskers span 3 repeat rates. *Correctness unresolved.",
    )

    fig, axes = plt.subplots(2, 5, figsize=(16, 7.5), sharex=True)
    for ax, model, label in zip(axes.flat, MODELS, LABELS):
        for mode, residency, color, linestyle in STYLES:
            selected = [index[model, b, mode, residency] for b in batches]
            rates = [throughput(r) if r["status"] == "pass" else np.nan for r in selected]
            ax.plot(batches, rates, marker="o", ms=4, color=color, ls=linestyle, label=f"{mode.title()} / {residency}")
        ax.set(xscale="log", yscale="log", title=label, xticks=batches)
        ax.set_xticklabels(batches)
        ax.grid(alpha=0.15)
        if model == "resnet18":
            ax.text(0.03, 0.05, "Batch 1 invalid (BatchNorm)", transform=ax.transAxes, fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("Samples / second")
    for ax in axes[1, :]:
        ax.set_xlabel("Batch size")
    fig.suptitle("Batch scaling varies by workload and input residency", y=0.98, fontsize=16)
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=4)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.82, bottom=0.17, hspace=0.38, wspace=0.32)
    save(
        fig,
        "batch-scaling",
        "Both axes logarithmic; each panel has its own y range. Points use all 3 windows; failed cases are gaps. *Correctness unresolved.",
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 7))
    from matplotlib.colors import TwoSlopeNorm

    for ax, residency in zip(axes, ["resident", "upload"]):
        values = np.full((len(MODELS), len(batches)), np.nan)
        for i, m in enumerate(MODELS):
            for j, b in enumerate(batches):
                a, c = index[m, b, "eager", residency], index[m, b, "capture", residency]
                if a["status"] == c["status"] == "pass":
                    values[i, j] = throughput(c) / throughput(a)
        im = ax.imshow(np.log2(values), cmap="RdYlGn", norm=TwoSlopeNorm(vmin=-2, vcenter=0, vmax=3))
        ax.set(
            xticks=range(len(batches)),
            xticklabels=batches,
            yticks=range(len(MODELS)),
            yticklabels=LABELS,
            xlabel="Batch size",
            title=f"{residency.title()} inputs",
        )
        for (i, j), v in np.ndenumerate(values):
            ax.text(j, i, "N/A" if np.isnan(v) else f"{v:.2f}×", ha="center", va="center", fontsize=10)
    bar = fig.colorbar(im, cax=fig.add_axes([0.91, 0.22, 0.018, 0.62]), ticks=[-2, -1, 0, 1, 2, 3])
    bar.ax.set_yticklabels(["0.25×", "0.5×", "1×", "2×", "4×", "8×"])
    fig.suptitle("Capture usually helps resident inputs; upload results are mixed", fontsize=15, y=0.98)
    fig.subplots_adjust(left=0.13, right=0.87, bottom=0.18, top=0.91, wspace=0.65)
    save(
        fig,
        "capture-speedup",
        "Ratio = capture throughput / eager throughput. Green > 1; red < 1. N/A = failed BatchNorm cases. *Correctness unresolved.",
    )

    fig, ax = plt.subplots(figsize=(9, 5.6))
    sync_rows = {(r["params"]["mode"], r["params"]["sync_every"]): r for r in rows if "/sync/" in r["case_id"]}
    for i, mode in enumerate(["eager", "capture"]):
        selected = [sync_rows[mode, s] for s in [1, 5, 0]]
        ax.bar(
            np.arange(3) + (i - 0.5) * 0.3,
            [throughput(r) for r in selected],
            0.3,
            color=STYLES[i][2],
            label=mode.title(),
        )
        for j, r in enumerate(selected):
            ax.scatter(
                np.full(3, j + (i - 0.5) * 0.3),
                [32 / v for v in r["seconds_per_step"]],
                color="#243449",
                s=15,
                zorder=3,
            )
    ax.set(
        xticks=range(3),
        xticklabels=["Every step", "Every 5 steps", "End of window"],
        ylabel="Training samples / second",
        title="Frequent synchronization limits MLP throughput",
        xlabel="Synchronization policy (batch 32, uploaded inputs)",
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.15)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.26)
    save(
        fig,
        "synchronization",
        "Bars use total synchronized window time; dots show each of 3 repeats. Nonmonotonic results indicate timing variability.",
    )

    with (REPORT / "data/throughput.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "case_id",
                "model",
                "batch",
                "mode",
                "residency",
                "sync_every",
                "status",
                "samples_per_second",
                "repeat_seconds_per_step",
            ]
        )
        for r in rows:
            p = r["params"]
            writer.writerow(
                [
                    r["case_id"],
                    *[p[k] for k in ("model", "batch", "mode", "residency", "sync_every")],
                    r["status"],
                    throughput(r) if r["status"] == "pass" else "",
                    json.dumps(r["seconds_per_step"]),
                ]
            )
    print(f"Rendered 4 figures (PNG + SVG) and CSV from {data['run_id']}; {len(rows)} cases.")


if __name__ == "__main__":
    main()
