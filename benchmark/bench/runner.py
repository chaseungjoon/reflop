from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import uuid
from collections import Counter

from .catalog import SUITES, catalog
from .core import REPO, ROOT, environment, load_config, utc, write_json


def stop_process(p):
    try:
        os.killpg(p.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        p.wait(timeout=40)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()


def execute_case(case, run_dir, c, worker_module="benchmark.bench.worker"):
    path = run_dir / "cases" / case.id
    path.mkdir(parents=True)
    write_json(path / "request.json", {"case": case.json(), "config": c, "run_dir": str(run_dir)})
    result = {"schema_version": 1, "case": case.json(), "started_at": utc()}
    start = time.monotonic()
    p = None
    interrupted = False
    try:
        with (path / "stdout.log").open("w") as stdout, (path / "stderr.log").open("w") as stderr:
            env = dict(os.environ)
            env.pop("REFLOP_LOCAL", None)  # never silently use an in-process worker
            env["PYTHONUNBUFFERED"] = "1"
            env["OMP_NUM_THREADS"] = str(c["cpu_threads"])
            env["MKL_NUM_THREADS"] = str(c["cpu_threads"])
            p = subprocess.Popen([sys.executable, "-u", "-m", worker_module, str(path / "request.json")],
                                 cwd=REPO, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
            try:
                rc = p.wait(timeout=case.timeout)
                outcome = path / "outcome.json"
                if outcome.exists():
                    result.update(json.loads(outcome.read_text()))
                    if rc and result.get("status") == "pass":
                        result.update(status="error", failure_category="process_exit_after_result")
                else:
                    result.update(status="error", failure_category="worker_crash", message=f"no result; exit {rc}")
                result["exit_code"] = rc
            except subprocess.TimeoutExpired:
                stop_process(p)
                result.update(status="timeout", failure_category="timeout", message=f"exceeded {case.timeout}s")
    except (KeyboardInterrupt, SystemExit):
        if p:
            stop_process(p)
        result.update(status="interrupted", failure_category="interrupted")
        interrupted = True
    except Exception as e:
        if p:
            stop_process(p)
        result.update(status="error", failure_category="harness", message=f"{type(e).__name__}: {e}")
    result.update(finished_at=utc(), wall_seconds=time.monotonic() - start,
                  artifact_directory=str(path.relative_to(run_dir)))
    write_json(path / "result.json", result)
    with (run_dir / "results.jsonl").open("a") as stream:
        stream.write(json.dumps(result, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    if interrupted:
        raise KeyboardInterrupt
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1 or args[0] not in (*SUITES, "all", "smoke", "catalog"):
        raise SystemExit("Use an executable in benchmark/scripts/; no arguments are needed.")
    selected = args[0]
    c = load_config()
    suites = list(SUITES) if selected in ("all", "catalog") else [selected]
    if selected == "smoke":
        suites = ["inventory", "tensors", "correctness", "transfer", "checkpoint"]
    cases = catalog(c, suites)
    if selected == "smoke":
        cases = [x for x in cases if x.suite == "inventory" or
                 x.id in ("tensors/broadcast", "tensors/views", "checkpoint/weights", "transfer/roundtrip/1024") or
                 (x.suite == "correctness" and x.params["model"] == "mlp" and x.params["target"] == "remote_cuda")]
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + selected + "-" + uuid.uuid4().hex[:8]
    directory = ROOT / "results" / run_id
    directory.mkdir(parents=True)
    meta = {"schema_version": 1, "run_id": run_id, "selection": selected, "started_at": utc(),
            "state": "running", "planned_cases": len(cases), "config": c,
            "claims": {"exhaustive_input_space": False, "cloud_product_implemented": False,
                       "native_cpu_is_gpu_baseline": False, "network_shaping": "none"}}
    write_json(directory / "run.json", meta)
    write_json(directory / "environment.json", environment())
    write_json(directory / "catalog.json", [x.json() for x in cases])
    print(f"Results: {directory}", flush=True)
    counts = Counter()
    exit_code = 0
    try:
        if selected != "catalog":
            for i, case in enumerate(cases, 1):
                print(f"[{i}/{len(cases)}] {case.id}", flush=True)
                result = execute_case(case, directory, c)
                counts[result["status"]] += 1
                print(f"  {result['status']} ({result['wall_seconds']:.2f}s) {result.get('message', '')[:200]}", flush=True)
                if result["status"] in ("fail", "error", "timeout", "fixture_error"):
                    exit_code = 1
        meta["state"] = "catalog_only" if selected == "catalog" else "completed"
    except KeyboardInterrupt:
        meta["state"] = "interrupted"
        exit_code = 130
    finally:
        meta.update(finished_at=utc(), counts=dict(counts), exit_code=exit_code)
        write_json(directory / "run.json", meta)
    print(f"{meta['state']}: {dict(counts)}\nResults: {directory}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
