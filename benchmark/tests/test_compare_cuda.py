from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark.bench.compare_cuda import compare_runs
from benchmark.bench.core import write_json


class CudaComparison(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = dict(seed=1, cpu_threads=4, warmup_steps=5, measured_steps=30, repeats=3)
        self.p = dict(model="mlp", batch=32, residency="resident", sync_every=0, optimizer="momentum")
        self.cap = dict(
            devices=[dict(name="Test GPU", total_memory_bytes="12000", device={"kind": "DEVICE_CUDA"})],
            runtime_info={"torch_version": "2.test+cu132", "cuda_version": "13.2"},
            numerics={"tf32_cudnn": True},
        )
        self.gpu = dict(
            name="Test GPU",
            total_memory=12000,
            torch_version="2.test+cu132",
            torch_cuda="13.2",
            numerics=dict(tf32_cudnn=True, tf32_matmul=False, deterministic=False, cudnn_benchmark=False),
        )

    def make_run(self, selection, status="pass"):
        path = self.root / selection
        path.mkdir(exist_ok=True)
        remote = selection == "throughput"
        p = self.p | dict(mode="capture" if remote else "eager", target="remote_cuda" if remote else "native_cuda")
        case = dict(id=selection + "/mlp/b32", params=p)
        details = dict(repeat_step_average={"samples": [0.004, 0.008, 0.012] if remote else [0.002, 0.004, 0.006]})
        if remote:
            details["connection_history"] = [{"capabilities_file": "capabilities.json"}]
            write_json(path / "cases/example/capabilities.json", self.cap)
        else:
            details["native_gpu"] = self.gpu
        row = dict(case=case, status=status, details=details, artifact_directory="cases/example")
        import json

        (path / "results.jsonl").write_text(json.dumps(row) + "\n")
        write_json(
            path / "run.json",
            dict(run_id=selection, state="completed", selection=selection, planned_cases=1, config=self.config),
        )
        write_json(
            path / "environment.json",
            dict(source_hashes={"bench/models.py": "same-model", "bench/worker.py": "same-worker"}),
        )
        return path

    def test_ratio_uses_total_window_time(self):
        result = compare_runs(self.make_run("throughput"), self.make_run("native_cuda"))["records"][0]
        self.assertAlmostEqual(result["retained_percent"], 50)
        self.assertAlmostEqual(result["remote_samples_s"], 4000)
        self.assertAlmostEqual(result["native_ms_step"], 4)

    def test_rejects_mismatched_configuration_gpu_and_numerics(self):
        remote = self.make_run("throughput")
        native = self.make_run("native_cuda")
        self.config["measured_steps"] = 2
        self.make_run("native_cuda")
        with self.assertRaisesRegex(ValueError, "Configuration mismatch"):
            compare_runs(remote, native)
        self.config["measured_steps"] = 30
        self.gpu["name"] = "Other GPU"
        self.make_run("native_cuda")
        with self.assertRaisesRegex(ValueError, "GPU name"):
            compare_runs(remote, native)
        self.gpu["name"] = "Test GPU"
        self.gpu["numerics"]["tf32_cudnn"] = False
        self.make_run("native_cuda")
        with self.assertRaisesRegex(ValueError, "Numerical policy"):
            compare_runs(remote, native)

    def test_rejects_missing_baseline_and_wrong_source(self):
        remote, native = self.make_run("throughput"), self.make_run("native_cuda", "skipped")
        with self.assertRaisesRegex(ValueError, "No successful matched"):
            compare_runs(remote, native)
        self.make_run("native_cuda")
        write_json(native / "environment.json", dict(source_hashes={"bench/models.py": "other"}))
        with self.assertRaisesRegex(ValueError, "implementation mismatch"):
            compare_runs(remote, native)

    def test_invalid_batch_is_excluded_from_all_performance_targets(self):
        from benchmark.bench.catalog import catalog
        from benchmark.bench.core import load_config

        rows = catalog(load_config(), ["throughput", "native_cuda", "native_cpu"])
        self.assertFalse(any(c.params["model"] == "resnet18" and c.params["batch"] == 1 for c in rows))
        self.assertTrue(any(c.params["model"] == "resnet18" and c.params["batch"] == 8 for c in rows))

    def test_registry_falls_back_to_official_checkout(self):
        from importlib.metadata import PackageNotFoundError

        from benchmark.bench.core import REPO, opset_source

        with (
            patch("benchmark.bench.core.importlib.metadata.distribution", side_effect=PackageNotFoundError),
            patch.dict("os.environ", {}, clear=True),
        ):
            self.assertEqual(opset_source(), REPO / "client/python/reflop/_opset_generated.py")


if __name__ == "__main__":
    unittest.main()
