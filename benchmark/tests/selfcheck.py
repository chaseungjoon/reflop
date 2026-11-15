from __future__ import annotations

import importlib.abc
import importlib.util
import json
import os
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class NoReflop(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=="reflop" or fullname.startswith("reflop."):
            raise AssertionError("offline verification attempted to import real reflop")


sys.meta_path.insert(0,NoReflop())
original_connect = socket.socket.connect
def no_connect(*args,**kwargs):
    raise AssertionError("offline verification attempted a network connection")
socket.socket.connect = no_connect

from benchmark.bench.catalog import MODELS, SUITES, Case, catalog  # noqa: E402
from benchmark.bench.core import ROOT, distribution, load_config, read_opset, write_json  # noqa: E402
from benchmark.bench.runner import execute_case  # noqa: E402


class Harness(unittest.TestCase):
    def test_catalog_is_complete_unique_and_runnable(self):
        c = load_config()
        cases = catalog(c,SUITES)
        ids = [x.id for x in cases]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertTrue(all(x.timeout>0 for x in cases))
        entries = [x.p if hasattr(x,"p") else x.params for x in cases if x.handler=="operator"]
        self.assertEqual({x["name"] for x in entries},set(read_opset()))
        for suite in SUITES:
            path = ROOT/"scripts"/(suite+".sh")
            self.assertTrue(path.exists(),suite)
            self.assertTrue(os.access(path,os.X_OK),suite)
        print(f"Offline catalog: {len(cases)} cases, {len(SUITES)} suites, {len(entries)} registry entries")

    def test_shell_syntax(self):
        for path in sorted(ROOT.rglob("*.sh")):
            if ".venv" not in path.parts:
                self.assertEqual(subprocess.run(["bash","-n",str(path)],capture_output=True).returncode,0,str(path))

    def test_python_syntax(self):
        for path in ROOT.rglob("*.py"):
            if not any(x in path.relative_to(ROOT).parts for x in (".venv","cache","results")):
                compile(path.read_text(),str(path),"exec")

    def test_statistics_and_strict_json(self):
        stats = distribution([1.,2.,3.,4.])
        self.assertEqual(stats["median"],2.5)
        self.assertAlmostEqual(stats["p95"],3.85)
        with self.assertRaises(ValueError):
            distribution([float("nan")])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"nested/value.json"
            write_json(path,{"x":[1,2]})
            self.assertEqual(json.loads(path.read_text()),{"x":[1,2]})
            self.assertFalse(list(path.parent.glob(".writing-*")))

    def test_result_classification_and_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i,(what,want) in enumerate((("pass","pass"),("skipped","skipped"),("fail","fail"),
                                           ("crash","error"),("malformed","error"),
                                           ("exit_after_pass","error"),("timeout","timeout"))):
                case = Case(f"test/{i}","test","unused",{"what":what},1 if what=="timeout" else 10)
                value = execute_case(case,root,load_config(),"benchmark.tests.fake_worker")
                self.assertEqual(value["status"],want,value)
                self.assertTrue((root/value["artifact_directory"]/"stderr.log").exists())
            rows = [json.loads(line) for line in (root/"results.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows),7)
            self.assertTrue(all("finished_at" in row for row in rows))
            if importlib.util.find_spec("psutil"):
                import psutil
                child_pid = int((root/"cases/test/6/child.pid").read_text())
                try:
                    child = psutil.Process(child_pid)
                    self.assertEqual(child.status(),psutil.STATUS_ZOMBIE,"timeout left a grandchild running")
                except psutil.NoSuchProcess:
                    pass

    def test_bad_config_is_rejected(self):
        old = os.environ.get("BENCH_CONFIG")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                config = load_config()
                config["measured_steps"] = 0
                path = Path(tmp)/"config.json"
                write_json(path,config)
                os.environ["BENCH_CONFIG"] = str(path)
                with self.assertRaises(ValueError):
                    load_config()
        finally:
            if old is None:
                os.environ.pop("BENCH_CONFIG",None)
            else:
                os.environ["BENCH_CONFIG"] = old


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("safetensors"),
                     "native reference validation needs bootstrap dependencies")
class NativeReferences(unittest.TestCase):
    def test_native_workloads_and_gradients(self):
        import torch

        from benchmark.bench.models import build
        from benchmark.bench.operations import GRAD_NAMES, NAMES, grad_recipe, make
        torch.set_num_threads(2)
        for name in MODELS:
            model,x,y,loss_fn = build(name,2)
            loss = loss_fn(model(x),y)
            loss.backward()
            self.assertTrue(torch.isfinite(loss),name)
            self.assertTrue(any(p.grad is not None for p in model.parameters()),name)
        for name in NAMES:
            fn,args = make(name)
            fn(*args)
        for name in GRAD_NAMES:
            fn,args = grad_recipe(name,1729)
            value = fn(*args)
            values = value if isinstance(value,(list,tuple)) else [value]
            sum(v.sum() for v in values).backward()
        self.assertFalse(any(n=="reflop" or n.startswith("reflop.") for n in sys.modules))

    def test_checkpoint_roundtrip_optimizer_and_rng(self):
        import torch

        from benchmark.bench.models import build, optimizer
        from benchmark.bench.state_probes import load_training, save_training
        model,x,y,loss_fn = build("mlp",2)
        opt = optimizer(model,"adamw")
        sched = torch.optim.lr_scheduler.StepLR(opt,2)
        loss_fn(model(x),y).backward()
        opt.step()
        sched.step()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            save_training(path,model,opt,sched,1,torch.arange(2),1729)
            loaded = load_training(path)
            self.assertEqual(loaded["step"],1)
            self.assertTrue(all(isinstance(k,int) for k in loaded["optimizer"]["state"]))
            for k,v in loaded["model"].items():
                torch.testing.assert_close(v,model.state_dict()[k],rtol=0,atol=0)
            torch.set_rng_state(loaded["torch_cpu_rng"])
            expected = torch.rand(4)
            torch.set_rng_state(loaded["torch_cpu_rng"])
            torch.testing.assert_close(expected,torch.rand(4),rtol=0,atol=0)
            raw = bytearray((path/"state.safetensors").read_bytes())
            raw[-1]^=1
            (path/"state.safetensors").write_bytes(raw)
            with self.assertRaises(ValueError):
                load_training(path)

    def test_native_operator_recipe_discovery_and_replay(self):
        import torch

        from benchmark.bench import fixtures
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        original_directory = fixtures.directory
        self.addCleanup(setattr,fixtures,"directory",original_directory)
        fixtures.directory = lambda: Path(temporary.name)
        from benchmark.bench.fixtures import load, prepare, resolve
        from benchmark.bench.worker import compare, host
        index = prepare()
        ops = read_opset()
        self.assertEqual(set(index["recipes"])|set(index["uncovered"]),set(ops))
        self.assertGreater(len(index["recipes"]),len(ops)//2)
        c = load_config()
        for name in index["recipes"]:
            if ops[name][1]&32:
                continue  # native uninitialized memory has no comparison value
            args,kw,_ = load(name)
            torch.manual_seed(1729)
            a = host(resolve(name)(*args,**kw))
            args,kw,_ = load(name)
            torch.manual_seed(1729)
            b = host(resolve(name)(*args,**kw))
            with self.subTest(operator=name):
                compare(a,b,c)
        print(f"Native-only validated recipes: {len(index['recipes'])}; explicit gaps: {len(index['uncovered'])}")

    def test_native_training_measurement_path(self):
        from benchmark.bench.worker import Context, train
        c = load_config()
        c.update(warmup_steps=3,measured_steps=2,repeats=2)
        params = {"model":"mlp","mode":"eager","target":"native_cpu","check":False,
                  "batch":2,"residency":"upload","sync_every":0,"optimizer":"momentum"}
        with tempfile.TemporaryDirectory() as tmp:
            ctx = Context({"case":{"params":params},"config":c,"run_dir":tmp},Path(tmp))
            result = train(ctx)
            self.assertEqual(result["repeat_step_average"]["count"],2)
            self.assertEqual(result["step_time"]["count"],4)
            self.assertEqual(len(result["samples_per_second"]),2)
            self.assertIsNone(ctx.rf)


from .test_compare_cuda import CudaComparison as CudaComparison  # noqa: E402

if __name__=="__main__":
    unittest.main(verbosity=2)
