from __future__ import annotations

import copy
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

from .core import CoverageGap, MissingPrerequisite, distribution, utc, write_json


class Context:
    def __init__(self, request, path):
        self.case = request["case"]
        self.p = self.case["params"]
        self.c = request["config"]
        self.path = path
        self.run_dir = Path(request["run_dir"])
        self.phase = "setup"
        self.devices = {}
        self.rf = None
        self.details = {}
        self.execution_metrics = []
        self.connection_count = 0
        if self.c.get("client_home"):
            os.environ["REFLOP_CLIENT_HOME"] = str(Path(self.c["client_home"]).expanduser())
        if self.c.get("server"):
            os.environ["REFLOP_SERVER"] = self.c["server"]
        if self.p.get("network"):
            info = self.c["network_targets"][self.p["network"]]
            if not info["server"]:
                raise MissingPrerequisite(f"configure network_targets.{self.p['network']}.server; no distance is inferred")
            os.environ["REFLOP_SERVER"] = info["server"]
            if info.get("client_home"):
                os.environ["REFLOP_CLIENT_HOME"] = str(Path(info["client_home"]).expanduser())
            self.details["network"] = {"label":self.p["network"],"description":info["description"],"server":info["server"]}

    def event(self, kind, **data):
        event = {"utc":utc(),"monotonic":time.monotonic(),"kind":kind,**data}
        with (self.path/"events.jsonl").open("a") as f:
            f.write(json.dumps(event,allow_nan=False)+"\n")

    def remote(self):
        if self.rf is None:
            import reflop
            self.rf = reflop
        return self.rf

    def device(self, target=None, index=0, server=None):
        import torch
        target = target or "remote_"+self.c["remote_kind"]
        if target == "native_cpu":
            return torch.device("cpu")
        if target == "native_cuda":
            if not torch.cuda.is_available():
                raise MissingPrerequisite("native CUDA baseline must run on the same owner GPU used for the remote run")
            info = torch.cuda.get_device_properties(0)
            self.details["native_gpu"] = {"name":info.name,"total_memory":info.total_memory,
                                          "label":self.c["native_cuda_label"],"torch_cuda":torch.version.cuda,
                                          "torch_version":torch.__version__,
                                          "numerics":{"tf32_cudnn":torch.backends.cudnn.allow_tf32,
                                                      "tf32_matmul":torch.backends.cuda.matmul.allow_tf32,
                                                      "deterministic":torch.are_deterministic_algorithms_enabled(),
                                                      "cudnn_benchmark":torch.backends.cudnn.benchmark}}
            return torch.device("cuda:0")
        if index not in self.devices:
            from google.protobuf.json_format import MessageToDict
            rf = self.remote()
            self.phase = "connect"
            device = rf.device(index, server=server or os.environ.get("REFLOP_SERVER"),
                               kind=target.removeprefix("remote_"), allow_local_fallback=False)
            from reflop import session
            s = session.by_index(index)
            deliver = s.conn._deliver
            def observed(env):
                if env.WhichOneof("body")=="execute_result":
                    value = env.execute_result
                    self.execution_metrics.append({"device_index":index,"phase":self.phase,
                        "duration_us":int(value.duration_us),"flops":int(value.flops),
                        "peak_memory_bytes":int(value.peak_memory_bytes)})
                return deliver(env)
            s.conn._deliver = observed
            cap_file = f"capabilities-{index}-{self.connection_count:04}.json"
            self.connection_count+=1
            write_json(self.path/cap_file,MessageToDict(s.caps,preserving_proto_field_name=True))
            record = {"kind":int(s.gdevice.kind),"device_index":index,"capabilities_file":cap_file,
                "enabled_ops":len(s.enabled),"session_id":getattr(s.conn,"session_id",b"").hex(),
                "server":getattr(getattr(s.conn,"entry",None),"name",server)}
            self.details.setdefault("remote_devices",{})[str(index)] = record
            self.details.setdefault("connection_history",[]).append(record)
            self.devices[index] = device
            self.phase = "remote_execute"
        return self.devices[index]

    def sync(self, dev):
        import torch
        if dev.type == "reflop":
            self.remote().synchronize()
        elif dev.type == "cuda":
            torch.cuda.synchronize(dev)

    def stats(self):
        return copy.deepcopy(self.rf.stats()) if self.rf else {}

    def close(self):
        try:
            if self.rf:
                from reflop import session
                self.details["unsupported_names"] = sorted({n for s in session._sessions.values() for n in s.unsupported_names})
                self.details["final_stats"] = self.stats()
                self.close_sessions(expected_errors=self.details.get("expected_terminal_error",False))
        finally:
            if self.execution_metrics:
                write_json(self.path/"execute_metrics.json",self.execution_metrics)

    def close_sessions(self,expected_errors=False):
        from reflop import session
        errors = []
        for s in list(session._sessions.values()):
            try:
                s.close()
            except Exception as e:
                errors.append(f"{type(e).__name__}: {e}")
        self.devices.clear()
        if errors:
            self.details.setdefault("session_close_errors",[]).extend(errors)
            if not expected_errors:
                raise RuntimeError("; ".join(errors))


def host(value):
    import torch
    if isinstance(value,torch.Tensor):
        return value.detach().cpu()
    if isinstance(value,(list,tuple)):
        return [host(x) for x in value]
    if isinstance(value,dict):
        return {k:host(v) for k,v in value.items()}
    return value


def compare(got,want,c,low=False):
    import torch
    if isinstance(want,torch.Tensor):
        got,want = host(got),host(want)
        torch.testing.assert_close(got,want,rtol=c["low_precision_rtol"] if low else c["rtol"],
                                   atol=c["low_precision_atol"] if low else c["atol"],equal_nan=True)
        if got.is_floating_point() or got.is_complex():
            finite = torch.isfinite(got)&torch.isfinite(want)
            delta = (got[finite]-want[finite]).abs()
            return {"max_abs_error":delta.max().item() if delta.numel() else 0.,"numel":got.numel()}
        return {"exact":True,"numel":got.numel()}
    if isinstance(want,(list,tuple)):
        assert len(got)==len(want),"output arity mismatch"
        return [compare(x,y,c,low) for x,y in zip(got,want)]
    if isinstance(want,dict):
        assert got.keys()==want.keys(),"output keys mismatch"
        return {k:compare(got[k],want[k],c,low) for k in want}
    if isinstance(want,float):
        import math
        assert math.isclose(got,want,rel_tol=c["rtol"],abs_tol=c["atol"]) or math.isnan(got) and math.isnan(want)
    else:
        assert got==want,(got,want)
    return {"equal":True}


def delta(before,after):
    return {str(k):{n:v-before.get(k,{}).get(n,0) for n,v in counters.items()} for k,counters in after.items()}


def train(ctx):
    import torch

    from .models import build, optimizer
    p,c = ctx.p,ctx.c
    dev = ctx.device(p["target"])
    remote = dev.type=="reflop"
    dtype_name = p.get("dtype","float32")
    dtype = getattr(torch,dtype_name) if not dtype_name.startswith("amp_") else torch.float32
    amp_dtype = torch.bfloat16 if dtype_name=="amp_bf16" else torch.float16
    low = dtype_name in ("float16","bfloat16","amp_bf16","amp_fp16")
    is_amp = dtype_name.startswith("amp_")
    model,x,y,loss_fn = build(p["model"],p["batch"],c["seed"])
    model = model.to(dtype=dtype)
    if x.is_floating_point():
        x = x.to(dtype)
    if y.is_floating_point():
        y = y.to(dtype)
    reference = copy.deepcopy(model) if p["check"] else None
    model = model.to(dev)
    opt = optimizer(model,p["optimizer"],remote=remote or dev.type=="cuda")
    ref_opt = optimizer(reference,p["optimizer"]) if reference is not None else None
    if dtype_name=="amp_fp16":
        scaler = torch.amp.GradScaler(dev.type)
    else:
        scaler = None
    resident = (x.to(dev),y.to(dev)) if p["residency"]=="resident" else None
    def step(a,b):
        a,b = (a,b) if resident else (a.to(dev),b.to(dev))
        def closure():
            opt.zero_grad(set_to_none=p.get("zero_none",True))
            with torch.autocast(dev.type,dtype=amp_dtype,enabled=is_amp):
                loss = loss_fn(model(a),b)
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            return loss
        if p["optimizer"]=="lbfgs":
            return opt.step(closure)
        loss = closure()
        if scaler:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        return loss
    fn = ctx.remote().capture(step) if p["mode"]=="capture" else step
    args = resident or (x,y)
    ctx.phase = "remote_execute" if remote else "native_execute"
    nparams = sum(v.numel() for v in model.parameters())
    ctx.details.update(model=p["model"],parameter_count=nparams,input_shape=list(x.shape),input_dtype=str(x.dtype),
                       comparison_key={k:p[k] for k in ("model","batch","residency","sync_every","optimizer")},
                       numerics={"rtol":c["low_precision_rtol"] if low else c["rtol"],
                                 "atol":c["low_precision_atol"] if low else c["atol"]})
    if p["check"]:
        losses,ref_losses = [],[]
        for i in range(c["correctness_steps"]):
            # Fresh, changing batches are necessary to detect stale captured uploads.
            a = x if not x.is_floating_point() else x+(i%3)*.01
            b = y
            result = fn(*(tuple(v.to(dev) for v in (a,b)) if resident else (a,b)))
            if remote and p["mode"]=="eager":
                ctx.remote().step_boundary()
            losses.append(result.item())
            assert math.isfinite(losses[-1]),"nonfinite training loss"
            ctx.phase = "cpu_reference"
            def ref_closure(a=a,b=b):
                ref_opt.zero_grad(set_to_none=p.get("zero_none",True))
                # CPU reference computes in the requested storage dtype. AMP
                # reference is fp32; tolerance is explicit in the result.
                value = loss_fn(reference(a),b)
                value.backward()
                return value
            if p["optimizer"]=="lbfgs":
                ref_loss = ref_opt.step(ref_closure)
            else:
                ref_loss = ref_closure()
                ref_opt.step()
            ref_losses.append(ref_loss.item())
            assert math.isfinite(ref_losses[-1]),"nonfinite native reference loss"
            ctx.phase = "correctness"
            torch.testing.assert_close(torch.tensor(losses[-1]),torch.tensor(ref_losses[-1]),
                rtol=c["low_precision_rtol"] if low else c["rtol"],atol=c["low_precision_atol"] if low else c["atol"])
            ctx.event("step",step=i,loss=losses[-1],reference_loss=ref_losses[-1])
        ctx.phase = "correctness"
        checks = compare(model.state_dict(),reference.state_dict(),c,low)
        # Compare gradients as well as final weights; finite loss alone is insufficient.
        gradients = compare({k:v.grad for k,v in model.named_parameters() if v.grad is not None},
                            {k:v.grad for k,v in reference.named_parameters() if v.grad is not None},c,low)
        write_json(ctx.path/"correctness.json",{"losses":losses,"reference_losses":ref_losses,
                                              "parameters":checks,"gradients":gradients})
        ctx.details["reference"] = "native CPU, identical initial state and batches"
    else:
        for _ in range(c["warmup_steps"]):
            out = fn(*args)
            if remote and p["mode"]=="eager":
                ctx.remote().step_boundary()
        ctx.sync(dev)
        averages,step_times,counts = [],[],[]
        for repeat in range(c["repeats"]):
            before = ctx.stats()
            ctx.sync(dev)
            start = time.perf_counter()
            for i in range(c["measured_steps"]):
                t0 = time.perf_counter()
                out = fn(*args)
                if remote and p["mode"]=="eager":
                    ctx.remote().step_boundary()
                if p["sync_every"] and (i+1)%p["sync_every"]==0:
                    ctx.sync(dev)
                step_times.append(time.perf_counter()-t0)
            ctx.sync(dev)
            elapsed = time.perf_counter()-start
            averages.append(elapsed/c["measured_steps"])
            counts.append(delta(before,ctx.stats()))
            loss_value = out.item()
            assert math.isfinite(loss_value),"nonfinite training loss"
            ctx.event("repeat",repeat=repeat,elapsed_seconds=elapsed,steps=c["measured_steps"],loss=loss_value)
        ctx.details.update(step_time=distribution(step_times),repeat_step_average=distribution(averages),
            step_timing="synchronized latency" if p["sync_every"]==1 else "submission or periodic synchronization; use repeat_step_average for throughput",
            samples_per_second=[p["batch"]/v for v in averages],stats_deltas=counts,
            timed_region="forward + loss + zero_grad + backward + optimizer + selected upload + final synchronization")
    if p["mode"]=="capture":
        ctx.details["capture_replays"] = fn.replays
        assert fn.replays>0,"capture ran without any replay"
    ctx.sync(dev)
    return ctx.details


def tensor(ctx):
    import torch

    from .operations import make
    name = ctx.p["name"]
    fn,args = make(name,ctx.c["seed"])
    dev = ctx.device()
    if name=="empty":
        out = fn(*[a.to(dev) for a in args]).cpu()
        assert torch.equal(out,torch.zeros_like(out)),"uninitialized remote allocation must be zeroed"
        return {"zero_initialized":True}
    if name in ("random","dropout"):
        ctx.remote().seed(ctx.c["seed"])
        a = host(fn(*[v.to(dev) for v in args]))
        ctx.remote().seed(ctx.c["seed"])
        b = host(fn(*[v.to(dev) for v in args]))
        return {"repeatability":compare(a,b,ctx.c),"cross_backend_rng_equality_required":False}
    ctx.phase = "cpu_reference"
    want = fn(*[a.clone() for a in args])
    ctx.phase = "remote_execute"
    start = time.perf_counter()
    got = fn(*[a.to(dev) for a in args])
    got = host(got)
    elapsed = time.perf_counter()-start
    ctx.phase = "correctness"
    return {"comparison":compare(got,want,ctx.c,name in ("half","bf16","dtype_conversion","float8")),
            "roundtrip_seconds":elapsed}


def autograd(ctx):
    import torch

    from .operations import grad_recipe
    dev = ctx.device()
    results = []
    for remote in (False,True):
        fn,args = grad_recipe(ctx.p["name"],ctx.c["seed"])
        leaves = [a for a in args if a.requires_grad]
        if remote:
            args = [a.detach().to(dev).requires_grad_(a.requires_grad) for a in args]
            leaves = [a for a in args if a.requires_grad]
        ctx.phase = "remote_execute" if remote else "cpu_reference"
        out = fn(*args)
        outputs = out if isinstance(out,(list,tuple)) else [out]
        value = sum(v.float().sum() for v in outputs)
        grads = torch.autograd.grad(value,leaves,create_graph=ctx.p["order"]==2)
        if ctx.p["order"]==2:
            differentiable = [v for v in grads if v.requires_grad]
            if not differentiable:
                results.append([torch.zeros_like(a).cpu() for a in leaves])
                continue
            grads = torch.autograd.grad(sum(v.sum() for v in differentiable),leaves,allow_unused=True)
            grads = [g if g is not None else torch.zeros_like(a) for g,a in zip(grads,leaves)]
        results.append(host(grads))
    ctx.phase = "correctness"
    return {"comparison":compare(results[1],results[0],ctx.c)}


def operator(ctx):
    import torch

    from .core import read_opset
    from .fixtures import directory, load, resolve
    name = ctx.p["name"]
    if name.startswith("reflop."):
        return native_operator(ctx,name)
    if ctx.p["flags"]&512:
        import torch
        dev = ctx.device()
        t = torch.ones(4).to(dev)
        try:
            t.resize_(8)
            ctx.sync(dev)
        except RuntimeError as e:
            return {"banned_operation_rejected":str(e)}
        raise AssertionError("banned resize_ was accepted")
    idx = directory()/"index.json"
    if not idx.exists():
        raise MissingPrerequisite("operator recipe preparation failed; inspect operators/prepare")
    index = json.loads(idx.read_text())
    if name not in index["recipes"]:
        raise CoverageGap(index["uncovered"].get(name,"no validated native recipe"))
    entry = read_opset()[name]
    dev = ctx.device()
    args,kwargs,source = load(name)
    ctx.details["recipe_source"] = source
    fn = resolve(name)
    ctx.phase = "cpu_reference"
    torch.manual_seed(ctx.c["seed"])
    want = host(fn(*args,**kwargs))
    ctx.phase = "remote_execute"
    args,kwargs,_ = load(name,dev)
    start = time.perf_counter()
    if entry[1]&32:
        got = host(fn(*args,**kwargs))
        check = compare(got,torch.zeros_like(want),ctx.c)
    elif entry[1]&2:
        ctx.remote().seed(ctx.c["seed"])
        got = host(fn(*args,**kwargs))
        args2,kwargs2,_ = load(name,dev)
        ctx.remote().seed(ctx.c["seed"])
        again = host(fn(*args2,**kwargs2))
        check = compare(got,again,ctx.c)
    else:
        got = host(fn(*args,**kwargs))
        ctx.phase = "correctness"
        # SVD/QR eigenvectors can differ by sign across CUDA/CPU. The high-level
        # decomposition probe additionally validates residuals; don't hide drift.
        check = compare(got,want,ctx.c,entry[6]=="fp8")
    return {"comparison":check,"seconds":time.perf_counter()-start,"recipe_source":source,
            "rng_comparison":"same remote seed" if entry[1]&2 else "native CPU"}


def native_operator(ctx,name):
    import torch
    from reflop import session
    from reflop.natives import op
    from reflop.tensor import dispatch
    base = name.split(".")[1]
    dev = ctx.device()
    sess = session.by_index(dev.index)
    p = torch.linspace(.1,.8,8)
    g = torch.full((8,),.2)
    args,kw = (),{}
    if base=="dataset_gather":
        args = (p.reshape(4,2),torch.tensor([0,3,1]))
        expected = args[0][args[1]]
    elif base=="bounds_check":
        args = (torch.tensor([0,1,3]),4,None)
        expected = args[0].clone()
    elif base in ("nonzero_count","mask_count"):
        args = (torch.tensor([0,1,0,2]),)
        expected = torch.tensor(2,dtype=torch.int64)
    elif base=="randperm_seeded":
        args = (16,ctx.c["seed"],torch.int64)
        expected = torch.randperm(16,generator=torch.Generator().manual_seed(ctx.c["seed"]))
    elif base in ("fused_adam","fused_adamw"):
        args = ([p],[g],[torch.zeros_like(p)],[torch.zeros_like(p)],[],[torch.tensor(1.)],
                .01,.9,.999,.01,1e-8,False,False)
        pp,mm,vv,ss = [p.clone()],[torch.zeros_like(p)],[torch.zeros_like(p)],[torch.tensor(1.)]
        fn = torch._fused_adamw_ if base=="fused_adamw" else torch._fused_adam_
        ctx.phase = "cpu_reference"
        fn(pp,[g],mm,vv,[],ss,lr=.01,beta1=.9,beta2=.999,weight_decay=.01,eps=1e-8,amsgrad=False,maximize=False)
        expected = (pp,mm,vv,[],ss)
    elif base=="fused_sgd":
        args = ([p],[g],[torch.zeros_like(p)],.01,.9,.01,0.,False,False,True)
        gradient = g+.01*p
        expected = ([p-.01*gradient],[gradient])
    elif base=="fused_adagrad":
        args = ([p],[g],[torch.zeros_like(p)],[torch.tensor(1.)],.01,0.,0.,1e-10,False)
        pp,ss,st = [p.clone()],[torch.zeros_like(p)],[torch.tensor(1.)]
        ctx.phase = "cpu_reference"
        torch._fused_adagrad_(pp,[g],ss,st,lr=.01,lr_decay=0.,weight_decay=0.,eps=1e-10,maximize=False)
        expected = (pp,ss,st)
    elif base=="amp_unscale_and_check":
        args = ([g],torch.tensor(0.),torch.tensor(.5))
        expected = ([g*.5],torch.tensor(0.))
    elif base=="amp_update_scale":
        args = (torch.tensor(1024.),torch.tensor(1,dtype=torch.int32),torch.tensor(0.),2.,.5,2)
        expected = (torch.tensor(2048.),torch.tensor(0,dtype=torch.int32))
    else:
        raise CoverageGap(f"no reference for newly added native operator {base}")
    def move(v):
        if isinstance(v,torch.Tensor):
            return v.to(dev)
        if isinstance(v,list):
            return [move(t) for t in v]
        if isinstance(v,tuple):
            return tuple(move(t) for t in v)
        return v
    ctx.phase = "remote_execute"
    result = dispatch(sess,op(base),move(args),kw)
    if base=="randperm_seeded":
        # Seeded random permutations need not have CPU/CUDA-identical order.
        got = host(result)
        compare(got.sort().values,torch.arange(16),ctx.c)
        again = dispatch(sess,op(base),move(args),kw)
        return {"permutation_verified":True,"same_remote_seed":compare(again,got,ctx.c),
                "cross_backend_rng_equality_required":False}
    ctx.phase = "correctness"
    return {"comparison":compare(result,expected,ctx.c),"reference":"explicit native PyTorch/mathematical reference"}


def prepare_ops(ctx):
    from .fixtures import directory, prepare
    index = prepare()
    write_json(ctx.path/"coverage.json",index)
    dedicated = [n for n in index["uncovered"] if n.startswith("reflop.") or n=="aten.resize_.default"]
    return {"cache":str(directory()),"registry_entries":index["total_registry_entries"],
            "validated_recipes":len(index["recipes"]),"dedicated_runtime_recipes":dedicated,
            "coverage_gaps":len(index["uncovered"])-len(dedicated)}


HANDLERS = {"train":train,"tensor":tensor,"autograd":autograd,"operator":operator,"prepare_ops":prepare_ops}


def main():
    path = Path(sys.argv[1]).resolve()
    request = json.loads(path.read_text())
    ctx = None
    result = {"status":"error"}
    import signal
    def interrupted(signum,frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM,interrupted)
    try:
        ctx = Context(request,path.parent)
        import torch
        torch.set_num_threads(ctx.c["cpu_threads"])
        torch.manual_seed(ctx.c["seed"])
        handler = request["case"]["handler"]
        if handler in HANDLERS:
            details = HANDLERS[handler](ctx)
        else:
            from . import probes
            details = getattr(probes,handler)(ctx)
        result = {"status":"pass","details":details or {}}
    except MissingPrerequisite as e:
        result = {"status":"skipped","failure_category":"prerequisite","message":str(e)}
    except CoverageGap as e:
        result = {"status":"coverage_gap","failure_category":"missing_recipe","message":str(e)}
    except (KeyboardInterrupt,SystemExit) as e:
        result = {"status":"interrupted","failure_category":"interrupted","message":str(e)}
    except Exception as e:
        category = "correctness" if isinstance(e,AssertionError) else "execution"
        status = "fail"
        if ctx and ctx.phase=="cpu_reference":
            status,category = "fixture_error","native_reference"
        elif type(e).__name__ in ("UnsupportedOperatorError","CaptureError","NotImplementedError"):
            status,category = "unsupported","compatibility"
        elif isinstance(e,TimeoutError):
            status,category = "timeout","deadline"
        if hasattr(e,"stage"):
            category = "remote_validation" if e.stage else "remote_execution"
        result = {"status":status,"failure_category":category,"message":str(e),"exception_type":type(e).__name__,
                  "phase":ctx.phase if ctx else "setup","traceback":traceback.format_exc(),
                  "remote_error":{n:getattr(e,n) for n in ("code","stage","node","op_name") if hasattr(e,n)}}
    finally:
        if ctx:
            try:
                ctx.close()
            except Exception as e:
                result["cleanup_error"] = f"{type(e).__name__}: {e}"
                if result["status"]=="pass":
                    result.update(status="fail",failure_category="cleanup")
            result.setdefault("details",{}).update(ctx.details)
        write_json(path.parent/"outcome.json",result)
    return 0 if result["status"] in ("pass","unsupported","skipped","coverage_gap") else 1


if __name__ == "__main__":
    raise SystemExit(main())
