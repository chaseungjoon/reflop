from __future__ import annotations

import json
import os
import random
import secrets
import subprocess
import sys
import threading
import time
from pathlib import Path

from .core import REPO, MissingPrerequisite, write_json
from .worker import compare, host


def pack_state(value,tensors):
    import torch
    if isinstance(value,torch.Tensor):
        key = f"tensor_{len(tensors)}"
        tensors[key] = value.detach().cpu().contiguous().clone()
        return {"tensor":key}
    if isinstance(value,dict):
        return {"mapping":[[pack_state(k,tensors),pack_state(v,tensors)] for k,v in value.items()]}
    if isinstance(value,(tuple,list)):
        return {"tuple" if isinstance(value,tuple) else "list":[pack_state(v,tensors) for v in value]}
    if value is None or isinstance(value,(str,int,float,bool)):
        return value
    raise TypeError(f"unsupported checkpoint value {type(value)}")


def unpack_state(value,tensors):
    if not isinstance(value,dict):
        return value
    if "tensor" in value:
        return tensors[value["tensor"]].clone()
    if "mapping" in value:
        return {unpack_state(k,tensors):unpack_state(v,tensors) for k,v in value["mapping"]}
    if "tuple" in value:
        return tuple(unpack_state(v,tensors) for v in value["tuple"])
    if "list" in value:
        return [unpack_state(v,tensors) for v in value["list"]]
    raise ValueError("invalid checkpoint tree")


def save_training(path,model,opt,scheduler,step,sampler,seed):
    import hashlib
    import importlib.metadata

    import torch
    from safetensors.torch import save_file
    state = {"model":model.state_dict(),"optimizer":opt.state_dict(),"scheduler":scheduler.state_dict(),
             "step":step,"sampler":sampler,"sampler_cursor":step,"torch_cpu_rng":torch.get_rng_state(),
             "python_rng":random.getstate(),"per_step_remote_seed_base":seed,
             "versions":{p:importlib.metadata.version(p) for p in ("torch","reflop")}}
    tensors = {}
    tree = pack_state(state,tensors)
    path.mkdir(parents=True,exist_ok=True)
    save_file(tensors,str(path/"state.safetensors"))
    write_json(path/"tree.json",tree)
    manifest = {"schema_version":1,"files":{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in (path/"state.safetensors",path/"tree.json")},
                "step":step,"rng_policy":"remote generator is explicitly reseeded by global step; arbitrary remote RNG state is not snapshotted"}
    write_json(path/"manifest.json",manifest)
    return manifest


def load_training(path):
    import hashlib

    from safetensors.torch import load_file
    manifest = json.loads((path/"manifest.json").read_text())
    for name,digest in manifest["files"].items():
        if name not in ("tree.json","state.safetensors"):
            raise ValueError("unexpected checkpoint filename")
        if hashlib.sha256((path/name).read_bytes()).hexdigest()!=digest:
            raise ValueError(f"checkpoint integrity mismatch: {name}")
    return unpack_state(json.loads((path/"tree.json").read_text()),load_file(str(path/"state.safetensors")))


def checkpoint(ctx):
    import torch
    from safetensors.torch import load_file

    from .models import build
    what = ctx.p["what"]
    dev = ctx.device()
    rf = ctx.remote()
    model,x,y,loss_fn = build("mlp",8,ctx.c["seed"])
    model.to(dev)
    if what=="server_snapshot":
        from reflop import session
        from reflop._pb import compute_pb2 as cpb
        tensor = torch.arange(16.).to(dev)
        ctx.sync(dev)
        s = session.by_index(dev.index)
        name = "bench_"+secrets.token_hex(10)
        try:
            saved = s.conn.call(cpb.Envelope(snapshot=cpb.Snapshot(name=name,handles=[tensor.reflop_handle],keys=["x"])),timeout=60)
            # Restore into a fresh handle in the same session and verify bytes.
            h = s.new_handle()
            s.conn.call(cpb.Envelope(restore=cpb.Restore(name=name,keys=["x"],handles=[h])),timeout=60)
            got = s.fetch_handles([h])[0]
            compare(got,torch.arange(16.),ctx.c)
            s.release(h)
            return {"server_snapshot_bytes":saved.snapshot_saved.bytes,"restored_values_verified":True}
        finally:
            s.conn.call(cpb.Envelope(delete_snapshot=cpb.DeleteSnapshot(name=name)),timeout=60)
    if what in ("weights","tied_weights"):
        if what=="tied_weights":
            w = model[0].weight
            state = {"weight":w,"alias":w}
            rf.fetch(state,str(ctx.path/"weights.safetensors"))
            loaded = load_file(str(ctx.path/"weights.safetensors"))
            compare(loaded["weight"],loaded["alias"],ctx.c)
            return {"duplicate_storage_saved":True,"alias_identity_restored":False}
        expected = model(x.to(dev)).cpu()
        rf.fetch(model.state_dict(),str(ctx.path/"weights.safetensors"))
        local,_,_,_ = build("mlp",8,ctx.c["seed"])
        local.load_state_dict(load_file(str(ctx.path/"weights.safetensors")))
        return {"comparison":compare(local(x),expected,ctx.c),"bytes":(ctx.path/"weights.safetensors").stat().st_size}
    return full_state(ctx,dev,model,x,y,loss_fn,corrupt=what=="corrupt_checkpoint")


def full_state(ctx,dev,model,x,y,loss_fn,corrupt=False,target_server=None):
    import torch

    from .models import build, optimizer
    rf = ctx.remote()
    opt = optimizer(model,"adamw_capturable",remote=True)
    sched = torch.optim.lr_scheduler.StepLR(opt,step_size=2,gamma=.9)
    permutation = torch.randperm(8,generator=torch.Generator().manual_seed(ctx.c["seed"]))
    checkpoint_step = max(2,ctx.c["correctness_steps"]//2)
    total_steps = checkpoint_step+4
    def advance(m,o,s,d,start,end):
        losses = []
        for step in range(start,end):
            rf.seed(ctx.c["seed"]+step)
            torch.manual_seed(ctx.c["seed"]+step)
            ids = permutation.roll(step)
            a,b = x[ids].to(d),y[ids].to(d)
            o.zero_grad()
            loss = loss_fn(m(a),b)
            loss.backward()
            o.step()
            s.step()
            losses.append(loss.item())
        return losses
    advance(model,opt,sched,dev,0,checkpoint_step)
    start = time.perf_counter()
    manifest = save_training(ctx.path/"checkpoint",model,opt,sched,checkpoint_step,permutation,ctx.c["seed"])
    save_seconds = time.perf_counter()-start
    if corrupt:
        file = ctx.path/"checkpoint/state.safetensors"
        data = bytearray(file.read_bytes())
        data[-1]^=1
        file.write_bytes(data)
        try:
            load_training(ctx.path/"checkpoint")
        except ValueError as e:
            return {"corruption_rejected":str(e),"manifest":manifest}
        raise AssertionError("corrupted checkpoint accepted")
    expected_losses = advance(model,opt,sched,dev,checkpoint_step,total_steps)
    expected = host(model.state_dict())
    expected_opt = host(opt.state_dict())
    failure_at = time.perf_counter()
    # This is manual restoration into a new session, never automatic failover.
    rf.close()
    ctx.devices.clear()
    if target_server:
        target = ctx.c["migration_target"]
        if target.get("client_home"):
            os.environ["REFLOP_CLIENT_HOME"] = str(Path(target["client_home"]).expanduser())
        dev = ctx.device(server=target_server)
    else:
        dev = ctx.device()
    restored = load_training(ctx.path/"checkpoint")
    m,_,_,_ = build("mlp",8,ctx.c["seed"])
    m.to(dev)
    m.load_state_dict(restored["model"])
    o = optimizer(m,"adamw_capturable",remote=True)
    o.load_state_dict(restored["optimizer"])
    s = torch.optim.lr_scheduler.StepLR(o,step_size=2,gamma=.9)
    s.load_state_dict(restored["scheduler"])
    torch.set_rng_state(restored["torch_cpu_rng"])
    random.setstate(restored["python_rng"])
    permutation = restored["sampler"]
    restore_seconds = time.perf_counter()-failure_at
    actual_losses = advance(m,o,s,dev,checkpoint_step,total_steps)
    first_progress = time.perf_counter()-failure_at
    compare(torch.tensor(actual_losses),torch.tensor(expected_losses),ctx.c)
    compare(m.state_dict(),expected,ctx.c)
    compare(host(o.state_dict()),expected_opt,ctx.c)
    return {"checkpoint_save_seconds":save_seconds,"restore_ready_seconds":restore_seconds,
            "restore_plus_four_steps_seconds":first_progress,"manual_restore":True,"automatic_failover":False,
            "checkpoint_step":checkpoint_step,"simulated_failure_step":total_steps,
            "rollback_steps":total_steps-checkpoint_step,"expected_losses":expected_losses,"restored_losses":actual_losses,
            "manifest":manifest,"optimizer_scheduler_sampler_verified":True}


def migration(ctx):
    from .models import build
    target = ctx.c["migration_target"]["server"]
    if not target:
        raise MissingPrerequisite("configure migration_target.server for manual cross-server restoration")
    dev = ctx.device()
    from reflop import session
    source = session.by_index(dev.index).conn.entry.name
    if source==target:
        raise MissingPrerequisite("migration target must differ from the source paired server")
    model,x,y,loss_fn = build("mlp",8,ctx.c["seed"])
    model.to(dev)
    return full_state(ctx,dev,model,x,y,loss_fn,target_server=target)


def disconnect(ctx):
    import torch
    from reflop import session
    rf = ctx.remote()
    dev = ctx.device()
    s = session.by_index(dev.index)
    what = ctx.p["what"]
    x = torch.arange(32.).to(dev)
    ctx.sync(dev)
    if what=="resume_expired":
        # A random, never-issued resume ID is a bounded substitute for waiting
        # the default 30-minute TTL. It does not measure expiry duration.
        from reflop._pb import compute_pb2 as cpb
        from reflop.client import admin
        conn = admin(os.environ.get("REFLOP_SERVER"))
        try:
            try:
                r = conn.call(cpb.Envelope(open_session=cpb.OpenSession(device=s.gdevice,
                                    resume_session_id=secrets.token_bytes(16))),timeout=30)
            except Exception as e:
                assert getattr(e,"code",None)==cpb.ERROR_UNKNOWN_SESSION,str(e)
            else:
                assert not r.session_opened.resumed,"unknown resume ID was accepted"
            return {"unknown_resume_id_rejected":True,"ttl_duration_measured":False}
        finally:
            conn.close()
    if what=="idle_reconnect":
        s.conn.fr.close()
        deadline = time.monotonic()+5
        while not s.conn.closed and time.monotonic()<deadline:
            time.sleep(.01)
        start = time.perf_counter()
        compare(x+2,torch.arange(32.)+2,ctx.c)
        return {"reconnect_seconds":time.perf_counter()-start,"preserved_remote_tensor":True}
    # Wait for an actual execute frame to be submitted, then sever transport.
    from reflop._pb import compute_pb2 as cpb
    original = s.conn._send
    triggered = threading.Event()
    def sever(env):
        original(env)
        if env.WhichOneof("body")=="execute":
            triggered.set()
            s.conn.fr.close()
    s.conn._send = sever
    start = time.perf_counter()
    try:
        out = (x*3).cpu()
        compare(out,torch.arange(32.)*3,ctx.c)
        outcome = "completed_correctly_before_disconnect"
    except rf.SessionLostError as e:
        outcome = "explicit_session_lost"
        ctx.details["error"] = str(e)
        ctx.details["expected_terminal_error"] = True
    finally:
        s.conn._send = original
    assert triggered.is_set(),"no execute was intercepted"
    return {"observed":outcome,"detection_seconds":time.perf_counter()-start}


def hook(ctx,name):
    from reflop import session
    argv = ctx.c["owner_hooks"][name]
    if not argv:
        raise MissingPrerequisite(f"configure owner_hooks.{name} as an argv array")
    s = session.by_index(0)
    variables = {"session_id":s.conn.session_id.hex(),"identity":s.conn.entry.identity,"server":s.conn.entry.name}
    args = [a.format(**variables) for a in argv]
    env = dict(os.environ)
    env.update({"BENCH_"+k.upper():v for k,v in variables.items()})
    t0 = time.perf_counter()
    p = subprocess.run(args,capture_output=True,text=True,env=env,timeout=30)
    with (ctx.path/"hooks.log").open("a") as f:
        f.write(json.dumps({"hook":name,"exit_code":p.returncode,"seconds":time.perf_counter()-t0})+"\n")
        f.write(p.stdout+p.stderr+"\n")
    assert p.returncode==0,f"owner hook {name} exited {p.returncode}; inspect hooks.log"


def faults(ctx):
    import torch
    from reflop import session
    rf = ctx.remote()
    what = ctx.p["what"]
    names = {"pause_resume":("pause","resume"),"kill_session":("kill_session",None),
             "quota":("quota_apply","quota_restore"),"network_partition":("network_down","network_up"),
             "worker_kill":("worker_kill","worker_restore")}
    apply,restore = names[what]
    for name in (apply,restore):
        if name and not ctx.c["owner_hooks"][name]:
            raise MissingPrerequisite(f"configure owner_hooks.{name}; no owner state was changed")
    dev = ctx.device()
    x = torch.ones(16).to(dev)
    assert x.sum().item()==16
    events = []
    s = session.by_index(0)
    original_event = s.conn.on_event
    s.conn.on_event = lambda e: events.append({"kind":int(e.kind),"text":e.text,"at":time.monotonic()})
    restored = threading.Event()
    restore_error = []
    def undo_later():
        if restore and not restored.wait(ctx.c["fault_wait_seconds"]):
            try:
                hook(ctx,restore)
                restored.set()
            except BaseException as e:
                restore_error.append(str(e))
    timer = None
    start = time.monotonic()
    try:
        hook(ctx,apply)
        if restore:
            timer = threading.Thread(target=undo_later,daemon=True)
            timer.start()
        observation = []
        def observe():
            try:
                observation.append((True,(x+1).cpu()))
            except BaseException as e:
                observation.append((False,e))
        computation = threading.Thread(target=observe,daemon=True)
        computation.start()
        computation.join(timeout=ctx.c["recovery_deadline_seconds"])
        if computation.is_alive():
            s.conn.fr.close()
            computation.join(timeout=5)
            raise TimeoutError("fault outcome exceeded recovery_deadline_seconds")
        try:
            ok,value = observation[0]
            if not ok:
                raise value
            compare(value,torch.full((16,),2.),ctx.c)
            outcome = "continued_correctly"
        except rf.SessionLostError as e:
            outcome = "explicit_session_lost"
            ctx.details["error"] = str(e)
            ctx.details["expected_terminal_error"] = True
        except Exception as e:
            from reflop._pb import compute_pb2 as cpb
            if what!="quota" or getattr(e,"code",None) not in (cpb.ERROR_QUOTA,cpb.ERROR_BUSY,cpb.ERROR_OUT_OF_MEMORY):
                raise
            outcome = "quota_rejected"
            ctx.details["expected_terminal_error"] = True
        if timer:
            timer.join(timeout=ctx.c["fault_wait_seconds"]+35)
        assert not restore_error,restore_error
        if what in ("kill_session","worker_kill"):
            assert outcome=="explicit_session_lost",f"fault not observed: {outcome}"
        if what=="quota":
            assert outcome=="quota_rejected","hook quota did not constrain the benchmark; use a quota this tiny operation exceeds"
        if what=="pause_resume":
            from reflop._pb import compute_pb2 as cpb
            kinds = [e["kind"] for e in events]
            assert cpb.Event.KIND_PAUSED in kinds and cpb.Event.KIND_RESUMED in kinds,"pause/resume events were not observed"
        return {"observed":outcome,"fault_to_outcome_seconds":time.monotonic()-start,"events":events,
                "automatic_migration_tested":False,"hooks_executed":True}
    finally:
        if restore and not restored.is_set():
            hook(ctx,restore)
            restored.set()
        if timer:
            timer.join(timeout=35)
        s.conn.on_event = original_event


def isolation(ctx):
    import torch
    from reflop import session
    from reflop._pb import compute_pb2 as cpb
    from reflop._pb import data_pb2 as dpb
    from reflop.client import admin
    what = ctx.p["what"]
    if what.startswith("cross_identity") and len(ctx.c["guest_homes"])<2:
        raise MissingPrerequisite("configure two distinct paired guest_homes to test identity isolation")
    ctx.remote()
    homes = ctx.c["guest_homes"]
    if what.startswith("cross_identity"):
        os.environ["REFLOP_CLIENT_HOME"] = str(Path(homes[0]).expanduser())
    d0 = ctx.device(index=0)
    a = torch.arange(16.).to(d0)
    ctx.sync(d0)
    handle = a.reflop_handle
    if what.startswith("cross_identity"):
        os.environ["REFLOP_CLIENT_HOME"] = str(Path(homes[1]).expanduser())
    ctx.device(index=1)
    s0,s1 = session.by_index(0),session.by_index(1)
    if what.startswith("cross_identity"):
        assert s0.conn.entry.identity_id!=s1.conn.entry.identity_id,"guest homes authenticate as the same identity"
    if what=="cross_identity_dataset":
        # Create a uniquely named object as A; B must not see or delete it.
        from reflop.data import write_pack
        from reflop.data.push import push
        name = "bench_"+secrets.token_hex(10)
        pack = write_pack({"x":torch.arange(16.).reshape(4,4)},ctx.path/"pack")
        os.environ["REFLOP_CLIENT_HOME"] = str(Path(homes[0]).expanduser())
        owner = admin(os.environ.get("REFLOP_SERVER"))
        info = None
        try:
            info = push(owner,str(pack),name,progress=False)
            lst = s1.conn.call(cpb.Envelope(list_datasets=dpb.ListDatasets()),timeout=30).dataset_list.datasets
            assert name not in [x.name for x in lst],"other identity's dataset listed"
            try:
                s1.conn.call(cpb.Envelope(delete_dataset=dpb.DeleteDataset(dataset_id=info.dataset_id)),timeout=30)
            except Exception as e:
                assert getattr(e,"code",None) in (cpb.ERROR_UNKNOWN_DATASET,cpb.ERROR_PERMISSION_DENIED,cpb.ERROR_DATASET),str(e)
            else:
                raise AssertionError("other identity's dataset deleted")
            return {"dataset_hidden_and_delete_denied":True}
        finally:
            if info is not None:
                owner.call(cpb.Envelope(delete_dataset=dpb.DeleteDataset(dataset_id=info.dataset_id)),timeout=30)
            owner.close()
    try:
        s1.fetch_handles([handle])
    except Exception as e:
        assert getattr(e,"code",None) in (cpb.ERROR_UNKNOWN_HANDLE,cpb.ERROR_PERMISSION_DENIED),str(e)
        compare(a,torch.arange(16.),ctx.c)
        return {"foreign_handle_rejected":str(e),"original_session_healthy":True}
    raise AssertionError("foreign tensor handle was accessible")


def concurrency(ctx):
    def group(n,label):
        folder = ctx.path/label
        folder.mkdir()
        jobs,streams = [],[]
        try:
            for i in range(n):
                path = folder/str(i)
                path.mkdir()
                req = {"config":ctx.c,"path":str(path),"barrier":str(folder/"start"),"index":i}
                write_json(path/"job.json",req)
                log = (path/"client.log").open("w")
                streams.append(log)
                jobs.append(subprocess.Popen([sys.executable,"-u","-m","benchmark.bench.concurrent_client",str(path/"job.json")],
                                             cwd=REPO,stdout=log,stderr=log))
            deadline = time.monotonic()+90
            while not all((folder/str(i)/"ready.json").exists() for i in range(n)):
                if any(p.poll() is not None for p in jobs):
                    raise AssertionError("a concurrency client failed before the barrier; inspect client logs")
                if time.monotonic()>deadline:
                    raise TimeoutError("concurrent clients did not become ready within 90 seconds")
                time.sleep(.05)
            start = time.perf_counter()
            (folder/"start").touch()
            for p in jobs:
                p.wait(timeout=max(1,ctx.case["timeout"]-(time.perf_counter()-start)-100))
            wall = time.perf_counter()-start
            assert all(p.returncode==0 for p in jobs),"concurrency client failed; inspect logs"
            results = [json.loads((folder/str(i)/"client.json").read_text()) for i in range(n)]
            rates = [r["steps"]/r["seconds"] for r in results]
            return {"clients":results,"wall_seconds":wall,"aggregate_steps_per_second":sum(r["steps"] for r in results)/wall,
                    "jain_fairness":sum(rates)**2/(n*sum(r*r for r in rates))}
        finally:
            for p in jobs:
                if p.poll() is None:
                    p.terminate()
                    try:
                        p.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        p.kill()
                        p.wait()
            for f in streams:
                f.close()
    rounds = []
    for i in range(ctx.c["concurrency_rounds"]):
        # Alternate order to make cache/thermal ordering visible.
        result = {}
        order = ((1,"alone"),(ctx.p["n"],"together")) if i%2==0 else ((ctx.p["n"],"together"),(1,"alone"))
        for n,label in order:
            result[label] = group(n,f"round{i}_{label}")
        result["aggregate_scaling"] = result["together"]["aggregate_steps_per_second"]/result["alone"]["aggregate_steps_per_second"]
        rounds.append(result)
    return {"rounds":rounds,"isolation":"multiple independent client processes and server sessions",
            "identity_mode":"configured guest homes, cycled" if ctx.c["guest_homes"] else "same paired identity"}
