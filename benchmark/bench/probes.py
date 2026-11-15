from __future__ import annotations

import copy
import gc
import hashlib
import os
import secrets
import ssl
import subprocess
import sys
import time
from pathlib import Path

from .core import distribution, environment
from .worker import compare, delta, host


def inventory(ctx):
    import torch
    what = ctx.p["what"]
    if what=="client":
        from .core import read_opset
        return {**environment(),"registry_entries":len(read_opset()),"local_cuda":torch.cuda.is_available(),
                "mps_available":bool(hasattr(torch.backends,"mps") and torch.backends.mps.is_available())}
    if what.startswith("cli_"):
        args = [sys.executable,"-m","reflop",what.removeprefix("cli_")]
        p = subprocess.run(args,capture_output=True,text=True,timeout=ctx.c["timeout_seconds"]-10)
        (ctx.path/"cli.stdout.log").write_text(p.stdout)
        (ctx.path/"cli.stderr.log").write_text(p.stderr)
        assert p.returncode==0,f"CLI exited {p.returncode}; inspect cli logs"
        return {"exit_code":p.returncode,"command":args}
    if what=="transport":
        from reflop.client import admin
        sockets = []
        original = ssl.SSLContext.wrap_socket
        def record(self,*a,**kw):
            result = original(self,*a,**kw)
            sockets.append({"version":result.version(),"cipher":result.cipher()})
            return result
        ssl.SSLContext.wrap_socket = record
        try:
            conn = admin(os.environ.get("REFLOP_SERVER"))
            try:
                transport = conn.entry.transport
                if transport=="tls":
                    assert sockets and all(s["version"]=="TLSv1.3" for s in sockets)
                return {"transport":transport,"tls":sockets,"pinned_authentication_connected":True,
                        "ssh_evidence":"authenticated SSH client connection" if transport=="ssh" else None}
            finally:
                conn.close()
        finally:
            ssl.SSLContext.wrap_socket = original
    ctx.device()
    return ctx.details


def link(ctx):
    from reflop import session
    from reflop._pb import compute_pb2 as cpb
    dev = ctx.device()
    conn = session.by_index(dev.index).conn
    samples = []
    for i in range(ctx.c["warmup_steps"]+ctx.c["measured_steps"]):
        nonce = secrets.randbits(63)
        start = time.perf_counter()
        reply = conn.call(cpb.Envelope(ping=cpb.Ping(nonce=nonce)),timeout=20)
        elapsed = time.perf_counter()-start
        assert reply.pong.nonce==nonce
        if i>=ctx.c["warmup_steps"]:
            samples.append(elapsed)
    return {"application_rtt":distribution(samples),"method":"authenticated protocol ping, unshaped real connection"}


def transfer(ctx):
    import torch
    dev = ctx.device()
    n = ctx.p["nbytes"]
    data = (torch.arange(n,dtype=torch.int64)%251).to(torch.uint8)
    direction = ctx.p["direction"]
    resident = data.to(dev)
    ctx.sync(dev)
    samples = []
    before = ctx.stats()
    timed_counters = []
    for i in range(ctx.c["warmup_steps"]+ctx.c["repeats"]):
        iteration_before = ctx.stats()
        t0 = time.perf_counter()
        if direction=="upload":
            out = data.to(dev)
            ctx.sync(dev)
        elif direction=="fetch":
            out = resident.cpu()
        else:
            out = data.to(dev).cpu()
        elapsed = time.perf_counter()-t0
        if i>=ctx.c["warmup_steps"]:
            samples.append(elapsed)
            timed_counters.append(delta(iteration_before,ctx.stats()))
        compare(out,data,ctx.c)
        del out
        gc.collect()
    return {"bytes":n,"direction":direction,"timing":distribution(samples),
            "payload_megabytes_per_second":[n/max(v,1e-12)/1e6 for v in samples],
            "counter_delta_including_warmup_and_validation":delta(before,ctx.stats()),
            "timed_counter_deltas":timed_counters,"content_verified":True}


def graphs(ctx):
    import torch
    dev = ctx.device()
    rf = ctx.remote()
    what = ctx.p["what"]
    size = ctx.p.get("size",32)
    if what=="matmul":
        x = torch.randn(size,size)
        a = x.to(dev)
        samples = []
        before = ctx.stats()
        for i in range(ctx.c["warmup_steps"]+ctx.c["measured_steps"]):
            t0 = time.perf_counter()
            out = a@a
            rf.step_boundary()
            ctx.sync(dev)
            if i>=ctx.c["warmup_steps"]:
                samples.append(time.perf_counter()-t0)
        check = compare(out,x@x,ctx.c)
        return {"timing":distribution(samples),"gflops_per_second":[2*size**3/v/1e9 for v in samples],
                "counters":delta(before,ctx.stats()),"comparison":check}
    if what=="chain":
        a = torch.ones(64).to(dev)
        before = ctx.stats()
        start = time.perf_counter()
        for _ in range(size):
            a = a*.999+.001
            if ctx.p["boundary"]=="each":
                rf.step_boundary()
        out = a.cpu()
        compare(out,torch.ones(64),ctx.c)
        return {"seconds":time.perf_counter()-start,"nodes_requested":size*2,"counters":delta(before,ctx.stats())}
    before = ctx.stats()
    a = torch.arange(64.).reshape(8,8).to(dev)
    if what=="cache_reuse":
        for _ in range(10):
            out = (a*2+1).sum()
            rf.step_boundary()
            out.item()
        check = compare(out,(torch.arange(64.)*2+1).sum(),ctx.c)
    elif what=="shape_churn":
        for n in range(2,18):
            out = torch.ones(n,n).to(dev).sum()
            assert out.item()==n*n
        check = {"shapes":16}
    elif what=="scalar_churn":
        for i in range(10):
            out = (a+i).sum()
            assert out.item()==2016+i*64
        check = {"scalars":10}
    elif what=="multi_output":
        check = compare((a.sum(),a.max(0),a.t()),(torch.arange(64.).sum(),torch.arange(64.).reshape(8,8).max(0),torch.arange(64.).reshape(8,8).t()),ctx.c)
    elif what=="dead_code":
        unused = (a@a).sin().cos()
        check = compare(a+1,torch.arange(64.).reshape(8,8)+1,ctx.c)
        del unused
    else:
        v = a.t()
        v[0].add_(3)
        expected = torch.arange(64.).reshape(8,8)
        expected[:,0]+=3
        check = compare(a,expected,ctx.c)
    ctx.sync(dev)
    return {"comparison":check,"counters":delta(before,ctx.stats())}


def capture(ctx):
    import torch
    rf = ctx.remote()
    dev = ctx.device()
    what = ctx.p["what"]
    if what in ("gradient_accumulation","lr_schedule"):
        from .models import build, optimizer
        model,x,y,loss_fn = build("mlp",8,ctx.c["seed"])
        reference = copy.deepcopy(model)
        model.to(dev)
        opt,ropt = optimizer(model,"momentum"),optimizer(reference,"momentum")
        def step(a,b):
            loss = loss_fn(model(a.to(dev)),b.to(dev))
            loss.backward()
            if what=="lr_schedule":
                opt.step()
                opt.zero_grad()
            return loss
        fn = rf.capture(step)
        for i in range(8):
            if what=="lr_schedule":
                opt.param_groups[0]["lr"] = .01/(i+1)
                ropt.param_groups[0]["lr"] = .01/(i+1)
            try:
                value = fn(x,y).item()
            except rf.CaptureError as e:
                if what=="lr_schedule":
                    return {"safely_refused_python_schedule":str(e),"replays":fn.replays}
                raise
            loss = loss_fn(reference(x),y)
            loss.backward()
            if what=="lr_schedule":
                ropt.step()
                ropt.zero_grad()
            compare(torch.tensor(value),loss.detach(),ctx.c)
        return {"comparison":compare({k:p.grad for k,p in model.named_parameters()},
                                       {k:p.grad for k,p in reference.named_parameters()},ctx.c),"replays":fn.replays}
    scalar = [1.]
    def step(x):
        if what=="mutated_argument":
            x.mul_(10)
        a = x.to(dev)
        if what=="python_branch":
            return a.sum()*(2 if x[0].item()>0 else 3)
        if what=="changed_scalar":
            return a.sum()*scalar[0]
        if what=="stale_view":
            b = a.t() if a.ndim==2 else a.view(-1)
            out = b+1
            rf.step_boundary()
            return out.sum()
        if what=="rng":
            return torch.rand_like(a)
        if what=="nested_output":
            return {"sum":a.sum(),"values":(a+1,a*2)}
        return (a*2).sum()
    fn = rf.capture(step)
    previous = None
    for i in range(8):
        shape = (8,) if what!="changed_shape" or i<4 else (12,)
        dtype = torch.float32 if what!="changed_dtype" or i<4 else torch.float64
        value = float(i+1) if what!="python_branch" or i%2==0 else -float(i+1)
        x = torch.full(shape,value,dtype=dtype)
        scalar[0] = float(i+1)
        try:
            got = host(fn(x))
        except rf.CaptureError as e:
            if what in ("changed_scalar","python_branch"):
                return {"safely_refused_dynamic_python":str(e),"replays":fn.replays}
            raise
        if what=="rng":
            assert got.shape==x.shape and ((got>=0)&(got<1)).all()
            if previous is not None:
                assert not torch.equal(got,previous),"captured RNG froze"
            previous = got
        elif what=="nested_output":
            compare(got,{"sum":x.sum(),"values":(x+1,x*2)},ctx.c)
        else:
            expected = x.sum()*(scalar[0] if what=="changed_scalar" else
                ((2 if value>0 else 3) if what=="python_branch" else (1 if what=="stale_view" else 2)))
            if what=="stale_view":
                expected+=x.numel()
            compare(got,expected,ctx.c)
    return {"replays":fn.replays,"fresh_values_verified":True}


def sessions(ctx):
    import torch
    from reflop import session
    rf = ctx.remote()
    what = ctx.p["what"]
    dev = ctx.device()
    if what=="open_close":
        timings = []
        for _ in range(min(10,ctx.c["memory_cycles"])):
            assert torch.ones(3).to(dev).sum().item()==3
            t0 = time.perf_counter()
            session.by_index(0).close()
            ctx.devices.pop(0,None)
            dev = ctx.device()
            timings.append(time.perf_counter()-t0)
        return {"close_open":distribution(timings)}
    if what=="seed":
        rf.seed(42)
        a = torch.randn(256,device=dev).cpu()
        rf.seed(42)
        b = torch.randn(256,device=dev).cpu()
        assert torch.equal(a,b)
        return {"seed_reproducible":True}
    if what=="closed_tensor":
        x = torch.ones(3).to(dev)
        ctx.sync(dev)
        session.by_index(0).close()
        try:
            (x+1).cpu()
        except (RuntimeError,KeyError) as e:
            return {"rejected_stale_tensor":type(e).__name__}
        raise AssertionError("closed session tensor remained usable")
    if what=="threads":
        from concurrent.futures import ThreadPoolExecutor
        def work(i):
            return (torch.ones(16).to(dev)*(i+1)).sum().item()
        with ThreadPoolExecutor(4) as pool:
            values = list(pool.map(work,range(16)))
        assert values==[16*(i+1) for i in range(16)],values
        return {"thread_results":values}
    other = ctx.device("remote_cpu",index=1)
    x = torch.ones(3).to(dev).requires_grad_()
    y = torch.ones(3).to(other)
    if what=="cross_session_reject":
        try:
            (x+y).cpu()
        except RuntimeError as e:
            assert "session" in str(e).lower(),str(e)
            return {"rejected":str(e)}
        raise AssertionError("mixed sessions were accepted implicitly")
    if what=="move_gradients":
        (x.to(other)*2).sum().backward()
        return {"gradients":compare(x.grad,torch.full((3,),2.),ctx.c)}
    return {"gpu":compare(x*3,torch.full((3,),3.),ctx.c),"cpu":compare(y*5,torch.full((3,),5.),ctx.c),
            "roundtrip":compare(x.to(other).to(dev),torch.ones(3),ctx.c)}


def placement(ctx):
    import torch
    rf = ctx.remote()
    dev = ctx.device()
    x = torch.arange(16.).to(dev)
    with rf.placement(ctx.p["mode"]):
        out = x*2+1
        check = compare(out,torch.arange(16.)*2+1,ctx.c)
        actual = str(out.device)
    if ctx.p["mode"]=="local":
        assert out.device.type=="cpu",actual
    elif ctx.p["mode"]=="remote":
        assert out.device.type=="reflop",actual
    if ctx.p["mode"]=="remote-cpu":
        with rf.placement("remote-cpu"):
            a = torch.ones(4).to(dev)
        s = a._rf_storage.session
        from reflop._pb import graph_pb2 as gpb
        assert s.gdevice.kind==gpb.DEVICE_CPU
        compare(a*3,torch.full((4,),3.),ctx.c)
    return {"comparison":check,"result_device":actual}


def datasets(ctx):
    import torch
    from reflop._pb import compute_pb2 as cpb
    from reflop._pb import data_pb2 as dpb
    from reflop.client import admin
    from reflop.data import write_pack
    from reflop.data.push import push
    from reflop.data.tensorpack import read_manifest
    what = ctx.p["what"]
    dev = ctx.device()
    rf = ctx.remote()
    count = ctx.c["dataset_samples"]
    # Each run/case needs fresh content: CAS reuse from an earlier case would
    # make the interrupted-upload probe silently skip actual chunk transfer.
    identity = ctx.run_dir.name+ctx.case["id"]
    data_seed = (ctx.c["seed"]^int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8],"big"))%(1<<63)
    gen = torch.Generator().manual_seed(data_seed)
    data = {"x":torch.randn(count,256,generator=gen),"y":torch.randint(0,10,(count,),generator=gen)}
    pack = write_pack(data,ctx.path/"pack",shard_bytes=ctx.c["dataset_shard_bytes"])
    raw,manifest = read_manifest(pack)
    name = "bench_"+secrets.token_hex(12)
    conn = admin(os.environ.get("REFLOP_SERVER"))
    dataset_id = None
    start = time.perf_counter()
    detail = {"data_seed":data_seed}
    try:
        if what=="bad_manifest":
            try:
                conn.call(cpb.Envelope(pack_begin=dpb.PackBegin(name=name,manifest=b'{"invalid":true}')),timeout=30)
            except Exception as e:
                assert getattr(e,"code",None) in (cpb.ERROR_DATASET,cpb.ERROR_INVALID_GRAPH,cpb.ERROR_PROTOCOL,cpb.ERROR_RESOURCE_LIMIT),str(e)
                return {"rejected_manifest":str(e)}
            raise AssertionError("invalid TensorPack manifest accepted")
        if what in ("resume_upload","bad_checksum"):
            status = conn.call(cpb.Envelope(pack_begin=dpb.PackBegin(name=name,manifest=raw)),timeout=60).pack_status
            todo = [(s.shard,ch) for s in status.shards for ch in s.missing_chunks]
            assert todo,"new randomized pack should have missing chunks"
            sh,ch = todo[0]
            with (Path(pack)/manifest["shards"][sh]["file"]).open("rb") as f:
                f.seek(ch*status.chunk_bytes)
                blob = f.read(status.chunk_bytes)
            message = dpb.PackChunk(pack_id=status.pack_id,shard=sh,chunk=ch,data=blob,
                                    sha256=b"\x00"*32 if what=="bad_checksum" else hashlib.sha256(blob).digest())
            if what=="bad_checksum":
                try:
                    conn.call(cpb.Envelope(pack_chunk=message),timeout=60)
                except Exception as e:
                    assert getattr(e,"code",None) in (cpb.ERROR_DATASET,cpb.ERROR_PROTOCOL,cpb.ERROR_RUNTIME_CHECK),str(e)
                    detail["bad_checksum_rejected"] = str(e)
                else:
                    raise AssertionError("corrupt chunk checksum accepted")
            else:
                conn.call(cpb.Envelope(pack_chunk=message),timeout=60)
                conn.close()
                conn = admin(os.environ.get("REFLOP_SERVER"))
                resumed = conn.call(cpb.Envelope(pack_begin=dpb.PackBegin(name=name,manifest=raw)),timeout=60).pack_status
                remaining = sum(len(s.missing_chunks) for s in resumed.shards)
                assert remaining==len(todo)-1,(remaining,len(todo))
                detail["chunks_preserved_after_disconnect"] = 1
        info = push(conn,str(pack),name,progress=False)
        dataset_id = info.dataset_id
        detail["upload_seconds"] = time.perf_counter()-start
        assert info.num_samples==count
        if what=="dedup_upload":
            status = conn.call(cpb.Envelope(pack_begin=dpb.PackBegin(name=name,manifest=raw)),timeout=60).pack_status
            assert sum(len(s.missing_chunks) for s in status.shards)==0
            detail["missing_chunks_second_upload"] = 0
        ds = rf.dataset(name,device=dev)
        ids = torch.tensor([0,count-1,1,1])
        batch = ds.batch(ids)
        compare(batch,{k:v[ids] for k,v in data.items()},ctx.c)
        if what in ("loader","capture_loader"):
            from .models import build, optimizer
            model,_,_,loss_fn = build("mlp",32,ctx.c["seed"])
            model.to(dev)
            opt = optimizer(model,"momentum")
            def step(x,y):
                opt.zero_grad()
                loss = loss_fn(model(x),y)
                loss.backward()
                opt.step()
                return loss
            fn = rf.capture(step) if what=="capture_loader" else step
            # Validate shuffle, epochs and remainder outside the timed training.
            order_gen = torch.Generator().manual_seed(ctx.c["seed"])
            expected_orders = [torch.randperm(count,generator=order_gen) for _ in range(2)]
            batches = ds.loader(37,shuffle=True,seed=ctx.c["seed"],drop_last=False,epochs=2)
            for order in expected_orders:
                for offset in range(0,count,37):
                    actual = next(batches)
                    ids = order[offset:offset+37]
                    compare(actual,{k:v[ids] for k,v in data.items()},ctx.c)
            try:
                next(batches)
            except StopIteration:
                pass
            else:
                raise AssertionError("dataset loader yielded extra batches")
            seen = 0
            before = ctx.stats()
            t0 = time.perf_counter()
            for b in ds.loader(32,shuffle=True,seed=ctx.c["seed"],drop_last=False,epochs=1):
                out = fn(b["x"],b["y"])
                if what=="loader":
                    rf.step_boundary()
                seen+=b["x"].shape[0]
            out.item()
            ctx.sync(dev)
            assert seen==count
            detail.update(samples=seen,seconds=time.perf_counter()-t0,counters=delta(before,ctx.stats()),
                          replays=getattr(fn,"replays",None),input_residency="server TensorPack",
                          shuffle_epochs_remainder_verified=True)
        return detail|{"dataset_rows":count,"batch_values_verified":True}
    finally:
        if dataset_id is not None:
            conn.call(cpb.Envelope(delete_dataset=dpb.DeleteDataset(dataset_id=dataset_id)),timeout=60)
        conn.close()


def limits(ctx):
    import psutil
    import torch
    from reflop import session
    rf = ctx.remote()
    dev = ctx.device()
    what = ctx.p["what"]
    if what in ("bounds_error","bad_shape","division_zero"):
        if what=="bounds_error":
            def fn():
                return torch.arange(8.).to(dev)[torch.tensor([99]).to(dev)].cpu()
            codes = ("ERROR_INDEX_OUT_OF_BOUNDS","ERROR_EXECUTION")
        elif what=="bad_shape":
            def fn():
                return (torch.ones(2,3).to(dev)@torch.ones(4,2).to(dev)).cpu()
            codes = ("ERROR_SHAPE_MISMATCH","ERROR_INVALID_GRAPH")
        else:
            def fn():
                return (torch.ones(8,dtype=torch.int64).to(dev)//torch.zeros(8,dtype=torch.int64).to(dev)).cpu()
            codes = ("ERROR_RUNTIME_CHECK","ERROR_EXECUTION")
        try:
            fn()
        except (RuntimeError,IndexError) as e:
            detail = {"rejected":type(e).__name__,"message":str(e),"remote_code":getattr(e,"code",None)}
            # A local shape check is valid, but don't call it server validation.
            if hasattr(e,"code"):
                from reflop._pb import compute_pb2 as cpb
                assert e.code in [getattr(cpb,n) for n in codes],str(e)
            detail["validation_location"] = "remote" if hasattr(e,"code") else "client"
            # A rejected trace may remain dirty: reopen, then prove service health.
            ctx.close_sessions(expected_errors=True)
            fresh = ctx.device()
            assert torch.ones(8).to(fresh).sum().item()==8
            return detail|{"fresh_session_healthy":True}
        raise AssertionError(f"invalid {what} was accepted")
    if what=="session_limit":
        samples = []
        for i in range(ctx.c["max_sessions"]):
            try:
                d = ctx.device(index=i)
                assert torch.ones(1).to(d).item()==1
                samples.append({"index":i,"opened":True})
            except Exception as e:
                if not hasattr(e,"code"):
                    raise
                samples.append({"index":i,"opened":False,"error":str(e)})
                break
        return {"attempts":samples,"bounded_by_config":True,"limit_found":not samples[-1]["opened"]}
    if what=="graph_limit":
        return graphs_with_depth(ctx)
    process = psutil.Process()
    samples = []
    tensors = []
    if what=="allocation":
        budget = ctx.c["memory_budget_bytes"]
        used = 0
        start = time.perf_counter()
        while used<budget:
            n = min(16<<20,budget-used)
            try:
                tensors.append(torch.zeros(n,dtype=torch.uint8).to(dev))
                ctx.sync(dev)
                used+=n
                samples.append({"requested_live_bytes":used,"client_rss":process.memory_info().rss})
            except rf.OutOfMemoryError as e:
                tensors.clear()
                gc.collect()
                return {"oom_at_requested_bytes":used+n,"samples":samples,"error":str(e),"seconds":time.perf_counter()-start}
        tensors.clear()
        gc.collect()
        ctx.sync(dev)
        assert torch.ones(8).to(dev).sum().item()==8
        return {"samples":samples,"ceiling_bytes":budget,"limit_found":False,"post_free_healthy":True}
    for i in range(ctx.c["memory_cycles"]):
        x = torch.ones(16384).to(dev)
        assert (x*2).sum().item()==32768
        del x
        if i%10==0:
            gc.collect()
            ctx.sync(dev)
            s = session.by_index(dev.index)
            samples.append({"cycle":i,"client_rss":process.memory_info().rss,"pending_frees":len(s.pending_free),
                            "registered_graphs":len(s.registered)})
    return {"samples":samples,"cycles":ctx.c["memory_cycles"]}


def graphs_with_depth(ctx):
    import torch
    dev = ctx.device()
    ctx.remote()
    before = ctx.stats()
    x = torch.ones(4).to(dev)
    for _ in range(ctx.c["max_graph_chain"]):
        x=x*.999+.001
    compare(x,torch.ones(4),ctx.c)
    return {"requested_chain_length":ctx.c["max_graph_chain"],"counters":delta(before,ctx.stats()),
            "note":"bounded client graph chunking probe; not an unbounded resource exhaustion test"}


def soak(ctx):
    import psutil
    import torch

    from .models import build, optimizer
    dev = ctx.device()
    rf = ctx.remote()
    model,x,y,loss_fn = build("gpt",8,ctx.c["seed"])
    model.to(dev)
    opt = optimizer(model,"momentum")
    x,y = x.to(dev),y.to(dev)
    def step():
        opt.zero_grad()
        loss = loss_fn(model(x),y)
        loss.backward()
        opt.step()
        return loss
    fn = rf.capture(step) if ctx.p["mode"]=="capture" else step
    for _ in range(ctx.c["warmup_steps"]):
        fn().item()
    before = ctx.stats()
    proc = psutil.Process()
    start = last = time.monotonic()
    steps = 0
    samples = []
    while time.monotonic()-start<ctx.c["soak_seconds"]:
        t0 = time.perf_counter()
        loss = fn()
        if ctx.p["mode"]=="eager":
            rf.step_boundary()
        value = loss.item()
        assert torch.isfinite(torch.tensor(value)),"nonfinite loss during soak"
        steps+=1
        now = time.monotonic()
        if now-last>=ctx.c["soak_sample_seconds"]:
            sample = {"elapsed_seconds":now-start,"steps":steps,"loss":value,"step_seconds":time.perf_counter()-t0,
                      "rss_bytes":proc.memory_info().rss,"threads":proc.num_threads(),"counters":ctx.stats(),
                      "capture_replays":getattr(fn,"replays",None)}
            samples.append(sample)
            ctx.event("soak_sample",**sample)
            last = now
    slope = None
    if len(samples)>=2:
        slope = (samples[-1]["rss_bytes"]-samples[0]["rss_bytes"])/(samples[-1]["elapsed_seconds"]-samples[0]["elapsed_seconds"])
    return {"steps":steps,"elapsed_seconds":time.monotonic()-start,"rss_growth_bytes_per_second":slope,
            "samples":samples,"counters":delta(before,ctx.stats()),"automatic_leak_verdict":False}


# Feature groups with independent resource/cleanup lifetimes live in separate modules.
from .compatibility import compatibility  # noqa: E402
from .interface_probes import c_api, protocol, shim  # noqa: E402
from .state_probes import checkpoint, concurrency, disconnect, faults, isolation, migration  # noqa: E402

__all__ = ["checkpoint", "migration", "disconnect", "faults", "isolation", "concurrency", "shim", "c_api", "protocol", "compatibility"]
