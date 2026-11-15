from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import secrets
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

from .core import REPO, ROOT, MissingPrerequisite


def shim(ctx):
    args = [sys.executable,"-m","reflop","run",sys.executable,str(ROOT/"bench/shim_program.py"),ctx.p["what"]]
    result = subprocess.run(args,capture_output=True,text=True,timeout=ctx.c["timeout_seconds"]-15)
    (ctx.path/"shim.stdout.log").write_text(result.stdout)
    (ctx.path/"shim.stderr.log").write_text(result.stderr)
    assert result.returncode==0,f"CUDA shim subprocess exited {result.returncode}; inspect shim logs"
    rows = []
    for line in result.stdout.splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass
    assert rows,"shim produced no machine-readable measurement"
    return {"measurements":rows,"exit_code":result.returncode}


def c_api(ctx):
    for tool in ("go","cc","c++"):
        if not shutil.which(tool):
            raise MissingPrerequisite(f"C/C++ API requires {tool}")
    src = Path(os.environ.get("BENCH_REFLOP_SOURCE",REPO)).resolve()
    if not (src/"clib/go").is_dir():
        raise MissingPrerequisite("set BENCH_REFLOP_SOURCE to a reflop checkout containing clib/go")
    # Cgo cannot read macOS Keychain. Never export the private key implicitly.
    home = os.environ.get("REFLOP_CLIENT_HOME")
    base = Path(home).expanduser() if home else Path(os.environ.get("XDG_CONFIG_HOME",Path.home()/".config"))/"reflop"
    if not (base/"id_ed25519").exists():
        raise MissingPrerequisite("C API requires the paired file key; run reflop key --write-file once for this client home")
    build = ctx.path/"build"
    build.mkdir()
    lib = build/"libreflop.so"
    env = dict(os.environ,CGO_ENABLED="1")
    command = ["go","build","-trimpath"]
    if platform.system()=="Darwin":
        command.append("-ldflags=-extldflags=-Wl,-install_name,@rpath/libreflop.so")
    command += ["-buildmode=c-shared","-o",str(lib),"./clib/go"]
    commands = [command]
    program = build/"benchmark"
    cpp = ctx.p["what"]=="cpp_mlp"
    compile_cmd = ["c++" if cpp else "cc","-O2","-Wall","-Wextra"]
    if cpp:
        compile_cmd += ["-std=c++17"]
    compile_cmd += ["-I"+str(src/"clib/include"),str(ROOT/"native"/("mlp.cpp" if cpp else "matmul.c")),
                    "-L"+str(build),"-lreflop","-lm","-Wl,-rpath,"+str(build),"-o",str(program)]
    commands.append(compile_cmd)
    with (ctx.path/"build.log").open("w") as log:
        for cmd in commands:
            result = subprocess.run(cmd,cwd=src,stdout=log,stderr=log,env=env,timeout=600)
            assert result.returncode==0,"C/C++ build failed; inspect build.log"
    t0 = time.perf_counter()
    result = subprocess.run([str(program),ctx.c["remote_kind"]],capture_output=True,text=True,timeout=90)
    (ctx.path/"native.stdout.log").write_text(result.stdout)
    (ctx.path/"native.stderr.log").write_text(result.stderr)
    assert result.returncode==0,f"native API exited {result.returncode}; inspect native logs"
    return {"seconds":time.perf_counter()-t0,"measurement":json.loads(result.stdout),"source_checkout":str(src),
            "library_sha256":hashlib.sha256(lib.read_bytes()).hexdigest(),
            "scope":"C/C++ source version may differ from installed Python client; source is recorded separately"}


def protocol(ctx):
    import torch
    from reflop import session
    from reflop._pb import compute_pb2 as cpb
    from reflop._pb import graph_pb2 as gpb
    what = ctx.p["what"]
    if what=="terminal_sanitization":
        from reflop.errors import sanitize
        text = "visible\x1b[31mred\x1b[0m\x00\x07\u202eevil"
        out = sanitize(text)
        assert "\x1b" not in out and "\x00" not in out and "\x07" not in out and "\u202e" not in out
        assert len(sanitize("x"*10000))<=4000
        return {"control_characters_removed":True,"bounded_length":True}
    if what=="frame_limit":
        from reflop.client import Framer
        limit = 64<<20
        stream = io.BytesIO(struct.pack(">I",limit+1))
        fr = Framer(stream,io.BytesIO(),lambda:None)
        fr.max_frame = limit
        try:
            fr.read()
        except RuntimeError as e:
            assert "frame" in str(e).lower(),str(e)
            return {"oversized_header_rejected_without_payload_allocation":True}
        raise AssertionError("oversized frame accepted")
    if what=="pin_mismatch":
        from reflop.client import dial_tls
        from reflop.config import ClientConfig
        from reflop.errors import ServerKeyMismatch
        entry = ClientConfig.load().get(os.environ.get("REFLOP_SERVER"))
        if entry.transport!="tls":
            raise MissingPrerequisite("TLS pin mismatch probe requires a paired TLS endpoint")
        fp = bytes.fromhex(entry.fingerprint)
        wrong = bytes([fp[0]^1])+fp[1:]
        try:
            conn = dial_tls(entry.addresses,wrong,timeout=10)
        except ServerKeyMismatch as e:
            return {"wrong_pin_rejected_before_authentication":True,"error":str(e)}
        else:
            conn.close()
            raise AssertionError("wrong server pin accepted")
    dev = ctx.device()
    s = session.by_index(dev.index)
    if what=="unknown_handle":
        def fn():
            return s.fetch_handles([secrets.token_bytes(16)])
        allowed = (cpb.ERROR_UNKNOWN_HANDLE,cpb.ERROR_PERMISSION_DENIED)
    else:
        graph = gpb.Graph(opset_version=1,device=s.gdevice)
        if what=="unknown_op":
            graph.nodes.append(gpb.Node(op_id=0x7fffffff))
        else:
            # Output points past the graph: structural rejection is mandatory.
            graph.outputs.append(gpb.ValueRef(node_output=gpb.NodeOutput(node=999,output=0)))
            graph.output_meta.append(gpb.TensorMeta(dtype=gpb.DTYPE_FLOAT32,shape=[1]))
        def fn():
            return s.conn.call(cpb.Envelope(register_graph=cpb.RegisterGraph(graph=graph)),timeout=30)
        allowed = (cpb.ERROR_INVALID_GRAPH,cpb.ERROR_UNSUPPORTED_VERSION,cpb.ERROR_PROTOCOL)
    try:
        fn()
    except Exception as e:
        assert getattr(e,"code",None) in allowed,str(e)
        detail = {"error":str(e),"code":e.code,"stage":getattr(e,"stage",None)}
        assert torch.ones(4).to(dev).sum().item()==4
        return detail|{"session_healthy_after_rejection":True}
    raise AssertionError("invalid protocol request accepted")
