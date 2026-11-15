"""CPU-only, native-PyTorch operator recipe discovery; never imports reFLOP.

Real forwards/backwards provide valid low-level arguments, including saved
workspace tensors. Schema synthesis only supplements those observed recipes.
Every supplement must execute successfully on CPU before it becomes a recipe.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .core import ROOT, read_opset, write_json


def key(name):
    return hashlib.sha256(name.encode()).hexdigest()[:24]


def directory():
    import torch

    from .core import opset_source
    digest = hashlib.sha256(opset_source().read_bytes())
    for name in ("fixtures.py","models.py","operations.py"):
        digest.update((Path(__file__).parent/name).read_bytes())
    return ROOT / "cache" / "operators" / (digest.hexdigest()[:16]+"-"+torch.__version__.replace("/","_"))


def encode(value, tensors):
    import torch
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided or value.is_quantized:
            raise ValueError("non-strided/quantized tensor requires a dedicated workload")
        name = f"t{len(tensors)}"
        tensors[name] = value.detach().cpu().resolve_conj().resolve_neg().contiguous().clone()
        return {"tensor": name, "stride": list(value.stride())}
    if isinstance(value, (list,tuple)):
        return {"tuple" if isinstance(value,tuple) else "list": [encode(x,tensors) for x in value]}
    if isinstance(value,dict):
        return {"dict": {k:encode(v,tensors) for k,v in value.items()}}
    if isinstance(value,torch.dtype):
        return {"dtype": str(value).removeprefix("torch.")}
    if isinstance(value,torch.device):
        return {"device": str(value)}
    if isinstance(value,torch.memory_format):
        return {"memory_format": str(value).removeprefix("torch.")}
    if isinstance(value,torch.layout):
        return {"layout": str(value).removeprefix("torch.")}
    if value is None or isinstance(value,(int,float,str,bool)):
        return value
    raise ValueError(f"unserializable recipe type {type(value)}")


def decode(value, tensors, device="cpu"):
    import torch
    if not isinstance(value,dict):
        return value
    if "tensor" in value:
        source = tensors[value["tensor"]]
        # Values are the contract here; aliasing is tested separately. Preserve
        # non-overlapping strides where possible, without giant allocations.
        stride = value["stride"]
        if all(s>0 for s in stride) and source.numel() and max(stride,default=0) < 1000000:
            out = torch.empty_strided(source.shape,stride,dtype=source.dtype)
            try:
                out.copy_(source)
            except RuntimeError:
                out = source.clone()
        else:
            out = source.clone()
        return out.to(device)
    if "list" in value:
        return [decode(x,tensors,device) for x in value["list"]]
    if "tuple" in value:
        return tuple(decode(x,tensors,device) for x in value["tuple"])
    if "dict" in value:
        return {k:decode(v,tensors,device) for k,v in value["dict"].items()}
    for kind in ("dtype","memory_format","layout"):
        if kind in value:
            return getattr(torch,value[kind])
    if "device" in value:
        return torch.device(device)
    raise ValueError(value)


def resolve(name):
    import torch
    namespace,base,overload = name.split(".")
    return getattr(getattr(getattr(torch.ops,namespace),base),overload)


def synthesize(name,entry):
    import torch as t
    base = name.split(".")[1]
    shape = (4,4)
    positive = t.linspace(.2,.8,16).reshape(shape)
    integral = any(s in base for s in ("bitwise","bincount","lshift","rshift"))
    tensor = t.ones(shape,dtype=t.int64) if integral else positive
    if any(s in base for s in ("acosh",)):
        tensor = tensor+1
    if any(s in base for s in ("linalg","cholesky","triangular_solve")):
        tensor = positive@positive.T+t.eye(4)
    vals = {
        "dim": 1,"dim0": 0,"dim1": 1,"start_dim": 0,"end_dim": -1,"index": 1,
        "k": 2,"n": 4,"p": 2.,"groups": 1,"eps": 1e-5,"momentum": .1,"training": False,
        "threshold": 0.,"value": 0.,"fill_value": 1.,"alpha": 1,"beta": 1,"min": 0.,"max": 1.,
        "size": [4,4],"shape": [4,4],"stride": [4,1],"dims": [1,0],"repeats": [1,2],
        "split_sizes": [1,3],"split_size": 2,"kernel_size": [2,2],"output_size": [2,2],
        "padding": [0,0],"dilation": [1,1],"normalized_shape": [4],"normalized_ndim": 1,
        "keepdim": False,"upper": False,"left": True,"transpose": False,"unitriangular": False,
        "correction": 1,"low": 0,"high": 5,"end": 5,"start": 0,"step": 1,"reduction": 1,
        "normalization": 0,"forward": True,"onesided": True,"last_dim_size": 4,
        "dtype": t.float32,"input_dtype": t.float32,"out_dtype": t.float32,"device": t.device("cpu"),
    }
    if base in ("gather","scatter","scatter_add","scatter_reduce"):
        vals["index"] = t.zeros(shape,dtype=t.int64)
    elif base in ("index_select","embedding"):
        vals["index"] = t.tensor([0,2])
        vals["indices"] = t.tensor([0,2])
    vals["condition"] = tensor>0
    vals["mask"] = tensor>0
    if base.startswith("_foreach_"):
        vals["scalars"] = [1.,2.]
        vals["scalar"] = t.tensor(1.) if name.endswith(".Tensor") else 1.
        if "addcmul" in base or "addcdiv" in base:
            vals["scalars"] = t.tensor([1.,2.]) if name.endswith(".Tensor") else [1.,2.]
        elif name.endswith(".Tensor"):
            vals["other"] = t.tensor(1.)
    if base in ("clamp","clamp_min","clamp_max","masked_fill") and name.endswith(".Tensor"):
        vals.update(min=t.tensor(.1),max=t.tensor(.9),value=t.tensor(.5))
    if base=="lerp":
        vals["end"] = tensor*.5
        vals["weight"] = t.tensor(.3) if name.endswith(".Tensor") else .3
    if base in ("index_add","index_copy","index_fill","take"):
        vals["index"] = t.tensor([0,1,2,3])
    if base=="index_fill" and name.endswith("Tensor"):
        vals["value"] = t.tensor(.5)
    if base in ("index_put","_unsafe_index","_unsafe_index_put"):
        vals["indices"] = [t.tensor([0,1,2,3])]
    if base in ("select_scatter","diagonal_scatter"):
        vals["src"] = t.ones(4)
    if base in ("outer","vdot","dot","mv","addmv"):
        vals["self"] = t.ones(4) if base!="mv" else tensor
        vals["other"] = t.ones(4)
        vals["vec"] = t.ones(4)
        vals["vec2"] = t.ones(4)
    if base in ("addbmm","baddbmm"):
        vals["batch1"],vals["batch2"] = t.ones(2,4,4),t.ones(2,4,4)
    if base=="_int_mm":
        vals["self"],vals["mat2"] = t.ones(4,4,dtype=t.int8),t.ones(4,4,dtype=t.int8)
    if base=="bucketize":
        vals["boundaries"] = t.arange(8.)
    if base in ("imag","real","view_as_real","_fft_c2c"):
        vals["self"] = tensor.to(t.complex64)
    if base=="view_as_complex":
        vals["self"] = t.ones(4,2)
    if base in ("bernoulli","native_dropout"):
        vals["p"] = t.full_like(tensor,.5) if name.endswith(".Tensor") else .5
    if base=="nonzero_static":
        vals["size"] = 16
    if base.startswith("unfold"):
        vals["size"] = 2
    if base=="repeat_interleave" and name.endswith("self_int"):
        vals["repeats"] = 2
    if "native_batch_norm_legit" in base:
        vals.update(input=t.ones(2,4,4,4),running_mean=t.zeros(4),running_var=t.ones(4),
                    training=name.endswith(".no_stats"))
    if base=="_scaled_dot_product_attention_math":
        vals.update(query=t.ones(2,2,4,8),key=t.ones(2,2,4,8),value=t.ones(2,2,4,8))
    if "upsample" in base and "backward" not in base:
        d = 3 if "3d" in base else (2 if "2d" in base else 1)
        vals.update(input=t.ones(1,2,*([4]*d)),self=t.ones(1,2,*([4]*d)),output_size=[6]*d,scale_factors=None)
    if base in ("avg_pool1d","adaptive_avg_pool1d"):
        vals.update(self=t.ones(2,2,4),kernel_size=[2],output_size=[2])
    if base=="embedding_renorm":
        vals.update(indices=t.tensor([0,2]),max_norm=1.,norm_type=2.)
    if base=="affine_grid_generator":
        vals.update(theta=t.eye(2,3).unsqueeze(0),size=[1,2,4,4])
    if base=="diagonal_backward":
        vals.update(grad_output=t.ones(4),input_sizes=[4,4],dim1=0,dim2=1,offset=0)
    if base=="linalg_lu_solve":
        lu,pivots = t.linalg.lu_factor(tensor)
        vals.update(LU=lu,pivots=pivots,B=t.ones(4,4))
    args,kwargs = [],{}
    for arg,typ,kw,has_default,default in entry[3]:
        if arg in ("device",):
            v = t.device("cpu")
        elif arg in vals and (not has_default or base in ("clamp","clamp_min","clamp_max","native_dropout","bernoulli","lerp")):
            v = vals[arg]
        elif has_default:
            # Python's enum defaults are integers in generated schemas.
            v = t.float32 if typ.startswith("ScalarType") and default is not None else default
            if typ.startswith("MemoryFormat") and default is not None:
                v = t.contiguous_format
        elif typ.endswith("?"):
            v = None
        elif typ in ("Tensor[]","Tensor?[]"):
            v = [tensor.clone(),tensor.clone()]
        elif typ == "Tensor":
            v = tensor.clone()
        elif typ.startswith(("int[","SymInt[","float[","bool[")):
            length = re.search(r"\[(\d+)\]",typ)
            v = [1]*(int(length.group(1)) if length else 1)
        elif typ in ("int","SymInt","Scalar"):
            v = 1
        elif typ == "float":
            v = .5
        elif typ == "bool":
            v = False
        elif typ == "str":
            v = entry[5].get(arg,["none"])[0]
        elif typ == "Scalar[]":
            v = [1.,2.]
        else:
            raise ValueError(f"no synthesis rule for {typ}")
        # Argument names are reused for different schema types. Adapt values
        # by type rather than passing a list to a scalar or vice versa.
        if typ in ("int","SymInt") and isinstance(v,list):
            v = v[0]
        if typ.startswith(("int[","SymInt[")) and isinstance(v,int):
            v = [v]
        if kw:
            kwargs[arg] = v
        else:
            args.append(v)
    return tuple(args),kwargs


def prepare():
    import torch
    from safetensors.torch import save_file
    from torch.utils._python_dispatch import TorchDispatchMode

    from .catalog import MODELS
    from .models import build, optimizer
    from .operations import GRAD_NAMES, NAMES, grad_recipe, make

    target = directory()
    if (target/"index.json").exists():
        return json.loads((target/"index.json").read_text())
    target.mkdir(parents=True,exist_ok=True)
    ops = read_opset()
    recorded,errors = {},{}

    def store(name,args,kwargs,source):
        if name not in ops or name in recorded:
            return
        tensors = {}
        spec = {"name":name,"source":source,"args":encode(args,tensors),"kwargs":encode(kwargs,tensors)}
        save_file(tensors,str(target/(key(name)+".safetensors")))
        write_json(target/(key(name)+".json"),spec)
        recorded[name] = {"source":source,"recipe":key(name)}

    class Observe(TorchDispatchMode):
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            kwargs = kwargs or {}
            name = str(func)
            # Snapshot inputs before any mutation. Only retain successful calls.
            if name in ops and name not in recorded:
                try:
                    tensors = {}
                    enc_args,enc_kw = encode(args,tensors),encode(kwargs,tensors)
                    result = func(*args,**kwargs)
                    store(name,decode(enc_args,tensors),decode(enc_kw,tensors),"observed_cpu_dispatch")
                    return result
                except (ValueError,TypeError):
                    pass
            return func(*args,**kwargs)

    torch.set_num_threads(2)
    for name in NAMES:
        try:
            fn,args = make(name)
            with Observe():
                fn(*args)
        except Exception as e:
            errors["workload/"+name] = f"{type(e).__name__}: {e}"
    for name in GRAD_NAMES:
        try:
            fn,args = grad_recipe(name,1729)
            with Observe():
                result = fn(*args)
                vals = result if isinstance(result,(list,tuple)) else [result]
                sum(v.float().sum() for v in vals if isinstance(v,torch.Tensor)).backward()
        except Exception as e:
            errors["gradient/"+name] = f"{type(e).__name__}: {e}"
    # Expand native dispatch coverage for spatial derivatives and overloads.
    import torch.nn.functional as F
    for dim in (1,2,3):
        for mode in ({1:("nearest","nearest-exact","linear","area"),
                      2:("nearest","nearest-exact","bilinear","bicubic","area"),
                      3:("nearest","nearest-exact","trilinear","area")}[dim]):
            try:
                a = torch.randn(2,2,*([4]*dim),requires_grad=True)
                with Observe():
                    extra = {"align_corners":False} if mode in ("linear","bilinear","bicubic","trilinear") else {}
                    F.interpolate(a,size=[6]*dim,mode=mode,**extra).sum().backward()
            except Exception as e:
                errors[f"interpolate/{dim}/{mode}"] = str(e)
        for kind in ("avg","max","adaptive_avg","adaptive_max"):
            try:
                a = torch.randn(2,2,*([4]*dim),requires_grad=True)
                with Observe():
                    getattr(F,f"{kind}_pool{dim}d")(a,2).sum().backward()
            except Exception as e:
                errors[f"pool/{dim}/{kind}"] = str(e)
        try:
            a = torch.randn(2,2,*([4]*dim),requires_grad=True)
            with Observe():
                F.pad(a,[1,1]*dim,mode="reflect").sum().backward()
                F.pad(a,[1,1]*dim,mode="replicate").sum().backward()
        except Exception as e:
            errors[f"pad/{dim}"] = str(e)
    for workload in range(11):
        try:
            a = torch.randn(2,4,4,4,requires_grad=True)
            with Observe():
                if workload==0:
                    F.interpolate(a,size=(6,6),mode="bilinear",align_corners=False,antialias=True).sum().backward()
                elif workload==1:
                    F.group_norm(a,2).sum().backward()
                elif workload==2:
                    F.glu(a).sum().backward()
                elif workload==3:
                    F.embedding_bag(torch.tensor([1,2,3,4]),a.reshape(16,8),torch.tensor([0,2]),mode="mean").sum().backward()
                elif workload==4:
                    F.grid_sample(a,torch.randn(2,3,3,2),align_corners=False).sum().backward()
                elif workload==5:
                    volume = a.unsqueeze(2).expand(2,4,4,4,4).contiguous()
                    F.grid_sample(volume,torch.randn(2,3,3,3,3),align_corners=False).sum().backward()
                elif workload==6:
                    F.pixel_shuffle(a,2)
                    F.pixel_unshuffle(a,2)
                    F.channel_shuffle(a,2)
                elif workload==7:
                    pooled,indices = F.max_pool2d(a,2,return_indices=True)
                    F.max_unpool2d(pooled,indices,2).sum().backward()
                    b = torch.randn(1,2,4,4,4,requires_grad=True)
                    pooled,indices = F.max_pool3d(b,2,return_indices=True)
                    F.max_unpool3d(pooled,indices,2).sum().backward()
                elif workload==8:
                    log_probs = torch.randn(8,2,5).log_softmax(2).requires_grad_()
                    F.ctc_loss(log_probs,torch.tensor([[1,2],[2,3]]),torch.tensor([8,8]),torch.tensor([2,2])).backward()
                elif workload==9:
                    log_probs = torch.randn(8,2,5).log_softmax(2).requires_grad_()
                    F.ctc_loss(log_probs,torch.tensor([[1,2],[2,3]]),[8,8],[2,2]).backward()
                else:
                    weight = torch.randn(8,4)
                    indices,offsets = torch.tensor([1,2,3,4]),torch.tensor([0,2])
                    out,offset2bag,bag_size,max_indices = torch.ops.aten._embedding_bag.default(
                        weight,indices,offsets,False,1,False,None,False,-1)
                    torch.ops.aten._embedding_bag_dense_backward.default(torch.ones_like(out),indices,
                        offset2bag,bag_size,max_indices,8,False,1,None,-1)
        except Exception as e:
            errors[f"extra/{workload}"] = str(e)
    try:
        a,b = torch.randn(4,4,requires_grad=True),torch.randn(3,4,requires_grad=True)
        with Observe():
            torch.pdist(a).sum().backward()
            torch.cdist(a,b).sum().backward()
    except Exception as e:
        errors["distance/backward"] = str(e)
    for name in MODELS:
        try:
            model,x,y,loss = build(name,batch=2)
            for opt_name in ("momentum","adamw","adagrad","rmsprop"):
                opt = optimizer(model,opt_name)
                with Observe():
                    opt.zero_grad()
                    loss(model(x),y).backward()
                    opt.step()
        except Exception as e:
            errors["model/"+name] = f"{type(e).__name__}: {e}"
    for name,entry in ops.items():
        if name in recorded or not name.startswith("aten."):
            continue
        # Internal fused CUDA/RNN workspaces require real kernel-generated
        # inputs. Never invent those or run potentially unsafe low-level calls.
        if any(s in name for s in ("cudnn","_thnn_fused","_scaled_mm","_fused_","_flash_attention",
                                   "_efficient_attention","_cudnn_attention")):
            errors[name] = "requires a device-specific kernel-generated workspace"
            continue
        if entry[1]&512:
            continue
        try:
            args,kwargs = synthesize(name,entry)
            # Validate before storing; don't label a bad CPU fixture a reFLOP failure.
            fn = resolve(name)
            tensors = {}
            a,k = encode(args,tensors),encode(kwargs,tensors)
            torch.manual_seed(1729)
            result1 = fn(*decode(a,tensors),**decode(k,tensors))
            torch.manual_seed(1729)
            result2 = fn(*decode(a,tensors),**decode(k,tensors))
            if not entry[1]&32:
                torch.testing.assert_close(result1,result2,rtol=0,atol=0,equal_nan=True)
            store(name,args,kwargs,"cpu_validated_schema_synthesis")
        except Exception as e:
            errors[name] = f"{type(e).__name__}: {e}"
    index = {"torch_version":torch.__version__,"total_registry_entries":len(ops),"recipes":recorded,
             "uncovered":{n:errors.get(n,"requires a dedicated stateful/native recipe") for n in ops if n not in recorded},
             "discovery_errors":errors}
    write_json(target/"index.json",index)
    return index


def load(name, device="cpu"):
    from safetensors.torch import load_file
    path = directory()
    spec = json.loads((path/(key(name)+".json")).read_text())
    tensors = load_file(str(path/(key(name)+".safetensors")))
    return decode(spec["args"],tensors,device), decode(spec["kwargs"],tensors,device), spec["source"]
