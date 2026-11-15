from __future__ import annotations

# These names are importable by the offline catalog without importing torch.
NAMES = (
    "unary", "trigonometry", "special", "rounding", "binary", "broadcast", "power", "where", "clamp",
    "integer", "boolean", "comparison", "sum_mean", "variance", "extrema", "scan", "logical_reduce",
    "norm", "sort_topk", "matmul", "bmm", "linear", "einsum", "conv1d", "conv2d", "conv3d",
    "conv_transpose1d", "conv_transpose2d", "conv_transpose3d", "depthwise", "maxpool1d", "maxpool2d",
    "maxpool3d", "avgpool1d", "avgpool2d", "avgpool3d", "adaptive_pool", "layer_norm", "group_norm",
    "batch_norm_train", "batch_norm_eval", "instance_norm", "softmax", "cross_entropy", "nll",
    "regression_losses", "binary_losses", "kl_div", "ctc", "embedding", "embedding_bag", "views",
    "noncontiguous", "cat_stack", "split", "expand_repeat", "triangular", "gather", "scatter", "scatter_reduce",
    "index_select", "advanced_index", "masked", "nonzero", "inplace_view", "inplace_out", "copy_alias",
    "factories", "scalar", "empty", "zero_size", "singleton", "high_rank", "dtype_conversion",
    "half", "bf16", "float64", "complex", "channels_last", "interpolate", "pad", "grid_sample",
    "unfold_fold", "sdpa", "sdpa_mask", "fft", "linalg_solve", "linalg_decomposition", "distance",
    "dropout", "random", "histogram", "unique", "searchsorted", "repeat_interleave", "roll_flip",
    "nan_inf", "signed_zero", "subnormal", "large_reduction", "sparse", "quantized", "float8",
)
GRAD_NAMES = ("smooth", "broadcast", "matmul", "conv2d", "embedding", "layer_norm", "sdpa", "view", "loss")


def make(name, seed=1729):
    import torch as t
    import torch.nn.functional as f

    g = t.Generator().manual_seed(seed)

    def r(*s):
        return t.randn(s, generator=g)

    x, y = r(4, 4), r(4, 4)
    recipes = {
        "unary": (lambda a: (a.abs(), a.exp(), a.sigmoid(), a.tanh(), a.square(), a.neg(), a.sign(),
                              a.abs().add(1).log(), a.abs().add(1).sqrt(), a.abs().add(1).rsqrt()), [x]),
        "trigonometry": (lambda a: (a.sin(), a.cos(), a.tan(), a.asin(), a.acos(), a.atan(), a.sinh(),
                                    a.cosh(), a.asinh(), (a + 2).acosh(), a.atanh()), [x.tanh() * .5]),
        "special": (lambda a: (t.erf(a), t.erfc(a), t.erfinv(a.tanh()), t.lgamma(a.abs()+1), t.digamma(a.abs()+1),
                               f.gelu(a), f.gelu(a, approximate="tanh"), f.silu(a), f.mish(a), f.hardswish(a),
                               f.hardsigmoid(a), f.softplus(a), f.softsign(a), f.logsigmoid(a), f.elu(a),
                               f.leaky_relu(a), f.hardtanh(a), f.threshold(a, .1, 0), f.prelu(a, a.new_ones(1))), [x]),
        "rounding": (lambda a: (a.floor(), a.ceil(), a.trunc(), a.round(), a.frac()), [x]),
        "binary": (lambda a, b: (a+b, a-b, a*b, a/(b.abs()+1), t.atan2(a,b), t.maximum(a,b), t.minimum(a,b),
                                  a.fmod(2), a.remainder(2), a.div(2, rounding_mode="floor")), [x,y]),
        "broadcast": (lambda a,b: (a*b + a/(b.abs()+1) - 2.5), [r(3,1,4), r(1,5,4)]),
        "power": (lambda a: (a.abs()**1.5, 2**a, t.pow(a.abs()+1,a))),
        "where": (lambda a,b: t.where(a>0,a,b), [x,y]),
        "clamp": (lambda a: (a.clamp(-.5,.7), a.clamp_min(.1), a.clamp_max(.3))),
        "integer": (lambda a,b: (a+b,a*b,a//b,a%b,a&b,a|b,a^b,~a,a<<1,a>>1),
                    [t.arange(-8,8).reshape(4,4), t.ones(4,4,dtype=t.int64)*3]),
        "boolean": (lambda a,b: (a&b,a|b,~a,t.logical_xor(a,b)), [x>0,y>0]),
        "comparison": (lambda a,b: (a<b,a<=b,a>b,a>=b,a==b,a!=b), [x,y]),
        "sum_mean": (lambda a: (a.sum(), a.mean(1), a.sum((0,1),keepdim=True))),
        "variance": (lambda a: (a.var(0), a.std(1,correction=0), t.var_mean(a))),
        "extrema": (lambda a: (a.amax(1),a.amin(0),a.argmax(),a.argmin(1),a.max(1),a.min(0))),
        "scan": (lambda a: (a.cumsum(0),a.cumprod(1),a.prod(1),t.cummax(a,1),t.cummin(a,0))),
        "logical_reduce": (lambda a: (a.any(),a.all(1)), [x>0]),
        "norm": (lambda a: (a.norm(),t.linalg.vector_norm(a,3),t.logsumexp(a,1))),
        "sort_topk": (lambda a: (a.sort(1),a.topk(2,1),t.kthvalue(a,2),t.argsort(a,stable=True))),
        "matmul": (lambda a,b: (a@b,t.addmm(a,a,b),t.mv(a,b[0]),t.dot(a[0],b[0]),t.outer(a[0],b[0])), [x,y]),
        "bmm": (lambda a,b: (t.bmm(a,b),t.baddbmm(a,a,b)), [r(2,4,4),r(2,4,4)]),
        "linear": (lambda a,b,c: f.linear(a,b,c), [r(3,5),r(4,5),r(4)]),
        "einsum": (lambda a,b: t.einsum("bi,oi->bo",a,b), [r(3,5),r(4,5)]),
        "depthwise": (lambda a,w: f.conv2d(a,w,padding=1,groups=4), [r(2,4,8,8),r(4,1,3,3)]),
        "adaptive_pool": (lambda a: (f.adaptive_avg_pool2d(a,(2,3)),f.adaptive_max_pool2d(a,2)), [r(2,3,8,8)]),
        "layer_norm": (lambda a: f.layer_norm(a,(4,))),
        "group_norm": (lambda a: f.group_norm(a,2), [r(2,4,5,5)]),
        "batch_norm_train": (lambda a,m,v: f.batch_norm(a,m,v,training=True), [r(3,4,5,5),t.zeros(4),t.ones(4)]),
        "batch_norm_eval": (lambda a,m,v: f.batch_norm(a,m,v,training=False), [r(3,4,5,5),t.zeros(4),t.ones(4)]),
        "instance_norm": (lambda a: f.instance_norm(a), [r(2,3,5,5)]),
        "softmax": (lambda a: (a.softmax(-1),a.log_softmax(0))),
        "cross_entropy": (lambda a,b: (f.cross_entropy(a,b),f.cross_entropy(a,b,reduction="none",label_smoothing=.1)),
                          [r(4,5),t.tensor([0,4,1,2])]),
        "nll": (lambda a,b: f.nll_loss(a.log_softmax(1),b,ignore_index=-100), [r(4,5),t.tensor([0,-100,1,2])]),
        "regression_losses": (lambda a,b: (f.mse_loss(a,b),f.l1_loss(a,b),f.smooth_l1_loss(a,b),f.huber_loss(a,b)),[x,y]),
        "binary_losses": (lambda a,b: (f.binary_cross_entropy_with_logits(a,b.sigmoid()),f.binary_cross_entropy(a.sigmoid(),b.sigmoid())),[x,y]),
        "kl_div": (lambda a,b: f.kl_div(a.log_softmax(-1),b.softmax(-1),reduction="batchmean"),[x,y]),
        "ctc": (lambda a,b: f.ctc_loss(a.log_softmax(2),b,[8,8],[3,3]),[r(8,2,5),t.tensor([[1,2,3],[2,3,1]])]),
        "embedding": (lambda a,b: f.embedding(b,a,padding_idx=0),[r(10,4),t.tensor([[0,1,9],[3,3,2]])]),
        "embedding_bag": (lambda a,b,c: f.embedding_bag(b,a,c,mode="mean"),[r(10,4),t.tensor([1,2,3,4]),t.tensor([0,2])]),
        "views": (lambda a: (a.view(2,8),a.permute(1,0),a.transpose(0,1).contiguous(),a.flatten(),a[:,1:3:2],a[1])),
        "noncontiguous": (lambda a: (a.t()*2).sum(0)+a[::2].mean()),
        "cat_stack": (lambda a,b: (t.cat([a,b],1),t.stack([a,b])),[x,y]),
        "split": (lambda a: (a.split([1,3],1),a.chunk(2,0),a.unbind(0))),
        "expand_repeat": (lambda a: (a.expand(3,4,5),a.repeat(2,1,3)),[r(1,4,5)]),
        "triangular": (lambda a: (a.tril(),a.triu(1),t.diag_embed(a[0]),a.diagonal())),
        "gather": (lambda a,i: a.gather(1,i),[x,t.tensor([[0,3],[1,1],[2,0],[3,2]])]),
        "scatter": (lambda a,i,b: (a.scatter(1,i,b),a.scatter_add(1,i,b)),[x,t.tensor([[0,3],[1,1],[2,0],[3,2]]),r(4,2)]),
        "scatter_reduce": (lambda a,i,b: a.scatter_reduce(1,i,b,reduce="mean"),[x,t.tensor([[0,3],[1,1],[2,0],[3,2]]),r(4,2)]),
        "index_select": (lambda a,i: a.index_select(1,i),[x,t.tensor([3,0,2])]),
        "advanced_index": (lambda a,i: (a[i],a[:,i],a[i,i]),[x,t.tensor([3,0,2])]),
        "masked": (lambda a,m: (a.masked_fill(m,0),a[m],a.masked_select(m),a.masked_scatter(m,a.flatten())),[x,x>0]),
        "nonzero": (lambda a: (a>0).nonzero()),
        "inplace_view": (_inplace_view,[x]),
        "inplace_out": (_inplace_out,[x]),
        "copy_alias": (_copy_alias,[x]),
        "factories": (lambda a: (t.zeros(3,4,device=a.device),t.ones_like(a),t.full((2,2),7.,device=a.device),
                                  t.arange(7,device=a.device),t.linspace(0,1,5,device=a.device),t.eye(3,device=a.device))),
        "scalar": (lambda a: (a.sum(),a.reshape(())),[t.tensor(1.5)]),
        "empty": (lambda a: t.empty(16,device=a.device)),
        "zero_size": (lambda a: (a.sum(0),a*2,t.cat([a,a])),[r(0,3)]),
        "singleton": (lambda a: a.squeeze().unsqueeze(0).expand(4,3),[r(1,1,3)]),
        "high_rank": (lambda a: a.sum((0,2,4,6)),[r(1,2,1,2,1,2,1,2)]),
        "dtype_conversion": (lambda a: (a.half().float(),a.bfloat16().float(),a.int(),a.double())),
        "half": (lambda a: (a@a,a.sum()),[x.half()]),
        "bf16": (lambda a: (a@a,a.sum()),[x.bfloat16()]),
        "float64": (lambda a: a@a,[x.double()]),
        "complex": (lambda a: (a@a,a.conj().resolve_conj(),a.abs()),[t.complex(x,y)]),
        "channels_last": (lambda a: (a+a).contiguous(memory_format=t.channels_last),[r(2,3,5,5).contiguous(memory_format=t.channels_last)]),
        "interpolate": (lambda a: (f.interpolate(a,scale_factor=2,mode="nearest"),f.interpolate(a,size=(5,7),mode="bilinear",align_corners=False)),[r(1,2,3,4)]),
        "pad": (lambda a: (f.pad(a,(1,2,0,1)),f.pad(a,(1,1,1,1),mode="reflect"),f.pad(a,(1,1,1,1),mode="replicate")),[r(1,2,4,4)]),
        "grid_sample": (lambda a,b: f.grid_sample(a,b,align_corners=False),[r(1,2,4,4),r(1,3,3,2).tanh()]),
        "unfold_fold": (lambda a: f.fold(f.unfold(a,3,padding=1),(4,4),3,padding=1),[r(1,2,4,4)]),
        "sdpa": (lambda q,k,v: f.scaled_dot_product_attention(q,k,v,is_causal=True),[r(2,2,5,8),r(2,2,5,8),r(2,2,5,8)]),
        "sdpa_mask": (lambda q,k,v,m: f.scaled_dot_product_attention(q,k,v,attn_mask=m),[r(2,2,5,8),r(2,2,5,8),r(2,2,5,8),t.ones(5,5,dtype=t.bool).tril()]),
        "fft": (lambda a: (t.fft.rfft(a),t.fft.irfft(t.fft.rfft(a),n=4),t.fft.fft(a))),
        "linalg_solve": (lambda a,b: (t.linalg.solve(a,b),t.linalg.inv(a),t.linalg.det(a),t.linalg.slogdet(a),t.linalg.pinv(a)),[x@x.T+t.eye(4),y]),
        "linalg_decomposition": (lambda a: (t.linalg.cholesky(a),t.linalg.qr(a),t.linalg.svd(a),t.linalg.eigh(a),t.linalg.lu_factor(a),t.linalg.matrix_exp(a)),[x@x.T+t.eye(4)]),
        "distance": (lambda a,b: (t.cdist(a,b),t.pdist(a)),[x,y]),
        "dropout": (lambda a: f.dropout(a,.3,training=True)),
        "random": (lambda a: (t.rand(8,device=a.device),t.randn(8,device=a.device),t.randint(0,5,(8,),device=a.device),t.randperm(8,device=a.device))),
        "histogram": (lambda a: (t.histc(a,bins=8,min=-3,max=3),t.bincount((a.flatten().abs()*2).long(),minlength=8))),
        "unique": (lambda a: (t.unique(a,return_inverse=True,return_counts=True),t.unique_consecutive(a))),
        "searchsorted": (lambda a,b: t.searchsorted(a,b),[t.arange(8.),r(4)]),
        "repeat_interleave": (lambda a: t.repeat_interleave(a,2,dim=1)),
        "roll_flip": (lambda a: (a.roll(2,-1),a.flip(-1))),
        "nan_inf": (lambda a: (a.isnan(),a.isinf(),t.nan_to_num(a),a+a),[t.tensor([float("nan"),float("inf"),-float("inf"),1.])]),
        "signed_zero": (lambda a: (a*2,t.signbit(a),t.reciprocal(a)),[t.tensor([-0.,0.])]),
        "subnormal": (lambda a: a*2,[t.tensor([1e-40,-1e-40,1e-35])]),
        "large_reduction": (lambda a: (a.sum(),a.mean()),[r(65536)]),
        "sparse": (lambda a: a.to_sparse().to_dense()),
        "quantized": (lambda a: t.quantize_per_tensor(a,.1,0,t.qint8).dequantize()),
        "float8": (lambda a: a.to(t.float8_e4m3fn).float()),
    }
    for dim in (1,2,3):
        args = [r(2,2,*([6]*dim)),r(3,2,*([3]*dim))]
        recipes[f"conv{dim}d"] = (lambda a,w,d=dim: getattr(f,f"conv{d}d")(a,w,padding=1),args)
        recipes[f"conv_transpose{dim}d"] = (lambda a,w,d=dim: getattr(f,f"conv_transpose{d}d")(a,w,stride=2),
                                               [args[0],r(2,3,*([3]*dim))])
        for pool in ("max", "avg"):
            recipes[f"{pool}pool{dim}d"] = (lambda a,d=dim,p=pool: getattr(f,f"{p}_pool{d}d")(a,2),[args[0]])
    value = recipes[name]
    return (value, [x]) if callable(value) else value


def _inplace_view(a):
    a[1:3].mul_(2)
    a[:,0] = 7
    a.t()[2].add_(1)
    a.view(-1)[::3].zero_()
    return a


def _inplace_out(a):
    import torch
    a.mul_(2).add_(1).clamp_(max=2)
    out = torch.empty_like(a)
    torch.add(a,a,out=out)
    return out


def _copy_alias(a):
    b = a.view(-1)
    b[0] = 12
    return a.clone(), b.clone()


def grad_recipe(name, seed):
    import torch as t
    import torch.nn.functional as f
    if name in ("broadcast", "matmul", "conv2d", "layer_norm", "sdpa", "embedding"):
        fn, args = make(name, seed)
        args = [a.requires_grad_() if a.is_floating_point() else a for a in args]
        return fn, args
    x = t.linspace(-.8,.8,12).reshape(3,4).requires_grad_()
    return {"smooth": (lambda a: (a.sin()*a.tanh()).sum(),[x]),
            "view": (lambda a: (a.t()[1:]**2).sum(),[x]),
            "loss": (lambda a: f.mse_loss(a,a.detach()*.5),[x])}[name]
