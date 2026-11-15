from __future__ import annotations

import copy

from .worker import compare


def compatibility(ctx):
    import torch
    from torch import nn
    what = ctx.p["what"]
    dev = ctx.device()
    ctx.remote()
    x = torch.linspace(-1,1,128).reshape(8,16)
    if what=="amp_overflow":
        p = nn.Parameter(torch.ones(4,device=dev))
        opt = torch.optim.SGD([p],lr=.1)
        scaler = torch.amp.GradScaler("reflop",init_scale=128.,growth_interval=2)
        loss = (p*float("inf")).sum()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        compare(p,torch.ones(4),ctx.c)
        assert scaler.get_scale()==64.,scaler.get_scale()
        return {"nonfinite_update_skipped":True,"backoff_scale":scaler.get_scale()}
    if what in ("dataloader","dataloader_spawn"):
        dataset = torch.utils.data.TensorDataset(x,torch.arange(8))
        args = {"num_workers":2,"multiprocessing_context":"spawn"} if what=="dataloader_spawn" else {"num_workers":0}
        loader = torch.utils.data.DataLoader(dataset,batch_size=3,shuffle=False,**args)
        output = []
        for data,_labels in loader:
            output.append((data.to(dev)*2).cpu())
        compare(torch.cat(output),x*2,ctx.c)
        return {"samples":8,"num_workers":args["num_workers"],"last_partial_batch_verified":True}
    if what=="custom_autograd":
        class Square(torch.autograd.Function):
            @staticmethod
            def forward(context,a):
                context.save_for_backward(a)
                return a.square()
            @staticmethod
            def backward(context,g):
                a, = context.saved_tensors
                return g*2*a
        a = x.to(dev).requires_grad_()
        Square.apply(a).sum().backward()
        return {"gradient":compare(a.grad,x*2,ctx.c)}
    if what=="gradient_hook":
        a = x.to(dev).requires_grad_()
        token = a.register_hook(lambda g:g*.5)
        a.square().sum().backward()
        token.remove()
        return {"gradient":compare(a.grad,x,ctx.c)}
    model = nn.Sequential(nn.Linear(16,16),nn.Tanh(),nn.Linear(16,16))
    if what=="shared_parameters":
        model[2].weight = model[0].weight
    local = copy.deepcopy(model)
    if what=="export":
        model = torch.export.export(model,(x,)).module()
    elif what=="torchscript":
        model = torch.jit.script(model)
    model.to(dev)
    if what=="compile":
        model = torch.compile(model)
    opt = torch.optim.SGD(model.parameters(),lr=.01,momentum=.9)
    ropt = torch.optim.SGD(local.parameters(),lr=.01,momentum=.9)
    sched = torch.optim.lr_scheduler.StepLR(opt,step_size=2,gamma=.8) if what=="scheduler" else None
    rsched = torch.optim.lr_scheduler.StepLR(ropt,step_size=2,gamma=.8) if sched else None
    losses = []
    for i in range(4):
        opt.zero_grad()
        ropt.zero_grad()
        a = (x+i*.01).to(dev).requires_grad_()
        b = (x+i*.01).requires_grad_()
        if what=="activation_checkpoint":
            from torch.utils.checkpoint import checkpoint
            out = checkpoint(model,a,use_reentrant=False,preserve_rng_state=True)
        else:
            out = model(a)
        expected = local(b)
        loss,ref_loss = out.square().mean(),expected.square().mean()
        loss.backward()
        ref_loss.backward()
        if what=="gradient_clip":
            norm = nn.utils.clip_grad_norm_(model.parameters(),.1)
            ref_norm = nn.utils.clip_grad_norm_(local.parameters(),.1)
            compare(norm,ref_norm,ctx.c)
        compare(a.grad,b.grad,ctx.c)
        compare(loss,ref_loss,ctx.c)
        opt.step()
        ropt.step()
        if sched:
            sched.step()
            rsched.step()
        losses.append(loss.item())
    # Compiled module state keys use an _orig_mod prefix; compare parameters
    # in order so an incidental serialization prefix cannot decide correctness.
    checks = compare(list(model.parameters()),list(local.parameters()),ctx.c)
    return {"losses":losses,"parameters":checks,"feature":what}
