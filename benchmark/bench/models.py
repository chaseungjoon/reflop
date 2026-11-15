from __future__ import annotations


def build(name, batch=32, seed=1729):
    import torch
    import torch.nn.functional as F
    from torch import nn

    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed + 1)

    class ConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.BatchNorm2d(16),nn.ReLU(),nn.MaxPool2d(2),
                                     nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1),
                                     nn.Flatten(),nn.Linear(32,10))
        def forward(self,x):
            return self.net(x)

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Conv2d(3,8,3,padding=1)
            self.mid = nn.Conv2d(8,16,3,padding=1)
            self.up = nn.ConvTranspose2d(16,8,2,stride=2)
            self.out = nn.Conv2d(16,4,1)
        def forward(self,x):
            a = self.enc(x).relu()
            b = self.mid(F.max_pool2d(a,2)).relu()
            return self.out(torch.cat([a,self.up(b)],dim=1))

    class Recurrent(nn.Module):
        def __init__(self,kind):
            super().__init__()
            self.rnn = getattr(nn,kind)(16,32,2,batch_first=True)
            self.head = nn.Linear(32,10)
        def forward(self,x):
            return self.head(self.rnn(x)[0][:,-1])

    class Attention(nn.Module):
        def __init__(self,lm):
            super().__init__()
            self.lm = lm
            self.emb = nn.Embedding(128,64) if lm else nn.Linear(16,64)
            self.pos = nn.Embedding(32,64)
            self.ln1 = nn.LayerNorm(64)
            self.qkv = nn.Linear(64,192)
            self.proj = nn.Linear(64,64)
            self.ln2 = nn.LayerNorm(64)
            self.ff = nn.Sequential(nn.Linear(64,128),nn.GELU(),nn.Linear(128,64))
            self.head = nn.Linear(64,128 if lm else 10)
        def forward(self,x):
            b,n = x.shape[:2]
            h = self.emb(x)+self.pos(torch.arange(n,device=x.device))
            q,k,v = self.qkv(self.ln1(h)).chunk(3,dim=-1)
            def heads(a):
                return a.reshape(b,n,4,16).transpose(1,2)
            a = F.scaled_dot_product_attention(heads(q),heads(k),heads(v),is_causal=self.lm)
            h = h+self.proj(a.transpose(1,2).contiguous().reshape(b,n,64))
            h = h+self.ff(self.ln2(h))
            return self.head(h).transpose(1,2) if self.lm else self.head(h[:,-1])

    class LoRA(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = nn.Linear(256,256)
            self.base.requires_grad_(False)
            self.a,self.b = nn.Linear(256,8,bias=False),nn.Linear(8,256,bias=False)
            self.head = nn.Linear(256,10)
        def forward(self,x):
            return self.head((self.base(x)+self.b(self.a(x))).relu())

    if name == "mlp":
        model = nn.Sequential(nn.Linear(256,128),nn.ReLU(),nn.Linear(128,10))
        x = torch.randn(batch,256,generator=g)
    elif name in ("cnn","resnet18","unet"):
        if name == "cnn":
            model = ConvNet()
        elif name == "unet":
            model = UNet()
        else:
            from torchvision.models import resnet18
            model = resnet18(weights=None,num_classes=10)
        x = torch.randn(batch,3,32,32,generator=g)
    elif name in ("lstm","gru"):
        model = Recurrent(name.upper())
        x = torch.randn(batch,16,16,generator=g)
    elif name in ("gpt","transformer"):
        model = Attention(name == "gpt")
        x = torch.randint(0,128,(batch,32),generator=g) if name == "gpt" else torch.randn(batch,32,16,generator=g)
    elif name == "lora":
        model = LoRA()
        x = torch.randn(batch,256,generator=g)
    elif name == "autoencoder":
        model = nn.Sequential(nn.Linear(256,64),nn.GELU(),nn.Linear(64,16),nn.GELU(),nn.Linear(16,256))
        x = torch.randn(batch,256,generator=g)
    else:
        raise ValueError(name)
    if name == "autoencoder":
        y = x.clone()
        loss = F.mse_loss
    else:
        shape = (batch,32,32) if name == "unet" else ((batch,32) if name == "gpt" else (batch,))
        classes = 4 if name == "unet" else (128 if name == "gpt" else 10)
        y = torch.randint(0,classes,shape,generator=g)
        loss = F.cross_entropy
    return model, x, y, loss


def optimizer(model,name,remote=False):
    import torch
    params = [p for p in model.parameters() if p.requires_grad]
    table = {
        "sgd": ("SGD", {"lr": .01}),
        "momentum": ("SGD", {"lr": .01,"momentum": .9}),
        "nesterov": ("SGD", {"lr": .01,"momentum": .9,"nesterov": True}),
        "adam": ("Adam", {"lr": .001}),
        "adamw": ("AdamW", {"lr": .001}),
        "adamw_capturable": ("AdamW", {"lr": .001,"capturable": remote}),
        "adamw_fused": ("AdamW", {"lr": .001,"fused": remote}),
        "sgd_fused": ("SGD", {"lr": .01,"fused": remote}),
        "adagrad": ("Adagrad", {"lr": .01}),
        "adagrad_fused": ("Adagrad", {"lr": .01,"fused": remote}),
        "rmsprop": ("RMSprop", {"lr": .01}),
        "adadelta": ("Adadelta", {"lr": .1}),
        "adamax": ("Adamax", {"lr": .001}),
        "rprop": ("Rprop", {"lr": .01}),
        "asgd": ("ASGD", {"lr": .01}),
        "lbfgs": ("LBFGS", {"lr": .1,"max_iter": 3}),
    }
    cls,kw = table[name]
    return getattr(torch.optim,cls)(params,**kw)
