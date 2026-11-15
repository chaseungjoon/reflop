import json
import sys
import time

import torch


def main():
    torch.manual_seed(1729)
    assert torch.cuda.is_available()
    x = torch.randn(8,16).cuda()
    assert x.device.type=="reflop",f"shim failed to map CUDA: {x.device}"
    model = torch.nn.Sequential(torch.nn.Linear(16,32),torch.nn.ReLU(),torch.nn.Linear(32,4)).cuda()
    opt = torch.optim.SGD(model.parameters(),.01)
    samples,losses = [],[]
    for _ in range(8):
        t0 = time.perf_counter()
        opt.zero_grad()
        with torch.autocast("cuda",dtype=torch.bfloat16,enabled=sys.argv[1]=="amp"):
            out = model(x)
            loss = out.square().mean()
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        samples.append(time.perf_counter()-t0)
        losses.append(loss.item())
    if sys.argv[1]=="cuda_api":
        assert torch.cuda.device_count()>=1
        # APIs with local-CUDA dependencies expose useful shim limitations.
        properties = str(torch.cuda.get_device_properties(0))
    else:
        properties = None
    assert all(torch.isfinite(torch.tensor(losses)))
    print(json.dumps({"device":str(x.device),"losses":losses,"step_seconds":samples,"properties":properties}))


if __name__=="__main__":
    main()
