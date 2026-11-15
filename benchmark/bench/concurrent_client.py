from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from .core import distribution, write_json


def main():
    import torch

    from .models import build, optimizer
    from .worker import Context
    req = json.loads(Path(sys.argv[1]).read_text())
    c,path = req["config"],Path(req["path"])
    if c["guest_homes"]:
        c["client_home"] = c["guest_homes"][req["index"]%len(c["guest_homes"])]
    ctx = Context({"case":{"params":{}},"config":c,"run_dir":str(path.parent)},path)
    try:
        torch.set_num_threads(c["cpu_threads"])
        dev = ctx.device()
        model,x,y,loss_fn = build("mlp",32,c["seed"])
        model.to(dev)
        opt = optimizer(model,"momentum")
        x,y = x.to(dev),y.to(dev)
        def step():
            opt.zero_grad()
            loss = loss_fn(model(x),y)
            loss.backward()
            opt.step()
            return loss.item()
        for _ in range(c["warmup_steps"]):
            step()
        write_json(path/"ready.json",{"ready":True})
        deadline = time.monotonic()+100
        while not Path(req["barrier"]).exists():
            if time.monotonic()>deadline:
                raise TimeoutError("parent did not release concurrency barrier")
            time.sleep(.01)
        start = time.perf_counter()
        samples = []
        for _ in range(c["measured_steps"]):
            t0 = time.perf_counter()
            value = step()
            assert torch.isfinite(torch.tensor(value))
            samples.append(time.perf_counter()-t0)
        write_json(path/"client.json",{"steps":len(samples),"seconds":time.perf_counter()-start,
                                      "step_times":distribution(samples),"last_loss":value,"stats":ctx.stats()})
    finally:
        ctx.close()


if __name__=="__main__":
    main()
