# reflop

The client for [reFLOP](https://github.com/chaseungjoon/reflop): train with
PyTorch on someone else's GPU from your own machine. Your Python program runs
locally. Only tensor math (as operator graphs) and tensor data go to the
server, and only the values you ask for come back. Your machine needs no CUDA,
so a MacBook works.

## Install

Add it to the environment your training code runs in:

```sh
uv add reflop          # or: pip install reflop
```

That installs both the library (`import reflop`) and the `reflop` command in
that environment. Update with `reflop update`, or `pip install -U reflop`.

Requires Python 3.10+ and PyTorch 2.6+, on macOS or Linux.

## Use

Pair once with the invite code the GPU Owner gives you:

```sh
reflop pair reflop1-…
reflop status
```

Then move your model to the reflop device:

```python
import torch, reflop

dev = reflop.device()          # the GPU on the Owner's workstation
model = MyModel().to(dev)      # weights live on the server
for x, y in loader:
    loss = loss_fn(model(x.to(dev)), y.to(dev))
    loss.backward()
    opt.step(); opt.zero_grad()
    print(loss.item())         # only this number comes back
```

A script written for CUDA runs without changes:

```sh
reflop run python train.py
```

## Links

- [Guest guide](https://github.com/chaseungjoon/reflop/blob/main/docs/GUEST_GUIDE.md)
- [Owner guide](https://github.com/chaseungjoon/reflop/blob/main/docs/OWNER_GUIDE.md) (the server, `reflopd`, is not on PyPI)
- [Changelog](https://github.com/chaseungjoon/reflop/blob/main/CHANGELOG.md)
- [Security](https://github.com/chaseungjoon/reflop/blob/main/SECURITY.md)

Every file on PyPI is the exact file from the matching GitHub release, whose
`checksums.txt` is signed with the project's offline release key.
