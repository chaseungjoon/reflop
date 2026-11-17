# SPDX-License-Identifier: Apache-2.0
"""Issue #7: a program that starts the thread pools of torch's OpenMP
runtime and scikit-learn's (a second libomp on macOS), in both orders, with
reflop tensors in between. Run by tests/client/test_openmp_clash.py under
`reflop run`, and once by the macOS CI job without it, to see whether the
clash reproduces there.

It asks for the device itself (in-process worker via REFLOP_LOCAL) rather than
through the shim's "cuda": with the in-process worker, the shim's seeding
re-enters the worker and deadlocks, a test-only path unrelated to OpenMP.
"""

import faulthandler

faulthandler.dump_traceback_later(240, exit=True)  # a hang ends with stacks

import numpy as np  # noqa: E402
import reflop  # noqa: E402
import torch  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402

dev = reflop.device()
rng = np.random.default_rng(0)
x = torch.randn(256, 256)
for i in range(3):
    torch.mm(x, x).sum().item()
    KMeans(n_clusters=4, n_init=1, random_state=i).fit(rng.random((5000, 8)))
    m = torch.nn.Linear(32, 32).to(dev)
    m(torch.randn(16, 32).to(dev)).sum().item()
    torch.nn.functional.conv2d(torch.randn(4, 3, 32, 32), torch.randn(8, 3, 3, 3)).sum().item()
print("openmp-ok")
