# SPDX-License-Identifier: Apache-2.0
"""reFLOP client: use a remote GPU as a calculator.

    import reflop
    dev = reflop.device()
    model.to(dev)

Your program runs on this machine. Only tensor math (as operator graphs)
and tensor data go to the server.
"""

from .api import close, device, fetch, placement, seed, stats, step_boundary, synchronize, usage
from .capture import CaptureError, capture
from .errors import OutOfMemoryError, ReflopError, ReflopRemoteError, SessionLostError
from .trace import UnsupportedOperatorError

__version__ = "0.1.10"

__all__ = [
    "device", "capture", "CaptureError", "fetch", "placement", "seed", "step_boundary", "synchronize", "close", "stats", "usage",
    "ReflopError", "ReflopRemoteError", "OutOfMemoryError", "SessionLostError", "UnsupportedOperatorError",
    "dataset", "__version__",
]


def dataset(name: str, *, device=None):
    """A server-resident dataset uploaded with `reflop data push`."""
    from .data.remote import RemoteDataset

    return RemoteDataset(name, device=device)
