# SPDX-License-Identifier: Apache-2.0
"""reFLOP client: use a remote GPU as a calculator.

    import reflop
    dev = reflop.device()
    model.to(dev)

Your program runs on this machine. Only tensor math (as operator graphs)
and tensor data go to the server.
"""

from .api import (
    checkpoint_tag,
    close,
    delete_snapshot,
    device,
    fetch,
    load,
    placement,
    recover,
    save,
    seed,
    snapshots,
    stats,
    step_boundary,
    synchronize,
    usage,
)
from .capture import CaptureError, capture
from .errors import OutOfMemoryError, ReflopError, ReflopRemoteError, SessionLostError
from .trace import UnsupportedOperatorError

__version__ = "0.2.6"

__all__ = [
    "device", "capture", "CaptureError", "fetch", "placement", "seed", "step_boundary", "synchronize", "close", "stats", "usage",
    "save", "load", "snapshots", "delete_snapshot", "checkpoint_tag", "recover",
    "ReflopError", "ReflopRemoteError", "OutOfMemoryError", "SessionLostError", "UnsupportedOperatorError",
    "dataset", "__version__",
]


def dataset(name: str, *, device=None):
    """A server-resident dataset uploaded with `reflop data push`."""
    from .data.remote import RemoteDataset

    return RemoteDataset(name, device=device)
