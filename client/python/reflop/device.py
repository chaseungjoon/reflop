# SPDX-License-Identifier: Apache-2.0
"""The `reflop` PyTorch device.

PyTorch's PrivateUse1 dispatch key is renamed to "reflop" and set up as a
Python backend, so `torch.device("reflop")`, `model.to("reflop")`,
`torch.randn(..., device="reflop")` and `torch.autocast("reflop")` work.
No CUDA is needed on the client: tensors on this device are RemoteTensor
wrappers whose data lives on the server.
"""

from __future__ import annotations

import threading

import torch

BACKEND = "reflop"
_lock = threading.Lock()
_registered = False


class _BackendModule:
    """Module registered as `torch.reflop` (what PyTorch expects of a backend)."""

    @staticmethod
    def is_available() -> bool:
        from . import session

        return session.has_default()

    @staticmethod
    def device_count() -> int:
        from . import session

        return session.device_count()

    @staticmethod
    def current_device() -> int:
        return 0

    @staticmethod
    def get_amp_supported_dtype() -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @staticmethod
    def _is_in_bad_fork() -> bool:
        return False

    @staticmethod
    def manual_seed(seed: int) -> None:
        # torch.manual_seed(n) also seeds the server session, so client code
        # that seeds for reproducibility keeps working.
        from . import session

        session.seed_if_open(seed)

    manual_seed_all = manual_seed

    @staticmethod
    def synchronize(device=None) -> None:
        from . import session

        session.synchronize()

    @staticmethod
    def is_initialized() -> bool:
        from . import session

        return session.has_default()


def register() -> None:
    """Register the backend once per process (idempotent)."""
    global _registered
    with _lock:
        if _registered:
            return
        name = torch._C._get_privateuse1_backend_name()
        if name not in ("privateuseone", BACKEND):
            raise RuntimeError(
                f"PrivateUse1 is already used by the '{name}' backend in this process; "
                "reflop cannot share it"
            )
        if name == "privateuseone":
            from torch.utils.backend_registration import _setup_privateuseone_for_python_backend

            _setup_privateuseone_for_python_backend(BACKEND, backend_module=_BackendModule())
        from . import (
            autocast,
            tensor,  # noqa: F401  registers factory kernels
        )

        tensor.register_backend_kernels()
        autocast.register()
        _registered = True


def torch_device(index: int = 0) -> torch.device:
    register()
    return torch.device(BACKEND, index)
