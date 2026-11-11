# SPDX-License-Identifier: Apache-2.0
"""reflop-worker: the sandboxed compute process of reFLOP.

Executes validated FlopIR graphs with a static operator table. It never
executes client-supplied code (invariants I1, I2).
"""

__all__ = ["__version__"]
__version__ = "0.1.5"
