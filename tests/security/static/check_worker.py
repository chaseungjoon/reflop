#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Static analysis for server-side Python (PLAN.md §17.4, rule R6).

Enforces invariants I1 and I2 over server/worker/reflop_worker: no code
loading, no deserialization of executable formats, no process creation,
no dynamic attribute lookup or import. Exits non-zero on any finding.
Never add exceptions here (R6); the only scoped allowance is ctypes inside
sandbox.py with literal libc symbol names (ADR-004).

    python tests/security/static/check_worker.py [paths...]
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT = [ROOT / "server/worker/reflop_worker"]

BANNED_MODULES = {
    "pickle", "cPickle", "_pickle", "marshal", "dill", "cloudpickle", "shelve", "subprocess", "pty", "multiprocessing",
    "importlib", "imp", "pkgutil", "runpy", "code", "codeop", "cffi", "triton", "torch.utils.cpp_extension",
    "torch.package", "torch._dynamo", "torch._inductor", "torch.jit", "torch.export", "torch.library", "zipimport",
    "webbrowser", "http", "urllib", "requests", "ftplib", "smtplib", "telnetlib", "xmlrpc", "asyncio.subprocess",
}
CTYPES_ALLOWED_IN = {"sandbox.py"}
BANNED_CALLS = {"eval", "exec", "compile", "__import__", "breakpoint", "input"}
BANNED_ATTRS = {
    ("os", "system"), ("os", "popen"), ("os", "fork"), ("os", "forkpty"), ("os", "execv"), ("os", "execve"),
    ("os", "execvp"), ("os", "execvpe"), ("os", "execl"), ("os", "execle"), ("os", "execlp"), ("os", "execlpe"),
    ("os", "spawnl"), ("os", "spawnv"), ("os", "spawnve"), ("os", "posix_spawn"), ("os", "posix_spawnp"),
    ("torch", "load"), ("torch", "compile"), ("jit", "load"), ("export", "load"), ("ops", "load_library"),
    ("torch", "_C._jit_pass"), ("library", "define"), ("library", "impl"), ("Library", "__init__"),
}
DYNAMIC_ATTR_FUNCS = {"getattr", "setattr", "delattr", "hasattr", "__getattribute__", "vars", "globals", "locals"}


def _is_literal(n: ast.AST) -> bool:
    return isinstance(n, ast.Constant) and isinstance(n.value, str)


class Checker(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.findings: list[str] = []

    def bad(self, node: ast.AST, msg: str) -> None:
        self.findings.append(f"{self.path.relative_to(ROOT)}:{getattr(node, 'lineno', 0)}: {msg}")

    def _module(self, node, name: str) -> None:
        top = name.split(".")[0]
        if name in BANNED_MODULES or top in BANNED_MODULES or any(name.startswith(m + ".") for m in BANNED_MODULES):
            self.bad(node, f"import of banned module {name!r} (I1)")
        if top == "ctypes" and self.path.name not in CTYPES_ALLOWED_IN:
            self.bad(node, "ctypes is only allowed in sandbox.py (ADR-004)")

    def visit_Import(self, node: ast.Import) -> None:
        for a in node.names:
            self._module(node, a.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module and node.level == 0:
            self._module(node, node.module)
            for a in node.names:
                self._module(node, f"{node.module}.{a.name}")

    def visit_Call(self, node: ast.Call) -> None:
        f = node.func
        if isinstance(f, ast.Name):
            if f.id in BANNED_CALLS:
                self.bad(node, f"call to {f.id}() (I1)")
            if f.id in DYNAMIC_ATTR_FUNCS and f.id in ("getattr", "setattr", "delattr", "hasattr"):
                if len(node.args) < 2 or not _is_literal(node.args[1]):
                    self.bad(node, f"{f.id}() with a non-literal attribute name (I2)")
            if f.id in ("vars", "globals", "locals"):
                self.bad(node, f"{f.id}() (dynamic namespace access, I2)")
            if f.id == "open" and len(node.args) >= 2 and _is_literal(node.args[1]) and any(
                    c in node.args[1].value for c in "wax+"):
                self.bad(node, "open() for writing: workers write only through fds passed by reflopd")
        if isinstance(f, ast.Attribute):
            base = f.value
            bname = base.id if isinstance(base, ast.Name) else (base.attr if isinstance(base, ast.Attribute) else "")
            if (bname, f.attr) in BANNED_ATTRS:
                self.bad(node, f"call to {bname}.{f.attr} (I1)")
            if f.attr == "__getattribute__":
                self.bad(node, "__getattribute__ call (I2)")
            if bname == "socket" and f.attr == "socket" and not any(k.arg == "fileno" for k in node.keywords):
                self.bad(node, "creating a new socket (only inherited fds may be wrapped)")
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        # torch.ops.aten[<expr>] style dynamic lookups
        v = node.value
        if isinstance(v, ast.Attribute) and v.attr in ("aten", "ops", "prims", "reflop"):
            self.bad(node, "subscript lookup on an operator namespace (I2)")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in ("__dict__", "__class__", "__subclasses__", "__globals__", "__builtins__", "__code__"):
            self.bad(node, f"access to {node.attr} (reflection)")
        self.generic_visit(node)


def check(paths: list[Path]) -> list[str]:
    out: list[str] = []
    for root in paths:
        files = [root] if root.is_file() else sorted(root.rglob("*.py"))
        for p in files:
            if "_pb" in p.parts:
                continue  # protoc output: descriptors only
            c = Checker(p)
            c.visit(ast.parse(p.read_text(), filename=str(p)))
            out.extend(c.findings)
    return out


def main() -> int:
    paths = [Path(a).resolve() for a in sys.argv[1:]] or DEFAULT
    findings = check(paths)
    for f in findings:
        print(f)
    print(f"static check: {len(findings)} finding(s)", file=sys.stderr)
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
