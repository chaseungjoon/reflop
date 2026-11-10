# Security policy

reFLOP's purpose is to let untrusted people run tensor math on someone else's
machine without being able to harm it. Any way for a Guest to execute code,
read or write files, open network connections, read another Guest's data,
disrupt the Owner's machine beyond the documented rails, or attack a Guest's
machine from a server is a security bug. The guarantees are listed in
[docs/THREAT_MODEL.md](docs/THREAT_MODEL.md).

## Reporting

Please report vulnerabilities privately through a
[GitHub security advisory](https://github.com/chaseungjoon/reflop/security/advisories/new),
not in a public issue. Include the reflopd and client versions, the output of
`reflopd doctor`, and the smallest graph or message sequence that shows the
problem.

## Scope

In scope: `reflopd`, the worker, the sandbox, the protocol, the client library,
`libreflop`, and `install.sh`.

Out of scope, as documented limitations: the Owner reading Guest data (the
server is trusted), GPU timing side channels, and bugs in PyTorch, CUDA or the
NVIDIA driver that need no reFLOP-specific trigger (please report those
upstream; we still want to hear if reFLOP's validation could have blocked them).

Every security fix adds a reproducer to `tests/security/`.
