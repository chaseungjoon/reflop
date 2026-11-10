# SPDX-License-Identifier: Apache-2.0
"""Client configuration: paired servers and the default server.

Stored in ~/.config/reflop/servers.json (mode 0600). Nothing in here is
ever sent to a server (invariant I12).
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path


def config_dir() -> Path:
    if h := os.environ.get("REFLOP_CLIENT_HOME"):
        return Path(h)
    base = os.environ.get("XDG_CONFIG_HOME")
    return Path(base) / "reflop" if base else Path.home() / ".config" / "reflop"


@dataclass
class ServerEntry:
    name: str
    transport: str                     # "tls" or "ssh"
    addresses: list[str]
    fingerprint: str                   # hex SHA-256 of the server SPKI (pinned)
    identity: str                      # our identity name on that server
    identity_id: str = ""
    ssh_port: int | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class ClientConfig:
    default: str = ""
    servers: dict[str, ServerEntry] = field(default_factory=dict)

    @classmethod
    def load(cls) -> ClientConfig:
        p = config_dir() / "servers.json"
        if not p.exists():
            return cls()
        d = json.loads(p.read_text())
        return cls(default=d.get("default", ""),
                   servers={k: ServerEntry(**v) for k, v in d.get("servers", {}).items()})

    def save(self) -> None:
        d = config_dir()
        d.mkdir(parents=True, exist_ok=True, mode=0o700)
        data = json.dumps({"default": self.default, "servers": {k: asdict(v) for k, v in self.servers.items()}},
                          indent=2) + "\n"
        fd, tmp = tempfile.mkstemp(dir=d, prefix=".servers-")
        with os.fdopen(fd, "w") as f:
            f.write(data)
        os.chmod(tmp, 0o600)
        os.replace(tmp, d / "servers.json")

    def get(self, name: str | None) -> ServerEntry:
        name = name or os.environ.get("REFLOP_SERVER") or self.default
        if not name:
            raise RuntimeError("no reflop server is paired; run `reflop pair <invite-code>` first")
        if name not in self.servers:
            raise RuntimeError(f"no paired server named {name!r} (see `reflop servers`)")
        return self.servers[name]
