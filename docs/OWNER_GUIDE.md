# reFLOP Owner guide

You own the machine with the GPU. This guide covers installing `reflopd`,
inviting Guests, choosing how they connect, the controls you have while they
train, and what to check when something looks wrong.

- [Requirements](#requirements)
- [Install and first start](#install-and-first-start)
- [What `reflopd up` does](#what-reflopd-up-does)
- [Inviting Guests](#inviting-guests)
- [Network exposure](#network-exposure)
- [SSH pipe mode](#ssh-pipe-mode)
- [Level 1: the system install](#level-1-the-system-install)
- [Safety rails and quotas](#safety-rails-and-quotas)
- [Controls while Guests train](#controls-while-guests-train)
- [Monitoring and the audit log](#monitoring-and-the-audit-log)
- [Files and directories](#files-and-directories)
- [Configuration file](#configuration-file)
- [Updating](#updating)
- [Troubleshooting](#troubleshooting)

## Requirements

| | |
|---|---|
| OS | Linux, x86_64 or aarch64, glibc ≥ 2.28 (Ubuntu 20.04+, Debian 11+, RHEL 8+, Fedora, Arch) |
| GPU | Optional. NVIDIA with driver ≥ 580 and a Turing or newer GPU uses PyTorch 2.15 + CUDA 13.2; driver ≥ 525 with a Maxwell to Hopper GPU uses PyTorch 2.14.1 + CUDA 12.6; anything else serves the CPU only |
| Kernel | Landlock (5.13+) and/or seccomp. Both are present on every mainstream distribution kernel |
| Disk | About 5 GB for the CUDA runtime, plus Guest datasets |
| Root | Not needed. `sudo` is used only for the optional Level 1 install |

WSL2 with an NVIDIA GPU works: reflopd detects `/dev/dxg` and the WSL driver
libraries. Containers work if the container gives the process its GPU device
nodes; `reflopd doctor` reports which sandbox layers the container allows.

## Install and first start

```sh
curl -fsSL https://raw.githubusercontent.com/chaseungjoon/reflop/main/install.sh | sh -s -- --server
reflopd up --listen 100.64.0.7:7460
```

Without `--listen`, reflopd binds to `127.0.0.1:7460` and only programs on this
machine can reach it.

## What `reflopd up` does

1. Detects the GPU, the driver, Landlock, seccomp, user namespaces and cgroup
   delegation, and picks a runtime: `cu132`, `cu126` or `cpu`.
2. Downloads that runtime (a standalone CPython and the PyTorch wheels) and
   checks every file against SHA-256 sums compiled into this reflopd build.
   Guests cannot trigger, choose or change this. The download resumes if
   interrupted.
3. Runs the self-test: one sandboxed worker per device, a matmul, a check of
   the result. A GPU whose memory is all in use by other programs is skipped
   with a note, and reflopd starts anyway as long as another device (usually
   the CPU) passed.
4. Installs and starts the `reflopd` systemd user service (`Delegate=yes`, so
   each worker gets its own cgroup for the memory and process rails). Without a
   systemd user manager, or with `--foreground`, it serves in the terminal.
5. On the first run, prints a one-time invite for a Guest named `guest`.

If reflopd should keep running after you log out:

```sh
loginctl enable-linger $USER
```

## Inviting Guests

```sh
reflopd invite --name alice              # valid 24 hours by default
reflopd invite --name alice --expires 2h
reflopd invite --name alice --address gpu.tailnet.example:7460
```

The code carries the address(es), the fingerprint of this server's key, and a
one-time secret. The Guest runs `reflop pair <code>`. Pairing registers their
Ed25519 key; from then on they authenticate by signing a fresh challenge, and
their client refuses any server whose key differs from the pinned one.

An invite works once. A failed attempt (wrong secret, tampered code) also
consumes it, so a leaked code cannot be brute-forced. To see or remove access:

```sh
reflopd users
reflopd revoke alice      # keys stop working, running sessions end now
```

## Network exposure

| Bind | Who can connect | When |
|---|---|---|
| `127.0.0.1:7460` (default) | programs on this machine | testing, SSH pipe mode |
| a Tailscale or WireGuard address | members of your tailnet | recommended |
| a LAN address | the local network | a lab |
| `0.0.0.0:7460` | everyone who can route to you | only behind a firewall you control |

Authentication is required on every bind. TLS 1.3 protects the traffic.

## SSH pipe mode

If `sshd` already runs and you would rather not open a port, Guests can reach
reflopd through SSH. SSH then works as a pipe that can only start
`reflopd serve --stdio`: no shell, no PTY, no port or agent forwarding.

1. The Guest prints their public key: `reflop key --ssh`
2. You register it and add the restricted line to `~/.ssh/authorized_keys`:

   ```sh
   reflopd invite --name bob --via ssh --key "ssh-ed25519 AAAA… bob@laptop" --ssh-target you@gpu-box
   ```

   The line it adds:

   ```
   restrict,command="/home/you/.local/bin/reflopd serve --stdio --identity 09ef…" ssh-ed25519 AAAA… reflop-identity=09ef…
   ```

   A reflopd path or `REFLOP_HOME` with spaces or shell characters is quoted
   in the forced command.

3. The Guest pairs with the code it prints.

The invite also carries this machine's sshd host keys
(`/etc/ssh/ssh_host_*_key.pub`). The Guest's ssh trusts those keys and no
others, whatever the Guest's own `known_hosts` or `ssh_config` say. If the
Guest reaches an sshd somewhere else (a container, a port forward to another
host), pass that sshd's public keys with `--ssh-host-keys '<glob>'`. If you
replace the host keys, Guests need a new invite.

Inside the pipe, the client still authenticates with its key, and the key must
belong to the identity named in the forced command. `reflopd doctor` scans
`authorized_keys` and reports any reflop key missing `restrict` or the forced
command as critical. `reflopd revoke bob` removes the line.

## Level 1: the system install

At Level 0, reflopd and its workers run as you. The sandbox makes an escape
unlikely, but if one happened, it would land in your account, next to your SSH
keys and browser profile. Level 1 moves everything into a dedicated account:

```sh
sudo reflopd install --system --listen 100.64.0.7:7460
sudo reflopd --system invite --name alice
sudo reflopd --system status
```

It creates the system user `reflop` (no login shell, home `/var/lib/reflop`),
installs the binary to `/usr/local/lib/reflop/reflopd`, writes a hardened unit
(`ProtectSystem=strict`, `ProtectHome=yes`, `PrivateTmp`, an empty capability
set, `RestrictAddressFamilies`, `SystemCallFilter`, `DevicePolicy=closed` with
only the NVIDIA nodes allowed, `Delegate=yes`), installs the runtime as `reflop`,
and starts the service. Recommended whenever reflopd is reachable from outside
your own machines.

SSH pipe mode at Level 1 needs a shell for the `reflop` user, because sshd runs
forced commands through the user's shell: `sudo usermod -s /bin/sh reflop`. The
`restrict` line still prevents interactive use.

Remove it with `sudo reflopd uninstall --system` (add `--purge` to delete data
and the user).

## Safety rails and quotas

Rails keep the machine alive. They are always on, and normal training should
never hit them:

| Rail | Default | Effect |
|---|---|---|
| Disk free floor | max(10%, 20 GiB) of the filesystem holding reflop's state | dataset uploads and snapshots that would cross it are refused, partial uploads removed |
| Host RAM floor | 4 GiB stays available to you | workers share a cgroup memory limit that tracks it; without a delegated cgroup, the largest worker is killed when available memory drops below the floor |
| Worker processes/threads | 4096 | `pids.max` per worker (or `RLIMIT_NPROC`) |
| Protocol limits | 64 MiB frames, 1M nodes per graph, 16 dims, 2^40 elements per tensor | malformed or absurd requests are rejected before any work |
| Physical feasibility | estimated peak memory ≤ the GPU's total memory | graphs that cannot fit are rejected without allocating |

```sh
reflopd rails show
reflopd rails set --ram 8GiB --disk 50GiB --disk-percent 15
reflopd rails set --ram 2GiB --force     # below a default: needs --force
```

No value turns a rail off. The minimums are 1% and 1 GiB of disk, 512 MiB of
RAM and 64 processes, and reflopd refuses to start with anything lower in
`reflopd.toml` or `settings.json`. Every protocol limit in `[limits]` has a
range, and 0 is outside all of them.

Quotas are about fairness, and there are none by default: Guests and you agree
on usage among yourselves. When the GPU is full, the Guest whose allocation fails
gets an `OUT_OF_MEMORY` error and their session keeps working. Sessions whose
client disconnected without closing are kept for `[session] ttl`, but the
oldest of them on that GPU is ended (audit event `session_evicted`) when a new
session cannot start or an allocation fails for lack of memory while running.
A graph rejected because it needs more memory than the whole GPU never ends
another session. If you need limits:

```sh
reflopd quota set alice --device-memory 6GiB --max-sessions 1
reflopd quota set alice --execute-wall-ms 60000      # longer executes kill the worker
reflopd quota set alice --storage 50GiB --upload-bandwidth 20MiB
reflopd quota set alice --storage unlimited          # remove one quota
reflopd quota show
reflopd quota clear alice                            # remove all of them
```

A quota of 0 is rejected: an unset quota means no limit, so 0 would read as
"nothing" and act as "everything". To block a Guest, revoke them. The storage
quota counts uploads still in progress as well as stored datasets, and an
upload with no progress for 7 days is deleted.

## Controls while Guests train

```sh
reflopd pause            # new work waits; running work finishes
reflopd pause --now      # also kills running workers (their sessions are lost)
reflopd resume
reflopd kill alice       # end alice's sessions (or pass a session id prefix)
reflopd say "pausing in 10 minutes for a deadline run"
reflopd yield --auto     # pause Guests whenever a non-reflop process uses the GPU
reflopd yield --off
```

Guests see pause, resume and your messages printed in their terminal. These
events are informational only: nothing in them makes a Guest's client do anything.

## Monitoring and the audit log

```sh
reflopd status           # listen address, key, runtime, devices, sandbox, sessions
reflopd top -w           # live: executes, TFLOP, host and GPU memory per session
reflopd usage            # compute used per Guest (device time, FLOPs, traffic)
reflopd audit --name alice --since 2h
```

The audit log records connections, pairing, graph registrations (node count and
an operator histogram), executes (duration, FLOP estimate), transfers,
rejections with their validation stage, rail hits and kills. It never records
tensor contents. Logs are kept 30 days in `~/.local/share/reflop/audit/`.

### Compute used per Guest

```sh
reflopd usage                # lifetime totals for every Guest
reflopd usage alice          # one Guest, with a per-device breakdown
reflopd usage --since 24h    # only the last day (from the audit log)
reflopd usage --json
```

```
compute used since 2026-11-10 21:54, live from reflopd
IDENTITY  DEVICE TIME  SHARE  BUSY  EST. TFLOP  EXECUTES  SESSIONS  SESSION TIME  UP       DOWN       LAST ACTIVE
alice     22m08s       85%    67%   3467.6      196579    12        33m02s        5.6 GiB  408.4 MiB  just now
bob       1m50s        7%     59%   307.9       13481     7         3m06s         339 MiB  998.0 KiB  20m00s ago
```

- **DEVICE TIME** is the main measure: how long the device ran the Guest's
  executes, measured by the worker (the reFLOP equivalent of GPU-hours). It is
  measured rather than estimated and charges memory-bound work fairly.
- **EST. TFLOP** is the server's static FLOP estimate; it undercounts work that
  is limited by memory bandwidth rather than arithmetic.
- **SESSION TIME** is how long the Guest's sessions held a worker and its device
  memory; **BUSY** is device time divided by session time.
- **UP/DOWN** is traffic between the Guest and the server.

Totals are kept per identity name in `usage.json` in the state directory and
survive restarts. The first `reflopd` that has it starts the ledger from what
the audit log still holds. `--since` recomputes from the audit log, so it only
reaches back as far as the log's retention.

Guests see their own lifetime totals with `reflop usage` (and a summary in
`reflop status`); they never see other Guests or their share.

## Files and directories

| Path (Level 0) | Contents |
|---|---|
| `~/.config/reflop/` | `server.key` (mode 0600) and `server.crt`, optional `reflopd.toml` |
| `~/.local/share/reflop/identities/` | paired identities and pending invites |
| `~/.local/share/reflop/store/<identity>/` | Guest datasets, content-addressed |
| `~/.local/share/reflop/snapshots/<identity>/` | Guest snapshots |
| `~/.local/share/reflop/runtime/` | pinned runtimes and the worker package, read-only |
| `~/.local/share/reflop/audit/` | audit log, one JSON-lines file per day |
| `~/.cache/reflop/scratch/` | per-session scratch, deleted when a session ends |
| `$XDG_RUNTIME_DIR/reflop/` | control socket and SSH bridge socket (owner-only) |

Level 1 uses `/etc/reflop`, `/var/lib/reflop` and `/run/reflop`. Guests never
name a path; every location is chosen by reflopd.

## Configuration file

No configuration file is needed. To change defaults, create
`~/.config/reflop/reflopd.toml`; unknown keys are rejected.

```toml
listen = ["100.64.0.7:7460"]
advertise = ["gpu.tailnet.example:7460"]   # addresses written into invites

[rails]
disk_free_floor_percent = 10
disk_free_floor_bytes = "20GiB"
host_ram_floor_bytes = "4GiB"

[devices]
cuda = [0]        # physical GPU indices to expose (default: all)
cpu = true        # also expose the server CPU as a device

[ops]
disabled = ["aten._scaled_mm.default"]
capabilities = { rnn_fused = false, fp8 = true }

[numerics]
tf32_matmul = false
tf32_cudnn = true

[session]
ttl = "30m"       # how long a disconnected session's state is kept; a kept
                  # session is evicted early when a new session needs its GPU memory

[sandbox]
accept_weak_isolation = false
```

## Updating

```sh
reflopd update                    # new binary, verified against the release signature and checksums
reflopd runtime install           # if the release pins a new runtime
reflopd stop && reflopd up
reflopd runtime prune             # remove old runtimes and the download cache
```

Every release since 0.1.6 signs its `checksums.txt` with an offline key.
`install.sh`, `reflopd update` and `reflop update` check that signature
before anything from the release runs, and refuse releases without one, so
they cannot install 0.1.5 or older. `REFLOP_UPDATE_URL` must be an `https://`
URL (plain `http://` works only for `localhost`).

## Troubleshooting

Start with `reflopd doctor`. It explains each sandbox layer, the rails, the
runtime, the network binding and SSH mode in plain language, and exits non-zero
if something is unsafe.

| Symptom | Cause and fix |
|---|---|
| `the cu132 compute runtime is not installed` | run `reflopd up` or `reflopd runtime install` |
| `neither Landlock nor seccomp works` | the kernel or container lacks both; fix that, or start with `--accept-weak-isolation` knowing workers then run unconfined |
| `cgroup rails: not running in a reflopd scope or unit` | you ran `serve` without systemd; rails fall back to `RLIMIT_NPROC` and memory polling. `reflopd up` sets up the service |
| reflopd stops when you log out | `loginctl enable-linger $USER` |
| self-test: `skipped: the GPU has no free memory for a worker` | other programs hold the GPU's memory; `nvidia-smi` lists them. A desktop session (`gnome-shell`, `Xwayland`) holding gigabytes is usually a compositor leak that logging out and back in clears. Guests get `OUT_OF_MEMORY` on that GPU until memory is free; then `reflopd selftest` should pass |
| a Guest gets `SESSION_LOST` | `reflopd audit --name <guest>` shows why: killed, rail, watchdog, or a worker crash |
| `another reflopd is already running` | `reflopd status`, then `reflopd stop` |
| CUDA errors after a driver update | `reflopd selftest`; if the variant changed, `reflopd runtime install` |
