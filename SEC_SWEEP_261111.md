# reFLOP security sweep — 2026-11-11

Initial security sweep of reFLOP at `17c0eec` (Release 0.1.5), branch
`security/initial-sweep`. No code was changed; this file is the only output.

## Method

- **Skill:** Trail of Bits `sharp-edges`
  (`npx skills add trailofbits/skills --skill sharp-edges`, installed to
  `~/.claude/skills/sharp-edges`, source commit `e4dd0bb`). The skill looks for
  dangerous defaults, configuration cliffs, silent failures, footgun APIs and
  primitive-vs-semantic confusion, and frames each against three adversaries:
  an attacker who controls input or config, a lazy integrator, and a confused
  integrator. The first choice, Trail of Bits `insecure-defaults`, has been
  removed from their repository.
- **Scope:** `reflopd` (auth, handshake, framing, config, rails, sandbox
  spawn, dataset store, SSH pipe, control sockets, self-update), the worker's
  sandbox step, the Python client (transport, pinning, keys, fetch, update),
  `libreflop` (`clib/go`), and `install.sh`. Reference: `docs/THREAT_MODEL.md`.
- **Depth:** a manual read, not a full audit. Nothing was run against a live
  server, and none of these findings has a reproducer yet. The opset
  validators (`validators.py`, `execute.py`) and the seccomp syscall lists
  were only spot-checked. They need their own pass.

Severity follows the skill's scale, adjusted for exposure: **Critical** means
the default or obvious usage is insecure, **High** means an easy
misconfiguration breaks security, **Medium** means a realistic setup or a
latent defect, and **Low** means it takes deliberate misuse or the impact is
small.

## Summary

| ID | Severity | Area | Finding |
|----|----------|------|---------|
| M1 | Medium | reflopd | Unauthenticated peers get the full 64 MiB frame budget, and pending connections are not counted against `max_connections` |
| M2 | Medium | sandbox | Seccomp fails open if the filter cannot be built; Landlock fails closed |
| M3 | Medium | clients | No cap on total response size (`max_response_bytes` is never used); the server sets the Python client's frame cap |
| M4 | Medium | updates | Releases are checked against a checksum from the same release, with no signature; `REFLOP_UPDATE_URL` accepts any scheme |
| L1 | Low | libreflop | `reflop_last_error()` is documented as thread-local but is one global buffer freed on every call |
| L2 | Low | libreflop | `reflop_upload` truncates `size_t nbytes` to `C.int` |
| L3 | Low | libreflop | `reflop_fetch` reports success when it truncates |
| L4 | Low | store | The storage quota is checked only at `PackBegin`; parallel uploads exceed it; abandoned uploads never expire |
| L5 | Low | config | Zero values silently turn off rails, quotas and audit pruning; most `limits.*` are not validated |
| L6 | Low | clients | SSH mode trusts the user's OpenSSH host-key policy; libreflop's SSH options are weaker than the Python client's |
| L7 | Low | client keys | The macOS Keychain write puts the private key seed on a command line |
| L8 | Low | reflopd | Failed handshakes write audit records with no rate or size limit |
| I1 | Info | sshpipe | Forced-command quoting escapes only `"` |

Nothing Critical or High was found. The core design holds up well under this
lens; see [What held up](#what-held-up).

---

## Medium

### M1. Pre-authentication memory exhaustion in reflopd

**Where**
- `pkg/wire/wire.go:46-47`: `ReadFrame` allocates `make([]byte, n)` for the
  length the peer declares, before any payload arrives.
- `server/reflopd/internal/server/conn.go:54-56`: every connection, including
  one that has not authenticated, gets a reader capped at
  `limits.max_frame_bytes` (64 MiB by default, up to 1 GiB).
- `server/reflopd/internal/server/conn.go:58-62` and `server.go:677-683`:
  `max_connections` is compared with `len(s.conns)`, but a connection joins
  `s.conns` only after `handshake()` succeeds. Connections in the TLS
  handshake or the Hello/Authenticate phase are not counted.
- `server/reflopd/internal/cli/install.go:143-161`: the Level 1 unit sets
  `Delegate=yes` for worker cgroups but has no `MemoryMax`/`MemoryHigh` for
  reflopd itself, and `ramRail` (`server/monitor.go:46`) budgets worker
  memory only.

**Scenario:** any host that can reach the TLS port opens many connections,
finishes TLS, and sends a 4-byte header declaring a 64 MiB frame. reflopd
allocates the buffer and waits up to 15 s for the Hello (`conn.go:114`). If
the attacker then sends the bytes, each connection holds about 64 MiB of RSS
(more while it is decoded). Hundreds of parallel connections can exhaust host
RAM, and the host-RAM rail does not cover reflopd's own memory. No invite or
key is needed.

**Exposure:** the default listener is `127.0.0.1:7460`
(`config.go:184`). Owners who follow the guide and listen on a Tailscale or
LAN address expose the port to every peer on that network.

**Fix**
1. Use a small pre-auth frame cap (Hello, Pair and Authenticate fit in a few
   KiB) and raise it to `max_frame_bytes` only after `AuthResult`.
2. Count connections from accept onward (increment before the handshake
   goroutine starts, decrement on close). Optionally add a per-source-IP cap.
3. In `ReadFrame`, grow the buffer as data arrives (for example
   `io.CopyN` into a `bytes.Buffer` capped at `n`) instead of trusting the
   declared length.
4. Consider a `MemoryMax`/`MemoryHigh` for reflopd in the Level 1 unit.

### M2. Seccomp fails open when the filter cannot be built

**Where**
- `server/reflopd/internal/server/server.go:365-374`: if
  `sandbox.Policy{...}.Filter()` returns an error, the callback logs it and
  returns `nil`. The worker still starts.
- `server/worker/reflop_worker/main.py:88-95`: `require_seccomp` is checked
  only when installing a *non-empty* filter fails. If `seccomp_filter` is
  empty, the worker skips seccomp without error, even when
  `require_seccomp` is true.
- `server/reflopd/internal/supervisor/supervisor.go:311-318`: after `Ready`,
  reflopd does not check `Ready.Sandbox.Seccomp` against `RequireSeccomp`.
- For contrast, Landlock fails closed:
  `sandbox/exec_linux.go:120-123` calls `fail(err)` when
  `RequireLandlock` is set.

**Trigger today:** `Filter()` fails only when the program exceeds 4096
instructions (`sandbox/seccomp_linux.go:120-121`). That is static, so this is
a latent defect, not a live bypass. If a future change to the allow or deny
lists (a new architecture, more syscalls, more `cond` blocks) crosses the
limit, every Guest worker runs with no seccomp filter: socket creation,
`connect`, `ptrace`, `io_uring` and the rest of the deny list become
reachable. The only sign would be a log line and a missing `seccomp` in the
sandbox description.

**Fix:** fail closed in all three places. Abort `spawnWorker` when
`RequireSeccomp` is set and `Filter()` errors. In the worker, call
`fatal(...)` when `require_seccomp` is set and `seccomp_filter` is empty.
After `Ready`, kill the worker if `RequireSeccomp && !Ready.Sandbox.Seccomp`.
A unit test that pins the filter length below 4096 would also catch this.

### M3. Clients have no cap on total response size

The threat model lists *size bombs* as mitigated by "frame size caps on
every response". The per-frame cap exists, but nothing limits how many frames
one response can have.

**Where**
- `client/python/reflop/connection.py:35,40`: `max_response_bytes`
  (16 GiB) is stored but never read anywhere.
- `client/python/reflop/connection.py:73-76`: stream frames are appended to
  a list until one arrives with `end=true`.
- `client/python/reflop/session.py:420-433`: `fetch_handles` concatenates
  every chunk before checking it against the tensor's metadata.
- `clib/go/client.go:254-276` and `clib/go/capi.go:168-170`: libreflop does
  the same, then appends everything into one slice.
- `client/python/reflop/client.py:304-305`: the Python client replaces its
  64 MiB frame cap with `server.max_frame_bytes + 1 MiB`, a value the server
  chooses. A hostile server can advertise `2^64-1` and send 4 GiB frames.
  libreflop keeps its fixed cap (`clib/go/client.go:30,193`).

**Scenario:** a compromised or impersonated server (for example over SSH
mode, see L6) answers a `Fetch` with an endless series of `FetchChunk`
frames with `end=false`. The Guest process grows until it is OOM-killed. This
is the "server attacks the Guest" class the threat model counts as in scope.

**Fix:** track the total bytes for each pending request and fail once it
passes `max_response_bytes`. For fetches, compute the expected size from the
first chunk's `meta` (numel × itemsize) and reject anything past it. Use
`min(server_advertised, client_constant)` as the frame cap, never the server's
value alone.

### M4. Release integrity depends only on the release's own checksum file

**Where**
- `server/reflopd/internal/selfupdate/selfupdate.go:111-137`:
  `checksums.txt` and the archive come from the same GitHub release.
- `install.sh:70-75,220`: the same check in the installer.
- `client/python/reflop/update.py:58-64`: the same check for the client
  wheel.
- `selfupdate.go:202-207`: the downloaded binary is *executed*
  (`<tmp> version`) before it replaces the current one, so the check is
  already the only gate before code runs.
- `selfupdate.go:47-50` and `update.py:16`: `REFLOP_UPDATE_URL` replaces the
  release host with any URL, including `http://`. The Python `_get`
  (`update.py:35-37`) also has no size limit, unlike Go's 256 MiB.

**Scenario:** anyone who can publish a release asset (a stolen GitHub token
or session, a compromised CI job, a malicious workflow change) can replace
both the archive and `checksums.txt`. Every Owner who runs `reflopd update`,
and every new install, then runs attacker code, at Level 0 as the Owner.
SHA-256 here protects against corruption, not against tampering.

**Fix:** sign `checksums.txt` (minisign, signify or Sigstore/cosign) in
`release.yml` and embed the public key in `reflopd`, the client and
`install.sh`. Verify the signature before checking the hash. Allow only
`https://` in `REFLOP_UPDATE_URL`, or put it behind a build tag for tests.
Cap the Python download size.

---

## Low

### L1. `reflop_last_error()` is not thread-local, and the pointer it returns is freed on the next call

- `clib/include/reflop.h:10-11` documents the message as *thread-local*.
- `clib/go/capi.go:42-66`: `errMsg` is one global, and `errC` is one global
  C string that is `free`d and replaced on every call.

Thread A calls `reflop_last_error()` and keeps the pointer. Thread B calls it
and frees that buffer, so A now reads freed memory. Errors also cross
threads: A can read B's message. Any multi-threaded C or C++ host that trusts
the header is affected. **Fix:** make the code match the header (per-thread
storage via a C `__thread` variable), or return a copy the caller frees, or
change the header to say "not thread-safe; the pointer is valid until the
next call".

### L2. `reflop_upload` truncates the byte count

`clib/go/capi.go:146`: `C.GoBytes(data, C.int(nbytes))` narrows `size_t` to
`int`. For 2–4 GiB the value goes negative and GoBytes panics, which aborts
the host process. Above 4 GiB it wraps to a small positive number and uploads
a prefix of the buffer. Whether the server then rejects it depends on its
size check against `meta`. **Fix:** reject `nbytes` above the protocol limit
and check it against shape × dtype size before copying, or use
`unsafe.Slice((*byte)(data), int(nbytes))` after a bounds check.

### L3. `reflop_fetch` reports success when it truncates

`clib/go/capi.go:171-181` copies at most `cap` bytes and returns `0`. The
real size goes to `*nbytes` only if the caller passes a non-NULL pointer.
With `nbytes == NULL`, a short buffer goes unnoticed and the caller reads a
partial tensor as complete. **Fix:** return a distinct error code (for
example `-ERANGE`) when `len(data) > cap`, and require `nbytes`.

### L4. The storage quota can be exceeded, and abandoned uploads stay on disk

- `server/reflopd/internal/store/store.go:134-138`: the quota is checked
  once, in `Begin`, against current usage plus that pack's missing bytes.
- `store.go:192-246`: `PutChunk` checks the disk rail but not the quota.

A Guest sends N `PackBegin` requests for different manifests, each within
quota while nothing has been written yet, and then uploads all of them,
reaching N × quota. Nothing removes uploads that never commit:
`.begin`/`.part`/`.chunks` files are deleted only on `Commit` or a disk-rail
abort (`store.go:248-252,306`). The disk rail still protects the machine, so
this is a fairness issue, not a safety one. **Fix:** check the quota in
`PutChunk`, or reserve the bytes at `Begin` (count every pending manifest's
total). Add a TTL sweep for stale uploads.

### L5. Zero values silently turn protections off

- Rails: `reflopd rails set --ram 0 --disk 0 --disk-percent 0` is accepted.
  `Config.Validate` (`config/config.go:257-274`) checks only the upper bound
  of the disk percent and the minimum `max_pids`. The docs call the rails
  "always on".
- Quotas: `0` means unset (`config.go:114`). An Owner who sets
  `device_memory = 0` to block a Guest gives them unlimited memory instead.
- `audit.retention_days <= 0` means keep everything forever
  (`audit/audit.go:89-92`).
- `limits.max_nodes`, `max_fetch_bytes`, `max_manifest_bytes`,
  `max_handles_per_session`, `max_graphs_per_session`, `max_connections` and
  others have no range checks. What `0` means is decided separately at each
  place the limit is enforced (in Go and in the worker). That should be
  audited, not assumed.
- `ramRail` returns without doing anything when `/proc/meminfo` cannot be
  read (`server/monitor.go:46-49`), a fail-open path. The disk rail fails
  closed.

**Fix:** set minimum floors for the rails (or require `--force` to go below
the defaults). Reject `0` where it is ambiguous, or add an explicit
`unlimited` value. Give every `limits.*` field a range in `Validate`.

### L6. SSH mode depends on the user's OpenSSH host-key policy

- In SSH mode, the server-key check inside the protocol
  (`client/python/reflop/client.py:238,299-302`) compares a value the server
  *sends* with the pinned one. That proves nothing about who is on the other
  end. The real server authentication is OpenSSH's `known_hosts` check.
- The Python client passes `BatchMode=yes`, which stops at an unknown host
  key under OpenSSH's default policy, but it does not force
  `StrictHostKeyChecking=yes`. A user with `StrictHostKeyChecking no` or
  `accept-new` in `~/.ssh/config` will accept an impersonating host.
- Relaying a Guest's signature to the real server does not work: the
  signature is bound to the server fingerprint and the identity in the
  forced command (`auth/identities.go:415`). An impersonating server can
  still return false results and use the client-side attack surface in M3.
- libreflop passes fewer options than the Python client: there is no
  `ForwardX11=no` or `ClearAllForwardings=yes` (`clib/go/client.go:128`
  vs. `client/python/reflop/client.py:156-157`). Forwardings set in the
  user's ssh config therefore apply to reflop connections.
- Both clients run whatever `REFLOP_SSH_COMMAND` names
  (`client.py:152`, `clib/go/client.go:124`).

**Fix:** pass `-o StrictHostKeyChecking=yes`. Ideally, record the server's
SSH host key in the SSH invite and pin it with a reflop-owned
`UserKnownHostsFile`. Give libreflop the same SSH options as the Python
client.

### L7. The macOS Keychain write exposes the key seed in the process list

`client/python/reflop/keys.py:53-56` runs
`security add-generic-password ... -w <base64 seed>`. While that runs, the
private key seed is in argv, which other local users can read with `ps`.
Separately, SSH mode always writes the key to `~/.config/reflop/id_ed25519`
(`keys.py:88-93`), so on macOS the Keychain protection is lost once a Guest
uses SSH mode. **Fix:** send the secret on stdin (`security -i` reads
commands from stdin) or go through the Security framework or the `keyring`
package. Document that SSH mode keeps a file copy of the key.

### L8. Unauthenticated audit log growth

Every failed `Pair` or `Authenticate` writes an audit record
(`server/conn.go:150,167`). The 500 ms delay applies per connection. Since
pre-auth connections are not limited (M1), an attacker who can reach the
port can grow the audit directory without bound, limited only by retention
days, not size. **Fix:** rate-limit failure records per source address,
and/or cap the audit directory size.

---

## Informational

### I1. Forced-command quoting

`server/reflopd/internal/sshpipe/sshpipe.go:63-71` builds
`command="REFLOP_HOME=<h> <reflopd> serve --stdio --identity <id>"` and
escapes only `"`. sshd runs the command through the Owner's shell, so a
`REFLOP_HOME` or reflopd path containing spaces, `;` or `$()` changes the
forced command. Only the Owner controls those values, so this is not
exploitable by a Guest, but a path with spaces would break SSH mode quietly.
Shell-quote both values (single quotes with `'\''` escaping).

### Supply-chain notes (not findings)

- `install.sh` and `reflop update` install the verified wheel with `pip`,
  which resolves dependencies from PyPI without pins or hashes. A
  `--require-hashes` constraints file in the release would close this gap.
- `REFLOP_TEST_PROBE` is read from reflopd's environment in production paths
  (`server/server.go:359`). The worker exits if `_probes` is not packaged,
  so this is safe as long as release builds keep `_probes` out
  (§17.5). A CI check on the release artifact would enforce that.

---

## What held up

These were checked specifically and looked correct:

- **Authentication.** 32-byte `crypto/rand` nonce per connection.
  Domain-separated messages (`reflop-auth-v1\0`, `reflop-pair-v1\0`) bound to
  the server SPKI fingerprint. Ed25519 verification and key lengths checked
  before lookup. Constant-time compares (`subtle`, `hmac.Equal`,
  `hmac.compare_digest`). Revocation is checked on every request
  (`conn.go:209`).
- **Invites.** The invite is consumed on the first attempt that names it,
  whether or not it succeeds. All failure causes return the same error. The
  secret is 32 random bytes, and expiry is checked on both client and server.
- **TLS.** TLS 1.3 only, with ALPN. Both clients replace CA validation with
  an SPKI pin that is checked before any application byte is sent. An empty
  or wrong-length pin is rejected.
- **Strict decoding.** Unknown protobuf fields and enum values are rejected
  at every depth on both sides. TOML and JSON config are decoded with
  unknown keys disallowed.
- **Key files.** The server key and the client key are refused if group or
  other can access them. Writes are atomic with mode 0600.
- **Dataset store.** Every path component is either the server-side identity
  id, a validated hex digest, or a fixed-pattern shard name. No path comes
  from a Guest. Each chunk is length- and SHA-256-checked, and shards are
  re-verified at commit.
- **Local sockets.** Created under umask 0177, chmod 0600, and the peer is
  checked with `SO_PEERCRED` (Owner uid or root) on every accept.
- **Sandbox.** Landlock fails closed when required. The seccomp program
  kills on an architecture mismatch, rejects x32, allows only thread
  `clone`, sends `clone3` to `ENOSYS`, and allows signals only to the
  worker's own pid. Level 1 adds systemd hardening.
- **Worker code.** No `eval`/`exec`/pickle/`importlib`/`subprocess` or
  dynamic dispatch by name in `reflop_worker`. The three `getattr` calls use
  literal attribute names on internal objects.
- **Self-update hygiene.** The tag regex is applied to `/latest` redirects,
  downloads are capped (Go), and the binary is replaced atomically by rename.

## Suggested next steps

1. Fix M1 and M2 first. Both are small, and M2 is a fail-open path in the
   sandbox. Add reproducers under `tests/security/`, as `SECURITY.md`
   requires.
2. Decide on release signing (M4) before the next release goes out.
3. Run the opset validators and meta-kernel checks through a dedicated
   review, for example Trail of Bits `audit-context-building` followed by
   `fp-check`, or fuzzing (the repo already fuzzes framing, safetensors and
   manifests). That surface was out of reach for this sweep.
4. Fix the libreflop ABI issues (L1–L3) before C/C++ users depend on the
   current behavior.
