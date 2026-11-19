# Isolation probes

The sandbox isolation probes (invariant I5, [PLAN.md §17.5](../../../docs/PLAN.md))
live next to the code that embeds them, because `go:embed` cannot reach this
directory:

| File | What |
|---|---|
| `server/worker/testprobes/_probes.py` | the probes: each tries something the sandbox must block (sockets, reading the Owner's files, exec, fork, ptrace, mount, unshare, setuid, …) |
| `server/bundle_probes.go` | embeds them, only with `-tags reflop_probes`; release builds never contain them |
| `server/reflopd/internal/server/isolation_test.go` | starts one fully sandboxed worker per probe and checks it was blocked |
| `.github/workflows/isolation.yml` | runs them in CI (sandbox PRs, pushes to `main`, nightly) |

Run them by hand:

```sh
make build
REFLOP_HOME=/tmp/rf ./bin/reflopd runtime install --variant cpu
REFLOP_TEST_HOME=/tmp/rf make test-isolation
```

`REFLOP_TEST_STRICT=1` turns every skip into a failure: no runtime, no
Landlock, no seccomp, a probe target that does not exist, or a user namespace
mode other than `REFLOP_TEST_USERNS` (`0` off, `1` on). CI always sets it.
