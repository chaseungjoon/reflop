# SPDX-License-Identifier: Apache-2.0
"""Reproducers for the client-side findings of the 2026-11-11 security sweep
(M3 response bombs, M4 release signatures, L6 SSH host keys, L7 Keychain)."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SIGDATA = ROOT / "server/reflopd/internal/selfupdate/testdata"


# ---------------------------------------------------------------- M3


def _conn():
    from reflop.connection import Connection

    sent = []
    c = Connection(max_response_bytes=1 << 20)
    c._send = sent.append
    return c


def _fetch(c, n=1):
    from reflop._pb import compute_pb2 as cpb

    return c.request_stream(cpb.Envelope(fetch=cpb.Fetch(handles=[b"\x01" * 16] * n)))


def _chunk(rid, index=0, data=b"", offset=0, end=False, shape=None):
    from reflop._pb import compute_pb2 as cpb
    from reflop._pb import graph_pb2 as gpb

    c = cpb.FetchChunk(index=index, offset=offset, data=data)
    if shape is not None:
        c.meta.CopyFrom(gpb.TensorMeta(dtype=gpb.DTYPE_FLOAT32, shape=shape))
    return cpb.Envelope(request_id=rid, end=end, fetch_chunk=c)


def test_endless_fetch_stream_is_cut_off():
    """A server answering Fetch with chunks forever no longer grows the
    client without bound."""
    from reflop.errors import ReflopError

    c = _conn()
    fut = _fetch(c)
    c._deliver(_chunk(1, data=b"x" * 4, shape=[1]))
    for i in range(1, 10_000):
        c._deliver(_chunk(1, data=b"x" * 4096, offset=4 + (i - 1) * 4096))
        if fut.done():
            break
    with pytest.raises(ReflopError, match="more data than"):
        fut.result(0)
    assert not c._pending


def test_fetch_total_capped_by_max_response_bytes():
    from reflop.errors import ReflopError

    c = _conn()
    fut = _fetch(c)
    c._deliver(_chunk(1, shape=[1 << 20]))  # 4 MiB declared, cap is 1 MiB
    with pytest.raises(ReflopError, match="exceeds"):
        fut.result(0)


@pytest.mark.parametrize("bad", ["before_meta", "unrequested", "out_of_order", "short"])
def test_malformed_fetch_streams_rejected(bad):
    from reflop.errors import ReflopError

    c = _conn()
    fut = _fetch(c)
    if bad == "before_meta":
        c._deliver(_chunk(1, data=b"abcd"))
    elif bad == "unrequested":
        c._deliver(_chunk(1, index=5, shape=[1], data=b"abcd", end=True))
    elif bad == "out_of_order":
        c._deliver(_chunk(1, shape=[2]))
        c._deliver(_chunk(1, data=b"abcd", offset=4))
    elif bad == "short":
        c._deliver(_chunk(1, shape=[2], data=b"abcd", end=True))
    with pytest.raises(ReflopError):
        fut.result(0)


def test_valid_fetch_still_works():
    c = _conn()
    fut = _fetch(c, n=2)
    c._deliver(_chunk(1, index=0, shape=[2], data=b"abcd"))
    c._deliver(_chunk(1, index=0, data=b"efgh", offset=4))
    c._deliver(_chunk(1, index=1, shape=[0], end=True))
    assert len(fut.result(0)) == 3


def test_server_cannot_raise_client_frame_cap():
    from reflop.client import MAX_FRAME, MAX_FRAME_CEILING, frame_limit

    assert frame_limit(2**64 - 1) == MAX_FRAME_CEILING
    assert frame_limit(64 << 20) == (64 << 20) + (1 << 20)
    assert frame_limit(0) == MAX_FRAME


# ---------------------------------------------------------------- M4


def test_release_signature_verifies():
    from reflop.update import verify_signature

    key = (SIGDATA / "test-signer.pub").read_text().strip()
    verify_signature((SIGDATA / "checksums.txt").read_bytes(), (SIGDATA / "checksums.txt.sig").read_bytes(), [key])


@pytest.mark.parametrize("bad", ["tampered", "untrusted", "namespace", "garbage"])
def test_bad_release_signatures_rejected(bad):
    from reflop.update import RELEASE_KEYS, SignatureError, verify_signature

    key = (SIGDATA / "test-signer.pub").read_text().strip()
    msg = (SIGDATA / "checksums.txt").read_bytes()
    sig = (SIGDATA / "checksums.txt.sig").read_bytes()
    keys = [key]
    if bad == "tampered":
        msg = msg.replace(b"abc123", b"abc124")
    elif bad == "untrusted":
        keys = list(RELEASE_KEYS)
    elif bad == "namespace":
        sig = (SIGDATA / "wrong-namespace.sig").read_bytes()
    elif bad == "garbage":
        sig = b"-----BEGIN SSH SIGNATURE-----\nU1NIU0lH\n-----END SSH SIGNATURE-----\n"
    with pytest.raises(SignatureError):
        verify_signature(msg, sig, keys)


def test_update_url_must_be_https():
    from reflop.update import check_base_url

    assert check_base_url("https://mirror.example/r/") == "https://mirror.example/r"
    assert check_base_url("http://127.0.0.1:8000/r")
    for bad in ("http://mirror.example/r", "file:///tmp", "ftp://x/y", "mirror.example"):
        with pytest.raises(RuntimeError):
            check_base_url(bad)


# ---------------------------------------------------------------- L6

ED25519 = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAILs7graug4rm/Iz4hvKqazO5MLF8HcSzVedtFmkYjbTH"


def test_ssh_mode_never_accepts_unknown_host_keys(tmp_path, monkeypatch):
    from reflop.client import ssh_command

    monkeypatch.setenv("REFLOP_CLIENT_HOME", str(tmp_path))
    monkeypatch.setenv("REFLOP_SSH_COMMAND", "/usr/bin/ssh")
    legacy = ssh_command("me@gpu", tmp_path / "k", "ab" * 32, [])
    assert "StrictHostKeyChecking=yes" in legacy and "ClearAllForwardings=yes" in legacy
    assert legacy[-3:] == ["--", "me@gpu", "reflop-stdio"]  # a target cannot be read as an option
    pinned = ssh_command("me@gpu", tmp_path / "k", "ab" * 32, [ED25519])
    alias = "reflop-" + "ab" * 32
    assert f"HostKeyAlias={alias}" in pinned and "GlobalKnownHostsFile=/dev/null" in pinned
    kh = tmp_path / "ssh_known_hosts" / ("ab" * 32)
    assert kh.read_text() == f"{alias} {ED25519}\n"
    assert (kh.stat().st_mode & 0o077) == 0


def test_invite_host_keys_validated():
    from reflop._sshhost import normalize

    assert normalize([ED25519 + " root@gpu"]) == [ED25519]
    for bad in (["ssh-rsa " + ED25519.split()[1]], ["ssh-dss AAAA"], ["ssh-ed25519 not-base64!"], [ED25519] * 5):
        with pytest.raises(ValueError):
            normalize(bad)


# ---------------------------------------------------------------- L7


def test_keychain_seed_never_on_command_line(monkeypatch):
    import base64
    import subprocess

    from reflop import keys

    calls = []
    store = {}

    def fake_run(argv, input=None, **kw):
        calls.append((list(argv), input))
        if argv[:2] == ["security", "-i"]:
            store["v"] = input.split(" -w ")[1].strip()
            return subprocess.CompletedProcess(argv, 0, "", "")
        if argv[:2] == ["security", "find-generic-password"]:
            return subprocess.CompletedProcess(argv, 0 if store else 44, store.get("v", "") + "\n", "")
        raise AssertionError(argv)

    monkeypatch.setattr(subprocess, "run", fake_run)
    seed = bytes(range(32))
    assert keys._keychain_put(seed)
    secret = base64.b64encode(seed).decode()
    for argv, _ in calls:
        assert all(secret not in a for a in argv), argv
    assert any(inp and secret in inp for _, inp in calls)
