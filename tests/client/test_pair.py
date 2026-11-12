# SPDX-License-Identifier: Apache-2.0
"""Issue #5: `reflop pair` with an expired invite says when it expired and
what this machine's clock says, so a wrong clock can be told apart."""

import base64

import pytest
from reflop import client
from reflop._pb import auth_pb2 as apb
from reflop.errors import ReflopError


def invite_code(expires_unix: int) -> str:
    inv = apb.InviteCode(version=1, transport=apb.TRANSPORT_TLS, addresses=["127.0.0.1:7460"],
                         server_key_fingerprint=b"\x01" * 32, invite_id=b"\x02" * 16, secret=b"\x03" * 32,
                         expires_unix=expires_unix, identity_name="guest")
    return client.INVITE_PREFIX + base64.b32encode(inv.SerializeToString()).decode().rstrip("=").lower()


def test_expired_invite_shows_both_times(monkeypatch):
    expires = 1_794_571_200  # 2026-11-13 12:00:00 UTC
    monkeypatch.setattr(client.time, "time", lambda: expires + 40 * 60 + 7)
    monkeypatch.setattr(client.keys, "load_or_create", lambda: pytest.fail("touched the key"))
    with pytest.raises(ReflopError) as e:
        client.pair(invite_code(expires))
    msg = str(e.value)
    assert "expired at 2026-11-13 12:00:00 UTC" in msg
    assert "clock says 2026-11-13 12:40:07 UTC (40m later)" in msg
    assert "date -u" in msg


def test_expired_message_formats_long_and_short_gaps():
    assert "(1d 2h 3m later)" in client._expired_message(0, 86400 + 2 * 3600 + 3 * 60 + 59)
    assert "(5s later)" in client._expired_message(0, 5.9)
