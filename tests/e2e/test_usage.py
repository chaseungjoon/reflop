# SPDX-License-Identifier: Apache-2.0
"""A Guest's own compute usage (reflop.usage, `reflop usage`)."""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest


def report():
    from reflop._pb import compute_pb2 as cpb

    return cpb.UsageReport(identity_name="alice\x1b[2J", since_unix=1_700_000_000, executes=4, device_us=90_000_000,
                           flops=3.6e14, peak_memory_bytes=3 << 30, sessions=2, session_us=180_000_000,
                           bytes_up=5 << 20, bytes_down=1024, first_unix=1_700_000_100, last_unix=1_700_000_200,
                           by_device=[cpb.DeviceUsage(device="cuda:0", device_us=89_000_000),
                                      cpb.DeviceUsage(device="cpu", device_us=1_000_000)])


def test_usage_dict_and_lines():
    from reflop.api import _usage_dict
    from reflop.cli.main import _usage_lines

    u = _usage_dict(report())
    assert "\x1b" not in u["identity"]
    assert u["device_seconds"] == 90 and u["session_seconds"] == 180 and u["executes"] == 4
    assert u["by_device"] == {"cuda:0": 89, "cpu": 1}
    assert u["last"] == datetime.fromtimestamp(1_700_000_200, timezone.utc)
    text = "\n".join(_usage_lines(u, now=1_700_000_200 + 3600))
    for want in ("device time:  1m30s (cuda:0 1m29s, cpu 1.0s)", "devices busy 50%", "22500.0 ms average",
                 "360.0 TFLOP (4.0 TFLOP/s", "3.0 GiB", "5.0 MiB up, 1.0 KiB down", "last active 1h00m ago"):
        assert want in text, (want, text)
    small = dict(u, device_seconds=0.0068, by_device={"cpu": 0.0068}, flops=5.4e9, executes=1)
    text = "\n".join(_usage_lines(small))
    assert "device time:  7ms (cpu 7ms)" in text and "5.4 GFLOP (794.1 GFLOP/s" in text, text


def test_usage_on_an_old_server_says_so():
    from reflop._pb import capabilities_pb2 as kpb
    from reflop.api import _usage_on
    from reflop.errors import ReflopError

    class Old:
        caps = kpb.ServerCapabilities(runtime_info=kpb.RuntimeInfo(reflopd_version="v0.1.3"))

    with pytest.raises(ReflopError, match="needs reflopd 0.1.4"):
        _usage_on(Old())


@pytest.mark.skipif(os.environ.get("REFLOP_E2E") != "1", reason="needs a running reflopd (REFLOP_E2E=1)")
def test_usage_from_the_server():
    import reflop

    u = reflop.usage()
    assert u["identity"] and u["since"] is not None
    assert u["device_seconds"] >= 0 and u["executes"] >= 0
