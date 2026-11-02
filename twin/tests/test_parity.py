"""Parity of ReplayTwin with the organisers' local adapter (needs TWINHOST_DIR; skipped otherwise)."""

import json
import os
import sys
from pathlib import Path

import pytest

from twin import parity


def test_compare_detects_differences():
    """Negative control: the comparator is not vacuous."""
    stats = {"fields": 0, "max_rel": 0.0, "worst": None, "mismatches": []}
    parity.compare({"a": [1.0, 2.0], "b": True}, {"a": [1.0, 2.0 * (1 + 1e-5)], "b": False}, "s", stats)
    assert len(stats["mismatches"]) == 2 and stats["worst"] == "s.a[1]"
    stats = {"fields": 0, "max_rel": 0.0, "worst": None, "mismatches": []}
    parity.compare({"a": 2.0}, {"a": 2.0 * (1 + 1e-8)}, "s", stats)  # inside 1e-6
    assert stats["mismatches"] == [] and stats["max_rel"] > 0


@pytest.mark.parity
def test_adapter_vs_replay(tmp_path):
    twinhost = os.environ.get("TWINHOST_DIR")
    if not twinhost or not (Path(twinhost) / "src" / "ran_rl_host").is_dir():
        pytest.skip("TWINHOST_DIR not set to a twin-host checkout")
    if not parity._port_free(8797):
        pytest.skip("port 8797 busy")
    code = parity.main(["--twinhost", twinhost, "--python", sys.executable, "--port", "8797", "--out-dir", str(tmp_path)])
    if code in (2, 3):
        pytest.skip(f"adapter could not start / run (exit {code})")
    report = json.loads(next(tmp_path.glob("parity-*.json")).read_text())
    assert code == 0 and report["passed"]
    assert report["max_rel_diff"] <= 1e-6
    assert report["state_fields_compared"] > 1000
    assert all(c["steps_with_clamping"] > 0 for c in report["cases"])
    assert any(c["steps_with_alarm_freeze"] > 0 for c in report["cases"])
    assert parity._port_free(8797)  # adapter was killed
