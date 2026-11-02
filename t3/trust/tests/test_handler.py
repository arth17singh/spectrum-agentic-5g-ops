import time

import pytest

from t3.trust import handler as h
from t3.trust.evio import EvidenceWriter
from t3.trust.gate import TrustGate
from t3.trust.policy import PolicyClient
from t3.trust.store import MemoryTrustStore

GOOD = {"coverage_ratio": 0.05, "sla_ratio": 0.02, "sinr_p10_db": 0.5}
BAD = {k: -v for k, v in GOOD.items()}


def rec(i, pred):
    return {"kind": "change", "run_id": "run-1", "change_id": f"chg-{i}", "predicted_delta": dict(pred),
            "measured_delta": dict(GOOD), "predictor": {"model_id": "m", "version": "v1"},
            "scored": True, "probe": False}


def test_sync_response_shape_and_pause(thresholds):
    w = EvidenceWriter()
    gate = TrustGate(MemoryTrustStore(), thresholds, PolicyClient(), w)
    r1 = h.handle(gate, {"record": rec(1, GOOD)})
    for k in ("trust_before", "trust_after", "e", "score", "action", "thresholds_sha256", "elapsed_ms"):
        assert k in r1
    assert r1["trust_before"] is None and r1["trust_after"] == pytest.approx(1.0) and r1["action"] is None
    t0 = time.perf_counter()
    outs = [h.handle(gate, rec(i, BAD)) for i in range(2, 6)]
    assert (time.perf_counter() - t0) < 2.0
    paused = [o for o in outs if o["action"] == "pause"]
    assert len(paused) == 1 and paused[0]["policy_name"].startswith("team13_ranpause_")
    assert "control_record_key" in paused[0]
    assert len(w.memory[("run-1", "control")]) == 1


def test_retry_of_the_same_change_is_not_scored_twice(thresholds):
    gate = TrustGate(MemoryTrustStore(), thresholds, PolicyClient(), EvidenceWriter())
    a = h.handle(gate, rec(1, BAD))
    b = h.handle(gate, rec(1, BAD))
    assert b["skipped"].startswith("duplicate") and b["trust_after"] == a["trust_after"]
    assert gate.state("m", "v1")["n"] == 1


def test_non_change_events_are_ignored(thresholds):
    gate = TrustGate(MemoryTrustStore(), thresholds, PolicyClient(), EvidenceWriter())
    assert h.handle(gate, {"kind": "control"})["skipped"]
    assert h.handle(gate, "nope")["skipped"]


def test_handler_builds_a_dry_run_gate_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("T3_TRUST_STORE", f"file:{tmp_path / 'trust.json'}")
    monkeypatch.delenv("T3_TRUST_LIVE", raising=False)
    monkeypatch.delenv("EVIDENCE_S3", raising=False)
    monkeypatch.delenv("EVIDENCE_DIR", raising=False)
    monkeypatch.setattr(h, "_GATE", None)
    out = h.handler({"record": rec(1, GOOD)})
    assert out["trust_after"] == pytest.approx(1.0) and h._GATE.policy_client.dry_run
    assert (tmp_path / "trust.json").exists()
