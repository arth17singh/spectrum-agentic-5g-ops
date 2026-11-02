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


# ------------------------------------------------------------------ role limits (/bb26/team13-trust-gate)
ALLOWED_AC = {"create_policy", "delete_policy", "get_policy", "list_policies"}
GW = "arn:aws:bedrock-agentcore:us-east-2:123456789012:gateway/gw-abc"


class RecordingAC:
    def __init__(self, fail=False):
        self.ops, self.fail, self.policies = [], fail, {}

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)

        def call(**kw):
            self.ops.append(name)
            if name == "list_policies":
                return {"policies": list(self.policies.values())}
            if name == "create_policy":
                if self.fail:
                    raise RuntimeError("AccessDenied")
                p = {"name": kw["name"], "policyId": "id-" + kw["name"], "status": "CREATING"}
                self.policies[p["policyId"]] = p
                return p
            raise AssertionError(f"{name} is outside the gate role")
        return call


def live_gate(thresholds, ac, writer=None):
    pc = PolicyClient(gateway_arn=GW, engine_id="team13_policies-x", ac=ac, dry_run=False, wait_active=False)
    return TrustGate(MemoryTrustStore(), thresholds, pc, writer or EvidenceWriter())


def test_pause_path_uses_only_allowed_policy_calls(thresholds):
    ac = RecordingAC()
    gate = live_gate(thresholds, ac)
    outs = [h.handle(gate, {"record": rec(i, BAD), "model_version": "v1"}) for i in range(4)]
    assert [o["action"] for o in outs].count("pause") == 1
    assert set(ac.ops) <= ALLOWED_AC and "create_policy" in ac.ops  # no waiter polling, no UpdatePolicy


def test_policy_failure_still_returns_trust_values(thresholds):
    gate = live_gate(thresholds, RecordingAC(fail=True))
    outs = [h.handle(gate, rec(i, BAD)) for i in range(3)]
    last = outs[-1]
    assert last["action"] is None and "AccessDenied" in last["error"]
    assert last["trust_after"] is not None and last["e"] is not None
    assert gate.state("m", "v1")["status"] == "active"  # claim released: the next change retries


def test_control_write_failure_keeps_the_pause(thresholds):
    class Denied:
        def append_records(self, *a):
            raise PermissionError("s3:PutObject denied")

    gate = TrustGate(MemoryTrustStore(), thresholds, PolicyClient(), Denied())
    outs = [h.handle(gate, rec(i, BAD)) for i in range(3)]
    assert outs[-1]["action"] == "pause" and "PutObject" in outs[-1]["control_write_error"]
    assert gate.state("m", "v1")["status"] == "paused"


def test_model_version_mismatch_is_reported(thresholds):
    gate = TrustGate(MemoryTrustStore(), thresholds, PolicyClient(), EvidenceWriter())
    out = h.handle(gate, {"record": rec(1, GOOD), "model_version": "v9"})
    assert out["model_version_mismatch"] == {"payload": "v9", "record": "v1"}


def test_evlib_writer_never_reads_s3(monkeypatch):
    import sys
    import types

    class S3Store:
        def __init__(self):
            self.puts = []

        def put_new(self, rel, data):
            if rel in self.puts:
                raise FileExistsError(rel)
            self.puts.append(rel)
            return "s3://b/evidence/" + rel

        def uri(self, rel):
            return "s3://b/evidence/" + rel

        def get(self, rel):
            raise AssertionError("s3:GetObject is outside the gate role")

    store = S3Store()

    def never(*a, **k):
        raise AssertionError("evlib.append_records would read S3 on a repeated key")

    fake = types.SimpleNamespace(S3Store=S3Store, default_store=lambda: store,
                                 _prepare=lambda run_id, kind, recs: [b'{"x":1}\n' for _ in recs],
                                 record_key=lambda line: "k" * 24, append_records=never)
    monkeypatch.setitem(sys.modules, "tools", types.SimpleNamespace(evlib=fake))
    monkeypatch.setitem(sys.modules, "tools.evlib", fake)
    w = h._EvlibWriter()
    a = w.append_records("run-1", "control", [{}])
    b = w.append_records("run-1", "control", [{}])  # same key again: treated as written, no read
    assert a == b == ["s3://b/evidence/records/run-1/control/" + "k" * 24 + ".json"]
