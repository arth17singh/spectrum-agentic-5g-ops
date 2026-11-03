import hashlib
import json

import pytest

from t3.trust import policy as pol
from t3.trust.records import (canon_change, change_key, direction_agreement, error_e, make_change_record,
                              make_control_record, parse_change)

K = {"coverage_ratio": 0.6, "sla_ratio": 0.9, "sinr_p10_db": 1.0, "sinr_mean_db": 8.0}


def test_canonical_change_text_a2():
    ch = [{"cell": 2, "d_power_db": -0.0, "d_tilt_deg": 1.04}, {"cell": 0, "d_power_db": 3, "d_tilt_deg": 0}]
    text = canon_change(ch)
    assert text == "cell=0;d_power_db=+3.0;d_tilt_deg=+0.0|cell=2;d_power_db=+0.0;d_tilt_deg=+1.0"
    assert canon_change(parse_change(text)) == text


def test_change_key_matches_kit_formula():
    text = "cell=0;d_power_db=+3.0;d_tilt_deg=+0.0"
    want = "chg-" + hashlib.sha256(json.dumps(["case-1", "step-0", text]).encode()).hexdigest()[:12]
    assert change_key("case-1", "step-0", text) == want


def test_change_record_has_a15_fields_and_exact_error():
    meas = dict(K, coverage_ratio=0.6123456789)
    rec = make_change_record(run_id="r1", case_id="case-1", item_id="step-0",
                             change=[{"cell": 1, "d_power_db": 2, "d_tilt_deg": -1}],
                             twin={"kind": "fake", "version": "1", "scenario": "s", "seed": 0, "step": 1},
                             baseline_kpis=K, measured_kpis=meas,
                             predicted_delta={"coverage_ratio": 0.01, "sla_ratio": 0, "sinr_p10_db": 0, "sinr_mean_db": 0},
                             predictor={"model_id": "m", "version": "v"},
                             scales={"coverage_ratio": 0.05, "sla_ratio": 0.03, "sinr_p10_db": 0.5},
                             deadband={"coverage_ratio": 0.001, "sla_ratio": 0.01, "sinr_p10_db": 0.1}, synthetic=True)
    for f in ("schema_version", "kind", "run_id", "team", "created_at", "aws_time", "aws_time_source", "synthetic",
              "case_id", "change_id", "item_id", "change", "change_text", "twin", "baseline_kpis", "measured_kpis",
              "measured_delta", "predicted_delta", "predictor", "error", "direction_agree", "scored", "probe",
              "trust_before", "trust_after"):
        assert f in rec, f
    assert rec["kind"] == "change" and rec["scored"] is True and rec["probe"] is False
    assert rec["error"]["coverage_ratio"] == abs(0.01 - (0.6123456789 - 0.6))  # never rounded
    assert rec["direction_agree"] is True
    assert set(rec["twin"]) == {"kind", "version", "scenario", "seed", "step"}


def test_probe_and_clipped_records_are_unscored():
    base = dict(run_id="r1", case_id="case-1", item_id="step-0", change=[{"cell": 0, "d_power_db": -1.0, "d_tilt_deg": 0.0}], twin={}, baseline_kpis=K,
                measured_kpis=K, synthetic=True)
    probe = make_change_record(probe=True, **base)
    assert probe["probe"] and probe["scored"] is False and probe["unscored_reason"] == "probe"
    assert "predictor" not in probe and "error" not in probe
    clipped = make_change_record(scored=False, unscored_reason="clipped", **base)
    assert clipped["scored"] is False and clipped["unscored_reason"] == "clipped"


def test_direction_agreement_rules():
    band = {"a": 0.1, "b": 0.1, "c": 0.1}
    assert direction_agreement({"a": 1, "b": 1, "c": 1}, {"a": 1, "b": 1, "c": -1}, band, "abc") is True
    assert direction_agreement({"a": -1, "b": -1, "c": 1}, {"a": 1, "b": 1, "c": 1}, band, "abc") is False
    assert direction_agreement({"a": 1, "b": 1, "c": 1}, {"a": 0.05, "b": 0, "c": 0}, band, "abc") is None
    assert direction_agreement({"a": 1, "b": -1, "c": 1}, {"a": 1, "b": 1, "c": 0}, band, "abc") is None  # tie


def test_error_e_is_mean_of_scaled_abs_errors():
    e = error_e({"a": 0.1, "b": 0.0}, {"a": 0.0, "b": 0.3}, {"a": 0.1, "b": 0.1})
    assert e == pytest.approx((1 + 3) / 2)


def test_control_record_validates_by():
    r = make_control_record(run_id="r", model_id="m", version="v", action="pause", by="trust-gate",
                            reason="x", policy_name="p", time="t", synthetic=True)
    assert r["kind"] == "control"
    with pytest.raises(ValueError):
        make_control_record(run_id="r", model_id="m", version="v", action="pause", by="someone",
                            reason="x", policy_name="p", time="t", synthetic=True)
    with pytest.raises(ValueError):
        make_control_record(run_id="r", model_id="m", version="v", action="stop", by="trust-gate",
                            reason="x", policy_name="p", time="t", synthetic=True)


# ------------------------------------------------------------------ policy
GW = "arn:aws:bedrock-agentcore:us-east-2:123456789012:gateway/gw-abc"


def test_forbid_text_matches_a8():
    s = pol.forbid_statement(GW, pol.PauseTarget(**pol.DEFAULT_TARGET), "v2")
    assert s == ('forbid (principal, action == AgentCore::Action::"ran___predict", '
                 f'resource == AgentCore::Gateway::"{GW}")\n'
                 'unless { context.input has model_version && context.input.model_version != "v2" };')
    plain = pol.forbid_statement(GW, pol.PauseTarget("ran___predict_v2"), "v2")
    assert plain.endswith(f'resource == AgentCore::Gateway::"{GW}");')


def test_forbid_refuses_injection():
    with pytest.raises(ValueError):
        pol.forbid_statement(GW, pol.PauseTarget(**pol.DEFAULT_TARGET), 'v2" || true || "')
    with pytest.raises(ValueError):
        pol.PauseTarget("predict")


def test_policy_names():
    n = pol.policy_name("standin-linear", "v1-flipped", 1762000000)
    assert n.startswith("team13_ranpause_") and len(n) <= 48 and pol.NAME_RE.match(n)
    long = pol.policy_name("a-very-long-model-identifier-for-testing", "2026-11-02-candidate", 1762000000)
    assert long.startswith("team13_ranpause_") and len(long) <= 48
    assert not n.startswith("team13_pause_nlm")


def test_dry_run_client_is_idempotent_and_records_calls():
    pc = pol.PolicyClient(clock=lambda: 100)
    a = pc.pause("m", "v1", "bad")
    b = pc.pause("m", "v1", "bad again")
    assert a["created"] and not b["created"] and a["policy_name"] == b["policy_name"]
    assert pc.is_paused("m", "v1") and not pc.is_paused("m", "v2")
    assert [c["op"] for c in pc.calls] == ["create_policy"]
    kw = pc.calls[0]["kwargs"]
    assert kw["validationMode"] == "IGNORE_ALL_FINDINGS" and "statement" in kw["definition"]["policy"]
    assert pc.resume("m", "v1")["removed"] == [a["policy_name"]]
    assert not pc.is_paused("m", "v1")


class FakeAC:
    def __init__(self):
        self.policies, self.calls = {}, []

    def list_policies(self, **kw):
        return {"policies": list(self.policies.values())}

    def create_policy(self, **kw):
        self.calls.append(("create", kw))
        p = {"name": kw["name"], "policyId": "id-" + kw["name"], "status": "ACTIVE"}
        self.policies[p["policyId"]] = p
        return p

    def delete_policy(self, **kw):
        self.calls.append(("delete", kw))
        self.policies.pop(kw["policyId"])

    def get_waiter(self, name):
        assert name == "policy_active"
        return type("W", (), {"wait": lambda self, **kw: None})()


def test_live_client_uses_injected_client_only():
    with pytest.raises(ValueError):
        pol.PolicyClient(dry_run=False)
    with pytest.raises(ValueError):
        pol.PolicyClient(dry_run=False, ac=FakeAC())  # placeholder gateway refused
    ac = FakeAC()
    pc = pol.PolicyClient(gateway_arn=GW, engine_id="eng-1", ac=ac, dry_run=False, clock=lambda: 5)
    r = pc.pause("m", "v1", "why")
    assert r["created"] and ac.calls[0][1]["policyEngineId"] == "eng-1"
    assert pc.pause("m", "v1", "again")["created"] is False
    pc.resume("m", "v1")
    assert [c[0] for c in ac.calls] == ["create", "delete"]
