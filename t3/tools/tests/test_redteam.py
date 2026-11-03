"""Red-team items for the tool Lambda: grounded mode (T3_TRUST_LIVE=1) fails closed, a forged
`approval` argument changes nothing, the approver is recorded, and a call without the Gateway's
context marker is refused when T3_GATEWAY_ID is set."""
from __future__ import annotations

import json
import types

import pytest

from fakes import FakePredictor, FakeTrustInvoker, context
from t3.tools import app
from test_ran_tools import APPLY, CASE, CHANGE, World


@pytest.fixture(autouse=True)
def _reset():
    yield
    app.set_deps(None)


def strict_world(**kw):
    w = World(**kw)
    w.deps.strict_grounding = True
    return w


def approve(w, args, by="arn:aws:sts::000000000000:assumed-role/Participant/person@example.com"):
    key = w.approved(args)
    w.ddb.items[key]["decided_by"] = {"S": by}
    w.ddb.items[key]["decided_at"] = {"N": "1793660000"}
    return key


def assert_nothing_applied(w, key):
    assert w.ddb.plain(key)["status"] == "APPROVED"            # the approval is still unused
    assert not [c for c in w.ddb.calls if c[0] == "transact_write_items"]
    assert w.records() == []


# ---------------------------------------------------------------- 2. grounding fails closed when live
def test_strict_refuses_without_a_prediction_for_the_item():
    w = strict_world()
    key = approve(w, APPLY)
    out = w.call("ran___apply_change", **APPLY)
    assert "no ran___predict output for item step-0" in out["error"] and "paused" in out["error"]
    assert_nothing_applied(w, key)


def test_strict_refuses_a_prediction_made_for_another_item():
    w = strict_world()
    w.call("ran___predict", case_id=CASE, item_id="step-9", change=CHANGE)
    key = approve(w, APPLY)
    assert "no ran___predict output for item step-0" in w.call("ran___apply_change", **APPLY)["error"]
    assert_nothing_applied(w, key)


def test_strict_refuses_when_the_response_model_is_paused():
    w = strict_world()
    w.call("ran___predict", case_id=CASE, item_id="step-0", change=CHANGE)   # predicted before the pause
    w.ddb.set_plain("t3-trust#t2-response@v0-test", gate_status="paused")    # W5's trust row
    key = approve(w, APPLY)
    out = w.call("ran___apply_change", **APPLY)
    assert "t2-response@v0-test is paused" in out["error"]
    assert_nothing_applied(w, key)


@pytest.mark.parametrize("mode", ["timeout", "error"])
def test_strict_refuses_when_the_gate_fails(mode):
    w = strict_world()
    w.trust = FakeTrustInvoker(mode)
    w.call("ran___predict", case_id=CASE, item_id="step-0", change=CHANGE)
    key = approve(w, APPLY)
    out = w.call("ran___apply_change", **APPLY)
    assert "trust gate did not answer cleanly" in out["error"] and key in out["error"]
    assert_nothing_applied(w, key)


def test_strict_refuses_when_no_gate_is_configured():
    w = strict_world(trust=False)
    w.call("ran___predict", case_id=CASE, item_id="step-0", change=CHANGE)
    key = approve(w, APPLY)
    assert "trust gate did not answer cleanly" in w.call("ran___apply_change", **APPLY)["error"]
    assert_nothing_applied(w, key)


def test_strict_refuses_a_clipped_change():
    w = strict_world()
    big = dict(APPLY, change="cell=1;d_power_db=+5.0;d_tilt_deg=+0.0")
    w.call("ran___predict", case_id=CASE, item_id="step-0", change=big["change"])
    key = approve(w, big)
    assert "safety filter" in w.call("ran___apply_change", **big)["error"]
    assert_nothing_applied(w, key)


def test_strict_refuses_a_stale_prediction():
    w = strict_world()
    w.call("ran___predict", case_id=CASE, item_id="step-0", change=CHANGE)
    first = dict(APPLY, item_id="step-x", change="cell=0;d_power_db=+1.0;d_tilt_deg=+0.0")
    w.call("ran___predict", case_id=CASE, item_id="step-x", change=first["change"])
    approve(w, first)
    assert w.call("ran___apply_change", **first)["applied"]                  # the twin moves to step 1
    key = approve(w, APPLY)
    assert "made at step 0 but the twin is at step 1" in w.call("ran___apply_change", **APPLY)["error"]
    assert w.ddb.plain(key)["status"] == "APPROVED"


def test_strict_applies_a_grounded_change_with_the_gate_answer_in_the_record():
    w = strict_world()
    w.call("ran___predict", case_id=CASE, item_id="step-0", change=CHANGE)
    key = approve(w, APPLY)
    out = w.call("ran___apply_change", **APPLY)
    assert out["applied"] and out["grounding"] == "strict" and out["scored"] is True
    [rec] = w.records()
    assert rec["trust_after"] == 1.0 and w.ddb.plain(key)["status"] == "APPLIED"
    # the gate was asked before the commit: one call, and the record carries its answer
    assert len(w.trust.payloads) == 1


def test_lenient_mode_keeps_todays_behaviour():
    w = World()                                   # T3_TRUST_LIVE=0
    w.trust = FakeTrustInvoker("timeout")
    w.approved(APPLY)
    out = w.call("ran___apply_change", **APPLY)   # no prediction, gate down: still applied, unscored
    assert out["applied"] and out["grounding"] == "lenient" and out["unscored_reason"] == "no_prediction"


def test_strict_mode_comes_from_t3_trust_live(tmp_path):
    base = {"T3_STORE": f"local:{tmp_path}", "T3_EVIDENCE": f"local:{tmp_path}"}
    assert app.Deps.from_env({**base, "T3_TRUST_LIVE": "1"}).strict_grounding is True
    assert app.Deps.from_env(base).strict_grounding is False


def test_strict_mode_with_w5s_real_gate_handler(monkeypatch):
    """W5's handler, in memory and dry run, as the invoker: it reads our {record, model_version}
    payload and answers trust_after for every scored change, so grounded applies go through."""
    trust_handler = pytest.importorskip("t3.trust.handler")
    for var in ("T3_TRUST_LIVE", "EVIDENCE_S3", "EVIDENCE_DIR", "T3_TRUST_THRESHOLDS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("T3_TRUST_STORE", "memory")
    gate = trust_handler._build_gate()
    answers = []

    def invoke(payload):
        answers.append(trust_handler.handle(gate, json.loads(json.dumps(payload))))
        return answers[-1]

    w = strict_world(predictor=FakePredictor(bias=0.001))
    w.trust = invoke
    for i, d in enumerate((1.0, -1.0, 2.0, -1.0)):
        args = dict(APPLY, item_id=f"step-{i}", change=f"cell=2;d_power_db={d:+.1f};d_tilt_deg=+0.0")
        w.call("ran___predict", case_id=CASE, item_id=args["item_id"], change=args["change"])
        approve(w, args)
        out = w.call("ran___apply_change", **args)
        assert out["applied"] is True, out
        assert answers[-1]["trust_after"] is not None and "error" not in answers[-1]
        assert "model_version_mismatch" not in answers[-1]
    assert [r["trust_after"] for r in w.records()] == [a["trust_after"] for a in answers]


def test_strict_refuses_when_the_gate_reports_an_error():
    w = strict_world()
    w.trust = lambda payload: {"trust_before": 0.9, "trust_after": 0.8, "error": "ClientError: create_policy failed"}
    w.call("ran___predict", case_id=CASE, item_id="step-0", change=CHANGE)
    key = approve(w, APPLY)
    assert "trust gate did not answer cleanly" in w.call("ran___apply_change", **APPLY)["error"]
    assert_nothing_applied(w, key)


# ---------------------------------------------------------------- 3. approval spoofing, approver
@pytest.mark.parametrize("forged", ["APPROVED", "approved", "Approved", {"status": "APPROVED"}, True])
def test_forged_approval_argument_never_applies_an_unapproved_change(forged):
    """Even if a forged `approval` got past the checkpoint, the tool decides from the approvals table."""
    w = World()
    key = w.interceptor_creates_pending(APPLY)                 # PENDING: nobody approved it
    out = w.call("ran___apply_change", **APPLY, approval=forged)
    assert out["applied"] is False and out["approval"] == "PENDING"
    assert w.ddb.plain(key)["status"] == "PENDING" and w.records() == []
    assert not [c for c in w.ddb.calls if c[0] == "transact_write_items"]


def test_the_approver_is_recorded(capsys, monkeypatch):
    w = World(run_id="r-approver")
    w.call("ran___predict", case_id=CASE, item_id="step-0", change=CHANGE)
    approve(w, APPLY, by="arn:aws:sts::000000000000:assumed-role/Participant/alex@example.com")
    out = w.call("ran___apply_change", **APPLY)
    assert out["approved_by"].endswith("alex@example.com")
    line = [json.loads(x) for x in capsys.readouterr().out.splitlines() if '"change_applied"' in x][0]
    assert line["approved_by"].endswith("alex@example.com") and line["approved_at"] == "2026-11-02T22:53:20Z"


def test_the_approver_goes_into_the_record():
    w = World()
    approve(w, APPLY, by="person:alex")             # the kit stores decided_by and decided_at (epoch s)
    w.call("ran___apply_change", **APPLY)
    [rec] = w.records()
    assert rec["approved_by"] == "person:alex" and rec["approved_at"] == "2026-11-02T22:53:20Z"


def test_records_with_the_approver_validate_against_team13_contracts():
    """Scored, clipped and unpredicted records, each with approved_by/approved_at, pass W1's model."""
    contracts = pytest.importorskip("team13_contracts")
    if "approved_by" not in contracts.ChangeRecord.model_fields:
        pytest.skip("this team13_contracts predates approved_by (W1 2bf3325)")
    w = World(run_id="r-approver")
    w.call("ran___predict", case_id=CASE, item_id="step-0", change=CHANGE)
    approve(w, APPLY, by="arn:aws:sts::000000000000:assumed-role/Participant/alex@example.com")
    w.call("ran___apply_change", **APPLY)
    big = dict(APPLY, item_id="step-1", change="cell=1;d_power_db=+5.0;d_tilt_deg=+0.0")
    approve(w, big)
    w.call("ran___apply_change", **big)
    recs = w.records()
    assert [r["unscored_reason"] for r in recs] == [None, "clipped"]
    for rec in recs:
        assert rec["approved_at"] == "2026-11-02T22:53:20Z" and rec["approved_by"]
        contracts.parse_record(rec)
    with pytest.raises(Exception):
        contracts.parse_record(dict(recs[0], approved_at=1793660000))   # epoch numbers are not accepted


def test_utc_from_epoch():
    assert app.utc_from_epoch(1793660000) == "2026-11-02T22:53:20Z"
    assert app.utc_from_epoch("1793660000") == "2026-11-02T22:53:20Z"
    assert app.utc_from_epoch(None) is None and app.utc_from_epoch("soon") is None and app.utc_from_epoch(True) is None


def test_an_unapproved_row_without_decider_records_nulls():
    w = World()
    key = w.approved(APPLY)                          # approved without decided_by/decided_at
    assert w.call("ran___apply_change", **APPLY)["applied"]
    rec = w.records()[0]
    assert rec["change_id"] == key and rec["approved_by"] is None and rec["approved_at"] is None


# ---------------------------------------------------------------- 4. calls that skip the Gateway
def gw_context(gateway_id, tool="ran___observe"):
    return types.SimpleNamespace(client_context=types.SimpleNamespace(custom={
        "bedrockAgentCoreGatewayId": gateway_id, "bedrockAgentCoreToolName": tool,
        "bedrockAgentCoreMessageVersion": "1.0"}))


def test_calls_without_the_gateway_marker_are_refused_when_expected():
    w = World()
    w.deps.expected_gateway_id = "team13-checkpoint-abc"
    for ctx in (None, context("ran___observe"), gw_context("someone-else"),
                types.SimpleNamespace(client_context=None)):
        out = app.handler({"case_id": CASE}, ctx)
        assert "only answers calls that come through the team13 checkpoint" in out["error"]
    assert w.ddb.items == {} and w.ddb.calls == []   # refused before any read or write
    assert "kpis" in app.handler({"case_id": CASE}, gw_context("team13-checkpoint-abc"))


def test_without_an_expected_gateway_the_marker_is_not_required():
    World()
    assert "kpis" in app.handler({"case_id": CASE}, context("ran___observe"))
