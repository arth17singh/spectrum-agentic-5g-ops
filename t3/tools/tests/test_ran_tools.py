"""Tests for the RAN tool Lambda with fakes for DynamoDB, S3, the twin, the models and the trust gate."""
from __future__ import annotations

import hashlib
import json

import pytest

from fakes import (FakeBedrock, FakePolicy, FakePredictor, FakeS3, FakeSageMaker, FakeTable, FakeTrust, FakeTwin,
                   ListWriter, context, fake_kpis)
from t3.tools import app, catalog, evidence, nlm, store

CASE = "case-1a2b3c4d"
CHANGE = {"cells": [{"cell": 2, "d_power_db": 0, "d_tilt_deg": 2}]}
KIT_STYLE_ARGS = {"case_id": CASE, "item_id": "cell-2", "change": CHANGE, "reason": "raise coverage"}


def kit_key(args):
    """The approval id exactly as the organisers' interceptor computes it (written out here independently)."""
    return "chg-" + hashlib.sha256(json.dumps([args.get("case_id"), args.get("item_id"), args.get("change")])
                                   .encode()).hexdigest()[:12]


class World:
    def __init__(self, emulate=False, predictor=None, trust=None, policy=None, nlm_obj=None):
        self.cases_table, self.approvals_table = FakeTable("case_id"), FakeTable("pk")
        self.writer, self.trust = ListWriter(), trust if trust is not None else FakeTrust()
        self.predictor = predictor if predictor is not None else FakePredictor()
        self.deps = app.Deps(
            cases=store.DynamoCaseStore(self.cases_table), approvals=store.DynamoApprovalStore(self.approvals_table),
            twin_factory=FakeTwin, kpis=fake_kpis, predictor=lambda: self.predictor,
            policy=lambda: policy if policy is not None else FakePolicy(), trust=lambda: self.trust,
            writer=lambda: self.writer, nlm=nlm_obj, emulate_interceptor=emulate, synthetic=True,
            now=lambda: "2026-11-02T20:00:00.000Z")
        app.set_deps(self.deps)

    def call(self, tool, **args):
        return app.handler(args, context(tool))

    # what the kit's interceptor and `approve` command do to the approvals table
    def interceptor_creates_pending(self, args):
        key = kit_key(args)
        self.approvals_table.items.setdefault(key, {"pk": key, "status": "PENDING", "case_id": args["case_id"],
                                                    "agent": "noc_lead", "created_at": 0})
        return key

    def person_approves(self, key):
        assert self.approvals_table.items[key]["status"] == "PENDING"
        self.approvals_table.items[key].update(status="APPROVED", decided_by="person:test")


@pytest.fixture(autouse=True)
def _reset():
    yield
    app.set_deps(None)


# ---------------------------------------------------------------- dispatch and case_id
def test_dispatch_by_gateway_tool_name():
    w = World(nlm_obj=nlm.NLM(env={}))
    assert set(app._dispatch_table(app.Tools(w.deps))) == set(catalog.all_tools())
    obs = w.call("ran___observe", case_id=CASE)
    assert obs["twin"]["step"] == 0 and "coverage_ratio" in obs["kpis"]
    prop = w.call("ran___propose", case_id=CASE)
    assert prop["change"] == {"cells": [{"cell": 2, "d_power_db": 0.0, "d_tilt_deg": 2.0}]}
    assert prop["policy"] == {"model_id": "ppo-maveric", "version": "v0-test"}
    pred = w.call("ran___predict", case_id=CASE, change=CHANGE)
    assert set(pred["predicted"]) == set(app.KPI_KEYS)
    pre = w.call("ran___preflight", case_id=CASE, change=CHANGE)
    assert pre["after"]["coverage_ratio"] > pre["before"]["coverage_ratio"]
    assert w.call("ran___observe", case_id=CASE)["twin"]["step"] == 0  # preflight changed nothing
    ask = w.call("nlm13___ask", case_id=CASE, question="What is tilt?")
    assert "No NLM is configured" in ask["error"]
    assert "unknown tool" in w.call("work___lookup", case_id=CASE)["error"]
    assert "unknown tool" in app.handler({"case_id": CASE}, None)["error"]


def test_tool_name_from_event_for_local_runs():
    w = World()
    out = app.handler({"__tool__": "ran___observe", "case_id": CASE}, None)
    assert out["case_id"] == CASE and "__tool__" not in w.cases_table.items[CASE]["doc"]


@pytest.mark.parametrize("tool", catalog.all_tools())
@pytest.mark.parametrize("case_id", [None, "", "   ", 7, "../etc/passwd"])
def test_every_tool_refuses_without_a_case_id(tool, case_id):
    w = World(nlm_obj=nlm.NLM(env={}))
    args = {"change": CHANGE, "item_id": "cell-2", "reason": "r", "question": "q"}
    if case_id is not None:
        args["case_id"] = case_id
    out = app.handler(args, context(tool))
    assert "error" in out and "case_id" in out["error"]
    assert w.cases_table.items == {} and w.writer.records == []


def test_malformed_change_is_refused_before_any_approval_is_used():
    w = World()
    bad = dict(KIT_STYLE_ARGS, change={"cells": [{"cell": "two", "d_power_db": 0, "d_tilt_deg": 1}]})
    key = w.interceptor_creates_pending(bad)
    w.person_approves(key)
    out = w.call("ran___apply_change", **bad)
    assert "cell index" in out["error"]
    assert w.approvals_table.items[key]["status"] == "APPROVED"  # still usable


# ---------------------------------------------------------------- approvals (kit semantics)
def test_change_key_is_the_kits_key():
    assert app.change_key(KIT_STYLE_ARGS) == kit_key(KIT_STYLE_ARGS)
    assert app.change_key({"case_id": "c", "item_id": "i", "change": "setting=6"}) == kit_key(
        {"case_id": "c", "item_id": "i", "change": "setting=6"})
    # The key hashes the change as sent: the same cells in another key order are a different approval.
    reordered = dict(KIT_STYLE_ARGS, change={"cells": [{"d_tilt_deg": 2, "cell": 2, "d_power_db": 0}]})
    assert app.change_key(reordered) != app.change_key(KIT_STYLE_ARGS)
    # ...but the same change for matching a prediction.
    assert app.change_hash(app.parse_change(reordered["change"])) == app.change_hash(app.parse_change(CHANGE))


def test_apply_refused_while_pending():
    w = World()
    key = w.interceptor_creates_pending(KIT_STYLE_ARGS)
    out = w.call("ran___apply_change", **KIT_STYLE_ARGS, approval="PENDING")
    assert out["applied"] is False and out["approval_id"] == key and out["approval"] == "PENDING"
    assert "needs a person's approval" in out["error"] and key in out["error"]
    assert w.approvals_table.items[key]["status"] == "PENDING"
    assert w.writer.records == [] and CASE not in w.cases_table.items


def test_apply_refused_without_any_approval_row():
    w = World()
    out = w.call("ran___apply_change", **KIT_STYLE_ARGS)
    assert out["applied"] is False and out["approval"] == "MISSING"
    assert w.approvals_table.items == {}  # in AWS only the kit's interceptor creates approval rows


def test_apply_refused_when_rejected():
    w = World()
    key = w.interceptor_creates_pending(KIT_STYLE_ARGS)
    w.approvals_table.items[key]["status"] = "REJECTED"
    out = w.call("ran___apply_change", **KIT_STYLE_ARGS)
    assert out["applied"] is False and "already rejected" in out["error"]


def test_applied_once_after_approval_then_refused_as_already_applied():
    w = World()
    key = w.interceptor_creates_pending(KIT_STYLE_ARGS)
    w.person_approves(key)
    # a forged approval argument does nothing: only the table decides
    first = w.call("ran___apply_change", **KIT_STYLE_ARGS, approval="APPROVED",
                   special_context={"x-auth-email": "someone"})
    assert first["applied"] is True and first["approval_id"] == key
    assert w.approvals_table.items[key]["status"] == "APPLIED"
    update = [c for c in w.approvals_table.calls if c[0] == "update_item"]
    assert update == [("update_item", {"pk": key}, "SET #s = :applied, applied_at = :t", "#s = :approved")]
    assert w.call("ran___observe", case_id=CASE)["twin"]["step"] == 1

    second = w.call("ran___apply_change", **KIT_STYLE_ARGS, approval="APPROVED")
    assert second["applied"] is False and "already applied" in second["error"]
    assert w.call("ran___observe", case_id=CASE)["twin"]["step"] == 1  # the twin moved once
    assert len(w.writer.records) == 1


# ---------------------------------------------------------------- the change record
def test_change_record_has_run_id_predicted_and_measured():
    w = World()
    w.cases_table.items[CASE] = {"case_id": CASE, "run_id": "r-20261102-a"}  # written by the runner
    pred = w.call("ran___predict", case_id=CASE, change=CHANGE)
    w.call("ran___propose", case_id=CASE)
    key = w.interceptor_creates_pending(KIT_STYLE_ARGS)
    w.person_approves(key)
    out = w.call("ran___apply_change", **KIT_STYLE_ARGS)
    assert out["applied"] and out["run_id"] == "r-20261102-a"

    [(run_id, kind, rec)] = w.writer.records
    assert (run_id, kind) == ("r-20261102-a", "change")
    assert rec["run_id"] == "r-20261102-a" and rec["kind"] == "change" and rec["team"] == "team13"
    assert rec["schema_version"] == "1" and rec["synthetic"] is True
    assert rec["case_id"] == CASE and rec["change_id"] == key
    assert rec["change"] == [{"cell": 2, "d_power_db": 0.0, "d_tilt_deg": 2.0}]
    assert rec["twin"] == {"kind": "maveric-inprocess", "version": "unknown", "scenario": "nominal", "seed": 0,
                           "step": 1}
    assert rec["predicted"] == pred["predicted"]
    assert set(rec["measured"]) == set(app.KPI_KEYS)
    observed = w.call("ran___observe", case_id=CASE)["kpis"]
    assert rec["measured"] == {k: observed[k] for k in app.KPI_KEYS}
    assert rec["predictor"] == {"model_id": "t2-response", "version": "v0-test"}
    assert rec["policy"] == {"model_id": "ppo-maveric", "version": "v0-test"}  # the proposal had the same change
    assert rec["error"]["coverage_ratio"] == pytest.approx(0.0, abs=1e-12)  # unbiased fake predictor
    assert rec["direction_agree"] is True
    assert rec["trust_before"] == 1.0 and rec["trust_after"] == 1.0
    assert w.trust.updates[0]["predicted"] == pred["predicted"]
    json.dumps(rec, allow_nan=False)


def test_wrong_direction_prediction_lowers_trust_and_reports_pause():
    w = World(predictor=FakePredictor(flip=True))
    w.call("ran___predict", case_id=CASE, change=CHANGE)
    key = w.interceptor_creates_pending(KIT_STYLE_ARGS)
    w.person_approves(key)
    out = w.call("ran___apply_change", **KIT_STYLE_ARGS)
    assert out["direction_agree"] is False and out["trust"]["action"] == "pause"
    rec = w.writer.records[0][2]
    assert rec["direction_agree"] is False and rec["trust_after"] == 0.5
    assert rec["error"]["summary"] > 0


def test_apply_without_prediction_writes_unscored_record_and_skips_trust():
    w = World()
    key = w.interceptor_creates_pending(KIT_STYLE_ARGS)
    w.person_approves(key)
    out = w.call("ran___apply_change", **KIT_STYLE_ARGS)
    assert out["applied"] and "nothing to score" in out["note"]
    rec = w.writer.records[0][2]
    assert rec["predicted"] is None and rec["direction_agree"] is None and rec["measured"]
    assert w.trust.updates == []


def test_run_id_sources():
    w = World()
    assert w.call("ran___observe", case_id="case-00000001")["run_id"] == "adhoc-case-00000001"
    assert w.call("ran___observe", case_id="case-00000002", run_id="run_id: r-7")["run_id"] == "r-7"
    assert w.call("ran___observe", case_id="case-00000003", run_id="r-8")["run_id"] == "r-8"
    w.cases_table.items["case-00000004"] = {"case_id": "case-00000004", "run_id": "r-runner", "seed": 5}
    out = w.call("ran___observe", case_id="case-00000004", run_id="r-other")
    assert out["run_id"] == "r-runner" and out["twin"]["seed"] == 5  # the runner's row wins
    # the first call fixes the run_id for the life of the case
    assert w.call("ran___observe", case_id="case-00000001", run_id="r-late")["run_id"] == "adhoc-case-00000001"


def test_replay_is_deterministic_and_stateless():
    w = World()
    for i, d in enumerate((1, -1, 2)):
        args = dict(KIT_STYLE_ARGS, item_id=f"cell-2-{i}", change={"cells": [{"cell": 2, "d_power_db": d,
                                                                                "d_tilt_deg": 0}]})
        w.person_approves(w.interceptor_creates_pending(args))
        assert w.call("ran___apply_change", **args)["applied"]
    obs = w.call("ran___observe", case_id=CASE)
    assert obs["twin"]["step"] == 3
    assert obs["kpis"]["per_cell"][2]["power_dbm"] == 42.0  # 40 + 1 - 1 + 2, rebuilt from the action log
    doc = json.loads(w.cases_table.items[CASE]["doc"])
    assert set(doc) == set(store.CASE_FIELDS) and len(doc["action_log"]) == 3


def test_case_row_conflict_is_detected():
    w = World()
    w.call("ran___observe", case_id=CASE)
    s = store.DynamoCaseStore(w.cases_table)
    case = s.get(CASE)
    s.put(dict(case), case["version"])
    with pytest.raises(store.ConflictError):
        s.put(dict(case), case["version"])


# ---------------------------------------------------------------- local mode (W3b's Docker runs)
def test_local_store_emulates_the_interceptor(tmp_path):
    cases, approvals, emulate = store.from_env({"T3_STORE": f"local:{tmp_path}"})
    assert emulate is True
    writer = evidence.LocalWriter(tmp_path / "ev")
    app.set_deps(app.Deps(cases=cases, approvals=approvals, twin_factory=FakeTwin, kpis=fake_kpis,
                          predictor=lambda: FakePredictor(), writer=lambda: writer, emulate_interceptor=True))
    ctx = context("ran___apply_change")
    app.handler({"case_id": CASE, "change": CHANGE}, context("ran___predict"))
    first = app.handler(dict(KIT_STYLE_ARGS), ctx)
    assert first["applied"] is False and first["approval"] == "PENDING"
    assert [r["pk"] for r in approvals.pending()] == [kit_key(KIT_STYLE_ARGS)]
    assert store.main([str(tmp_path), "approve", first["approval_id"], "looks", "fine"]) == 0
    assert app.handler(dict(KIT_STYLE_ARGS), ctx)["applied"] is True
    assert "already applied" in app.handler(dict(KIT_STYLE_ARGS), ctx)["error"]
    lines = (tmp_path / "ev/records/adhoc-case-1a2b3c4d/change.jsonl").read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["predicted"] is not None


def test_store_selection_errors():
    with pytest.raises(ValueError):
        store.from_env({})
    with pytest.raises(ValueError):
        store.from_env({"T3_STORE": "redis:x"})
    with pytest.raises(ValueError):
        store.LocalCaseStore("/tmp/unused-never-written")._path("../x")


# ---------------------------------------------------------------- evidence writers
def test_s3_writer_appends_with_conditions():
    s3 = FakeS3()
    w = evidence.S3Writer("bucket", "evidence", client=s3)
    assert w.append_records("r-1", "change", [{"a": 1}]) == "s3://bucket/evidence/records/r-1/change.jsonl"
    w.append_records("r-1", "change", [{"a": 2}])
    body = s3.objects[("bucket", "evidence/records/r-1/change.jsonl")][0].decode().splitlines()
    assert [json.loads(x)["a"] for x in body] == [1, 2]
    assert s3.puts[0][2] == "*" and s3.puts[1][1] is not None
    with pytest.raises(ValueError):
        w.append_records("../r", "change", [{}])


def test_evidence_selection():
    assert isinstance(evidence.from_env({"T3_EVIDENCE": "local:/x"}), evidence.LocalWriter)
    s3w = evidence.from_env({"T3_EVIDENCE": "s3://b/p/q"})
    assert (s3w.bucket, s3w.prefix) == ("b", "p/q")
    with pytest.raises(ValueError):
        evidence.from_env({"T3_EVIDENCE": "ftp://x"})


def test_evidence_failure_does_not_hide_the_applied_change():
    w = World()

    class Broken:
        def append_records(self, *a):
            raise OSError("disk full")
    w.deps.writer = lambda: Broken()
    w.person_approves(w.interceptor_creates_pending(KIT_STYLE_ARGS))
    out = w.call("ran___apply_change", **KIT_STYLE_ARGS)
    assert out["applied"] is True and "disk full" in out["evidence_error"]


# ---------------------------------------------------------------- NLM
def test_nlm_sagemaker_endpoint_is_labelled():
    sm = FakeSageMaker({"choices": [{"message": {"content": " Tilt aims the beam. "}}]})
    w = World(nlm_obj=nlm.NLM(env={"NLM_ENDPOINT": "team13-otel-tuned", "NLM_STANDIN_MODEL": "x"},
                              client_factory=lambda svc: {"sagemaker-runtime": sm}[svc]))
    out = w.call("nlm13___ask", case_id=CASE, question="What is tilt?")
    assert out["answer"] == "Tilt aims the beam."
    assert out["answered_by"]["kind"] == "sagemaker-endpoint" and out["answered_by"]["stand_in"] is False
    body = json.loads(sm.calls[0]["Body"])
    assert sm.calls[0]["EndpointName"] == "team13-otel-tuned" and body["messages"][1]["content"] == "What is tilt?"


def test_nlm_standin_is_labelled_as_standin():
    br = FakeBedrock("A stand-in answer.")
    w = World(nlm_obj=nlm.NLM(env={"NLM_STANDIN_MODEL": "openai.gpt-oss-20b-1:0"},
                              client_factory=lambda svc: {"bedrock-runtime": br}[svc]))
    out = w.call("nlm13___ask", case_id=CASE, question="What is SINR?")
    assert out["answer"] == "A stand-in answer."
    assert out["answered_by"] == {"kind": "bedrock-standin", "id": "openai.gpt-oss-20b-1:0",
                                  "model_label": "stand-in: NOT our tuned model", "stand_in": True}
    assert br.calls[0]["modelId"] == "openai.gpt-oss-20b-1:0"


def test_nlm_tgi_payload_and_parsing():
    sm = FakeSageMaker([{"generated_text": "ok"}])
    n = nlm.NLM(env={"NLM_ENDPOINT": "e", "NLM_PAYLOAD": "tgi"}, client_factory=lambda s: sm)
    assert n.ask("q")["answer"] == "ok"
    assert "inputs" in json.loads(sm.calls[0]["Body"])
    with pytest.raises(ValueError):
        nlm.parse_generation({"nothing": 1})


def test_nlm_target_name_is_ours_not_the_kits(monkeypatch):
    monkeypatch.setenv("T3_NLM_TARGET", "nlm")
    with pytest.raises(ValueError):
        catalog.nlm_target()
    monkeypatch.setenv("T3_NLM_TARGET", "bad_name")
    with pytest.raises(ValueError):
        catalog.nlm_target()


# ---------------------------------------------------------------- change parsing
def test_change_may_arrive_as_json_text():
    assert app.parse_change(json.dumps(CHANGE)) == [{"cell": 2, "d_power_db": 0.0, "d_tilt_deg": 2.0}]
    for bad in ({}, {"cells": []}, {"cells": [{"cell": 1}, {"cell": 1}]}, {"cells": [{"cell": True}]},
                {"cells": [{"cell": 1, "d_power_db": float("nan")}]}, "not json", [1]):
        with pytest.raises(app.ToolError):
            app.parse_change(bad)
