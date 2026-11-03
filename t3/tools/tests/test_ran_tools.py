"""Tests for the RAN tool Lambda with fakes for DynamoDB, S3, the twin, the models and the trust gate."""
from __future__ import annotations

import hashlib
import json

import pytest

from fakes import (FakeBedrock, FakeDynamo, FakePolicy, FakePredictor, FakeS3, FakeSageMaker, FakeTrustInvoker,
                   FakeTwin, FakeTwinCF, MemoryWriter, context, fake_kpis)
from t3.tools import app, catalog, changes, evidence, nlm, store

CASE = "case-1a2b3c4d"
CHANGE = "cell=2;d_power_db=+0.0;d_tilt_deg=+2.0"
APPLY = {"case_id": CASE, "item_id": "step-0", "change": CHANGE, "reason": "raise coverage"}


def kit_key(args):
    """The approval id exactly as the organisers' interceptor computes it (written out independently)."""
    return "chg-" + hashlib.sha256(json.dumps([args.get("case_id"), args.get("item_id"), args.get("change")])
                                   .encode()).hexdigest()[:12]


class World:
    def __init__(self, predictor=None, policy=None, nlm_obj=None, run_id=None, trust=True, twin=FakeTwinCF):
        self.ddb = FakeDynamo()
        if run_id:
            self.ddb.set_plain("t3-active", run_id=run_id, unit="u")  # the runner: active run ...
            self.ddb.set_plain(f"t3-{CASE}", case_id=CASE, run_id=run_id, unit="u", created_at=1793600000,
                               source="t3/suite/runner.py")  # ... and map_case: a row WITHOUT doc
        self.writer = MemoryWriter()
        self.trust = FakeTrustInvoker()
        self.predictor = predictor if predictor is not None else FakePredictor()
        self.deps = app.Deps(
            store=store.DynamoStore("team13-approvals", client=self.ddb), twin_factory=twin, kpis=fake_kpis,
            predictor=lambda: self.predictor, policy=lambda: policy if policy is not None else FakePolicy(),
            trust_invoker=(lambda: self.trust) if trust else (lambda: None), writer=lambda: self.writer,
            nlm=nlm_obj or nlm.NLM(env={}), synthetic=True, now=lambda: "2026-11-02T20:00:00.000Z",
            scales={"coverage_ratio": 0.01, "sla_ratio": 0.01, "sinr_p10_db": 0.5})
        app.set_deps(self.deps)

    def call(self, tool, **args):
        return app.handler(args, context(tool))

    def interceptor_creates_pending(self, args):
        key = kit_key(args)
        if key not in self.ddb.items:
            self.ddb.set_plain(key, status="PENDING", case_id=args["case_id"], agent="noc_lead", created_at=0)
        return key

    def person_approves(self, key):
        assert self.ddb.plain(key)["status"] == "PENDING"
        self.ddb.items[key]["status"] = {"S": "APPROVED"}

    def approved(self, args):
        key = self.interceptor_creates_pending(args)
        self.person_approves(key)
        return key

    def case_doc(self, case_id=CASE):
        return json.loads(self.ddb.plain(f"t3-{case_id}")["doc"])

    def records(self):
        return list(self.writer.records.values())


@pytest.fixture(autouse=True)
def _reset():
    yield
    app.set_deps(None)


# ---------------------------------------------------------------- dispatch and case_id
def test_dispatch_by_gateway_tool_name():
    w = World()
    assert set(app._dispatch_table(app.Tools(w.deps))) == set(catalog.all_tools()) | {"nlm___ask_nlm"}
    obs = w.call("ran___observe", case_id=CASE)
    assert obs["twin"]["step"] == 0 and "coverage_ratio" in obs["kpis"]
    prop = w.call("ran___propose", case_id=CASE)
    assert prop == {"case_id": CASE, "run_id": f"adhoc-{CASE}", "item_id": "step-0", "change": CHANGE,
                    "policy": {"model_id": "ppo-maveric", "version": "v0-test"}, "note": prop["note"]}
    pred = w.call("ran___predict", case_id=CASE, item_id="step-0", change=CHANGE)
    assert set(pred["predicted_delta"]) == set(app.KPI_KEYS)
    pre = w.call("ran___preflight", case_id=CASE, change=CHANGE)
    assert pre["measured_delta"]["coverage_ratio"] == pytest.approx(pred["predicted_delta"]["coverage_ratio"])
    assert w.call("ran___observe", case_id=CASE)["twin"]["step"] == 0  # preflight changed nothing
    for tool in ("nlm___ask", "nlm___ask_nlm"):
        assert "No NLM is configured" in w.call(tool, case_id=CASE, question="What is tilt?")["error"]
    assert "unknown tool" in w.call("work___lookup", case_id=CASE)["error"]
    assert "unknown tool" in app.handler({"case_id": CASE}, None)["error"]


def test_tool_name_fallback_only_when_enabled():
    w = World()
    event = {"_tool": "ran___observe", "case_id": CASE}
    assert "unknown tool" in app.handler(dict(event), None)["error"]
    w.deps.tool_name_fallback = True
    out = app.handler(dict(event), None)
    assert out["case_id"] == CASE and "_tool" not in w.case_doc()
    # the Gateway's name always wins over the event
    assert "kpis" in app.handler({"_tool": "ran___apply_change", "case_id": CASE}, context("ran___observe"))


@pytest.mark.parametrize("tool", catalog.all_tools())
@pytest.mark.parametrize("case_id", [None, "", "   ", 7, "../etc/passwd"])
def test_every_tool_refuses_without_a_case_id(tool, case_id):
    w = World()
    args = {"change": CHANGE, "item_id": "step-0", "reason": "r", "question": "q"}
    if case_id is not None:
        args["case_id"] = case_id
    out = app.handler(args, context(tool))
    assert "error" in out and "case_id" in out["error"]
    assert w.ddb.items == {} and w.writer.records == {}


def test_bad_change_is_refused_before_the_approval_is_used():
    w = World()
    bad = dict(APPLY, change="cell=two;d_power_db=+0.0;d_tilt_deg=+1.0")
    key = w.approved(bad)
    out = w.call("ran___apply_change", **bad)
    assert "bad change" in out["error"]
    assert w.ddb.plain(key)["status"] == "APPROVED"  # still usable
    for text in ("cell=1;d_power_db=+0.25;d_tilt_deg=+0.0", "cell=1;d_power_db=+0.0;d_tilt_deg=+0.0|"
                 "cell=1;d_power_db=+1.0;d_tilt_deg=+0.0", ""):
        assert "bad change" in w.call("ran___predict", case_id=CASE, change=text)["error"]


# ---------------------------------------------------------------- the canonical change (A2)
def test_canonical_change_text():
    cells = [{"cell": 3, "d_power_db": -1.0, "d_tilt_deg": 0.0}, {"cell": 1, "d_power_db": 0.5, "d_tilt_deg": -0.0}]
    text = changes.canon(cells)
    assert text == "cell=1;d_power_db=+0.5;d_tilt_deg=+0.0|cell=3;d_power_db=-1.0;d_tilt_deg=+0.0"
    assert changes.canon(changes.parse(text)) == text
    assert changes.parse("cell=3;d_power_db=-1;d_tilt_deg=0|cell=1;d_power_db=0.5;d_tilt_deg=-0.0") == \
        changes.parse(text)
    assert changes.canon(changes.round_items([{"cell": 0, "d_power_db": 1.04, "d_tilt_deg": -0.96}])) == \
        "cell=0;d_power_db=+1.0;d_tilt_deg=-1.0"


# ---------------------------------------------------------------- approvals (kit semantics)
def test_change_key_is_the_kits_key():
    assert app.change_key(APPLY) == kit_key(APPLY)
    assert app.change_key({"case_id": "c", "item_id": "i", "change": "setting=6"}) == kit_key(
        {"case_id": "c", "item_id": "i", "change": "setting=6"})


def test_apply_refused_while_pending():
    w = World()
    key = w.interceptor_creates_pending(APPLY)
    out = w.call("ran___apply_change", **APPLY, approval="PENDING")
    assert out["applied"] is False and out["approval_id"] == key and out["approval"] == "PENDING"
    assert "needs a person's approval" in out["error"] and key in out["error"]
    assert w.ddb.plain(key)["status"] == "PENDING"
    assert w.writer.records == {} and f"t3-{CASE}" not in w.ddb.items


def test_apply_refused_without_any_approval_row():
    w = World()
    out = w.call("ran___apply_change", **APPLY)
    assert out["applied"] is False and out["approval"] == "MISSING"
    assert w.ddb.items == {}  # in AWS only the kit's interceptor creates approval rows


def test_apply_refused_when_rejected():
    w = World()
    key = w.interceptor_creates_pending(APPLY)
    w.ddb.items[key]["status"] = {"S": "REJECTED"}
    assert "already rejected" in w.call("ran___apply_change", **APPLY)["error"]


def test_applied_once_after_approval_then_refused_as_already_applied():
    w = World()
    key = w.approved(APPLY)
    # a forged approval argument does nothing: only the table decides
    first = w.call("ran___apply_change", **APPLY, approval="APPROVED", special_context={"x-auth-email": "someone"})
    assert first["applied"] is True and first["approval_id"] == key
    assert w.ddb.plain(key)["status"] == "APPLIED"
    [tx] = [c for c in w.ddb.calls if c[0] == "transact_write_items"]
    upd, put = tx[1]["TransactItems"]
    assert upd["Update"]["Key"] == {"pk": {"S": key}}
    assert upd["Update"]["UpdateExpression"] == "SET #s = :applied, applied_at = :t"
    assert upd["Update"]["ConditionExpression"] == "#s = :approved"
    assert put["Put"]["Item"]["pk"] == {"S": f"t3-{CASE}"} and "status" not in put["Put"]["Item"]
    assert w.call("ran___observe", case_id=CASE)["twin"]["step"] == 1

    second = w.call("ran___apply_change", **APPLY, approval="APPROVED")
    assert second["applied"] is False and "already applied" in second["error"]
    assert w.call("ran___observe", case_id=CASE)["twin"]["step"] == 1  # the twin moved once
    assert len(w.records()) == 1


def test_lost_race_on_the_approval_applies_nothing():
    """The status read said APPROVED, but another call applied it first: the transaction refuses."""
    w = World()
    key = w.approved(APPLY)
    real = w.deps.store.approval_status
    w.deps.store.approval_status = lambda k: "APPROVED"
    w.ddb.items[key]["status"] = {"S": "APPLIED"}
    out = w.call("ran___apply_change", **APPLY)
    w.deps.store.approval_status = real
    assert out["applied"] is False and "already applied" in out["error"]
    assert w.case_doc()["action_log"] == [] and w.records() == []


def test_case_row_conflict_is_retried():
    w = World()
    key = w.approved(APPLY)
    w.call("ran___observe", case_id=CASE)
    s = w.deps.store
    real_get = s.get_case
    bumped = {"done": False}

    def racing_get(case_id):
        case = real_get(case_id)
        if not bumped["done"]:  # someone else writes the row after we read it
            bumped["done"] = True
            s.put_case(dict(case), case["version"])
        return case
    s.get_case = racing_get
    out = w.call("ran___apply_change", **APPLY)
    assert out["applied"] is True and w.ddb.plain(key)["status"] == "APPLIED"
    assert len(w.case_doc()["action_log"]) == 1


# ---------------------------------------------------------------- the change record (A15)
def test_change_record_has_run_id_predicted_and_measured():
    w = World(run_id="r-20261102-a")
    prop = w.call("ran___propose", case_id=CASE)
    pred = w.call("ran___predict", case_id=CASE, item_id=prop["item_id"], change=prop["change"])
    args = {"case_id": CASE, "item_id": prop["item_id"], "change": prop["change"], "reason": "r"}
    key = w.approved(args)
    out = w.call("ran___apply_change", **args)
    assert out["applied"] and out["run_id"] == "r-20261102-a"

    [(path, rec)] = list(w.writer.records.items())
    assert path == f"records/r-20261102-a/change/{key}.json"
    assert rec["run_id"] == "r-20261102-a" and rec["kind"] == "change" and rec["team"] == "team13"
    assert rec["schema_version"] == "1" and rec["synthetic"] is True and rec["probe"] is False
    assert rec["case_id"] == CASE and rec["change_id"] == key and rec["item_id"] == "step-0"
    assert rec["change"] == [{"cell": 2, "d_power_db": 0.0, "d_tilt_deg": 2.0}] and rec["change_text"] == CHANGE
    assert {k: v for k, v in rec["twin"].items() if k != "preset"} == {"kind": "maveric-inprocess", "version": "unknown", "scenario": "nominal", "seed": 0,
                           "step": 0}
    assert rec["predicted_delta"] == pred["predicted_delta"]
    assert rec["predictor"] == {"model_id": "t2-response", "version": "v0-test"}
    assert rec["policy"] == {"model_id": "ppo-maveric", "version": "v0-test"}
    for k in app.KPI_KEYS:
        assert rec["measured_delta"][k] == rec["measured_kpis"][k] - rec["baseline_kpis"][k]
    observed = w.call("ran___observe", case_id=CASE)["kpis"]
    assert rec["measured_kpis"] == {k: observed[k] for k in app.KPI_KEYS}
    assert rec["scored"] is True and rec["unscored_reason"] is None
    assert rec["error"]["coverage_ratio"] == pytest.approx(0.0, abs=1e-12)  # unbiased fake predictor
    assert set(rec["error"]) == set(app.KPI_KEYS) | {"summary"}
    assert rec["direction_agree"] is True
    # the gate was asked synchronously BEFORE the record was written, so the record carries its answer
    assert rec["trust_before"] == 1.0 and rec["trust_after"] == 1.0
    [payload] = w.trust.payloads
    assert payload["model_version"] == "v0-test"
    assert {k: v for k, v in payload["record"].items() if not k.startswith("trust_")} == \
        {k: v for k, v in rec.items() if not k.startswith("trust_")}
    assert out["trust"]["action"] is None and out["trust"]["e"] == rec["error"]["summary"]
    json.dumps(rec, allow_nan=False)


def test_apply_reads_the_stored_prediction_and_never_runs_the_model():
    w = World(predictor=FakePredictor(flip=True))
    w.call("ran___predict", case_id=CASE, item_id="step-0", change=CHANGE)
    assert len(w.predictor.calls) == 1
    w.approved(APPLY)
    out = w.call("ran___apply_change", **APPLY)
    assert len(w.predictor.calls) == 1   # apply did not call the predictor (A4)
    assert out["direction_agree"] is False and out["scored"] is True
    rec = w.records()[0]
    assert rec["error"]["summary"] > 0
    assert rec["error"]["summary"] == pytest.approx(sum(rec["error"][k] / s for k, s in w.deps.scales.items()) / 3)
    assert (rec["trust_before"], rec["trust_after"]) == (1.0, 0.5)
    assert out["trust"]["action"] == "pause" and out["trust"]["control_record_key"] == "ctl-0001"


def test_clipped_change_is_recorded_unscored():
    w = World()
    big = dict(APPLY, change="cell=1;d_power_db=+5.0;d_tilt_deg=+0.0")
    w.call("ran___predict", case_id=CASE, item_id="step-0", change=big["change"])
    w.approved(big)
    out = w.call("ran___apply_change", **big)
    assert out["applied"] and out["clipped"] is True
    rec = w.records()[0]
    assert rec["scored"] is False and rec["unscored_reason"] == "clipped"
    assert rec["error"] is None and rec["direction_agree"] is None
    assert w.trust.records == []  # nothing to score


def test_apply_without_prediction_is_unscored():
    w = World()
    w.approved(APPLY)
    out = w.call("ran___apply_change", **APPLY)
    assert out["applied"] and out["unscored_reason"] == "no_prediction"
    assert w.records()[0]["predicted_delta"] is None and w.trust.records == []


@pytest.mark.parametrize("mode,text", [("timeout", "ReadTimeoutError"), ("error", "Unhandled")])
def test_trust_gate_timeout_or_error_never_fails_the_apply(mode, text):
    w = World()
    w.trust = FakeTrustInvoker(mode)
    w.call("ran___predict", case_id=CASE, item_id="step-0", change=CHANGE)
    key = w.approved(APPLY)
    out = w.call("ran___apply_change", **APPLY)
    assert out["applied"] is True and not out.get("error") and w.ddb.plain(key)["status"] == "APPLIED"
    assert text in out["trust"]["note"] and "left null" in out["trust"]["note"]
    [rec] = w.records()   # written once, with null trust fields
    assert rec["trust_before"] is None and rec["trust_after"] is None and rec["scored"] is True
    assert len(w.trust.payloads) == 1


def test_trust_gate_off_leaves_trust_null():
    w = World(trust=False)
    w.call("ran___predict", case_id=CASE, item_id="step-0", change=CHANGE)
    w.approved(APPLY)
    out = w.call("ran___apply_change", **APPLY)
    assert out["applied"] and "not configured" in out["trust"]["note"]
    assert w.records()[0]["trust_after"] is None


def test_default_invoker_is_synchronous_with_a_short_timeout(monkeypatch):
    """Deps.from_env builds a RequestResponse invoker with a ~5 s read timeout and no retries."""
    import io as _io
    import boto3
    calls = {}

    class FakeLambda:
        def invoke(self, **kw):
            calls["invoke"] = kw
            return {"StatusCode": 200, "Payload": _io.BytesIO(b'{"trust_before": 0.9, "trust_after": 0.8}')}

    def fake_client(service, config=None, **kw):
        calls["service"], calls["config"] = service, config
        return FakeLambda()
    monkeypatch.setattr(boto3, "client", fake_client)
    deps = app.Deps.from_env({"T3_STORE": "local:/tmp/unused-t3", "T3_EVIDENCE": "local:/tmp/unused-t3"})
    out = deps.trust_invoker()({"record": {"kind": "change"}, "model_version": "v1"})
    assert out == {"trust_before": 0.9, "trust_after": 0.8}
    assert calls["invoke"]["InvocationType"] == "RequestResponse"
    assert calls["invoke"]["FunctionName"] == "team13-trust-gate"
    assert calls["config"].read_timeout == 5.0 and calls["config"].retries == {"max_attempts": 0}

    class Failing(FakeLambda):
        def invoke(self, **kw):
            return {"StatusCode": 200, "FunctionError": "Unhandled", "Payload": _io.BytesIO(b'{"errorMessage": "x"}')}
    monkeypatch.setattr(boto3, "client", lambda service, config=None, **kw: Failing())
    deps = app.Deps.from_env({"T3_STORE": "local:/tmp/unused-t3", "T3_EVIDENCE": "local:/tmp/unused-t3"})
    with pytest.raises(RuntimeError):
        deps.trust_invoker()({"record": {}})


def map_case(w, case_id, run_id):
    """Exactly what t3/suite/runner.py map_case writes: a t3-<case> row with NO doc."""
    w.ddb.set_plain(f"t3-{case_id}", case_id=case_id, run_id=run_id, unit="t3.allow", created_at=1793600000,
                    source="t3/suite/runner.py")


def test_runner_mapped_case_without_doc_works_and_gets_its_doc():
    w = World()
    map_case(w, "case-r7aaaaaa-01", "r-7")
    out = w.call("ran___observe", case_id="case-r7aaaaaa-01")
    assert not out.get("error") and out["run_id"] == "r-7"           # used to fail: KeyError 'doc'
    row = w.ddb.plain("t3-case-r7aaaaaa-01")
    assert row["doc"] and row["version"] == 1 and row["run_id"] == "r-7"
    assert row["unit"] == "t3.allow" and row["source"] == "t3/suite/runner.py"   # the runner's fields kept
    assert "status" not in row                                                  # never listed by approve
    puts = [c for c in w.ddb.calls if c[0] == "put_item"]
    assert puts[0][1]["ConditionExpression"] == "attribute_not_exists(pk) OR attribute_not_exists(doc)"
    # and the full flow on that case: predict, approve, apply, record under the runner's run
    args = dict(APPLY, case_id="case-r7aaaaaa-01")
    w.call("ran___predict", case_id=args["case_id"], item_id="step-0", change=CHANGE)
    w.approved(args)
    assert w.call("ran___apply_change", **args)["applied"] is True
    assert w.records()[0]["run_id"] == "r-7"


def test_second_create_of_the_same_case_is_still_guarded():
    w = World()
    map_case(w, "case-r7aaaaaa-02", "r-7")
    s = w.deps.store
    first = {"case_id": "case-r7aaaaaa-02", "run_id": "r-7", "scenario": "nominal", "seed": 0, "config": "official",
             "action_log": [], "twin_kind": "maveric-inprocess", "predictions": [], "created_at": "t"}
    s.put_case(dict(first), None)                        # fills in the doc-less mapping row
    with pytest.raises(store.ConflictError):
        s.put_case(dict(first, run_id="r-other"), None)  # a concurrent second create must not overwrite
    assert s.get_case("case-r7aaaaaa-02")["run_id"] == "r-7"
    w.call("ran___observe", case_id="case-r7aaaaaa-02")  # load_case survives the race path too
    assert s.get_case("case-r7aaaaaa-02")["version"] == 1


def test_chat_cases_are_adhoc_even_with_an_active_suite_run():
    w = World()
    w.ddb.set_plain("t3-active", run_id="r-suite")      # never cleared by the runner
    assert w.call("ran___observe", case_id="case-chat0001")["run_id"] == "adhoc-case-chat0001"
    map_case(w, "case-chat0002", "r-suite")
    w.ddb.set_plain("t3-active", run_id="r-next")
    assert w.call("ran___observe", case_id="case-chat0002")["run_id"] == "r-suite"   # the mapping wins
    assert w.call("ran___observe", case_id="case-chat0002")["run_id"] == "r-suite"   # fixed for the case's life
    assert "status" not in w.ddb.items["t3-case-chat0002"]


def test_local_store_mapping_rows(tmp_path):
    s = store.LocalStore(tmp_path)
    s.map_case("case-loc00001", "r-local")
    assert s.get_case("case-loc00001") is None and s.mapped_run_id("case-loc00001") == "r-local"


def test_replay_is_deterministic_and_the_twin_is_cached():
    w = World()
    FakeTwin.instances = 0
    for i, d in enumerate((1.0, -1.0, 2.0)):
        args = dict(APPLY, item_id=f"step-{i}", change=f"cell=2;d_power_db={d:+.1f};d_tilt_deg=+0.0")
        w.approved(args)
        assert w.call("ran___apply_change", **args)["applied"]
    obs = w.call("ran___observe", case_id=CASE)
    assert obs["twin"]["step"] == 3
    assert obs["kpis"]["per_cell"][2]["power_dbm"] == 42.0  # 40 + 1 - 1 + 2, rebuilt from the action log
    assert FakeTwin.instances == 1  # one cached twin object, reset per use (A16)
    doc = w.case_doc()
    assert set(doc) == set(store.CASE_FIELDS) and [e["step"] for e in doc["action_log"]] == [0, 1, 2]


def test_replay_mismatch_is_reported():
    w = World()
    w.approved(APPLY)
    w.call("ran___apply_change", **APPLY)
    doc = w.case_doc()
    doc["action_log"][0]["filtered"]["tilt"][2] = 1.5
    row = w.ddb.items[f"t3-{CASE}"]
    row["doc"] = {"S": json.dumps(doc)}
    assert "ReplayMismatch" in w.call("ran___observe", case_id=CASE)["error"]


def test_model_version_must_be_the_served_one():
    w = World()
    assert "not being served" in w.call("ran___predict", case_id=CASE, change=CHANGE, model_version="v9")["error"]
    assert "predicted_delta" in w.call("ran___predict", case_id=CASE, change=CHANGE, model_version="v0-test")
    assert "not being served" in w.call("ran___propose", case_id=CASE, model_version="v9")["error"]


# ---------------------------------------------------------------- local mode (W3b's Docker runs)
def test_local_store_emulates_the_interceptor(tmp_path):
    s = store.from_env({"T3_STORE": f"local:{tmp_path}"})
    assert s.emulate_interceptor is True
    s.set_active_run_id("r-other")   # never used for a case (t3-active is not cleared)
    s.map_case(CASE, "r-local")      # the runner maps the suite case
    writer = evidence.LocalWriter(tmp_path / "ev")
    app.set_deps(app.Deps(store=s, twin_factory=FakeTwin, kpis=fake_kpis, predictor=lambda: FakePredictor(),
                          writer=lambda: writer, tool_name_fallback=True))
    app.handler({"_tool": "ran___predict", "case_id": CASE, "item_id": "step-0", "change": CHANGE}, None)
    first = app.handler({"_tool": "ran___apply_change", **APPLY}, None)
    assert first["applied"] is False and first["approval"] == "PENDING"
    assert [r["pk"] for r in s.pending()] == [kit_key(APPLY)]
    assert store.main([str(tmp_path), "approve", first["approval_id"], "looks", "fine"]) == 0
    assert app.handler({"_tool": "ran___apply_change", **APPLY}, None)["applied"] is True
    assert "already applied" in app.handler({"_tool": "ran___apply_change", **APPLY}, None)["error"]
    rec = json.loads((tmp_path / f"ev/records/r-local/change/{first['approval_id']}.json").read_text())
    assert rec["predicted_delta"] is not None and rec["run_id"] == "r-local"


def test_w3b_sample_event_shape_still_parses():
    """W3b's events send `change` as a list; the AWS schema sends a string. Both parse the same."""
    assert changes.parse([{"cell": 0, "d_power_db": 1.0, "d_tilt_deg": -1.0}]) == \
        changes.parse("cell=0;d_power_db=+1.0;d_tilt_deg=-1.0")


def test_store_selection_errors(tmp_path):
    with pytest.raises(ValueError):
        store.from_env({})
    with pytest.raises(ValueError):
        store.from_env({"T3_STORE": "redis:x"})
    assert isinstance(store.from_env({"APPROVALS_TABLE": "team13-approvals"}), store.DynamoStore)
    with pytest.raises(ValueError):
        store.LocalStore(tmp_path).get_case("../x")


# ---------------------------------------------------------------- evidence writers (A11)
def test_s3_writer_one_object_per_record_never_overwritten():
    s3 = FakeS3()
    w = evidence.S3Writer("bucket", "evidence", client=s3)
    loc = w.put_record("r-1", "change", "chg-0123456789ab", {"a": 1})
    assert loc == "s3://bucket/evidence/records/r-1/change/chg-0123456789ab.json"
    with pytest.raises(evidence.RecordExists):
        w.put_record("r-1", "change", "chg-0123456789ab", {"a": 2})
    assert json.loads(s3.objects[("bucket", "evidence/records/r-1/change/chg-0123456789ab.json")]) == {"a": 1}
    with pytest.raises(ValueError):
        w.put_record("../r", "change", "k", {})


def test_local_writer_and_selection(tmp_path):
    w = evidence.from_env({"T3_EVIDENCE": f"local:{tmp_path}"})
    w.put_record("r-1", "change", "k1", {"a": 1})
    assert (tmp_path / "records/r-1/change/k1.json").exists()
    with pytest.raises(evidence.RecordExists):
        w.put_record("r-1", "change", "k1", {"a": 1})
    s3w = evidence.from_env({"T3_EVIDENCE": "s3://b/p/q"})
    assert (s3w.bucket, s3w.prefix) == ("b", "p/q")
    with pytest.raises(ValueError):
        evidence.from_env({})


def test_evidence_failure_does_not_hide_the_applied_change():
    w = World()

    class Broken:
        def put_record(self, *a):
            raise OSError("disk full")
    w.deps.writer = lambda: Broken()
    w.approved(APPLY)
    out = w.call("ran___apply_change", **APPLY)
    assert out["applied"] is True and "disk full" in out["evidence_error"]


def test_change_applied_log_line(capsys):
    w = World(run_id="r-9")
    key = w.approved(APPLY)
    w.call("ran___apply_change", **APPLY)
    lines = [json.loads(x) for x in capsys.readouterr().out.splitlines()]
    [line] = [x for x in lines if x.get("record") == "change_applied"]
    assert (line["case_id"], line["run_id"], line["approval_id"], line["change_text"]) == (CASE, "r-9", key, CHANGE)


# ---------------------------------------------------------------- NLM
def test_nlm_sagemaker_endpoint_is_labelled():
    sm = FakeSageMaker({"choices": [{"message": {"content": " Tilt aims the beam. "}}]})
    w = World(nlm_obj=nlm.NLM(env={"NLM_ENDPOINT": "team13-otel-tuned", "NLM_STANDIN_MODEL": "x"},
                              client_factory=lambda svc: {"sagemaker-runtime": sm}[svc]))
    out = w.call("nlm___ask", case_id=CASE, question="What is tilt?")
    assert out["answer"] == "Tilt aims the beam."
    assert out["answered_by"]["kind"] == "sagemaker-endpoint" and out["answered_by"]["stand_in"] is False
    body = json.loads(sm.calls[0]["Body"])
    assert sm.calls[0]["EndpointName"] == "team13-otel-tuned" and body["messages"][1]["content"] == "What is tilt?"
    assert w.call("nlm___ask_nlm", case_id=CASE, question="q")["answer"]  # the kit's tool name, same code
    assert "not being served" in w.call("nlm___ask", case_id=CASE, question="q", model_version="v7")["error"]


def test_nlm_standin_is_labelled_as_standin():
    br = FakeBedrock("A stand-in answer.")
    w = World(nlm_obj=nlm.NLM(env={"NLM_STANDIN_MODEL": "openai.gpt-oss-20b-1:0"},
                              client_factory=lambda svc: {"bedrock-runtime": br}[svc]))
    out = w.call("nlm___ask", case_id=CASE, question="What is SINR?")
    assert out["answer"] == "A stand-in answer."
    assert out["answered_by"]["kind"] == "bedrock-standin" and out["answered_by"]["stand_in"] is True
    assert out["answered_by"]["model_label"] == "stand-in: NOT our tuned model"
    assert br.calls[0]["modelId"] == "openai.gpt-oss-20b-1:0"


def test_nlm_tgi_payload_and_parsing():
    sm = FakeSageMaker([{"generated_text": "ok"}])
    n = nlm.NLM(env={"NLM_ENDPOINT": "e", "NLM_PAYLOAD": "tgi"}, client_factory=lambda s: sm)
    assert n.ask("q")["answer"] == "ok"
    assert "inputs" in json.loads(sm.calls[0]["Body"])
    with pytest.raises(ValueError):
        nlm.parse_generation({"nothing": 1})


# ---------------------------------------------------------------- follow-ups: schemas, counterfactual, contracts
def test_case_id_is_optional_in_every_schema_so_cedar_decides():
    """A call without case_id must reach Cedar (the case-id forbid decides and records it)."""
    seen = set()
    for target, tools in catalog.targets().items():
        for t in tools:
            full = catalog.full(target, t["name"])
            seen.add(full)
            schema = t["inputSchema"]
            assert "case_id" in schema["properties"] and "case_id" not in schema.get("required", []), full
    assert set(catalog.all_tools()) <= seen
    apply_schema = next(t for t in catalog.ran_tool_schemas() if t["name"] == "apply_change")["inputSchema"]
    assert apply_schema["required"] == ["item_id", "change"]


def _apply_once(twin):
    w = World(twin=twin, run_id="r-cf")
    w.call("ran___predict", case_id=CASE, item_id="step-0", change=CHANGE)
    w.approved(APPLY)
    out = w.call("ran___apply_change", **APPLY)
    return w, out


def test_counterfactual_from_twin_matches_the_two_replay_fallback():
    w_cf, cf = _apply_once(FakeTwinCF)
    w_rp, rp = _apply_once(FakeTwin)
    assert w_cf.call("ran___observe", case_id=CASE)["twin"]["step"] == 1
    for k in ("baseline_kpis", "measured_kpis", "measured_delta", "prediction_error", "direction_agree"):
        assert cf[k] == rp[k], k
    assert w_cf.case_doc()["action_log"] == w_rp.case_doc()["action_log"]


def test_change_records_validate_against_team13_contracts():
    """Every record shape apply writes (scored, clipped, no prediction) passes W1's A15 model."""
    contracts = pytest.importorskip("team13_contracts")
    validate = getattr(contracts, "validate_record", None) or contracts.parse_record
    w = World(run_id="r-contracts")
    w.call("ran___predict", case_id=CASE, item_id="step-0", change=CHANGE)
    w.approved(APPLY)
    w.call("ran___apply_change", **APPLY)                                   # scored
    big = dict(APPLY, item_id="step-1", change="cell=1;d_power_db=+5.0;d_tilt_deg=+0.0")
    w.call("ran___predict", case_id=CASE, item_id="step-1", change=big["change"])
    w.approved(big)
    w.call("ran___apply_change", **big)                                     # clipped
    plain = dict(APPLY, item_id="step-2", change="cell=0;d_power_db=-1.0;d_tilt_deg=+0.0")
    w.approved(plain)
    w.call("ran___apply_change", **plain)                                   # no prediction
    recs = w.records()
    assert [r["unscored_reason"] for r in recs] == [None, "clipped", "no_prediction"]
    for rec in recs:
        validate(rec)
    bad = dict(recs[0], measured_delta=dict(recs[0]["measured_delta"], coverage_ratio=0.5))
    with pytest.raises(Exception):
        validate(bad)   # the validator really checks (measured_delta must equal measured - baseline)
