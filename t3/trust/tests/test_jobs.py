"""Job entry points (AWS re-measurements): fakes only; no twin SDK, no AWS."""

import json

import pytest

from t3.trust import jobs
from t3.trust.retention import wilson

from .conftest import CHANGES, FakeCFTwin, FakePredictor, fake_kpis

from t3.trust._evlib import get_evlib

evlib = get_evlib()
SHA40 = "a" * 40


def test_wilson_interval():
    lo, hi = wilson(20, 20)
    assert hi == 1.0 and 0.83 < lo < 0.84
    lo, hi = wilson(6, 20)
    assert lo < 0.3 < hi and wilson(0, 0) is None


def gate_run(model, thresholds, seeds=(1, 2, 3)):
    p = FakePredictor("v1") if model == "healthy" else FakePredictor("v1-bad", -1)
    return jobs.run_gate_eval(version="v1", model=model, run_id="t3.gate-test", seeds=list(seeds),
                              twin_factory=FakeCFTwin, kpis=fake_kpis, predictor=p, changes=CHANGES,
                              thresholds_path=thresholds, testset_sha="b" * 64, code_sha=SHA40)


def test_gate_eval_items_and_result_recompute(thresholds):
    for model, metric, value, passed in (("healthy", "fail_count", 0.0, True), ("degraded", "pass_count", 3.0, False)):
        out = gate_run(model, thresholds)
        res, items = out["result"], out["items"]
        assert res["metric"] == metric and res["value"] == value and res["passed"] is passed  # 3 of >= 19: missed
        assert res["n_items"] == len(items) == 3 and len({i["item_id"] for i in items}) == 3
        assert evlib.summarize_items(out["sink"].records["item"], metric) == value  # what check_unit recomputes
        d = res["details"]
        assert d["thresholds_version"] == "v1" and "sha256" in d["thresholds"] and len(d["paused_ci95"]) == 2
        assert "ed7d3c4" in d["note"] and "used once before" in d["note"]
        out["sink"].validate()
        evlib.validate(evlib._envelope(dict(res), "result", "t3.gate-test"))
        changes = out["sink"].records["change"]
        assert len({c["case_id"] for c in changes}) == 3 and all(c["synthetic"] is False for c in changes)
        if model == "degraded":
            assert all(c["synthetic"] is True for c in out["sink"].records["control"])  # dry-run pause


LADDER_RUNGS = [{"name": "maveric", "factory": FakeCFTwin, "scenario": "s", "config": "official", "twin_kind": "maveric-inprocess"},
                {"name": "rehearsal", "factory": FakeCFTwin, "scenario": "s", "config": "viavi-like", "twin_kind": "maveric-inprocess"}]


def ladder_run(model="healthy"):
    p = FakePredictor("v1") if model == "healthy" else FakePredictor("v1-bad", -1)
    return jobs.run_ladder_job(model=model, run_id="t3.ladder-test", seeds=[1, 2, 3], rungs=LADDER_RUNGS, kpis=fake_kpis,
                               predictor=p, changes=CHANGES[:5], testset_sha="c" * 64, code_sha=SHA40)


def test_ladder_job_items_are_one_per_seed_and_details_carry_the_shift(thresholds):
    out = ladder_run()
    res = out["result"]
    items = out["sink"].records["item"]
    assert set(out["sink"].records) == {"item"}  # no extra record kind
    assert len(items) == 3 and all(i["item_id"].startswith("healthy/seed-") for i in items)
    for it in items:  # the R1 gain is the score; both rungs' cases ride in extra
        assert set(it["extra"]["rungs"]) == {"maveric", "rehearsal"}
        assert it["score"] == it["extra"]["rungs"]["maveric"]["gain"]
    assert res["unit"] == "t3.ladder" and res["metric"] == "gain_ci_low" and res["target"]["op"] == ">"
    g = res["details"]["gain"]
    assert g["value"] == pytest.approx(evlib.summarize_items(items, "mean_score")) and res["value"] == g["ci_low"]
    assert res["passed"] is (g["ci_low"] > 0) and g["paired_n"] == 3 and g["method"] == "paired-bootstrap-v1"
    d = res["details"]
    assert d["retention"] is None and d["has_real_viavi"] is False and "NOT retention" in d["settings_shift"]["what"]
    assert d["settings_shift"]["point"] == pytest.approx(0.5)
    out["sink"].validate()
    evlib.validate(evlib._envelope(dict(res), "result", "t3.ladder-test"))


def test_ladder_interval_is_what_check_unit_replays():
    """check_unit part e replays details.gain (paired-bootstrap-v1, CI_FIELDS) from the items: same numbers."""
    import importlib.util

    from .conftest import ROOT

    spec = importlib.util.spec_from_file_location("t5_check_unit", ROOT / "tools" / "check_unit.py")
    cu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cu)
    out = ladder_run()
    g = out["result"]["details"]["gain"]
    assert all(k in g for k in cu.CI_FIELDS) and g["method"] == cu.CI_METHOD
    assert cu.CI_METRIC_RE.search(out["result"]["metric"])
    items = out["sink"].records["item"]
    ids = sorted(i["item_id"] for i in items)
    by = {i["item_id"]: i["score"] for i in items}
    value, lo, hi, _ = cu.paired_bootstrap_v1([by[i] for i in ids], ids, g["seed"], g["resamples"], g["level"])
    assert abs(value - g["value"]) <= 1e-9 and abs(lo - g["ci_low"]) <= 1e-9 and abs(hi - g["ci_high"]) <= 1e-9


def test_runner_declarations_for_runs_17_to_19():
    for name, (kind, v, m, ts) in jobs.RUNNER_RUNS.items():
        doc = json.loads((jobs.DECLARATIONS / f"{name}.json").read_text())
        want, entry = jobs.runner_declaration(kind, v, m, ts, doc["twin"]["version"])
        assert doc == want, name
        assert doc["test_set"] == jobs.rel(ts) and (jobs.HERE.parent.parent / doc["test_set"]).is_file()
        assert "--testset " + doc["test_set"] in entry and doc["twin"]["config"]["command"] == "python " + entry
        assert not ({"run_id", "code_sha", "region"} & set(doc))  # the runner adds them
        seeds = json.loads((jobs.HERE.parent.parent / doc["test_set"]).read_text())["seeds"]
        assert seeds == (list(range(3000, 3020)) if kind == "gate" else list(range(4000, 4020)))
        rec = dict(doc)
        rec.pop("test_set")
        rec.update({"run_id": "t3.x", "code_sha": SHA40, "region": "us-east-2", "test_set_sha256": "d" * 64})
        rec = evlib._envelope(rec, "declared", "t3.x")
        rec["declared_at"] = evlib.utc_now()
        evlib.validate(rec)
        contracts_validate(rec)


class FakeS3:
    def __init__(self):
        self.objects = {}

    def put_object(self, Bucket, Key, Body, IfNoneMatch=None, **kw):
        if IfNoneMatch == "*" and Key in self.objects:
            raise RuntimeError("PreconditionFailed")
        self.objects[Key] = Body
        return {}


def test_flush_writes_one_object_per_record_on_s3(thresholds):
    out = gate_run("degraded", thresholds, seeds=(1,))
    s3 = FakeS3()
    store = evlib.S3Store("s3://bucket/evidence/", client=s3)
    paths = out["sink"].flush(store=store)
    n = sum(len(v) for v in out["sink"].records.values())
    assert len(paths) == n == len(s3.objects)
    assert all(k.startswith("evidence/records/t3.gate-test/") and k.count("/") == 4 for k in s3.objects)


def test_flush_writes_jsonl_locally(tmp_path, thresholds):
    out = gate_run("healthy", thresholds, seeds=(1,))
    paths = out["sink"].flush(store=evlib.LocalStore(tmp_path))
    assert sorted(p.rsplit("/", 1)[1] for p in paths) == ["change.jsonl", "item.jsonl"]  # no pause: no control


def test_committed_testsets_and_templates_match_the_builders():
    for path, doc in jobs.build_testsets().items():
        assert json.loads(path.read_text()) == doc, path.name
    for name, doc in jobs.template_docs().items():
        assert json.loads((jobs.DECLARATIONS / f"{name}.template.json").read_text()) == doc, name
    assert len(jobs.template_docs()) == 6


def test_templates_are_valid_declarations_once_filled():
    for name, doc in jobs.template_docs().items():
        d = json.loads(json.dumps(doc).replace(jobs.PLACEHOLDERS["run_id"], "t3.x-run")
                       .replace(jobs.PLACEHOLDERS["code_sha"], SHA40).replace(jobs.PLACEHOLDERS["region"], "us-east-2"))
        rec = evlib._envelope(d, "declared", "t3.x-run")
        rec["declared_at"] = evlib.utc_now()
        evlib.validate(rec)
        assert rec["n_items"] == 20 and len(rec["test_set_sha256"]) == 64, name
        cfg = rec["twin"]["config"]
        assert isinstance(cfg, dict) and cfg["preset"] in ("official", "viavi-like") and cfg["command"].startswith("python -m")
        assert rec["twin"]["kind"] == "maveric-inprocess"
        if name.startswith("gate_eval"):
            assert cfg["references"]["declaration_commit"] == "ed7d3c4" and "used once before" in cfg["note"]
            assert len(cfg["thresholds_sha256"]) == 64
        else:
            assert cfg["measured_not_targeted"] and rec["target"]["metric"] == ("gain_ci_low" if "healthy" in name else "mean_score")
        contracts_validate(rec)


def _contracts():
    """team13_contracts from <repo>/contracts (the job image ships it); None if pydantic is missing."""
    import importlib
    import sys

    from .conftest import ROOT

    try:
        import pydantic  # noqa: F401
    except ImportError:
        return None
    p = str(ROOT / "contracts")
    if p not in sys.path:
        sys.path.insert(0, p)
    return importlib.import_module("team13_contracts")


def contracts_validate(rec):
    tc = _contracts()
    if tc is not None:
        tc.validate_record(rec)


def test_every_job_record_passes_the_contracts(thresholds):
    """With team13_contracts importable (as in the job image): change, control, item and result records."""
    tc = _contracts()
    if tc is None:
        pytest.skip("pydantic not installed")
    for out, run_id in ((gate_run("healthy", thresholds), "t3.gate-test"), (gate_run("degraded", thresholds), "t3.gate-test"),
                        (ladder_run("healthy"), "t3.ladder-test"), (ladder_run("degraded"), "t3.ladder-test")):
        n = 0
        for kind, recs in out["sink"].records.items():
            for r in recs:
                tc.validate_record(r)
                evlib.validate(r)
                n += 1
        tc.validate_record(evlib._envelope(dict(out["result"]), "result", run_id))
        assert n > 0
    kinds = set(gate_run("degraded", thresholds)["sink"].records)
    assert kinds == {"change", "control", "item"}


def test_cli_needs_a_run_id_unless_dry_run(monkeypatch, capsys):
    from t3.trust import job_gate_eval, job_ladder

    for k in ("RUN_ID", "EVIDENCE_RUN_ID"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(SystemExit):
        job_gate_eval.main(["--version", "v1", "--model", "healthy"])
    assert job_gate_eval.main(["--version", "v2", "--model", "degraded", "--print-declaration", "--run-id", "t3.r1"]) == 0
    decl = json.loads(capsys.readouterr().out)
    assert decl["run_id"] == "t3.r1" and decl["twin"]["config"]["thresholds_version"] == "v2" and decl["target"]["op"] == ">="
    monkeypatch.setenv("RUN_ID", "t3.r2")
    assert job_ladder.main(["--model", "degraded", "--print-declaration"]) == 0
    decl = json.loads(capsys.readouterr().out)
    assert decl["run_id"] == "t3.r2" and decl["target"] == {"metric": "mean_score", "op": "<", "value": 0,
                                                            "unit": "objective gain on MAVERIC (R1)"}


def test_dry_run_writes_nothing(tmp_path, monkeypatch, thresholds, capsys):
    monkeypatch.setenv("EVIDENCE_DIR", str(tmp_path))
    out = gate_run("healthy", thresholds, seeds=(1,))
    assert jobs.finish(run_id="t3.gate-test", declaration={"run_id": "t3.gate-test"}, out=out, dry_run=True,
                       declare_first_done=False) == 0
    assert list(tmp_path.iterdir()) == [] and json.loads(capsys.readouterr().out)["dry_run"] is True
