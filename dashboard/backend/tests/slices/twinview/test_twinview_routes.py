"""Twin view routes over the committed SYNTHETIC fixtures (read through the shared RecordsReader)."""

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from spectrum.slices import twinview
from spectrum.slices._shared.records import RecordsReader
from spectrum.slices.twinview import routes

FIX = Path(__file__).parent / "fixtures"
SILOS = Path(__file__).resolve().parents[5] / "silos"


def client(monkeypatch, reader=None, silos=SILOS):
    monkeypatch.setattr(routes, "_reader", lambda: reader if reader is not None else RecordsReader(str(FIX)))
    monkeypatch.setenv("SPECTRUM_SILOS_DIR", str(silos))
    app = FastAPI()
    app.include_router(twinview.router)
    return TestClient(app)


def test_cases_lists_both_fixture_cases_as_synthetic(monkeypatch):
    j = client(monkeypatch).get("/api/twinview/cases").json()
    assert j["n_cases"] == 2 and j["n_changes"] == 5
    by = {c["case_id"]: c for c in j["cases"]}
    assert by["case-twin-01"]["rung"]["code"] == "R1" and by["case-twin-02"]["rung"]["code"] == "R2"
    assert all(c["provenance"]["state"] == "synthetic" for c in j["cases"])
    assert j["provenance"]["label"] == "SYNTHETIC" and j["field_spec"]["field"] == "twin_state"
    json.dumps(j, allow_nan=False)


def test_case_windows_carry_four_cells_before_and_after(monkeypatch):
    j = client(monkeypatch).get("/api/twinview/case/case-twin-01").json()
    assert j["run_id"] == "t3-twin-maveric-20261102-a" and len(j["windows"]) == 3
    w = j["windows"][0]
    assert w["has_state"] and [c["name"] for c in w["before"]["per_cell"]] == ["S1/B3/C1", "S2/B3/C1", "S3/B3/C1", "S4/B3/C1"]
    for c in w["after"]["per_cell"]:
        assert {"power_dbm", "tilt_deg", "ue_count", "coverage_share", "sla_share", "sinr_p10_db", "rsrp_mean_dbm"} <= set(c)
    # the first change asked for a tilt the safety filter did not pass: shown, not hidden
    assert w["unscored_reason"] == "clipped" and w["cells"][0]["differs"] is True
    assert w["cells"][0]["applied"] == {"d_power_db": 3.0, "d_tilt_deg": 0.0}
    sources = [x["approval"]["source"] for x in j["windows"]]
    assert sources == ["change.approved_by", "item.extra.approval", "decision"]
    assert j["windows"][0]["approval"]["by"] == "person:navigator-1" and j["windows"][0]["approval"]["at"]
    assert j["windows"][2]["approval"]["by"] is None and j["windows"][2]["approval"]["allowed_at"]
    assert j["provenance"]["state"] == "synthetic" and j["provenance"]["n_records"] == 6  # the 3 windows and the 3 records that name their approvers; the declaration is no row here
    assert any(m["field"] == "approved_by" for m in j["missing"])


def test_rehearsal_case_shows_gap_and_unrecorded_approver(monkeypatch):
    j = client(monkeypatch).get("/api/twinview/case/case-twin-02").json()
    assert j["rung"]["code"] == "R2" and j["rung"]["preset"] == "viavi-like" and "change.twin.preset" in j["rung"]["basis"]
    assert [w["gap_steps"] for w in j["windows"]] == [0, 1]
    assert j["windows"][0]["probe"] is True and j["windows"][1]["approval"]["source"] is None


def test_unknown_case_is_json_404(monkeypatch):
    r = client(monkeypatch).get("/api/twinview/case/nope")
    assert r.status_code == 404 and r.headers["content-type"].startswith("application/json")


def test_empty_store_is_an_honest_empty_list(monkeypatch, tmp_path):
    j = client(monkeypatch, RecordsReader(str(tmp_path))).get("/api/twinview/cases").json()
    assert j["cases"] == [] and j["provenance"]["state"] == "empty"


def test_reader_failure_is_json_503(monkeypatch):
    class Broken:
        def all_records(self, kind):
            raise OSError("evidence dir unreadable")

        def runs(self):
            raise OSError("evidence dir unreadable")

    c = client(monkeypatch, Broken())
    for path in ("/api/twinview/cases", "/api/twinview/case/x", "/api/twinview/consortium"):
        r = c.get(path)
        assert r.status_code == 503 and "unreadable" in r.json()["detail"]


def test_consortium_shows_manifest_facts_and_not_run_yet(monkeypatch, tmp_path):
    j = client(monkeypatch, RecordsReader(str(tmp_path))).get("/api/twinview/consortium").json()
    silos = {s["silo"]: s for s in j["facts"]["silos"]}
    assert [silos[s]["clusters"] for s in "ABC"] == [8, 3, 1] and [silos[s]["cells"] for s in "ABC"] == [32, 12, 4]
    assert silos["A"]["mix"]["lean"] == "mostly tilt changes" and silos["B"]["mix"]["lean"] == "mostly power changes"
    assert len(j["facts"]["shared_test"]["sha256"]) == 64
    assert j["runs"] == [] and j["runs_state"] == "not run yet"
    assert j["preview"]["label"] == "LOCAL" and "not the FL service" in j["preview"]["what"]
    assert j["hub"]["name"] == "the FL service"
    text = json.dumps(j)
    assert "seeds\": [" not in text and "user_share" not in text  # aggregates only: no seed lists, no columns


def test_consortium_reads_a_t2_fl_result_and_checks_the_test_set(monkeypatch, tmp_path):
    manifest = json.loads((SILOS / "manifest.json").read_text())
    shared = next(f["sha256"] for f in manifest["files"] if f["role"] == "shared_test" and f["split"] == "disjoint")
    (tmp_path / "results").mkdir()
    rec = {"schema_version": "1", "kind": "result", "run_id": "t2-fl-x", "team": "team13", "created_at": "2026-11-02T22:00:00Z",
           "aws_time": None, "aws_time_source": None, "synthetic": False,
           "unit": "t2.fl", "metric": "skill", "value": 0.44, "passed": True,
           "details": {"split": "disjoint", "shared_test_sha256": shared, "rounds_completed": 20, "final_round": 20,
                       "dp": {"enabled": True, "epsilon": 8.0, "delta": 1e-5},
                       "per_silo": {"A": {"local_only_skill": 0.4, "federated_skill": 0.43, "n_train_rows": 24000}}}}
    (tmp_path / "results" / "t2-fl-x.json").write_text(json.dumps(rec))
    # the FL job attests the file in CloudWatch: that is the run's AWS time (the result carries none of its own)
    import hashlib
    (tmp_path / ".sync" / "attest").mkdir(parents=True)
    sha = hashlib.sha256((tmp_path / "results" / "t2-fl-x.json").read_bytes()).hexdigest()
    (tmp_path / ".sync" / "attest" / "t2-fl-x.json").write_text(json.dumps({"files": {
        "results/t2-fl-x.json": {"job": "j", "log_time": "2026-11-02T22:00:05Z", "sha256": sha, "state": "match"}}}))
    j = client(monkeypatch, RecordsReader(str(tmp_path))).get("/api/twinview/consortium").json()
    assert j["runs_state"] == "run" and j["runs_provenance"]["state"] == "real"
    assert j["runs_provenance"]["aws_time_summary"]["by_source"]["cloudwatch_attest"] == 1
    assert j["runs"][0]["aws_time_source"] == "cloudwatch_attest" and j["runs"][0]["aws_time"] == "2026-11-02T22:00:05Z"
    run = j["runs"][0]
    assert run["same_test_set"] is True and run["dp"]["epsilon"] == 8.0 and run["final_round"] == 20
    assert run["per_silo"]["A"]["federated_skill"] == 0.43


def test_consortium_without_manifest_says_so(monkeypatch, tmp_path):
    j = client(monkeypatch, RecordsReader(str(tmp_path)), silos=tmp_path).get("/api/twinview/consortium").json()
    assert j["facts"] is None and "not found" in j["facts_note"] and j["preview"] is None


def test_bad_split_is_400(monkeypatch):
    assert client(monkeypatch).get("/api/twinview/consortium", params={"split": "x"}).status_code == 400
