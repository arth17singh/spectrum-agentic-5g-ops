"""Trace routes with a fake RecordsReader (the shared reader belongs to another stream)."""

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from spectrum.slices import trace

FIX = Path(__file__).parent / "fixtures"


class FakeReader:
    def __init__(self, names=("complete", "deny", "paused", "no_approval")):
        self.data = {"decision": [], "change": [], "control": []}
        for n in names:
            for kind, rows in json.loads((FIX / f"{n}.json").read_text()).items():
                self.data[kind] += rows

    def all_records(self, kind):
        return list(self.data.get(kind, []))

    def source(self):
        return {"kind": "local", "uri": "fixtures"}


def client(monkeypatch, reader):
    monkeypatch.setattr(trace, "_reader", lambda: reader)
    app = FastAPI()
    app.include_router(trace.router)
    return TestClient(app)


def test_cases_lists_all_four(monkeypatch):
    r = client(monkeypatch, FakeReader()).get("/api/trace/cases")
    assert r.status_code == 200
    j = r.json()
    assert j["n_cases"] == 4 and j["n_records"] == 24 + 3 + 1  # decisions + changes + controls
    row = next(c for c in j["cases"] if c["case_id"] == "case-complete")
    assert set(row) >= {"case_id", "run_id", "first_time", "last_time", "n_events", "denies", "allows", "pauses", "applied"}
    assert j["source"] == {"kind": "local", "uri": "fixtures"}


def test_case_detail_complete(monkeypatch):
    j = client(monkeypatch, FakeReader()).get("/api/trace/case/case-complete").json()
    assert j["run_id"] == "run-synthetic-1" and j["check"]["passed"] is True
    assert j["show_record"] == "python3 tools/show_record.py --run-id run-synthetic-1 --item case-complete"
    assert len(j["timeline"]) == 8 and j["synthetic"] is True


def test_case_detail_flags_missing_approval(monkeypatch):
    j = client(monkeypatch, FakeReader()).get("/api/trace/case/case-no-approval").json()
    assert j["check"]["passed"] is False and j["check"]["missing"] == ["approval"]


def test_unknown_case_is_json_404(monkeypatch):
    r = client(monkeypatch, FakeReader()).get("/api/trace/case/nope")
    assert r.status_code == 404 and r.headers["content-type"].startswith("application/json")
    assert "unknown case" in r.json()["detail"]


def test_no_records_is_an_honest_empty_list(monkeypatch):
    j = client(monkeypatch, FakeReader(names=())).get("/api/trace/cases").json()
    assert j["cases"] == [] and j["n_records"] == 0


def test_reader_failure_is_json_503_not_empty(monkeypatch):
    class Broken:
        def all_records(self, kind):
            raise OSError("evidence dir unreadable")

    c = client(monkeypatch, Broken())
    for path in ("/api/trace/cases", "/api/trace/case/x"):
        r = c.get(path)
        assert r.status_code == 503 and "unreadable" in r.json()["detail"]


def test_missing_shared_reader_is_json_503(monkeypatch):
    app = FastAPI()
    app.include_router(trace.router)
    r = TestClient(app).get("/api/trace/cases")
    assert r.status_code in (200, 503)  # 503 until the shared reader module exists; 200 once it does
    assert r.headers["content-type"].startswith("application/json")


@pytest.mark.parametrize("path", ["/api/trace/cases", "/api/trace/case/case-deny"])
def test_responses_are_json(monkeypatch, path):
    assert client(monkeypatch, FakeReader()).get(path).headers["content-type"].startswith("application/json")


# ---------------------------------------------------------------------------------------------- provenance chip

def _flip(reader, *, synthetic, aws_time=True):
    """Make the fixture records look like a real run (or not): synthetic flag and AWS time."""
    for rows in reader.data.values():
        for r in rows:
            r["synthetic"] = synthetic
            if not aws_time:
                r["aws_time"] = None
    return reader


def _prov(monkeypatch, reader):
    j = client(monkeypatch, reader).get("/api/trace/cases").json()
    return j["provenance"], j


def test_provenance_synthetic_fixtures_are_never_real(monkeypatch):
    p, j = _prov(monkeypatch, FakeReader())
    assert p["state"] == "synthetic" and p["label"] == "SYNTHETIC"
    assert p["note"] == "Synthetic records (test fixtures): not a real run."
    assert p["claim"] == "" and "simulated" not in p["note"].lower()
    assert p["n_synthetic"] == p["n_records"] > 0 and p["n_unverified"] == 0
    assert all(c["provenance"]["state"] == "synthetic" for c in j["cases"])


def test_provenance_real_only_when_every_record_has_aws_time_and_is_not_synthetic(monkeypatch):
    p, j = _prov(monkeypatch, _flip(FakeReader(), synthetic=False))
    assert p["state"] == "real" and p["label"] == "REAL" and p["claim"] == "Nothing here is simulated."
    assert p["n_synthetic"] == 0 and p["n_unverified"] == 0
    assert all(c["provenance"]["state"] == "real" and c["synthetic"] is False for c in j["cases"])
    # the case detail agrees with the list
    d = client(monkeypatch, _flip(FakeReader(), synthetic=False)).get("/api/trace/case/case-complete").json()
    assert d["provenance"]["state"] == "real" and all(e["real"] and not e["synthetic"] for e in d["timeline"])


def test_provenance_mixed_counts_synthetic_records(monkeypatch):
    real = _flip(FakeReader(names=("complete",)), synthetic=False)
    syn = FakeReader(names=("no_approval",))
    for kind in real.data:
        real.data[kind] += syn.data[kind]
    p, j = _prov(monkeypatch, real)
    assert p["state"] == "mixed" and p["label"] == "MIXED" and p["claim"] == ""
    # complete = 7 decisions + 1 change (real); no-approval = 5 decisions + 1 change (synthetic)
    assert p["note"] == "6 of 14 records synthetic." and p["n_records"] == 14


def test_provenance_unverified_when_aws_time_missing(monkeypatch):
    p, _ = _prov(monkeypatch, _flip(FakeReader(), synthetic=False, aws_time=False))
    assert p["state"] == "unverified" and p["claim"] == "" and p["n_unverified"] == p["n_records"] > 0
    assert "no AWS-recorded event time" in p["note"]


def test_provenance_empty_and_control_counted_once(monkeypatch):
    assert _prov(monkeypatch, FakeReader(names=()))[0]["state"] == "empty"
    p, _ = _prov(monkeypatch, FakeReader(names=("paused", "complete")))
    # paused case: 4 decisions + 1 control; complete case: 7 decisions + 1 change; the control (20:03:10) sits
    # outside the complete case's window, so it is counted exactly once
    assert p["n_records"] == 4 + 1 + 7 + 1
