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
