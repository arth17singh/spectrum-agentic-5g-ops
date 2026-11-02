import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent))

from scorecard_fixtures import GOLDEN_STATES, FakeReader  # noqa: E402
from spectrum.slices.scorecard import router, routes  # noqa: E402


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(routes, "_reader", lambda: FakeReader(n_items=40))
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_router_prefix():
    assert router.prefix == "/api/scorecard"
    assert {r.path for r in router.routes} == {"/api/scorecard/units", "/api/scorecard/run/{run_id}"}


def test_units_golden(client):
    r = client.get("/api/scorecard/units")
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/json")
    j = r.json()
    assert {u["run_id"]: u["state"] for t in j["tracks"] for u in t["units"]} == GOLDEN_STATES
    assert j["totals"] == {"declared": 5, "passed": 2, "failed": 1, "not_checked": 1, "inconsistent": 1}
    assert j["tracks"][1]["track"] == "track-3" and j["tracks"][1]["declared"] == 4 and j["tracks"][1]["passed"] == 1
    assert [o["run_id"] for o in j["orphans"]] == ["syn-orphan"]
    assert j["source"]["synthetic"] is True
    deny = next(u for t in j["tracks"] for u in t["units"] if u["run_id"] == "syn-t3-deny")
    assert deny["gain"]["value"] == 1.3 and deny["gain"]["ci_low"] == 1.1 and deny["gain"]["ci_high"] == 1.5


def test_units_empty(monkeypatch):
    monkeypatch.setattr(routes, "_reader", lambda: FakeReader(declared={}, results={}, runs=[]))
    app = FastAPI()
    app.include_router(router)
    j = TestClient(app).get("/api/scorecard/units").json()
    assert j["empty"] is True and j["tracks"] == [] and j["totals"]["declared"] == 0


def test_units_reads_fresh_each_call(monkeypatch):
    calls = []
    monkeypatch.setattr(routes, "_reader", lambda: calls.append(1) or FakeReader())
    app = FastAPI()
    app.include_router(router)
    c = TestClient(app)
    c.get("/api/scorecard/units")
    c.get("/api/scorecard/units")
    assert len(calls) == 2


def test_run_detail(client):
    j = client.get("/api/scorecard/run/syn-t3-retention").json()
    assert j["command"] == "python3 tools/show_record.py --run-id syn-t3-retention"
    assert j["row"]["state"] == "inconsistent"
    assert j["declared"]["target"]["value"] == 0.8 and j["result"]["passed"] is True
    assert j["item_records"] == 40


def test_run_detail_not_checked(client):
    j = client.get("/api/scorecard/run/syn-t3-trace").json()
    assert j["row"]["state"] == "not checked" and j["result"] is None and j["item_records"] is None
    assert j["command"].endswith("--run-id syn-t3-trace")


def test_run_detail_without_declaration(client):
    j = client.get("/api/scorecard/run/syn-orphan").json()
    assert j["row"] is None and "not a scored unit" in j["note"] and j["result"]["synthetic"] is True


def test_run_unknown_404(client):
    assert client.get("/api/scorecard/run/nope").status_code == 404


@pytest.mark.parametrize("bad", ["..", "a;b", "a b", "-x", ".hidden", "x" * 200])
def test_run_invalid_id_400_or_404_never_echoed_as_command(client, bad):
    r = client.get(f"/api/scorecard/run/{bad}")
    assert r.status_code in (400, 404) and "command" not in r.json()


def test_unreadable_run_ids_are_ignored(monkeypatch):
    monkeypatch.setattr(routes, "_reader", lambda: FakeReader(runs=["syn-t1-acc", "bad id; rm", {"run_id": "syn-t3-pause"}, 7, "syn-t1-acc"]))
    app = FastAPI()
    app.include_router(router)
    j = TestClient(app).get("/api/scorecard/units").json()
    assert sorted(u["run_id"] for t in j["tracks"] for u in t["units"]) == ["syn-t1-acc", "syn-t3-pause"]


def test_reader_failure_is_503_json(monkeypatch):
    def boom():
        raise RuntimeError("secret path /x/y")

    monkeypatch.setattr(routes, "_reader", boom)
    app = FastAPI()
    app.include_router(router)
    c = TestClient(app)
    for path in ("/api/scorecard/units", "/api/scorecard/run/syn-t1-acc"):
        r = c.get(path)
        assert r.status_code == 503 and r.json() == {"detail": "evidence unavailable: RuntimeError"}
