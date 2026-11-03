import json
import sys
import types

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from spectrum.slices.trustgate import router
from spectrum.slices.trustgate.routes import get_reader
from slices.trustgate.trustgate_fakes import FakeReader


def client(reader) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_reader] = lambda: reader
    return TestClient(app)


def test_models_lists_predictors_with_status_pause_and_probes():
    r = client(FakeReader()).get("/api/trustgate/models")
    assert r.status_code == 200
    j = r.json()
    assert j["error"] is None and j["source"] == {"kind": "fixtures"}
    by = {(m["model_id"], m["version"]): m for m in j["models"]}
    assert set(by) == {("t2-response-mlp", "v1"), ("t2-response-mlp", "v1-degraded")}
    h, d = by[("t2-response-mlp", "v1")], by[("t2-response-mlp", "v1-degraded")]
    assert h["paused"] is False and h["pauses"] == [] and h["pause_below"] == 0.55 and h["synthetic"] is True
    assert [p["by_kind"] for p in d["pauses"]] == ["trust-gate", "person"]
    assert "rows" not in h and len(h["points"]) == 10
    assert len(j["probes"]) == 2 and all(p["probe"] for p in j["probes"])


def test_changes_returns_rows_with_predicted_and_measured_deltas():
    r = client(FakeReader()).get("/api/trustgate/changes", params={"model_id": "t2-response-mlp", "version": "v1-degraded"})
    assert r.status_code == 200
    j = r.json()
    assert j["version"] == "v1-degraded" and len(j["rows"]) == 5
    row = j["rows"][2]
    assert {k["key"] for k in row["kpis"]} == {"coverage_ratio", "sla_ratio", "sinr_p10_db", "sinr_mean_db"}
    assert row["error_summary"] == 1.5 and row["direction_agree"] is False and row["over_tau"] is True
    assert j["thresholds"]["tau"] == 1.0
    json.dumps(j, allow_nan=False)  # strictly JSON: no NaN / Infinity


def test_changes_validates_and_404s():
    c = client(FakeReader())
    assert c.get("/api/trustgate/changes").status_code == 422
    assert c.get("/api/trustgate/changes", params={"model_id": "t2-response-mlp"}).status_code == 422
    assert c.get("/api/trustgate/changes", params={"model_id": "../../etc", "version": "v1"}).status_code == 422
    r = c.get("/api/trustgate/changes", params={"model_id": "nope", "version": "v1"})
    assert r.status_code == 404 and "nope" in r.json()["detail"]


def test_ladder_returns_both_rungs_sets():
    j = client(FakeReader()).get("/api/trustgate/ladder").json()
    assert {(l["version"], l["top_label"]) for l in j["ladders"]} == {("v3", "REHEARSAL (Viavi-like Maveric)"), ("v4", "VIAVI")}
    assert all(l["synthetic"] for l in j["ladders"])


def test_synthetic_viavi_fixture_is_not_a_real_viavi_result():
    ls = client(FakeReader()).get("/api/trustgate/ladder").json()["ladders"]
    assert not any(l["real_viavi"] for l in ls)
    v = next(l for l in ls if l["top_label"] == "VIAVI")
    assert v["has_viavi"] is True and v["synthetic"] is True


def test_no_viavi_run_means_no_viavi_ladder():
    j = client(FakeReader(only_runs={"t3-ladder-rehearsal-20261102-a"})).get("/api/trustgate/ladder").json()
    assert [l["top_label"] for l in j["ladders"]] == ["REHEARSAL (Viavi-like Maveric)"]
    assert not any(l["has_viavi"] for l in j["ladders"])


def test_empty_store_gives_empty_lists_not_errors():
    c = client(FakeReader(only_runs=set()))
    assert c.get("/api/trustgate/models").json()["models"] == []
    assert c.get("/api/trustgate/ladder").json()["ladders"] == []


def test_unreadable_store_is_reported_in_band():
    class Broken:
        def all_records(self, kind):
            raise OSError("boom")

        def runs(self):
            raise OSError("boom")

        def source(self):
            raise OSError("boom")

    j = client(Broken()).get("/api/trustgate/models").json()
    assert j["models"] == [] and "change" in j["error"] and "result" in j["error"] and j["source"] is None


def test_default_reader_is_imported_lazily(monkeypatch):
    for name in ("spectrum.slices", "spectrum.slices._shared"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    mod = types.ModuleType("spectrum.slices._shared.records")
    mod.RecordsReader = FakeReader  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "spectrum.slices._shared.records", mod)
    app = FastAPI()
    app.include_router(router)  # no override: the real dependency runs
    assert TestClient(app).get("/api/trustgate/ladder").status_code == 200


def test_missing_shared_reader_fails_loudly_not_with_fake_data():
    try:
        import spectrum.slices._shared.records  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("W7a's shared reader is present")
    with pytest.raises(ImportError):
        get_reader()
