import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from spectrum.slices._shared.redact import RedactMiddleware
from spectrum.slices.realrun import logic, router

T = {"metric": "pass_rate", "op": ">=", "unit": "ratio", "value": 0.9}


def w(root: Path, rel: str, obj) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj))


def store(tmp_path: Path) -> Path:
    runner = {"entry": ["--client", "local-hf"], "source_uri": "s3://bucket-123456789012-x/src.tgz"}
    w(tmp_path, "declared/t1.interface-otel-base-1.json", {
        "run_id": "t1.interface-otel-base-1", "unit": "t1.interface", "track": "track-1", "model_id": "m-base",
        "target": T, "declared_at": "2026-11-03T04:00:00Z", "twin": {"config": {"runner": runner}}})
    w(tmp_path, "results/t1.interface-otel-base-1.json", {
        "run_id": "t1.interface-otel-base-1", "unit": "t1.interface", "metric": "pass_rate", "value": 0.5, "passed": False,
        "n_items": 10, "checked_at": "2026-11-03T04:10:00Z",
        "details": {"pass_rate": 0.5, "latency_and_price": {"p95_ms": 900.0, "cost_per_1k_usd": 0.4}}})
    w(tmp_path, "records/t1.interface-otel-base-1/job/abc.json", {
        "job_name": "j-1", "BillableTimeInSeconds": 360, "CreationTime": "2026-11-03T04:00:05Z",
        "arn": "arn:aws:sagemaker:us-east-2:123456789012:training-job/j-1"})
    w(tmp_path, "outputs/t1.interface-otel-base-1/job-map.json", {"job_name": "j-1", "price_usd_per_hour": 1.5})
    return tmp_path


def test_track1_matrix_shows_missing_variants_as_absent(tmp_path):
    st = logic.Store(store(tmp_path))
    t1 = logic.track1(st, st.run_ids())
    unit = next(u for u in t1["units"] if u["unit"] == "t1.interface")
    assert unit["cells"]["base"]["state"] == "fail" and unit["cells"]["base"]["p95_ms"] == 900.0
    assert unit["cells"]["tuned"] is None and unit["cells"]["bedrock"] is None
    s = logic.run_summary(st, "t1.interface-otel-base-1")
    assert s["jobs"][0]["cost_usd"] == 360 * 1.5 / 3600


def test_routes_redact_and_404(tmp_path, monkeypatch):
    monkeypatch.setenv("SPECTRUM_RECORDS_ROOT", str(store(tmp_path)))
    app = FastAPI()
    app.add_middleware(RedactMiddleware)
    app.include_router(router)
    c = TestClient(app)
    body = c.get("/api/realrun/run/t1.interface-otel-base-1").text
    assert "123456789012" not in body and "<acct>" in body
    assert c.get("/api/realrun/run/nope").status_code == 404
    assert c.get("/api/realrun/run/..%2Fx").status_code in (400, 404)
    assert c.get("/api/realrun/overview").json()["t1"]["columns"][0]["variant"] == "base"
