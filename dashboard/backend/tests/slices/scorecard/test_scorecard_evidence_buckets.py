import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from spectrum.slices.scorecard import router

T = {"metric": "pass_rate", "op": ">=", "unit": "ratio", "value": 0.5}


def w(root: Path, rel: str, obj) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj))


def run(root, rid, **det):
    w(root, f"declared/{rid}.json", {"run_id": rid, "unit": "t3.x", "track": "track-3", "target": T, "n_items": 1})
    w(root, f"results/{rid}.json", {"run_id": rid, "unit": "t3.x", "metric": "pass_rate", "value": 1.0, "passed": True,
                                     "n_items": 1, "target": T, "details": det})


def check(root, rid, exit_code, parts):
    w(root, f".sync/checks/{rid}.json", {"exit_code": exit_code, "summary": {"parts": {
        k: {"state": v, "lines": [f"FAIL: {k} broke"] if v == "FAIL" else ["x"]} for k, v in parts.items()}}})


def test_buckets_are_named_and_superseded_is_apart(tmp_path, monkeypatch):
    ok = {"a": "PASS", "b": "PASS", "c": "PASS", "d": "PASS", "e": "PASS"}
    run(tmp_path, "r-pass"); check(tmp_path, "r-pass", 0, ok)
    run(tmp_path, "r-bfail"); check(tmp_path, "r-bfail", 1, {**ok, "b": "FAIL"})
    run(tmp_path, "r-unv"); check(tmp_path, "r-unv", 2, {**ok, "c": "CANNOT CHECK"})
    run(tmp_path, "r-inc", complete=False)
    run(tmp_path, "r-old")
    w(tmp_path, "superseded.json", {"superseded": [{"run_ids": ["r-old"], "reason": "Harness defect: x", "superseded_by": {"run_ids": []}}]})
    monkeypatch.setenv("SPECTRUM_RECORDS_ROOT", str(tmp_path))
    app = FastAPI()
    app.include_router(router)
    card = TestClient(app).get("/api/scorecard/units").json()
    states = {u["run_id"]: u["state"] for t in card["tracks"] for u in t["units"]}
    assert states == {"r-pass": "pass", "r-bfail": "check failed", "r-unv": "unverified", "r-inc": "incomplete", "r-old": "superseded"}
    t = card["totals"]
    assert (t["declared"], t["passed"], t["check_failed"], t["unverified"], t["incomplete"], t["superseded"], t["inconsistent"]) == (4, 1, 1, 1, 1, 1, 0)
