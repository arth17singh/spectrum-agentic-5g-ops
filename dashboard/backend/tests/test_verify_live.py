"""POST/GET /api/evidence/verify/{run_id}: guards, a mocked check_unit (a stand-in script), cache write, timeout,
one-at-a-time refusal. Nothing here calls AWS: EVIDENCE_S3 is set, and check_unit is replaced by a local script."""

import json
import sys
import textwrap

import pytest
from fastapi.testclient import TestClient

from spectrum.access import TOKEN
from spectrum.evidence import live_verify
from spectrum.main import app

RUN = "t1.rl.viavi_operate-20990101T000000Z-x"
LOCAL = {"base_url": "http://localhost:8787", "client": ("127.0.0.1", 50000)}
HDRS = {"X-Spectrum-Client": "verify-live", "Origin": "http://localhost:8787", "Sec-Fetch-Site": "same-origin"}
SUMMARY = {"run_id": RUN, "exit": 2, "parts": {
    "a": {"state": "PASS", "title": "declared before the run", "lines": ["declared at 2099-01-01T00:00:00Z (s3:LastModified)"]},
    "b": {"state": "PASS", "title": "code in our repo", "lines": ["code_sha abc is a commit in agent-xyz"]},
    "c": {"state": "CANNOT CHECK", "title": "records keyed to the run, times agree with AWS",
          "lines": ["cannot check: no AWS window for this run (no SageMaker job and no CloudWatch event mentions the run id)"]},
    "d": {"state": "PASS", "title": "unchanged", "lines": ["ledger ok"]},
    "e": {"state": "PASS", "title": "result recomputes", "lines": ["n_items 5 == 5"]}},
    "derived_aws_times": [{"record": "item/x.json", "aws_time": "2099-01-01T00:01:00Z", "source": "s3:LastModified records/x"}]}


@pytest.fixture()
def root(tmp_path, monkeypatch):
    (tmp_path / "declared").mkdir()
    (tmp_path / "declared" / f"{RUN}.json").write_text("{}")
    monkeypatch.setenv("SPECTRUM_EVIDENCE_DIR", str(tmp_path))
    monkeypatch.setenv("EVIDENCE_S3", "s3://bucket-for-tests/evidence/")
    return tmp_path


def fake_check_unit(tmp_path, body: str):
    p = tmp_path / "fake_check_unit.py"
    p.write_text(textwrap.dedent(body))
    return p


def test_guards_refuse_lan_and_missing_header(root):
    lan = TestClient(app, base_url="http://192.168.1.20:8787", client=("192.168.1.20", 50000))
    r = lan.post(f"/api/evidence/verify/{RUN}", headers={**HDRS, "X-Spectrum-Token": TOKEN, "Origin": "http://192.168.1.20:8787"})
    assert r.status_code == 403 and r.json()["error"] == "laptop_only"
    c = TestClient(app, **LOCAL)
    assert c.post(f"/api/evidence/verify/{RUN}", headers={k: v for k, v in HDRS.items() if k != "X-Spectrum-Client"}).status_code == 403
    assert c.post(f"/api/evidence/verify/{RUN}", headers={**HDRS, "Origin": "http://evil.example"}).json()["error"] == "bad_origin"
    assert c.get(f"/api/evidence/verify/{RUN}").status_code == 403


def test_unknown_run_is_refused(root):
    c = TestClient(app, **LOCAL)
    assert c.post("/api/evidence/verify/t9.nope-1", headers=HDRS).status_code == 404


def test_parse_and_cache_write_in_sync_format(root, monkeypatch):
    script = fake_check_unit(root, f"""
        import json, sys
        print(json.dumps({json.dumps(SUMMARY)}))
        sys.exit(2)
    """)
    monkeypatch.setattr(live_verify, "CHECK_UNIT", script)
    c = TestClient(app, **LOCAL)
    r = c.post(f"/api/evidence/verify/{RUN}", headers=HDRS)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["exit_code"] == 2 and d["live"] is True and d["checked_at"]
    parts = {p["id"]: p for p in d["parts"]}
    assert parts["c"]["state"] == "CANNOT CHECK" and "no AWS window" in parts["c"]["lines"][0]
    assert parts["b"]["lines"][0].endswith("in the team repo")  # the checkout name is not shown
    assert d["derived_aws_times"][0]["source"].startswith("s3:LastModified")
    cached = json.loads((root / ".sync" / "checks" / f"{RUN}.json").read_text())
    assert set(cached) >= {"run_id", "exit_code", "checked_at", "elapsed_s", "summary", "stdout", "stderr"}
    assert cached["summary"]["parts"]["c"]["state"] == "CANNOT CHECK"
    g = c.get(f"/api/evidence/verify/{RUN}", headers={"X-Spectrum-Client": "verify-live"}).json()
    assert g["running"] is False and g["last"]["exit_code"] == 2


def test_timeout_caches_nothing(root, monkeypatch):
    script = fake_check_unit(root, "import time; time.sleep(5)\n")
    monkeypatch.setattr(live_verify, "CHECK_UNIT", script)
    monkeypatch.setattr(live_verify, "TIMEOUT_S", 0.5)
    r = TestClient(app, **LOCAL).post(f"/api/evidence/verify/{RUN}", headers=HDRS)
    assert r.status_code == 504 and r.json()["error"] == "timeout"
    assert not (root / ".sync" / "checks" / f"{RUN}.json").exists()


def test_concurrent_request_is_refused(root, monkeypatch):
    class Busy:
        def locked(self):
            return True

    monkeypatch.setattr(live_verify, "_lock", Busy())
    monkeypatch.setitem(live_verify._running, "run_id", "t1.other-1")
    r = TestClient(app, **LOCAL).post(f"/api/evidence/verify/{RUN}", headers=HDRS)
    assert r.status_code == 409 and r.json()["running_run_id"] == "t1.other-1"


def test_no_credentials_is_said_plainly(root, monkeypatch):
    monkeypatch.delenv("EVIDENCE_S3")
    monkeypatch.delenv("TEAM13_DATA_BUCKET", raising=False)
    monkeypatch.setattr(live_verify, "s3_prefix", lambda: (None, "No AWS credentials are available to this server (NoCredentialsError)."))
    r = TestClient(app, **LOCAL).post(f"/api/evidence/verify/{RUN}", headers=HDRS)
    assert r.status_code == 503 and "No AWS credentials" in r.json()["message"]


def test_python_is_the_servers_own():
    assert sys.executable  # the subprocess uses the server's interpreter (boto3 and its AWS environment)
