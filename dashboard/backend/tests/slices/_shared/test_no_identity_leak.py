"""Walk every GET route's JSON over an evidence root whose records carry an approver's email and STS ARN (as the
real t3.allow change records do) and fail on any email or STS/IAM ARN in a response."""

import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
ARN = re.compile(r"arn:aws[a-z-]*:(?:sts|iam)::")
ACCT = "123456789012"
WHO = f"arn:aws:sts::{ACCT}:assumed-role/AWSReservedSSO_bb26-Participant_0123/jane.doe@example.com"
RUN = "t3.allow-20990101T000000Z"


def w(root: Path, rel: str, obj) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj))


@pytest.fixture()
def root(tmp_path, monkeypatch):
    t = {"metric": "pass_rate", "op": ">=", "unit": "ratio", "value": 1.0}
    w(tmp_path, f"declared/{RUN}.json", {"run_id": RUN, "unit": "t3.allow", "track": "track-3", "model_id": "checkpoint:x",
                                          "target": t, "declared_at": "2099-01-01T00:00:00Z", "n_items": 1,
                                          "grader": "g", "test_set_sha256": "0" * 64, "contact": "ops@example.org"})
    w(tmp_path, f"results/{RUN}.json", {"run_id": RUN, "unit": "t3.allow", "metric": "pass_rate", "value": 1.0, "passed": True,
                                         "n_items": 1, "checked_at": "2099-01-01T00:10:00Z", "target": t,
                                         "details": {"approved_by": [WHO]}})
    w(tmp_path, f"records/{RUN}/change/aaaaaaaaaaaaaaaaaaaaaaaa.json", {
        "kind": "change", "run_id": RUN, "case_id": "case-1", "change_id": "chg-1", "item_id": "step-0",
        "approved_by": WHO, "approved_at": "2099-01-01T00:05:00Z", "created_at": "2099-01-01T00:05:01Z",
        "change": [{"cell": 2, "d_power_db": -1.0, "d_tilt_deg": 1.0}], "change_text": "cell=2;d_power_db=-1.0;d_tilt_deg=+1.0",
        "twin": {"kind": "maveric-inprocess", "step": 0}, "synthetic": False})
    w(tmp_path, f"records/{RUN}/decision/bbbbbbbbbbbbbbbbbbbbbbbb.json", {
        "kind": "decision", "run_id": RUN, "case_id": "case-1", "item_id": "step-0", "created_at": "2099-01-01T00:04:00Z",
        "actor": WHO, "decision": "allow"})
    w(tmp_path, f"outputs/{RUN}/job-map.json", {"job_name": "j-1", "role_arn": f"arn:aws:iam::{ACCT}:role/bb26/team13-x",
                                                 "launched_by": "jane.doe@example.com"})
    monkeypatch.setenv("SPECTRUM_RECORDS_ROOT", str(tmp_path))
    return tmp_path


def test_no_email_or_person_arn_in_any_get_route(root):
    from spectrum.access import TOKEN
    from spectrum.main import app

    paths = [r.path for r in app.routes if "GET" in (getattr(r, "methods", None) or set()) and "{" not in r.path
             and r.path.startswith("/api/")]
    paths += ["/api/scorecard/units", f"/api/scorecard/run/{RUN}", "/api/realrun/overview", f"/api/realrun/run/{RUN}",
              "/api/twinview/cases", "/api/twinview/case/case-1", "/api/twinview/consortium",
              "/api/trace/cases", "/api/trace/case/case-1",
              "/api/trustgate/models", "/api/trustgate/changes", "/api/trustgate/ladder", f"/api/evidence/runs/{RUN}"]
    seen_person = False
    with TestClient(app, headers={"X-Spectrum-Token": TOKEN}) as c:
        for p in sorted(set(paths)):
            r = c.get(p)
            body = r.text
            assert not EMAIL.search(body), f"{p}: email in response"
            assert not ARN.search(body), f"{p}: STS/IAM ARN in response"
            assert ACCT not in body, f"{p}: account id in response"
            seen_person |= "person:team13-op1" in body
            assert r.status_code < 500, f"{p}: HTTP {r.status_code}"
    assert seen_person  # the approver is shown, as the pseudonym
