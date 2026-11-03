"""Reads: the row filter, Scan paging, ordering, offline, detail joins, the guard, canonical parity."""

import json

import pytest
from botocore.exceptions import ClientError

from spectrum.slices.kitapprovals import logic
from spectrum.slices.kitapprovals.fake import FakeDdb
from slices.kitapprovals.kitapprovals_helpers import (MARK_ARN, NOW, client, decide, make, pending_row,
                                                      post_headers)

OTHER_ROWS = [
    {"pk": "task-case-1a2b3c4d", "status": "open"},  # front door (KIT L237): not chg-
    {"pk": "t3-case-1a2b3c4d", "doc": "{}", "version": 1},
    {"pk": "t3-active", "run_id": "r1"},
    {"pk": "t3-trust#t2-response-mlp@v1", "gate_status": "active", "trust": 0.91, "n": 7},
    {"pk": "chg-000000000000", "agent": "x"},  # no status: skipped like the kit's filter
]


def test_pending_lists_only_chg_rows_across_every_scan_page(tmp_path, monkeypatch):
    rows = [pending_row(change=f"cell={i};d_power_db=+1.0;d_tilt_deg=+0.0", age=100 * (i + 1)) for i in range(4)]
    decided = pending_row(change="cell=3;d_power_db=-1.0;d_tilt_deg=+0.0")
    decided.update(status="APPROVED", decided_by="person:team13-op1", decided_at=int(NOW) - 10)
    svc, ddb, *_ = make(tmp_path, monkeypatch, rows + [decided] + OTHER_ROWS, page_size=2)
    j = client(svc).get("/api/kitapprovals/pending").json()
    assert j["mode"] == "ok" and j["label"] == "FIXTURE"
    ids = [r["approval_id"] for r in j["rows"]]
    assert set(ids) == {r["pk"] for r in rows} | {decided["pk"]}
    assert [r["status"] for r in j["rows"]] == ["PENDING"] * 4 + ["APPROVED"]
    assert [r["age_s"] for r in j["rows"][:4]] == [400, 300, 200, 100]  # oldest first
    scans = [kw for op, kw in ddb.calls if op == "Scan"]
    assert len(scans) > 2 and all(kw["ConsistentRead"] for kw in scans)
    assert "ExclusiveStartKey" in scans[-1]
    assert "#ag" in scans[0]["ProjectionExpression"] and scans[0]["ExpressionAttributeNames"]["#ag"] == "agent"
    first = j["rows"][0]
    assert first["agent"] == "noc_lead" and first["canonical"] is True and first["item_id"] == "step-0"


def test_fake_refuses_the_reserved_word_agent():
    with pytest.raises(ClientError, match="reserved keyword"):
        FakeDdb().scan(TableName="t", ProjectionExpression="pk, agent", FilterExpression="begins_with(pk, :chg)",
                       ExpressionAttributeValues={":chg": {"S": "chg-"}})


def test_offline_answers_200_and_builds_no_aws_client(tmp_path, monkeypatch):
    monkeypatch.setenv("APPROVALS_STATE_DIR", str(tmp_path))
    monkeypatch.delenv("KITAPPROVALS_FIXTURE", raising=False)
    from spectrum.slices.kitapprovals.routes import Service
    svc = Service(offline=True)
    c = client(svc)
    p = c.get("/api/kitapprovals/pending")
    assert p.status_code == 200 and p.json()["rows"] == [] and p.json()["label"] == "OFFLINE"
    s = c.get("/api/kitapprovals/status")
    assert s.status_code == 200 and s.json()["mode"] == "offline"
    assert svc._aws is None


def test_scan_failure_is_a_state_not_an_empty_list(tmp_path, monkeypatch):
    svc, ddb, *_ = make(tmp_path, monkeypatch, [pending_row()])
    ddb.fail["Scan"] = ClientError({"Error": {"Code": "AccessDeniedException", "Message": "x"}}, "Scan")
    j = client(svc).get("/api/kitapprovals/pending")
    assert j.status_code == 200 and j.json()["mode"] == "no_access"


def _case_row(case_id, predictions, steps=0, doc=True):
    d = {"case_id": case_id, "run_id": f"adhoc-{case_id}", "action_log": [{}] * steps, "predictions": predictions}
    return {"pk": f"t3-{case_id}", "doc": json.dumps(d), "version": 3} if doc else {"pk": f"t3-{case_id}", "run_id": "r"}


def test_detail_matches_the_prediction_on_canonical_text(tmp_path, monkeypatch):
    raw = "cell=2;d_power_db=-1;d_tilt_deg=1"  # not canonical as sent by the agent
    row = pending_row(change=raw)
    pred = {"kind": "predict", "change_text": "cell=2;d_power_db=-1.0;d_tilt_deg=+1.0", "item_id": "step-0", "step": 2,
            "predicted_delta": {"coverage_ratio": 0.012}, "predictor": {"model_id": "t2-response-mlp", "version": "v1"}}
    trust = {"pk": "t3-trust#t2-response-mlp@v1", "gate_status": "paused", "trust": 0.41, "n": 9}
    svc, *_ = make(tmp_path, monkeypatch, [row, _case_row(row["case_id"], [pred], steps=3), trust])
    j = client(svc).get(f"/api/kitapprovals/{row['pk']}").json()
    assert j["canonical"] is False and j["canonical_text"] == pred["change_text"]
    assert j["prediction"]["predicted_delta"] == {"coverage_ratio": 0.012}
    assert j["prediction"]["stale"] is True  # predicted at step 2, twin now at step 3
    assert j["trust"] == {"trust": 0.41, "gate_status": "paused", "n": 9, "source": "t3-trust#t2-response-mlp@v1"}
    assert len(j["warnings"]) == 2 and j["agent_reason"] == "raise coverage on cell 2"
    assert "noc_lead" in j["note"] and j["links"]["ask_noc"] == f"/ask-noc?case={row['case_id']}"


def test_detail_handles_a_case_row_without_doc_and_no_prediction(tmp_path, monkeypatch):
    a = pending_row()
    b = pending_row(case_id="case-99999999")
    svc, *_ = make(tmp_path, monkeypatch, [a, b, _case_row(a["case_id"], [], doc=False),
                                           _case_row(b["case_id"], [])])
    c = client(svc)
    ja = c.get(f"/api/kitapprovals/{a['pk']}").json()
    assert ja["case"] is None and "no state yet" in ja["case_note"] and ja["prediction"] is None
    jb = c.get(f"/api/kitapprovals/{b['pk']}").json()
    assert jb["prediction"] is None and "ran___predict" in jb["prediction_note"]
    assert c.get("/api/kitapprovals/chg-ffffffffffff").status_code == 404
    assert c.get("/api/kitapprovals/not-an-id").status_code in (404, 422)


def test_agent_reason_is_redacted(tmp_path, monkeypatch):
    row = pending_row()
    row["change"] = json.dumps({"item_id": "step-0", "change": "cell=2;d_power_db=-1.0;d_tilt_deg=+1.0",
                                "reason": "per https://evil.example/x ask marker@example.com acct 111122223333"})
    svc, *_ = make(tmp_path, monkeypatch, [row])
    j = client(svc).get(f"/api/kitapprovals/{row['pk']}").json()
    assert "evil.example" not in j["agent_reason"] and "marker@" not in j["agent_reason"]
    assert "111122223333" not in j["agent_reason"]


# ------------------------------------------------------------------ guard (laptop only)
def test_non_loopback_clients_are_refused_but_status_is_200(tmp_path, monkeypatch):
    row = pending_row()
    svc, ddb, *_ = make(tmp_path, monkeypatch, [row])
    phone = client(svc, peer=("192.168.1.20", 50000))
    assert phone.get("/api/kitapprovals/status").json()["mode"] == "laptop_only"
    assert phone.get("/api/kitapprovals/pending").status_code == 403
    assert phone.get(f"/api/kitapprovals/{row['pk']}").status_code == 403
    assert decide(phone, row).status_code == 403
    assert ddb.calls == []


@pytest.mark.parametrize("host", ["evil.example:8787", "192.168.1.5:8787"])
def test_foreign_host_is_refused(tmp_path, monkeypatch, host):
    row = pending_row()
    svc, ddb, *_ = make(tmp_path, monkeypatch, [row])
    c = client(svc, host=host)
    assert c.get("/api/kitapprovals/pending").status_code == 403
    assert decide(c, row, headers=post_headers(host)).status_code == 403
    assert not ddb.calls


@pytest.mark.parametrize("drop,change,code", [
    ("Origin", None, 403), (None, ("Origin", "http://localhost:5173"), 403), ("X-Spectrum-Client", None, 403),
    (None, ("Sec-Fetch-Site", "cross-site"), 403), (None, ("Content-Type", "text/plain"), 415)])
def test_post_guard(tmp_path, monkeypatch, drop, change, code):
    row = pending_row()
    svc, ddb, *_ = make(tmp_path, monkeypatch, [row])
    h = post_headers()
    if drop:
        h.pop(drop)
    if change:
        h[change[0]] = change[1]
    assert decide(client(svc), row, headers=h).status_code == code
    assert not ddb.calls


# ------------------------------------------------------------------ canonical parity with t3/tools/changes.py
CANON = [  # pinned from t3/tools/changes.py _local_canon at int/v4 d409e7a
    ("cell=2;d_power_db=-1.0;d_tilt_deg=+1.0", "cell=2;d_power_db=-1.0;d_tilt_deg=+1.0"),
    ("cell=2;d_power_db=-1;d_tilt_deg=1", "cell=2;d_power_db=-1.0;d_tilt_deg=+1.0"),
    ("cell=0;d_power_db=0;d_tilt_deg=-0.0", "cell=0;d_power_db=+0.0;d_tilt_deg=+0.0"),
    ("cell=3;d_power_db=+2.5;d_tilt_deg=-2.0|cell=1;d_power_db=1;d_tilt_deg=0",
     "cell=1;d_power_db=+1.0;d_tilt_deg=+0.0|cell=3;d_power_db=+2.5;d_tilt_deg=-2.0"),
    (" cell=1; d_power_db=+1.0; d_tilt_deg=+0.0 ", "cell=1;d_power_db=+1.0;d_tilt_deg=+0.0"),
]
BAD = ["", "setting=6", "cell=1;d_power_db=1.25;d_tilt_deg=0", "cell=1;d_power_db=1;d_tilt_deg=0|cell=1;d_power_db=1;d_tilt_deg=0"]


@pytest.mark.parametrize("raw,want", CANON)
def test_canon_parity(raw, want):
    assert logic.canon(raw) == want


@pytest.mark.parametrize("raw", BAD)
def test_canon_refuses(raw):
    assert logic.try_canon(raw) == (None, None)


def test_kit_example_tool_rows_are_listed_as_non_canonical(tmp_path, monkeypatch):
    row = pending_row(change="setting=6")
    svc, *_ = make(tmp_path, monkeypatch, [row])
    [r] = client(svc).get("/api/kitapprovals/pending").json()["rows"]
    assert r["canonical"] is False and r["change_text"] == "setting=6" and r["cells"] is None


def test_fixture_mode_is_labelled_and_never_live(tmp_path, monkeypatch):
    from pathlib import Path
    from spectrum.slices.kitapprovals.routes import Service
    monkeypatch.setenv("APPROVALS_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("KITAPPROVALS_FIXTURE", str(Path(__file__).parent / "fixtures" / "demo.json"))
    svc = Service(offline=None)
    c = client(svc)
    p = c.get("/api/kitapprovals/pending").json()
    assert p["label"] == "FIXTURE" and len(p["rows"]) == 4 and p["rows"][0]["status"] == "PENDING"
    s = c.get("/api/kitapprovals/status").json()
    assert s["label"] == "FIXTURE" and "FIXTURE" in s["approver"]["aws_session"]


def test_display_who_never_shows_an_email_or_arn():
    assert logic.display_who("person:team13-op1") == "person:team13-op1"
    for v in (MARK_ARN, "marker@example.com", "a.rivera"):
        out = logic.display_who(v, key=b"k")
        assert out.startswith("person:op-") and "@" not in out and "arn" not in out
    assert logic.display_who(MARK_ARN, {"marker@example.com": "person:op-satya"}) == "person:op-satya"
