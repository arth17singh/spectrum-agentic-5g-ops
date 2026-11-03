"""The decision: the kit's exact update and control record, races, PIN, AWS errors, audit outbox."""

import json
import os
import stat
import time

import pytest
from botocore.exceptions import ClientError, SSOTokenLoadError, TokenRetrievalError, UnauthorizedSSOTokenError

from spectrum.slices.kitapprovals import audit
from slices.kitapprovals.kitapprovals_helpers import (MARK_ARN, MARK_EMAIL, NOW, PIN, client, decide, make,
                                                      pending_row, sha)

KIT_CONTROL_KEYS = ["ts", "by", "record", "decision", "approval_id", "case_id", "change", "reason"]  # KIT L975-977


def _updates(ddb):
    return [kw for op, kw in ddb.calls if op == "UpdateItem"]


def test_update_is_the_kits_plus_the_change_condition(tmp_path, monkeypatch):
    row = pending_row()
    svc, ddb, logs, sts, clock = make(tmp_path, monkeypatch, [row])
    r = decide(client(svc), row, "approve", "checked: coverage gain predicted")
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["status"] == "APPROVED" and j["by"] == "person:team13-op1" and j["audit"] == "written"
    assert j["decided_at"] == int(NOW) and "noc_lead" in j["note"]
    [kw] = _updates(ddb)
    # KIT L992-999, field by field, plus `AND #c = :change` and ALL_OLD on a failed condition
    assert kw["Key"] == {"pk": {"S": row["pk"]}}
    assert kw["ReturnValues"] == "ALL_NEW"
    assert kw["UpdateExpression"] == "SET #s = :s, decided_by = :by, decided_at = :t, decision_reason = :why"
    assert kw["ConditionExpression"] == "#s = :pending AND #c = :change"
    assert kw["ExpressionAttributeNames"] == {"#s": "status", "#c": "change"}
    assert kw["ExpressionAttributeValues"] == {
        ":s": {"S": "APPROVED"}, ":by": {"S": "person:team13-op1"}, ":t": {"N": str(int(NOW))},
        ":why": {"S": "checked: coverage gain predicted"}, ":pending": {"S": "PENDING"}, ":change": {"S": row["change"]}}
    assert kw["ReturnValuesOnConditionCheckFailure"] == "ALL_OLD"
    assert kw["TableName"] == "team13-approvals"
    stored = ddb.items[row["pk"]]
    assert stored["status"] == "APPROVED" and stored["decided_by"] == "person:team13-op1"
    assert stored["decided_at"] == int(NOW) and stored["decision_reason"] == "checked: coverage gain predicted"
    assert sts.calls >= 1  # the identity is re-read for the decision


def test_reject_writes_rejected(tmp_path, monkeypatch):
    row = pending_row()
    svc, ddb, *_ = make(tmp_path, monkeypatch, [row])
    r = decide(client(svc), row, "reject", "smoke cleanup")
    assert r.status_code == 200 and r.json()["status"] == "REJECTED" and r.json()["note"] is None
    assert _updates(ddb)[0]["ExpressionAttributeValues"][":s"] == {"S": "REJECTED"}


def test_control_record_is_the_kits_shape_with_the_arn_and_original_time(tmp_path, monkeypatch):
    row = pending_row()
    svc, ddb, logs, *_ = make(tmp_path, monkeypatch, [row])
    decide(client(svc), row, "approve", "checked it")
    ops = [op for op, _ in logs.calls]
    assert ops == ["CreateLogStream", "PutLogEvents"]  # KIT L971-978: create (exists is fine), then put
    put = logs.calls[1][1]
    assert put["logGroupName"] == "/team13/checkpoint" and put["logStreamName"] == "control"
    [ev] = put["logEvents"]
    assert ev["timestamp"] == int(NOW * 1000)
    msg = json.loads(ev["message"])
    assert list(msg) == KIT_CONTROL_KEYS + ["decision_id"]
    assert msg == {"ts": NOW, "by": MARK_ARN, "record": "approval", "decision": "APPROVED", "approval_id": row["pk"],
                   "case_id": row["case_id"], "change": row["change"], "reason": "checked it",
                   "decision_id": f"{row['pk']}:{int(NOW)}"}


def test_missing_control_stream_is_created_first(tmp_path, monkeypatch):
    row = pending_row()
    svc, ddb, logs, *_ = make(tmp_path, monkeypatch, [row], streams=())
    r = decide(client(svc), row)
    assert r.json()["audit"] == "written"
    assert "control" in logs.streams and len(logs.events) == 1
    assert [op for op, _ in logs.calls] == ["CreateLogStream", "PutLogEvents"]


def test_race_with_the_kit_cli_gives_409_with_a_pseudonym(tmp_path, monkeypatch):
    row = pending_row()
    svc, ddb, *_ = make(tmp_path, monkeypatch, [row])

    def cli_wins(fake):  # the kit CLI decides between our GetItem and our UpdateItem, writing the ARN
        fake.items[row["pk"]].update(status="APPROVED", decided_by=MARK_ARN, decided_at=int(NOW) - 5)

    ddb.before_update = cli_wins
    r = decide(client(svc), row)
    assert r.status_code == 409 and r.json()["error"] == "not_pending"
    j = r.json()
    assert j["status"] == "APPROVED" and j["by"].startswith("person:op-") and j["decided_at"] == int(NOW) - 5
    assert MARK_EMAIL not in r.text and "111122223333" not in r.text and "arn:" not in r.text


def test_second_decision_is_409(tmp_path, monkeypatch):
    row = pending_row()
    svc, ddb, *_ = make(tmp_path, monkeypatch, [row])
    c = client(svc)
    assert decide(c, row, "approve").status_code == 200
    r = decide(c, row, "reject", "too late")
    assert r.status_code == 409 and r.json()["error"] == "not_pending" and r.json()["by"] == "person:team13-op1"
    assert len(_updates(ddb)) == 1


@pytest.mark.parametrize("status", ["APPROVED", "REJECTED", "APPLIED"])
def test_decided_rows_are_refused_without_an_update(tmp_path, monkeypatch, status):
    row = pending_row(status=status)
    row["status"] = status
    svc, ddb, *_ = make(tmp_path, monkeypatch, [row])
    r = decide(client(svc), row)
    assert r.status_code == 409 and not _updates(ddb)


def test_change_mismatch(tmp_path, monkeypatch):
    row = pending_row()
    svc, ddb, *_ = make(tmp_path, monkeypatch, [row])
    r = decide(client(svc), row, change_sha=sha("something else"))
    assert r.status_code == 409 and r.json()["error"] == "change_mismatch" and not _updates(ddb)

    def swapped(fake):  # the change string changes between the read and the update
        fake.items[row["pk"]]["change"] = row["change"].replace("-1.0", "-3.0")

    ddb.before_update = swapped
    r = decide(client(svc), row)
    assert r.status_code == 409 and r.json()["error"] == "change_mismatch"
    assert ddb.items[row["pk"]]["status"] == "PENDING"


@pytest.mark.parametrize("reason", ["", "ok", "x" * 301, "see https://example.com", "ask marker@example.com",
                                    "acct 111122223333", "bad​char", "line\x07"])
def test_reason_is_required_and_clean_for_approve_too(tmp_path, monkeypatch, reason):
    row = pending_row()
    svc, ddb, *_ = make(tmp_path, monkeypatch, [row])
    r = decide(client(svc), row, "approve", reason)
    assert r.status_code == 422 and not _updates(ddb)


def test_body_must_be_exact_and_is_never_echoed(tmp_path, monkeypatch):
    row = pending_row()
    svc, ddb, *_ = make(tmp_path, monkeypatch, [row])
    c = client(svc)
    from slices.kitapprovals.kitapprovals_helpers import post_headers
    r = c.post(f"/api/kitapprovals/{row['pk']}/decide", headers=post_headers(),
               content=json.dumps({"decision": "approve", "reason": "fine here", "change_sha256": sha(row["change"]),
                                   "pin": "99999999", "extra": 1}))
    assert r.status_code == 422 and "99999999" not in r.text and not _updates(ddb)


# ------------------------------------------------------------------ PIN, loopback included (review F4)
def test_pin_is_required_on_loopback(tmp_path, monkeypatch):
    row = pending_row()
    svc, ddb, *_ = make(tmp_path, monkeypatch, [row])
    c = client(svc)
    r = decide(c, row, pin="")
    assert r.status_code == 403 and r.json()["error"] == "pin_wrong" and not _updates(ddb)
    r = decide(c, row, pin="135791")
    assert r.status_code == 403 and r.json()["error"] == "pin_wrong" and not _updates(ddb)
    assert ddb.calls == []  # a wrong PIN reaches nothing in AWS, not even a read
    assert decide(c, row).status_code == 200


def test_five_wrong_pins_lock_decisions(tmp_path, monkeypatch):
    row = pending_row()
    svc, ddb, logs, sts, clock = make(tmp_path, monkeypatch, [row])
    c = client(svc)
    codes = [decide(c, row, pin="000000").json()["error"] for _ in range(5)]
    assert codes == ["pin_wrong"] * 4 + ["pin_locked"]
    assert decide(c, row).json()["error"] == "pin_locked"  # even the right PIN, while locked
    clock.t += 15 * 60 + 1
    assert decide(c, row).status_code == 200


def test_missing_or_loose_pin_file_refuses(tmp_path, monkeypatch):
    row = pending_row()
    svc, ddb, *_ = make(tmp_path, monkeypatch, [row], pin=False)
    c = client(svc)
    assert decide(c, row).json()["error"] == "pin_not_set"
    from spectrum.slices.kitapprovals.pin import write_pin
    p = write_pin(PIN, tmp_path / "state" / "pin")
    assert stat.S_IMODE(os.stat(p).st_mode) == 0o600
    os.chmod(p, 0o644)
    assert decide(c, row).json()["error"] == "pin_not_set"
    assert not _updates(ddb)


def test_pin_file_holds_a_salted_scrypt_hash_not_the_pin(tmp_path, monkeypatch):
    make(tmp_path, monkeypatch)
    text = (tmp_path / "state" / "pin").read_text()
    assert PIN not in text and text.startswith("scrypt$")


# ------------------------------------------------------------------ AWS errors (review F10)
@pytest.mark.parametrize("exc", [
    UnauthorizedSSOTokenError(), TokenRetrievalError(provider="sso", error_msg="expired"),
    SSOTokenLoadError(error_msg="no token"),
    ClientError({"Error": {"Code": "ExpiredTokenException", "Message": "x"}}, "GetCallerIdentity")])
def test_expired_session_is_503_with_the_login_copy(tmp_path, monkeypatch, exc):
    row = pending_row()
    svc, ddb, logs, sts, _ = make(tmp_path, monkeypatch, [row])
    sts.fail = exc
    r = decide(client(svc), row)
    assert r.status_code == 503 and r.json()["error"] == "aws_session_expired"
    assert "aws sso login --profile fyuz" in r.json()["message"] and not _updates(ddb)
    s = client(svc).get("/api/kitapprovals/status").json()
    assert s["mode"] == "aws_session_expired"


def test_access_denied_on_update_is_no_access(tmp_path, monkeypatch):
    row = pending_row()
    svc, ddb, *_ = make(tmp_path, monkeypatch, [row])
    ddb.fail["UpdateItem"] = ClientError({"Error": {"Code": "AccessDeniedException", "Message": "x"}}, "UpdateItem")
    r = decide(client(svc), row)
    assert r.status_code == 503 and r.json()["error"] == "no_access"


# ------------------------------------------------------------------ audit outbox (review F2, F14)
def test_audit_failure_is_decided_not_audited_and_fails_closed(tmp_path, monkeypatch):
    a, b = pending_row(), pending_row(change="cell=1;d_power_db=+1.0;d_tilt_deg=+0.0")
    svc, ddb, logs, *_ = make(tmp_path, monkeypatch, [a, b])
    logs.fail["PutLogEvents"] = ClientError({"Error": {"Code": "AccessDeniedException", "Message": "x"}}, "PutLogEvents")
    c = client(svc)
    r = decide(c, a)
    assert r.status_code == 200 and r.json()["audit"] == "not_audited" and "NOT audited" in r.json()["audit_message"]
    assert ddb.items[a["pk"]]["status"] == "APPROVED"  # the decision stands, as in the kit
    p = audit.outbox_path()
    assert stat.S_IMODE(os.stat(p).st_mode) == 0o600
    [entry] = [json.loads(x) for x in p.read_text().splitlines()]
    assert entry["timestamp"] == int(NOW * 1000) and json.loads(entry["message"])["ts"] == NOW
    assert c.get("/api/kitapprovals/status").json()["audit_outbox"] == 1
    # fail closed: no new decision while the line is missing
    r = decide(c, b)
    assert r.status_code == 503 and r.json()["error"] == "audit_pending"
    assert ddb.items[b["pk"]]["status"] == "PENDING"
    # CloudWatch works again: the queued line goes out with its ORIGINAL time, then b is decided
    del logs.fail["PutLogEvents"]
    r = decide(c, b)
    assert r.status_code == 200 and r.json()["audit"] == "written"
    assert [e["timestamp"] for e in logs.events] == [int(NOW * 1000), int(NOW * 1000)]
    assert [json.loads(e["message"])["approval_id"] for e in logs.events] == [a["pk"], b["pk"]]
    assert not p.exists()


def test_a_line_that_landed_before_a_timeout_is_never_written_twice(tmp_path, monkeypatch):
    a = pending_row()
    svc, ddb, logs, *_ = make(tmp_path, monkeypatch, [a])
    logs.land_then_fail = True  # PutLogEvents writes, then the read times out
    r = decide(client(svc), a)
    assert r.json()["audit"] == "not_audited" and len(logs.events) == 1
    entry = json.loads(audit.outbox_path().read_text())
    assert entry["not_before"] > time.time() + 30  # ambiguous: wait before replaying
    assert audit.drain(logs, now=lambda: entry["not_before"] - 1) == 1 and len(logs.events) == 1
    assert audit.drain(logs, now=lambda: entry["not_before"] + 1) == 0
    assert len(logs.events) == 1  # found by decision_id, dropped, not re-sent
    assert any(op == "FilterLogEvents" for op, _ in logs.calls)


def test_status_and_pending_need_no_pin(tmp_path, monkeypatch):
    svc, *_ = make(tmp_path, monkeypatch, [pending_row()], pin=False)
    c = client(svc)
    s = c.get("/api/kitapprovals/status").json()
    assert s["mode"] == "ok" and s["pin"] == {"set": False, "locked_for_s": 0}
    assert s["approver"]["pseudonym"] == "person:team13-op1"
