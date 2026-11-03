"""No email, ARN or account id in any response or log line, in any state (review F3)."""

import logging

from botocore.exceptions import ClientError, UnauthorizedSSOTokenError

from slices.kitapprovals.kitapprovals_helpers import (MARK_ACCOUNT, MARK_ARN, MARK_EMAIL, NOW, client, decide, make,
                                                      pending_row)

FORBIDDEN = (MARK_ACCOUNT, MARK_EMAIL, "arn:aws", "marker%40example")


def _rows():
    pend = pending_row()
    cli = pending_row(change="cell=1;d_power_db=+1.0;d_tilt_deg=+0.0")
    cli.update(status="APPROVED", decided_by=MARK_ARN, decided_at=int(NOW) - 60,
               decision_reason=f"ok from {MARK_EMAIL} in {MARK_ACCOUNT}")
    applied = pending_row(change="cell=0;d_power_db=+1.0;d_tilt_deg=+0.0")
    applied.update(status="APPLIED", decided_by=MARK_EMAIL, decided_at=int(NOW) - 90, applied_at=int(NOW) - 30)
    return pend, cli, applied


def _clean(text):
    for f in FORBIDDEN:
        assert f not in text, f"{f!r} leaked: {text[:300]}"


def test_every_route_in_every_state_hides_identity(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    pend, cli, applied = _rows()
    svc, ddb, logs, sts, _ = make(tmp_path, monkeypatch, [pend, cli, applied])
    c = client(svc)
    texts = [c.get("/api/kitapprovals/status").text, c.get("/api/kitapprovals/pending").text]
    texts += [c.get(f"/api/kitapprovals/{r['pk']}").text for r in (pend, cli, applied)]
    texts += [decide(c, cli).text, decide(c, applied).text]  # 409 copy names who decided
    texts += [decide(c, pend, pin="000000").text]
    logs.fail["PutLogEvents"] = ClientError({"Error": {"Code": "ResourceNotFoundException", "Message": "x"}}, "Put")
    texts += [decide(c, pend).text]  # decided, not audited
    texts += [c.get("/api/kitapprovals/status").text]
    sts.fail = UnauthorizedSSOTokenError()
    svc._sts_cache = None
    texts += [c.get("/api/kitapprovals/status").text, decide(c, pend).text]
    phone = client(svc, peer=("10.0.0.9", 1))
    texts += [phone.get("/api/kitapprovals/status").text, phone.get("/api/kitapprovals/pending").text]
    for t in texts:
        _clean(t)
    _clean("\n".join(r.getMessage() for r in caplog.records))
    # the ARN IS in the CloudWatch control line, exactly as the kit writes it (never in the UI)
    assert any(MARK_ARN in str(kw) for op, kw in logs.calls if op == "PutLogEvents")
