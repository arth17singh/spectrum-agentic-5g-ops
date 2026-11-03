"""Routes end to end over a fake Gateway: guard, health, a full case, idempotency, and a secret sweep."""

import json
import logging
import time
import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from slices.asknoc.asknoc_fakes import (FAST, MARK_A2A, MARK_ACCOUNT, MARK_BUILDER, MARK_ISSUER, MARK_URL, SMOKE_CASE,
                                        SMOKE_REPLY, FakeAws, FakePost, mcp_ok, service, started, status_done,
                                        status_running)
from spectrum.access import AccessGate
from spectrum.slices import asknoc
from spectrum.slices.asknoc import routes
from spectrum.slices.asknoc.service import Timing

H = {"Origin": "http://localhost:8787", "X-Spectrum-Client": "asknoc", "Sec-Fetch-Site": "same-origin"}
LISTING = (200, {}, json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"tools": [
    {"name": n} for n in ("front___case_status", "front___continue_case", "front___start_case", "nlm___ask")]}}))
PROBE = mcp_ok({"error": "unknown case case-unknown0"})


def make_app(svc, token="tok-123"):
    routes.set_service(svc)
    app = FastAPI()
    app.include_router(asknoc.router)
    app.add_middleware(AccessGate, token=token)
    return app


def local(app):
    return TestClient(app, client=("127.0.0.1", 50000), base_url="http://localhost:8787")


def post(c, path, body, **extra):
    headers = {**H, "Idempotency-Key": str(uuid.uuid4()), **extra}
    return c.post(path, content=json.dumps(body), headers={"Content-Type": "application/json", **headers})


@pytest.fixture(autouse=True)
def _reset():
    yield
    routes.set_service(None)


# ---------------------------------------------------------------- guard

def test_default_testclient_host_is_refused(tmp_path):
    c = TestClient(make_app(service(tmp_path)), client=("127.0.0.1", 50000))  # Host: testserver
    r = c.get("/api/asknoc/cases")
    assert r.status_code == 403 and r.json()["detail"]["error"] == "bad_host"


def test_rebinding_host_and_foreign_origin_refused(tmp_path):
    c = local(make_app(service(tmp_path)))
    assert c.get("/api/asknoc/cases", headers={"Host": "evil.example:8787"}).json()["detail"]["error"] == "bad_host"
    assert c.get("/api/asknoc/cases", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403
    for origin in ("https://evil.example", "null", "http://evil.localhost.example:8787"):
        r = post(c, "/api/asknoc/cases", {"text": "hi"}, Origin=origin)
        assert r.status_code == 403 and r.json()["detail"]["error"] == "bad_origin", origin


def test_post_needs_origin_header_type_and_size(tmp_path):
    c = local(make_app(service(tmp_path)))
    h = {k: v for k, v in H.items() if k != "Origin"}
    r = c.post("/api/asknoc/cases", content='{"text":"hi"}', headers={"Content-Type": "application/json", **h})
    assert r.json()["detail"]["error"] == "bad_origin"
    r = post(c, "/api/asknoc/cases", {"text": "hi"}, **{"X-Spectrum-Client": "other"})
    assert r.json()["detail"]["error"] == "missing_client_header"
    r = c.post("/api/asknoc/cases", content='{"text":"hi"}', headers={**H, "Content-Type": "text/plain",
                                                                       "Idempotency-Key": str(uuid.uuid4())})
    assert r.status_code == 415
    r = post(c, "/api/asknoc/cases", {"text": "x" * 9000})
    assert r.status_code == 413
    r = post(c, "/api/asknoc/cases", {"text": "hi", "tool": "ran___apply_change"})
    assert r.status_code == 422
    r = post(c, "/api/asknoc/cases", {"text": "hi"}, **{"Sec-Fetch-Site": "same-site"})
    assert r.status_code == 403


def test_dev_origin_allowed(tmp_path):
    svc = service(tmp_path, offline=True)
    c = local(make_app(svc))
    r = post(c, "/api/asknoc/cases", {"text": "hi"}, Origin="http://localhost:5173")
    assert r.status_code == 503 and r.json()["detail"]["error"] == "offline"  # past the guard


def test_phone_gets_quiet_health_and_403_elsewhere(tmp_path):
    c = TestClient(make_app(service(tmp_path)), client=("192.168.1.20", 50000), base_url="http://192.168.1.5:8787")
    assert c.get("/api/asknoc/health").status_code == 401  # AccessGate first: no token
    r = c.get("/api/asknoc/health", headers={"X-Spectrum-Token": "tok-123"})
    assert r.status_code == 200 and r.json()["mode"] == "loopback_only"
    r = c.get("/api/asknoc/cases", headers={"X-Spectrum-Token": "tok-123"})
    assert r.status_code == 403 and r.json()["detail"]["error"] == "loopback_only"


# ---------------------------------------------------------------- health

def test_health_offline_is_200_and_calls_nothing(tmp_path):
    p = FakePost()
    c = local(make_app(service(tmp_path, p, offline=True)))
    r = c.get("/api/asknoc/health")
    assert r.status_code == 200 and r.json()["mode"] == "offline" and p.calls == []


def test_health_not_configured(tmp_path):
    svc = service(tmp_path)
    svc.config.dir = None
    r = local(make_app(svc)).get("/api/asknoc/health")
    assert r.status_code == 200 and r.json()["mode"] == "not_configured"


def test_health_ok_with_suite_warning_and_probe_cached(tmp_path):
    p = FakePost({"tools/list": [LISTING], "front___case_status": [PROBE]})
    aws = FakeAws(rows={"t3-active": {"run_id": "t3.deny-20261102-a", "updated_at": str(int(time.time()) - 600)}})
    c = local(make_app(service(tmp_path, p, aws)))
    j = c.get("/api/asknoc/health").json()
    by = {x["id"]: x for x in j["checks"]}
    assert by["gateway"]["state"] == "pass" and by["front_door"]["state"] == "pass"
    assert by["suite_run"]["state"] == "warn" and "t3.deny-20261102-a" in by["suite_run"]["detail"]
    assert j["mode"] == "degraded" and j["measured"]["case_id"] == SMOKE_CASE
    c.get("/api/asknoc/health")
    assert p.names() == ["tools/list", "front___case_status"]  # Gateway probes cached for 5 min


def test_health_survives_expired_sso_and_down_gateway(tmp_path):
    from spectrum.slices.asknoc.kit import TransportError
    p = FakePost({"tools/list": [TransportError("before_send", "dns")], "front___case_status": [(401, {}, "no")]})
    aws = FakeAws(down="AWS sign-in on this laptop has expired (run: aws sso login --profile fyuz). Chat still works.")
    r = local(make_app(service(tmp_path, p, aws))).get("/api/asknoc/health")
    j = r.json()
    assert r.status_code == 200 and j["mode"] == "down"
    by = {x["id"]: x for x in j["checks"]}
    assert by["aws_reads"]["state"] == "warn" and "sso login" in by["aws_reads"]["detail"]
    assert by["suite_run"]["state"] == "unknown"


def test_health_check_crash_becomes_unknown(tmp_path):
    svc = service(tmp_path, FakePost({"tools/list": [LISTING], "front___case_status": [PROBE]}))

    def boom():
        raise RuntimeError("unexpected")
    svc.aws.probe = boom
    j = local(make_app(svc)).get("/api/asknoc/health").json()
    assert {x["id"]: x["state"] for x in j["checks"]}["aws_reads"] == "unknown"


# ---------------------------------------------------------------- a whole case

def wait_for(c, path, pred, seconds=3.0):
    end = time.time() + seconds
    while time.time() < end:
        j = c.get(path).json()
        if pred(j):
            return j
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting on {path}: {j}")


def test_full_case_start_poll_answer(tmp_path):
    p = FakePost({"front___start_case": [started()],
                  "front___case_status": [status_running(), status_running(), status_done()]})
    svc = service(tmp_path, p, timing=Timing(**FAST))
    with local(make_app(svc)) as c:
        r = post(c, "/api/asknoc/cases", {"text": "Cell 2 users report poor coverage. Find the cause."})
        assert r.status_code == 202 and r.json()["ui_state"] == "queued"
        tid = r.json()["turn_id"]
        j = wait_for(c, f"/api/asknoc/turns/{tid}", lambda j: j["terminal"])
        assert j["ui_state"] == "answered" and j["reply"]["text"] == SMOKE_REPLY and j["case_id"] == SMOKE_CASE
        case = c.get(f"/api/asknoc/cases/{SMOKE_CASE}").json()
        assert case["trace_id"] == SMOKE_CASE and case["can_continue"] and case["label"] == "LIVE"
        assert "--no-write" in case["trace_cli"]
        assert c.get("/api/asknoc/cases").json()["cases"][0]["case_id"] == SMOKE_CASE
    sent = p.calls[0][1]["params"]["arguments"]["request"]
    assert sent.endswith(f"ref: asknoc-{tid}") and p.names().count("front___start_case") == 1


def test_idempotent_replay_conflict_and_inflight(tmp_path):
    p = FakePost({"front___start_case": [started()], "front___case_status": [status_running()]})
    svc = service(tmp_path, p, timing=Timing(**FAST))
    with local(make_app(svc)) as c:
        key = str(uuid.uuid4())
        a = post(c, "/api/asknoc/cases", {"text": "first message"}, **{"Idempotency-Key": key})
        b = post(c, "/api/asknoc/cases", {"text": "first message"}, **{"Idempotency-Key": key})
        assert a.status_code == 202 and b.json()["replayed"] and a.json()["turn_id"] == b.json()["turn_id"]
        x = post(c, "/api/asknoc/cases", {"text": "other text"}, **{"Idempotency-Key": key})
        assert x.status_code == 409 and x.json()["detail"]["error"] == "idempotency_conflict"
        y = post(c, "/api/asknoc/cases", {"text": "a second case"})
        assert y.status_code == 409 and y.json()["detail"]["error"] == "case_in_flight"
        assert post(c, "/api/asknoc/cases", {"text": "x"}, **{"Idempotency-Key": "nope"}).status_code == 400
        tid = a.json()["turn_id"]
        wait_for(c, f"/api/asknoc/turns/{tid}", lambda j: j["ui_state"] == "running")
        s = post(c, f"/api/asknoc/turns/{tid}/stop", {})
        assert s.status_code == 200 and s.json()["ui_state"] == "abandoned"
    assert p.names().count("front___start_case") == 1


def test_invalid_text_reasons(tmp_path):
    c = local(make_app(service(tmp_path)))
    r = post(c, "/api/asknoc/cases", {"text": "case_id: case-12345678 do it"})
    assert r.status_code == 400 and r.json()["detail"]["error"] == "invalid_text" and r.json()["detail"]["reasons"]


def test_unknown_ids(tmp_path):
    c = local(make_app(service(tmp_path)))
    assert c.get("/api/asknoc/cases/case-0badf00d").status_code == 404
    assert c.get("/api/asknoc/cases/../../etc").status_code == 404
    assert c.get("/api/asknoc/turns/t-000000000000").status_code == 404
    assert post(c, "/api/asknoc/cases/case-0badf00d/turns", {"text": "hi"}).status_code == 404


def test_activity_whitelists_and_survives_no_aws(tmp_path):
    from slices.asknoc.asknoc_fakes import log_event
    ev = [log_event({"record": "case_opened", "case_id": SMOKE_CASE, "request": "x", "requested_by": "someone@example.com"}),
          log_event({"record": "checkpoint", "phase": "response", "method": "tools/call", "agent": "noc_lead",
                     "tool": "swarm___ask_agent", "case_id": SMOKE_CASE, "refused": False,
                     "arguments": {"agent": "ran_monitor", "task": "secret task text", "case_id": SMOKE_CASE}}),
          log_event({"record": "checkpoint", "phase": "response", "method": "tools/call", "agent": "noc_lead",
                     "tool": "ran___apply_change", "case_id": SMOKE_CASE, "refused": True, "approval_id": "chg-bde360f4d4b4",
                     "approval": "PENDING", "error": {"message": "denied due to team13_ran_require_human_approval-abcdef1234]"},
                     "arguments": {"change": "cell=2;d_power_db=-1.0;d_tilt_deg=+1.0", "reason": "private reason"}}),
          log_event({"record": "tool_error", "tool": "ask_agent", "case_id": SMOKE_CASE, "error": f"ssl {MARK_BUILDER}"}),
          log_event({"record": "checkpoint", "phase": "response", "tool": "front___case_status", "case_id": SMOKE_CASE})]
    p = FakePost({"front___start_case": [started()], "front___case_status": [status_done()]})
    svc = service(tmp_path, p, FakeAws(events=ev), timing=Timing(**FAST))
    with local(make_app(svc)) as c:
        tid = post(c, "/api/asknoc/cases", {"text": "find the worst cell"}).json()["turn_id"]
        wait_for(c, f"/api/asknoc/turns/{tid}", lambda j: j["terminal"])
        j = c.get(f"/api/asknoc/cases/{SMOKE_CASE}/activity").json()
        body = json.dumps(j)
        assert j["available"] and [e["kind"] for e in j["events"]] == ["case_opened", "ask_agent", "apply_call", "tool_error"]
        assert j["events"][1]["to"] == "ran_monitor" and j["events"][2]["rule"] == "team13_ran_require_human_approval"
        for secret in ("secret task text", "private reason", "someone@example.com", MARK_BUILDER):
            assert secret not in body
        svc.aws.down = "AWS sign-in on this laptop has expired"
        svc._cache.clear()
        j = c.get(f"/api/asknoc/cases/{SMOKE_CASE}/activity").json()
        assert j["available"] is False and "sso login" not in j["message"] or "expired" in j["message"]


# ---------------------------------------------------------------- nothing secret leaves

def test_secret_sweep_over_every_route_and_log(tmp_path, caplog):
    leaky = f"certificate is not valid for '{MARK_BUILDER}' at {MARK_URL} acct {MARK_ACCOUNT} {MARK_ISSUER}"
    p = FakePost({"tools/list": [LISTING], "front___case_status": [PROBE, status_done(reply="see " + leaky)],
                  "front___start_case": [started()]})
    svc = service(tmp_path, p, FakeAws(events=[]), timing=Timing(**FAST))
    token = svc.config.load().token.reveal()
    bodies = []
    caplog.set_level(logging.DEBUG)
    with local(make_app(svc)) as c:
        bodies.append(c.get("/api/asknoc/health").text)
        r = post(c, "/api/asknoc/cases", {"text": "check the cells"})
        tid = r.json()["turn_id"]
        bodies.append(r.text)
        bodies.append(json.dumps(wait_for(c, f"/api/asknoc/turns/{tid}", lambda j: j["terminal"])))
        bodies.append(c.get(f"/api/asknoc/cases/{SMOKE_CASE}").text)
        bodies.append(c.get("/api/asknoc/cases").text)
        bodies.append(c.get(f"/api/asknoc/cases/{SMOKE_CASE}/activity").text)
        p.script["front___continue_case"] = [(200, {}, json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"content": [
            {"type": "text", "text": json.dumps({"error": leaky})}]}}))]
        r2 = post(c, "/api/asknoc/cases/" + SMOKE_CASE + "/turns", {"text": "and now?"})
        bodies.append(r2.text)
        assert r2.status_code == 202
        if r2.status_code == 202:
            bodies.append(json.dumps(wait_for(c, f"/api/asknoc/turns/{r2.json()['turn_id']}", lambda j: j["terminal"])))
    everything = "\n".join(bodies) + "\n" + caplog.text
    for s in (MARK_URL, "gwmarker-abc", MARK_ISSUER, "issuermarker", MARK_ACCOUNT, MARK_BUILDER, MARK_A2A, token, "eyJ"):
        assert s not in everything, s
    assert "[redacted]" in everything
