"""The turn state machine, driven step by step with a fake clock (no event loop: spawn is a no-op)."""

import asyncio

import pytest

from slices.asknoc.asknoc_fakes import (CHG, SMOKE_CASE, SMOKE_REPLY, Clock, FakeAws, FakePost, log_event, mcp_ok,
                                        service, started, status_done, status_running, transport, waiting_row)
from spectrum.slices.asknoc.kit import classify
from spectrum.slices.asknoc.service import RefusedRequest, Timing
from spectrum.slices.asknoc.store import Store


def sent(svc, text="Cell 2 users report poor coverage at the cell edge. Find the cause."):
    turn = svc._turn("start", text, None)
    turn["ui_state"], turn["sent_at"] = "sending", svc.now()
    return turn


def accepted(svc, clock):
    turn = sent(svc)
    svc.apply_send(turn, classify(*started()))
    return turn


def test_start_accepted_then_answered_with_the_smoke_reply(tmp_path):
    clock = Clock()
    svc = service(tmp_path, clock=clock)
    turn = accepted(svc, clock)
    assert turn["ui_state"] == "running" and turn["case_id"] == SMOKE_CASE and turn["seen_running"]
    clock.t += 60
    svc.apply_status(turn, classify(*status_running()))
    assert turn["ui_state"] == "running" and not turn["overdue"]
    clock.t += 60
    svc.apply_status(turn, classify(*status_done()))
    assert turn["ui_state"] == "answered" and turn["reply"]["text"] == SMOKE_REPLY
    v = svc.turn_view(turn)
    assert v["trace_id"] == SMOKE_CASE and v["label"] == "LIVE" and v["elapsed_s"] == 120.0


def test_waiting_approval_shows_while_running_and_ends_awaiting(tmp_path):
    clock = Clock()
    svc = service(tmp_path, clock=clock)
    turn = accepted(svc, clock)
    svc.apply_status(turn, classify(*status_running(waiting=waiting_row())))
    assert turn["ui_state"] == "running" and turn["waiting_approvals"][0]["approval_id"] == CHG
    assert svc.store.cases[SMOKE_CASE]["approvals_seen"] == [CHG]
    svc.apply_status(turn, classify(*status_done(reply="waiting for approval " + CHG, waiting=waiting_row())))
    assert turn["ui_state"] == "awaiting_approval"
    w = turn["waiting_approvals"][0]
    assert w["change"] == "cell=2;d_power_db=-1.0;d_tilt_deg=+1.0" and w["canonical"] and w["item_id"] == "step-0"


def test_non_canonical_change_is_not_echoed(tmp_path):
    svc = service(tmp_path, clock=Clock())
    turn = accepted(svc, None)
    svc.apply_status(turn, classify(*status_done(waiting=waiting_row(change="setting=6"))))
    assert turn["waiting_approvals"][0]["change"] is None and not turn["waiting_approvals"][0]["canonical"]


@pytest.mark.parametrize("elapsed,cause", [(100, "not_reported"), (290, "stopped_near_limit")])
def test_failed_cause_is_inferred_only_from_elapsed_time(tmp_path, elapsed, cause):
    clock = Clock()
    svc = service(tmp_path, clock=clock)
    turn = accepted(svc, clock)
    clock.t += elapsed
    svc.apply_status(turn, classify(*status_done(reply="", state="TASK_STATE_FAILED")))
    assert turn["ui_state"] == "failed" and turn["failed_cause"] == cause and turn["reply"]["empty"]


def test_unknown_state_and_input_required(tmp_path):
    svc = service(tmp_path, clock=Clock())
    t1 = accepted(svc, None)
    svc.apply_status(t1, classify(*status_done(state="TASK_STATE_AUTH_REQUIRED")))
    assert t1["failed_cause"] == "unrecognised_state:TASK_STATE_AUTH_REQUIRED"
    t2 = accepted(svc, None)
    svc.apply_status(t2, classify(*status_done(state="TASK_STATE_INPUT_REQUIRED")))
    assert t2["ui_state"] == "needs_input"


def test_overdue_then_lost(tmp_path):
    clock = Clock()
    svc = service(tmp_path, clock=clock)
    turn = accepted(svc, clock)
    clock.t += 340
    svc.apply_status(turn, classify(*status_running()))
    assert turn["ui_state"] == "running" and turn["overdue"]
    clock.t += 600
    svc.apply_status(turn, classify(*status_running()))
    assert turn["ui_state"] == "lost"


def test_status_errors_back_off_and_flag_unavailable(tmp_path):
    clock = Clock()
    svc = service(tmp_path, clock=clock)
    turn = accepted(svc, clock)
    for _ in range(5):
        assert svc.apply_status(turn, classify(502, {}, "bad")) is True
    assert turn["ui_state"] == "running" and turn["status_unavailable"]
    svc.apply_status(turn, classify(*mcp_ok({"error": "<urlopen error timed out>"})))
    assert turn["ui_state"] == "running"
    svc.apply_status(turn, classify(*status_running()))
    assert not turn["status_unavailable"]


def test_unknown_case_on_status(tmp_path):
    svc = service(tmp_path, clock=Clock())
    turn = accepted(svc, None)
    svc.apply_status(turn, classify(*mcp_ok({"error": f"unknown case {SMOKE_CASE}"})))
    assert turn["ui_state"] == "failed" and turn["failed_cause"] == "kit_lost_case"


@pytest.mark.parametrize("answer,state", [
    ((transport("before_send", "dns")), "kit_unreachable"),
    ((transport("after_send", "TimeoutError")), "outcome_unknown"),
    ((502, {}, "x"), "outcome_unknown"),
    ((504, {}, "x"), "outcome_unknown"),
    ((429, {}, "x"), "outcome_unknown"),
])
def test_send_failures(tmp_path, answer, state):
    post = FakePost({"front___start_case": [answer]})
    svc = service(tmp_path, post, clock=Clock())
    turn = svc._turn("start", "hello there", None)
    asyncio.run(svc.send_turn(turn["turn_id"]))
    assert turn["ui_state"] == state
    assert post.names() == ["front___start_case"]  # never resent


def test_cedar_deny_and_auth(tmp_path):
    svc = service(tmp_path, clock=Clock())
    t = sent(svc)
    svc.apply_send(t, classify(200, {}, '{"jsonrpc":"2.0","id":1,"error":{"code":-32002,"message":"Tool Execution Denied: x"}}'))
    assert t["ui_state"] == "refused_policy"
    t = sent(svc)
    svc.apply_send(t, classify(401, {}, "no"))
    assert t["ui_state"] == "refused_auth"


def test_auth_with_expired_token_is_token_expired(tmp_path):
    svc = service(tmp_path, clock=Clock(4_000_000_000), exp_offset=10)
    t = sent(svc)
    svc.apply_send(t, classify(401, {}, "no"))
    assert t["ui_state"] == "token_expired"


def test_a2a_error_on_start_is_lead_unconfirmed_never_nothing_ran(tmp_path):
    svc = service(tmp_path, clock=Clock())
    t = sent(svc)
    svc.apply_send(t, classify(*mcp_ok({"error": "<urlopen error timed out>"})))
    assert t["ui_state"] == "lead_unconfirmed"
    t = sent(svc)
    svc.apply_send(t, classify(*mcp_ok({"error": "A2A error -32000: HTTP Error 401: Unauthorized"})))
    assert t["ui_state"] == "lead_unconfirmed"


def _resp_line(tag, tool="front___start_case", case_id=SMOKE_CASE, refused=False):
    return log_event({"record": "checkpoint", "phase": "response", "agent": "team_app", "method": "tools/call",
                      "tool": tool, "case_id": case_id, "arguments": {"request": f"x\n\nref: {tag}"},
                      "refused": refused, "status_code": 200, "error": None})


def _opened_line(tag, case_id=SMOKE_CASE):
    return log_event({"record": "case_opened", "case_id": case_id, "request": f"x\n\nref: {tag}",
                      "requested_by": "front door"})


def _timing(**kw):
    return Timing(reconcile_every_s=0.0, reconcile_window_s=0.0, **kw)


def test_reconcile_start_found_by_response_line(tmp_path):
    svc = service(tmp_path, clock=Clock(), timing=_timing())
    t = sent(svc)
    t["ui_state"] = "outcome_unknown"
    svc.aws.events = [_opened_line(t["tag"]), _resp_line(t["tag"])]
    asyncio.run(svc.reconcile(t["turn_id"]))
    assert t["ui_state"] == "running" and t["case_id"] == SMOKE_CASE


def test_reconcile_case_opened_alone_is_not_success(tmp_path):
    svc = service(tmp_path, clock=Clock(), timing=_timing())
    t = sent(svc)
    t["ui_state"] = "outcome_unknown"
    svc.aws.events = [_opened_line(t["tag"]), _resp_line(t["tag"], case_id=None)]
    asyncio.run(svc.reconcile(t["turn_id"]))
    assert t["ui_state"] == "lead_unconfirmed" and t["case_id"] == SMOKE_CASE


def test_reconcile_nothing_found_and_no_aws(tmp_path):
    svc = service(tmp_path, clock=Clock(), timing=_timing())
    t = sent(svc)
    t["ui_state"] = "outcome_unknown"
    asyncio.run(svc.reconcile(t["turn_id"]))
    assert t["ui_state"] == "not_received"
    svc.aws = FakeAws(down="AWS sign-in on this laptop has expired")
    t = sent(svc)
    t["ui_state"] = "outcome_unknown"
    asyncio.run(svc.reconcile(t["turn_id"]))
    assert t["ui_state"] == "unconfirmed"


def test_continue_timeout_old_reply_not_reused(tmp_path):
    clock = Clock()
    svc = service(tmp_path, clock=clock, timing=_timing())
    first = accepted(svc, clock)
    svc.apply_status(first, classify(*status_done()))
    nxt = svc._turn("continue", "And cell 1?", SMOKE_CASE)
    nxt["ui_state"], nxt["sent_at"] = "outcome_unknown", clock.t
    svc.aws.events = [_resp_line(nxt["tag"], tool="front___continue_case")]
    asyncio.run(svc.reconcile(nxt["turn_id"]))
    assert nxt["ui_state"] == "running" and nxt["confirming"]
    svc.apply_status(nxt, classify(*status_done()))  # the kit still shows the OLD task
    assert nxt["ui_state"] == "running" and nxt["reply"] is None
    clock.t += 200
    svc.apply_status(nxt, classify(*status_done()))
    assert nxt["ui_state"] == "lead_unconfirmed" and nxt["reply"] is None
    other = svc._turn("continue", "And cell 0?", SMOKE_CASE)
    other.update(ui_state="running", sent_at=clock.t, accepted_at=clock.t, case_id=SMOKE_CASE, confirming=True)
    svc.apply_status(other, classify(*status_done(reply="cell 0 is fine")))
    assert other["ui_state"] == "answered" and other["reply"]["text"] == "cell 0 is fine"


def test_restart_never_resends(tmp_path):
    store = Store(tmp_path / "state")
    svc = service(tmp_path, clock=Clock(), store=store)
    q = svc._turn("start", "queued one", None)
    s = svc._turn("start", "sending one", None)
    s["ui_state"], s["sent_at"] = "sending", svc.now()
    svc.save()
    post = FakePost()
    svc2 = service(tmp_path, post, clock=Clock(), store=Store(tmp_path / "state"))
    svc2.resume()
    assert svc2.store.turns[q["turn_id"]]["ui_state"] == "cancelled_by_restart"
    assert svc2.store.turns[s["turn_id"]]["ui_state"] == "outcome_unknown"
    assert post.calls == []


def test_store_file_is_private_and_logs_hold_no_text(tmp_path):
    store = Store(tmp_path / "state")
    svc = service(tmp_path, clock=Clock(), store=store)
    t = accepted(svc, None)
    svc.apply_status(t, classify(*status_done()))
    assert oct((tmp_path / "state" / "state.json").stat().st_mode)[-3:] == "600"
    logs = list((tmp_path / "state" / "logs").glob("turns-*.jsonl"))
    body = logs[0].read_text()
    assert "Cell 2" not in body and "ran_monitor" not in body and t["turn_id"] in body


def test_stop_following_frees_the_slot(tmp_path):
    clock = Clock()
    svc = service(tmp_path, clock=clock)
    t = accepted(svc, clock)
    with pytest.raises(RefusedRequest) as e:
        svc.check_start()
    assert e.value.code == "case_in_flight"
    svc.stop_following(t["turn_id"])
    assert t["ui_state"] == "abandoned"
    svc.check_start()


def test_running_turn_releases_the_slot_after_360s(tmp_path):
    clock = Clock()
    svc = service(tmp_path, clock=clock)
    accepted(svc, clock)
    clock.t += 361
    svc.check_start()


def test_offline_and_expired_token_refuse_before_any_call(tmp_path):
    post = FakePost()
    svc = service(tmp_path, post, clock=Clock(), offline=True)
    with pytest.raises(RefusedRequest) as e:
        svc.start("hi")
    assert e.value.code == "offline"
    svc = service(tmp_path, post, clock=Clock(4_000_000_000), exp_offset=10)
    with pytest.raises(RefusedRequest) as e:
        svc.start("hi")
    assert e.value.code == "token_expired" and post.calls == []


# ---------------------------------------------------------------- follow-ups and the approval relay

def answered_case(svc, waiting=None):
    t = accepted(svc, None)
    svc.apply_status(t, classify(*status_done(waiting=waiting)))
    return t


def follow(svc, text):
    return asyncio.run(svc.check_follow_up(SMOKE_CASE, text))


def test_follow_up_rules(tmp_path):
    svc = service(tmp_path, clock=Clock())
    with pytest.raises(RefusedRequest) as e:
        follow(svc, "hi")
    assert e.value.code == "unknown_case"
    t = accepted(svc, None)
    with pytest.raises(RefusedRequest) as e:
        follow(svc, "and?")
    assert e.value.code == "turn_in_progress"
    svc.apply_status(t, classify(*status_done()))
    follow(svc, "Thanks. Which cell is second worst?")
    with pytest.raises(RefusedRequest) as e:
        follow(svc, "Approved.")
    assert e.value.code == "no_approval_known"


def test_relay_refused_while_pending(tmp_path):
    svc = service(tmp_path, FakePost(), FakeAws(rows={CHG: {"status": "PENDING"}}), clock=Clock())
    answered_case(svc, waiting_row())
    with pytest.raises(RefusedRequest) as e:
        follow(svc, "Approved.")
    assert e.value.code == "approval_still_pending"


def test_relay_refused_without_aws_reads(tmp_path):
    svc = service(tmp_path, FakePost(), FakeAws(down="AWS sign-in on this laptop has expired"), clock=Clock())
    answered_case(svc, waiting_row())
    with pytest.raises(RefusedRequest) as e:
        follow(svc, "anything at all")
    assert e.value.code == "approval_state_unknown"


def test_relay_refused_while_a_suite_run_is_active(tmp_path):
    aws = FakeAws(rows={CHG: {"status": "APPROVED"}, "t3-active": {"run_id": "t3.deny-20261102-a", "updated_at": "1"}})
    svc = service(tmp_path, FakePost(), aws, clock=Clock())
    answered_case(svc, waiting_row())
    with pytest.raises(RefusedRequest) as e:
        follow(svc, "Approved.")
    assert e.value.code == "judged_run_active" and "t3.deny-20261102-a" in e.value.message


def test_relay_allowed_after_decision_with_no_suite_run(tmp_path):
    aws = FakeAws(rows={CHG: {"status": "APPROVED"}})
    svc = service(tmp_path, FakePost(), aws, clock=Clock())
    answered_case(svc, waiting_row())
    follow(svc, "Approved.")


def test_relay_allowed_when_lead_marks_the_run_non_judged(tmp_path):
    aws = FakeAws(rows={CHG: {"status": "REJECTED"}, "t3-active": {"run_id": "t3.scratch-1", "updated_at": "1"}})
    svc = service(tmp_path, FakePost(), aws, clock=Clock(), env={"ASKNOC_NONJUDGED_RUNS": "t3.scratch-1"})
    answered_case(svc, waiting_row())
    follow(svc, "Rejected.")


def test_non_aws_gateway_is_refused_unless_allowed_and_labelled_test(tmp_path):
    post = FakePost({"front___start_case": [started()]})
    svc = service(tmp_path, post, clock=Clock(), url="http://127.0.0.1:8795/mcp")
    with pytest.raises(RefusedRequest) as e:
        svc.check_start()
    assert e.value.code == "not_real_gateway" and post.calls == []
    svc = service(tmp_path, post, clock=Clock(), url="http://127.0.0.1:8795/mcp", env={"ASKNOC_ALLOW_TEST_GATEWAY": "1"})
    svc.check_start()
    t = accepted(svc, None)
    assert svc.turn_view(t)["label"] == "TEST"
