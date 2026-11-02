"""Offline tests of the Track 3 suite, runner, check, pause floor and trace. No network, no AWS."""
from __future__ import annotations

import copy
import json
import random
import re
from pathlib import Path

import pytest

from t3.suite import check, common, logsrc, pause_floor, runner, suite_hash, trace
from t3.suite.tests.fakes import (Clock, FakeDDB, FakeEvlib, FakeGateway, FakePolicyClient, PERMITS,
                                  tokens_env)

CFG = {"gateway_url": "https://gateway.invalid/mcp", "region": "us-east-2",
       "approvals_table": "team13-approvals", "model_id": "checkpoint:test", "operator": "person:tester"}


# ---------- the suite ----------
def test_suite_is_valid_and_covers_every_class():
    items = suite_hash.load_suite()
    assert suite_hash.validate(items) == []
    assert 30 <= len(items) <= 45
    classes = {i["rule_class"] for i in items}
    assert classes == suite_hash.RULE_CLASSES
    callers = {i["caller"] for i in items}
    assert callers == suite_hash.CALLERS
    # each agent has at least one allowed and one denied call
    for agent in suite_hash.CALLERS:
        assert any(i["caller"] == agent and i["expect"] == "ALLOW" for i in items), agent
        assert any(i["caller"] == agent and i["expect"] == "DENY" for i in items), agent
    # specialists calling ask_agent are all denied
    spec_asks = [i for i in items if i["tool"] == "swarm___ask_agent" and i["caller"] != "noc_lead"
                 and i["caller"] != "team_app"]
    assert len(spec_asks) == 3 and all(i["expect"] == "DENY" for i in spec_asks)
    steps = [i.get("approval_step") for i in items if i.get("approval_step")]
    assert steps == ["request", "after_approval", "replay"]
    windows = [i.get("pause_window") for i in items if i.get("pause_window")]
    assert windows == ["inside", "after_resume"]
    assert all("{case}" in json.dumps(i["args"]) or "case_id" not in i["args"] or i["rule_class"] ==
               "require_case_id" for i in items)


def test_suite_hash_is_stable_and_sensitive(tmp_path):
    items = suite_hash.load_suite()
    h = suite_hash.suite_sha256(items)
    assert h == suite_hash.suite_sha256(suite_hash.DEFAULT_SUITE)
    # re-formatting (key order, spacing) does not change it
    p = tmp_path / "s.jsonl"
    p.write_text("\n".join(json.dumps(dict(reversed(list(i.items()))), indent=None, separators=(", ", " : "))
                           for i in items) + "\n\n")
    assert suite_hash.suite_sha256(p) == h
    # any change in content or order does
    changed = copy.deepcopy(items)
    changed[0]["expect"] = "DENY"
    assert suite_hash.suite_sha256(changed) != h
    assert suite_hash.suite_sha256(items[::-1]) != h


def test_suite_expectations_agree_with_the_intended_rules():
    """The fake Gateway encodes the rules as the contract describes them; every expectation that does not
    depend on state (approval, pause) must agree with it. This catches a typo in the suite."""
    gw = FakeGateway(Clock(), FakeDDB())
    for it in suite_hash.load_suite():
        if it.get("approval_step") or it.get("pause_window"):
            continue
        args = common.fill(it["args"], runner.item_case_id("run-x-01", it), "run-x-01")
        decision, _ = gw.decide(it["caller"], it["tool"], args)
        assert decision == it["expect"], it["id"]


# ---------- runner against a fake Gateway ----------
def make_runner(clock, ddb, gw, ev, policy=None, approve_after_s=30.0, approver=True, timeout=900.0):
    tokens = common.Tokens(env=tokens_env())
    gateway = common.Gateway(CFG["gateway_url"], tokens, post=gw, clock=clock)
    state = {"waited": 0.0}

    def sleep(s):  # the test plays the PERSON: approves pending changes after a while
        clock.sleep(s)
        state["waited"] += s
        if approver and state["waited"] >= approve_after_s:
            for row in ddb.approvals().values():
                if row.get("status") == "PENDING":
                    row.update(status="APPROVED", decided_by="arn:aws:sts::000000000000:assumed-role/x/person")

    pauser = (pause_floor.CedarPause(policy, "engine-1", "arn:aws:bedrock-agentcore:us-east-2:0:gateway/gw",
                                     sleep=clock.sleep, clock=clock) if policy else None)
    return runner.Runner(run_id="t3-test-run", gateway=gateway, evlib=ev, ddb=ddb, cfg=CFG, pauser=pauser,
                         approval_timeout_s=timeout, poll_s=5.0, sleep=sleep, clock=clock, log=lambda *a: None,
                         synthetic=True)


def run_suite(flip=None, approver=True, with_pause=True):
    clock, ddb, ev = Clock(), FakeDDB(), FakeEvlib()
    policy = FakePolicyClient(clock) if with_pause else None
    gw = FakeGateway(clock, ddb, policy, flip=flip)
    r = make_runner(clock, ddb, gw, ev, policy, approver=approver, timeout=120.0)
    write = ev.write_declared

    def write_declared(d):
        ev.posts_at_declaration = len(gw.posts)
        return write(d)
    ev.write_declared = write_declared
    summary = r.run()
    return summary, ev, gw, ddb, clock


def test_runner_declares_first_carries_run_id_and_waits_for_a_person():
    summary, ev, gw, ddb, clock = run_suite()
    # the declaration is the very first thing written, before any Gateway call
    assert ev.calls[0] == ("declared", "t3-test-run")
    decl = ev.declared[0]
    assert decl["test_set_sha256"] == suite_hash.suite_sha256(suite_hash.load_suite())
    assert decl["n_items"] == 41 and decl["unit"] == "t3.deny_allow" and decl["run_id"] == "t3-test-run"
    assert ev.posts_at_declaration == 0  # nothing had been sent when the declaration was written
    assert gw.posts[0][1:] == ("team_app", "front___start_case")
    # amendment A10: the t3-active run row and every case row map to the run, with no status attribute
    rows = ddb.approvals()
    assert rows["t3-active"]["run_id"] == "t3-test-run" and "status" not in rows["t3-active"]
    case = summary["case_id"]
    assert rows[f"t3-{case}"]["run_id"] == "t3-test-run"
    item_cases = {r["extra"]["case_id"] for r in ev.records[("t3-test-run", "item")] if r["extra"].get("case_id")}
    assert "case-t3-test--33" in item_cases and "case-t3-test--01" in item_cases
    for c in item_cases - {"no-case"}:
        assert rows[f"t3-{c}"]["run_id"] == "t3-test-run" and "status" not in rows[f"t3-{c}"]
    items = ev.records[("t3-test-run", "item")]
    setup = items[0]
    assert setup["extra"]["setup"] and "run_id: t3-test-run" in setup["extra"]["args"]["request"]
    assert all(r["run_id"] == "t3-test-run" for r in items)
    graded = [r for r in items if not r["extra"].get("setup")]
    assert len(graded) == 41
    # every call carries the caller's own identity and the client view matches all expectations
    assert summary["client_agree"] == 41 and summary["skipped"] == []
    # the approval: refused first, allowed only after the person, refused again as already applied
    appr = [r for r in graded if r["extra"].get("approval_step")]
    assert [r["extra"]["observed"] for r in appr] == ["DENY", "ALLOW", "DENY"]
    assert appr[0]["extra"]["approval_id"] == appr[0]["extra"]["approval_id_expected"]
    assert appr[1]["extra"]["approval"]["status"] == "APPROVED"
    assert "person" in appr[1]["extra"]["approval"]["decided_by"]
    # the runner never approved anything: its only writes are t3- rows, never an approval (chg-) row
    assert all(w[1] == "put" and w[2].startswith("t3-") for w in ddb.writes)
    # A12: one call at a time, at least 2 s apart
    gaps = [b[0] - a[0] for a, b in zip(gw.posts, gw.posts[1:])]
    assert min(gaps) >= 2.0
    # pause and resume are control records
    ctl = ev.records[("t3-test-run", "control")]
    assert [c["action"] for c in ctl] == ["pause", "resume"]
    assert ctl[0]["policy_name"].startswith("team13_ranpause_predict_")
    # AWS time from the Gateway's Date header
    assert all(r["aws_time_source"] == "gateway:http-date" for r in graded)


def test_runner_skips_the_change_when_nobody_approves():
    summary, ev, gw, ddb, clock = run_suite(approver=False)
    items = {r["item_id"]: r for r in ev.records[("t3-test-run", "item")]}
    after = [r for r in items.values() if r["extra"].get("approval_step") == "after_approval"][0]
    assert "approval timeout" in after["extra"]["skipped"]
    assert after["item_id"] in summary["skipped"]
    # no approval was faked: the change row is still PENDING
    assert all(r["status"] == "PENDING" for k, r in ddb.approvals().items() if k.startswith("chg-"))


def test_runner_refuses_a_run_id_whose_cases_exist():
    clock, ddb, ev = Clock(), FakeDDB(), FakeEvlib()
    ddb.approvals()["t3-case-t3-test--01"] = {"pk": "t3-case-t3-test--01", "run_id": "t3-test-old"}
    gw = FakeGateway(clock, ddb)
    r = make_runner(clock, ddb, gw, ev)
    with pytest.raises(RuntimeError, match="first 8 characters"):
        r.run()
    assert gw.posts == []  # refused before any call


def test_runner_without_policy_client_skips_pause_items():
    summary, ev, *_ = run_suite(with_pause=False)
    pause_ids = [i["id"] for i in suite_hash.load_suite() if i.get("pause_window")]
    assert summary["skipped"] == pause_ids


def test_runner_dry_run_sends_nothing(capsys):
    assert runner.main(["--run-id", "t3-dry"]) == 0
    out = capsys.readouterr().out
    assert "DRY RUN: nothing sent" in out and "waits for a person" in out


def test_tokens_never_come_from_the_repo(tmp_path, monkeypatch):
    t = common.Tokens(env={})
    with pytest.raises(KeyError):
        t.get("noc_lead")
    (tmp_path / "noc_lead.txt").write_text("abc\n")
    assert common.Tokens(env={}, tokens_dir=tmp_path).get("noc_lead") == "abc"

    class Secrets:
        def get_secret_value(self, SecretId):
            assert SecretId == "team13/agent-tokens"
            return {"SecretString": json.dumps({"ran_monitor": "s1"})}
    assert common.Tokens(env={"T3_TOKENS_SECRET": "team13/agent-tokens"}, secrets_client=Secrets()).get(
        "ran_monitor") == "s1"


def test_gateway_reads_http_errors_and_streams_as_answers():
    def post(url, body, headers, timeout):
        return 403, {"Date": "Mon, 02 Nov 2026 20:00:00 GMT"}, '{"message":"Forbidden"}'
    g = common.Gateway("https://x.invalid", common.Tokens(env=tokens_env()), post=post, clock=Clock())
    out = g.call("noc_lead", "ran___observe", {"case_id": "case-1"})
    assert out["decision"] == "DENY" and out["http_status"] == 403
    assert out["gateway_date"] == "2026-11-02T20:00:00.000Z"


# ---------- check against fixture rows ----------
def records_and_rows(**kw):
    summary, ev, gw, ddb, clock = run_suite(**kw)
    return ev.records[("t3-test-run", "item")], gw.cp_rows, gw.span_rows, ev.declared[0]


def test_check_all_match_exits_0():
    records, cp, spans, decl = records_and_rows()
    ev = FakeEvlib()
    code = check.check("t3-test-run", records, cp, spans, declared=decl, spans_mode="required", evlib=ev,
                       synthetic=True, log=lambda *a: None)
    assert code == 0
    res = ev.results[0]
    assert res["passed"] and res["value"] == 1.0 and res["n_items"] == 41
    assert res["details"]["by_unit"]["t3.deny"]["passed"] and res["details"]["by_unit"]["t3.allow"]["passed"]
    assert res["details"]["spans_joined"] == 41
    decisions = ev.records[("t3-test-run", "decision")]
    assert len(decisions) == 41 and all(d["aws_time_source"] == "cloudwatch:/team13/checkpoint" for d in decisions)
    rules = {d["item_id"]: d["deciding_rule"] for d in decisions}
    assert any(r == "team13_require_case_id" for r in rules.values())
    assert any(r == "default_deny" for r in rules.values())
    items = ev.records[("t3-test-run", "item")]
    assert all(i["extra"]["view"] == "aws" and i["passed"] for i in items)


def test_check_one_mismatch_exits_1():
    # the Gateway gets ran___preflight wrong: AWS records DENY where the suite expects ALLOW
    records, cp, spans, decl = records_and_rows(flip={"ran___preflight"})
    ev = FakeEvlib()
    code = check.check("t3-test-run", records, cp, spans, declared=decl, evlib=ev, log=lambda *a: None)
    assert code == 1
    mism = ev.results[0]["details"]["mismatches"]
    assert [m["item_id"] for m in mism] == ["t3-006"]


def test_check_uses_aws_records_not_the_runner_view():
    """If the runner's own view is tampered with, the grade still comes from the logs."""
    records, cp, spans, decl = records_and_rows()
    tampered = copy.deepcopy(records)
    for r in tampered:
        r["extra"]["observed"] = "ALLOW"
    assert check.check("t3-test-run", tampered, cp, spans, declared=decl, log=lambda *a: None) == 0
    # and a tampered log row (one DENY rewritten as ALLOW) is caught
    cp2 = copy.deepcopy(cp)
    for row in cp2:
        msg = json.loads(row["@message"])
        if msg.get("phase") == "response" and msg.get("tool") == "swarm___ask_agent" and msg.get("refused"):
            msg["refused"], msg["error"] = False, None
            row["@message"] = json.dumps(msg)
            break
    assert check.check("t3-test-run", records, cp2, spans, declared=decl, log=lambda *a: None) == 1


def test_check_flags_sources_that_disagree_and_missing_spans():
    records, cp, spans, decl = records_and_rows()
    spans2 = copy.deepcopy(spans)
    msg = json.loads(spans2[3]["@message"])
    attr = msg["attributes"]
    attr["aws.agentcore.policy.authorization_decision"] = (
        "DENY" if attr["aws.agentcore.policy.authorization_decision"] == "ALLOW" else "ALLOW")
    spans2[3]["@message"] = json.dumps(msg)
    assert check.check("t3-test-run", records, cp, spans2, declared=decl, log=lambda *a: None) == 1
    assert check.check("t3-test-run", records, cp, [], declared=decl, spans_mode="required",
                       log=lambda *a: None) == 1
    assert check.check("t3-test-run", records, cp, [], declared=decl, spans_mode="optional",
                       log=lambda *a: None) == 0


def test_check_counts_skipped_items_as_not_passed():
    summary, ev, gw, ddb, clock = run_suite(with_pause=False)
    code = check.check("t3-test-run", ev.records[("t3-test-run", "item")], gw.cp_rows, gw.span_rows,
                       declared=ev.declared[0], log=lambda *a: None)
    assert code == 1


def test_check_reads_logs_insights_shapes_offline(tmp_path):
    records, cp, spans, decl = records_and_rows()
    # the get_query_results shape: rows of {field, value}, @timestamp as text
    insights = {"status": "Complete", "results": [
        [{"field": "@timestamp", "value": common.to_iso(r["@timestamp"]).replace("T", " ").rstrip("Z")},
         {"field": "@message", "value": r["@message"]}, {"field": "@ptr", "value": r["@ptr"]}] for r in cp]}
    p = tmp_path / "cp.json"
    p.write_text(json.dumps(insights))
    rows = logsrc.load_rows(p, logsrc.CHECKPOINT_LOG)
    assert len(rows) == len(cp) and rows[0]["@log"] == "/team13/checkpoint"
    assert check.check("t3-test-run", records, rows, spans, declared=decl, log=lambda *a: None) == 0
    # the filter_log_events shape
    ev_shape = {"events": [{"timestamp": r["@timestamp"], "message": r["@message"], "eventId": r["@ptr"],
                            "logStreamName": "s1"} for r in cp]}
    p2 = tmp_path / "cp2.json"
    p2.write_text(json.dumps(ev_shape))
    assert check.check("t3-test-run", records, logsrc.load_rows(p2, logsrc.CHECKPOINT_LOG), spans, declared=decl,
                       log=lambda *a: None) == 0


def test_check_refuses_a_suite_changed_after_the_declaration():
    records, cp, spans, decl = records_and_rows()
    decl = dict(decl, test_set_sha256="0" * 64)
    assert check.check("t3-test-run", records, cp, spans, declared=decl, log=lambda *a: None) == 1


def test_check_cli_offline(tmp_path):
    records, cp, spans, decl = records_and_rows()
    (tmp_path / "item.jsonl").write_text("\n".join(json.dumps(r) for r in records))
    (tmp_path / "cp.json").write_text(json.dumps(cp))
    (tmp_path / "spans.json").write_text(json.dumps(spans))
    args = ["--run-id", "t3-test-run", "--runner-records", str(tmp_path / "item.jsonl"), "--checkpoint-json",
            str(tmp_path / "cp.json"), "--spans-json", str(tmp_path / "spans.json"), "--no-write",
            "--evidence-dir", str(tmp_path)]
    assert check.main(args) == 0
    # the per-record layout of amendment A11 reads the same
    folder = tmp_path / "records" / "t3-test-run" / "item"
    folder.mkdir(parents=True)
    for i, r in enumerate(records):
        (folder / f"{i:03d}.json").write_text(json.dumps(r))
    assert check.read_records("t3-test-run", "item", tmp_path) == records
    assert check.main(args[:2] + args[4:]) == 0


# ---------- pause floor ----------
def make_floor(propagate_s=6.0, leak=False):
    clock, ddb, ev = Clock(), FakeDDB(), FakeEvlib()
    policy = FakePolicyClient(clock, propagate_s=propagate_s)
    gw = FakeGateway(clock, ddb, policy)
    if leak:  # a broken pause: the forbid never takes effect
        policy.forbids = lambda tool: False
    r = make_runner(clock, ddb, gw, ev, policy)
    floor = pause_floor.PauseFloor(run_id="t3-test-run", runner=r, pauser=r.pauser, n_calls=5, interval_s=3.0,
                                   sleep=clock.sleep, clock=clock, log=lambda *a: None)
    return floor, ev, gw, clock


def test_pause_floor_passes_and_times_the_first_refusal():
    floor, ev, gw, clock = make_floor()
    res = floor.run()
    assert ev.calls[0] == ("declared", "t3-test-run")
    assert ev.declared[0]["unit"] == "t3.pause" and ev.declared[0]["target"]["value"] == 0
    d = res["details"]
    assert res["passed"] and d["allowed_after_forbid"] == 0 and d["calls_after_first_refusal"] == 5
    assert 0 <= d["first_refusal_s"] <= 60 and d["forbid_time_source"] == "agentcore:policy.createdAt"
    assert d["control_before_allowed"] and d["control_after_resume_allowed"]
    assert [c["action"] for c in ev.records[("t3-test-run", "control")]] == ["pause", "resume"]
    assert all(p["deleted"] for p in floor.p.c.policies.values())  # resumed: the forbid is gone
    assert d["tool"] == "ran___predict" and d["policy_name"].startswith("team13_ranpause_predict_")
    assert ev.records[("t3-test-run", "item")][0]["extra"]["caller"] == "ran_optimizer"


def test_pause_floor_fails_when_the_forbid_never_refuses():
    floor, ev, gw, clock = make_floor(leak=True)
    res = floor.run()
    assert not res["passed"] and res["details"]["first_refusal_s"] is None


def test_pause_floor_counts_a_call_allowed_after_the_first_refusal():
    floor, ev, gw, clock = make_floor()
    policy = floor.p.c
    real = policy.forbids
    seen = {"n": 0}

    def flaky(tool):  # refuses once, then leaks one call, then holds
        on = real(tool)
        if on:
            seen["n"] += 1
            return seen["n"] != 2
        return on
    policy.forbids = flaky
    res = floor.run()
    assert not res["passed"] and res["details"]["allowed_after_forbid"] == 1


def test_pause_floor_fails_when_the_first_refusal_is_late():
    floor, ev, gw, clock = make_floor(propagate_s=75.0)
    floor.max_wait_s = 200
    res = floor.run()
    assert res["details"]["first_refusal_s"] > 60 and not res["passed"]


def test_pause_recheck_from_logs_matches():
    floor, ev, gw, clock = make_floor()
    res = floor.run()
    ctl = ev.records[("t3-test-run", "control")]
    pol = next(iter(floor.p.c.policies.values()))
    v = pause_floor.recheck(floor.case, gw.cp_rows, {
        "pause_time": ctl[0]["aws_time"], "active_time": common.to_iso(pol["created"] + 6.0),
        "resume_time": ctl[1]["time"]}, 5)
    assert v["passed"] and v["allowed_after_forbid"] == 0


# ---------- trace ----------
def trace_rows(case="case-0a1b2c3d", shuffle=False, drop=None):
    t0 = 1_793_650_000_000
    seq = [
        {"record": "case_opened", "case_id": case, "requested_by": "front door"},
        {"record": "checkpoint", "phase": "request", "agent": "noc_lead", "method": "tools/call",
         "tool": "swarm___ask_agent", "case_id": case, "arguments": {"case_id": case, "agent": "ran_monitor"}},
        {"record": "checkpoint", "phase": "request", "agent": "ran_monitor", "method": "tools/call",
         "tool": "ran___observe", "case_id": case, "arguments": {"case_id": case}},
        {"record": "checkpoint", "phase": "response", "agent": "ran_monitor", "method": "tools/call",
         "tool": "ran___observe", "case_id": case, "refused": False},
        {"record": "checkpoint", "phase": "response", "agent": "noc_lead", "method": "tools/call",
         "tool": "swarm___ask_agent", "case_id": case, "refused": False, "arguments": {"agent": "ran_monitor"}},
        {"record": "checkpoint", "phase": "request", "agent": "noc_lead", "method": "tools/call",
         "tool": "ran___apply_change", "case_id": case, "approval": "PENDING", "approval_id": "chg-aaaaaaaaaaaa"},
        {"record": "checkpoint", "phase": "response", "agent": "noc_lead", "method": "tools/call",
         "tool": "ran___apply_change", "case_id": case, "refused": True, "approval_id": "chg-aaaaaaaaaaaa"},
        {"record": "approval", "decision": "APPROVED", "approval_id": "chg-aaaaaaaaaaaa", "case_id": case,
         "by": "arn:aws:sts::0:assumed-role/x/person"},
        {"record": "checkpoint", "phase": "request", "agent": "noc_lead", "method": "tools/call",
         "tool": "ran___apply_change", "case_id": case},

        {"record": "checkpoint", "phase": "response", "agent": "noc_lead", "method": "tools/call",
         "tool": "ran___apply_change", "case_id": case, "refused": False},
        {"record": "checkpoint", "phase": "request", "agent": "noc_lead", "method": "tools/call",
         "tool": "ran___observe", "case_id": case},
        {"record": "checkpoint", "phase": "response", "agent": "noc_lead", "method": "tools/call",
         "tool": "ran___observe", "case_id": case, "refused": False},
        {"record": "checkpoint", "phase": "response", "agent": "noc_lead", "method": "tools/call",
         "tool": "ran___observe", "case_id": "case-ffffffff", "refused": False},  # another case: ignored
    ]
    rows = [{"@timestamp": t0 + 1000 * i, "@message": json.dumps(m), "@ptr": f"p{i}"} for i, m in enumerate(seq)
            if m["record"] != drop]
    # amendment A11: change_applied comes from our tools' log group, between the two apply records
    i_apply = next(i for i, m in enumerate(seq) if m["record"] == "approval") + 1
    tools = [{"@timestamp": t0 + 1000 * i_apply + 500, "@log": "/aws/lambda/team13-ran-tools", "@message":
              "START RequestId: x"},
             {"@timestamp": t0 + 1000 * i_apply + 600, "@log": "/aws/lambda/team13-ran-tools", "@message": json.dumps(
                 {"record": "change_applied", "case_id": case, "run_id": "t3-trace-01",
                  "approval_id": "chg-aaaaaaaaaaaa", "change_text": "cell=2;d_power_db=-1.0;d_tilt_deg=+1.0"})}]
    if shuffle:
        random.Random(7).shuffle(rows)
        tools = tools[::-1]
    if drop == "change_applied":
        tools = []
    bedrock = [{"@timestamp": t0 + 500, "@message": json.dumps({"modelId": "openai.gpt-oss-120b-1:0",
                                                                "input": f"case_id: {case} ..."})}]
    return rows, bedrock, tools


def test_trace_rebuilds_in_aws_time_order_and_passes():
    rows, bedrock, tools = trace_rows(shuffle=True)
    out = trace.rebuild("case-0a1b2c3d", rows, [], bedrock, tool_rows=tools)
    assert out["passed"], out["missing"]
    kinds = [e["kind"] for e in out["events"]]
    assert kinds[0] == "case_opened" and kinds[1] == "bedrock_call"
    times = [e["time"] for e in out["events"]]
    assert times == sorted(times)
    assert [s["stage"] for s in out["stages"]] == trace.EXPECTED
    assert out["refused_apply_calls"] == 1 and out["approval_id"] == "chg-aaaaaaaaaaaa"
    assert all("case-ffffffff" not in json.dumps(e) for e in out["events"])
    applied = [e for e in out["events"] if e["kind"] == "change_applied"]
    assert len(applied) == 1 and applied[0]["source"] == "/aws/lambda/team13-ran-tools"
    assert applied[0]["detail"]["run_id"] == "t3-trace-01"


def test_trace_fails_without_the_approval_or_out_of_order():
    rows, _, tools = trace_rows(drop="approval")
    out = trace.rebuild("case-0a1b2c3d", rows, tool_rows=tools)
    assert not out["passed"] and "approval" in out["missing"]
    # change applied BEFORE the approval: out of order, so it fails
    rows, _, tools = trace_rows()
    i_appr = next(i for i, r in enumerate(rows) if json.loads(r["@message"])["record"] == "approval")
    rows[i_appr]["@timestamp"] = tools[1]["@timestamp"] + 1
    assert not trace.rebuild("case-0a1b2c3d", rows, tool_rows=tools)["passed"]
    # without the tools' log group there is no change_applied at all
    rows, _, tools = trace_rows()
    assert "change_applied" in trace.rebuild("case-0a1b2c3d", rows)["missing"]


def test_trace_cli_offline(tmp_path):
    rows, bedrock, tools = trace_rows()
    (tmp_path / "cp.json").write_text(json.dumps(rows))
    (tmp_path / "b.json").write_text(json.dumps(bedrock))
    (tmp_path / "tools.json").write_text(json.dumps(tools))
    assert trace.main(["check", "--case", "case-0a1b2c3d", "--checkpoint-json", str(tmp_path / "cp.json"),
                       "--bedrock-json", str(tmp_path / "b.json"), "--tools-json", str(tmp_path / "tools.json"),
                       "--out", str(tmp_path / "t.json")]) == 0
    assert json.loads((tmp_path / "t.json").read_text())["passed"]


def test_trace_drive_declares_then_waits_for_the_person():
    clock, ddb, ev = Clock(), FakeDDB(), FakeEvlib()

    class LeadGateway(FakeGateway):
        def __call__(self, url, body, headers, timeout):
            req = json.loads(body)
            tool = req["params"]["name"]
            if tool == "front___case_status":
                waiting = [{"approval_id": "chg-bbbbbbbbbbbb", "change": "{}"}] if not self.cases[1:] else []
                ddb.approvals().setdefault("chg-bbbbbbbbbbbb", {"pk": "chg-bbbbbbbbbbbb", "status": "PENDING"})
                return 200, self._headers(), json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"content": [
                    {"type": "text", "text": json.dumps({"state": "TASK_STATE_COMPLETED", "reply": "ok",
                                                         "waiting_approvals": waiting})}]}})
            if tool == "front___continue_case":
                self.cases.append("continued")
            return super().__call__(url, body, headers, timeout)

    gw = LeadGateway(clock, ddb)
    r = make_runner(clock, ddb, gw, ev)
    out = trace.drive(r, "Cell 2 has poor coverage. Find the cause and fix it.", sleep=r.sleep, clock=clock,
                      log=lambda *a: None)
    assert ev.calls[0] == ("declared", "t3-test-run") and ev.declared[0]["unit"] == "t3.trace"
    assert out["approved"]["status"] == "APPROVED"
    assert [p[2] for p in gw.posts][:1] == ["front___start_case"]
    assert "front___continue_case" in [p[2] for p in gw.posts]
    assert all(w[2].startswith("t3-") for w in ddb.writes)


def test_permits_table_matches_the_agents_prompts():
    """Each student's prompt names only the tools its rules allow."""
    from pathlib import Path
    agents = Path(__file__).resolve().parents[2] / "agents"
    for agent in ("ran_monitor", "ran_optimizer", "ran_explainer"):
        text = (agents / f"{agent}.md").read_text()
        for tool in ("ran___observe", "ran___propose", "ran___predict", "ran___preflight", "nlm___ask"):
            short = tool.split("___")[1]
            if tool in PERMITS[agent]:
                assert short in text, (agent, tool)


# ---------- agents ----------
AGENTS_DIR = Path(__file__).resolve().parents[2] / "agents"


def prompt_example(text):
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith('{"role"'))
    for end in range(start + 1, len(lines) + 1):
        try:
            return json.loads("\n".join(lines[start:end]))
        except ValueError:
            continue
    raise AssertionError("no JSON example in the prompt")


@pytest.mark.parametrize("agent", ["ran_monitor", "ran_optimizer", "ran_explainer"])
def test_student_prompt_example_matches_its_schema(agent):
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((AGENTS_DIR / "schemas" / f"{agent}.schema.json").read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    example = prompt_example((AGENTS_DIR / f"{agent}.md").read_text())
    assert example["case_id"] == "case-..."  # the prompt shows a placeholder; a real answer has the case id
    example["case_id"] = "case-0a1b2c3d"
    jsonschema.validate(example, schema)
    assert len(json.dumps(example).split()) <= 120
    if "change" in example:  # a non-canonical change string is rejected
        bad = dict(example, change="cell 2: -1 dB power, +1 deg tilt")
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bad, schema)


def test_kit_template_holds_placeholders_only():
    kit = json.loads((AGENTS_DIR / "kit.json.template").read_text())
    assert kit["name"] == "team13" and kit["region"] == "us-east-2" and kit["lead"] == "noc_lead"
    assert kit["specialists"] == ["ran_monitor", "ran_optimizer", "ran_explainer"]
    assert kit["may_change"] == ["noc_lead"]
    assert "YOUR-IP" in kit["builder_host"] and "paste" in kit["a2a_token"]
    assert all("paste" in v for v in kit["a2a_agents"].values())
    text = (AGENTS_DIR / "kit.json.template").read_text() + (AGENTS_DIR / "BUILDER_CHECKLIST.md").read_text()
    assert not re.search(r"\b\d{12}\b", text) and not re.search(r"\d+-\d+-\d+-\d+\.sslip", text)
    assert not re.search(r"eyJ[A-Za-z0-9_-]{10,}", text)  # no JWT


def test_lead_prompt_carries_the_rules():
    text = (AGENTS_DIR / "noc_lead.md").read_text()
    for must in ("open_case", "case_id", "apply_change", "approval id", "Approved.", "exactly the same",
                 "verify", "character for character", "run_id"):
        assert must in text, must
    assert text.index("ask ran_monitor") < text.index("ask ran_optimizer") < text.index("ask ran_explainer")


def test_brain_smoke_dry_run_and_fake_conversation(capsys):
    from t3.agents import brain_smoke
    assert brain_smoke.main([]) == 0
    assert "Nothing sent" in capsys.readouterr().out

    clock, ddb = Clock(), FakeDDB()
    state = {"turn": 0, "case": "case-1234abcd"}

    def post(url, body, headers, timeout):
        req = json.loads(body)
        tool, args = req["params"]["name"], req["params"]["arguments"]
        if tool == "front___start_case":
            assert "run_id: smoke-test" in args["request"]
            payload = {"case_id": state["case"], "state": "TASK_STATE_SUBMITTED"}
        elif tool == "front___continue_case":
            state["turn"] += 1
            payload = {"case_id": state["case"], "state": "TASK_STATE_SUBMITTED"}
        else:
            replies = [f"{state['case']}: waiting for approval chg-cccccccccccc",
                       f"{state['case']}: change applied and verified", f"{state['case']}: it held"]
            waiting = [{"approval_id": "chg-cccccccccccc", "change": "{}"}] if state["turn"] == 0 else []
            payload = {"state": "TASK_STATE_COMPLETED", "reply": replies[state["turn"]],
                       "waiting_approvals": waiting}
        return 200, {}, json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"content": [
            {"type": "text", "text": json.dumps(payload)}]}})

    gw = common.Gateway("https://x.invalid", common.Tokens(env=tokens_env()), post=post, clock=clock)
    t_start = clock()

    def sleep(s):  # the person approves after a while
        clock.sleep(s)
        if clock() - t_start > 30:
            ddb.approvals()["chg-cccccccccccc"] = {"pk": "chg-cccccccccccc", "status": "APPROVED"}
    out = brain_smoke.converse(gw, ddb, "team13-approvals", tag="smoke-test", sleep=sleep, clock=clock,
                               log=lambda *a: None)
    assert out["passed"], out["checks"]
    assert ddb.writes == []  # read only
