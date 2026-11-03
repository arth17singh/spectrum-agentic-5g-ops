"""Offline tests of the Track 3 suite, runner, check, pause floor and trace. No network, no AWS."""
from __future__ import annotations

import copy
import json
import random
import re
from pathlib import Path

import pytest

from t3.suite import check, common, logsrc, pause_floor, runner, suite_hash, trace, units
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
def make_runner(clock, ddb, gw, ev, policy=None, approve_after_s=30.0, approver=True, timeout=900.0,
                unit="deny", run_id="t3-test-run"):
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
    sink = getattr(ev, "runner_calls", None)
    if sink is None:
        sink = ev.runner_calls = {}
    return runner.Runner(run_id=run_id, gateway=gateway, evlib=ev, ddb=ddb, cfg=CFG, pauser=pauser,
                         approval_timeout_s=timeout, poll_s=5.0, sleep=sleep, clock=clock, log=lambda *a: None,
                         synthetic=True, unit=unit,
                         calls_sink=lambda rid, rec: sink.setdefault(rid, []).append(rec))


def run_suite(unit="deny", flip=None, approver=True, with_pause=True, apply_noop=False):
    clock, ddb, ev = Clock(), FakeDDB(), FakeEvlib()
    policy = FakePolicyClient(clock) if with_pause else None
    gw = FakeGateway(clock, ddb, policy, flip=flip)
    gw.apply_noop = apply_noop
    r = make_runner(clock, ddb, gw, ev, policy, approver=approver, timeout=120.0, unit=unit)
    write = ev.write_declared

    def write_declared(d):
        ev.posts_at_declaration = len(gw.posts)
        return write(d)
    ev.write_declared = write_declared
    summary = r.run()
    return summary, ev, gw, ddb, clock


SUITE = suite_hash.load_suite()
N_ALLOW = sum(1 for i in SUITE if i["expect"] == "ALLOW" or i.get("approval_step"))
N_DENY = len(SUITE) - N_ALLOW


def common_runner_checks(summary, ev, gw, ddb, unit_name, n):
    assert ev.calls[0] == ("declared", "t3-test-run")   # the declaration is written first
    assert ev.posts_at_declaration == 0                  # before any Gateway call
    decl = ev.declared[0]
    assert decl["test_set_sha256"] == suite_hash.suite_sha256(SUITE)  # the FULL frozen suite
    assert decl["unit"] == unit_name and decl["n_items"] == n and decl["subset"]
    assert gw.posts[0][1:] == ("team_app", "front___start_case")
    rows = ddb.approvals()
    assert rows["t3-active"]["run_id"] == "t3-test-run" and "status" not in rows["t3-active"]
    assert rows[f"t3-{summary['case_id']}"]["run_id"] == "t3-test-run"
    assert ("t3-test-run", "item") not in ev.records   # the client view is not evidence
    items = ev.runner_calls["t3-test-run"]
    assert all(r["kind"] == "call" for r in items)
    assert items[-1]["item_id"] == "setup-close" and "run_id: t3-test-run" in items[-1]["extra"]["args"]["message"]
    setup = items[0]
    assert setup["extra"]["setup"] and "run_id: t3-test-run" in setup["extra"]["args"]["request"]
    graded = [r for r in items if not r["extra"].get("setup")]
    assert len(graded) == n and summary["client_agree"] == n and summary["skipped"] == []
    prefix = common.sha256_text("t3-test-run")[:8]
    for c in {r["extra"]["case_id"] for r in graded if r["extra"].get("case_id")} - {"no-case"}:
        assert re.fullmatch(rf"case-{prefix}-\d\d", c), c
        assert rows[f"t3-{c}"]["run_id"] == "t3-test-run" and "status" not in rows[f"t3-{c}"]
    assert all(w[1] == "put" and w[2].startswith("t3-") for w in ddb.writes)  # never an approval row
    gaps = [b[0] - a[0] for a, b in zip(gw.posts, gw.posts[1:])]
    assert min(gaps) >= 2.0                                                     # A12 pacing
    assert all(r["aws_time_source"] == "gateway:http-date" for r in graded)
    return graded


def test_runner_deny_unit_declares_first_and_runs_only_its_items():
    summary, ev, gw, ddb, clock = run_suite("deny")
    graded = common_runner_checks(summary, ev, gw, ddb, "t3.deny", N_DENY)
    assert ev.declared[0]["subset"] == "expect=DENY;exclude=approval_step"
    assert {r["item_id"] for r in graded} == {i["id"] for i in units.select(SUITE, "deny")}
    assert all(r["extra"]["expect"] == "DENY" and not r["extra"].get("approval_step") for r in graded)
    ctl = ev.records[("t3-test-run", "control")]   # the pause window ran inside the deny unit, then resumed
    assert [c["action"] for c in ctl] == ["pause", "resume"]
    assert ctl[0]["policy_name"].startswith("team13_ranpause_predict_")


def test_runner_allow_unit_holds_the_approval_chain_and_waits_for_a_person():
    summary, ev, gw, ddb, clock = run_suite("allow")
    graded = common_runner_checks(summary, ev, gw, ddb, "t3.allow", N_ALLOW)
    assert ev.declared[0]["subset"] == "expect=ALLOW;include=approval_step"
    ids = [r["item_id"] for r in graded]
    assert {"t3-033", "t3-034", "t3-035"} <= set(ids)
    appr = [r for r in graded if r["extra"].get("approval_step")]
    assert [r["extra"]["observed"] for r in appr] == ["DENY", "ALLOW", "DENY"]
    assert appr[0]["extra"]["approval_id"] == appr[0]["extra"]["approval_id_expected"]
    assert appr[1]["extra"]["approval"]["status"] == "APPROVED"
    assert "person" in appr[1]["extra"]["approval"]["decided_by"]
    assert ("t3-test-run", "control") not in ev.records   # no pause in the allow unit


def test_units_partition_the_suite_and_all_uses_two_run_ids():
    deny, allow = units.select(SUITE, "deny"), units.select(SUITE, "allow")
    assert len(deny) + len(allow) == len(SUITE) == 41
    assert not {i["id"] for i in deny} & {i["id"] for i in allow}
    assert units.run_ids("1102a", "all") == [("1102a-deny", "deny"), ("1102a-allow", "allow")]
    # both units back to back on one account: separate declarations, records and case ids
    clock, ddb, ev = Clock(), FakeDDB(), FakeEvlib()
    policy = FakePolicyClient(clock)
    gw = FakeGateway(clock, ddb, policy)
    for rid, key in units.run_ids("1102a", "all"):
        make_runner(clock, ddb, gw, ev, policy, unit=key, run_id=rid).run()
    assert [(d["run_id"], d["unit"], d["n_items"]) for d in ev.declared] == [
        ("1102a-deny", "t3.deny", N_DENY), ("1102a-allow", "t3.allow", N_ALLOW)]
    assert ddb.approvals()["t3-active"]["run_id"] == "1102a-allow"
    d_cases = {r["extra"]["case_id"] for r in ev.runner_calls["1102a-deny"] if r["extra"].get("case_id")}
    a_cases = {r["extra"]["case_id"] for r in ev.runner_calls["1102a-allow"] if r["extra"].get("case_id")}
    assert not (d_cases & a_cases) - {"no-case"}
    for rid, key in units.run_ids("1102a", "all"):
        decl = next(d for d in ev.declared if d["run_id"] == rid)
        assert check.check(rid, ev.runner_calls[rid], gw.cp_rows, gw.span_rows, declared=decl,
                           spans_mode="required", log=lambda *a: None, tool_rows=gw.tool_rows) == 0


def test_case_ids_do_not_collide_for_similar_run_ids():
    it = {"id": "t3-007"}
    a, b = runner.item_case_id("t3-suite-20261102-01", it), runner.item_case_id("t3-suite-20261102-02", it)
    assert a != b and re.fullmatch(r"case-[0-9a-f]{8}-07", a)


def test_runner_skips_the_change_when_nobody_approves():
    summary, ev, gw, ddb, clock = run_suite("allow", approver=False)
    items = {r["item_id"]: r for r in ev.runner_calls["t3-test-run"]}
    after = [r for r in items.values() if r["extra"].get("approval_step") == "after_approval"][0]
    assert "approval timeout" in after["extra"]["skipped"]
    assert after["item_id"] in summary["skipped"]
    assert all(r["status"] == "PENDING" for k, r in ddb.approvals().items() if k.startswith("chg-"))


def test_runner_refuses_a_run_id_whose_cases_exist():
    clock, ddb, ev = Clock(), FakeDDB(), FakeEvlib()
    used = f"case-{common.sha256_text('t3-test-run')[:8]}-12"   # t3-012 is in the deny unit
    ddb.approvals()[f"t3-{used}"] = {"pk": f"t3-{used}", "run_id": "t3-test-old"}
    gw = FakeGateway(clock, ddb)
    r = make_runner(clock, ddb, gw, ev)
    with pytest.raises(RuntimeError, match="pick a new run id"):
        r.run()
    assert gw.posts == []  # refused before any call


def test_runner_without_policy_client_skips_the_pause_item():
    summary, ev, *_ = run_suite("deny", with_pause=False)
    assert summary["skipped"] == [i["id"] for i in SUITE if i.get("pause_window") == "inside"]
    summary, ev, *_ = run_suite("allow", with_pause=False)
    assert summary["skipped"] == []   # the after-resume call is a plain allowed call


def test_runner_dry_run_sends_nothing(capsys):
    assert runner.main(["--run-id", "t3-dry"]) == 0
    out = capsys.readouterr().out
    assert "DRY RUN: nothing sent" in out and "waits for a person" in out
    assert "run t3-dry-deny" in out and "run t3-dry-allow" in out


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
def records_and_rows(unit="deny", **kw):
    summary, ev, gw, ddb, clock = run_suite(unit, **kw)
    return ev.runner_calls["t3-test-run"], gw.cp_rows, gw.span_rows, ev.declared[0], gw.tool_rows


@pytest.mark.parametrize("unit,n", [("deny", N_DENY), ("allow", N_ALLOW)])
def test_check_each_unit_all_match_exits_0(unit, n):
    records, cp, spans, decl, tools = records_and_rows(unit)
    ev = FakeEvlib()
    code = check.check("t3-test-run", records, cp, spans, declared=decl, spans_mode="required", evlib=ev,
                       synthetic=True, log=lambda *a: None, tool_rows=tools)
    assert code == 0
    res = ev.results[0]
    assert res["passed"] and res["value"] == 1.0 and res["n_items"] == n
    assert res["unit"] == f"t3.{unit}" and res["details"]["subset"] == units.UNITS[unit]["subset"]
    assert res["details"]["spans_joined"] == n
    decisions = ev.records[("t3-test-run", "decision")]
    assert len(decisions) == n and all(d["aws_time_source"] == "cloudwatch:/team13/checkpoint" for d in decisions)
    items = ev.records[("t3-test-run", "item")]
    assert all(i["extra"]["view"] == "aws" and i["passed"] for i in items)
    rules = {d["deciding_rule"] for d in decisions}
    if unit == "deny":
        assert {"team13_require_case_id", "default_deny"} <= rules
    else:
        assert "team13_require_human_approval" in rules


def test_check_one_mismatch_exits_1():
    # the Gateway gets ran___preflight wrong: AWS records DENY where the suite expects ALLOW
    records, cp, spans, decl, tools = records_and_rows("allow", flip={"ran___preflight"})
    ev = FakeEvlib()
    code = check.check("t3-test-run", records, cp, spans, declared=decl, evlib=ev, log=lambda *a: None,
                       tool_rows=tools)
    assert code == 1
    assert [m["item_id"] for m in ev.results[0]["details"]["mismatches"]] == ["t3-006"]


def test_check_will_not_grade_one_unit_as_the_other():
    records, cp, spans, decl, _tools = records_and_rows("deny")
    assert check.check("t3-test-run", records, cp, spans, declared=decl, unit="allow", log=lambda *a: None) == 1
    ev = FakeEvlib()   # without a declaration, the deny items are outside the allow subset
    assert check.check("t3-test-run", records, cp, spans, unit="allow", evlib=ev, log=lambda *a: None) == 1
    assert all("not in the t3.allow subset" in " ".join(m["problems"])
               for m in ev.results[0]["details"]["mismatches"])
    assert check.check("t3-test-run", records, cp, spans, log=lambda *a: None) == 1   # no unit at all


def test_check_uses_aws_records_not_the_runner_view():
    """If the runner's own view is tampered with, the grade still comes from the logs."""
    records, cp, spans, decl, _tools = records_and_rows("deny")
    tampered = copy.deepcopy(records)
    for r in tampered:
        r["extra"]["observed"] = "ALLOW"
    assert check.check("t3-test-run", tampered, cp, spans, declared=decl, log=lambda *a: None) == 0
    cp2 = copy.deepcopy(cp)   # a tampered log row (one DENY rewritten as ALLOW) is caught
    for row in cp2:
        msg = json.loads(row["@message"])
        if msg.get("phase") == "response" and msg.get("tool") == "swarm___ask_agent" and msg.get("refused"):
            msg["refused"], msg["error"] = False, None
            row["@message"] = json.dumps(msg)
            break
    assert check.check("t3-test-run", records, cp2, spans, declared=decl, log=lambda *a: None) == 1


def test_check_flags_sources_that_disagree_and_missing_spans():
    records, cp, spans, decl, _tools = records_and_rows("deny")
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
    summary, ev, gw, ddb, clock = run_suite("deny", with_pause=False)
    code = check.check("t3-test-run", ev.runner_calls["t3-test-run"], gw.cp_rows, gw.span_rows,
                       declared=ev.declared[0], log=lambda *a: None)
    assert code == 1


def test_check_reads_logs_insights_shapes_offline(tmp_path):
    records, cp, spans, decl, _tools = records_and_rows("deny")
    insights = {"status": "Complete", "results": [
        [{"field": "@timestamp", "value": common.to_iso(r["@timestamp"]).replace("T", " ").rstrip("Z")},
         {"field": "@message", "value": r["@message"]}, {"field": "@ptr", "value": r["@ptr"]}] for r in cp]}
    p = tmp_path / "cp.json"
    p.write_text(json.dumps(insights))
    rows = logsrc.load_rows(p, logsrc.CHECKPOINT_LOG)
    assert len(rows) == len(cp) and rows[0]["@log"] == "/team13/checkpoint"
    assert check.check("t3-test-run", records, rows, spans, declared=decl, log=lambda *a: None) == 0
    ev_shape = {"events": [{"timestamp": r["@timestamp"], "message": r["@message"], "eventId": r["@ptr"],
                            "logStreamName": "s1"} for r in cp]}
    p2 = tmp_path / "cp2.json"
    p2.write_text(json.dumps(ev_shape))
    assert check.check("t3-test-run", records, logsrc.load_rows(p2, logsrc.CHECKPOINT_LOG), spans, declared=decl,
                       log=lambda *a: None) == 0


def test_check_refuses_a_suite_changed_after_the_declaration():
    records, cp, spans, decl, _tools = records_and_rows("deny")
    decl = dict(decl, test_set_sha256="0" * 64)
    assert check.check("t3-test-run", records, cp, spans, declared=decl, log=lambda *a: None) == 1


def test_check_cli_offline(tmp_path):
    records, cp, spans, decl, tools = records_and_rows("allow")
    (tmp_path / "tools.json").write_text(json.dumps(tools))
    (tmp_path / "item.jsonl").write_text("\n".join(json.dumps(r) for r in records))
    (tmp_path / "cp.json").write_text(json.dumps(cp))
    (tmp_path / "spans.json").write_text(json.dumps(spans))
    args = ["--run-id", "t3-test-run", "--unit", "allow", "--runner-records", str(tmp_path / "item.jsonl"),
            "--checkpoint-json", str(tmp_path / "cp.json"), "--spans-json", str(tmp_path / "spans.json"),
            "--tools-json", str(tmp_path / "tools.json"), "--no-write", "--evidence-dir", str(tmp_path)]
    assert check.main(args) == 0
    folder = tmp_path / "records" / "t3-test-run" / "item"   # the per-record layout of amendment A11
    folder.mkdir(parents=True)
    for i, r in enumerate(records):
        (folder / f"{i:03d}.json").write_text(json.dumps(r))
    assert check.read_records("t3-test-run", "item", tmp_path) == records
    assert check.main(args[:4] + args[6:]) == 0


def test_check_cli_all_grades_both_run_ids(tmp_path):
    clock, ddb, ev = Clock(), FakeDDB(), FakeEvlib()
    policy = FakePolicyClient(clock)
    gw = FakeGateway(clock, ddb, policy)
    for rid, key in units.run_ids("1102a", "all"):
        make_runner(clock, ddb, gw, ev, policy, unit=key, run_id=rid).run()
        (tmp_path / "records" / rid).mkdir(parents=True)
        (tmp_path / "records" / rid / "item.jsonl").write_text(
            "\n".join(json.dumps(r) for r in ev.runner_calls[rid]))
        (tmp_path / "declared").mkdir(exist_ok=True)
        decl = next(d for d in ev.declared if d["run_id"] == rid)
        (tmp_path / "declared" / f"{rid}.json").write_text(json.dumps(decl))
    (tmp_path / "cp.json").write_text(json.dumps(gw.cp_rows))
    (tmp_path / "spans.json").write_text(json.dumps(gw.span_rows))
    (tmp_path / "tools.json").write_text(json.dumps(gw.tool_rows))
    args = ["--run-id", "1102a", "--unit", "all", "--tools-json", str(tmp_path / "tools.json"),
            "--checkpoint-json", str(tmp_path / "cp.json"),
            "--spans-json", str(tmp_path / "spans.json"), "--spans", "required", "--no-write",
            "--evidence-dir", str(tmp_path)]
    assert check.main(args) == 0
    # the declared unit is enough when --unit is left out
    assert check.main(["--run-id", "1102a-deny"] + args[4:]) == 0
    # one broken unit fails `all`
    (tmp_path / "records" / "1102a-allow" / "item.jsonl").write_text("")
    assert check.main(args) == 1


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
    assert ev.declared[0]["unit"] == "t3.pause" and ev.declared[0]["target"] == {
        "metric": "pass_rate", "op": ">=", "value": 1.0, "unit": "ratio"} and ev.declared[0]["n_items"] == 9
    d = res["details"]
    assert res["passed"] and d["allowed_after_forbid"] == 0 and d["calls_after_first_refusal"] == 5
    assert 0 <= d["first_refusal_s"] <= 60 and d["forbid_time_source"] == "agentcore:policy.createdAt"
    assert d["control_before_allowed"] and d["control_after_resume_allowed"]
    assert [c["action"] for c in ev.records[("t3-test-run", "control")]] == ["pause", "resume"]
    assert all(p["deleted"] for p in floor.p.c.policies.values())  # resumed: the forbid is gone
    assert d["tool"] == "ran___predict" and d["policy_name"].startswith("team13_ranpause_predict_")
    calls = ev.runner_calls["t3-test-run"]
    assert calls[0]["extra"]["caller"] == "ran_optimizer"
    graded = ev.records[("t3-test-run", "item")]
    assert len(graded) == 9 and all(i["passed"] for i in graded) and res["value"] == 1.0 and res["n_items"] == 9
    assert [i["item_id"] for i in graded][:3] == ["pause-before", "pause-first-after-active",
                                                   "pause-first-refusal-latency"]
    # the floor waited for GetPolicy to say ACTIVE before the "next call is denied" step, and recorded it
    wait = d["wait_until_active"]
    assert wait["want"] == "ACTIVE" and wait["statuses"][-1] == "ACTIVE" and wait["polls"] >= 2
    assert wait["waited_s"] >= 6.0 and wait["since_forbid_s"] >= 6.0      # the fake propagates in 6 s
    assert d["first_call_after_active"] == "DENY" and d["allowed_while_propagating"] == 0
    w = next(r for r in calls if r["item_id"] == "setup-wait-active")
    calls_after = [r for r in calls if r["extra"].get("phase") == "after_active"]
    assert w["extra"]["wait"]["seen_at"] <= calls_after[0]["extra"]["sent_at"]


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


# ---------- red-team fixes: allowed means applied; Gateway bypass; one trace id ----------
def test_check_allow_needs_a_change_applied_line_from_our_tools():
    # the approval is consumed (the Gateway allows the call) but our tools Lambda never applies anything
    records, cp, spans, decl, tools = records_and_rows("allow", apply_noop=True)
    ev = FakeEvlib()
    assert check.check("t3-test-run", records, cp, spans, declared=decl, evlib=ev, log=lambda *a: None,
                       tool_rows=tools) == 1
    mism = ev.results[0]["details"]["mismatches"]
    assert [m["item_id"] for m in mism] == ["t3-034"]
    assert "allowed but not applied" in mism[0]["problems"][0]
    # and with no tools log at all, the allowed apply cannot pass either
    assert check.check("t3-test-run", records, cp, spans, declared=decl, log=lambda *a: None) == 1


def test_check_counts_only_our_tools_lines_inside_the_run_window():
    records, cp, spans, decl, tools = records_and_rows("allow")
    ok = check.check("t3-test-run", records, cp, spans, declared=decl, log=lambda *a: None, tool_rows=tools)
    assert ok == 0

    def mutate(fn):
        rows = copy.deepcopy(tools)
        for r in rows:
            if json.loads(r["@message"]).get("record") == "change_applied":
                fn(r)
        return check.check("t3-test-run", records, cp, spans, declared=decl, log=lambda *a: None, tool_rows=rows)
    assert mutate(lambda r: r.update({"@log": "/team13/checkpoint"})) == 1          # another log group
    assert mutate(lambda r: r.update({"@timestamp": r["@timestamp"] + 86_400_000})) == 1   # outside the window
    assert mutate(lambda r: r.update({"@message": r["@message"].replace('"source": "team13-ran-tools"',
                                                                        '"source": "kit"')})) == 1
    assert mutate(lambda r: r.update({"@message": r["@message"].replace('"item_id": "step-0"',
                                                                        '"item_id": "step-9"')})) == 1


def bypass_rows():
    rows, bedrock, tools = trace_rows()
    t0 = 1_793_650_000_000
    stray = [{"@timestamp": t0 + 13_500, "@log": "/aws/lambda/team13-ran-tools", "@message": json.dumps(
        {"record": "ran_tool", "source": "team13-ran-tools", "tool": "ran___apply_change",
         "case_id": "case-0a1b2c3d", "ok": True})},
        {"@timestamp": t0 + 13_600, "@log": "/aws/lambda/team13-ran-tools", "@message": json.dumps(
            {"record": "change_applied", "source": "team13-ran-tools", "case_id": "case-0a1b2c3d",
             "approval_id": "chg-dddddddddddd", "change_text": "cell=1;d_power_db=+3.0;d_tilt_deg=+0.0"})}]
    return rows, bedrock, tools, stray


def test_trace_flags_tool_lines_without_a_gateway_decision():
    rows, bedrock, tools, stray = bypass_rows()
    # with the tools' own lines (each inside an allowed Gateway call) the trace passes
    calls_lines = [{"@timestamp": 1_793_650_000_000 + 2500, "@log": "/aws/lambda/team13-ran-tools",
                    "@message": json.dumps({"record": "ran_tool", "source": "team13-ran-tools",
                                            "tool": "ran___observe", "case_id": "case-0a1b2c3d", "ok": True})}]
    good = trace.rebuild("case-0a1b2c3d", rows, [], bedrock, tool_rows=tools + calls_lines)
    assert good["passed"] and good["no_gateway_decision"] == []
    tool_exec = [e for e in good["events"] if e["kind"] == "tool_exec"][0]
    assert tool_exec["gateway_call"]["agent"] == "ran_monitor"
    # a tool run and a change with no Gateway call around them: listed, and the trace FAILS
    bad = trace.rebuild("case-0a1b2c3d", rows, [], bedrock, tool_rows=tools + calls_lines + stray)
    assert not bad["passed"]
    assert [(b["kind"], b["problem"]) for b in bad["no_gateway_decision"]] == [
        ("tool_exec", "no Gateway decision"), ("change_applied", "no Gateway decision")]
    assert all("no Gateway decision" in p for p in bad["problems"])


def test_trace_line_with_a_trace_id_must_match_that_call():
    rows, bedrock, tools = trace_rows()
    t0 = 1_793_650_000_000
    spans = [{"@timestamp": t0 + 2200, "@message": json.dumps({"traceId": "1-abc", "attributes": {
        "aws.agentcore.policy.authorization_decision": "ALLOW", "mcp.tool.name": "ran___observe"}})}]
    line = {"@timestamp": t0 + 2500, "@log": "/aws/lambda/team13-ran-tools", "@message": json.dumps(
        {"record": "ran_tool", "source": "team13-ran-tools", "tool": "ran___observe", "case_id": "case-0a1b2c3d",
         "trace_id": "1-abc"})}
    assert trace.rebuild("case-0a1b2c3d", rows, spans, bedrock, tool_rows=tools + [line])["no_gateway_decision"] == []
    other = copy.deepcopy(line)
    other["@message"] = other["@message"].replace("1-abc", "1-zzz")
    out = trace.rebuild("case-0a1b2c3d", rows, spans, bedrock, tool_rows=tools + [other])
    assert [b["kind"] for b in out["no_gateway_decision"]] == ["tool_exec"]


def test_trace_is_one_case_id_end_to_end_with_span_trace_ids_joined():
    rows, bedrock, tools = trace_rows()
    t0 = 1_793_650_000_000
    msgs = [json.loads(r["@message"]) for r in rows]
    spans = []
    for r, m in zip(rows, msgs):  # one Cedar span per Gateway call, no case id in it (as AWS writes them)
        if m.get("record") == "checkpoint" and m.get("phase") == "request" and m["case_id"] == "case-0a1b2c3d":
            spans.append({"@timestamp": r["@timestamp"] + 200, "@message": json.dumps({
                "traceId": f"1-{len(spans):03d}", "attributes": {
                    "aws.agentcore.policy.authorization_decision": "ALLOW", "mcp.tool.name": m["tool"]}})})
    spans.append({"@timestamp": t0 + 99_000_000, "@message": json.dumps({"traceId": "1-far", "attributes": {
        "aws.agentcore.policy.authorization_decision": "DENY", "mcp.tool.name": "ran___observe"}})})
    out = trace.rebuild("case-0a1b2c3d", rows, spans, bedrock, tool_rows=tools, spans_required=True)
    assert out["passed"], out["problems"]
    assert out["trace_id"] == "case-0a1b2c3d" and out["first_event"] <= out["last_event"]
    assert set(out["sources"]) == {"/team13/checkpoint", "/aws/lambda/team13-ran-tools",
                                   "/team13/bedrock-invocations"}
    assert "1-far" not in out["span_trace_ids"] and len(out["span_trace_ids"]) == len(spans) - 1
    assert out["calls_without_span"] == []
    # every case-scoped record names exactly this case; a record of another case never enters
    assert all("case-ffffffff" not in json.dumps(e) for e in out["events"])
    # spans required but missing: FAIL
    assert not trace.rebuild("case-0a1b2c3d", rows, [], bedrock, tool_rows=tools, spans_required=True)["passed"]


def test_trace_ignores_a_record_of_another_case_that_mentions_this_one():
    rows, bedrock, tools = trace_rows()
    foreign = {"@timestamp": 1_793_650_000_000 + 3000, "@message": json.dumps(
        {"record": "approval", "decision": "APPROVED", "approval_id": "chg-eeeeeeeeeeee", "case_id": "case-ffffffff",
         "reason": "see case-0a1b2c3d"})}
    out = trace.rebuild("case-0a1b2c3d", rows + [foreign], [], bedrock, tool_rows=tools)
    assert all(e["detail"].get("approval_id") != "chg-eeeeeeeeeeee" for e in out["events"])


def test_records_validate_against_the_contracts_when_present():
    """Green with and without the contracts package: skipped when it is not importable."""
    import sys as _sys
    contracts_dir = Path(__file__).resolve().parents[3] / "contracts"
    if contracts_dir.is_dir() and str(contracts_dir) not in _sys.path:
        _sys.path.insert(0, str(contracts_dir))
    tc = pytest.importorskip("team13_contracts")
    recs = []
    for unit in ("deny", "allow"):
        records, cp, spans, decl, tools = records_and_rows(unit)
        ev = FakeEvlib()
        check.check("t3-test-run", records, cp, spans, declared=decl, evlib=ev, log=lambda *a: None, tool_rows=tools)
        assert all(r["kind"] == "call" for r in records)   # the client view: not evidence, never written as such
        recs += [decl] + [r for v in ev.records.values() for r in v] + ev.results
    floor, ev, gw, clock = make_floor()
    floor.run()
    recs += ev.declared + [r for v in ev.records.values() for r in v] + ev.results
    for r in recs:
        tc.validate_record(r)


def test_runner_binding_row_is_read_by_the_real_tools_store():
    """map_case (runner) -> t3/tools/store.py DynamoStore on a botocore-validated fake table: the binding is
    "no case yet" with the run id, the first put replaces it, and the runner can never overwrite a case."""
    from t3.tools import store as tstore
    from t3.tools.tests.fakes import FakeDynamo
    ddb = FakeDynamo()
    clock = Clock()
    r = runner.Runner(run_id="1102b-allow", gateway=None, evlib=FakeEvlib(), ddb=ddb, cfg=CFG, clock=clock,
                      log=lambda *a: None, unit="allow")
    r.map_case("case-abc12345-34")
    s = tstore.DynamoStore("team13-approvals", client=ddb)
    assert s.get_case("case-abc12345-34") is None and s.case_run_id("case-abc12345-34") == "1102b-allow"
    doc = {"case_id": "case-abc12345-34", "run_id": "1102b-allow", "scenario": "s", "seed": 1, "config": {},
           "action_log": [], "twin_kind": "maveric-inprocess", "predictions": [], "created_at": "t"}
    s.put_case(doc, None)
    assert s.get_case("case-abc12345-34")["version"] == 1
    with pytest.raises(Exception, match="ConditionalCheckFailed"):
        r.map_case("case-abc12345-34")   # the runner cannot clobber a real case
    assert s.get_case("case-abc12345-34")["version"] == 1
