"""The swarm eval split: the live phase records proposals through the front door; the scored run replays them.

Offline: a fake Gateway plays the kit's front door and the lead (it proposes a fixed change and gets it
refused for approval); the recorded client is then driven by the real SwarmOperator exactly as run.py does
(start log from the harness twin's action log), with a stub twin that only keeps that log."""
from __future__ import annotations

import json

import pytest

from t3.evals import live_swarm as ls
from t3.evals.changes import canon, nonzero
from t3.evals.recorded_client import RecordedSwarmClient, step_text
from t3.evals.swarm import SwarmOperator
from t3.suite import common
from t3.suite.tests.fakes import Clock, FakeDDB, FakeEvlib, tokens_env

CFG = {"region": "us-east-2", "approvals_table": "team13-approvals", "gateway_url": "https://x.invalid"}


def incidents(n=2):
    return [{"incident_id": f"inc-test-{k}", "scenario": "nominal", "seed": 4100 + k, "preset": "official",
             "tier": "masked", "commissioning": [{"cell": 1, "d_power_db": 1.0, "d_tilt_deg": 0.0}],
             "fault": [{"cell": 0, "d_power_db": 0.0, "d_tilt_deg": 5.0}],
             "pre_fault_kpis": {"coverage_ratio": 0.95, "sla_ratio": 0.9, "sinr_p10_db": 1.0},
             "post_fault_kpis": {"coverage_ratio": 0.80, "sla_ratio": 0.7, "sinr_p10_db": -3.0}} for k in range(n)]


TARGETS = {"episode": {"fault_step": 2, "operator_steps": 3, "grade_step": 5}}
PROPOSALS = ["cell=0;d_power_db=+0.0;d_tilt_deg=-3.0", "", "cell=0;d_power_db=+0.0;d_tilt_deg=-1.0"]


class FakeFrontDoor:
    """team_app calls: ran___seed_case, front___continue_case, front___case_status (with the lead's refusal)."""

    def __init__(self, clock, fail_seed_for=()):
        self.clock, self.fail_seed_for = clock, set(fail_seed_for)
        self.seeded, self.messages, self.calls = {}, {}, []

    def __call__(self, url, body, headers, timeout):
        req = json.loads(body)
        tool, args = req["params"]["name"], req["params"]["arguments"]
        agent = headers["Authorization"].split()[-1].replace("tok-", "")
        self.calls.append((agent, tool, args.get("case_id")))
        assert agent == "team_app"
        cid = args.get("case_id")
        if tool == "ran___seed_case":
            if cid in self.fail_seed_for:
                payload = {"error": "twin replay failed"}
            else:
                self.seeded[cid] = json.loads(args["start_action_log_json"])
                payload = {"case_id": cid, "seeded": True, "step": len(self.seeded[cid])}
        elif tool == "front___continue_case":
            assert cid in self.seeded and "run_id=" in args["message"]
            self.messages[cid] = args["message"]
            payload = {"case_id": cid, "state": "TASK_STATE_SUBMITTED"}
        else:  # case_status: the lead finished and its apply waits for approval (unless it proposed nothing)
            step = int(cid[-1])
            change = PROPOSALS[step]
            waiting = [{"approval_id": f"chg-{step:012d}", "change": json.dumps(
                {"item_id": f"step-{2 + step}", "change": change, "reason": "fix"})}] if change else []
            payload = {"case_id": cid, "state": "TASK_STATE_COMPLETED", "reply": f"{cid}: done", "waiting_approvals": waiting}
        self.clock.sleep(0.5)
        return 200, {"Date": "Mon, 02 Nov 2026 22:00:00 GMT"}, json.dumps(
            {"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": json.dumps(payload)}]}})


def live(fail=()):
    clock, ddb, ev = Clock(), FakeDDB(), FakeEvlib()
    fd = FakeFrontDoor(clock, fail)
    gw = common.Gateway(CFG["gateway_url"], common.Tokens(env=tokens_env()), post=fd, clock=clock)
    run = ls.LiveSwarm(run_id="t3.swarm-live-test", gateway=gw, ddb=ddb, evlib=ev, cfg=CFG, targets=TARGETS,
                       sleep=clock.sleep, clock=clock, log=lambda *a: None, synthetic=True)
    out = run.run(incidents(), "d" * 64, steps=3, workers=1)
    return run, out, fd, ddb, ev


def test_live_phase_declares_first_seeds_each_case_and_records_the_refused_proposal():
    run, out, fd, ddb, ev = live()
    assert ev.calls[0] == ("declared", "t3.swarm-live-test")
    decl = ev.declared[0]
    assert decl["unit"] == "t3.evals.swarm-proposals" and decl["n_items"] == 6 and decl["subset"] == "tier=masked"
    assert "proposal-only" in decl["twin"]["config"]["approval_policy"]
    # every call used the app's token; every case was mapped to the run, seeded, then sent the ticket
    assert {c[0] for c in fd.calls} == {"team_app"}
    rows = run.rows
    assert [r["change_text"] for r in rows if r["incident_id"] == "inc-test-0"] == PROPOSALS
    # step k starts from commissioning, the fault and the earlier proposals (no-ops as "")
    s2 = next(r for r in rows if r["incident_id"] == "inc-test-0" and r["step"] == 2)
    assert s2["start_action_log"] == ["cell=1;d_power_db=+1.0;d_tilt_deg=+0.0", "cell=0;d_power_db=+0.0;d_tilt_deg=+5.0",
                                      PROPOSALS[0], ""]
    assert fd.seeded[s2["case_id"]] == s2["start_action_log"]
    assert all(ddb.approvals()[f"t3-{r['case_id']}"]["run_id"] == "t3.swarm-live-test" for r in rows)
    res = out["result"]
    assert res["passed"] and res["n_items"] == 6 and res["value"] == 1.0
    assert len(ev.records[("t3.swarm-live-test", "item")]) == 6


def test_a_failed_seed_is_recorded_as_a_failed_case_not_a_proposal():
    bad = ls.case_id_for("t3.swarm-live-test", 1, 0)
    run, out, *_ = live(fail=[bad])
    row = next(r for r in run.rows if r["case_id"] == bad)
    assert row["ok"] is False and "seed" in row["problem"] and row["change_text"] == ""
    assert not out["result"]["passed"]


class StubTwin:
    def __init__(self):
        self.action_log = []

    def apply(self, change):
        self.action_log.append({"step": len(self.action_log), "change": change})


def test_recorded_client_replays_the_live_proposals_through_the_real_operator():
    run, out, *_ = live()
    client = RecordedSwarmClient([json.loads(line) for line in out["body"].decode().splitlines()])
    assert client.model_id == ls.DEFAULT_MODEL
    for inc in incidents():
        op = SwarmOperator(client, run_id="t3.evals.swarm-masked-test", mode="proposal", proposal_steps=3)
        twin = StubTwin()
        for change in [list(inc["commissioning"]), list(inc["fault"])]:   # incidents.script, fault_step 2
            twin.apply(change)
        op.begin(inc, {"twin": twin, "run_id": "t3.evals.swarm-masked-test"})
        history, got = [], []
        for _ in range(3):
            change = nonzero(op.act({}, list(history)))
            twin.apply(change)
            got.append(canon(change) if change else "")
            history.append({"change": change})
        assert got == PROPOSALS


def test_recorded_client_refuses_a_harness_that_diverged():
    run, out, *_ = live()
    client = RecordedSwarmClient([json.loads(line) for line in out["body"].decode().splitlines()])
    inc = incidents()[0]
    with pytest.raises(ValueError, match="earlier proposals differ"):
        client.open_case({"incident_id": inc["incident_id"], "step": 1,
                          "start_action_log": [inc["commissioning"], inc["fault"],
                                               [{"cell": 3, "d_power_db": 1.0, "d_tilt_deg": 0.0}]]})
    with pytest.raises(KeyError):
        client.open_case({"incident_id": "inc-unknown", "step": 0, "start_action_log": []})
    assert step_text([]) == "" and step_text("cell=0;d_power_db=+0.0;d_tilt_deg=+0.0") == ""
