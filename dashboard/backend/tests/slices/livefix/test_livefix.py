"""Live Fix: the state machine against a fake kit, approvals table and evidence writer (no AWS)."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from spectrum.slices.asknoc.kit import Outcome
from spectrum.slices.livefix import suite
from spectrum.slices.livefix.routes import set_service
from spectrum.slices.livefix.service import LiveFix

FIX = "cell=0;d_power_db=+6.0;d_tilt_deg=+0.0"
INC = "inc-vvl-vis-4161-power-6"


def kp(cov, sla, p10, shares):
    return {"coverage_ratio": cov, "sla_ratio": sla, "sinr_p10_db": p10, "sinr_mean_db": 9.0,
            "per_cell": [{"cell": i, "coverage_share": s, "sla_share": s} for i, s in enumerate(shares)]}


HEALTHY = kp(0.95, 0.96, 2.5, [1, 1, 1, 1])
FAULTED = kp(0.90, 0.90, 0.7, [1, 1, 0.6, 0.6])


class FakeTable:
    def __init__(self):
        self.rows: dict[str, dict] = {}
        self.updates = 0

    def put_item(self, TableName, Item, ConditionExpression=None):
        pk = Item["pk"]["S"]
        assert pk.startswith("t3-case-"), "Live Fix only writes case mapping rows"
        assert pk not in self.rows
        self.rows[pk] = {k: next(iter(v.values())) for k, v in Item.items()}

    def get_item(self, TableName, Key, ConsistentRead=True):
        row = self.rows.get(Key["pk"]["S"])
        return {"Item": {k: {"S": str(v)} for k, v in row.items()}} if row else {}

    def update_item(self, **kw):  # a decision: never Live Fix's to make
        self.updates += 1
        raise AssertionError("Live Fix must never write a decision")


class FakeKit:
    """ran___seed_case answers KPIs from the steps; the lead proposes FIX once; "Approved." applies it."""

    def __init__(self, table: FakeTable, propose=True, fixes=True):
        self.t, self.propose, self.fixes = table, propose, fixes
        self.calls: list[tuple[str, dict]] = []
        self.logs: dict[str, list] = {}

    def call(self, tool, args, timeout):
        self.calls.append((tool, args))
        case = args["case_id"]
        if tool == "ran___seed_case":
            steps = json.loads(args["start_action_log_json"])
            faulted = any("d_power_db=-6.0" in s for s in steps)
            fixed = any(s == FIX for s in steps)
            self.logs[case] = [{"change": suite.repo_import("t3.evals.changes").nonzero(s)} for s in steps]
            k = FAULTED if faulted and not (fixed and self.fixes) else HEALTHY
            return Outcome("ok", 200, {"kpis": k, "seeded": True})
        if tool == "front___continue_case":
            if args["message"] == "Approved.":
                row = self.t.rows["chg-000000000001"]
                if row["status"] == "APPROVED":
                    row["status"] = "APPLIED"
                    self.logs[case].append({"change": [{"cell": 0, "d_power_db": 6.0, "d_tilt_deg": 0.0}]})
                    self.t.rows[f"t3-{case}"]["doc"] = json.dumps({"action_log": self.logs[case]})
            elif self.propose:
                self.t.rows["chg-000000000001"] = {"pk": "chg-000000000001", "status": "PENDING", "case_id": case,
                                                   "change": json.dumps({"change": FIX, "item_id": "step-2"})}
            self.t.rows[f"t3-{case}"]["doc"] = json.dumps({"action_log": self.logs[case]})
            return Outcome("ok", 200, {"ok": True}, gateway_date="2026-11-04T06:00:00.000Z")
        if tool == "front___case_status":
            waiting = [{"approval_id": a, "change": r["change"]} for a, r in self.t.rows.items()
                       if a.startswith("chg-") and r["status"] != "APPLIED"]
            return Outcome("ok", 200, {"state": "TASK_STATE_COMPLETED", "reply": "done", "waiting_approvals": waiting})
        raise AssertionError(tool)


class FakeWriter:
    def __init__(self):
        self.recs: list[tuple[str, dict]] = []

    def declared(self, rec):
        self.recs.append(("declared", rec))
        return ["mem://declared"]

    def item(self, rec):
        self.recs.append(("item", rec))
        return ["mem://item"]

    def result(self, rec):
        self.recs.append(("result", rec))
        return ["mem://result"]


def make(person=None, **kit_kw):
    table = FakeTable()
    kit = FakeKit(table, **kit_kw)
    writer = FakeWriter()

    def sleep(_s):  # the only place time passes: a "person" may decide while the service waits
        if person:
            for aid, row in table.rows.items():
                if aid.startswith("chg-") and row["status"] == "PENDING":
                    row["status"] = person

    svc = LiveFix(kit=kit, ddb=table, writer_factory=lambda: writer, sleep=sleep, spawn=lambda fn: fn(),
                  poll_s=0, approval_timeout_s=5)
    return svc, kit, table, writer


@pytest.fixture(autouse=True)
def _no_git(monkeypatch):
    monkeypatch.setattr("spectrum.slices.livefix.evidence.code_sha", lambda: "a" * 40)


def test_live_loop_recovers_only_after_a_person_approves():
    svc, kit, table, writer = make(person="APPROVED")
    run = svc.start(INC, "live")
    assert run.state == "healthy"
    assert [c["status"] for c in svc.view(run)["cells"]] == ["green"] * 4
    assert writer.recs[0][0] == "declared" and kit.calls[0][0] == "ran___seed_case"  # declared before any call
    svc.inject(run.run_id)
    v = svc.view(run)
    assert v["state"] == "recovered", v["problem"]
    assert v["grade"]["success"] is True
    assert v["applied"] == ["chg-000000000001"]
    tools = [c[0] for c in kit.calls]
    assert "front___continue_case" in tools and table.updates == 0
    ticket = next(a["message"] for t, a in kit.calls if t == "front___continue_case")
    assert "power" not in ticket and "4161" not in ticket and "cell" not in ticket  # symptoms only
    assert [k for k, _ in writer.recs] == ["declared", "item", "result"]
    assert writer.recs[-1][1]["passed"] is True
    for pk, row in table.rows.items():
        if pk.startswith("t3-"):
            assert row["run_id"] == run.run_id and row["unit"] == "t3.livefix"


def test_rejected_change_is_never_applied():
    svc, kit, table, writer = make(person="REJECTED")
    run = svc.start(INC, "live")
    svc.inject(run.run_id)
    assert run.state == "rejected"
    assert not any(a.get("message") == "Approved." for t, a in kit.calls if t == "front___continue_case")
    assert writer.recs[-1][1]["passed"] is False


def test_no_decision_times_out_and_nothing_is_applied():
    svc, kit, table, writer = make(person=None)
    run = svc.start(INC, "live")
    svc.clock = iter(range(0, 10_000, 10)).__next__  # time moves 10 s per read
    svc.inject(run.run_id)
    assert run.state == "timed_out" and run.applied == []


def test_no_proposal():
    svc, *_ = make(person="APPROVED", propose=False)
    run = svc.start(INC, "live")
    svc.inject(run.run_id)
    assert run.state == "no_proposal"


def test_not_recovered_when_the_fix_does_not_close_the_gap():
    svc, *_ = make(person="APPROVED", fixes=False)
    run = svc.start(INC, "live")
    svc.inject(run.run_id)
    assert run.state == "not_recovered"


def test_only_dev_visible_incidents_run_live():
    svc, *_ = make()
    test_ids = {json.loads(x)["incident_id"] for x in suite.TEST_PATH.read_text().splitlines() if x.strip()}
    live = {i["incident_id"] for i in svc.incidents()["live"]}
    assert live and not (live & test_ids)
    with pytest.raises(KeyError):
        svc.start(next(iter(test_ids)), "live")


def test_one_run_at_a_time():
    svc, *_ = make()
    svc.start(INC, "live")
    with pytest.raises(ValueError):
        svc.start(INC, "live")


def test_recorded_mode_needs_no_aws_and_waits_for_continue():
    rec = {"incident_id": "inc-x", "fault_cells": [0], "label": {"incident_id": "inc-x", "fault": "cell 0"},
           "steps": [{"step": 0, "change_text": FIX, "approval_id": "chg-1", "reply_excerpt": "r"}],
           "healthy": {"coverage_ratio": 0.9, "sla_ratio": 0.9, "sinr_p10_db": 1.0},
           "unfixed": {"coverage_ratio": 0.8, "sla_ratio": 0.8, "sinr_p10_db": 0.0},
           "final": {"coverage_ratio": 0.9, "sla_ratio": 0.9, "sinr_p10_db": 1.0}, "success": True, "recovery": 1.0,
           "live_run": "t3.swarm-live-x", "score_run": "t3.evals.x"}
    jobs = []
    svc = LiveFix(recorded_loader=lambda: [rec], sleep=lambda s: None, spawn=jobs.append, step_delay_s=0)
    run = svc.start("inc-x", "recorded")
    jobs.pop()()
    assert run.state == "healthy" and svc.view(run)["label"] == "RECORDED"
    svc.inject(run.run_id)
    svc.continue_requested = True
    run.continue_requested = True
    jobs.pop()()
    assert run.state == "recovered" and svc.view(run)["evidence_link"] == "/evidence?run=t3.swarm-live-x"


def test_live_refused_without_kit():
    svc = LiveFix()
    with pytest.raises(RuntimeError):
        svc.start(INC, "live")


def test_routes_guard_and_read():
    from spectrum.main import app

    set_service(LiveFix(recorded_loader=lambda: []))
    try:
        c = TestClient(app, client=("127.0.0.1", 50000), base_url="http://127.0.0.1:8787")
        r = c.get("/api/livefix/incidents")
        assert r.status_code == 200 and r.json()["live_ready"] is False and r.json()["live"]
        body = {"incident_id": INC, "mode": "live"}
        assert c.post("/api/livefix/runs", json=body).status_code == 403  # no Origin / client header
        h = {"Origin": "http://127.0.0.1:8787", "X-Spectrum-Client": "livefix"}
        r = c.post("/api/livefix/runs", json=body, headers=h)
        assert r.status_code == 503  # live needs the kit
        assert c.get("/api/livefix/runs/t3.livefix-20261104T000000Z-abcd").status_code == 404
        c2 = TestClient(app, client=("127.0.0.1", 50000), base_url="http://evil.example:8787")
        assert c2.get("/api/livefix/incidents").status_code == 403
    finally:
        set_service(None)
