"""Live Fix wiring: fault buttons, lessons in the ticket, confidence after the proposal, learning after a verified
recovery, attacks through the Gateway. The twin-side helper and the Gateway are fakes."""
from __future__ import annotations

import pytest

from spectrum.slices.livefix import suite
from spectrum.slices.livefix.service import LiveFix
from slices.livefix.test_livefix import INC, FakeKit, FakeTable, FakeWriter, make


@pytest.fixture(autouse=True)
def _no_git(monkeypatch):
    monkeypatch.setattr("spectrum.slices.livefix.evidence.code_sha", lambda: "a" * 40)


class FakeTwinSide:
    """Stands in for t3/evals/livefix_worker.py."""
    available = True

    def __init__(self):
        self.calls: list[str] = []

    def call(self, op, **a):
        self.calls.append(op)
        if op == "make_incident":
            inc = dict(next(i for i in suite.dev_incidents() if i["incident_id"] == INC))
            return {**inc, "incident_id": f"lf-{a['fault_key']}-c{a['cell']}-off-vis-7002", "fault_key": a["fault_key"],
                    "fault_label": "Power amplifier fault", "chosen_cell": a["cell"], "fault_about": "x"}
        if op == "health":
            return [{"cell": r["cell"], "health": "red" if r.get("coverage_share", 1) < 0.9 else "green", "why": []}
                    for r in a["kpis"]]
        if op == "lessons":
            return {"block": "Verified past lessons: L-01 raise power", "best": 0.8, "n_lessons": 1,
                    "hits": [{"lesson_id": "L-01", "similarity": 0.8, "pattern_text": "raise power"}]}
        if op == "score":
            return {"confidence": 0.82, "tier": "two-person", "preflight_ok": True, "explain": "e",
                    "breakdown": {"federated": {"signal": 0.9, "deltas": {}, "model_id": "m", "version": "v"}}}
        if op == "learn":
            return {"lesson": {"lesson_id": "L-02", "status": "verified"}, "verification": {"recovered": True}}
        raise AssertionError(op)


def _approve_pending(table):
    def sleep(_s):
        for k, r in table.rows.items():
            if k.startswith("chg-") and r["status"] == "PENDING":
                r["status"] = "APPROVED"
    return sleep


def test_fault_button_run_wires_lessons_confidence_and_learning():
    table = FakeTable()
    kit = FakeKit(table)
    ts = FakeTwinSide()
    svc = LiveFix(kit=kit, ddb=table, writer_factory=FakeWriter, sleep=_approve_pending(table),
                  spawn=lambda fn: fn(), poll_s=0, twinside=ts)
    run = svc.start(mode="live", fault_key="pa_fault", cell=2)
    assert run.incident["incident_id"].startswith("lf-pa_fault-c2")
    svc.inject(run.run_id)
    assert run.state == "recovered", run.problem
    ticket = next(a["message"] for t, a in kit.calls if t == "front___continue_case")
    head = ticket.split("\n\n")[0]
    assert "lf-" not in ticket and "pa_fault" not in ticket and "power" not in head and "cell" not in head
    assert "Verified past lessons" in ticket
    v = svc.view(run)
    assert v["confidence"]["tier"] == "two_person" and v["confidence"]["calibrated"] is False
    assert "MISSED" in v["confidence"]["label"]
    assert any(e.get("kind") == "federated" for e in v["events"])
    assert v["lesson_learned"]["status"] == "verified" and ts.calls.count("learn") == 1
    assert {c["health"] for c in v["cells"]} <= {"good", "degraded", "critical", "unknown"}


def test_unknown_fault_key_refused():
    svc, *_ = make()
    svc.twinside = FakeTwinSide()
    with pytest.raises(KeyError):
        svc.start(mode="live", fault_key="meteor", cell=1)


def test_fault_button_without_helper_says_so():
    svc, *_ = make()
    with pytest.raises(RuntimeError, match="LIVEFIX_T3_PYTHON"):
        svc.start(mode="live", fault_key="pa_fault", cell=1)


def test_attack_goes_through_the_gateway_and_is_labelled():
    svc, kit, table, writer = make(person=None)
    gw = suite.repo_import("t3.evals.attacks").FakeGateway(decision="DENY", text="denied by team13_ran_x")
    svc.attack_gateway = lambda: gw
    run = svc.start(INC, "live")
    svc.attack(run.run_id, "rogue_agent")
    a = run.attacks[-1]
    assert a["refused"] is True and a["judged"] is True and gw.calls[0][0] == "ran_monitor"
    assert a["case_id"] == run.cases["healthy"]
