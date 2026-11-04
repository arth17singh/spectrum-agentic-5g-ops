"""Live Fix, first do no harm: the per-cell verdict next to the network grade, side effects before approval (hold and
"ask for an alternative"), and the after-change correction. The kit, approvals table and twin-side helper are fakes."""
from __future__ import annotations

import json

import pytest

from spectrum.slices.asknoc.kit import Outcome
from spectrum.slices.livefix import suite
from spectrum.slices.livefix.service import LiveFix
from slices.livefix.test_livefix import INC, FakeTable, FakeWriter

changes = suite.repo_import("t3.evals.changes")
BOOST = changes.canon(changes.nonzero("cell=0;d_power_db=-1.0;d_tilt_deg=+0.0|cell=2;d_power_db=+8.0;d_tilt_deg=+0.0"))
TARGET = changes.canon(changes.nonzero("cell=0;d_power_db=+6.0;d_tilt_deg=+0.0"))
FAULT_WORDS = ("power amplifier", "pa_fault", "tilt drift", "-6", "4161", "lf-")


@pytest.fixture(autouse=True)
def _no_git(monkeypatch):
    monkeypatch.setattr("spectrum.slices.livefix.evidence.code_sha", lambda: "a" * 40)


def kp(marker: str) -> dict:
    bad = marker == "faulted"
    return {"coverage_ratio": 0.90 if bad else 0.95, "sla_ratio": 0.90 if bad else 0.96,
            "sinr_p10_db": 0.7 if bad else 2.5, "sinr_mean_db": 9.0,
            "per_cell": [{"cell": i, "coverage_share": 1.0, "sla_share": 1.0, "marker": marker} for i in range(4)]}


class SeqKit:
    """The lead proposes the next change of `proposals` after every message that is not "Approved."; "Approved."
    applies the approved change. Network KPIs: healthy once BOOST or TARGET is applied (the marker tells the
    fake twin-side helper which one)."""

    def __init__(self, table: FakeTable, proposals: list[str]):
        self.t, self.proposals = table, list(proposals)
        self.calls: list[tuple[str, dict]] = []
        self.messages: list[str] = []
        self.logs: dict[str, list] = {}
        self.n = 0

    def call(self, tool, args, timeout):
        self.calls.append((tool, args))
        case = args["case_id"]
        if tool == "ran___seed_case":
            steps = json.loads(args["start_action_log_json"])
            self.logs[case] = [{"change": changes.nonzero(s)} for s in steps]
            faulted = any("d_power_db=-6.0" in s for s in steps)
            m = "healthy" if not faulted else "target" if TARGET in steps else "boost" if BOOST in steps else "faulted"
            return Outcome("ok", 200, {"kpis": kp(m), "seeded": True})
        if tool == "front___continue_case":
            msg = args["message"]
            self.messages.append(msg)
            if msg == "Approved.":
                for aid, row in self.t.rows.items():
                    if aid.startswith("chg-") and row.get("case_id") == case and row["status"] == "APPROVED":
                        row["status"] = "APPLIED"
                        self.logs[case].append({"change": changes.nonzero(json.loads(row["change"])["change"])})
            elif self.proposals:
                self.n += 1
                aid = f"chg-{self.n:012d}"
                self.t.rows[aid] = {"pk": aid, "status": "PENDING", "case_id": case,
                                    "change": json.dumps({"change": self.proposals.pop(0), "item_id": f"step-{self.n}"})}
            self.t.rows[f"t3-{case}"]["doc"] = json.dumps({"action_log": self.logs[case]})
            return Outcome("ok", 200, {"ok": True}, gateway_date="2026-11-04T06:00:00.000Z")
        if tool == "front___case_status":
            waiting = [{"approval_id": a, "change": r["change"]} for a, r in self.t.rows.items()
                       if a.startswith("chg-") and r.get("case_id") == case and r["status"] != "APPLIED"]
            return Outcome("ok", 200, {"state": "TASK_STATE_COMPLETED", "reply": "done", "waiting_approvals": waiting})
        raise AssertionError(tool)


class PercellTwinSide:
    """Stands in for t3/evals/livefix_worker.py: percell-v1 says BOOST leaves cells 0 and 3 degraded, either newly
    degraded by the change (collateral, `collateral=True`) or merely not restored (run 9695's real shape); TARGET
    recovers every cell. Side effects: BOOST is predicted to turn cells 0 and 3 red when `predict_harm`."""
    available = True

    def __init__(self, predict_harm: bool = False, collateral: bool = True):
        self.predict_harm, self.collateral = predict_harm, collateral
        self.calls: list[tuple[str, dict]] = []

    def call(self, op, **a):
        self.calls.append((op, a))
        if op == "percell":
            if (a["cells_after"] or [{}])[0].get("marker") == "boost":
                hit = self.collateral
                cells = [{"cell": c, "health": "amber" if c in (0, 3) else "green", "worse_than_before": hit and c in (0, 3),
                          "fault_cell": c == 2, "reasons": ["vs healthy: coverage_share down 0.05"] if c in (0, 3) else []}
                         for c in range(4)]
                return {"version": "percell-v1", "recovered": False, "cells": cells, "degraded": [0, 3],
                        "collateral": [0, 3] if hit else [], "overshoot": {"power_above_healthy_db": 14.1}}
            return {"version": "percell-v1", "recovered": True, "degraded": [], "collateral": [], "overshoot": {},
                    "cells": [{"cell": c, "health": "green", "worse_than_before": False, "reasons": []} for c in range(4)]}
        if op == "side_effects":
            harm = self.predict_harm and a["proposal"] == BOOST
            cells = [{"cell": c, "without": {"coverage_share": 1.0}, "with": {"coverage_share": 0.7 if harm else 1.0},
                      "health_without": "green", "health_with": "red" if harm and c in (0, 3) else "green",
                      "worse": harm and c in (0, 3),
                      "reasons": ["vs healthy: sinr_p10_db down 3.2"] if harm and c in (0, 3) else []} for c in range(4)]
            return {"version": "percell-v1", "cells": cells, "recovered_with": not harm, "grade_step": 5}
        if op == "score":
            return {"confidence": 0.9, "tier": "one-person", "preflight_ok": True, "explain": "e", "breakdown": {}}
        if op == "health":
            return [{"cell": r["cell"], "health": "green", "why": []} for r in a["kpis"]]
        if op == "lessons":
            return {"block": "", "best": 0.0, "hits": [], "n_lessons": 0}
        if op == "learn":
            return {"lesson": {"lesson_id": "L-9", "status": "verified"}, "verification": {}}
        raise AssertionError(op)


def make(proposals, *, predict_harm=False, alternatives=0, reviews=(), deny_on_hold=False, collateral=True):
    table = FakeTable()
    kit = SeqKit(table, proposals)
    writer = FakeWriter()
    box: dict = {"alt": alternatives, "reviews": list(reviews)}

    def sleep(_s):  # the presenter and the person act while the service waits
        svc, run = box["svc"], box.get("run")
        if run is None:
            return
        if run.state == "review" and box["reviews"] and not run.review_choice:
            svc.review(run.run_id, box["reviews"].pop(0))
            return
        aid = (run.approval or {}).get("approval_id")
        row = table.rows.get(aid or "")
        if run.state != "awaiting_approval" or not row or row["status"] != "PENDING":
            return
        if (run.tier or {}).get("tier") == "hold":
            if box["alt"] and not run.alternative_requested:
                box["alt"] -= 1
                svc.alternative(run.run_id)
            elif deny_on_hold and not box["alt"]:
                row["status"] = "REJECTED"
            return
        row["status"] = "APPROVED"

    svc = LiveFix(kit=kit, ddb=table, writer_factory=lambda: writer, sleep=sleep, spawn=lambda fn: fn(), poll_s=0,
                  approval_timeout_s=5, twinside=PercellTwinSide(predict_harm, collateral))
    box["svc"] = svc

    def go():
        run = svc.start(INC, "live")
        box["run"] = run
        svc.inject(run.run_id)
        return run
    return svc, kit, table, writer, go


def item_extra(writer):
    return next(r for k, r in writer.recs if k == "item")["extra"]


def test_network_recovered_but_cells_degraded_is_not_recovered():
    svc, kit, table, writer, go = make([BOOST], reviews=["finish"])
    run = go()
    v = svc.view(run)
    assert v["state"] == "not_recovered", v["problem"]
    assert v["grade"]["success"] is True                      # the network rule alone would say recovered
    assert v["percell"]["recovered"] is False and v["percell"]["collateral"] == [0, 3]
    last = v["events"][-1]["text"]
    assert "per cell NOT recovered: cells 0, 3 newly degraded by the change (collateral)" in last
    assert "not restored" not in last
    assert {c["cell"]: c["status"] for c in v["cells"]} == {0: "amber", 1: "green", 2: "green", 3: "amber"}
    assert [c["cell"] for c in v["cells"] if c.get("collateral")] == [0, 3]
    decl = writer.recs[0][1]
    assert "percell" in decl["grader"] and decl["target"]["rule"].startswith("success = network grade AND per-cell")
    res = writer.recs[-1][1]
    assert res["passed"] is False and res["details"]["network_success"] is True
    assert res["details"]["percell"]["collateral"] == [0, 3] and res["details"]["percell"]["declared"] is True
    assert item_extra(writer)["percell"]["recovered"] is False
    assert kit.messages.count("Approved.") == 1 and table.updates == 0


def test_cells_still_degraded_by_the_fault_are_not_restored_not_collateral():
    # The real shape of run t3.livefix-20261104T073206Z-9695: cells 0 and 3 degraded, collateral empty.
    svc, kit, table, writer, go = make([BOOST], reviews=["finish"], collateral=False)
    run = go()
    assert run.state == "not_recovered"
    last = run.events[-1]["text"]
    assert "per cell NOT recovered: cells 0, 3 not restored." in last and "collateral" not in last
    assert not any(c.get("collateral") for c in svc.view(run)["cells"])
    assert writer.recs[-1][1]["details"]["percell"]["collateral"] == []


def test_correction_goes_back_through_the_hold_and_recovers_every_cell():
    svc, kit, table, writer, go = make([BOOST, TARGET], reviews=["correct"])
    run = go()
    assert run.state == "recovered", run.problem
    assert [f["kind"] for f in run.followups] == ["correction"] and run.followups[0]["cells"] == [0, 3]
    text = run.followups[0]["text"]
    assert "cell 0" in text and "cell 3" in text and "coverage_share down 0.05" in text
    assert not any(w in text for w in FAULT_WORDS)                  # symptoms only
    assert text in kit.messages and kit.messages.count("Approved.") == 2  # a person approved the correction too
    assert run.percell["recovered"] is True and run.applied == ["chg-000000000001", "chg-000000000002"]
    extra = item_extra(writer)
    assert [f["kind"] for f in extra["followups"]] == ["correction"] and "text_sha256" in extra["followups"][0]
    assert len(extra["side_effects_shown"]) == 2
    assert writer.recs[-1][1]["passed"] is True


def test_predicted_red_holds_the_change_and_an_alternative_replaces_it():
    svc, kit, table, writer, go = make([BOOST, TARGET], predict_harm=True, alternatives=1)
    run = go()
    assert run.state == "recovered", run.problem
    held = run.shown[0]
    assert held["approval_id"] == "chg-000000000001" and held["tier"] == "hold" and held["red"] == [0, 3]
    assert run.superseded == ["chg-000000000001"]
    assert table.rows["chg-000000000001"]["status"] == "PENDING"     # the app never decides: the held row is untouched
    assert run.applied == ["chg-000000000002"] and kit.messages.count("Approved.") == 1
    alt = run.followups[0]
    assert alt["kind"] == "alternative" and alt["approval_id"] == "chg-000000000001" and alt["cells"] == [0, 3]
    assert "on hold" in alt["text"] and "cell 0 red, made worse by the change" in alt["text"]
    assert "worse than doing nothing" in alt["text"] and "cell 1" not in alt["text"]  # only the harmed cells
    assert not any(w in alt["text"] for w in FAULT_WORDS)
    assert run.shown[1]["tier"] == "one_person"
    assert any(e.get("kind") == "hold" for e in run.events) and any(e.get("kind") == "side_effects" for e in run.events)
    v = svc.view(run)
    assert v["tier"]["tier"] == "one_person" and v["superseded"] == ["chg-000000000001"]


def test_no_alternative_keeps_the_held_change_on_hold_until_a_person_denies_it():
    svc, kit, table, writer, go = make([BOOST], predict_harm=True, alternatives=1, deny_on_hold=True)
    run = go()
    assert run.state == "rejected"
    assert run.superseded == [] and run.applied == []
    assert any("proposed no alternative" in e["text"] for e in run.events)
    assert "Approved." not in kit.messages


def test_alternative_and_review_are_refused_out_of_turn():
    svc, kit, table, writer, go = make([TARGET])
    run = go()
    assert run.state == "recovered"
    with pytest.raises(ValueError):
        svc.alternative(run.run_id)
    with pytest.raises(ValueError):
        svc.review(run.run_id, "correct")
    with pytest.raises(ValueError):
        svc.review(run.run_id, "approve")


def test_side_effects_use_the_preflight_copy_and_the_current_start_log():
    svc, kit, table, writer, go = make([TARGET])
    run = go()
    op, a = next(c for c in svc.twinside.calls if c[0] == "side_effects")
    assert a["proposal"] == TARGET and a["incident"]["incident_id"] == INC
    assert a["start_log"] == suite.steps_for(run.incident, svc.targets, True)  # healthy history + the fault, no fix yet
    pc = next(c for c in svc.twinside.calls if c[0] == "percell")[1]
    assert pc["fault_cells"] == run.incident["fault_cells"] and pc["cells_before"][0]["marker"] == "faulted"


def test_without_the_helper_the_run_is_declared_on_the_network_rule():
    from slices.livefix.test_livefix import make as make_plain

    svc, kit, table, writer = make_plain(person="APPROVED")
    run = svc.start(INC, "live")
    svc.inject(run.run_id)
    assert run.state == "recovered" and run.percell is None and run.percell_declared is False
    assert "percell" not in writer.recs[0][1]["grader"]
    assert writer.recs[-1][1]["details"]["percell"]["declared"] is False


def test_routes_for_alternative_and_review():
    from fastapi.testclient import TestClient

    from spectrum.main import app
    from spectrum.slices.livefix.routes import set_service

    svc, kit, table, writer, go = make([TARGET])
    run = go()
    set_service(svc)
    try:
        c = TestClient(app, client=("127.0.0.1", 50000), base_url="http://127.0.0.1:8787")
        h = {"Origin": "http://127.0.0.1:8787", "X-Spectrum-Client": "livefix"}
        assert c.post(f"/api/livefix/runs/{run.run_id}/alternative", json={}, headers=h).status_code == 409
        assert c.post(f"/api/livefix/runs/{run.run_id}/review", json={"choice": "correct"}, headers=h).status_code == 409
        assert c.post(f"/api/livefix/runs/{run.run_id}/review", json={"choice": "approve"}, headers=h).status_code == 422
        assert c.post(f"/api/livefix/runs/{run.run_id}/alternative", json={}).status_code == 403  # no Origin / header
    finally:
        set_service(None)
