"""Live Fix: one incident, end to end, on our in-process Maveric twin behind the AgentCore Gateway.

    healthy  ran___seed_case (the app's token; Cedar refuses it to every agent): the twin at the fault step with
             no fault, so every cell shows its healthy level
    faulted  ran___seed_case with the fault: the cells the fault hits go red
    diagnosing  front___continue_case sends noc_lead a symptoms-only ticket (t3/evals/swarm.task_text, neutral
             label); front___case_status is polled. The agents observe, preflight and propose through the Gateway
    awaiting_approval  the lead's ran___apply_change was refused by the approval rule (the change waits in the
             kit's approvals table, PENDING). A PERSON decides on the Approvals path (PIN) or with
             the kit's CLI. This module never writes a decision; it only reads the row
    applying the person approved: the lead is told "Approved." on the same case (what the Approvals page tells a
             person to do) and calls ran___apply_change again; the kit marks the row APPLIED
    recovered / not_recovered  the twin is replayed with the case's applied action log (another ran___seed_case),
             next to the healthy and unfixed twins at the same step, and graded with the suite's own rule
             (t3/evals/graders.grade_kpis, targets.json tolerance)
    rejected / no_proposal / failed / timed_out  end states that say why

Evidence (tools/evlib): declared before the first Gateway call, one item per run, one result. The run id maps
every case of the run (t3-<case> rows), so the Lambda files its change record under the same run id.

Mode "recorded": no AWS at all. The judged live swarm run's recorded proposal and reply for a masked test
incident, with the KPIs its scoring run measured; the approval is a "continue" click, not a decision.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from . import suite
from .gateway import CONTINUE, SEED, STATUS, GatewayFailed, ok_or_raise, parse_waiting

UNIT = "t3.livefix"
SOURCE = "dashboard/backend/spectrum/slices/livefix"
RUNNING = ("TASK_STATE_SUBMITTED", "TASK_STATE_WORKING")
DEFAULT_MODEL = "noc_lead=openai.gpt-oss-120b-1:0;students=openai.gpt-oss-20b-1:0"
TARGET = {"metric": "success", "op": ">=", "value": 1.0, "unit": "ratio"}
GRADER = ("t3/evals/graders.py:grade_kpis (twin replay of the case's applied action log, against the healthy "
          "and unfixed twins at the same step; t3/evals/targets.json tolerance)")
MAX_CHANGES = 3
END = {"recovered", "not_recovered", "rejected", "no_proposal", "failed", "timed_out"}
APPROVED_TEXT = "Approved."


def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


@dataclass
class Run:
    run_id: str
    mode: str
    incident: dict
    n: int
    state: str = "preparing"
    created: float = 0.0
    events: list = field(default_factory=list)
    stages: dict = field(default_factory=dict)       # stage -> {"start": t, "end": t}
    cases: dict = field(default_factory=dict)        # role -> case id
    kpis: dict = field(default_factory=dict)         # healthy, faulted, final, healthy_final, unfixed_final
    approval: dict | None = None
    applied: list = field(default_factory=list)      # approval ids applied
    reply: str | None = None
    grade: dict | None = None
    problem: str | None = None
    evidence: dict = field(default_factory=dict)
    recorded: dict | None = None
    inject_requested: bool = False
    continue_requested: bool = False
    finishing: bool = False


class LiveFix:
    def __init__(self, *, kit=None, ddb=None, writer_factory: Callable[[], Any] | None = None,
                 table: str = "team13-approvals", model_id: str | None = None, clock=time.time, sleep=time.sleep,
                 poll_s: float = 4.0, lead_timeout_s: float = 420.0, approval_timeout_s: float = 900.0,
                 recorded_loader: Callable[[], list] = suite.recorded_incidents, step_delay_s: float = 1.5,
                 spawn: Callable[[Callable[[], None]], None] | None = None):
        self.kit, self.ddb, self.writer_factory = kit, ddb, writer_factory
        self.table = table
        self.model_id = model_id or os.environ.get("T3_SWARM_MODEL_ID", DEFAULT_MODEL)
        self.clock, self.sleep, self.poll_s = clock, sleep, poll_s
        self.lead_timeout_s, self.approval_timeout_s = lead_timeout_s, approval_timeout_s
        self.recorded_loader, self.step_delay_s = recorded_loader, step_delay_s
        self.spawn = spawn or (lambda fn: threading.Thread(target=fn, daemon=True).start())
        self.runs: dict[str, Run] = {}
        self.lock = threading.Lock()
        self.targets = suite.load_targets()
        self._incidents = {i["incident_id"]: i for i in suite.dev_incidents()}
        self._counter = 0
        self._writer = None

    # ------------------------------------------------------------------ public
    def incidents(self) -> dict:
        rec = self._recorded()
        return {"live": [suite.label(i) for i in self._incidents.values()],
                "recorded": [{**r["label"], "success": r["success"]} for r in rec],
                "live_source": "t3/evals/suite/incidents-dev.jsonl (dev split, visible tier; the scored test set is "
                               "never used live)",
                "recorded_source": f"{suite.RECORDED_LIVE_RUN} proposals, scored by {suite.RECORDED_SCORE_RUN}"}

    def active(self) -> Run | None:
        return next((r for r in self.runs.values() if r.state not in END or r.finishing), None)

    def start(self, incident_id: str, mode: str) -> Run:
        with self.lock:
            busy = self.active()
            if busy is not None:
                raise ValueError(f"run {busy.run_id} is still in progress ({busy.state})")
            if mode == "live":
                inc = self._incidents.get(incident_id)
                if inc is None:
                    raise KeyError(incident_id)
                if self.kit is None or self.ddb is None:
                    raise RuntimeError("live mode needs the kit and AWS (not configured or offline)")
                recorded = None
            elif mode == "recorded":
                recorded = next((r for r in self._recorded() if r["incident_id"] == incident_id), None)
                if recorded is None:
                    raise KeyError(incident_id)
                inc = {"incident_id": incident_id, "fault_cells": recorded["fault_cells"]}
            else:
                raise ValueError("mode must be live or recorded")
            self._counter += 1
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            run_id = f"t3.livefix-{stamp}-{secrets.token_hex(2)}" if mode == "live" else f"recorded-{stamp}"
            run = Run(run_id=run_id, mode=mode, incident=inc, n=self._counter, created=self.clock(),
                      recorded=recorded)
            self.runs[run_id] = run
        self.spawn(lambda: self._guarded(run, self._prepare_live if mode == "live" else self._prepare_recorded))
        return run

    def inject(self, run_id: str) -> Run:
        run = self._get(run_id)
        with self.lock:
            if run.state != "healthy" or run.inject_requested:
                raise ValueError(f"can't inject now (state {run.state})")
            run.inject_requested = True
        self.spawn(lambda: self._guarded(run, self._fix_live if run.mode == "live" else self._fix_recorded))
        return run

    def recorded_continue(self, run_id: str) -> Run:
        run = self._get(run_id)
        if run.mode != "recorded" or run.state != "awaiting_approval":
            raise ValueError("only a recorded run waiting at the gate can continue")
        run.continue_requested = True
        return run

    def view(self, run: Run) -> dict:
        healthy = run.kpis.get("healthy")
        now_k = run.kpis.get("final") or run.kpis.get("faulted") or healthy
        ref = run.kpis.get("healthy_final") if run.kpis.get("final") else healthy
        return {
            "run_id": run.run_id, "mode": run.mode, "label": "LIVE" if run.mode == "live" else "RECORDED",
            "state": run.state, "end": run.state in END and not run.finishing, "incident": suite.label(run.incident)
            if run.mode == "live" else (run.recorded or {}).get("label"),
            "events": run.events[-200:], "stages": self._stage_view(run), "cases": run.cases,
            "headline": {k: suite.headline(v) for k, v in run.kpis.items()},
            "cells": suite.cell_view(now_k, ref) if run.mode == "live" else self._recorded_cells(run),
            "cells_healthy": suite.cell_view(healthy, healthy) if run.mode == "live" else None,
            "approval": run.approval, "applied": run.applied, "reply": run.reply, "grade": run.grade,
            "problem": run.problem, "evidence": run.evidence,
            "evidence_link": f"/evidence?run={run.run_id}" if run.mode == "live" else
            f"/evidence?run={(run.recorded or {}).get('live_run', '')}",
            "twin": "our in-process Maveric twin (ran-rl 1.2.9) in the team's RAN-tools Lambda, behind the AgentCore "
                    "Gateway; not the organisers' hosted twin",
            "models": self.model_id,
        }

    # ------------------------------------------------------------------ helpers
    def _get(self, run_id: str) -> Run:
        run = self.runs.get(run_id)
        if run is None:
            raise KeyError(run_id)
        return run

    def _recorded(self) -> list:
        try:
            return self.recorded_loader()
        except (OSError, ValueError):
            return []

    def _event(self, run: Run, stage: str, text: str, **extra) -> None:
        run.events.append({"t": utc(), "stage": stage, "text": text, **extra})

    def _stage(self, run: Run, stage: str) -> None:
        now = self.clock()
        for s in run.stages.values():
            s.setdefault("end", now)
        run.stages[stage] = {"start": now}
        run.state = stage

    def _stage_view(self, run: Run) -> list:
        now = self.clock()
        return [{"stage": k, "seconds": round((v.get("end", now) - v["start"]), 1), "open": "end" not in v}
                for k, v in run.stages.items()]

    def _guarded(self, run: Run, fn) -> None:
        try:
            fn(run)
        except GatewayFailed as e:
            self._fail(run, f"The Gateway answered {e.out.kind} to {e.tool}"
                       + (f" ({e.out.category})" if e.out.category else "") + ".")
        except Exception as e:  # noqa: BLE001 - every failure ends the run with a reason on screen
            self._fail(run, f"{type(e).__name__}: {str(e)[:300]}")

    def _fail(self, run: Run, why: str) -> None:
        run.problem = why
        run.finishing = True
        self._event(run, "failed", why)
        self._stage(run, "failed")
        run.stages["failed"]["end"] = self.clock()
        try:
            if run.mode == "live":
                self._write_result(run, False)
        finally:
            run.finishing = False

    # ------------------------------------------------------------------ gateway, table
    def _call(self, tool: str, args: dict, timeout: float = 90.0):
        return ok_or_raise(tool, self.kit.call(tool, args, timeout))

    def _new_case(self, run: Run, role: str) -> str:
        case_id = f"case-{secrets.token_hex(4)}"
        self.ddb.put_item(TableName=self.table, ConditionExpression="attribute_not_exists(pk)", Item={
            "pk": {"S": f"t3-{case_id}"}, "case_id": {"S": case_id}, "run_id": {"S": run.run_id},
            "unit": {"S": UNIT}, "role": {"S": role}, "created_at": {"N": str(int(self.clock()))},
            "source": {"S": SOURCE}})
        run.cases[role] = case_id
        return case_id

    def _seed(self, run: Run, role: str, steps: list[str]) -> dict:
        case_id = self._new_case(run, role)
        inc = run.incident
        out = self._call(SEED, {"case_id": case_id, "scenario": inc["scenario"], "seed": int(inc["seed"]),
                                "preset": inc["preset"], "start_action_log_json": json.dumps(steps)}, timeout=120)
        return out.data.get("kpis") or {}

    def _row(self, pk: str) -> dict | None:
        item = self.ddb.get_item(TableName=self.table, Key={"pk": {"S": pk}}, ConsistentRead=True).get("Item")
        if not item:
            return None
        return {k: next(iter(v.values())) for k, v in item.items()}

    def _status(self, case_id: str) -> dict:
        out = self._call(STATUS, {"case_id": case_id}, timeout=60)
        return out.data

    def _wait_lead(self, run: Run, case_id: str) -> dict:
        deadline = self.clock() + self.lead_timeout_s
        status: dict = {}
        while True:
            self.sleep(self.poll_s)
            status = self._status(case_id)
            if status.get("state") not in RUNNING:
                break
            if self.clock() >= deadline:
                status["timed_out"] = True
                break
        run.reply = str(status.get("reply") or "")[:1500] or run.reply
        return status

    # ------------------------------------------------------------------ evidence
    def _w(self):
        if self._writer is None:
            self._writer = self.writer_factory() if self.writer_factory else None
        return self._writer

    def _write(self, run: Run, kind: str, rec: dict) -> None:
        w = self._w()
        if w is None:
            return
        try:
            where = getattr(w, kind)(rec)
            run.evidence[kind] = where
        except Exception as e:  # noqa: BLE001 - the run continues; the screen says the record failed
            run.evidence[f"{kind}_error"] = f"{type(e).__name__}: {str(e)[:200]}"

    def _declare(self, run: Run) -> None:
        from .evidence import code_sha
        inc = run.incident
        self._write(run, "declared", {
            "run_id": run.run_id, "unit": UNIT, "track": "track-3", "target": dict(TARGET), "n_items": 1,
            "test_set_sha256": suite.dev_digest(), "model_id": self.model_id, "grader": GRADER,
            "code_sha": code_sha(), "region": os.environ.get("AWS_REGION") or "us-east-2",
            "subset": f"dev split, visible tier, incident {inc['incident_id']}",
            "approval_policy": "a person approves every change (kit approvals table); the app never decides",
            "twin": {"kind": "maveric-inprocess", "version": "ran-rl-1.2.9", "scenario": inc["scenario"],
                     "seed": int(inc["seed"]), "preset": inc["preset"],
                     "host": "team13 RAN-tools Lambda behind the AgentCore Gateway"}})

    def _write_result(self, run: Run, passed: bool) -> None:
        if "result" in run.evidence:
            return
        from .evidence import code_sha
        g = run.grade or {}
        self._write(run, "item", {
            "run_id": run.run_id, "item_id": run.incident["incident_id"],
            "input_sha256": sha(json.dumps({"incident_id": run.incident["incident_id"], "cases": run.cases},
                                           sort_keys=True)),
            "output_sha256": sha(json.dumps({"applied": run.applied, "final": suite.headline(run.kpis.get("final"))},
                                            sort_keys=True)),
            "score": 1.0 if passed else 0.0, "passed": passed, "grader": GRADER, "backend": "agentcore-gateway",
            "aws_time": run.evidence.get("_gw_date"), "aws_time_source": "gateway:http-date"
            if run.evidence.get("_gw_date") else None,
            "extra": {"state": run.state, "cases": run.cases, "applied": run.applied, "grade": g,
                      "stages": self._stage_view(run), "problem": run.problem,
                      "approval_id": (run.approval or {}).get("approval_id")}})
        self._write(run, "result", {
            "run_id": run.run_id, "unit": UNIT, "metric": TARGET["metric"], "value": 1.0 if passed else 0.0,
            "target": dict(TARGET), "passed": passed, "n_items": 1,
            "checker": {"script": SOURCE + "/service.py", "code_sha": code_sha()},
            "details": {"state": run.state, "recovery": g.get("recovery"), "applied": run.applied}})

    # ------------------------------------------------------------------ live
    def _prepare_live(self, run: Run) -> None:
        inc = run.incident
        self._declare(run)
        if "declared_error" in run.evidence:
            raise RuntimeError("the declaration could not be written, so the run did not start: "
                               + run.evidence["declared_error"])
        self._event(run, "preparing", "Declared the run before any call: target, grader, twin and incident.")
        fs = self.targets["episode"]["fault_step"]
        run.kpis["healthy"] = self._seed(run, "healthy", suite.steps_for(inc, self.targets, False))
        self._stage(run, "healthy")
        self._event(run, "healthy", f"Twin at step {fs} with no fault: every cell at its healthy level.")

    def _fix_live(self, run: Run) -> None:
        inc = run.incident
        self._stage(run, "faulted")
        run.kpis["faulted"] = self._seed(run, "faulted", suite.steps_for(inc, self.targets, True))
        self._event(run, "faulted", "Fault injected on the twin. The agents get symptoms only, never the cell.")
        case = run.cases["faulted"]
        self._stage(run, "diagnosing")
        sent = self._call(CONTINUE, {"case_id": case, "message": suite.ticket(inc, run.run_id, run.n)}, timeout=60)
        run.evidence["_gw_date"] = sent.gateway_date
        self._event(run, "diagnosing", "Ticket sent to noc_lead through the Gateway (front___continue_case).")
        status = self._wait_lead(run, case)
        while True:
            pending = self._pending(case, status)
            if pending is None:
                if status.get("timed_out"):
                    self._end(run, "timed_out", "The lead did not finish in time.")
                elif not run.applied:
                    self._end(run, "no_proposal", "The lead finished without proposing a change.")
                else:
                    self._grade(run)
                return
            run.approval = pending
            self._stage(run, "awaiting_approval")
            self._event(run, "awaiting_approval", f"Cedar held ran___apply_change: {pending['approval_id']} waits for "
                        f"a person.", approval_id=pending["approval_id"], change=pending["change_text"])
            decision = self._wait_person(pending["approval_id"])
            if decision != "APPROVED":
                self._end(run, "rejected" if decision == "REJECTED" else "timed_out",
                          "The person rejected the change." if decision == "REJECTED"
                          else "No decision before the approval timeout.")
                return
            run.approval = {**pending, "status": "APPROVED"}
            self._event(run, "applying", f"A person approved {pending['approval_id']}. Telling the lead.")
            self._stage(run, "applying")
            self._call(CONTINUE, {"case_id": case, "message": APPROVED_TEXT}, timeout=60)
            status = self._wait_lead(run, case)
            row = self._row(pending["approval_id"]) or {}
            if row.get("status") != "APPLIED":
                self._end(run, "failed", f"The approved change was not applied (row {row.get('status')}).")
                return
            run.applied.append(pending["approval_id"])
            run.approval = {**pending, "status": "APPLIED"}
            self._event(run, "applying", f"Applied on the twin: {pending['change_text']}.")
            if self._grade(run, final=False) or len(run.applied) >= MAX_CHANGES:
                self._grade(run)
                return

    def _pending(self, case_id: str, status: dict) -> dict | None:
        for w in parse_waiting(status):
            aid = w.get("approval_id")
            if not aid:
                continue
            row = self._row(aid)
            # PENDING, or already decided by a person faster than our poll (then _wait_person sees it at once).
            # An APPLIED row is a change the lead already made: not a new proposal.
            if not row or row.get("status") not in ("PENDING", "APPROVED", "REJECTED"):
                continue
            raw = row.get("change") if isinstance(row.get("change"), str) else ""
            return {"approval_id": aid, "change_text": w.get("change_text"), "item_id": w.get("item_id"),
                    "status": "PENDING", "change_sha256": sha(raw), "reason": (w.get("reason") or "")[:600],
                    "approve_link": f"/approvals?case={case_id}"}
        return None

    def _wait_person(self, approval_id: str) -> str:
        deadline = self.clock() + self.approval_timeout_s
        while self.clock() < deadline:
            row = self._row(approval_id) or {}
            if row.get("status") in ("APPROVED", "REJECTED", "APPLIED"):
                return "APPROVED" if row["status"] == "APPLIED" else row["status"]
            self.sleep(2.0)
        return "TIMEOUT"

    def _grade(self, run: Run, final: bool = True) -> bool:
        """Replay the faulted case's applied log on a fresh twin and grade it. Returns success."""
        row = self._row(f"t3-{run.cases['faulted']}") or {}
        try:
            doc = json.loads(row.get("doc") or "{}")
        except ValueError:
            doc = {}
        changes = suite.repo_import("t3.evals.changes")
        log = [changes.canon(changes.nonzero(e.get("change"))) if changes.nonzero(e.get("change")) else ""
               for e in doc.get("action_log") or []]
        n = len(log)
        if not log:
            raise RuntimeError("the case's action log could not be read from the approvals table")
        inc = run.incident
        run.kpis["final"] = self._seed(run, f"verify{len(run.applied)}", log)
        run.kpis["healthy_final"] = self._seed(run, f"healthy{n}", suite.steps_for(inc, self.targets, False, n))
        run.kpis["unfixed_final"] = self._seed(run, f"unfixed{n}", suite.steps_for(inc, self.targets, True, n))
        hl = {k: suite.headline(run.kpis[k]) for k in ("final", "healthy_final", "unfixed_final")}
        run.grade = {**suite.grade(hl["final"], hl["healthy_final"], hl["unfixed_final"], self.targets), "step": n}
        ok = bool(run.grade["success"])
        if final:
            self._end(run, "recovered" if ok else "not_recovered",
                      "Every graded KPI is back within tolerance of the healthy twin." if ok
                      else "Some graded KPI is still outside tolerance of the healthy twin.")
        return ok

    def _end(self, run: Run, state: str, why: str) -> None:
        run.finishing = True  # "end" is reported only once the evidence is written
        self._stage(run, state)
        run.stages[state]["end"] = self.clock()
        self._event(run, state, why)
        try:
            if run.mode == "live":
                self._write_result(run, state == "recovered")
        finally:
            run.finishing = False

    # ------------------------------------------------------------------ recorded
    def _prepare_recorded(self, run: Run) -> None:
        r = run.recorded or {}
        run.kpis["healthy"] = r.get("healthy") or {}
        self._stage(run, "healthy")
        self._event(run, "healthy", "RECORDED: the healthy network for this incident, as the scoring run measured it.")

    def _fix_recorded(self, run: Run) -> None:
        r = run.recorded or {}
        self._stage(run, "faulted")
        run.kpis["faulted"] = r.get("unfixed") or {}
        self._event(run, "faulted", "RECORDED: the fault as frozen in the test suite.")
        self.sleep(self.step_delay_s)
        self._stage(run, "diagnosing")
        self._event(run, "diagnosing", f"RECORDED: the ticket noc_lead got in {r.get('live_run')}.")
        self.sleep(self.step_delay_s * 2)
        step = next((s for s in r.get("steps") or [] if s.get("change_text")), None)
        if step is None:
            self._end(run, "no_proposal", "RECORDED: the lead proposed no change for this incident.")
            return
        run.reply = step.get("reply_excerpt")
        run.approval = {"approval_id": step.get("approval_id"), "change_text": step.get("change_text"),
                        "status": "PENDING (recorded)", "recorded": True}
        self._stage(run, "awaiting_approval")
        self._event(run, "awaiting_approval", "RECORDED: the change stopped at the approval gate in the judged run "
                    "(proposal-only: nobody approved it then).")
        while not run.continue_requested:
            self.sleep(0.5)
        self._stage(run, "applying")
        self._event(run, "applying", "RECORDED: the scoring run replayed the proposals on the twin.")
        self.sleep(self.step_delay_s)
        run.kpis["final"] = r.get("final") or {}
        run.kpis["healthy_final"] = r.get("healthy") or {}
        run.grade = {"success": r.get("success"), "recovery": r.get("recovery")}
        self._end(run, "recovered" if r.get("success") else "not_recovered",
                  f"RECORDED: scored by {r.get('score_run')}.")

    def _recorded_cells(self, run: Run) -> list:
        faulty = set(run.incident.get("fault_cells") or [])
        hit = run.state not in ("preparing", "healthy")
        fixed = run.state == "recovered"
        return [{"cell": c, "status": "green" if (fixed or not hit or c not in faulty) else "red"} for c in range(4)]
