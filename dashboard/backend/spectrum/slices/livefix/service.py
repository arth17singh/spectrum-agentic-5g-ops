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
             next to the healthy and unfixed twins at the same step, and graded twice: the suite's network rule
             (t3/evals/graders.grade_kpis, targets.json tolerance) AND the per-cell rule (t3/evals/percell.py
             percell-v1: every cell back to its own healthy level; a cell the fix made worse is collateral).
             "recovered" needs both; network-only recovery with degraded cells is not_recovered (collateral)
    review   the per-cell check failed after a change and steps remain: the presenter asks the agents to correct
             (a follow-up with per-cell symptoms, back through Cedar's hold and a person's approval) or finishes
    rejected / no_proposal / failed / timed_out  end states that say why

Side effects before approval: every held change is tried on the PREFLIGHT copy of the twin (t3/evals/confidence.py,
seed + PREFLIGHT_SEED_OFFSET: not the grader's twin) and each cell is shown without -> with the change, graded by
percell-v1. A cell the change makes red puts it on hold (Approve disabled on the card; "Ask for an alternative"
sends the lead the predicted harm, symptoms only); a cell it makes amber asks for two people.

Evidence (tools/evlib): declared before the first Gateway call, one item per run, one result. The run id maps
every case of the run (t3-<case> rows), so the Lambda files its change record under the same run id. The item
carries the per-cell verdict, the side effects each approver was shown and every follow-up sent to the lead.

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
from .twinside import CALIBRATION, FAULT_KEYS, HEALTH, TEMPLATE, TIER, TwinSide, Unavailable

UNIT = "t3.livefix"
SOURCE = "dashboard/backend/spectrum/slices/livefix"
RUNNING = ("TASK_STATE_SUBMITTED", "TASK_STATE_WORKING")
DEFAULT_MODEL = "noc_lead=openai.gpt-oss-120b-1:0;students=openai.gpt-oss-20b-1:0"
TARGET = {"metric": "success", "op": ">=", "value": 1.0, "unit": "ratio"}
GRADER = ("t3/evals/graders.py:grade_kpis (twin replay of the case's applied action log, against the healthy "
          "and unfixed twins at the same step; t3/evals/targets.json tolerance)")
PERCELL_GRADER = ("t3/evals/percell.py:percell_verdict percell-v1 (every cell of the replayed twin against the same "
                  "cell of the healthy twin at the same step: recovered = every cell green; and against the network "
                  "right before the fix: a non-faulted cell the fix made worse is collateral)")
TARGET_RULE = "success = network grade AND per-cell verdict (percell-v1)"
MAX_CHANGES = 3
MAX_FOLLOWUPS = 3       # alternatives + corrections per run
END = {"recovered", "not_recovered", "rejected", "no_proposal", "failed", "timed_out"}
APPROVED_TEXT = "Approved."
TIER_ORDER = {"one_person": 0, "two_person": 1, "hold": 2}


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
    health: dict = field(default_factory=dict)       # kpis key -> faults.cell_health rows (when the helper runs)
    cells_at_proposal: list | None = None
    confidence: dict | None = None
    lessons: dict | None = None
    lesson_learned: dict | None = None
    attacks: list = field(default_factory=list)
    applied_changes: list = field(default_factory=list)  # canonical change texts, in apply order
    percell_declared: bool = False                     # the declaration named the per-cell grader
    percell: dict | None = None                        # percell-v1 verdict after the latest applied change
    side_effects: dict | None = None                   # predicted per-cell effects of the held change
    tier: dict | None = None                           # effective approval tier: confidence + side effects
    shown: list = field(default_factory=list)          # side effects each approver was shown (evidence)
    followups: list = field(default_factory=list)      # alternatives and corrections sent to the lead
    superseded: list = field(default_factory=list)     # held approvals replaced by an alternative
    alternative_requested: bool = False
    review_choice: str | None = None                   # "correct" | "finish" (the presenter, in state review)


class LiveFix:
    def __init__(self, *, kit=None, ddb=None, writer_factory: Callable[[], Any] | None = None,
                 table: str = "team13-approvals", model_id: str | None = None, clock=time.time, sleep=time.sleep,
                 poll_s: float = 4.0, lead_timeout_s: float = 420.0, approval_timeout_s: float = 900.0,
                 recorded_loader: Callable[[], list] = suite.recorded_incidents, step_delay_s: float = 1.5,
                 spawn: Callable[[Callable[[], None]], None] | None = None, twinside: Any = None,
                 attack_gateway: Callable[[], Any] | None = None):
        self.kit, self.ddb, self.writer_factory = kit, ddb, writer_factory
        self.twinside = twinside if twinside is not None else TwinSide(python="")
        self.attack_gateway = attack_gateway
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

    def start(self, incident_id: str | None = None, mode: str = "live", *, fault_key: str | None = None,
              cell: int | None = None, preset: str = "official") -> Run:
        """A fault button (fault_key = the UI's FaultKey, cell 0-3; the twin-side helper builds a dev-split incident)
        or a dev incident by id. Recorded mode takes a recorded incident id."""
        made = None
        if mode == "live" and fault_key is not None:
            if fault_key not in FAULT_KEYS or cell not in (0, 1, 2, 3):
                raise KeyError(f"{fault_key} on cell {cell}")
            if self.kit is None or self.ddb is None:
                raise RuntimeError("live mode needs the kit and AWS (not configured or offline)")
            if self.active() is not None:
                raise ValueError(f"run {self.active().run_id} is still in progress")
            try:
                made = self.twinside.call("make_incident", fault_key=FAULT_KEYS[fault_key], cell=int(cell),
                                          preset=preset)
            except Unavailable as e:
                raise RuntimeError(str(e)) from None
        with self.lock:
            busy = self.active()
            if busy is not None:
                raise ValueError(f"run {busy.run_id} is still in progress ({busy.state})")
            if mode == "live":
                inc = made or self._incidents.get(incident_id or "")
                if inc is None:
                    raise KeyError(incident_id)
                if self.kit is None or self.ddb is None:
                    raise RuntimeError("live mode needs the kit and AWS (not configured or offline)")
                recorded = None
            elif mode == "recorded":
                recorded = next((r for r in self._recorded() if r["incident_id"] == incident_id), None) \
                    if fault_key is None else self._recorded_for(fault_key)
                if recorded is None:
                    raise KeyError(incident_id)
                inc = {"incident_id": recorded["incident_id"], "fault_cells": recorded["fault_cells"]}
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

    def alternative(self, run_id: str) -> Run:
        """The held change is on hold (predicted harm or low confidence): ask the lead for another one. The held row
        stays PENDING in the approvals table (this app never decides); the run waits on the new proposal instead."""
        run = self._get(run_id)
        with self.lock:
            if run.mode != "live" or run.state != "awaiting_approval" or not run.approval:
                raise ValueError(f"no held change to replace (state {run.state})")
            if (run.tier or {}).get("tier") != "hold":
                raise ValueError("only a change on hold can be sent back for an alternative")
            if len(run.followups) >= MAX_FOLLOWUPS:
                raise ValueError(f"at most {MAX_FOLLOWUPS} follow-ups per run")
            run.alternative_requested = True
        return run

    def review(self, run_id: str, choice: str) -> Run:
        """After a change whose per-cell check failed: "correct" sends the lead the degraded cells (the correction
        goes through Cedar's hold and a person's approval again); "finish" ends the run as graded."""
        run = self._get(run_id)
        if choice not in ("correct", "finish"):
            raise ValueError("choice must be correct or finish")
        with self.lock:
            if run.state != "review":
                raise ValueError(f"nothing to review (state {run.state})")
            if choice == "correct" and len(run.followups) >= MAX_FOLLOWUPS:
                raise ValueError(f"at most {MAX_FOLLOWUPS} follow-ups per run")
            run.review_choice = choice
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
            "cells": self._cells(run, now_k, ref) if run.mode == "live" else self._recorded_cells(run),
            "cells_at_proposal": run.cells_at_proposal,
            "fault": {k: run.incident.get(k) for k in ("fault_key", "fault_label", "fault_about", "chosen_cell",
                                                       "fault_cells", "preset")} if run.mode == "live" else None,
            "approval": run.approval, "applied": run.applied, "reply": run.reply, "grade": run.grade,
            "confidence": run.confidence, "lessons": run.lessons, "lesson_learned": run.lesson_learned,
            "attacks": run.attacks,
            "percell": run.percell, "percell_declared": run.percell_declared, "side_effects": run.side_effects,
            "tier": run.tier, "followups": [{k: f.get(k) for k in ("kind", "t", "approval_id", "cells", "text")}
                                            for f in run.followups],
            "superseded": run.superseded,
            "can_follow_up": run.mode == "live" and len(run.followups) < MAX_FOLLOWUPS
            and len(run.applied) < MAX_CHANGES,
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

    def _recorded_for(self, fault_key: str) -> dict | None:
        """A recorded incident with the same fault template (a recovered one first)."""
        tmpl = TEMPLATE.get(fault_key)
        rows = [r for r in self._recorded() if tmpl and r["incident_id"].endswith("-" + tmpl)]
        return next((r for r in rows if r.get("success")), rows[0] if rows else None)

    def _health(self, run: Run, key: str, ref_key: str | None) -> None:
        """faults.cell_health of run.kpis[key] against run.kpis[ref_key], via the twin-side helper (if running)."""
        if not self.twinside.available:
            return
        try:
            cur = (run.kpis.get(key) or {}).get("per_cell") or []
            ref = (run.kpis.get(ref_key) or {}).get("per_cell") if ref_key else None
            run.health[key] = self.twinside.call("health", kpis=cur, reference=ref,
                                                 fault=run.incident.get("fault") or [])
        except Exception:  # noqa: BLE001 - the map falls back to the slice's own colouring
            run.health.pop(key, None)

    def _cells(self, run: Run, now_k: dict | None, ref: dict | None) -> list:
        rows = suite.cell_view(now_k, ref)
        key = "final" if run.kpis.get("final") else "faulted" if run.kpis.get("faulted") else "healthy"
        h = {r["cell"]: r for r in run.health.get(key) or []}
        pc = {c["cell"]: c for c in (run.percell or {}).get("cells") or []} if key == "final" else {}
        for r in rows:
            if r["cell"] in pc:  # after a fix the map shows the per-cell verdict (t3/evals/percell.py)
                r["status"] = pc[r["cell"]].get("health") or r["status"]
                r["why"] = pc[r["cell"]].get("reasons")
                r["collateral"] = bool(pc[r["cell"]].get("worse_than_before"))
            elif r["cell"] in h:  # before a fix: t3.evals.faults.cell_health (the same thresholds)
                r["status"] = h[r["cell"]]["health"]
                r["why"] = h[r["cell"]].get("why")
            r["health"] = HEALTH.get(r["status"], "unknown")
        return rows

    def attack(self, run_id: str, key: str, variant: str = "judged") -> Run:
        """One attack call through the Gateway on this run's case (t3.evals.attacks.run_attack). Expected: refused."""
        run = self._get(run_id)
        if run.mode != "live":
            raise ValueError("attacks run on live runs only")
        case = run.cases.get("faulted") or run.cases.get("healthy")
        if not case:
            raise ValueError("load the network first")
        if self.attack_gateway is None:
            raise RuntimeError("no Gateway for attacks (kit not configured)")

        def go():
            attacks = suite.repo_import("t3.evals.attacks")
            try:
                out = attacks.run_attack(key, self.attack_gateway(), case, run_id=run.run_id, variant=variant,
                                         cell=int(run.incident.get("chosen_cell") or 0))
            except Exception as e:  # noqa: BLE001
                out = {"attack": key, "decision": "ERROR", "refused": False, "text_excerpt": f"{type(e).__name__}",
                       "dry": False}
            out["text_excerpt"] = (self.redact or (lambda s: s))(str(out.get("text_excerpt") or ""))
            out["case_id"] = case
            run.attacks.append(out)
            self._event(run, "attack", f"Attack {out.get('label') or key} ({'judged case ' + out['suite_id'] if out.get('judged') else 'NOT a judged case'}): "
                        f"{out.get('decision')}" + (f" by {out.get('policy')}" if out.get("policy") else "") + ".")
        self.spawn(go)
        return run

    redact: Callable[[str], str] | None = None

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
        # The per-cell grader runs in the twin-side helper; a run without it is declared on the network rule alone.
        run.percell_declared = bool(self.twinside.available)
        target = {**TARGET, "rule": TARGET_RULE} if run.percell_declared else \
            {**TARGET, "rule": "success = network grade (the per-cell grader needs the twin-side helper, not running)"}
        grader = f"{GRADER} AND {PERCELL_GRADER}" if run.percell_declared else GRADER
        self._write(run, "declared", {
            "run_id": run.run_id, "unit": UNIT, "track": "track-3", "target": target, "n_items": 1,
            "test_set_sha256": suite.dev_digest(), "model_id": self.model_id, "grader": grader,
            "code_sha": code_sha(), "region": os.environ.get("AWS_REGION") or "us-east-2",
            "subset": f"dev split, visible tier, incident {inc['incident_id']}",
            "approval_policy": "a person approves every change (kit approvals table); the app never decides; a change "
                               "the preflight copy predicts to turn a cell red is on hold (Approve disabled on the "
                               "Live Fix card), one it predicts to turn a cell amber asks for two people",
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
                      "approval_id": (run.approval or {}).get("approval_id"),
                      "percell_declared": run.percell_declared, "percell": run.percell,
                      "side_effects_shown": run.shown,
                      "followups": [{k: f.get(k) for k in ("kind", "t", "approval_id", "cells", "text_sha256")}
                                    for f in run.followups],
                      "superseded": run.superseded}})
        pc = run.percell or {}
        self._write(run, "result", {
            "run_id": run.run_id, "unit": UNIT, "metric": TARGET["metric"], "value": 1.0 if passed else 0.0,
            "target": {**TARGET, "rule": TARGET_RULE if run.percell_declared else "network grade"},
            "passed": passed, "n_items": 1,
            "checker": {"script": SOURCE + "/service.py", "code_sha": code_sha()},
            "details": {"state": run.state, "recovery": g.get("recovery"), "applied": run.applied,
                        "network_success": g.get("success"),
                        "percell": {"declared": run.percell_declared, "available": bool(run.percell),
                                    "recovered": pc.get("recovered"), "degraded": pc.get("degraded"),
                                    "collateral": pc.get("collateral"), "version": pc.get("version")},
                        "followups": len(run.followups)}})

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
        self._health(run, "healthy", "healthy")
        self._stage(run, "healthy")
        self._event(run, "healthy", f"Twin at step {fs} with no fault: every cell at its healthy level.")

    def _fix_live(self, run: Run) -> None:
        inc = run.incident
        self._stage(run, "faulted")
        run.kpis["faulted"] = self._seed(run, "faulted", suite.steps_for(inc, self.targets, True))
        self._health(run, "faulted", "healthy")
        self._event(run, "faulted", "Fault injected on the twin. The agents get symptoms only, never the cell.")
        case = run.cases["faulted"]
        self._stage(run, "diagnosing")
        text = suite.ticket(inc, run.run_id)
        run.lessons = self._lessons(inc)
        if run.lessons and run.lessons.get("block"):
            text += "\n\n" + run.lessons["block"]
            self._event(run, "diagnosing", f"{len(run.lessons['hits'])} verified lesson(s) added to the ticket "
                        f"(best similarity {run.lessons['best']:.2f}).")
        sent = self._call(CONTINUE, {"case_id": case, "message": text}, timeout=60)
        run.evidence["_gw_date"] = sent.gateway_date
        run.evidence["ticket_sha256"] = sha(text)
        self._event(run, "diagnosing", "Ticket sent to noc_lead through the Gateway (front___continue_case): "
                    "symptoms only, under an opaque incident label.")
        status = self._wait_lead(run, case)
        held_before: dict | None = None  # the held change an alternative was asked for
        while True:
            pending = self._pending(case, status, skip=run.superseded)
            if pending is None and held_before is not None:
                # No alternative came back: the held change is still the one on the table (still on hold).
                run.superseded.remove(held_before["approval_id"])
                pending = held_before
                self._event(run, "awaiting_approval", f"The lead proposed no alternative. {pending['approval_id']} "
                            "is still on hold: deny it, or let the approval time out.")
            held_before = None
            if pending is None:
                if status.get("timed_out"):
                    self._end(run, "timed_out", "The lead did not finish in time.")
                elif not run.applied:
                    self._end(run, "no_proposal", "The lead finished without proposing a change.")
                else:
                    self._finish_graded(run, self._success(run, bool((run.grade or {}).get("success"))))
                return
            run.approval = pending
            run.cells_at_proposal = self._cells(run, run.kpis.get("final") or run.kpis.get("faulted"),
                                                run.kpis.get("healthy_final") or run.kpis.get("healthy"))
            run.side_effects = self._side_effects(run, pending.get("change_text"))
            run.confidence = self._confidence(run, pending.get("change_text"))
            run.tier = self._tier(run)
            run.shown.append(self._shown(run, pending))
            self._stage(run, "awaiting_approval")
            self._event(run, "awaiting_approval", f"Cedar held ran___apply_change: {pending['approval_id']} waits for "
                        f"a person.", approval_id=pending["approval_id"], change=pending["change_text"])
            if run.tier["tier"] == "hold":
                self._event(run, "awaiting_approval", "On hold: " + "; ".join(run.tier["why"]) + ". Approve is off on "
                            "this card; deny it, or ask the lead for an alternative.", kind="hold")
            decision = self._wait_person(pending["approval_id"], run)
            if decision == "ALTERNATIVE":
                self._follow_up(run, case, "alternative", self._alternative_text(run, pending),
                                approval_id=pending["approval_id"],
                                cells=[c["cell"] for c in (run.side_effects or {}).get("cells") or [] if c.get("worse")])
                run.superseded.append(pending["approval_id"])
                held_before = pending
                self._stage(run, "diagnosing")
                status = self._wait_lead(run, case)
                continue
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
            run.applied_changes.append(pending.get("change_text") or "")
            run.approval = {**pending, "status": "APPLIED"}
            self._event(run, "applying", f"Applied on the twin: {pending['change_text']}.")
            ok_net = self._grade(run, final=False)
            ok = self._success(run, ok_net)
            if ok:
                self._learn(run)
                self._finish_graded(run, True)
                return
            if len(run.applied) >= MAX_CHANGES:
                self._finish_graded(run, False)
                return
            if run.percell is None:  # no per-cell verdict: the network rule alone, as before
                if ok_net:
                    self._finish_graded(run, True)
                    return
                continue  # the lead may already have proposed the next change
            # The per-cell check failed and changes remain: the presenter asks for a correction or finishes.
            if len(run.followups) >= MAX_FOLLOWUPS:
                self._finish_graded(run, False)
                return
            self._stage(run, "review")
            self._event(run, "review", "Per-cell check: " + self._percell_text(run) + " Ask the agents to correct, "
                        "or finish the run as graded.", kind="review")
            if self._wait_review(run) != "correct":
                self._finish_graded(run, False)
                return
            bad = [c["cell"] for c in run.percell.get("cells") or []
                   if c.get("health") != "green" or c.get("worse_than_before")]
            self._follow_up(run, case, "correction", self._correction_text(run), cells=bad)
            self._stage(run, "diagnosing")
            status = self._wait_lead(run, case)

    def _pending(self, case_id: str, status: dict, skip: list | tuple = ()) -> dict | None:
        for w in parse_waiting(status):
            aid = w.get("approval_id")
            if not aid or aid in skip:
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

    def _wait_person(self, approval_id: str, run: Run | None = None) -> str:
        deadline = self.clock() + self.approval_timeout_s
        while self.clock() < deadline:
            row = self._row(approval_id) or {}
            if row.get("status") in ("APPROVED", "REJECTED", "APPLIED"):
                return "APPROVED" if row["status"] == "APPLIED" else row["status"]
            if run is not None and run.alternative_requested:
                run.alternative_requested = False
                return "ALTERNATIVE"
            self.sleep(2.0)
        return "TIMEOUT"

    def _wait_review(self, run: Run) -> str:
        """The presenter's choice after a failed per-cell check; no choice before the timeout finishes the run."""
        deadline = self.clock() + self.approval_timeout_s
        while self.clock() < deadline:
            if run.review_choice:
                choice, run.review_choice = run.review_choice, None
                return choice
            self.sleep(1.0)
        return "finish"

    def _follow_up(self, run: Run, case: str, kind: str, text: str, **extra) -> None:
        """A follow-up to noc_lead on the same case (symptoms only): an alternative to a held change, or a
        correction after a change that left cells degraded. Whatever it proposes stops at Cedar's hold again."""
        self._call(CONTINUE, {"case_id": case, "message": text}, timeout=60)
        run.followups.append({"kind": kind, "t": utc(), "text": text, "text_sha256": sha(text), **extra})
        what = "An alternative" if kind == "alternative" else "A correction"
        self._event(run, "diagnosing", f"{what} was asked of noc_lead on the same case: {text}", kind="followup")

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
        self._health(run, "final", "healthy_final")
        run.percell = self._percell(run)
        ok = bool(run.grade["success"])
        if final:
            self._finish_graded(run, self._success(run, ok))
        return ok

    def _success(self, run: Run, ok_net: bool) -> bool:
        """The declared rule: network grade AND per-cell verdict. A run declared without the per-cell grader, or
        whose per-cell verdict could not be computed, is graded on the network rule (the record says which)."""
        if not run.percell_declared or run.percell is None:
            return ok_net
        return ok_net and bool(run.percell.get("recovered"))

    def _percell(self, run: Run) -> dict | None:
        """percell-v1 on the replayed twin: each cell against the same cell of the healthy twin at the same step,
        and against the network right before the fix (the faulted state, as percell.grade_incident uses): a cell
        the fix made worse, other than the faulted cells, is collateral. Computed by the twin-side helper."""
        if not run.percell_declared or not self.twinside.available:
            return None
        k = run.kpis
        try:
            return self.twinside.call("percell", cells_after=(k.get("final") or {}).get("per_cell") or [],
                                      cells_healthy=(k.get("healthy_final") or {}).get("per_cell") or [],
                                      cells_before=(k.get("faulted") or {}).get("per_cell") or None,
                                      fault_cells=run.incident.get("fault_cells"),
                                      network_after=suite.headline(k.get("final")),
                                      network_healthy=suite.headline(k.get("healthy_final")))
        except Exception as e:  # noqa: BLE001 - the run is then graded on the network rule, and says so
            self._event(run, "applying", f"Per-cell verdict not computed ({type(e).__name__}); this change is "
                        "graded on the network rule only.")
            return None

    @staticmethod
    def _cell_bits(cells: list, health_key: str, worse_key: str) -> list[str]:
        out = []
        for c in cells:
            h, worse = c.get(health_key), c.get(worse_key)
            if h in (None, "green") and not worse:
                continue
            why = ", ".join(str(r) for r in (c.get("reasons") or [])[:3]) or "below its healthy level"
            out.append(f"cell {c['cell']} {h}{' (made worse)' if worse else ''}: {why}")
        return out

    def _percell_text(self, run: Run) -> str:
        pc = run.percell or {}
        bits = self._cell_bits(pc.get("cells") or [], "health", "worse_than_before")
        net = "Network KPIs recovered" if (run.grade or {}).get("success") else "Network KPIs not recovered"
        if not bits:
            return f"{net}; every cell is back to its healthy level."
        coll = pc.get("collateral") or []
        return (f"{net}, but per cell NOT recovered: " + "; ".join(bits)
                + (f" (collateral: cell{'s' if len(coll) > 1 else ''} {', '.join(str(c) for c in coll)})." if coll
                   else "."))

    def _finish_graded(self, run: Run, ok: bool) -> None:
        if run.percell is not None:
            why = ("Every graded KPI is back within tolerance of the healthy twin, and every cell is back to its "
                   "healthy level." if ok else self._percell_text(run))
        else:
            why = ("Every graded KPI is back within tolerance of the healthy twin." if ok
                   else "Some graded KPI is still outside tolerance of the healthy twin.")
            if run.percell_declared and run.grade:
                why += " (Per-cell verdict not available: graded on the network rule.)"
        self._end(run, "recovered" if ok else "not_recovered", why)

    # ------------------------------------------------------------------ side effects, tier, follow-up texts
    def _side_effects(self, run: Run, change_text: str | None) -> dict | None:
        """The held change tried on the PREFLIGHT copy of the twin (not the grader's twin): each cell without -> with
        the change at the grade step, graded by percell-v1. A cell the change makes red holds it; amber asks for two
        people. Computed by the twin-side helper; None if it is not running."""
        if not self.twinside.available or not change_text:
            return None
        try:
            se = self.twinside.call("side_effects", incident=run.incident, start_log=self._start_log(run),
                                    proposal=change_text)
        except Exception as e:  # noqa: BLE001 - the card then says the side effects were not computed
            self._event(run, "awaiting_approval", f"Side effects not computed ({type(e).__name__}).")
            return None
        cells = se.get("cells") or []
        red = [c["cell"] for c in cells if c.get("worse") and c.get("health_with") == "red"]
        amber = [c["cell"] for c in cells if c.get("worse") and c.get("health_with") == "amber"]
        tier = "hold" if red else "two_person" if amber else None
        why = ([f"the preflight copy predicts cell{'s' if len(red) > 1 else ''} {', '.join(map(str, red))} "
                f"turn{'' if len(red) > 1 else 's'} red because of the change"] if red else []) + \
              ([f"cell{'s' if len(amber) > 1 else ''} {', '.join(map(str, amber))} amber because of it"]
               if amber else [])
        se = {**se, "red": red, "amber": amber, "tier": tier, "why": why}
        bits = self._cell_bits(cells, "health_with", "worse")
        self._event(run, "awaiting_approval", "Side effects on the preflight copy (without -> with the change): "
                    + ("; ".join(bits) if bits else "no cell is made worse") + ".", kind="side_effects")
        return se

    def _tier(self, run: Run) -> dict:
        """The approval tier the card shows: the stricter of the confidence tier and the side-effect tier."""
        conf = (run.confidence or {}).get("tier")
        side = (run.side_effects or {}).get("tier")
        tiers = [t for t in (conf, side) if t]
        eff = max(tiers, key=lambda t: TIER_ORDER.get(t, 2)) if tiers else "one_person"
        why = list((run.side_effects or {}).get("why") or [])
        if conf and TIER_ORDER.get(conf, 2) >= TIER_ORDER.get(eff, 2):
            score = (run.confidence or {}).get("score")
            why.append(f"confidence {score:.2f}" if isinstance(score, (int, float)) else "no confidence score")
        return {"tier": eff, "confidence_tier": conf, "side_effect_tier": side, "why": why}

    def _shown(self, run: Run, pending: dict) -> dict:
        """What this approver was shown, for the evidence item."""
        se = run.side_effects or {}
        return {"approval_id": pending.get("approval_id"), "t": utc(), "tier": (run.tier or {}).get("tier"),
                "tier_why": (run.tier or {}).get("why"), "red": se.get("red"), "amber": se.get("amber"),
                "side_effects_sha256": sha(json.dumps(se, sort_keys=True, default=str)) if se else None,
                "cells": [{k: c.get(k) for k in ("cell", "health_without", "health_with", "worse", "reasons")}
                          for c in se.get("cells") or []]}

    def _alternative_text(self, run: Run, pending: dict) -> str:
        bits = self._cell_bits((run.side_effects or {}).get("cells") or [], "health_with", "worse")
        harm = ("A preflight on a copy of the twin predicts it would leave " + "; ".join(bits)
                + ", worse than doing nothing.") if bits else ("It is on hold: " + "; ".join((run.tier or {}).get("why")
                                                                                          or ["low confidence"]) + ".")
        return (f"The change waiting for approval ({pending.get('approval_id')}) is on hold and will not be approved. "
                f"{harm} Do not apply it. Propose a different change that restores service without degrading other "
                "cells, and request it with ran___apply_change as before.")

    def _correction_text(self, run: Run) -> str:
        bits = self._cell_bits((run.percell or {}).get("cells") or [], "health", "worse_than_before")
        net = "are back within tolerance" if (run.grade or {}).get("success") else "are still outside tolerance"
        left = MAX_CHANGES - len(run.applied)
        return (f"After the applied change the network-wide KPIs {net}, but these cells are below their healthy "
                f"level: {'; '.join(bits) or 'see the per-cell observation'}. Observe again, then propose one "
                "corrective change that restores these cells without losing the service-level recovery, and request "
                f"it with ran___apply_change. At most {left} more change{'s' if left != 1 else ''}.")

    # ------------------------------------------------------------------ helper-backed extras (optional)
    def _start_log(self, run: Run) -> list[str]:
        return suite.steps_for(run.incident, self.targets, True) + list(run.applied_changes)

    def _lessons(self, inc: dict) -> dict | None:
        if not self.twinside.available:
            return None
        try:
            return self.twinside.call("lessons", incident=inc, k=3)
        except Exception:  # noqa: BLE001 - no lessons is a valid state
            return None

    def _confidence(self, run: Run, change_text: str | None) -> dict | None:
        """t3.evals.confidence.score_proposal for the held change (the tier only changes how many approvers the
        card asks for; nothing is ever approved automatically)."""
        if not self.twinside.available or not change_text:
            return None
        try:
            lesson = (run.lessons or {}).get("best")
            c = self.twinside.call("score", incident=run.incident, start_log=self._start_log(run),
                                   proposal=change_text, lesson=lesson if lesson else None)
        except Exception as e:  # noqa: BLE001
            self._event(run, "awaiting_approval", f"Confidence not computed ({type(e).__name__}).")
            return None
        out = {"score": c.get("confidence"), "tier": TIER.get(c.get("tier"), "hold"), "explain": c.get("explain"),
               "preflight_ok": c.get("preflight_ok"), "breakdown": c.get("breakdown"), **CALIBRATION}
        fd = (c.get("breakdown") or {}).get("federated")
        if fd:
            d = fd.get("deltas") or {}
            self._event(run, "awaiting_approval",
                        f"Federated forecast ({fd.get('model_id')}@{fd.get('version')}, trained across 3 operator "
                        f"silos without pooling data): coverage {d.get('coverage_ratio', 0):+.3f}, SLA "
                        f"{d.get('sla_ratio', 0):+.3f}, p10 SINR {d.get('sinr_p10_db', 0):+.2f} dB; signal "
                        f"{fd.get('signal', 0):.2f}.", kind="federated", federated=fd)
        if out["score"] is not None:
            self._event(run, "awaiting_approval", f"Confidence {out['score']:.2f}: {out['tier'].replace('_', '-')} "
                        f"approval ({CALIBRATION['label']}).", kind="confidence")
        return out

    def _learn(self, run: Run) -> None:
        """After a verified recovery: replay the applied fixes on the local twin and store a lesson if they recover
        it (the lesson store refuses any test-split incident)."""
        if not self.twinside.available:
            return
        try:
            out = self.twinside.call("learn", incident=run.incident, proposals=list(run.applied_changes))
        except Exception as e:  # noqa: BLE001
            run.lesson_learned = {"status": "error", "reason": f"{type(e).__name__}: {str(e)[:200]}"}
            return
        run.lesson_learned = {**(out.get("lesson") or {}), "verification": out.get("verification")}
        st = run.lesson_learned.get("status")
        self._event(run, "recovered", f"Lesson {run.lesson_learned.get('lesson_id')} stored (twin-verified)."
                    if st == "verified" else f"No lesson stored: {run.lesson_learned.get('reason')}.")

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
