"""Swarm Tower: pure functions over the live swarm run's records (no I/O here).

The live swarm run (t3/evals/live_swarm.py, unit t3.evals.swarm-proposals) worked 18 masked incidents x 3 operator
steps on the team's Maveric twin through the AgentCore Gateway. Each (incident, step) is one case:
  item record        records/<run>/item/<key>.json: passed (the case ran to the lead's answer and the proposal was
                     read), latency_ms, aws_time (the Gateway response Date), extra{incident_id, step, case_id,
                     change_text, approval_id, refused, state, problem, reply_excerpt}
  actions line       outputs/<run>/swarm-actions.jsonl: the same case plus task_text, start_action_log, model_id
Words used on screen, from the grader:
  refused = true     the swarm's change stopped at the approval gate on ran___apply_change: it is ready but NOT
                     applied and waits for a person (approval_id names the waiting approval). The run is
                     proposal-only: nobody approved, nothing was applied on the twin.
  state              the lead agent's task state when the case was read: TASK_STATE_COMPLETED = finished,
                     TASK_STATE_WORKING = still working when the 300 s wait ran out (the case timed out).
  recovered          from the scoring run (t3/evals/run.py --operator swarm, x86): replaying the three proposals,
                     every graded KPI ends within tolerance of the healthy (no-fault) network at the grade step.
Incident id: inc-<off|vvl>-<tier>-<seed>-<template>: off = Maveric "official" preset, vvl = Maveric "viavi-like"
preset (Maveric settings, not the Viavi twin), tier mas = masked (the agents see symptoms only), seed = the twin
seed, template = the fault injected (e.g. tilt+5: one cell's tilt raised 5 degrees).
"""

from __future__ import annotations

import re
from typing import Any

Rec = dict[str, Any]

STATE_WORDS = {
    "TASK_STATE_COMPLETED": "the lead finished",
    "TASK_STATE_WORKING": "the lead was still working when the wait ran out",
    "TASK_STATE_SUBMITTED": "the task was still queued when the wait ran out",
}
PRESET_WORDS = {"off": "Maveric official preset", "vvl": "Maveric viavi-like preset (Maveric settings)"}
TIER_WORDS = {"mas": "masked: the agents see symptoms only", "vis": "visible"}
_ID = re.compile(r"^inc-(?P<preset>[a-z]+)-(?P<tier>[a-z]+)-(?P<seed>\d+)-(?P<template>.+)$")
_PART = re.compile(r"^cell=(?P<cell>\d+);d_power_db=(?P<p>[+-]?\d+(?:\.\d+)?);d_tilt_deg=(?P<t>[+-]?\d+(?:\.\d+)?)$")


def parse_change_text(s: Any) -> list[Rec]:
    """'cell=0;d_power_db=-1.2;d_tilt_deg=-0.3|cell=1;...' -> [{cell, d_power_db, d_tilt_deg}]. '' -> []."""
    if not isinstance(s, str) or not s.strip():
        return []
    out = []
    for part in s.split("|"):
        m = _PART.match(part.strip())
        if m:
            out.append({"cell": int(m["cell"]), "d_power_db": float(m["p"]), "d_tilt_deg": float(m["t"])})
    return out


def parse_incident_id(iid: Any) -> Rec:
    m = _ID.match(iid) if isinstance(iid, str) else None
    if not m:
        return {"incident_id": iid, "preset": None, "tier": None, "seed": None, "template": None}
    return {"incident_id": iid, "preset": PRESET_WORDS.get(m["preset"], m["preset"]), "tier": TIER_WORDS.get(m["tier"], m["tier"]),
            "seed": int(m["seed"]), "template": m["template"]}


def parse_models(model_id: Any) -> list[Rec]:
    """'noc_lead=openai.gpt-oss-120b-1:0;students=openai.gpt-oss-20b-1:0' -> [{role, model}]."""
    out = []
    for part in str(model_id or "").split(";"):
        if "=" in part:
            role, model = part.split("=", 1)
            out.append({"role": role.replace("_", " "), "model": model})
    return out


def outcome(row: Rec) -> Rec:
    """Plain words for one case: what happened to the swarm's proposal."""
    cells = row.get("cells") or []
    if row.get("problem"):
        return {"code": "problem", "words": f"No proposal read: {row['problem']}"}
    if row.get("refused") and cells:
        return {"code": "held", "words": "Held for approval: ready but not applied (it waits for a person; nobody approved in this run)"}
    if row.get("passed") and not cells:
        return {"code": "none", "words": "No change proposed at this step"}
    return {"code": "other", "words": "Proposal recorded"}


def group_incidents(items: list[Rec], actions: list[Rec], scores: dict[str, Rec]) -> list[Rec]:
    """One entry per incident with its steps 0..n in order; each step merges the item record and the action line."""
    act = {(a.get("incident_id"), a.get("step")): a for a in actions if isinstance(a, dict)}
    inc: dict[str, list[Rec]] = {}
    for it in items:
        ex = it.get("extra") if isinstance(it.get("extra"), dict) else {}
        iid, step = ex.get("incident_id"), ex.get("step")
        if not isinstance(iid, str) or not isinstance(step, int):
            continue
        a = act.get((iid, step), {})
        change_text = ex.get("change_text") if ex.get("change_text") is not None else a.get("change_text")
        row = {
            "step": step, "case_id": ex.get("case_id") or a.get("case_id"), "item_id": it.get("item_id"),
            "change_text": change_text or "", "cells": parse_change_text(change_text),
            "approval_id": ex.get("approval_id"), "refused": bool(ex.get("refused")), "state": ex.get("state"),
            "state_words": STATE_WORDS.get(ex.get("state"), ex.get("state") or "no state recorded"),
            "problem": ex.get("problem"), "passed": it.get("passed"), "score": it.get("score"),
            "latency_ms": it.get("latency_ms"), "reply_excerpt": ex.get("reply_excerpt"),
            "aws_time": it.get("aws_time_resolved"), "aws_time_source": it.get("aws_time_source_resolved"),
            "runner_time": it.get("runner_time"), "synthetic": it.get("synthetic") is True,
        }
        row["outcome"] = outcome(row)
        inc.setdefault(iid, []).append(row)
    out = []
    for iid, steps in inc.items():
        steps.sort(key=lambda r: r["step"])
        first = act.get((iid, steps[0]["step"]), {})
        start = first.get("start_action_log")
        sc = scores.get(iid) or {}
        meta = parse_incident_id(iid)
        incd = sc.get("incident") if isinstance(sc.get("incident"), dict) else {}
        out.append({
            **meta, "steps": steps,
            "fault": incd.get("fault") or [], "fault_cells": incd.get("fault_cells") or [],
            "start_action_log": start if isinstance(start, list) else None,
            "recovered": sc.get("success") if isinstance(sc.get("success"), bool) else None,
            "recovery": sc.get("recovery"), "models": parse_models(first.get("model_id")),
        })
    return sorted(out, key=lambda x: (x.get("seed") or 0, x["incident_id"]))
