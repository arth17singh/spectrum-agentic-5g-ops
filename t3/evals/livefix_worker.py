"""Live Fix's twin-side helper: one long-lived process that answers JSON lines on stdin.

The dashboard's own Python has no Maveric twin (ran-rl), so the console starts this module with an interpreter
that does (LIVEFIX_T3_PYTHON) and asks it, one line per request:

    {"id": 1, "op": "make_incident", "args": {"fault_key": "pa_fault", "cell": 2, "preset": "official"}}
    -> {"id": 1, "ok": true, "result": {...}}   or   {"id": 1, "ok": false, "error": "..."}

Ops (every one is a pure function of its arguments plus the lessons file):
  catalog                         the fault buttons (t3.evals.faults.FAULTS)
  make_incident fault_key cell preset   t3.evals.faults.make_incident (dev seeds only)
  health        kpis reference fault     t3.evals.faults.cell_health + mark_fault_cells on per-cell KPIs
  lessons       incident k               verified lessons nearest to the incident's symptoms, the ticket block,
                                         and the best similarity (the `lesson` confidence signal)
  score         incident start_log proposal lesson   t3.evals.confidence.score_proposal
  learn         incident proposals       t3.evals.lessons.verify on the local twin, then LessonStore.learn
                                         (refuses test-split incidents)
  percell       cells_after cells_healthy cells_before fault_cells network_after network_healthy
                                         t3.evals.percell.percell_verdict (percell-v1) on per-cell KPI rows
  side_effects  incident start_log proposal   the held change on the PREFLIGHT copy of the twin
                                         (confidence.PREFLIGHT_SEED_OFFSET, not the grader's twin): each cell
                                         without -> with the change at the grade step, graded by percell-v1

The lessons file is LIVEFIX_LESSONS (default ~/.spectrum/livefix/lessons.jsonl). Nothing here calls AWS.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any


def _lessons_path() -> Path:
    return Path(os.environ.get("LIVEFIX_LESSONS") or Path.home() / ".spectrum" / "livefix" / "lessons.jsonl")


_KEEP = ("power_dbm", "tilt_deg", "ue_count", "coverage_share", "sla_share", "sinr_p10_db", "alarm", "asleep")


def side_effects(incident: dict[str, Any], start_log: list[Any], proposal: Any, targets: dict[str, Any] | None = None,
                 twin_cls: Any = None) -> dict[str, Any]:
    """What the held change would do to every cell, predicted on the PREFLIGHT copy of the twin.

    The copy is confidence.preflight's: same scenario, preset and configuration history, but the seed is
    offset (PREFLIGHT_SEED_OFFSET), so users sit elsewhere: never the twin the grader scores on. Three replays
    to the grade step on that copy: with the change, without it (the network if nothing is done), and the
    healthy counterfactual (commissioning, no fault). percell-v1 grades "with" against healthy, with "without"
    as the state before the change, so ``worse`` marks a cell the change itself makes worse."""
    from t3.evals import confidence, percell
    from t3.evals import incidents as inc
    from t3.evals.changes import nonzero
    from t3.evals.graders import load_targets
    if twin_cls is None:
        from t3.evals.twin_backend import ReplayTwin as twin_cls
    targets = targets or load_targets()
    grade_step = int(targets["episode"]["grade_step"])
    steps = [nonzero(s) for s in start_log]
    remaining = grade_step - len(steps) - 1
    if remaining < 0:
        raise ValueError(f"start log has {len(steps)} steps; nothing left before grade step {grade_step}")
    seed = int(incident["seed"]) + confidence.PREFLIGHT_SEED_OFFSET

    def replay(script: list[Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        t = twin_cls(incident["scenario"], seed, incident["preset"])
        t.reset()
        for s in script:
            t.apply(s)
        return percell.cells_from_twin(t), t.kpis()

    with_rows, k_with = replay(steps + [nonzero(proposal)] + [[]] * remaining)
    without_rows, k_without = replay(steps + [[]] * (1 + remaining))
    healthy_script = inc.script(dict(incident), targets, False)
    healthy_rows, k_healthy = replay(healthy_script + [[]] * (grade_step - len(healthy_script)))
    v_with = percell.percell_verdict(with_rows, healthy_rows, without_rows, network_after=k_with,
                                     network_healthy=k_healthy)
    v_without = percell.percell_verdict(without_rows, healthy_rows, network_after=k_without,
                                        network_healthy=k_healthy)
    was = {c["cell"]: c for c in v_without["cells"]}
    rows_with = {int(r["cell"]): r for r in with_rows}
    rows_without = {int(r["cell"]): r for r in without_rows}
    cells = []
    for c in v_with["cells"]:
        n = c["cell"]
        # percell-v1 says "before the fix"; here the state before is the network without the change.
        reasons = [str(r).replace("before the fix", "without the change") for r in c["reasons"]]
        cells.append({"cell": n, "without": {k: rows_without[n].get(k) for k in _KEEP},
                      "with": {k: rows_with[n].get(k) for k in _KEEP},
                      "health_without": was[n]["health"], "health_with": c["health"],
                      "worse": bool(c["worse_than_before"]), "reasons": reasons})
    return {"version": percell.VERSION, "cells": cells, "recovered_with": v_with["recovered"],
            "overshoot_with": v_with["overshoot"], "grade_step": grade_step,
            "twin": {"kind": "preflight copy", "scenario": incident["scenario"], "preset": incident["preset"],
                     "seed_offset": confidence.PREFLIGHT_SEED_OFFSET,
                     "note": "users placed differently from the grader's twin: a prediction, not the network"}}


def handle(op: str, a: dict[str, Any]) -> Any:
    if op == "ping":
        return {"ok": True}
    if op == "percell":
        from t3.evals import percell
        return percell.percell_verdict(a["cells_after"], a["cells_healthy"], a.get("cells_before"),
                                       fault_cells=a.get("fault_cells"), network_after=a.get("network_after"),
                                       network_healthy=a.get("network_healthy"))
    if op == "side_effects":
        return side_effects(a["incident"], a["start_log"], a["proposal"])
    if op == "catalog":
        from t3.evals import faults
        return {"faults": faults.FAULTS, "version": faults.CATALOG_VERSION}
    if op == "make_incident":
        from t3.evals import faults
        return faults.make_incident(str(a["fault_key"]), int(a["cell"]), preset=a.get("preset") or "official")
    if op == "health":
        from t3.evals import faults
        h = faults.cell_health(a["kpis"], a.get("reference"))
        return faults.mark_fault_cells(h, a.get("fault") or [])
    if op == "lessons":
        from t3.evals import lessons as L
        store = L.LessonStore(_lessons_path())
        sy = L.symptoms_of(a["incident"])
        hits = store.retrieve(sy, k=int(a.get("k", 3)))
        return {"block": L.lesson_block(hits), "best": store.best_similarity(sy),
                "hits": [{"lesson_id": l["lesson_id"], "similarity": s, "pattern_text": l["pattern_text"],
                          "support": l.get("support", 1), "source": l.get("source"),
                          "symptoms": l["symptoms"]} for l, s in hits],
                "n_lessons": len(store.lessons), "path": str(_lessons_path())}
    if op == "score":
        from t3.evals import confidence
        return confidence.score_proposal(a["incident"], a["start_log"], a["proposal"], lesson=a.get("lesson"))
    if op == "learn":
        from t3.evals import lessons as L
        store = L.LessonStore(_lessons_path())
        v = L.verify(a["incident"], a["proposals"])
        rec = store.learn(a["incident"], a["proposals"], v, source=a.get("source") or "livefix")
        return {"lesson": {k: rec.get(k) for k in ("lesson_id", "status", "reason", "pattern_text", "support")},
                "verification": {k: v[k] for k in ("verifier", "backend", "recovered", "recovery", "harmful",
                                                   "steps_to_fix")}}
    raise ValueError(f"unknown op {op!r}")


def main() -> int:
    out = sys.stdout
    sys.stdout = sys.stderr  # library prints never corrupt the protocol
    for line in sys.stdin:
        if not line.strip():
            continue
        req: dict[str, Any] = {}
        try:
            req = json.loads(line)
            res = {"id": req.get("id"), "ok": True, "result": handle(req["op"], req.get("args") or {})}
        except Exception as e:  # noqa: BLE001 - every failure is an answer, the worker keeps running
            traceback.print_exc(file=sys.stderr)
            res = {"id": req.get("id"), "ok": False, "error": f"{type(e).__name__}: {str(e)[:500]}"}
        out.write(json.dumps(res, default=str) + "\n")
        out.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
