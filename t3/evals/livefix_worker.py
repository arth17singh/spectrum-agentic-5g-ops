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


def handle(op: str, a: dict[str, Any]) -> Any:
    if op == "ping":
        return {"ok": True}
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
