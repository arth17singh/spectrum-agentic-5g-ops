"""Deterministic graders. Every number comes from the twin (replay); no model is involved.

    grade_kpis(final, healthy, unfixed, targets) -> {recovery, success, per_kpi}
    action_stats(op_log, fault_cells) -> {changes_used, wrong_cell_actions, clamped, rules}

``final``, ``healthy`` and ``unfixed`` are KPI dicts at the same (grade) step: the operator's twin, the
no-fault counterfactual and the faulted twin left alone. The thresholds are in ``targets.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TARGETS_PATH = Path(__file__).with_name("targets.json")
GRADER_ID = "t3/evals/graders.py:v1 (twin replay, same-step healthy counterfactual)"


def load_targets(path: str | Path = TARGETS_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def grade_kpis(
    final: dict[str, float],
    healthy: dict[str, float],
    unfixed: dict[str, float],
    targets: dict[str, Any],
) -> dict[str, Any]:
    tol, mat = targets["tolerance"], targets["material_gap"]
    per: dict[str, Any] = {}
    recs = []
    for k in targets["graded_kpis"]:
        gap = healthy[k] - unfixed[k]
        material = gap >= mat[k]
        rec = _clip((final[k] - unfixed[k]) / gap, -1.0, 1.0) if material else None
        if rec is not None:
            recs.append(rec)
        per[k] = {
            "final": final[k],
            "healthy": healthy[k],
            "unfixed": unfixed[k],
            "gap": gap,
            "material": material,
            "recovery": rec,
            "within_tolerance": final[k] >= healthy[k] - tol[k],
        }
    return {
        "recovery": sum(recs) / len(recs) if recs else None,
        "success": all(p["within_tolerance"] for p in per.values()),
        "per_kpi": per,
    }


def action_stats(op_log: list[dict[str, Any]], fault_cells: set[int]) -> dict[str, Any]:
    """Counts over the operator's steps (``action_log`` entries after the fault step)."""
    used = wrong = 0
    rules: list[str] = []
    for entry in op_log:
        items = [
            i for i in entry["change"]
            if float(i.get("d_power_db", 0.0)) != 0.0 or float(i.get("d_tilt_deg", 0.0)) != 0.0
        ]
        if items:
            used += 1
        wrong += sum(1 for i in items if int(i["cell"]) not in fault_cells)
        rules += [f"step{entry['step']}:{r}" for r in entry.get("rules", [])]
    return {"changes_used": used, "wrong_cell_actions": wrong, "clamped": bool(rules), "rules": rules}
