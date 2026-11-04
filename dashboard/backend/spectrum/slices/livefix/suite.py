"""The incidents Live Fix can run, the grading rule, and the recorded fallback.

Live incidents come from the DEV split of the frozen incident suite (t3/evals/suite/incidents-dev.jsonl), visible
tier: the scored test set (incidents-test.jsonl) is never touched by a live demo. The grading rule is the suite's
own (t3/evals/targets.json tolerance, t3/evals/graders.py): every graded KPI must end within tolerance of the
healthy twin at the same step.

The recorded fallback replays the judged live swarm run's recorded proposals (outputs/<run>/swarm-actions.jsonl)
for the MASKED test incidents, with the KPIs the scoring run measured on the twin (records/<scoring run>/item).
Nothing is computed for it here; it is a replay of records, labelled RECORDED in the UI.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[5]
SUITE_DIR = REPO / "t3" / "evals" / "suite"
DEV_PATH = SUITE_DIR / "incidents-dev.jsonl"
TEST_PATH = SUITE_DIR / "incidents-test.jsonl"
TARGETS_PATH = REPO / "t3" / "evals" / "targets.json"
GRADED = ("coverage_ratio", "sla_ratio", "sinr_p10_db")
# the judged live swarm run and its scoring run (t3/evals/live_swarm.py, t3/evals/run.py)
RECORDED_LIVE_RUN = os.environ.get("LIVEFIX_RECORDED_RUN", "t3.swarm-live-20261103T060140Z")
RECORDED_SCORE_RUN = os.environ.get("LIVEFIX_RECORDED_SCORE_RUN", "t3.evals.swarm-masked-20261103T064410Z-x86")


def repo_import(name: str):
    """Import a repo-root module (t3.*, tools.*) without leaving the repo root on sys.path."""
    if name in sys.modules:
        return sys.modules[name]
    sys.path.insert(0, str(REPO))
    try:
        return importlib.import_module(name)
    finally:
        try:
            sys.path.remove(str(REPO))
        except ValueError:
            pass


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_targets() -> dict[str, Any]:
    return json.loads(TARGETS_PATH.read_text())


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def dev_incidents() -> list[dict[str, Any]]:
    """Visible-tier incidents of the dev split (never the scored test split)."""
    return [i for i in _lines(DEV_PATH) if i.get("tier") == "visible" and i.get("split") == "dev"]


def dev_digest() -> str:
    return sha256_bytes(DEV_PATH.read_bytes())


def label(inc: dict[str, Any]) -> dict[str, Any]:
    """What the picker shows: the cell and fault, for the person choosing. The agents never see this."""
    f = inc.get("fault") or []
    parts = []
    for c in f:
        bits = []
        if float(c.get("d_power_db", 0)):
            bits.append(f"power {float(c['d_power_db']):+.0f} dB")
        if float(c.get("d_tilt_deg", 0)):
            bits.append(f"tilt {float(c['d_tilt_deg']):+.0f}°")
        parts.append(f"cell {c['cell']}: " + ", ".join(bits))
    return {"incident_id": inc["incident_id"], "fault_cells": inc.get("fault_cells", []),
            "fault": "; ".join(parts), "preset": inc.get("preset"), "scenario": inc.get("scenario"),
            "seed": inc.get("seed")}


def steps_for(inc: dict[str, Any], targets: dict[str, Any], with_fault: bool, upto: int | None = None) -> list[str]:
    """The canonical change strings for steps 1..fault_step ("" = no-op), padded with no-ops up to `upto`."""
    changes = repo_import("t3.evals.changes")
    ep = targets["episode"]
    raw: list[list[dict]] = [[] for _ in range(ep["fault_step"])]
    raw[0] = list(inc.get("commissioning") or [])
    if with_fault:
        raw[ep["fault_step"] - 1] = list(inc["fault"])
    out = [changes.canon(changes.nonzero(s)) if changes.nonzero(s) else "" for s in raw]
    while upto is not None and len(out) < upto:
        out.append("")
    return out


def ticket(inc: dict[str, Any], run_id: str, n: int) -> str:
    """The noc_lead's ticket: t3/evals/swarm.task_text (symptoms only), with a neutral incident label in place
    of the suite id (the suite id names the fault template, e.g. "...-tilt+5")."""
    swarm = repo_import("t3.evals.swarm")
    return swarm.task_text({**inc, "incident_id": f"live-{n:02d}"}, run_id)


def grade(final: dict[str, float], healthy: dict[str, float], unfixed: dict[str, float],
          targets: dict[str, Any]) -> dict[str, Any]:
    graders = repo_import("t3.evals.graders")
    return graders.grade_kpis(final, healthy, unfixed, targets)


def headline(kpis: dict[str, Any] | None) -> dict[str, float] | None:
    if not isinstance(kpis, dict):
        return None
    try:
        return {k: float(kpis[k]) for k in (*GRADED, "sinr_mean_db") if k in kpis}
    except (TypeError, ValueError):
        return None


def cell_view(kpis: dict[str, Any] | None, healthy: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Per-cell rows for the hex map, with a status against the healthy twin at the same step.

    green: the cell's coverage and SLA shares are within 0.05 of healthy; amber: within 0.15; red: worse, or
    the cell raises an alarm. Without per-cell shares (no data), the status is "unknown"."""
    rows = (kpis or {}).get("per_cell") or []
    ref = {c.get("cell"): c for c in ((healthy or {}).get("per_cell") or [])}
    out = []
    for c in rows:
        h = ref.get(c.get("cell"), {})
        worst = None
        for k in ("coverage_share", "sla_share"):
            a, b = c.get(k), h.get(k)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                d = float(b) - float(a)
                worst = d if worst is None else max(worst, d)
        if c.get("alarm"):
            status = "red"
        elif worst is None:
            status = "unknown"
        elif worst <= 0.05:
            status = "green"
        elif worst <= 0.15:
            status = "amber"
        else:
            status = "red"
        out.append({"cell": c.get("cell"), "status": status,
                    **{k: c.get(k) for k in ("power_dbm", "tilt_deg", "ue_count", "coverage_share", "sla_share",
                                             "sinr_mean_db", "alarm")}})
    return out


# ------------------------------------------------------------------ recorded fallback
def evidence_root() -> Path:
    raw = os.environ.get("SPECTRUM_EVIDENCE_DIR") or os.environ.get("SPECTRUM_RECORDS_ROOT") or str(REPO / "evidence")
    return Path(raw)


def recorded_incidents(root: Path | None = None) -> list[dict[str, Any]]:
    """One entry per masked test incident the judged swarm run worked: its recorded proposals and replies, and the
    KPIs the scoring run measured (healthy, unfixed, final). Empty when the records are not on this laptop."""
    root = root or evidence_root()
    actions = root / "outputs" / RECORDED_LIVE_RUN / "swarm-actions.jsonl"
    items_dir = root / "records" / RECORDED_SCORE_RUN / "item"
    if not actions.is_file() or not items_dir.is_dir():
        return []
    by_inc: dict[str, list[dict]] = {}
    for row in _lines(actions):
        by_inc.setdefault(row["incident_id"], []).append(row)
    scores = {}
    for p in sorted(items_dir.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except ValueError:
            continue
        scores[d.get("item_id")] = d
    suite = {i["incident_id"]: i for i in _lines(TEST_PATH)}
    out = []
    for iid, rows in sorted(by_inc.items()):
        s, inc = scores.get(iid), suite.get(iid)
        if not s or not inc:
            continue
        per = (s.get("extra") or {}).get("per_kpi") or {}
        out.append({
            "incident_id": iid, "fault_cells": inc.get("fault_cells", []), "label": label(inc),
            "steps": [{k: r.get(k) for k in ("step", "case_id", "change_text", "approval_id", "reply_excerpt")}
                      for r in sorted(rows, key=lambda r: r["step"])],
            "healthy": {k: per[k]["healthy"] for k in GRADED if k in per},
            "unfixed": {k: per[k]["unfixed"] for k in GRADED if k in per},
            "final": (s.get("extra") or {}).get("final_kpis"),
            "pre_fault": inc.get("pre_fault_kpis"), "post_fault": inc.get("post_fault_kpis"),
            "success": bool((s.get("extra") or {}).get("success")),
            "recovery": s.get("score"), "live_run": RECORDED_LIVE_RUN, "score_run": RECORDED_SCORE_RUN,
        })
    return out
