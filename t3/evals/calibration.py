"""Is the confidence score calibrated? When it says 0.8, does the fix work about 80% of the time?

Measured on RECORDED swarm proposals (no Bedrock calls), graded by the twin, never by a model:

- input: a swarm-actions.jsonl recording (t3/evals/live_swarm.py writes one per live run; the judged
  one is ``outputs/t3.swarm-live-20261103T060140Z/swarm-actions.jsonl``, 18 masked test incidents x
  3 steps). Rows with no proposal ("no change") are skipped: there is nothing to approve.
- confidence: ``t3.evals.confidence.score_proposal`` (preflight on a mismatched twin + the federated
  forecast; agreement and lesson are not in the recording, so they are absent and renormalised away).
- outcome (the label): on the GRADER's twin (the incident's own scenario, seed and preset), apply
  the row's start log, then the proposal, then no-ops up to the grade step; success = every graded KPI
  within tolerance of the frozen healthy counterfactual (t3/evals/graders.py, the same rule run.py
  uses). That is "if the swarm had stopped after this change and a person approved it, would the
  network be healthy at the grade step?".
- metrics: Brier score, expected calibration error over 5 equal-width bins, and the reliability bins.

Target (declared before the run): ECE <= 0.15. Brier is reported, not a pass line.

    python -m t3.evals.calibration --recording <swarm-actions.jsonl> [--out calib.json]      # dry run
    python -m t3.evals.calibration --recording ... --run-id t3.evals.calibration-<UTC> --live # evidence

Dry run (default): computes everything and writes a local JSON only. --live writes the declaration
(before the first item), the items and the result through tools.evlib, like the other units.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

from t3.evals import confidence as conf_mod
from t3.evals import incidents as inc_mod
from t3.evals.changes import nonzero
from t3.evals.graders import GRADER_ID, grade_kpis, load_targets
from t3.evals.twin_backend import BACKEND, ReplayTwin, graded

UNIT = "t3.evals.confidence-calibration"
SCRIPT = "t3/evals/calibration.py"
N_BINS = 5
TARGET = {"metric": "ece", "op": "<=", "value": 0.15, "unit": "probability"}


# ---------- metrics (pure) ----------
def brier(p: Sequence[float], y: Sequence[int]) -> float:
    if not p:
        raise ValueError("no items")
    return sum((pi - yi) ** 2 for pi, yi in zip(p, y)) / len(p)


def reliability(p: Sequence[float], y: Sequence[int], n_bins: int = N_BINS) -> list[dict[str, Any]]:
    """Equal-width bins on [0, 1]; the last bin includes 1.0. Empty bins are kept (n = 0)."""
    bins = [{"lo": i / n_bins, "hi": (i + 1) / n_bins, "n": 0, "sum_p": 0.0, "sum_y": 0} for i in range(n_bins)]
    for pi, yi in zip(p, y):
        k = min(int(pi * n_bins), n_bins - 1)
        bins[k]["n"] += 1
        bins[k]["sum_p"] += pi
        bins[k]["sum_y"] += yi
    out = []
    for b in bins:
        n = b["n"]
        out.append({"lo": b["lo"], "hi": b["hi"], "n": n,
                    "mean_confidence": b["sum_p"] / n if n else None,
                    "success_rate": b["sum_y"] / n if n else None})
    return out


def ece(p: Sequence[float], y: Sequence[int], n_bins: int = N_BINS) -> float:
    n = len(p)
    if not n:
        raise ValueError("no items")
    return sum(b["n"] / n * abs(b["mean_confidence"] - b["success_rate"])
               for b in reliability(p, y, n_bins) if b["n"])


# ---------- the label: the grader's twin ----------
def outcome(incident: dict[str, Any], start_log: Sequence[Any], proposal: Any, targets: dict[str, Any],
            twin_factory=ReplayTwin) -> dict[str, Any]:
    g_step = int(targets["episode"]["grade_step"])
    steps = [nonzero(s) for s in start_log]
    remaining = g_step - len(steps) - 1
    if remaining < 0:
        raise ValueError("start log reaches the grade step")
    twin = twin_factory(incident["scenario"], int(incident["seed"]), incident["preset"])
    twin.reset()
    state = None
    for s in steps + [nonzero(proposal)] + [[] for _ in range(remaining)]:
        state = twin.apply(s)
    g = grade_kpis(graded(state), incident["healthy_kpis_by_step"][g_step],
                   incident["unfixed_kpis_by_step"][g_step], targets)
    return {"success": bool(g["success"]), "recovery": g["recovery"]}


def load_recording(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def evaluate(rows: list[dict[str, Any]], incidents: dict[str, dict[str, Any]], targets: dict[str, Any], *,
             scorer: Callable[..., dict[str, Any]] = conf_mod.score_proposal,
             grader: Callable[..., dict[str, Any]] = outcome) -> dict[str, Any]:
    items, skipped = [], []
    for r in rows:
        if not r.get("change_text"):
            skipped.append({"incident_id": r.get("incident_id"), "step": r.get("step"), "why": "no change proposed"})
            continue
        inc = incidents[r["incident_id"]]
        s = scorer(inc, r["start_action_log"], r["change_text"], targets=targets)
        o = grader(inc, r["start_action_log"], r["change_text"], targets)
        items.append({"item_id": f"{r['incident_id']}#s{r['step']}", "incident_id": r["incident_id"],
                      "step": r["step"], "case_id": r.get("case_id"), "proposal": r["change_text"],
                      "confidence": s["confidence"], "tier": s["tier"], "preflight_ok": s["preflight_ok"],
                      "success": o["success"], "recovery": o["recovery"],
                      "signals": {k: (v["signal"] if isinstance(v, dict) else v)
                                  for k, v in s["breakdown"].items()}})
    p = [it["confidence"] for it in items]
    y = [1 if it["success"] else 0 for it in items]
    by_tier: dict[str, dict[str, Any]] = {}
    for it in items:
        t = by_tier.setdefault(it["tier"], {"n": 0, "successes": 0})
        t["n"] += 1
        t["successes"] += int(it["success"])
    summary = {"n": len(items), "skipped": len(skipped), "base_rate": sum(y) / len(y) if y else None,
               "brier": brier(p, y) if p else None, "ece": ece(p, y) if p else None,
               "reliability": reliability(p, y) if p else [], "by_tier": by_tier,
               "brier_of_base_rate": brier([sum(y) / len(y)] * len(y), y) if y else None}
    return {"items": items, "skipped": skipped, "summary": summary}


def _envelope(kind, run_id, **f):
    from t3.suite.common import envelope

    return envelope(kind, run_id, **f)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--recording", required=True, help="swarm-actions.jsonl from a live swarm run")
    ap.add_argument("--run-id", default="t3.evals.calibration-adhoc")
    ap.add_argument("--out", help="write the full local result JSON here")
    ap.add_argument("--live", action="store_true", help="write declared/items/result evidence via tools.evlib")
    a = ap.parse_args(argv)
    targets = load_targets()
    incidents, digest = inc_mod.load_test()
    by_id = {i["incident_id"]: i for i in incidents}
    body = Path(a.recording).read_bytes()
    rows = load_recording(a.recording)
    ev = None
    if a.live:
        from t3.suite import common

        ev = common.load_evlib()
        ev.write_declared(_envelope(
            "declared", a.run_id, unit=UNIT, track="track-3", target=dict(TARGET),
            n_items=sum(1 for r in rows if r.get("change_text")), test_set_sha256=digest,
            recording_sha256=hashlib.sha256(body).hexdigest(), model_id=conf_mod.VERSION,
            grader=f"{SCRIPT} + {GRADER_ID}: success of the proposal followed by no-ops at the grade step",
            code_sha=common.code_sha(), declared_at=common.utc_now_iso(),
            twin={"kind": BACKEND, "confidence_preflight_seed_offset": conf_mod.PREFLIGHT_SEED_OFFSET},
            confidence={"weights": conf_mod.WEIGHTS, "one_person": conf_mod.ONE_PERSON,
                        "two_person": conf_mod.TWO_PERSON, "bins": N_BINS}))
    out = evaluate(rows, by_id, targets)
    s = out["summary"]
    if ev is not None:
        ev.append_records(a.run_id, "item", [
            _envelope("item", a.run_id, item_id=it["item_id"], score=it["confidence"], passed=it["success"],
                      grader=GRADER_ID, backend=BACKEND, extra=it) for it in out["items"]])
        ev.write_result(_envelope(
            "result", a.run_id, unit=UNIT, metric="ece", value=s["ece"], target=dict(TARGET),
            passed=s["ece"] is not None and s["ece"] <= TARGET["value"], n_items=s["n"],
            details={k: s[k] for k in ("brier", "brier_of_base_rate", "base_rate", "reliability", "by_tier",
                                       "skipped")}))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps({"n": s["n"], "skipped": s["skipped"], "base_rate": s["base_rate"], "brier": s["brier"],
                      "brier_of_base_rate": s["brier_of_base_rate"], "ece": s["ece"],
                      "target": TARGET, "passed": s["ece"] is not None and s["ece"] <= TARGET["value"],
                      "by_tier": s["by_tier"], "reliability": s["reliability"],
                      "mode": "live" if a.live else "dry-run (no evidence written)"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
