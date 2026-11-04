"""Confidence in a proposed change, built from evidence rather than from the agent's own opinion.

    from t3.evals.confidence import score_proposal
    out = score_proposal(incident, start_log, "cell=0;d_power_db=+0.0;d_tilt_deg=-5.0")
    out -> {"confidence": 0.71, "tier": "two-person", "preflight_ok": True,
            "breakdown": {"preflight": {...}, "federated": {...}, "agreement": None, "lesson": None},
            "weights_used": {...}, "explain": "..."}

Signals (each mapped to [0, 1]; a missing optional signal is dropped and the weights renormalised):

- preflight (weight 0.50). The change is tried on a COPY of the twin and compared, at the grade step,
  with doing nothing on that same copy. Circularity guard: the copy is deliberately NOT the twin the
  grader scores on. It uses the same scenario, preset and configuration history, but a different
  seed (``seed + PREFLIGHT_SEED_OFFSET``), so the users are dropped in different places and move
  differently. That models the gap between a twin and the real network: an operator can copy the
  network's configuration into a twin, but not where every user is. Preflight also never uses
  knowledge of which step was the fault or what the healthy KPIs were: it only compares "with the
  change" against "without the change".
- federated (0.20). The Track 2 response model (``t2/artifacts/t2_response_federated_v1.npz``,
  trained across the three silos) predicts the one-step KPI deltas of the change from the network's
  CURRENT observed state (the KPIs and configuration an operator can see; never seed or user
  positions).
- agreement (0.15, optional). Fraction of k independent agent proposals that match this one.
- lesson (0.15, optional). A score in [0, 1] from the lesson memory (a verified past fix that
  supports this change), or None.

Each KPI signal is z = mean over the graded KPIs of (delta_k / tolerance_k), clipped to [-5, 5],
mapped with logistic(z - 1): an average improvement of one tolerance unit gives 0.5.

The weights, the mapping and the tier thresholds were fixed BEFORE any calibration run and were not
fitted (calibration.py measures them on the masked test set; nothing here was tuned on it).

tier(conf, preflight_ok): "hold" if preflight shows harm or no gain, else "one-person" at >= 0.85,
"two-person" at >= 0.5, otherwise "hold". The tier only chooses how much human oversight a change
gets; no tier removes the human approval.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

from t3.evals.changes import canon, nonzero
from t3.evals.graders import load_targets
from t3.evals.twin_backend import GRADED_KPIS, ReplayTwin, graded

VERSION = "confidence-v1"
WEIGHTS = {"preflight": 0.50, "federated": 0.20, "agreement": 0.15, "lesson": 0.15}
ONE_PERSON = 0.85
TWO_PERSON = 0.50
PREFLIGHT_SEED_OFFSET = 7919
Z_CLIP = 5.0
FEDERATED_PATH = Path(__file__).resolve().parents[2] / "t2" / "artifacts" / "t2_response_federated_v1.npz"


def logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def z_of(deltas: dict[str, float], tolerance: dict[str, float], keys: Sequence[str] = GRADED_KPIS) -> float:
    """Mean improvement in tolerance units over the graded KPIs, clipped."""
    zs = [max(-Z_CLIP, min(Z_CLIP, float(deltas[k]) / float(tolerance[k]))) for k in keys]
    return sum(zs) / len(zs)


def signal_of(z: float) -> float:
    return logistic(z - 1.0)


def tier(conf: float | None, preflight_ok: bool) -> str:
    if conf is None or not preflight_ok:
        return "hold"
    if conf >= ONE_PERSON:
        return "one-person"
    if conf >= TWO_PERSON:
        return "two-person"
    return "hold"


def combine(signals: dict[str, float | None], weights: dict[str, float] = WEIGHTS) -> tuple[float, dict[str, float]]:
    """Weighted mean of the available signals; returns (confidence, weights actually used)."""
    used = {k: w for k, w in weights.items() if signals.get(k) is not None}
    total = sum(used.values())
    if total <= 0:
        raise ValueError("no signal available")
    used = {k: w / total for k, w in used.items()}
    return sum(used[k] * float(signals[k]) for k in used), used


def _steps(log: Sequence[Any]) -> list[list[dict[str, Any]]]:
    return [nonzero(s) for s in log]


# ---------- signal (a): preflight on a mismatched twin ----------
def preflight(incident: dict[str, Any], start_log: Sequence[Any], proposal: Any, targets: dict[str, Any],
              twin_factory=ReplayTwin) -> dict[str, Any]:
    ep, tol = targets["episode"], targets["tolerance"]
    grade_step = int(ep["grade_step"])
    steps = _steps(start_log)
    remaining = grade_step - len(steps) - 1
    if remaining < 0:
        raise ValueError(f"start log has {len(steps)} steps; nothing left before grade step {grade_step}")
    seed = int(incident["seed"]) + PREFLIGHT_SEED_OFFSET
    out = {}
    for name, first in (("with", nonzero(proposal)), ("without", [])):
        twin = twin_factory(incident["scenario"], seed, incident["preset"])
        twin.reset()
        state = None
        for s in steps + [first] + [[] for _ in range(remaining)]:
            state = twin.apply(s)
        out[name] = graded(state)
    deltas = {k: out["with"][k] - out["without"][k] for k in GRADED_KPIS}
    harm = [k for k in GRADED_KPIS if deltas[k] < -float(tol[k])]
    z = z_of(deltas, tol)
    return {"signal": signal_of(z), "z": z, "deltas": deltas, "harm": harm, "ok": (not harm) and z > 0,
            "twin": {"scenario": incident["scenario"], "preset": incident["preset"], "seed": seed,
                     "seed_offset": PREFLIGHT_SEED_OFFSET, "note": "mismatched copy: not the grader's seed"}}


# ---------- signal (b): federated response model ----------
@lru_cache(maxsize=2)
def load_federated(path: str = str(FEDERATED_PATH)):
    from t2.predict import load

    return load(path)


def observed_state(incident: dict[str, Any], start_log: Sequence[Any], twin_factory=ReplayTwin) -> dict[str, Any]:
    """The network's current state after the start log (what an operator observes)."""
    twin = twin_factory(incident["scenario"], int(incident["seed"]), incident["preset"])
    twin.reset()
    state = twin.state
    for s in _steps(start_log):
        state = twin.apply(s)
    return state


def federated(state: dict[str, Any], proposal: Any, targets: dict[str, Any], predictor=None) -> dict[str, Any]:
    p = predictor or load_federated()
    deltas = p.predict(state, nonzero(proposal))
    z = z_of(deltas, targets["tolerance"])
    return {"signal": signal_of(z), "z": z, "deltas": {k: deltas[k] for k in GRADED_KPIS},
            "model_id": p.model_id, "version": p.version}


def _clamp01(x: Any, name: str) -> float | None:
    if x is None:
        return None
    v = float(x)
    if not 0.0 <= v <= 1.0 or math.isnan(v):
        raise ValueError(f"{name} must be in [0, 1], got {x!r}")
    return v


def _explain(conf: float, t: str, pf: dict, fd: dict | None, agreement, lesson) -> str:
    parts = []
    if pf["harm"]:
        parts.append(f"preflight shows harm on {', '.join(pf['harm'])}")
    else:
        parts.append(f"preflight {'gain' if pf['ok'] else 'no gain'} "
                     f"(coverage {pf['deltas']['coverage_ratio']:+.3f}, SLA {pf['deltas']['sla_ratio']:+.3f})")
    if fd is not None:
        parts.append(f"federated forecast {fd['signal']:.2f}")
    if agreement is not None:
        parts.append(f"agent agreement {agreement:.0%}")
    if lesson is not None:
        parts.append(f"lesson match {lesson:.2f}")
    who = {"one-person": "one approver", "two-person": "two approvers", "hold": "held, no change proposed"}[t]
    return f"Confidence {conf:.2f}: " + "; ".join(parts) + f". Oversight: {who}."


def score_proposal(incident: dict[str, Any], start_log: Sequence[Any], proposal: Any, *,
                   agreement: float | None = None, lesson: float | None = None,
                   targets: dict[str, Any] | None = None, predictor=None, twin_factory=ReplayTwin,
                   use_federated: bool = True) -> dict[str, Any]:
    """JSON-serialisable confidence for one proposal. ``start_log``: the changes applied so far
    (commissioning, fault, earlier proposals), canonical strings or change lists."""
    targets = targets or load_targets()
    agreement, lesson = _clamp01(agreement, "agreement"), _clamp01(lesson, "lesson")
    change = nonzero(proposal)
    if not change:
        return {"version": VERSION, "confidence": None, "tier": "hold", "preflight_ok": False,
                "proposal": "", "breakdown": {}, "weights_used": {},
                "explain": "No change proposed, so there is nothing to approve."}
    pf = preflight(incident, start_log, change, targets, twin_factory=twin_factory)
    fd = None
    if use_federated:
        fd = federated(observed_state(incident, start_log, twin_factory=twin_factory), change, targets,
                       predictor=predictor)
    signals = {"preflight": pf["signal"], "federated": fd["signal"] if fd else None,
               "agreement": agreement, "lesson": lesson}
    conf, used = combine(signals)
    t = tier(conf, pf["ok"])
    return {"version": VERSION, "confidence": round(conf, 4), "tier": t, "preflight_ok": pf["ok"],
            "proposal": canon(change),
            "breakdown": {"preflight": pf, "federated": fd, "agreement": agreement, "lesson": lesson},
            "weights_used": used, "thresholds": {"one_person": ONE_PERSON, "two_person": TWO_PERSON},
            "explain": _explain(conf, t, pf, fd, agreement, lesson)}
