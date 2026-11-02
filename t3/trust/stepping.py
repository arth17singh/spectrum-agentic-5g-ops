"""One applied change with its no-change counterfactual (A5), shared by demo, calibration and ladder.

Real path (W2's ``twin.replay.ReplayTwin``): ``tw.counterfactual(change)`` gives the no-change
baseline KPIs, the KPIs after the change and ``clamped`` (the safety filter altered the request)
on a snapshot, exactly, without touching the live twin; then ``tw.apply(change)`` advances the live
twin and its KPIs are the measured ones (identical to the trial's, by determinism).

Fallback (twins without ``counterfactual``, e.g. test fakes): rebuild the twin with
``twin_factory.from_log`` and step it with an empty change; clamping is detected from the state
(the power/tilt actually moved vs the requested deltas).

Predictions: ``t2.predict`` returns KPI DELTAS (A5), so ``predictor.predict(state, change)`` is the
predicted delta; a ``predict_delta`` method, if a predictor has one, takes precedence.
"""

from __future__ import annotations

import copy
from typing import Any, Callable

TOL = 1e-6


def predicted_delta(predictor, state: dict, change: list[dict]) -> dict:
    fn = getattr(predictor, "predict_delta", None) or predictor.predict
    return {k: float(v) for k, v in fn(state, change).items()}


def was_clipped(before: dict, after: dict, change: list[dict]) -> bool:
    n = len(before["power_dbm"])
    want_p, want_t = [0.0] * n, [0.0] * n
    for c in change:
        i = int(c["cell"])
        want_p[i] += float(c.get("d_power_db", 0.0))
        want_t[i] += float(c.get("d_tilt_deg", 0.0))
    return any(abs((after["power_dbm"][i] - before["power_dbm"][i]) - want_p[i]) > TOL
               or abs((after["tilt_deg"][i] - before["tilt_deg"][i]) - want_t[i]) > TOL for i in range(n))


def baseline_kpis(tw, twin_factory, kpis: Callable, scenario, seed, config) -> dict:
    if hasattr(tw, "counterfactual"):
        return dict(tw.counterfactual([])["baseline"])
    ref = twin_factory.from_log(scenario, seed, config, copy.deepcopy(tw.action_log))
    return kpis(ref.apply([]))


def step_with_counterfactual(tw, twin_factory, kpis: Callable, change: list[dict], *, scenario, seed,
                             config, before_state: dict) -> dict[str, Any]:
    if hasattr(tw, "counterfactual"):
        cf = tw.counterfactual(change)
        after = tw.apply(change)
        measured = kpis(after)
        return {"before": before_state, "after": after, "baseline_kpis": dict(cf["baseline"]),
                "measured_kpis": measured, "clipped": bool(cf["clamped"])}
    base = baseline_kpis(tw, twin_factory, kpis, scenario, seed, config)
    after = tw.apply(change)
    return {"before": before_state, "after": after, "baseline_kpis": base, "measured_kpis": kpis(after),
            "clipped": was_clipped(before_state, after, change)}


def twin_desc(tw, scenario, seed) -> dict:
    kind = getattr(tw, "kind", None) or "maveric-inprocess"
    version = getattr(tw, "version", None)
    if version is None:
        try:
            from twin.replay import sdk_version
            from twin.presets import config_hash

            version = f"ran-rl {sdk_version()} config {config_hash(tw.config)[:12]} ({tw.config.get('preset')})"
        except Exception:
            version = None
    return {"kind": kind, "version": version, "scenario": scenario, "seed": seed, "step": len(tw.action_log)}
