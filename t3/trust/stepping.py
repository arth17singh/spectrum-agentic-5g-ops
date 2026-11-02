"""One applied change with its no-change counterfactual (A5), shared by demo, calibration and ladder.

Uses only the contract's twin interface: ``twin_factory(scenario, seed, config)``,
``twin_factory.from_log(scenario, seed, config, action_log)``, ``.reset()``, ``.apply(change)``,
``.action_log`` and ``kpis(state)``. The baseline is the same twin rebuilt from its action log
and stepped with an empty change, so on a deterministic twin it is exact.

Clipping (A5): a change is "clipped" when the power/tilt the twin actually moved differ from
what was asked (safety filter clip or alarm freeze). It is detected from the state itself
(after - before vs the requested deltas), so it needs nothing beyond the SDK's raw state.
"""

from __future__ import annotations

from typing import Any

TOL = 1e-6


def predicted_delta(predictor, state: dict, change: list[dict]) -> dict:
    """The predictor's KPI delta vs no change: ``predict_delta`` if it has one, else the
    difference of two ``predict`` calls (W9 interface)."""
    if hasattr(predictor, "predict_delta"):
        return {k: float(v) for k, v in predictor.predict_delta(state, change).items()}
    a, b = predictor.predict(state, change), predictor.predict(state, [])
    return {k: float(a[k]) - float(b[k]) for k in a}


def was_clipped(before: dict, after: dict, change: list[dict]) -> bool:
    n = len(before["power_dbm"])
    want_p, want_t = [0.0] * n, [0.0] * n
    for c in change:
        i = int(c["cell"])
        want_p[i] += float(c.get("d_power_db", 0.0))
        want_t[i] += float(c.get("d_tilt_deg", 0.0))
    for i in range(n):
        if abs((after["power_dbm"][i] - before["power_dbm"][i]) - want_p[i]) > TOL:
            return True
        if abs((after["tilt_deg"][i] - before["tilt_deg"][i]) - want_t[i]) > TOL:
            return True
    return False


def step_with_counterfactual(tw, twin_factory, kpis, change: list[dict], *, scenario, seed,
                             config, before_state: dict) -> dict[str, Any]:
    ref = twin_factory.from_log(scenario, seed, config, [list(c) for c in tw.action_log])
    baseline = kpis(ref.apply([]))
    after = tw.apply(change)
    measured = kpis(after)
    return {"before": before_state, "after": after, "baseline_kpis": baseline,
            "measured_kpis": measured, "clipped": was_clipped(before_state, after, change)}


def twin_desc(tw, scenario, seed) -> dict:
    d = tw.describe() if hasattr(tw, "describe") else {}
    return {"kind": d.get("kind", getattr(tw, "kind", "maveric-inprocess")),
            "version": d.get("version", getattr(tw, "version", None)),
            "scenario": scenario, "seed": seed, "step": len(tw.action_log)}
