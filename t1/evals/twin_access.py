"""The execution unit's view of the twin: W2's `twin.replay.ReplayTwin` (required, no stand-in)
and the organisers' SDK rules (safety filter, rule baseline), used as installed.

    case(scenario, seed, prefix) -> ReplayTwin    reset, then apply each prefix change in order
    outcome(tw, change) -> dict                   W2's counterfactual: the change and the no-change
                                                  step from the same state and random stream
"""

from __future__ import annotations

import importlib.metadata
from typing import Any

import numpy as np

from t1.evals import canon as C
from twin.replay import ReplayTwin  # W2; ImportError here is deliberate: no stand-in

TWIN_SOURCE = "twin.replay.ReplayTwin (W2)"
CONFIG = None  # the "official" preset: SDK MavericTwinConfig defaults, SafetyBounds, 3600 s steps


class TwinDrift(RuntimeError):
    """A replay disagrees with the value frozen in the item: the twin is not the one it was built on."""


def sdk_version() -> str:
    return importlib.metadata.version("ran-rl")


def case(scenario: str, seed: int, prefix: list[str]) -> ReplayTwin:
    tw = ReplayTwin(scenario, seed, CONFIG)
    tw.reset()
    for ch in prefix:
        tw.apply(C.parse(ch))
    return tw


def kpi(k: dict[str, Any]) -> float:
    return float(k["coverage_ratio"]) + float(k["sla_ratio"])


def outcome(tw: ReplayTwin, change: str) -> dict[str, Any]:
    """KPI after `change` and after no change, both one step from the current state."""
    cf = tw.counterfactual(C.parse(change))
    return {"after": kpi(cf["after"]), "noop": kpi(cf["baseline"]), "gain": kpi(cf["after"]) - kpi(cf["baseline"]),
            "clamped": bool(cf["clamped"]), "rules": list(cf["rules"]),
            "after_kpis": {k: float(cf["after"][k]) for k in ("coverage_ratio", "sla_ratio")}}


def filter_check(change: Any, state: dict[str, Any]) -> tuple[list[str], bool]:
    """(rule ids, hard_violation) the SDK safety filter fires for `change` against `state`."""
    from ran_rl.env.config import SafetyBounds
    from ran_rl.env.safety import ActionSafetyFilter

    n = len(state["power_dbm"])
    res = ActionSafetyFilter(SafetyBounds()).apply(np.asarray(C.to_vector(change, n), dtype=float), state)
    return list(res.rule_ids), bool(res.hard_violation)


def rule_baseline_change(state: dict[str, Any]) -> str:
    """The SDK rule baseline (ran_rl.evaluation.baseline) as a canonical change string: the cells it
    moves, scaled to raw units as the env does (ran_rl.env.observation.scale_action); "" if none."""
    from ran_rl.env.config import SafetyBounds
    from ran_rl.env.observation import scale_action
    from ran_rl.evaluation.baseline import RuleBasedController

    b = SafetyBounds()
    raw = scale_action(RuleBasedController(b).act(state), b)
    n = len(state["power_dbm"])
    moved = [{"cell": i, "d_power_db": float(raw[i]), "d_tilt_deg": float(raw[n + i])}
             for i in range(n) if raw[i] != 0 or raw[n + i] != 0]
    return C.canon(moved)
