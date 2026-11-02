"""STAND-IN for builder W2's `twin.replay.ReplayTwin` (swapped at merge; see get_replay_twin).

This is a thin, clearly labelled stand-in so the execution unit runs before the merge. It does what
the build contract fixes for `ReplayTwin` and nothing more:

    ReplayTwin(scenario, seed, config=None)    config None = the SDK's official Maveric defaults
    .reset() -> state                          MavericDigitalTwin.reset(scenario, seed)
    .apply(change) -> state                    the SDK ActionSafetyFilter against the current state,
                                               the filtered deltas to the twin, advance 3600 s
                                               (exactly what RANOptimizationEnv.step does)
    .action_log                                one entry per apply: change, filtered, rules
    ReplayTwin.from_log(scenario, seed, config, action_log)

It uses the organisers' SDK (ran-rl, in-process Maveric GP twin) as installed; nothing is vendored.
`get_replay_twin()` returns W2's class when `twin.replay` is importable, else this one.
"""

from __future__ import annotations

import copy
import importlib.metadata
from typing import Any

import numpy as np

from t1.evals import canon as C

STANDIN = True
STEP_SECONDS = 3600.0
_PROTO: dict[str, Any] = {}


def sdk_version() -> str:
    return importlib.metadata.version("ran-rl")


def _twin() -> Any:
    from ran_rl.twin.factory import MAVERIC_MODELS_DIR, MAVERIC_TOPOLOGY
    from ran_rl.twin.maveric.twin import MavericDigitalTwin, MavericTwinConfig

    if "proto" not in _PROTO:
        _PROTO["proto"] = MavericDigitalTwin(MAVERIC_MODELS_DIR, MAVERIC_TOPOLOGY, MavericTwinConfig())
    proto = _PROTO["proto"]
    tw = copy.copy(proto)  # shares the read-only GP models; reset() rebuilds the episode arrays
    for name in ("_power", "_tilt", "_alarm", "_asleep"):
        setattr(tw, name, getattr(proto, name).copy())
    return tw


class ReplayTwin:
    def __init__(self, scenario: str, seed: int, config: dict | str | None = None) -> None:
        if config not in (None, "official"):
            raise ValueError("the stand-in supports only the official config (None)")
        self.scenario, self.seed, self.config = scenario, int(seed), None
        self._twin = _twin()
        self._state: dict[str, Any] | None = None
        self.action_log: list[dict[str, Any]] = []

    def reset(self) -> dict[str, Any]:
        from ran_rl.env.config import SafetyBounds
        from ran_rl.env.safety import ActionSafetyFilter

        self._filter = ActionSafetyFilter(SafetyBounds())
        self._state = self._twin.reset(self.scenario, self.seed)
        self.action_log = []
        return copy.deepcopy(self._state)

    def apply(self, change: Any) -> dict[str, Any]:
        if self._state is None:
            raise RuntimeError("call reset() first")
        n = len(self._state["power_dbm"])
        raw = np.asarray(C.to_vector(change, n), dtype=float)
        res = self._filter.apply(raw, self._state)
        f = res.filtered
        self._twin.apply_action({"power_delta_db": f[:n].tolist(), "tilt_delta_deg": f[n:2 * n].tolist()})
        self._state = self._twin.advance(STEP_SECONDS)
        self.action_log.append(
            {
                "step": len(self.action_log),
                "change": C.parse(C.canon(change)),
                "filtered": {"power_delta_db": f[:n].tolist(), "tilt_delta_deg": f[n:2 * n].tolist()},
                "rules": list(res.rule_ids),
                "clamped": bool(res.rule_ids),
                "hard_violation": bool(res.hard_violation),
            }
        )
        return copy.deepcopy(self._state)

    @property
    def state(self) -> dict[str, Any]:
        if self._state is None:
            raise RuntimeError("call reset() first")
        return copy.deepcopy(self._state)

    @classmethod
    def from_log(cls, scenario: str, seed: int, config: Any, action_log: list[dict[str, Any]], **_: Any) -> "ReplayTwin":
        tw = cls(scenario, seed, config)
        tw.reset()
        for entry in action_log:
            tw.apply(entry["change"])
        return tw


def get_replay_twin() -> tuple[type, str]:
    """(ReplayTwin class, source label). W2's twin.replay when merged, else this stand-in."""
    try:
        from twin.replay import ReplayTwin as W2ReplayTwin  # type: ignore

        return W2ReplayTwin, "twin.replay (W2)"
    except Exception:  # noqa: BLE001 - not merged yet
        return ReplayTwin, "t1.evals.twin_standin (stand-in for W2 twin.replay)"


def filter_check(change: Any, state: dict[str, Any]) -> tuple[list[str], bool]:
    """(rule ids, hard_violation) the SDK safety filter fires for `change` against `state`."""
    from ran_rl.env.config import SafetyBounds
    from ran_rl.env.safety import ActionSafetyFilter

    n = len(state["power_dbm"])
    res = ActionSafetyFilter(SafetyBounds()).apply(np.asarray(C.to_vector(change, n), dtype=float), state)
    return list(res.rule_ids), bool(res.hard_violation)


def rule_baseline_change(state: dict[str, Any]) -> str:
    """The SDK rule baseline (ran_rl.evaluation.baseline) as a canonical change string, scaled to
    raw units the way the env scales an action (ran_rl.env.observation.scale_action)."""
    from ran_rl.env.config import SafetyBounds
    from ran_rl.env.observation import scale_action
    from ran_rl.evaluation.baseline import RuleBasedController

    b = SafetyBounds()
    raw = scale_action(RuleBasedController(b).act(state), b)
    n = len(state["power_dbm"])
    return C.canon(
        [{"cell": i, "d_power_db": float(raw[i]), "d_tilt_deg": float(raw[n + i])} for i in range(n)]
    )
