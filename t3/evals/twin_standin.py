"""STAND-IN for builder W2's ``twin.replay`` / ``twin.kpis`` (swapped out at merge).

This file exists only because ``twin/`` (stream W2) is not in this worktree yet. It speaks the
interface fixed in the build contract, and the same semantics W2 documents:

    t = ReplayTwin(scenario, seed, config)    # config: None / "official" / "viavi-like" / dict
    s0 = t.reset()
    s1 = t.apply([{"cell": 1, "d_power_db": -6.0, "d_tilt_deg": 0.0}])  # safety filter, then 1 step
    t.action_log                               # [{step, change, requested, filtered, rules, clamped, kpis}]
    ReplayTwin.from_log(scenario, seed, config, action_log)
    kpis(state) -> {coverage_ratio, sla_ratio, sinr_p10_db, sinr_mean_db, rsrp_p10_dbm, per_cell, ...}

``t3.evals.twin`` imports W2's modules when they are importable and falls back to this file
otherwise, so nothing else in ``t3/evals`` changes at merge. Only the parts the evals use are here.

Presets (same values W2 uses): "official" = the SDK's Maveric defaults (200 UEs, 23 dBm, 6 deg,
3600 s steps, SDK scenarios); "viavi-like" = 60 UEs that do not move (alpha 1, speed 0, no
resampling), 40 dBm, 6 deg, 60 s steps, scenario ``viavi_stationary``.

``kpis`` pools per-cell p10 values with the same Gaussian-mixture estimate W2 documents (the twin
reports only per-cell percentiles), so grades do not shift when W2's module replaces this one.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np

OFFICIAL = "official"
VIAVI_LIKE = "viavi-like"
VIAVI_SCENARIO = "viavi_stationary"
STANDIN = True


def _official() -> dict[str, Any]:
    from ran_rl.env.config import EnvConfig, SafetyBounds
    from ran_rl.twin.maveric.twin import MavericTwinConfig

    return {
        "preset": OFFICIAL,
        "twin": MavericTwinConfig().model_dump(),
        "safety": SafetyBounds().model_dump(),
        "step_seconds": float(EnvConfig().step_seconds),
        "noise_floor_dbm": None,
        "scenarios": {},
        "models_dir": None,
        "topology": None,
    }


def _viavi_like() -> dict[str, Any]:
    cfg = _official()
    cfg["preset"] = VIAVI_LIKE
    cfg["twin"].update(n_ues=60, initial_power_dbm=40.0, initial_tilt_deg=6.0)
    cfg["step_seconds"] = 60.0
    cfg["scenarios"] = {
        VIAVI_SCENARIO: {
            "mobility": {"alpha": 1.0, "velocity_m": 0.0},
            "ue_resample_fraction": 0.0,
            "length_steps": 5,
        }
    }
    return cfg


PRESETS = {OFFICIAL: _official, VIAVI_LIKE: _viavi_like}


def resolve_config(config: dict[str, Any] | str | None = None) -> dict[str, Any]:
    if config is None:
        return _official()
    if isinstance(config, str):
        return PRESETS[config]()
    if config.get("preset") in PRESETS and set(config) == set(_official()):
        return copy.deepcopy(config)
    base = PRESETS[config.get("preset", OFFICIAL)]()
    for key, value in config.items():
        if key == "preset":
            continue
        if key not in base:
            raise ValueError(f"unknown config key {key!r}")
        if isinstance(value, dict) and isinstance(base[key], dict):
            base[key] = {**base[key], **copy.deepcopy(value)}
        else:
            base[key] = copy.deepcopy(value)
    return base


_MODEL_CACHE: dict[str, Any] = {}


def _build_twin(cfg: dict[str, Any]) -> Any:
    from ran_rl.twin.factory import MAVERIC_MODELS_DIR, MAVERIC_TOPOLOGY
    from ran_rl.twin.maveric.twin import MavericDigitalTwin, MavericTwinConfig
    from ran_rl.twin.scenarios import Scenario, load_scenarios

    scenarios = load_scenarios()
    scenarios.update({k: Scenario(**v) for k, v in cfg["scenarios"].items()})
    key = repr((cfg["models_dir"], cfg["topology"], sorted(cfg["twin"].items()), sorted(cfg["scenarios"])))
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = MavericDigitalTwin(
            cfg["models_dir"] or MAVERIC_MODELS_DIR,
            cfg["topology"] or MAVERIC_TOPOLOGY,
            MavericTwinConfig(**cfg["twin"]),
            scenarios=scenarios,
        )
    proto = _MODEL_CACHE[key]
    twin = copy.copy(proto)  # shares the read-only GP models; reset() rebuilds the episode state
    for name in ("_power", "_tilt", "_alarm", "_asleep"):
        setattr(twin, name, getattr(proto, name).copy())
    return twin


def _f(x: float) -> float | str:
    x = float(x)
    if math.isfinite(x):
        return x
    return "nan" if math.isnan(x) else ("inf" if x > 0 else "-inf")


class ReplayTwin:
    """In-process Maveric twin plus the log of every change applied (stand-in for W2)."""

    def __init__(self, scenario: str, seed: int, config: dict[str, Any] | str | None = None) -> None:
        self.scenario = scenario
        self.seed = int(seed)
        self.config = resolve_config(config)
        self._twin = _build_twin(self.config)
        self._safety: Any = None
        self._state: dict[str, Any] | None = None
        self.action_log: list[dict[str, Any]] = []

    def reset(self) -> dict[str, Any]:
        from ran_rl.env.config import SafetyBounds
        from ran_rl.env.safety import ActionSafetyFilter

        self._safety = ActionSafetyFilter(SafetyBounds(**self.config["safety"]))
        kwargs = {}
        if self.config["noise_floor_dbm"] is not None:
            kwargs["noise_floor_dbm"] = self.config["noise_floor_dbm"]
        self._state = self._twin.reset(self.scenario, self.seed, **kwargs)
        self.action_log = []
        return copy.deepcopy(self._state)

    def apply(self, change: list[dict[str, Any]]) -> dict[str, Any]:
        if self._state is None:
            raise RuntimeError("call reset() first")
        n = len(self._state["power_dbm"])
        norm = []
        raw = np.zeros(2 * n)
        for item in change:
            extra = set(item) - {"cell", "d_power_db", "d_tilt_deg"}
            if extra:
                raise ValueError(f"unknown change keys {sorted(extra)}")
            cell = item["cell"]
            if isinstance(cell, bool) or not isinstance(cell, (int, np.integer)) or not 0 <= cell < n:
                raise ValueError(f"bad cell {cell!r}")
            dp, dt = float(item.get("d_power_db", 0.0)), float(item.get("d_tilt_deg", 0.0))
            norm.append({"cell": int(cell), "d_power_db": _f(dp), "d_tilt_deg": _f(dt)})
            raw[int(cell)] += dp
            raw[n + int(cell)] += dt
        result = self._safety.apply(raw, self._state)
        filt = result.filtered
        action = {"power_delta_db": filt[:n].tolist(), "tilt_delta_deg": filt[n : 2 * n].tolist()}
        self._twin.apply_action(action)
        self._state = self._twin.advance(self.config["step_seconds"])
        self.action_log.append(
            {
                "step": len(self.action_log),
                "change": norm,
                "requested": {
                    "power_delta_db": [_f(v) for v in raw[:n]],
                    "tilt_delta_deg": [_f(v) for v in raw[n:]],
                },
                "filtered": {k: list(map(float, v)) for k, v in action.items()},
                "rules": list(result.rule_ids),
                "clamped": bool(result.rule_ids),
                "kpis": {k: _f(v) for k, v in headline(self._state).items()},
            }
        )
        return copy.deepcopy(self._state)

    @property
    def state(self) -> dict[str, Any]:
        if self._state is None:
            raise RuntimeError("call reset() first")
        return copy.deepcopy(self._state)

    @property
    def step(self) -> int:
        return len(self.action_log)

    @classmethod
    def from_log(
        cls,
        scenario: str,
        seed: int,
        config: dict[str, Any] | str | None,
        action_log: list[dict[str, Any]],
        *,
        verify: bool = True,
    ) -> ReplayTwin:
        twin = cls(scenario, seed, config)
        twin.reset()
        for i, logged in enumerate(action_log):
            twin.apply(logged["change"])
            if verify and twin.action_log[-1]["filtered"] != logged["filtered"]:
                raise RuntimeError(f"replay mismatch at step {i}")
        return twin


# --- KPIs ----------------------------------------------------------------------------------------

_Z10 = 1.2815515655446004


def _pooled_p10(counts: list[int], means: list[float], p10s: list[float]) -> float:
    occ = [(c, m, p) for c, m, p in zip(counts, means, p10s, strict=True) if c > 0]
    if not occ:
        return float("nan")
    total = float(sum(c for c, _, _ in occ))
    parts = [(c / total, m, max((m - p) / _Z10, 1e-3)) for c, m, p in occ]

    def cdf(x: float) -> float:
        return sum(w * 0.5 * (1.0 + math.erf((x - m) / (s * math.sqrt(2.0)))) for w, m, s in parts)

    lo = min(m - 10.0 * s for _, m, s in parts)
    hi = max(m + 10.0 * s for _, m, s in parts)
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        if cdf(mid) < 0.10:
            lo = mid
        else:
            hi = mid
    return min(max(0.5 * (lo + hi), min(p for _, _, p in occ)), max(p for _, _, p in occ))


def kpis(state: dict[str, Any]) -> dict[str, Any]:
    counts = [int(c) for c in state["ue_count"]]
    total = sum(counts)
    sinr_means = [float(v) for v in state["sinr_mean_db"]]
    return {
        "coverage_ratio": float(state["coverage_ratio"]),
        "sla_ratio": float(state["sla_ratio"]),
        "sinr_p10_db": _pooled_p10(counts, sinr_means, [float(v) for v in state["sinr_p10_db"]]),
        "sinr_mean_db": (
            sum(c * m for c, m in zip(counts, sinr_means, strict=True)) / total if total else float("nan")
        ),
        "rsrp_p10_dbm": _pooled_p10(
            counts, [float(v) for v in state["rsrp_mean_dbm"]], [float(v) for v in state["rsrp_p10_dbm"]]
        ),
        "per_cell": [
            {"cell": i, **{k: state[k][i] for k in ("power_dbm", "tilt_deg", "ue_count", "alarm") if k in state}}
            for i in range(len(counts))
        ],
        "n_ues": total,
    }


def headline(state_or_kpis: dict[str, Any]) -> dict[str, float]:
    k = state_or_kpis if "per_cell" in state_or_kpis else kpis(state_or_kpis)
    return {key: float(k[key]) for key in ("coverage_ratio", "sla_ratio", "sinr_p10_db", "sinr_mean_db")}
