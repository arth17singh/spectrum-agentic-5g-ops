"""STAND-IN for W2's ``twin.replay.ReplayTwin``. Not the team's twin module.

W5 needs a twin to run the trust demo and the ladder before W2's module is merged. This wraps
the SDK's in-process Maveric twin (``ran_rl.twin.maveric``) behind the same interface the build
contract fixes for ``twin.replay``:

    ReplayTwin(scenario, seed, config) .reset() -> state  .apply(change) -> state  .action_log
    ReplayTwin.from_log(scenario, seed, config, action_log)      kpis(state) -> {...}

``apply`` maps ``[{cell, d_power_db, d_tilt_deg}]`` to the SDK action vector, runs it through the
SDK's ``ActionSafetyFilter`` (default ``SafetyBounds``), applies it and advances one step
(3600 s, the SDK's default step). ``kpis`` adds the network-wide SINR p10 and mean over every
UE, which the SDK state only gives per cell; the stand-in computes them from the twin's own
UE-level SINR (reads private attributes of the SDK twin, acceptable for a stand-in only).

Presets (``config``): ``"official"`` = the SDK's Maveric defaults (200 UEs, 23 dBm, 6 deg,
mobile UEs); ``"viavi-like"`` = 60 stationary UEs, 40 dBm, 6 deg (the contract's preset). The
"viavi-like" preset is a REHEARSAL on Maveric. It is not Viavi.

At merge, replace with ``twin.replay`` and delete this file.
"""

from __future__ import annotations

import copy
import json
from functools import lru_cache
from typing import Any

import numpy as np

STANDIN_LABEL = "t3.trust.standin_twin (stand-in for twin.replay; in-process Maveric via ran_rl)"
STEP_SECONDS = 3600.0

PRESETS: dict[str, dict[str, Any]] = {
    "official": {"n_ues": 200, "initial_power_dbm": 23.0, "initial_tilt_deg": 6.0, "stationary": False},
    "viavi-like": {"n_ues": 60, "initial_power_dbm": 40.0, "initial_tilt_deg": 6.0, "stationary": True},
}


@lru_cache(maxsize=1)
def _assets():
    from ran_rl.twin.factory import MAVERIC_MODELS_DIR, MAVERIC_TOPOLOGY
    from ran_rl.twin.maveric.features import load_topology
    from ran_rl.twin.maveric.model import load_gp_models

    return MAVERIC_MODELS_DIR, MAVERIC_TOPOLOGY, load_gp_models(MAVERIC_MODELS_DIR), load_topology(MAVERIC_TOPOLOGY)


@lru_cache(maxsize=16)
def _prototype(scenario: str, config_json: str):
    """One fully built SDK twin per (scenario, config); building one loads the GP models."""
    from ran_rl.twin.maveric.twin import MavericDigitalTwin, MavericTwinConfig
    from ran_rl.twin.scenarios import Mobility, load_scenarios

    cfg = json.loads(config_json)
    models_dir, topo, _, _ = _assets()
    scenarios = load_scenarios()
    if cfg["stationary"]:
        base = scenarios[scenario]
        scenarios = dict(scenarios)
        scenarios[scenario] = base.model_copy(update={
            "ue_resample_fraction": 0.0, "mobility": Mobility(alpha=1.0, velocity_m=0.0)})
    tcfg = MavericTwinConfig(n_ues=int(cfg["n_ues"]), initial_power_dbm=float(cfg["initial_power_dbm"]),
                             initial_tilt_deg=float(cfg["initial_tilt_deg"]))
    return MavericDigitalTwin(models_dir, topo, tcfg, scenarios=scenarios)


def _fresh_twin(scenario: str, config_json: str):
    """A shallow copy of the prototype that shares the read-only GP models and topology but owns
    every numpy array (the SDK twin mutates power/tilt/alarm arrays in place)."""
    tw = copy.copy(_prototype(scenario, config_json))
    for k, v in list(vars(tw).items()):
        if isinstance(v, np.ndarray):
            setattr(tw, k, v.copy())
    return tw


def _resolve(config) -> tuple[str, dict]:
    if config is None:
        config = "official"
    if isinstance(config, str):
        if config not in PRESETS:
            raise ValueError(f"unknown preset {config!r}; known: {sorted(PRESETS)}")
        return config, dict(PRESETS[config])
    cfg = dict(config)
    name = cfg.pop("preset", "official")
    out = dict(PRESETS[name])
    out.update(cfg)
    return name, out


class StandinReplayTwin:
    """Deterministic replay of the in-process Maveric twin from (scenario, seed, config, action_log)."""

    kind = "maveric-inprocess"

    def __init__(self, scenario: str, seed: int, config=None):
        from ran_rl.env.config import SafetyBounds
        from ran_rl.env.safety import ActionSafetyFilter
        from ran_rl.twin.maveric.twin import MavericDigitalTwin, MavericTwinConfig
        from ran_rl.twin.scenarios import Mobility, load_scenarios

        self.scenario, self.seed = scenario, int(seed)
        self.preset, self.config = _resolve(config)
        _, _, _, cells = _assets()
        self._tw = _fresh_twin(scenario, json.dumps(self.config, sort_keys=True))
        self._filter = ActionSafetyFilter(SafetyBounds())
        self.cell_ids = [c.cell_id for c in cells]
        self.n_cells = len(self.cell_ids)
        self.action_log: list[list[dict]] = []
        self.step = 0
        self._state: dict | None = None
        self.last_filter_rules: tuple[str, ...] = ()

    @property
    def version(self) -> str:
        import ran_rl

        return f"ran_rl {getattr(ran_rl, '__version__', '?')} maveric sim_tilt_sweep_4cell, preset {self.preset}"

    def describe(self) -> dict:
        return {"kind": self.kind, "version": self.version, "scenario": self.scenario,
                "seed": self.seed, "step": self.step, "config": self.preset, "impl": STANDIN_LABEL}

    def _cell_index(self, cell) -> int:
        if isinstance(cell, (int, np.integer)):
            i = int(cell)
        elif isinstance(cell, str) and cell in self.cell_ids:
            i = self.cell_ids.index(cell)
        else:
            raise ValueError(f"unknown cell {cell!r}; cells are 0..{self.n_cells - 1} or {self.cell_ids}")
        if not 0 <= i < self.n_cells:
            raise ValueError(f"cell index {i} out of range")
        return i

    def _with_globals(self, state: dict) -> dict:
        from ran_rl.twin.maveric.model import predict
        from ran_rl.twin.maveric.vendor.attachment import attach

        tw = self._tw
        pred = predict(tw._models, tw._cells, tw._ue, tw._tilt, tw._power)
        rsrp = pred.rsrp_dbm
        rsrp[:, tw._alarm | tw._asleep] = -np.inf
        _, served, sinr = attach(rsrp, tw._freq, tw._noise_floor_dbm)
        sinr = np.asarray(sinr, dtype=float)
        served = np.asarray(served, dtype=float)
        state = dict(state)
        state["extra"] = dict(state.get("extra", {}))
        state["extra"]["standin_global"] = {"sinr_p10_db": float(np.percentile(sinr, 10)),
                                            "sinr_mean_db": float(sinr.mean()),
                                            "rsrp_p10_dbm": float(np.percentile(served, 10))}
        return state

    def reset(self) -> dict:
        self.action_log, self.step = [], 0
        self._state = self._with_globals(self._tw.reset(self.scenario, self.seed))
        return self._state

    def apply(self, change: list[dict]) -> dict:
        if self._state is None:
            self.reset()
        n = self.n_cells
        raw = np.zeros(2 * n)
        for c in change:
            i = self._cell_index(c["cell"])
            raw[i] += float(c.get("d_power_db", 0.0))
            raw[n + i] += float(c.get("d_tilt_deg", 0.0))
        res = self._filter.apply(raw, self._state)
        self.last_filter_rules = res.rule_ids
        self._tw.apply_action({"power_delta_db": res.filtered[:n].tolist(),
                               "tilt_delta_deg": res.filtered[n:].tolist()})
        self._state = self._with_globals(self._tw.advance(STEP_SECONDS))
        self.action_log.append([dict(c) for c in change])
        self.step += 1
        return self._state

    def fork(self) -> "StandinReplayTwin":
        """An independent copy at the same step (same UEs, RNG state, settings and log). Equivalent
        to ``from_log(scenario, seed, config, action_log)`` but O(1); shares only the read-only GP
        models and topology."""
        new = copy.copy(self)
        tw = copy.copy(self._tw)
        for k, v in list(vars(tw).items()):
            if isinstance(v, np.ndarray):
                setattr(tw, k, v.copy())
        tw._rng = copy.deepcopy(self._tw._rng)
        new._tw = tw
        new.action_log = [[dict(c) for c in step] for step in self.action_log]
        new._state = copy.deepcopy(self._state)
        return new

    def observe(self) -> dict:
        return self._state if self._state is not None else self.reset()

    @classmethod
    def from_log(cls, scenario: str, seed: int, config, action_log: list[list[dict]]) -> "StandinReplayTwin":
        tw = cls(scenario, seed, config)
        tw.reset()
        for change in action_log:
            tw.apply(change)
        return tw


def kpis(state: dict) -> dict:
    """Network KPIs of a stand-in state (W2's ``twin.replay.kpis`` has the same output keys)."""
    g = state["extra"]["standin_global"]
    return {
        "coverage_ratio": float(state["coverage_ratio"]),
        "sla_ratio": float(state["sla_ratio"]),
        "sinr_p10_db": g["sinr_p10_db"],
        "sinr_mean_db": g["sinr_mean_db"],
        "rsrp_p10_dbm": g["rsrp_p10_dbm"],
        "per_cell": [{"cell": i, "ue_count": state["ue_count"][i], "power_dbm": state["power_dbm"][i],
                      "tilt_deg": state["tilt_deg"][i], "sinr_p10_db": state["sinr_p10_db"][i]}
                     for i in range(len(state["power_dbm"]))],
    }
