"""Twin configurations.

A resolved config is a plain JSON-safe dict (it is stored per case next to the action log):

    {
      "preset":          label of the preset it was built from (informational),
      "twin":            fields of ran_rl MavericTwinConfig
                         (n_ues, area_margin_m, initial_power_dbm, initial_tilt_deg,
                          coverage_rsrp_threshold_dbm, sla_sinr_threshold_db),
      "safety":          fields of ran_rl SafetyBounds (the SDK safety filter),
      "step_seconds":    simulated seconds per step, what the SDK env passes to advance(),
      "noise_floor_dbm": None (twin default) or a number, passed to reset(),
      "scenarios":       {name: fields of ran_rl Scenario} defined or overridden on top of the
                         SDK's scenarios.yaml (needed because a Scenario cannot be named by
                         the config dataclass alone),
      "models_dir":      None (the SDK's shipped sim_tilt_sweep_4cell GP models) or a path,
      "topology":        None (the shipped topology csv) or a path,
    }

``resolve_config(cfg)`` accepts None (the "official" preset), a preset name, or a dict that may
carry ``"preset": <name>`` plus partial overrides (deep-merged over the preset). Unknown keys are
errors (the SDK dataclasses refuse them). Resolving a resolved config returns it unchanged.

Presets
-------
"official"
    The SDK's Maveric training defaults: ``MavericTwinConfig()`` (200 UEs, 23 dBm, 6 degrees,
    -97 dBm coverage, 0 dB SLA), ``SafetyBounds()`` and ``EnvConfig().step_seconds`` (3600 s),
    scenarios from the SDK's scenarios.yaml (nominal, congestion, interference, outage, ...).

"viavi-like"
    Mirrors the organisers' Viavi scenario ``ran-rl-maveric-4cell`` (config/viavi_scenarios/
    maveric_4cell.json and config/viavi_rsg.yaml in the twin-host source): the same four sites
    (the shipped topology), 60 UEs (3 groups x 20), UEs that do not move (speed 0), every cell at
    40 dBm and 6 degrees, 60 s per step, 5-step episodes, -97 dBm / 0 dB thresholds. It defines
    the scenario ``viavi_stationary``; use that name (``VIAVI_SCENARIO``). Other scenario names
    keep the SDK meaning (e.g. the SDK "nominal" is a mobile scenario) but with this preset's UE
    count and power.

    What the SDK cannot express, so this is "like" and not "equal": Viavi places UEs in three
    named areas (a1..a3), the Maveric twin scatters them uniformly over the site bounding box plus
    300 m with the seeded generator; Viavi measures RSRP/SINR with its own radio model (Cosine-Urban
    ARF, per-beam interference) while Maveric predicts them with per-cell Gaussian processes
    trained at a 23 dBm reference (a 40 dBm cell is the GP plus a 17 dB offset, outside the
    reference the models were trained at); Viavi's seed is ignored, Maveric's is not.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

OFFICIAL = "official"
VIAVI_LIKE = "viavi-like"
VIAVI_SCENARIO = "viavi_stationary"

_STEP_SECONDS = 3600.0


def _sdk_defaults() -> dict[str, Any]:
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
    cfg = _sdk_defaults()
    cfg["preset"] = VIAVI_LIKE
    cfg["twin"].update(n_ues=60, initial_power_dbm=40.0, initial_tilt_deg=6.0)
    cfg["step_seconds"] = 60.0
    # alpha=1, velocity 0: the Gauss-Markov walk keeps speed 0 and adds no noise; no resampling.
    cfg["scenarios"] = {
        VIAVI_SCENARIO: {
            "mobility": {"alpha": 1.0, "velocity_m": 0.0},
            "ue_resample_fraction": 0.0,
            "length_steps": 5,
        }
    }
    return cfg


PRESETS: dict[str, Any] = {OFFICIAL: _sdk_defaults, VIAVI_LIKE: _viavi_like}


def preset(name: str) -> dict[str, Any]:
    """A fresh resolved config for the named preset."""
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; known: {sorted(PRESETS)}")
    return PRESETS[name]()


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict) and key != "scenarios":
            out[key] = _merge(out[key], value)
        elif isinstance(value, dict) and key == "scenarios":
            out[key] = {**out.get(key, {}), **copy.deepcopy(value)}
        else:
            out[key] = copy.deepcopy(value)
    return out


def resolve_config(config: dict[str, Any] | str | None = None) -> dict[str, Any]:
    """Resolve ``config`` to the full, validated, JSON-safe dict described in the module docstring."""
    from ran_rl.env.config import SafetyBounds
    from ran_rl.twin.maveric.twin import MavericTwinConfig
    from ran_rl.twin.scenarios import Scenario

    if config is None:
        config = {}
    elif isinstance(config, str):
        config = {"preset": config}
    elif not isinstance(config, dict):
        raise TypeError(f"config must be None, a preset name or a dict, got {type(config).__name__}")
    base = preset(config.get("preset", OFFICIAL))
    overrides = {k: v for k, v in config.items() if k != "preset"}
    unknown = set(overrides) - set(base)
    if unknown:
        raise ValueError(f"unknown config keys {sorted(unknown)}; known: {sorted(base)}")
    cfg = _merge(base, overrides)
    # Validate with the SDK's own models (StrictModel: unknown keys are errors), then dump them
    # back so defaults are explicit and the result is stable.
    cfg["twin"] = MavericTwinConfig(**cfg["twin"]).model_dump()
    cfg["safety"] = SafetyBounds(**cfg["safety"]).model_dump()
    cfg["scenarios"] = {
        name: Scenario(**fields).model_dump() for name, fields in cfg["scenarios"].items()
    }
    cfg["step_seconds"] = float(cfg["step_seconds"])
    if cfg["step_seconds"] <= 0:
        raise ValueError("step_seconds must be positive")
    if cfg["noise_floor_dbm"] is not None:
        cfg["noise_floor_dbm"] = float(cfg["noise_floor_dbm"])
    if (cfg["models_dir"] is None) != (cfg["topology"] is None):
        raise ValueError("models_dir and topology go together: both or neither")
    json.dumps(cfg)  # must be JSON-safe
    return cfg


def config_hash(resolved: dict[str, Any]) -> str:
    """sha256 of the canonical JSON of a resolved config (a stable identity for evidence)."""
    blob = json.dumps(resolved, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()
