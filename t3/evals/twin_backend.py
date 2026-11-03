"""Which twin the evals run on: W2's ``twin.replay`` when importable, else the labelled stand-in.

Everything in ``t3/evals`` imports ``ReplayTwin``, ``kpis`` and ``headline`` from here, so the swap at
merge is automatic. ``BACKEND`` names the one in use and is written into every item record.
"""

from __future__ import annotations

import importlib.metadata
import math
from typing import Any

try:  # stream W2 (merged)
    from twin.kpis import headline, kpis  # type: ignore[import-not-found]
    from twin.replay import ReplayTwin  # type: ignore[import-not-found]

    BACKEND = "maveric-inprocess"
except ImportError:  # not merged yet: same interface, see twin_standin.py
    from t3.evals.twin_standin import ReplayTwin, headline, kpis  # noqa: F401

    BACKEND = "maveric-inprocess-standin"

VIAVI_SCENARIO = "viavi_stationary"
PRESET_SCENARIO = {"official": "nominal", "viavi-like": VIAVI_SCENARIO}
GRADED_KPIS = ("coverage_ratio", "sla_ratio", "sinr_p10_db")


def sdk_version() -> str:
    return importlib.metadata.version("ran-rl")


def twin_version() -> str:
    return f"ran-rl-{sdk_version()}"


def graded(state: dict[str, Any]) -> dict[str, float]:
    """The KPIs the graders read (plus sinr_mean_db for information), all finite floats."""
    h = headline(state)
    out = {k: float(h[k]) for k in (*GRADED_KPIS, "sinr_mean_db")}
    for k, v in out.items():
        if not math.isfinite(v):
            raise ValueError(f"twin KPI {k} is not finite ({v}); no cell serves any UE")
    return out


def replay(scenario: str, seed: int, config: Any, changes: list[list[dict[str, Any]]]) -> Any:
    """A fresh twin, reset, with ``changes`` applied one step each (requested deltas)."""
    t = ReplayTwin(scenario, seed, config)
    t.reset()
    for ch in changes:
        t.apply(ch)
    return t


def requested_changes(twin: Any) -> list[list[dict[str, Any]]]:
    return [entry["change"] for entry in twin.action_log]
