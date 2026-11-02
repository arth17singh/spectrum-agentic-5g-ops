"""Shared change handling: validation, the SDK safety filter, and JSON-safe log entries.

Used by ReplayTwin (in-process) and AdapterClient (hosted twins) so both treat a change identically,
and exactly as ran_rl's RANOptimizationEnv.step does: raw per-cell deltas go through
``ActionSafetyFilter.apply(raw, state)`` against the current state, and the filtered vectors are
what the twin receives.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from twin.kpis import headline


def enc(x: float) -> float | str:
    """JSON-safe float: non-finite values become the strings "nan", "inf", "-inf"."""
    x = float(x)
    if math.isfinite(x):
        return x
    return "nan" if math.isnan(x) else ("inf" if x > 0 else "-inf")


def dec(x: float | str) -> float:
    return float(x)  # float("nan"), float("inf"), float("-inf") all parse


def normalise_change(change: list[dict[str, Any]], n_cells: int) -> list[dict[str, Any]]:
    """Validate a change and return it as ``[{cell, d_power_db, d_tilt_deg}]`` with plain types.

    Missing deltas default to 0. Two items for the same cell add up (the same as one item with the sum).
    """
    if not isinstance(change, (list, tuple)):
        raise TypeError("change must be a list of {cell, d_power_db, d_tilt_deg}")
    out = []
    for item in change:
        if not isinstance(item, dict) or "cell" not in item:
            raise ValueError(f"each change item needs a 'cell': {item!r}")
        extra = set(item) - {"cell", "d_power_db", "d_tilt_deg"}
        if extra:
            raise ValueError(f"unknown change keys {sorted(extra)} in {item!r}")
        cell = item["cell"]
        if isinstance(cell, bool) or not isinstance(cell, (int, np.integer)):
            raise ValueError(f"cell must be an integer index, got {cell!r}")
        if not 0 <= int(cell) < n_cells:
            raise ValueError(f"cell {int(cell)} out of range 0..{n_cells - 1}")
        out.append(
            {
                "cell": int(cell),
                "d_power_db": enc(item.get("d_power_db", 0.0)),
                "d_tilt_deg": enc(item.get("d_tilt_deg", 0.0)),
            }
        )
    return out


def to_vectors(change: list[dict[str, Any]], n_cells: int) -> np.ndarray:
    """Raw action vector (n power deltas then n tilt deltas) from a normalised change."""
    raw = np.zeros(2 * n_cells)
    for item in change:
        raw[item["cell"]] += dec(item["d_power_db"])
        raw[n_cells + item["cell"]] += dec(item["d_tilt_deg"])
    return raw


def filter_change(
    change: list[dict[str, Any]], state: dict[str, Any], safety: Any, step: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run ``change`` through the SDK safety filter against ``state``.

    Returns ``(twin_action, entry)``: the action dict for the twin (as the env builds it) and an
    action_log entry without the post-step ``kpis`` (the caller adds it after advancing).
    """
    n = len(state["power_dbm"])
    norm = normalise_change(change, n)
    raw = to_vectors(norm, n)
    result = safety.apply(raw, state)
    filtered = result.filtered
    twin_action = {
        "power_delta_db": filtered[:n].tolist(),
        "tilt_delta_deg": filtered[n : 2 * n].tolist(),
    }
    requested = {
        "power_delta_db": [enc(v) for v in raw[:n]],
        "tilt_delta_deg": [enc(v) for v in raw[n:]],
    }
    entry = {
        "step": step,
        "change": norm,
        "requested": requested,
        "filtered": {k: list(map(float, v)) for k, v in twin_action.items()},
        "rules": list(result.rule_ids),
        "clamped": bool(result.rule_ids),  # every altered value records a rule id
    }
    return twin_action, entry


def with_kpis(entry: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    entry["kpis"] = {k: enc(v) for k, v in headline(state).items()}
    return entry
