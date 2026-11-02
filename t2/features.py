"""Feature layout for the Track 2 network-response model (data contract v0.2).

One row = one 4-cell cluster x one twin step. Pure numpy; imported by the generator,
the trainer, the evaluator and the Lambda-safe predictor.

Inputs (38 numbers, field-major, cell order = the cluster's cell order):
    7 per-cell fields x 4 cells     user_share, rsrp_mean_dbm, rsrp_p10_dbm,
                                    sinr_mean_db, sinr_p10_db, power_dbm, tilt_deg
    2 cluster fields                coverage_ratio, sla_ratio  (the state the change starts from)
    2 change fields x 4 cells       d_power_db, d_tilt_deg     (the safety-filtered change)
Labels (cluster, state after the change is applied and the twin advances one step):
    coverage_ratio, sla_ratio, sinr_p10_db, sinr_mean_db

NEVER inputs (contract v0.2: leaks, or absent on Viavi): hour, phase, alarm, asleep, extra,
absolute user count, seed, scenario, silo id, generator settings.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

import numpy as np

CONTRACT_VERSION = "0.2"
N_CELLS = 4

CELL_FIELDS = (
    "user_share",
    "rsrp_mean_dbm",
    "rsrp_p10_dbm",
    "sinr_mean_db",
    "sinr_p10_db",
    "power_dbm",
    "tilt_deg",
)
CLUSTER_FIELDS = ("coverage_ratio", "sla_ratio")
CHANGE_FIELDS = ("d_power_db", "d_tilt_deg")
LABELS = ("coverage_ratio", "sla_ratio", "sinr_p10_db", "sinr_mean_db")

FORBIDDEN_INPUTS = frozenset(
    {
        "hour",
        "phase",
        "alarm",
        "alarms",
        "asleep",
        "extra",
        "ue_count",
        "seed",
        "scenario",
        "silo",
        "silo_id",
        "run_id",
        "step",
        "cluster",
        "backend",
        "start_power",
        "start_tilt",
        "n_ues",
    }
)

# Fixed contract ranges (the SDK's observation and safety ranges, the same in every silo).
RANGES = {
    "user_share": (0.0, 1.0),
    "rsrp_mean_dbm": (-140.0, -40.0),
    "rsrp_p10_dbm": (-140.0, -40.0),
    "sinr_mean_db": (-30.0, 40.0),
    "sinr_p10_db": (-30.0, 40.0),
    "power_dbm": (10.0, 46.0),
    "tilt_deg": (4.0, 15.0),
    "coverage_ratio": (0.0, 1.0),
    "sla_ratio": (0.0, 1.0),
    "d_power_db": (-8.0, 8.0),  # symmetric so that 0 stays 0; the real limit is -6..+8
    "d_tilt_deg": (-5.0, 5.0),
}
POWER_DELTA = (-6.0, 8.0)
TILT_DELTA = (-5.0, 5.0)
POWER_ABS = (10.0, 46.0)
TILT_ABS = (4.0, 15.0)
LABEL_RANGES = {
    "coverage_ratio": (0.0, 1.0),
    "sla_ratio": (0.0, 1.0),
    "sinr_p10_db": (-30.0, 40.0),
    "sinr_mean_db": (-30.0, 40.0),
}


def feature_names() -> list[str]:
    names = [f"{f}[{i}]" for f in CELL_FIELDS for i in range(N_CELLS)]
    names += list(CLUSTER_FIELDS)
    names += [f"{f}[{i}]" for f in CHANGE_FIELDS for i in range(N_CELLS)]
    return names


FEATURE_NAMES = feature_names()
N_FEATURES = len(FEATURE_NAMES)  # 38
_COL = {n: i for i, n in enumerate(FEATURE_NAMES)}


def col(field: str, cell: int | None = None) -> int:
    return _COL[field if cell is None else f"{field}[{cell}]"]


def _scale_vectors() -> tuple[np.ndarray, np.ndarray]:
    lo = np.zeros(N_FEATURES)
    hi = np.ones(N_FEATURES)
    for name, i in _COL.items():
        lo[i], hi[i] = RANGES[name.split("[")[0]]
    return lo, hi


_LO, _HI = _scale_vectors()


def encode(x_raw: np.ndarray) -> np.ndarray:
    """Raw feature rows -> [-1, 1] (fixed contract ranges; no data-dependent statistics)."""
    x = np.asarray(x_raw, dtype=np.float64)
    return 2.0 * (x - _LO) / (_HI - _LO) - 1.0


def cell_index(cell: Any) -> int:
    """Cell id of a change item -> 0..3. Accepts 2, "2", "cell-2" (last integer in the text)."""
    if isinstance(cell, (int, np.integer)):
        idx = int(cell)
    else:
        found = re.findall(r"\d+", str(cell))
        if not found:
            raise ValueError(f"cannot read a cell index from {cell!r}")
        idx = int(found[-1])
    if not 0 <= idx < N_CELLS:
        raise ValueError(f"cell index {idx} outside 0..{N_CELLS - 1}")
    return idx


def change_to_arrays(change: Sequence[Mapping[str, Any]] | None) -> tuple[np.ndarray, np.ndarray]:
    """[{cell, d_power_db, d_tilt_deg}] -> (dpower[4], dtilt[4]); repeated cells add up."""
    dp, dt = np.zeros(N_CELLS), np.zeros(N_CELLS)
    for item in change or []:
        i = cell_index(item["cell"])
        dp[i] += float(item.get("d_power_db", 0.0) or 0.0)
        dt[i] += float(item.get("d_tilt_deg", 0.0) or 0.0)
    return dp, dt


def arrays_to_change(dp: np.ndarray, dt: np.ndarray) -> list[dict[str, float]]:
    """(dpower[4], dtilt[4]) -> the contract's change list, nonzero cells only."""
    return [
        {"cell": i, "d_power_db": float(dp[i]), "d_tilt_deg": float(dt[i])}
        for i in range(len(dp))
        if dp[i] != 0.0 or dt[i] != 0.0
    ]


def _cell_values(state: Mapping[str, Any], key: str) -> np.ndarray:
    v = np.asarray(state[key], dtype=np.float64)
    if v.shape != (N_CELLS,):
        raise ValueError(f"state[{key!r}] must have {N_CELLS} values, got shape {v.shape}")
    return v


def user_share(state: Mapping[str, Any]) -> np.ndarray:
    """Per-cell share of users. From `user_share`, else from `ue_count` (counts are never an input)."""
    if "user_share" in state:
        s = _cell_values(state, "user_share")
    else:
        c = _cell_values(state, "ue_count")
        s = c / c.sum() if c.sum() > 0 else np.full(N_CELLS, 1.0 / N_CELLS)
    return s


def clip_change(
    dp: np.ndarray, dt: np.ndarray, power: np.ndarray, tilt: np.ndarray, alarm: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """The SDK safety filter's clipping, in numpy: step limits, absolute bounds, alarm freeze."""
    dp = np.clip(np.nan_to_num(dp, nan=0.0, posinf=0.0, neginf=0.0), *POWER_DELTA)
    dt = np.clip(np.nan_to_num(dt, nan=0.0, posinf=0.0, neginf=0.0), *TILT_DELTA)
    dp = np.clip(dp, np.clip(POWER_ABS[0] - power, *POWER_DELTA), np.clip(POWER_ABS[1] - power, *POWER_DELTA))
    dt = np.clip(dt, np.clip(TILT_ABS[0] - tilt, *TILT_DELTA), np.clip(TILT_ABS[1] - tilt, *TILT_DELTA))
    if alarm is not None:
        a = np.asarray(alarm, dtype=bool)
        dp, dt = np.where(a, 0.0, dp), np.where(a, 0.0, dt)
    return dp, dt


def state_change_to_row(
    state: Mapping[str, Any], change: Sequence[Mapping[str, Any]] | None, *, apply_safety: bool = True
) -> np.ndarray:
    """One (state, change) -> a raw 38-feature row. `state` is the SDK twin state dict
    (power_dbm, tilt_deg, ue_count, rsrp_*, sinr_*, coverage_ratio, sla_ratio) or the same
    with `user_share` instead of `ue_count`. Forbidden fields in `state` are ignored."""
    row = np.zeros(N_FEATURES)
    row[[col("user_share", i) for i in range(N_CELLS)]] = user_share(state)
    for f in CELL_FIELDS[1:]:
        for i, v in enumerate(_cell_values(state, f)):
            row[col(f, i)] = v
    row[col("coverage_ratio")] = float(state["coverage_ratio"])
    row[col("sla_ratio")] = float(state["sla_ratio"])
    dp, dt = change_to_arrays(change)
    if apply_safety:
        dp, dt = clip_change(dp, dt, _cell_values(state, "power_dbm"), _cell_values(state, "tilt_deg"), state.get("alarm"))
    for i in range(N_CELLS):
        row[col("d_power_db", i)] = dp[i]
        row[col("d_tilt_deg", i)] = dt[i]
    return row
