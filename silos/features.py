"""Row extraction for the Track 2 operator silos (data contract v0.2 + amendment 1).

One row = one 4-cell cluster x one twin step. The column layout is IDENTICAL to the pooled dataset
in ``t2/features.py`` (builder W9) so the two can be unified at merge:

    inputs (38, field-major, cell order = the cluster's cell order)
        7 per-cell fields x 4 cells   user_share, rsrp_mean_dbm, rsrp_p10_dbm, sinr_mean_db,
                                      sinr_p10_db, power_dbm, tilt_deg           -> ``field[i]``
        2 cluster fields              coverage_ratio, sla_ratio (state the change starts from)
        2 change fields x 4 cells     d_power_db, d_tilt_deg (safety-filtered)   -> ``field[i]``
    labels (amendment 1, A5): DELTAS against the no-op counterfactual at the same step
        d_coverage_ratio, d_sla_ratio, d_sinr_p10_db, d_sinr_mean_db
        = KPIs(after the change, advanced one step) - KPIs(same step, no change)

Never inputs: hour, phase, alarm, asleep, extra, absolute user counts, seed, scenario, silo id,
generator settings. Cluster KPIs are defined from the state dict alone (the same definition as
``t2/kpis.py``): coverage and SLA as reported; sinr_mean = user-share weighted mean of the per-cell
means (exact); sinr_p10 = user-share weighted mean of the per-cell 10th percentiles (an
APPROXIMATION of the pooled 10th percentile, which no twin state carries).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

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
KPI_NAMES = ("coverage_ratio", "sla_ratio", "sinr_p10_db", "sinr_mean_db")
LABELS = tuple(f"d_{k}" for k in KPI_NAMES)
BASE_KPIS = tuple(f"base_{k}" for k in KPI_NAMES)  # no-op next-step KPI levels (metadata)

# Metadata columns: recorded for every row, never model inputs.
METADATA = (
    "run_id",
    "segment",
    "silo",
    "cluster",
    "seed",
    "step",
    "episode",
    "n_ues",
    "moving",
    "noise_floor_dbm",
    "start_power_dbm",
    "start_tilt_deg",
    "change_kind",
    "clipped",
    "source",
    "twin_version",
    "backend",
    "contract_version",
    "generator_commit",
)


def _names(fields: Sequence[str]) -> list[str]:
    return [f"{f}[{i}]" for f in fields for i in range(N_CELLS)]


INPUT_COLUMNS: list[str] = _names(CELL_FIELDS) + list(CLUSTER_FIELDS) + _names(CHANGE_FIELDS)
N_INPUTS = len(INPUT_COLUMNS)  # 38
COLUMNS: list[str] = INPUT_COLUMNS + list(LABELS) + list(BASE_KPIS) + list(METADATA)
_COL = {n: i for i, n in enumerate(INPUT_COLUMNS)}

# Names that must never be an input column (and hour/phase/alarm/asleep/extra not anywhere).
FORBIDDEN_INPUTS = frozenset(
    {
        "hour", "phase", "alarm", "alarms", "asleep", "extra", "ue_count", "n_ues", "seed",
        "scenario", "silo", "silo_id", "run_id", "step", "cluster", "backend", "start_power",
        "start_tilt", "noise_floor_dbm", "moving",
    }
)
FORBIDDEN_ANYWHERE = frozenset({"hour", "phase", "alarm", "alarms", "asleep", "extra", "ue_count"})

# Fixed contract ranges: the SDK's observation and safety ranges, the same in every silo.
RANGES: dict[str, tuple[float, float]] = {
    "user_share": (0.0, 1.0),
    "rsrp_mean_dbm": (-140.0, -40.0),
    "rsrp_p10_dbm": (-140.0, -40.0),
    "sinr_mean_db": (-30.0, 40.0),
    "sinr_p10_db": (-30.0, 40.0),
    "power_dbm": (10.0, 46.0),
    "tilt_deg": (4.0, 15.0),
    "coverage_ratio": (0.0, 1.0),
    "sla_ratio": (0.0, 1.0),
    "d_power_db": (-6.0, 8.0),  # the SDK safety filter's per-step limit
    "d_tilt_deg": (-5.0, 5.0),
}
# A delta label is a difference of two in-range KPIs, so it is within +-(range width).
LABEL_RANGES: dict[str, tuple[float, float]] = {
    "d_coverage_ratio": (-1.0, 1.0),
    "d_sla_ratio": (-1.0, 1.0),
    "d_sinr_p10_db": (-70.0, 70.0),
    "d_sinr_mean_db": (-70.0, 70.0),
}
POWER_DELTA = RANGES["d_power_db"]
TILT_DELTA = RANGES["d_tilt_deg"]
POWER_ABS = RANGES["power_dbm"]
TILT_ABS = RANGES["tilt_deg"]


_LO = np.array([RANGES[c.split("[")[0]][0] for c in INPUT_COLUMNS])
_HI = np.array([RANGES[c.split("[")[0]][1] for c in INPUT_COLUMNS])


def col(field: str, cell: int | None = None) -> int:
    """Index of an input column."""
    return _COL[field if cell is None else f"{field}[{cell}]"]


def input_range(column: str) -> tuple[float, float]:
    return RANGES[column.split("[")[0]]


def _cells(state: Mapping[str, Any], key: str) -> np.ndarray:
    v = np.asarray(state[key], dtype=np.float64)
    if v.shape != (N_CELLS,):
        raise ValueError(f"state[{key!r}] needs {N_CELLS} values, got shape {v.shape}")
    return v


def user_share(state: Mapping[str, Any]) -> np.ndarray:
    """Per-cell share of users, from ``user_share`` or from ``ue_count`` (counts are never an input)."""
    if "user_share" in state:
        return _cells(state, "user_share")
    c = _cells(state, "ue_count")
    return c / c.sum() if c.sum() > 0 else np.full(N_CELLS, 1.0 / N_CELLS)


def cluster_kpis(state: Mapping[str, Any]) -> np.ndarray:
    """The four cluster KPIs (order KPI_NAMES) from a state dict alone."""
    w = user_share(state)
    return np.array(
        [
            float(state["coverage_ratio"]),
            float(state["sla_ratio"]),
            float(np.dot(w, _cells(state, "sinr_p10_db"))),
            float(np.dot(w, _cells(state, "sinr_mean_db"))),
        ]
    )


def state_change_to_inputs(
    state: Mapping[str, Any], d_power: Sequence[float], d_tilt: Sequence[float]
) -> np.ndarray:
    """One (state, filtered change) -> the 38 raw input values, in INPUT_COLUMNS order.

    Anything else in ``state`` (hour, phase, alarm, asleep, extra, ue_count, ...) is ignored.
    A reported value outside its fixed contract range is clipped to the range, as the SDK's own
    observation normalisation does (the twin can report a per-cell SINR p10 below -30 dB at a low
    power with a high noise floor). Labels are computed from the unclipped state.
    """
    row = np.zeros(N_INPUTS)
    for i, v in enumerate(user_share(state)):
        row[col("user_share", i)] = v
    for f in CELL_FIELDS[1:]:
        for i, v in enumerate(_cells(state, f)):
            row[col(f, i)] = v
    row[col("coverage_ratio")] = float(state["coverage_ratio"])
    row[col("sla_ratio")] = float(state["sla_ratio"])
    dp, dt = np.asarray(d_power, dtype=float), np.asarray(d_tilt, dtype=float)
    if dp.shape != (N_CELLS,) or dt.shape != (N_CELLS,):
        raise ValueError("a change needs 4 power and 4 tilt deltas")
    for i in range(N_CELLS):
        row[col("d_power_db", i)] = dp[i]
        row[col("d_tilt_deg", i)] = dt[i]
    return np.clip(row, _LO, _HI)


def delta_labels(after_change: Mapping[str, Any], after_noop: Mapping[str, Any]) -> np.ndarray:
    """Amendment 1 labels: KPIs(after the change) - KPIs(same step, no change)."""
    return cluster_kpis(after_change) - cluster_kpis(after_noop)


def check_input_ranges(x: np.ndarray, tol: float = 1e-9) -> list[str]:
    """Names of input columns whose values fall outside the fixed contract ranges (or are not finite)."""
    x = np.atleast_2d(x)
    bad = []
    for j, name in enumerate(INPUT_COLUMNS):
        lo, hi = input_range(name)
        v = x[:, j]
        if not np.isfinite(v).all() or v.min() < lo - tol or v.max() > hi + tol:
            bad.append(name)
    return bad
