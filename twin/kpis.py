"""Headline KPIs from a twin state dict (the SDK's own keys; see ran_rl.twin.base).

Returned dict:

    coverage_ratio   state["coverage_ratio"]   (twin field: share of UEs with serving RSRP above the
                                                coverage threshold)
    sla_ratio        state["sla_ratio"]        (twin field: share of UEs with SINR above the SLA threshold)
    sinr_mean_db     UE-count-weighted mean of the per-cell sinr_mean_db. Exact: the mean over all
                     UEs equals the count-weighted mean of the per-cell means.
    sinr_p10_db      ESTIMATE of the pooled 10th percentile of SINR over all UEs (see below)
    rsrp_p10_dbm     ESTIMATE of the pooled 10th percentile of serving RSRP (same method)
    per_cell         one dict per cell: the twin's own per-cell fields
    p10_method       "gaussian-mixture"

Why the cluster p10 values are estimates: the twin reports a 10th percentile per cell, not per
UE, and a percentile of a pool cannot be computed exactly from per-group percentiles. The twin
does not provide a pooled p10. We estimate it so that it uses only the twin's own fields:

  1. Per occupied cell (ue_count > 0) take mean m and p10 p; model the cell's values as a normal
     with sigma = (m - p) / 1.2816 (z of the 10th percentile), floored at 1e-3.
  2. The pooled distribution is the ue_count-weighted mixture of those normals; solve
     F(x) = 0.10 by bisection.

Bounds: for large cells the pooled p10 lies between the smallest and largest per-cell p10
(``sinr_p10_db_bounds``, ``rsrp_p10_dbm_bounds``; the estimate is clamped to them). With very few UEs
in a cell, percentile interpolation can put the exact value slightly outside (seen: up to 0.2 dB).

Accuracy, measured against the exact pooled percentile over 64 (preset x scenario x seed) states
(tests/test_kpis.py): SINR p10 max abs error 1.1 dB, mean +0.3 dB (biased high); RSRP p10 max abs
error 5.6 dB, mean +1.2 dB. ``ReplayTwin.exact_kpis()`` gives the exact pooled numbers for the
in-process twin. Use the SAME function (this one) on predicted and measured states so that
errors are comparable; the estimate preserves direction of change well but is not a UE-level value.

Cells with no UEs report the twin's floor values (-140 dBm / -30 dB); they are excluded from the
pooled numbers and listed in per_cell with ue_count 0.
"""

from __future__ import annotations

import math
from typing import Any

_Z10 = 1.2815515655446004  # standard normal 10th percentile magnitude
_SIGMA_FLOOR = 1e-3

_PER_CELL_KEYS = (
    "power_dbm",
    "tilt_deg",
    "ue_count",
    "rsrp_mean_dbm",
    "rsrp_p10_dbm",
    "sinr_mean_db",
    "sinr_p10_db",
    "alarm",
    "asleep",
)


def _cdf(x: float, parts: list[tuple[float, float, float]]) -> float:
    return sum(w * 0.5 * (1.0 + math.erf((x - m) / (s * math.sqrt(2.0)))) for w, m, s in parts)


def _pooled_p10(counts: list[int], means: list[float], p10s: list[float]) -> tuple[float, float, float]:
    """(estimate, lower bound, upper bound) of the pooled 10th percentile."""
    occ = [(c, m, p) for c, m, p in zip(counts, means, p10s, strict=True) if c > 0]
    if not occ:
        return float("nan"), float("nan"), float("nan")
    total = float(sum(c for c, _, _ in occ))
    parts = [(c / total, m, max((m - p) / _Z10, _SIGMA_FLOOR)) for c, m, p in occ]
    lo = min(m - 10.0 * s for _, m, s in parts)
    hi = max(m + 10.0 * s for _, m, s in parts)
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        if _cdf(mid, parts) < 0.10:
            lo = mid
        else:
            hi = mid
    low_bound = min(p for _, _, p in occ)
    high_bound = max(p for _, _, p in occ)
    estimate = min(max(0.5 * (lo + hi), low_bound), high_bound)
    return estimate, low_bound, high_bound


def kpis(state: dict[str, Any]) -> dict[str, Any]:
    """Headline KPIs of one twin state. See the module docstring for how each is obtained."""
    counts = [int(c) for c in state["ue_count"]]
    total = sum(counts)
    n = len(counts)
    sinr_means = [float(v) for v in state["sinr_mean_db"]]
    sinr_mean = (
        sum(c * m for c, m in zip(counts, sinr_means, strict=True)) / total if total else float("nan")
    )
    sinr_p10, sinr_lo, sinr_hi = _pooled_p10(counts, sinr_means, [float(v) for v in state["sinr_p10_db"]])
    rsrp_p10, rsrp_lo, rsrp_hi = _pooled_p10(
        counts,
        [float(v) for v in state["rsrp_mean_dbm"]],
        [float(v) for v in state["rsrp_p10_dbm"]],
    )
    per_cell = []
    for i in range(n):
        row: dict[str, Any] = {"cell": i}
        for key in _PER_CELL_KEYS:
            if key in state:
                row[key] = state[key][i]
        per_cell.append(row)
    return {
        "coverage_ratio": float(state["coverage_ratio"]),
        "sla_ratio": float(state["sla_ratio"]),
        "sinr_p10_db": sinr_p10,
        "sinr_mean_db": sinr_mean,
        "rsrp_p10_dbm": rsrp_p10,
        "per_cell": per_cell,
        "sinr_p10_db_bounds": [sinr_lo, sinr_hi],
        "rsrp_p10_dbm_bounds": [rsrp_lo, rsrp_hi],
        "p10_method": "gaussian-mixture",
        "n_ues": total,
    }


HEADLINE = ("coverage_ratio", "sla_ratio", "sinr_p10_db", "sinr_mean_db")


def headline(state_or_kpis: dict[str, Any]) -> dict[str, float]:
    """The four numbers a ``change`` record carries (predicted and measured)."""
    k = state_or_kpis if "per_cell" in state_or_kpis else kpis(state_or_kpis)
    return {key: float(k[key]) for key in HEADLINE}
