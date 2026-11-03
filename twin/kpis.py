"""Headline KPIs from a twin state dict (the SDK's own keys; see ran_rl.twin.base).

THE definition shared with the Track 2 predictor (t2/kpis.py cluster_kpis): predicted and measured
values must come from the same formulas or the trust gate compares different quantities.

    coverage_ratio   state["coverage_ratio"]   (twin field)
    sla_ratio        state["sla_ratio"]        (twin field)
    sinr_mean_db     sum_i share_i * sinr_mean_db[i]   (exact cluster mean)
    sinr_p10_db      sum_i share_i * sinr_p10_db[i]    (APPROXIMATION: user-share-weighted mean of
                     the per-cell p10s, not the pooled 10th percentile; no twin state carries it)
    rsrp_p10_dbm     sum_i share_i * rsrp_p10_dbm[i]  (same approximation)

share_i = state["user_share"][i] if present, else ue_count[i] / sum(ue_count), else uniform when
nobody is attached. Cells with no UEs have weight 0 (their floor values do not count).

Diagnostics only, NOT for scoring: ``sinr_p10_mixture_db`` / ``rsrp_p10_mixture_dbm`` estimate the
pooled p10 by fitting a normal per occupied cell (sigma = (mean - p10) / 1.2816) and solving the
count-weighted mixture CDF = 0.10 by bisection; ``*_bounds`` are the min and max per-cell p10.
Against the exact pooled percentile (``ReplayTwin.exact_kpis()``) over 64 states the mixture had
SINR p10 max abs error 1.1 dB and RSRP 5.6 dB (tests/test_kpis.py measures the share-weighted
error too). ``per_cell`` carries the twin's per-cell fields.

Per-cell ``coverage_share`` / ``sla_share`` (for the Twin View): not in the SDK state, which has only
the network ratios, so ``kpis(state)`` adds them only when the caller passes them in
(``kpis(state, shares=...)``; ``ReplayTwin.kpis()`` does, from the twin's per-UE arrays). They are
the share of the cell's served UEs meeting the SDK's own rules, strictly greater: coverage is
serving RSRP > coverage_rsrp_threshold_dbm (-97 dBm; the SDK's coverage_ratio does NOT also
require SINR > 0), SLA is SINR > sla_sinr_threshold_db (0 dB). A cell with no UEs gets None.
Their UE-weighted mean equals the network coverage_ratio / sla_ratio. Hosted adapters give no
per-UE data, so for them the two keys are absent.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

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


def user_share(state: dict[str, Any]) -> np.ndarray:
    """Per-cell share of users: ``user_share`` if given, else ue_count / sum, else uniform."""
    if "user_share" in state:
        return np.asarray(state["user_share"], dtype=np.float64)
    c = np.asarray(state["ue_count"], dtype=np.float64)
    return c / c.sum() if c.sum() > 0 else np.full(len(c), 1.0 / len(c))


def kpis(state: dict[str, Any], shares: dict[str, list] | None = None) -> dict[str, Any]:
    """Headline KPIs of one twin state. See the module docstring for how each is obtained.

    ``shares`` = ``{"coverage_share": [...], "sla_share": [...]}`` per cell (None for empty cells)
    adds those two keys to every per_cell entry.
    """
    counts = [int(c) for c in state["ue_count"]]
    n = len(counts)
    w = user_share(state)
    sinr_means = [float(v) for v in state["sinr_mean_db"]]
    sinr_p10s = [float(v) for v in state["sinr_p10_db"]]
    rsrp_means = [float(v) for v in state["rsrp_mean_dbm"]]
    rsrp_p10s = [float(v) for v in state["rsrp_p10_dbm"]]
    mix_sinr, sinr_lo, sinr_hi = _pooled_p10(counts, sinr_means, sinr_p10s)
    mix_rsrp, rsrp_lo, rsrp_hi = _pooled_p10(counts, rsrp_means, rsrp_p10s)
    per_cell = []
    for i in range(n):
        row: dict[str, Any] = {"cell": i}
        for key in _PER_CELL_KEYS:
            if key in state:
                row[key] = state[key][i]
        if shares is not None:
            row["coverage_share"] = shares["coverage_share"][i]
            row["sla_share"] = shares["sla_share"][i]
        per_cell.append(row)
    return {
        "coverage_ratio": float(state["coverage_ratio"]),
        "sla_ratio": float(state["sla_ratio"]),
        "sinr_p10_db": float(np.dot(w, np.asarray(sinr_p10s, dtype=np.float64))),
        "sinr_mean_db": float(np.dot(w, np.asarray(sinr_means, dtype=np.float64))),
        "rsrp_p10_dbm": float(np.dot(w, np.asarray(rsrp_p10s, dtype=np.float64))),
        "per_cell": per_cell,
        "sinr_p10_mixture_db": mix_sinr,
        "rsrp_p10_mixture_dbm": mix_rsrp,
        "sinr_p10_db_bounds": [sinr_lo, sinr_hi],
        "rsrp_p10_dbm_bounds": [rsrp_lo, rsrp_hi],
        "p10_method": "user-share-weighted-mean",
        "n_ues": sum(counts),
    }


HEADLINE = ("coverage_ratio", "sla_ratio", "sinr_p10_db", "sinr_mean_db")


def headline(state_or_kpis: dict[str, Any]) -> dict[str, float]:
    """The four numbers a ``change`` record carries (predicted and measured)."""
    k = state_or_kpis if "per_cell" in state_or_kpis else kpis(state_or_kpis)
    return {key: float(k[key]) for key in HEADLINE}
