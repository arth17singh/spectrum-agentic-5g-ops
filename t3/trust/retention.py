"""Retention: how much of a model's gain survives the move up the ladder.

    retention = gain on rung 2 / gain on rung 1

Rule (guards): retention is **None** (undefined, reported as such, never as 0 or 1) when the
rung-1 gain is missing, not finite, or not above ``min_gain`` (default 1e-3 in objective units).
A ratio over a negative or near-zero denominator flips sign or explodes, so it says nothing
about transfer: a model that did not gain on the basic simulator has nothing to retain. A
missing or non-finite rung-2 gain also gives None. Otherwise the raw ratio is returned
unclipped: below 0 means the gain reversed on rung 2, above 1 means it grew.

``bootstrap_retention`` resamples cases with replacement, jointly across rungs (the gains are
paired: the same case seeds and scripted changes on both rungs), and returns the 2.5/97.5
percentiles of the ratio of mean gains. Resamples whose rung-1 mean fails the guard are
dropped and counted; if more than 5 % fail, the interval is None (too unstable to report).
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

MIN_GAIN = 1e-3


def retention(gain_rung1: float | None, gain_rung2: float | None, *, min_gain: float = MIN_GAIN) -> float | None:
    if gain_rung1 is None or gain_rung2 is None:
        return None
    g1, g2 = float(gain_rung1), float(gain_rung2)
    if not (math.isfinite(g1) and math.isfinite(g2)):
        return None
    if g1 <= min_gain:
        return None
    return g2 / g1


def bootstrap_mean_ci(values: Sequence[float], *, n_boot: int = 2000, seed: int = 0,
                      level: float = 0.95) -> tuple[float, float] | None:
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    lo, hi = np.percentile(means, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def bootstrap_retention(gains_rung1: Sequence[float], gains_rung2: Sequence[float], *,
                        n_boot: int = 2000, seed: int = 0, level: float = 0.95,
                        min_gain: float = MIN_GAIN, max_fail_frac: float = 0.05) -> dict:
    g1 = np.asarray(gains_rung1, dtype=float)
    g2 = np.asarray(gains_rung2, dtype=float)
    if g1.shape != g2.shape or g1.ndim != 1:
        raise ValueError("paired per-case gains: both rungs need the same number of cases")
    point = retention(float(g1.mean()) if g1.size else None,
                      float(g2.mean()) if g2.size else None, min_gain=min_gain)
    out = {"point": point, "ci": None, "level": level, "n_cases": int(g1.size), "n_boot": n_boot,
           "failed_resamples": 0, "rule": f"None unless mean rung-1 gain > {min_gain}"}
    if g1.size == 0 or point is None:
        return out
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, g1.size, size=(n_boot, g1.size))
    m1, m2 = g1[idx].mean(axis=1), g2[idx].mean(axis=1)
    ok = m1 > min_gain
    out["failed_resamples"] = int((~ok).sum())
    if (~ok).mean() > max_fail_frac:
        return out
    ratios = m2[ok] / m1[ok]
    lo, hi = np.percentile(ratios, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    out["ci"] = [float(lo), float(hi)]
    return out


def wilson(k: int, n: int, z: float = 1.959964) -> list[float] | None:
    """Wilson score 95 % interval for k successes in n trials (as [low, high]); None when n == 0."""
    if n <= 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [max(0.0, centre - half), min(1.0, centre + half)]
