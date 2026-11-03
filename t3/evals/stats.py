"""Paired gain with a percentile bootstrap interval (stdlib + numpy, deterministic)."""

from __future__ import annotations

from typing import Any

import numpy as np


def paired_bootstrap(
    candidate: dict[str, float],
    baseline: dict[str, float],
    resamples: int = 10000,
    seed: int = 20261102,
    ci: float = 0.95,
) -> dict[str, Any] | None:
    """Mean of (candidate - baseline) over the item ids both have, with a percentile bootstrap CI.

    ``candidate`` and ``baseline`` map item_id -> score. Returns None if no item is shared.
    """
    ids = sorted(set(candidate) & set(baseline))
    if not ids:
        return None
    d = np.array([candidate[i] - baseline[i] for i in ids], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(resamples, len(d)))
    means = d[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return {
        "value": float(d.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "paired_n": len(ids),
        "excludes_zero": bool(lo > 0 or hi < 0),
    }
