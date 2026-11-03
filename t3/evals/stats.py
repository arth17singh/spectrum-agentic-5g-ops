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


METHOD = "paired-bootstrap-v1"  # the name tools/check_unit.py replays; the algorithm is paired_bootstrap above


def ci_spec(boot: dict[str, Any], baseline_run_id: str | None, baseline_model_id: str | None,
            score_field: str = "score", subset: dict[str, Any] | None = None) -> dict[str, Any]:
    """Everything needed to replay an interval from the item records (check_unit part e).

    ``boot`` is targets.json's bootstrap block (seed, resamples, ci). Pairs are item ids; each pair is
    its own cluster; ``subset`` filters this run's items ({"field": dotted path, "equals": value})."""
    return {
        "method": METHOD,
        "seed": int(boot["seed"]),
        "resamples": int(boot["resamples"]),
        "level": float(boot["ci"]),
        "pair_key": "item_id",
        "score_field": score_field,
        "baseline_run_id": baseline_run_id,
        "baseline_model_id": baseline_model_id,
        "cluster_field": None,
        "subset": subset,
        "numpy_version": np.__version__,
    }
