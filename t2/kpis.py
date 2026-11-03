"""Cluster KPIs from a twin state dict. THE definition both the labels and the trust gate use.

The SDK state has per-cell SINR statistics but only cluster-level coverage_ratio and
sla_ratio. The four response-model labels are therefore defined from the state dict alone:

    coverage_ratio  state["coverage_ratio"]                       (as reported by the twin)
    sla_ratio       state["sla_ratio"]                            (as reported by the twin)
    sinr_mean_db    sum_i share_i * sinr_mean_db[i]               (exact cluster mean)
    sinr_p10_db     sum_i share_i * sinr_p10_db[i]                (APPROXIMATION: the weighted
                    mean of the per-cell 10th percentiles, not the pooled 10th percentile,
                    which no twin state carries)

share_i = ue_count[i] / sum(ue_count), or `user_share` when given. Cells with no users have
weight 0 (their floor values -140 dBm / -30 dB do not count).

Any other component that computes "measured" KPIs for a change record (the replay twin's
`kpis()`) MUST use this function or the same formulas, otherwise predicted and measured are not
comparable.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from t2.features import LABELS, N_CELLS, user_share


def cluster_kpis(state: Mapping[str, Any]) -> dict[str, float]:
    w = user_share(state)
    sm = np.asarray(state["sinr_mean_db"], dtype=np.float64)
    sp = np.asarray(state["sinr_p10_db"], dtype=np.float64)
    if sm.shape != (N_CELLS,) or sp.shape != (N_CELLS,):
        raise ValueError("expected 4 cells")
    return {
        "coverage_ratio": float(state["coverage_ratio"]),
        "sla_ratio": float(state["sla_ratio"]),
        "sinr_p10_db": float(np.dot(w, sp)),
        "sinr_mean_db": float(np.dot(w, sm)),
    }


def kpis_vector(state: Mapping[str, Any]) -> np.ndarray:
    k = cluster_kpis(state)
    return np.array([k[n] for n in LABELS])
