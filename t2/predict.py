"""Track 2 response model, inference side. NUMPY ONLY (runs in a Lambda without torch).

v0, local, not federated / not the official job.

    from t2.predict import load, load_degraded
    p = load("t2/artifacts/t2_response_v0.npz")
    p.predict(state, [{"cell": 1, "d_power_db": 2.0, "d_tilt_deg": -1.0}])
        -> {"coverage_ratio", "sla_ratio", "sinr_p10_db", "sinr_mean_db"}   # KPI DELTAS
    p.model_id, p.version

`state` is the SDK twin state dict (power_dbm, tilt_deg, ue_count or user_share, rsrp_mean_dbm,
rsrp_p10_dbm, sinr_mean_db, sinr_p10_db, coverage_ratio, sla_ratio; 4 cells). Other keys (hour,
phase, alarm [used only to freeze a changed cell], extra...) are never read as inputs. The change is
passed through the same clipping as the SDK safety filter before it is encoded.

`predict` returns KPI DELTAS (contract amendment 1, A5): cluster KPIs after the change and one twin
step MINUS the cluster KPIs of the same step with no change. The network predicts the standardised
delta, anchored so that a zero change gives exactly 0.

DEGRADED variant (a documented, deliberate fault for the trust-gate demo): `fault="tilt_reversed"`
feeds the network the NEGATED tilt change, so the model believes every tilt-up is a tilt-down. It is
identical to the healthy model on any change whose tilt component is zero. Its model_id is
`<id>-degraded-tilt-reversed`; every record of its output must carry that id.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from t2.features import (
    CELL_FIELDS,
    LABELS,
    N_CELLS,
    N_FEATURES,
    col,
    encode,
    state_change_to_row,
)

FAULTS = {
    "tilt_reversed": "the sign of every tilt change is flipped before the network sees it",
}
_TILT_COLS = [col("d_tilt_deg", i) for i in range(N_CELLS)]
_CHANGE_COLS = _TILT_COLS + [col("d_power_db", i) for i in range(N_CELLS)]
DELTA_RANGES = {"coverage_ratio": (-1.0, 1.0), "sla_ratio": (-1.0, 1.0),
                "sinr_p10_db": (-70.0, 70.0), "sinr_mean_db": (-70.0, 70.0)}
_SHARE = [col("user_share", i) for i in range(N_CELLS)]
_SM = [col("sinr_mean_db", i) for i in range(N_CELLS)]
_SP = [col("sinr_p10_db", i) for i in range(N_CELLS)]


def current_kpis(x_raw: np.ndarray) -> np.ndarray:
    """(n, 38) raw rows -> (n, 4) current cluster KPIs in LABELS order (t2/kpis.py definition)."""
    x = np.atleast_2d(x_raw)
    return np.stack(
        [x[:, col("coverage_ratio")], x[:, col("sla_ratio")],
         (x[:, _SHARE] * x[:, _SP]).sum(1), (x[:, _SHARE] * x[:, _SM]).sum(1)], axis=1)


class Predictor:
    def __init__(self, params: Mapping[str, np.ndarray], meta: Mapping[str, Any], fault: str | None = None):
        if fault is not None and fault not in FAULTS:
            raise ValueError(f"unknown fault {fault!r}; known: {sorted(FAULTS)}")
        self._layers = []
        i = 0
        while f"W{i}" in params:
            self._layers.append((np.asarray(params[f"W{i}"], dtype=np.float64),
                                 np.asarray(params[f"b{i}"], dtype=np.float64)))
            i += 1
        if len(self._layers) < 2 or self._layers[0][0].shape[0] != N_FEATURES:
            raise ValueError("weights do not match the t2 feature layout")
        self._y_std = np.asarray(params["y_std"], dtype=np.float64)
        self.meta = dict(meta)
        self.fault = fault
        base_id = str(meta["model_id"])
        self.base_model_id = base_id
        self.model_id = base_id if fault is None else f"{base_id}-degraded-{fault.replace('_', '-')}"
        self.version = str(meta["version"])
        self.degraded = fault is not None

    # ---- batch API (also used by eval) -------------------------------------------------
    def residual_scaled(self, x_raw: np.ndarray) -> np.ndarray:
        """Raw network output (standardised KPI delta), before anchoring and de-standardising."""
        x = np.array(x_raw, dtype=np.float64, ndmin=2)
        if self.fault == "tilt_reversed":
            x = x.copy()
            x[:, _TILT_COLS] *= -1.0
        h = encode(x)
        for k, (w, b) in enumerate(self._layers):
            h = h @ w + b
            if k < len(self._layers) - 1:
                h = np.maximum(h, 0.0)
        return h

    def predict_batch(self, x_raw: np.ndarray) -> np.ndarray:
        """(n, 38) raw rows -> (n, 4) predicted KPI DELTAS (after the change minus the no-change
        counterfactual at the same step), in LABELS order.

        Anchored: delta = net(state, change) - net(state, no change), so a zero change predicts
        exactly 0 whatever the state."""
        x = np.array(x_raw, dtype=np.float64, ndmin=2)
        x0 = x.copy()
        x0[:, _CHANGE_COLS] = 0.0
        d = (self.residual_scaled(x) - self.residual_scaled(x0)) * self._y_std
        for j, name in enumerate(LABELS):
            lo, hi = DELTA_RANGES[name]
            d[:, j] = np.clip(d[:, j], lo, hi)
        return d

    # ---- single-call API (the tool contract) ---------------------------------------------
    def predict(self, state: Mapping[str, Any], change: Sequence[Mapping[str, Any]] | None) -> dict[str, float]:
        row = state_change_to_row(state, change, apply_safety=True)
        y = self.predict_batch(row[None, :])[0]
        return {name: float(y[j]) for j, name in enumerate(LABELS)}

    def describe(self) -> dict[str, Any]:
        return {"model_id": self.model_id, "version": self.version, "fault": self.fault,
                "fault_doc": FAULTS.get(self.fault) if self.fault else None,
                "label": self.meta.get("label"), "dataset_sha256": self.meta.get("dataset_sha256")}


def load(path: str | Path, fault: str | None = None) -> Predictor:
    with np.load(path, allow_pickle=False) as z:
        params = {k: z[k] for k in z.files if k != "meta_json"}
        meta = json.loads(str(z["meta_json"]))
    return Predictor(params, meta, fault=fault)


def load_degraded(path: str | Path, fault: str = "tilt_reversed") -> Predictor:
    return load(path, fault=fault)
