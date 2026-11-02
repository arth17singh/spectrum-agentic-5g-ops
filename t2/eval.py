"""Evaluate the Track 2 response model on the HELD-OUT seeds. v0, local, not federated / not the official job.

    python -m t2.eval --data <dataset.npz> --model t2/artifacts/t2_response_v0.npz \
        --out-dir t2/models [--manifest <dataset manifest>]

Labels and predictions are KPI DELTAS: after the change and one twin step, minus the same step with
no change (contract amendment 1, A5). Metrics (test-split rows only; no test seed is in training):
  * MAE per label for the model and for baselines (all in delta terms)
      no_change      predicts a delta of 0
      no_change+dP   the power change added to each cell's served RSRP/SINR, interference held
                     fixed (the Maveric twin applies power as a pure RSRP offset); coverage and
                     SLA from a normal fit to each cell's mean and p10. See BASELINE_DEFS.
      ridge          anchored linear least squares on the same encoded inputs (extra reference)
  * Direction accuracy for coverage and SLA: sign of the predicted delta vs the measured delta, on
    rows with a NONZERO change and |measured delta| > 0.004 (the ratios are quantised at 0.005).
    "no_change" predicts 0, so it can never get a direction right; reported as 0.
  * The degraded (tilt_reversed) variant on the rows whose tilt change is nonzero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from t2.features import LABELS, N_CELLS, col, encode
from t2.predict import Predictor, load

LABEL = "v0, local, not federated / not the official job"
EPS = 0.004
BASELINE_DEFS = {
    "no_change": "next-step KPIs = current cluster KPIs",
    "no_change+dP": "no change plus the power change added to the served signal, interference held fixed: "
                    "each cell's served RSRP and SINR shift by that cell's d_power; coverage (RSRP > -97 dBm) and "
                    "SLA (SINR > 0 dB) follow from a normal fit to each cell's mean and p10, weighted by user "
                    "share; sinr_p10_db and sinr_mean_db += share-weighted mean d_power",
    "ridge": "linear least squares on the same encoded inputs, trained on the train seeds, predicting the residual",
}
_DP = [col("d_power_db", i) for i in range(N_CELLS)]
_DT = [col("d_tilt_deg", i) for i in range(N_CELLS)]
_SH = [col("user_share", i) for i in range(N_CELLS)]


def zero_change(x: np.ndarray) -> np.ndarray:
    z = x.copy()
    z[:, _DP + _DT] = 0.0
    return z


def baseline_no_change(x: np.ndarray) -> np.ndarray:
    """Predicts a delta of 0 for every KPI."""
    return np.zeros((len(x), len(LABELS)))


COVERAGE_RSRP_DBM = -97.0  # Maveric twin defaults (MavericTwinConfig): coverage threshold, SLA threshold
SLA_SINR_DB = 0.0


def _share_above(mean: np.ndarray, p10: np.ndarray, shift: np.ndarray, threshold: float) -> np.ndarray:
    """Per cell: share of users above `threshold` if the served-signal distribution (a normal with the
    cell's mean and p10) is shifted by `shift` dB."""
    from scipy.stats import norm

    sigma = np.maximum((mean - p10) / 1.2816, 1.0)
    return norm.sf((threshold - mean - shift) / sigma)


def baseline_dp(x: np.ndarray) -> np.ndarray:
    """The power change added to the served signal, interference held fixed, as a KPI DELTA: cell i's
    served RSRP and SINR shift by its own d_power. Coverage and SLA deltas follow from a normal fit to
    each cell's mean and p10; the SINR deltas are the share-weighted mean power change."""
    dp = x[:, _DP]
    y = np.zeros((len(x), len(LABELS)))
    for j, (field, unit, thr) in ((0, ("rsrp", "dbm", COVERAGE_RSRP_DBM)), (1, ("sinr", "db", SLA_SINR_DB))):
        m = x[:, [col(f"{field}_mean_{unit}", i) for i in range(N_CELLS)]]
        p = x[:, [col(f"{field}_p10_{unit}", i) for i in range(N_CELLS)]]
        y[:, j] = (x[:, _SH] * (_share_above(m, p, dp, thr) - _share_above(m, p, np.zeros_like(dp), thr))).sum(1)
    shift = (x[:, _SH] * dp).sum(1)
    y[:, 2] = shift
    y[:, 3] = shift
    return y


def fit_ridge(x: np.ndarray, y: np.ndarray, lam: float = 1.0) -> Any:
    """Linear least squares on the encoded inputs, anchored like the network: f(x) - f(x with no change)."""
    a = np.hstack([encode(x), np.ones((len(x), 1))])
    w = np.linalg.solve(a.T @ a + lam * np.eye(a.shape[1]), a.T @ y)

    def predict(xq: np.ndarray) -> np.ndarray:
        one = np.ones((len(xq), 1))
        return (np.hstack([encode(xq), one]) - np.hstack([encode(zero_change(xq)), one])) @ w

    return predict


def mae(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {n: float(np.abs(pred[:, j] - y[:, j]).mean()) for j, n in enumerate(LABELS)}


def direction(pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    """Share of rows (in `mask` and with |true delta| > EPS) where the sign of the predicted delta is right
    (a predicted delta of exactly 0 counts as wrong)."""
    out: dict[str, Any] = {}
    for j, n in enumerate(LABELS[:2]):  # coverage_ratio, sla_ratio
        m = mask & (np.abs(true[:, j]) > EPS)
        ok = (np.sign(pred[m, j]) == np.sign(true[m, j])) & (pred[m, j] != 0)
        out[n] = {"accuracy": float(ok.mean()) if m.any() else None, "n": int(m.sum())}
    return out


def evaluate(pred: Predictor, degraded: Predictor, data: dict[str, np.ndarray]) -> dict[str, Any]:
    test, train = data["split"] == 1, data["split"] == 0
    x, y = data["x"][test].astype(np.float64), data["y"][test].astype(np.float64)  # y = KPI delta vs no-op
    ridge = fit_ridge(data["x"][train].astype(np.float64), data["y"][train].astype(np.float64))
    changed = (np.abs(x[:, _DP]).sum(1) + np.abs(x[:, _DT]).sum(1)) > 0
    models = {
        "model": pred.predict_batch, "degraded": degraded.predict_batch,
        "no_change": baseline_no_change, "no_change+dP": baseline_dp, "ridge": ridge,
    }
    preds = {k: f(x) for k, f in models.items()}
    viavi = (data["start_kind"][test] == 1) & (data["step"][test] <= 1)
    others = ("model", "no_change", "no_change+dP", "ridge")

    def slice_mae(mask: np.ndarray) -> dict[str, Any]:
        return {k: mae(p[mask], y[mask]) for k, p in preds.items() if k != "degraded"} | {"n": int(mask.sum())}

    res: dict[str, Any] = {
        "label": LABEL, "labels_are": "KPI deltas vs the no-op counterfactual at the same step (amendment 1, A5)",
        "n_test_rows": int(test.sum()), "n_test_seeds": int(len(np.unique(data["seed"][test]))),
        "n_changed_rows": int(changed.sum()), "baselines": BASELINE_DEFS,
        "mean_abs_true_delta_changed_rows": {n: float(np.abs(y[changed, j]).mean()) for j, n in enumerate(LABELS)},
        "mae_all_rows": slice_mae(np.ones(len(x), bool)),
        "mae_changed_rows": slice_mae(changed),
        "mae_zero_change_rows": slice_mae(~changed),
        "mae_viavi_like_start_slice": slice_mae(viavi) | {"definition": "episodes starting at exactly 40 dBm / 6 deg, steps 0-1"},
        "mae_by_scenario": {n: slice_mae(data["scenario"][test] == i)["model"] | {"n": int((data["scenario"][test] == i).sum())}
                            for i, n in enumerate(data["scenario_names"])},
        "direction": {k: direction(preds[k], y, changed) for k in others},
    }
    sign_dp = np.zeros_like(y)
    sign_dp[:, :2] = np.sign((x[:, _SH] * x[:, _DP]).sum(1))[:, None]
    res["direction"]["heuristic_sign_of_dP"] = direction(sign_dp, y, changed)

    tilt = (np.abs(x[:, _DT]).sum(1) > 0)
    res["degraded_tilt_reversed"] = {
        "variant": degraded.model_id, "fault": degraded.fault,
        "rows_with_tilt_change": int(tilt.sum()),
        "mae_tilt_rows": {"healthy": mae(preds["model"][tilt], y[tilt]), "degraded": mae(preds["degraded"][tilt], y[tilt]),
                          "no_change": mae(preds["no_change"][tilt], y[tilt])},
        "mae_no_tilt_rows": {"healthy": mae(preds["model"][~tilt], y[~tilt]), "degraded": mae(preds["degraded"][~tilt], y[~tilt])},
        "max_abs_difference_on_no_tilt_rows": float(np.abs(preds["model"][~tilt] - preds["degraded"][~tilt]).max()),
        "direction_tilt_rows": {"healthy": direction(preds["model"], y, tilt), "degraded": direction(preds["degraded"], y, tilt)},
        "mae_ratio_tilt_rows_degraded_over_healthy": {
            n: float(np.abs(preds["degraded"][tilt, j] - y[tilt, j]).mean() / max(np.abs(preds["model"][tilt, j] - y[tilt, j]).mean(), 1e-9))
            for j, n in enumerate(LABELS)},
    }
    res["effect_sizes_in_data"] = effect_sizes(data)
    return res


def effect_sizes(data: dict[str, np.ndarray]) -> dict[str, Any]:
    """How much do power and tilt move coverage / SLA / SINR in the data (all splits)? For the degraded-variant choice."""
    x, e = data["x"].astype(np.float64), data["y"].astype(np.float64)
    dp_abs, dt_abs = np.abs(x[:, _DP]).sum(1), np.abs(x[:, _DT]).sum(1)
    out = {}
    for name, m, scale in (("power_only_rows_per_db", (dp_abs > 0) & (dt_abs == 0), dp_abs),
                           ("tilt_only_rows_per_deg", (dt_abs > 0) & (dp_abs == 0), dt_abs)):
        out[name] = {n: float(np.abs(e[m, j]).mean() / scale[m].mean()) for j, n in enumerate(LABELS)} | {"n": int(m.sum())}
    return out


def build_model_record(model_path: Path, pred: Predictor, ev: dict[str, Any], manifest: dict[str, Any] | None) -> dict[str, Any]:
    sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    m = ev["mae_changed_rows"]
    return {
        "schema_version": "1", "kind": "model", "run_id": "w9-models-v0-local", "team": "team13",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "aws_time": None, "aws_time_source": None, "synthetic": False,
        "model_id": pred.base_model_id, "version": pred.version, "model_kind": "t2-response",
        "artifact_uri": f"t2/artifacts/{model_path.name}", "artifact_sha256": sha,
        "status": "candidate",
        "trained_on": f"{LABEL}. Maveric in-process twin (ran_rl, shipped sim_tilt_sweep_4cell models), "
                      f"4-cell clusters, data contract v0.2"
                      + (f"; dataset sha256 {manifest['dataset_sha256']}, {manifest['n_rows']} rows" if manifest else ""),
        "metrics": {
            "label": LABEL,
            "n_test_rows": ev["n_test_rows"], "n_test_seeds": ev["n_test_seeds"],
            "mae_all_rows": ev["mae_all_rows"]["model"],
            "mae_all_rows_no_change": ev["mae_all_rows"]["no_change"],
            "mae_all_rows_no_change+dP": ev["mae_all_rows"]["no_change+dP"],
            "mae_changed_rows": m["model"],
            "mae_changed_rows_no_change": m["no_change"],
            "mae_changed_rows_no_change+dP": m["no_change+dP"],
            "direction": ev["direction"]["model"],
            "baselines": BASELINE_DEFS,
            "labels_are": "KPI deltas vs the no-op counterfactual at the same step",
            "direction_definition": f"sign of the predicted delta vs the measured delta, rows with a nonzero change and |measured delta| > {EPS}",
            "degraded_variant": {"model_id": f"{pred.base_model_id}-degraded-tilt-reversed", "fault": "tilt_reversed"},
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--manifest")
    a = ap.parse_args(argv)
    with np.load(a.data, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    pred, degraded = load(a.model), load(a.model, fault="tilt_reversed")
    manifest = json.loads(Path(a.manifest).read_text()) if a.manifest else None
    ev = evaluate(pred, degraded, data)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{pred.base_model_id}.eval.json").write_text(json.dumps(ev, indent=2) + "\n")
    rec = build_model_record(Path(a.model), pred, ev, manifest)
    (out / f"{pred.base_model_id}.json").write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps({k: ev[k] for k in ("n_test_rows", "n_test_seeds", "n_changed_rows")}))
    print("MAE changed rows  model        ", {k: round(v, 4) for k, v in ev["mae_changed_rows"]["model"].items()})
    print("                  no_change    ", {k: round(v, 4) for k, v in ev["mae_changed_rows"]["no_change"].items()})
    print("                  no_change+dP ", {k: round(v, 4) for k, v in ev["mae_changed_rows"]["no_change+dP"].items()})
    print("                  ridge        ", {k: round(v, 4) for k, v in ev["mae_changed_rows"]["ridge"].items()})
    for k, v in ev["direction"].items():
        print("direction", k, {n: (round(d["accuracy"], 3) if d["accuracy"] is not None else None, d["n"]) for n, d in v.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
