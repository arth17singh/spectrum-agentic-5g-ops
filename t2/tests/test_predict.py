"""Numpy predictor parity with torch, the degraded variant, input handling and artifact sanity."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from t2.export import export_npz
from t2.features import N_CELLS, N_FEATURES, col
from t2.model_torch import ResponseMLP, encoded
from t2.predict import FAULTS, load, load_degraded
from t2.train import fit, _zero_change

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "t2_response_v0.npz"


@pytest.fixture(scope="module")
def data():
    with np.load(ROOT / "data" / "t2_dataset_v0.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


@pytest.fixture(scope="module")
def tiny(tmp_path_factory, data):
    """A short fit on a slice of the TRAIN rows, exported to npz."""
    tr = np.flatnonzero(data["split"] == 0)[:1500]
    res = fit(data["x"][tr[:1200]], data["y"][tr[:1200]], data["x"][tr[1200:]], data["y"][tr[1200:]],
              hidden=(32, 32), epochs=3, seed=1)
    path = export_npz(res["state_dict"], res["y_std"], tmp_path_factory.mktemp("w") / "tiny.npz",
                      {"model_id": "t2-response-test", "version": "test"})
    return res, path


def torch_predict_double(res, x_raw: np.ndarray) -> np.ndarray:
    """Anchored standardised delta from the torch model in float64: net(x) - net(x with no change)."""
    net = ResponseMLP(res["hidden"])
    net.load_state_dict(res["state_dict"])
    net = net.double().eval()
    with torch.no_grad():
        a = net(torch.tensor(encoded(x_raw).astype(np.float64))).numpy()
        b = net(torch.tensor(encoded(_zero_change(x_raw)).astype(np.float64))).numpy()
    return a - b


def test_numpy_matches_torch_within_1e5(tiny, data):
    res, path = tiny
    pred = load(path)
    x = data["x"][-400:].astype(np.float64)
    ref = torch_predict_double(res, x) * np.asarray(res["y_std"], dtype=np.float64)
    ref = np.clip(ref, [-1, -1, -70, -70], [1, 1, 70, 70])
    assert np.abs(pred.predict_batch(x) - ref).max() < 1e-5


def test_zero_change_predicts_exactly_zero():
    p = load(ARTIFACT)
    assert p.predict(STATE, None) == {k: 0.0 for k in ("coverage_ratio", "sla_ratio", "sinr_p10_db", "sinr_mean_db")}
    assert load_degraded(ARTIFACT).predict(STATE, []) == p.predict(STATE, [])


def test_real_artifact_matches_checkpoint_format():
    p = load(ARTIFACT)
    assert p.model_id == "t2-response-v0" and p.version == "v0"
    assert "not federated" in p.meta["label"]


def test_predict_module_does_not_import_torch():
    code = ("import sys, numpy; import t2.predict as p; p.load(%r); "
            "assert 'torch' not in sys.modules, 'torch was imported'" % str(ARTIFACT))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT.parent)
    assert r.returncode == 0, r.stderr


STATE = {
    "power_dbm": [23.0, 25.0, 21.0, 24.0], "tilt_deg": [6.0, 7.0, 5.0, 6.0], "ue_count": [30, 60, 80, 30],
    "rsrp_mean_dbm": [-90.0, -100.0, -98.0, -92.0], "rsrp_p10_dbm": [-95.0, -115.0, -110.0, -98.0],
    "sinr_mean_db": [8.0, 6.0, 10.0, 5.0], "sinr_p10_db": [1.5, -0.2, 0.6, 0.0],
    "coverage_ratio": 0.6, "sla_ratio": 0.9,
    "hour": 3.0, "phase": 0.2, "alarm": [False] * 4, "asleep": [False] * 4, "extra": {"x": 1},
}


def test_predict_output_shape_and_keys():
    p = load(ARTIFACT)
    out = p.predict(STATE, [{"cell": 1, "d_power_db": 2.0, "d_tilt_deg": -1.0}])
    assert set(out) == {"coverage_ratio", "sla_ratio", "sinr_p10_db", "sinr_mean_db"}
    assert abs(out["coverage_ratio"]) <= 1 and abs(out["sla_ratio"]) <= 1  # deltas, not levels
    assert all(isinstance(v, float) and np.isfinite(v) for v in out.values())
    assert p.predict(STATE, None) == p.predict(STATE, [])


def test_forbidden_state_fields_and_absolute_counts_are_not_inputs():
    p = load(ARTIFACT)
    ch = [{"cell": 0, "d_power_db": 1.0, "d_tilt_deg": 0.0}]
    base = p.predict(STATE, ch)
    other = copy.deepcopy(STATE)
    other.update(hour=17.0, phase=0.9, extra={"y": 2}, asleep=[True, False, False, False])
    other["ue_count"] = [3 * c for c in STATE["ue_count"]]  # same shares, 3x the users
    assert p.predict(other, ch) == base
    shares = {k: v for k, v in STATE.items() if k != "ue_count"}
    shares["user_share"] = [c / 200 for c in STATE["ue_count"]]
    assert p.predict(shares, ch) == base


def test_change_is_clipped_like_the_safety_filter():
    p = load(ARTIFACT)
    big = p.predict(STATE, [{"cell": 2, "d_power_db": 50.0, "d_tilt_deg": 30.0}])
    lim = p.predict(STATE, [{"cell": 2, "d_power_db": 8.0, "d_tilt_deg": 5.0}])
    assert big == lim
    with pytest.raises(ValueError):
        p.predict(STATE, [{"cell": 7, "d_power_db": 1.0}])


def test_degraded_variant_differs_exactly_as_documented():
    good, bad = load(ARTIFACT), load_degraded(ARTIFACT)
    assert "tilt_reversed" in FAULTS and bad.fault == "tilt_reversed" and bad.degraded and not good.degraded
    assert bad.model_id == "t2-response-v0-degraded-tilt-reversed" and good.model_id == "t2-response-v0"
    assert bad.version == good.version
    # no tilt in the change -> identical
    pw = [{"cell": 1, "d_power_db": 3.0, "d_tilt_deg": 0.0}]
    assert good.predict(STATE, pw) == bad.predict(STATE, pw)
    # a tilt change -> different, and it equals the healthy model fed the negated tilt
    tl = [{"cell": 1, "d_power_db": 0.0, "d_tilt_deg": 3.0}]
    assert good.predict(STATE, tl) != bad.predict(STATE, tl)
    neg = [{"cell": 1, "d_power_db": 0.0, "d_tilt_deg": -3.0}]
    assert bad.predict(STATE, tl) == pytest.approx(good.predict(STATE, neg), abs=1e-12)
    with pytest.raises(ValueError):
        load(ARTIFACT, fault="nope")


def test_artifact_beats_no_change_on_held_out_seeds(data):
    """A regression guard on the committed model, not a claim of accuracy: test seeds only, delta terms."""
    t = data["split"] == 1
    x, y = data["x"][t].astype(np.float64), data["y"][t].astype(np.float64)  # y = delta vs no-op
    dcols = [col("d_power_db", i) for i in range(N_CELLS)] + [col("d_tilt_deg", i) for i in range(N_CELLS)]
    changed = np.abs(x[:, dcols]).sum(1) > 0
    p = load(ARTIFACT)
    err_model = np.abs(p.predict_batch(x[changed]) - y[changed]).mean(0)
    err_none = np.abs(y[changed]).mean(0)  # "no change" predicts a delta of 0
    assert (err_model < err_none).all()
    rec = json.loads((ROOT / "models" / "t2-response-v0.json").read_text())
    assert rec["kind"] == "model" and rec["model_kind"] == "t2-response" and "not federated" in rec["trained_on"]
    import hashlib
    assert rec["artifact_sha256"] == hashlib.sha256(ARTIFACT.read_bytes()).hexdigest()
    assert N_FEATURES == 38
