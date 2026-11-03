"""The SageMaker-trained models baked into the tools image: files match their provenance, the predictor
conversion keeps W11's function, and the policy runs on the SDK observation it was trained on."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from t3.deploy import import_models as im

REPO = Path(__file__).resolve().parents[3]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_committed_policy_matches_its_provenance_and_the_scored_checkpoint():
    prov = json.loads((REPO / "t1/rl/artifacts/ppo_sm_50k.provenance.json").read_text())
    assert _sha(im.POLICY_ZIP) == prov["artifact_sha256"]
    # the checkpoint the SageMaker eval run t1.rl.eval-20261103T043115Z-bf52 scored
    assert prov["artifact_sha256"] == "e04d8c907ca30fd3c876121b924c2836157cb546b1890a984184bb52f37484c1"
    assert prov["source_s3_uri"].startswith("s3://") and prov["source_s3_uri"].endswith("ppo_50000_steps.zip")
    ev = prov["evaluation"]
    assert ev["passed"] is False and ev["gates"]["UPLIFT"] is False   # the honest miss stays on file
    assert "UPLIFT FAILS" in ev["honest_summary"]
    meta = json.loads(im.POLICY_ZIP.with_suffix(".meta.json").read_text())
    assert meta["obs"] == "sdk" and meta["model_id"] == "t1-ppo-cco-sm"


def test_committed_predictor_matches_its_provenance():
    prov = json.loads((REPO / "t2/artifacts/t2_response_central_disjoint_v1.provenance.json").read_text())
    assert _sha(im.PREDICTOR_NPZ) == prov["artifact_sha256"]
    assert prov["source_sha256"] == "fe76800152d3dbcf4795cc62c791dc287ff3b919f0e19b0854e7e13d1bea42e4"
    assert prov["conversion_check"]["equal_to_1e-6"] is True
    assert prov["training"]["run_id"] == "t2.central.disjoint-20261103T041620Z"


def test_conversion_keeps_the_silos_function():
    from silos import nn_core as nn
    from t2.predict import load
    params = nn.init_params(seed=3)
    out = Path(__file__).parent / "_tmp_conv.npz"
    try:
        np.savez(out, meta_json=np.array(json.dumps({"model_id": "x", "version": "y"})), **im.silos_to_t2_params(params))
        check = im.verify_conversion(params, out, im.probe_rows(256, seed=1))
        assert check["equal_to_1e-6"], check
        p = load(out)
        rows = im.probe_rows(8, seed=2)
        zero = rows.copy()
        zero[:, 30:] = 0.0
        assert np.allclose(p.predict_batch(zero), 0.0)   # served anchored: no change predicts exactly 0
    finally:
        out.unlink(missing_ok=True)


def test_served_predictor_loads_and_predicts_deltas():
    from t2.predict import load
    p = load(im.PREDICTOR_NPZ)
    assert (p.model_id, p.version) == ("t2-response-central-disjoint", "v1-041620Z")
    state = {"power_dbm": [40.0] * 4, "tilt_deg": [8.0] * 4, "user_share": [0.25] * 4,
             "rsrp_mean_dbm": [-90.0] * 4, "rsrp_p10_dbm": [-105.0] * 4, "sinr_mean_db": [8.0] * 4,
             "sinr_p10_db": [-2.0] * 4, "coverage_ratio": 0.9, "sla_ratio": 0.9}
    y = p.predict(state, [{"cell": 2, "d_power_db": 2.0, "d_tilt_deg": -1.0}])
    assert set(y) == {"coverage_ratio", "sla_ratio", "sinr_p10_db", "sinr_mean_db"}
    assert p.predict(state, []) == {k: 0.0 for k in y}


def test_sagemaker_policy_uses_the_sdk_observation():
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("ran_rl")
    from t1.rl.policy import load
    from twin.replay import ReplayTwin
    pol = load(im.POLICY_ZIP)
    assert pol.obs_mode == "sdk" and (pol.model_id, pol.version) == ("t1-ppo-cco-sm", "50k-a09e")
    state = ReplayTwin("nominal", 7, None).reset()
    text = pol.propose_text(state)
    assert text == "" or text.startswith("cell=")
    v0 = load(REPO / "t1/rl/artifacts/ppo_v0.zip")
    assert v0.obs_mode == "masked"   # v0 keeps the masked view it was trained on
