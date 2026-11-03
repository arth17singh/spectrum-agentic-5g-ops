"""The federated aggregate as served by t2.predict: loads, equals the silos network, anchored, provenance consistent.

    PYTHONPATH=contracts:. python -m pytest t2/tests/test_federated_predictor.py -q
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from silos import nn_core as nn
from t2 import import_federated as imp
from t2.predict import load

REPO = Path(__file__).resolve().parents[2]
NPZ, SOURCE, PROV = imp.NPZ, imp.SOURCE, imp.PROV
STATE = {"power_dbm": [40.0] * 4, "tilt_deg": [8.0] * 4, "user_share": [0.25] * 4, "rsrp_mean_dbm": [-90.0] * 4,
         "rsrp_p10_dbm": [-105.0] * 4, "sinr_mean_db": [8.0] * 4, "sinr_p10_db": [-2.0] * 4, "coverage_ratio": 0.9, "sla_ratio": 0.9}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_loads_with_the_federated_identity():
    p = load(NPZ)
    assert (p.model_id, p.version) == ("t2-response-federated-disjoint", "v1-5fb29b2d")


def test_parity_with_the_silos_network_within_1e_6():
    params = dict(np.load(SOURCE))
    nn.check_params(params)
    p = load(NPZ)
    rng = np.random.default_rng(11)
    rows = imp.probe_rows(512, seed=5)
    unanchored = p.residual_scaled(rows) * p._y_std          # the network before anchoring, through the t2 encoding
    assert np.max(np.abs(unanchored - nn.forward_np(params, rows))) <= 1e-6
    assert rows.shape[1] == 38 and rng is not None


def test_anchored_a_zero_change_predicts_exactly_zero_and_a_change_does_not():
    p = load(NPZ)
    assert p.predict(STATE, []) == {"coverage_ratio": 0.0, "sla_ratio": 0.0, "sinr_p10_db": 0.0, "sinr_mean_db": 0.0}
    y = p.predict(STATE, [{"cell": 2, "d_power_db": 2.0, "d_tilt_deg": -1.0}])
    assert set(y) == {"coverage_ratio", "sla_ratio", "sinr_p10_db", "sinr_mean_db"} and any(abs(v) > 0 for v in y.values())
    rows = imp.probe_rows(8, seed=2)
    zero = rows.copy()
    zero[:, 30:] = 0.0                                        # the eight change columns
    assert np.allclose(p.predict_batch(zero), 0.0)


def test_provenance_matches_the_files_and_names_no_service_or_account():
    prov = json.loads(PROV.read_text())
    assert prov["artifact_sha256"] == sha(NPZ)
    assert prov["weights_canonical_sha256"] == nn.sha256_bytes(nn.safetensors_dumps(dict(np.load(SOURCE))))
    assert prov["published_file_sha256"] == "20b8fa7c986c7952739ef6b3345d7838d4507c3aac2fb6de08f15281af5d9188"
    assert prov["conversion_check"]["equal_to_1e-6"] is True and prov["conversion_check"]["max_abs_diff_unanchored_vs_silos"] <= 1e-6
    fed = prov["federation"]
    assert (fed["session_id"], fed["rounds"], fed["final_iteration"]) == ("5fb29b2d83806227d5c80", 20, 21)
    assert prov["published_s3_uri"].endswith("/t2-fl/5fb29b2d83806227d5c80/federated.safetensors") and "<account>" in prov["published_s3_uri"]
    ev = prov["evaluation"]
    assert ev["run_id"].startswith("t2.fl_federated.disjoint-") and ev["n_items"] == 10000
    assert set(prov["baselines_same_shared_test"]) >= {"local_A", "local_B", "local_C", "central"}
    assert ev["mean_score"] < ev["mean_score_noop"]
    assert set(prov["comparisons"]) == {"federated_vs_B", "federated_vs_C", "distance_to_central", "federated_vs_localfedavg"}
    blob = (PROV.read_text() + NPZ.read_bytes().decode("latin1") + SOURCE.read_bytes().decode("latin1")).lower()
    assert ("cross" + "trainer") not in blob and ("4835" + "98626742") not in blob


def test_importer_refuses_a_file_the_evaluate_run_did_not_score(tmp_path):
    params = nn.init_params(1)
    f = tmp_path / "other.safetensors"
    f.write_bytes(nn.safetensors_dumps(params, {"k": "v"}))
    with pytest.raises(SystemExit):
        imp.import_federated(f, "s3://b/k", "sid", 20, 21, {"details": {"artifact_sha256": "0" * 64}}, {}, {}, out_dir=tmp_path / "out")
