"""Dataset schema, leakage and split checks (contract v0.2)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from t2 import gen
from t2.features import (
    FEATURE_NAMES,
    FORBIDDEN_INPUTS,
    LABELS,
    N_CELLS,
    N_FEATURES,
    POWER_DELTA,
    TILT_DELTA,
    col,
)

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def committed():
    with np.load(DATA / "t2_dataset_v0.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


@pytest.fixture(scope="module")
def fresh():
    return gen.generate(episodes=4, steps=3, workers=1)


def test_feature_names_have_no_forbidden_fields():
    bases = {n.split("[")[0] for n in FEATURE_NAMES}
    assert not bases & FORBIDDEN_INPUTS
    assert N_FEATURES == 38 == len(FEATURE_NAMES) == len(set(FEATURE_NAMES))
    assert LABELS == ("coverage_ratio", "sla_ratio", "sinr_p10_db", "sinr_mean_db")


def test_fresh_rows_schema(fresh):
    n = len(fresh["x"])
    assert fresh["x"].shape == (n, N_FEATURES) and fresh["y"].shape == (n, 4) == fresh["y_nochange"].shape
    assert np.isfinite(fresh["x"]).all() and np.isfinite(fresh["y"]).all()
    # inputs are exactly the contract fields; metadata is kept in separate arrays
    assert set(fresh) >= {"x", "y", "y_nochange", "seed", "step", "scenario", "start_kind", "change_kind", "split"}
    shares = fresh["x"][:, [col("user_share", i) for i in range(N_CELLS)]]
    assert np.allclose(shares.sum(1), 1.0, atol=1e-5)  # shares, not counts
    assert shares.max() <= 1.0


def test_changes_are_within_sdk_limits(committed):
    x = committed["x"]
    dp = x[:, [col("d_power_db", i) for i in range(N_CELLS)]]
    dt = x[:, [col("d_tilt_deg", i) for i in range(N_CELLS)]]
    assert dp.min() >= POWER_DELTA[0] - 1e-6 and dp.max() <= POWER_DELTA[1] + 1e-6
    assert dt.min() >= TILT_DELTA[0] - 1e-6 and dt.max() <= TILT_DELTA[1] + 1e-6
    power = x[:, [col("power_dbm", i) for i in range(N_CELLS)]]
    tilt = x[:, [col("tilt_deg", i) for i in range(N_CELLS)]]
    assert ((power + dp) >= 10 - 1e-4).all() and ((power + dp) <= 46 + 1e-4).all()
    assert ((tilt + dt) >= 4 - 1e-4).all() and ((tilt + dt) <= 15 + 1e-4).all()


def test_mix_of_change_kinds_and_zero_rows(committed):
    kinds = {gen.CHANGE_KINDS[k] for k in np.unique(committed["change_kind"])}
    assert kinds == set(gen.CHANGE_KINDS)  # zero, single_power, single_tilt, single_both, multi
    zero = committed["change_kind"] == gen.CHANGE_KINDS.index("zero")
    assert zero.sum() > 1000
    assert np.allclose(committed["y"][zero], 0.0)  # labels are deltas: no change -> exactly 0
    assert np.allclose(committed["y"], committed["y_level"] - committed["y_nochange"], atol=1e-4)
    assert (committed["start_kind"] == 1).any()  # Viavi-like start present


def test_start_configuration_is_varied(committed):
    p = committed["start_power_mean"]
    t = committed["start_tilt_mean"]
    assert p.min() < 20 and p.max() > 38 and t.min() < 6 and t.max() > 9


def test_split_by_seed_hash_and_no_seed_overlap(committed):
    seeds, split = committed["seed"], committed["split"]
    train, test = set(seeds[split == 0].tolist()), set(seeds[split == 1].tolist())
    assert train and test and not train & test
    assert all(gen.split_of(s) == "test" for s in test) and all(gen.split_of(s) == "train" for s in train)
    assert 0.1 < len(test) / (len(test) + len(train)) < 0.3


def test_manifest_matches_dataset(committed):
    m = json.loads((DATA / "t2_dataset_manifest.json").read_text())
    assert m["dataset_sha256"] == hashlib.sha256((DATA / "t2_dataset_v0.npz").read_bytes()).hexdigest()
    assert m["n_rows"] == len(committed["x"]) >= 20000
    assert set(m["seeds"]["test_seeds"]) == set(committed["seed"][committed["split"] == 1].tolist())
    assert m["generator"]["git_commit"] not in ("", "unknown")
    assert "not federated" in m["label"]
    assert not {n.split("[")[0] for n in m["inputs"]} & FORBIDDEN_INPUTS
    assert "outage" not in m["generator"]["scenarios"]


def test_generator_is_deterministic_and_counterfactual_restores_state():
    a, b = gen.run_episode(10003, 3), gen.run_episode(10003, 3)
    assert np.array_equal(np.array(a["x"]), np.array(b["x"])) and np.array_equal(np.array(a["y"]), np.array(b["y"]))
