"""Tests for the silo datasets: features, generator determinism on a tiny config, checks pass and fail.

    <python> -m pytest silos/tests -q        (from the repo root; needs ran-rl with the maveric extra)
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from silos import checks, features as F, generate as G


# ---------------------------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------------------------
def _state(n_ues=(10, 20, 30, 40)):
    return {
        "power_dbm": [40.0] * 4, "tilt_deg": [6.0] * 4, "ue_count": list(n_ues),
        "rsrp_mean_dbm": [-80.0, -85.0, -90.0, -95.0], "rsrp_p10_dbm": [-95.0] * 4,
        "sinr_mean_db": [8.0, 7.0, 6.0, 5.0], "sinr_p10_db": [1.0, 0.0, -1.0, -2.0],
        "alarm": [False] * 4, "asleep": [False] * 4, "coverage_ratio": 0.8, "sla_ratio": 0.9,
        "hour": 13.0, "phase": 0.5, "extra": {"ue_displacement_m_mean": 3.0},
    }


def test_layout_matches_the_pooled_dataset_contract():
    assert F.N_INPUTS == 38 and len(F.INPUT_COLUMNS) == 38
    assert F.INPUT_COLUMNS[:4] == [f"user_share[{i}]" for i in range(4)]
    assert F.INPUT_COLUMNS[4:8] == [f"rsrp_mean_dbm[{i}]" for i in range(4)]
    assert F.INPUT_COLUMNS[24:28] == [f"tilt_deg[{i}]" for i in range(4)]
    assert F.INPUT_COLUMNS[28:30] == ["coverage_ratio", "sla_ratio"]
    assert F.INPUT_COLUMNS[30:] == [f"d_power_db[{i}]" for i in range(4)] + [f"d_tilt_deg[{i}]" for i in range(4)]
    assert F.LABELS == ("d_coverage_ratio", "d_sla_ratio", "d_sinr_p10_db", "d_sinr_mean_db")


def test_extraction_uses_shares_not_counts_and_ignores_forbidden_state_fields():
    x = F.state_change_to_inputs(_state(), [0, 2, 0, 0], [0, 0, -1, 0])
    assert x.shape == (38,)
    np.testing.assert_allclose(x[:4], [0.1, 0.2, 0.3, 0.4])
    assert x[F.col("d_power_db", 1)] == 2.0 and x[F.col("d_tilt_deg", 2)] == -1.0
    # a state with other hour/phase/alarm/extra/counts but the same shares gives the same row
    s2 = _state((1, 2, 3, 4))
    s2.update(hour=2.0, phase=0.1, alarm=[True, False, False, False], extra={})
    np.testing.assert_allclose(F.state_change_to_inputs(s2, [0, 2, 0, 0], [0, 0, -1, 0]), x)
    assert not {c.split("[")[0] for c in F.INPUT_COLUMNS} & F.FORBIDDEN_INPUTS
    assert F.check_input_ranges(x) == []


def test_reported_values_outside_the_fixed_ranges_are_clipped_in_inputs_only():
    s = _state()
    s["sinr_p10_db"] = [-33.0, 0.0, 0.0, 0.0]
    x = F.state_change_to_inputs(s, [0] * 4, [0] * 4)
    assert x[F.col("sinr_p10_db", 0)] == -30.0 and F.check_input_ranges(x) == []
    assert F.cluster_kpis(s)[2] == pytest.approx(-3.3)  # labels use the raw state


def test_range_check_flags_bad_values():
    x = F.state_change_to_inputs(_state(), [0] * 4, [0] * 4)
    x[F.col("power_dbm", 0)] = 50.0
    x[F.col("coverage_ratio")] = float("nan")
    assert set(F.check_input_ranges(x)) == {"power_dbm[0]", "coverage_ratio"}


def test_delta_labels_and_cluster_kpis():
    a, b = _state(), _state()
    b["coverage_ratio"], b["sinr_mean_db"] = 0.7, [8.0, 7.0, 6.0, 4.0]
    d = F.delta_labels(a, b)
    assert d[0] == pytest.approx(0.1) and d[1] == 0.0 and d[2] == 0.0
    assert d[3] == pytest.approx(0.4 * 1.0)  # share 0.4 x 1 dB
    assert F.cluster_kpis(a)[3] == pytest.approx(0.1 * 8 + 0.2 * 7 + 0.3 * 6 + 0.4 * 5)


# ---------------------------------------------------------------------------------------------
# seeds
# ---------------------------------------------------------------------------------------------
def test_seed_split_is_by_hash_and_picks_are_disjoint():
    tr, te = G.pick_seeds(100000, 20, "train"), G.pick_seeds(100000, 20, "test")
    assert not set(tr) & set(te)
    assert all(int(hashlib.sha256(str(s).encode()).hexdigest(), 16) % 10 >= 2 for s in tr)
    assert all(int(hashlib.sha256(str(s).encode()).hexdigest(), 16) % 10 < 2 for s in te)


# ---------------------------------------------------------------------------------------------
# generator on a tiny config
# ---------------------------------------------------------------------------------------------
def tiny_cfg():
    cfg = G.load_config()
    cfg["steps_per_cluster"] = 20
    for s in cfg["silos"].values():
        s["clusters"] = 1
    cfg["silos"]["A"]["clusters"] = 2  # proportional rows: A has twice the rows of B and C
    cfg["test"]["clusters_per_slice"] = 1
    cfg["test"]["steps_per_cluster"] = 10
    return cfg


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny")
    cfg = tiny_cfg()
    m1 = G.generate(cfg, root / "a", workers=4, manifest_path=root / "a_manifest.json")
    m2 = G.generate(cfg, root / "b", workers=2, manifest_path=root / "b_manifest.json")
    return root, cfg, m1, m2


def test_generation_is_deterministic(tiny):
    root, _, m1, m2 = tiny
    assert [(f["path"], f["sha256"]) for f in m1["files"]] == [(f["path"], f["sha256"]) for f in m2["files"]]
    assert len(m1["files"]) == 14


def test_generated_rows_follow_the_design(tiny):
    root, cfg, m1, _ = tiny
    for split in ("disjoint", "overlap"):
        a = pd.read_csv(root / "a" / split / "A" / "train.csv")
        b = pd.read_csv(root / "a" / split / "B" / "train.csv")
        assert len(a) == 2 * len(b) == 40
        assert list(a.columns) == F.COLUMNS
    d = pd.read_csv(root / "a" / "disjoint" / "A" / "train.csv")
    assert d["start_power_dbm"].between(23, 30).all()
    c = pd.read_csv(root / "a" / "disjoint" / "C" / "train.csv")
    assert c["start_power_dbm"].between(43, 46).all() and set(c["moving"]) == {0} and set(c["n_ues"]) == {100}
    o = pd.read_csv(root / "a" / "overlap" / "C" / "train.csv")
    assert o["start_power_dbm"].between(23, 46).all()
    v = pd.read_csv(root / "a" / "disjoint" / "shared_test" / "test.csv")
    first = v[(v["segment"] == "viavi_like") & (v["step"] % 5 == 0)]
    assert (first[[f"power_dbm[{i}]" for i in range(4)]] == 40.0).all().all()
    assert (first[[f"tilt_deg[{i}]" for i in range(4)]] == 6.0).all().all()
    assert (d["clipped"] == 0).all()
    z = d[d["change_kind"] == "zero"]
    assert (z[list(F.LABELS)] == 0).all().all()


def test_no_seed_of_any_train_file_is_in_a_test_file(tiny):
    root, _, m1, _ = tiny
    train = {s["seed"] for f in m1["files"] if f["role"] == "train" for s in f["seeds"]}
    test = {s["seed"] for f in m1["files"] if f["role"] in ("test", "shared_test") for s in f["seeds"]}
    assert train and test and not train & test


# ---------------------------------------------------------------------------------------------
# checks: pass on good data, fail on hand-broken data
# ---------------------------------------------------------------------------------------------
def _copy_case(tiny, tmp_path):
    root, _, m1, _ = tiny
    data = tmp_path / "data"
    import shutil

    shutil.copytree(root / "a", data)
    man = tmp_path / "manifest.json"
    man.write_text(json.dumps(m1))
    return data, man, json.loads(man.read_text())


def _rewrite(data, man_path, manifest, rel, df):
    """Replace a file and bring the manifest's sha256 and rows in line, so only the other checks can fire."""
    df.to_csv(data / rel, index=False, lineterminator="\n")
    for f in manifest["files"]:
        if f["path"] == rel:
            f["sha256"], f["rows"] = G.file_sha256(data / rel), len(df)
    man_path.write_text(json.dumps(manifest))


def _failed(rep):
    return {name for name, ok, _ in rep.results if not ok}


def test_checks_pass_on_good_data(tiny, tmp_path):
    data, man, _ = _copy_case(tiny, tmp_path)
    rep = checks.run_checks(data, man, config_path=None)
    assert rep.ok, [r for r in rep.results if not r[1]]


def test_cli_exit_codes(tiny, tmp_path):
    data, man, manifest = _copy_case(tiny, tmp_path)
    assert checks.main(["--data", str(data), "--manifest", str(man), "--config", str(tmp_path / "none.yaml")]) == 0
    p = data / "disjoint/A/train.csv"
    p.write_text(p.read_text().replace(",", ";", 1))
    assert checks.main(["--data", str(data), "--manifest", str(man), "--config", str(tmp_path / "none.yaml")]) == 1


def test_checks_catch_a_changed_file(tiny, tmp_path):
    data, man, _ = _copy_case(tiny, tmp_path)
    df = pd.read_csv(data / "disjoint/B/train.csv")
    df.loc[3, "d_sla_ratio"] += 0.01
    df.to_csv(data / "disjoint/B/train.csv", index=False, lineterminator="\n")
    assert any("sha256" in n for n in _failed(checks.run_checks(data, man, None)))


def test_checks_catch_out_of_range_values(tiny, tmp_path):
    data, man, manifest = _copy_case(tiny, tmp_path)
    df = pd.read_csv(data / "disjoint/B/train.csv")
    df.loc[2, "power_dbm[1]"] = 60.0
    _rewrite(data, man, manifest, "disjoint/B/train.csv", df)
    assert any("ranges" in n for n in _failed(checks.run_checks(data, man, None)))


def test_checks_catch_a_forbidden_field(tiny, tmp_path):
    data, man, manifest = _copy_case(tiny, tmp_path)
    df = pd.read_csv(data / "overlap/C/train.csv")
    df["hour"] = 3.0
    _rewrite(data, man, manifest, "overlap/C/train.csv", df)
    failed = _failed(checks.run_checks(data, man, None))
    assert any("forbidden" in n for n in failed) and any("schema" in n for n in failed)


def test_checks_catch_a_train_test_seed_leak(tiny, tmp_path):
    data, man, manifest = _copy_case(tiny, tmp_path)
    test_seed = int(pd.read_csv(data / "disjoint/B/test.csv")["seed"].iloc[0])
    df = pd.read_csv(data / "disjoint/B/train.csv")
    df["seed"] = test_seed
    _rewrite(data, man, manifest, "disjoint/B/train.csv", df)
    failed = _failed(checks.run_checks(data, man, None))
    assert any("overlap" in n for n in failed)


def test_checks_catch_unproportional_rows(tiny, tmp_path):
    data, man, manifest = _copy_case(tiny, tmp_path)
    df = pd.read_csv(data / "disjoint/A/train.csv").iloc[:30]
    _rewrite(data, man, manifest, "disjoint/A/train.csv", df)
    assert any("proportional" in n for n in _failed(checks.run_checks(data, man, None)))


def test_checks_catch_an_operating_point_violation(tiny, tmp_path):
    data, man, manifest = _copy_case(tiny, tmp_path)
    df = pd.read_csv(data / "disjoint/A/train.csv")
    df["start_power_dbm"] = 40.0
    _rewrite(data, man, manifest, "disjoint/A/train.csv", df)
    assert any("operating points" in n for n in _failed(checks.run_checks(data, man, None)))


def test_checks_catch_a_nonzero_label_on_a_zero_change_row(tiny, tmp_path):
    data, man, manifest = _copy_case(tiny, tmp_path)
    df = pd.read_csv(data / "disjoint/C/train.csv")
    i = df.index[df["change_kind"] == "zero"][0]
    df.loc[i, "d_coverage_ratio"] = 0.2
    _rewrite(data, man, manifest, "disjoint/C/train.csv", df)
    assert any("zero-change" in n for n in _failed(checks.run_checks(data, man, None)))
