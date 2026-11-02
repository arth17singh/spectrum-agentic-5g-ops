import math

import pytest

from t3.trust.demo import run_demo
from t3.trust.ladder import LABELS, ladder, ladder_result_record
from t3.trust.local_writer import LocalRecordWriter
from t3.trust.policy import PolicyClient
from t3.trust.retention import MIN_GAIN, bootstrap_mean_ci, bootstrap_retention, retention

from .conftest import CHANGES, FakePredictor, FakeTwin, fake_kpis


# ------------------------------------------------------------------ retention
def test_retention_guards():
    assert retention(0.1, 0.05) == pytest.approx(0.5)
    assert retention(0.1, -0.05) == pytest.approx(-0.5)  # reversed gain is reported, not clipped
    assert retention(0.1, 0.2) == pytest.approx(2.0)
    assert retention(None, 0.1) is None and retention(0.1, None) is None
    assert retention(0.0, 0.1) is None and retention(-0.2, -0.1) is None
    assert retention(MIN_GAIN, 0.1) is None and retention(MIN_GAIN * 2, 0.1) is not None
    assert retention(float("nan"), 0.1) is None and retention(0.1, float("inf")) is None
    assert retention(0.004, 0.002, min_gain=0.005) is None


def test_bootstrap_intervals():
    g1 = [0.1, 0.12, 0.08, 0.11, 0.09, 0.1]
    g2 = [x * 0.5 for x in g1]
    b = bootstrap_retention(g1, g2)
    assert b["point"] == pytest.approx(0.5) and b["ci"][0] == pytest.approx(0.5) and b["ci"][1] == pytest.approx(0.5)
    lo, hi = bootstrap_mean_ci(g1)
    assert lo <= sum(g1) / len(g1) <= hi
    with pytest.raises(ValueError):
        bootstrap_retention([0.1], [0.1, 0.2])
    assert bootstrap_retention([-0.1, 0.0], [0.1, 0.1])["point"] is None
    unstable = bootstrap_retention([0.5, -0.45, 0.002, -0.001], [0.1, 0.1, 0.1, 0.1])
    assert unstable["ci"] is None  # too many resamples fail the guard


# ------------------------------------------------------------------ ladder
def test_ladder_labels_and_fixed_shape_without_viavi():
    rungs = [{"name": "maveric", "factory": FakeTwin, "config": "official", "twin_kind": "maveric-inprocess"},
             {"name": "rehearsal", "factory": FakeTwin, "config": "viavi-like", "twin_kind": "maveric-inprocess"}]
    lad = ladder(FakePredictor("v1"), rungs, fake_kpis, seeds=[1, 2, 3], changes=CHANGES[:6])
    d = lad["details"]
    assert set(d) == {"rungs", "retention", "retention_ci", "model_id", "version"}
    assert [r["label"] for r in d["rungs"]] == ["MAVERIC", "REHEARSAL (Viavi-like Maveric)"]
    for r in d["rungs"]:
        assert set(r) == {"name", "label", "twin_kind", "n_cases", "gain", "gain_ci"}
        assert r["n_cases"] == 3 and len(r["gain_ci"]) == 2
    assert d["retention"] is None and d["retention_ci"] is None  # no VIAVI rung: null by rule
    assert d["rungs"][0]["gain"] > 0 and d["rungs"][1]["gain"] == pytest.approx(d["rungs"][0]["gain"] / 2)
    assert "NOT retention" in lad["settings_shift"]["what"]
    assert lad["settings_shift"]["point"] == pytest.approx(0.5)
    rec = ladder_result_record(lad, run_id="r1", n_items=6)
    assert rec["kind"] == "result" and rec["unit"] == "t3.ladder" and rec["details"] is d


def test_ladder_retention_is_viavi_over_rehearsal():
    rungs = [{"name": "maveric", "factory": FakeTwin, "config": "official", "twin_kind": "maveric-inprocess"},
             {"name": "rehearsal", "factory": FakeTwin, "config": "official", "twin_kind": "maveric-inprocess"},
             {"name": "viavi", "factory": FakeTwin, "config": "viavi-like", "twin_kind": "viavi"}]
    d = ladder(FakePredictor("v1"), rungs, fake_kpis, seeds=[1, 2, 3], changes=CHANGES[:6])["details"]
    assert d["rungs"][2]["label"] == "VIAVI"
    assert d["retention"] == pytest.approx(0.5) and d["retention_ci"] == pytest.approx([0.5, 0.5])


def test_ladder_rejects_unknown_rungs_and_degraded_has_no_retention():
    with pytest.raises(ValueError):
        ladder(FakePredictor("v1"), [{"name": "viavi-ish", "factory": FakeTwin, "config": None, "twin_kind": "x"}],
               fake_kpis, seeds=[1], changes=CHANGES[:2])
    assert set(LABELS.values()) == {"MAVERIC", "REHEARSAL (Viavi-like Maveric)", "VIAVI"}
    rungs = [{"name": "rehearsal", "factory": FakeTwin, "config": "official", "twin_kind": "maveric-inprocess"},
             {"name": "viavi", "factory": FakeTwin, "config": "viavi-like", "twin_kind": "viavi"}]
    d = ladder(FakePredictor("bad", sign=-1), rungs, fake_kpis, seeds=[1, 2], changes=CHANGES[:6])["details"]
    assert d["rungs"][0]["gain"] <= 0 and d["retention"] is None  # nothing to retain


# ------------------------------------------------------------------ demo
def test_demo_pauses_degraded_only(thresholds):
    w = LocalRecordWriter()
    pc = PolicyClient(clock=lambda: 7)
    s = run_demo(twin_factory=FakeTwin, kpis=fake_kpis, predictors=[FakePredictor("v1"), FakePredictor("v1-bad", -1)],
                 changes=CHANGES, writer=w, run_id="t3trust-test", thresholds=thresholds, policy_client=pc)
    h, d = s["cases"]
    assert h["paused_at_change"] is None and h["refused_after_pause"] == 0 and h["scored"] == len(CHANGES)
    assert d["paused_at_change"] is not None and d["paused_at_change"] <= thresholds["max_changes_to_pause"]
    assert d["refused_after_pause"] == len(CHANGES) - d["paused_at_change"]
    ctrl = w.records[("t3trust-test", "control")]
    assert len(ctrl) == 1 and ctrl[0]["version"] == "v1-bad" and ctrl[0]["by"] == "trust-gate"
    changes = w.records[("t3trust-test", "change")]
    assert len(changes) == len(CHANGES) + d["paused_at_change"]
    assert all(c["synthetic"] and c["scored"] and c["item_id"].startswith("step-") for c in changes)
    healthy_e = [c["error"]["summary"] for c in changes if c["predictor"]["version"] == "v1"]
    assert max(healthy_e) == pytest.approx(0.0, abs=1e-9)
    results = w.records[("t3trust-test", "result")]
    assert [r["details"]["version"] for r in results] == ["v1", "v1-bad"]
    th = results[0]["details"]["thresholds"]
    for k in ("tau", "scales", "pause_below", "alpha", "n_min", "window", "sha256"):
        assert k in th
    assert th["window"] == {"needed": 2, "of_last": 3}


def test_demo_marks_clipped_changes_unscored(thresholds):
    w = LocalRecordWriter()
    alarm_changes = [[{"cell": 3, "d_power_db": 2.0, "d_tilt_deg": 1.0}]] * 4
    s = run_demo(twin_factory=FakeTwin, kpis=fake_kpis, predictors=[FakePredictor("v1-bad", -1)],
                 changes=alarm_changes, writer=w, run_id="t3trust-clip", thresholds=thresholds, config="alarm")
    recs = w.records[("t3trust-clip", "change")]
    assert all(r["scored"] is False and r["unscored_reason"] == "clipped" for r in recs)
    assert s["cases"][0]["paused_at_change"] is None and s["cases"][0]["scored"] == 0
