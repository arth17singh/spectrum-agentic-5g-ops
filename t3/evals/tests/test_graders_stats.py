import math

import pytest

from t3.evals.changes import canon, nonzero, parse
from t3.evals.graders import action_stats, grade_kpis, load_targets
from t3.evals.stats import paired_bootstrap

T = load_targets()
H = {"coverage_ratio": 0.60, "sla_ratio": 0.80, "sinr_p10_db": -2.0}
U = {"coverage_ratio": 0.50, "sla_ratio": 0.70, "sinr_p10_db": -4.0}


def test_full_recovery_is_one_and_success():
    g = grade_kpis(dict(H), H, U, T)
    assert g["recovery"] == pytest.approx(1.0)
    assert g["success"] is True


def test_no_action_is_zero_and_fails():
    g = grade_kpis(dict(U), H, U, T)
    assert g["recovery"] == pytest.approx(0.0)
    assert g["success"] is False


def test_half_recovery_and_tolerance_boundary():
    final = {"coverage_ratio": 0.55, "sla_ratio": 0.75, "sinr_p10_db": -3.0}
    g = grade_kpis(final, H, U, T)
    assert g["recovery"] == pytest.approx(0.5)
    assert g["success"] is False
    # exactly at healthy - tolerance counts as success
    edge = {"coverage_ratio": 0.58, "sla_ratio": 0.78, "sinr_p10_db": -2.5}
    assert grade_kpis(edge, H, U, T)["success"] is True
    just_out = dict(edge, sla_ratio=0.7799)
    assert grade_kpis(just_out, H, U, T)["success"] is False


def test_overshoot_and_damage_are_clipped():
    better = {"coverage_ratio": 0.9, "sla_ratio": 0.95, "sinr_p10_db": 5.0}
    assert grade_kpis(better, H, U, T)["recovery"] == pytest.approx(1.0)
    worse = {"coverage_ratio": 0.2, "sla_ratio": 0.3, "sinr_p10_db": -20.0}
    assert grade_kpis(worse, H, U, T)["recovery"] == pytest.approx(-1.0)


def test_immaterial_kpi_is_excluded_but_still_checked_for_success():
    u = dict(U, sla_ratio=0.795)  # gap 0.005 < material 0.01
    final = {"coverage_ratio": 0.60, "sla_ratio": 0.70, "sinr_p10_db": -2.0}
    g = grade_kpis(final, H, u, T)
    assert g["per_kpi"]["sla_ratio"]["recovery"] is None
    assert g["recovery"] == pytest.approx(1.0)  # mean of coverage and sinr only
    assert g["success"] is False  # but SLA fell below healthy - tolerance


def test_action_stats_counts_changes_wrong_cells_and_clamps():
    log = [
        {"step": 2, "change": [{"cell": 1, "d_power_db": 6.0, "d_tilt_deg": 0.0}], "rules": []},
        {"step": 3, "change": [], "rules": []},
        {"step": 4, "change": [{"cell": 2, "d_power_db": 0.0, "d_tilt_deg": -5.0},
                               {"cell": 1, "d_power_db": 0.0, "d_tilt_deg": 0.0}], "rules": ["tilt_abs_clip:2"]},
    ]
    s = action_stats(log, {1})
    assert s == {"changes_used": 2, "wrong_cell_actions": 1, "clamped": True, "rules": ["step4:tilt_abs_clip:2"]}


def test_bootstrap_constant_difference_has_zero_width():
    g = paired_bootstrap({"a": 1.0, "b": 0.5}, {"a": 0.5, "b": 0.0, "c": 9.0})
    assert g["value"] == pytest.approx(0.5)
    assert g["ci_low"] == pytest.approx(0.5) and g["ci_high"] == pytest.approx(0.5)
    assert g["paired_n"] == 2 and g["excludes_zero"]


def test_bootstrap_is_deterministic_and_brackets_the_mean():
    cand = {str(i): float(i % 3) for i in range(30)}
    base = {str(i): 1.0 for i in range(30)}
    a, b = paired_bootstrap(cand, base), paired_bootstrap(cand, base)
    assert a == b
    assert a["ci_low"] <= a["value"] <= a["ci_high"]
    assert a["value"] == pytest.approx(0.0)
    assert not a["excludes_zero"]


def test_bootstrap_without_shared_items_is_none():
    assert paired_bootstrap({"a": 1.0}, {"b": 1.0}) is None


def test_canonical_change_strings_round_trip():
    ch = [{"cell": 2, "d_power_db": 0.0, "d_tilt_deg": -5.0}, {"cell": 0, "d_power_db": 6.0, "d_tilt_deg": 0.0}]
    s = canon(ch)
    assert s == "cell=0;d_power_db=+6.0;d_tilt_deg=+0.0|cell=2;d_power_db=+0.0;d_tilt_deg=-5.0"
    assert canon(parse(s)) == s
    assert nonzero("") == [] and nonzero(None) == []
    assert canon([{"cell": 1, "d_power_db": -0.0, "d_tilt_deg": 0.04}]) == "cell=1;d_power_db=+0.0;d_tilt_deg=+0.0"
    assert math.isclose(parse(s)[1]["d_tilt_deg"], -5.0)
