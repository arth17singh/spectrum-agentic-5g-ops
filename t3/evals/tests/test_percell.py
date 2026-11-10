import importlib.util

import pytest

from t3.evals import faults as F
from t3.evals import percell as P

REAL_TWIN = importlib.util.find_spec("twin.replay") is not None
# The change Live Fix applied in run t3.livefix-20261104T073206Z-9695 (antenna tilt drift on cell 2).
OBSERVED = ("cell=0;d_power_db=-0.9;d_tilt_deg=-0.5|cell=1;d_power_db=+3.7;d_tilt_deg=+0.0|"
            "cell=2;d_power_db=+8.0;d_tilt_deg=-2.4|cell=3;d_power_db=+2.4;d_tilt_deg=+2.0")


def row(cell, power=23.0, tilt=6.0, ues=10, cov=1.0, sla=1.0, sinr=3.0, alarm=False, asleep=False):
    return {"cell": cell, "power_dbm": power, "tilt_deg": tilt, "ue_count": ues, "sinr_p10_db": sinr,
            "rsrp_p10_dbm": -90.0, "coverage_share": cov, "sla_share": sla, "alarm": alarm, "asleep": asleep}


HEALTHY = [row(c) for c in range(4)]


def test_identical_to_healthy_is_recovered_with_no_overshoot():
    v = P.percell_verdict([dict(r) for r in HEALTHY], HEALTHY, [dict(r) for r in HEALTHY])
    assert v["version"] == "percell-v1" and v["recovered"] is True
    assert v["degraded"] == [] and v["collateral"] == []
    assert v["overshoot"]["power_above_healthy_db"] == 0.0
    assert v["overshoot"]["coverage_above_healthy"] == 0.0
    assert v["overshoot"]["coverage_source"] == "ue-weighted-cells"
    assert all(c["health"] == "green" and c["worse_than_before"] is False for c in v["cells"])


def test_amber_on_one_cell_means_not_recovered():
    after = [dict(r) for r in HEALTHY]
    after[0]["sinr_p10_db"] = 3.0 - 1.5          # amber line is 1 dB, red 3 dB (faults.HEALTH_THRESHOLDS)
    v = P.percell_verdict(after, HEALTHY)
    assert v["recovered"] is False and v["degraded"] == [0]
    assert v["cells"][0]["health"] == "amber"
    assert v["cells"][0]["worse_than_before"] is None          # no cells_before given


def test_fix_that_hurts_neighbours_is_collateral_not_recovery():
    # The Live Fix pattern: the faulted cell 2 is back, while cells 0 and 3 end worse than before the fix.
    before = [row(0), row(1), row(2, tilt=11.0, cov=0.3, sla=0.3, sinr=-2.0, alarm=True), row(3)]
    after = [row(0, power=22.1, sinr=1.5), row(1, power=26.7), row(2, power=31.0, tilt=8.6),
             row(3, power=25.4, tilt=8.0, sla=0.9)]
    v = P.percell_verdict(after, HEALTHY, before, fault_cells=[2])
    assert v["recovered"] is False
    assert v["degraded"] == [0, 3] and v["collateral"] == [0, 3]
    cells = {c["cell"]: c for c in v["cells"]}
    assert cells[2]["health"] == "green" and cells[2]["fault_cell"] is True
    assert cells[0]["worse_than_before"] and any("before the fix" in r for r in cells[0]["reasons"])
    assert v["overshoot"]["power_above_healthy_db"] == pytest.approx(3.7 + 8.0 + 2.4)
    assert v["overshoot"]["power_off_healthy_db"] == pytest.approx(0.9 + 3.7 + 8.0 + 2.4)
    assert v["overshoot"]["tilt_off_healthy_deg"] == pytest.approx(2.6 + 2.0)


def test_faulted_cell_is_never_collateral_and_returning_to_healthy_is_not_harm():
    before = [row(0, sla=1.0), row(1), row(2, cov=0.3, sla=0.3, alarm=True), row(3)]
    after = [row(0, sla=1.0), row(1), row(2, cov=0.3, sla=0.3, alarm=True), row(3)]
    v = P.percell_verdict(after, HEALTHY, before, fault_cells=[2])
    assert v["degraded"] == [2] and v["collateral"] == []
    # A neighbour that looked better than healthy during the fault, and returns to healthy after it.
    healthy = [row(0, sla=0.9)] + HEALTHY[1:]
    before2 = [row(0, sla=1.0)] + HEALTHY[1:]
    after2 = [row(0, sla=0.9)] + HEALTHY[1:]
    v2 = P.percell_verdict(after2, healthy, before2)
    assert v2["recovered"] is True and v2["collateral"] == []
    assert any("not counted" in r for r in v2["cells"][0]["reasons"])


def test_a_kpi_drop_within_the_same_colour_counts_as_worse():
    before = [row(0, sinr=3.0 - 1.2)] + HEALTHY[1:]           # amber already
    after = [row(0, sinr=3.0 - 2.5)] + HEALTHY[1:]            # still amber, 1.3 dB worse
    v = P.percell_verdict(after, HEALTHY, before)
    assert v["cells"][0]["health"] == "amber" and v["cells"][0]["worse_than_before"] is True
    assert v["collateral"] == [0]


def test_new_alarm_counts_but_a_standing_alarm_does_not():
    before = [row(0, alarm=True)] + HEALTHY[1:]
    after = [row(0, alarm=True)] + HEALTHY[1:]
    v = P.percell_verdict(after, HEALTHY, before)
    assert v["cells"][0]["worse_than_before"] is False
    v2 = P.percell_verdict([row(0, alarm=True)] + HEALTHY[1:], HEALTHY, [dict(r) for r in HEALTHY])
    assert v2["cells"][0]["worse_than_before"] is True


def test_thresholds_default_to_faults_and_can_be_overridden():
    after = [row(0, sinr=3.0 - 0.6)] + HEALTHY[1:]
    assert P.percell_verdict(after, HEALTHY)["recovered"] is True
    strict = {"red": dict(F.HEALTH_THRESHOLDS["red"]), "amber": dict(F.HEALTH_THRESHOLDS["amber"], sinr_p10_db=0.5)}
    v = P.percell_verdict(after, HEALTHY, thresholds=strict)
    assert v["recovered"] is False and v["thresholds"] == "custom"


def test_network_headlines_win_over_ue_weighted_cells():
    v = P.percell_verdict(HEALTHY, HEALTHY, network_after={"coverage_ratio": 0.725},
                          network_healthy={"coverage_ratio": 0.605})
    assert v["overshoot"]["coverage_above_healthy"] == pytest.approx(0.12)
    assert v["overshoot"]["coverage_source"] == "network"


def test_mismatched_cells_are_refused():
    with pytest.raises(ValueError):
        P.percell_verdict(HEALTHY[:3], HEALTHY)
    with pytest.raises(ValueError):
        P.percell_verdict(HEALTHY, HEALTHY, HEALTHY[:2])


def test_cells_from_kpis_result_and_from_object():
    k = {"per_cell": [dict(r) for r in HEALTHY]}

    class FakeTwin:
        def kpis(self):
            return k

    assert P.cells_from_twin(k) == P.cells_from_twin(FakeTwin())
    assert [r["cell"] for r in P.cells_from_twin(k)] == [0, 1, 2, 3]
    with pytest.raises(TypeError):
        P.cells_from_twin(42)


@pytest.mark.skipif(not REAL_TWIN, reason="needs the in-process Maveric twin (ran-rl)")
def test_live_fix_run_is_reproduced_and_judged_per_cell():
    incident = F.make_incident("tilt_drift", 2, preset="official")
    v = P.grade_incident(incident, [OBSERVED, "", ""])
    after = {r["cell"]: r for r in v["cells_after"]}
    # The configuration Live Fix showed after the change (cell 2: +31.0 dBm, 8.6 deg).
    assert after[2]["power_dbm"] == pytest.approx(31.0) and after[2]["tilt_deg"] == pytest.approx(8.6)
    assert after[0]["power_dbm"] == pytest.approx(22.1) and after[3]["tilt_deg"] == pytest.approx(8.0)
    assert v["recovered"] is False and 2 not in v["degraded"] and {0, 3} <= set(v["degraded"])
    assert v["overshoot"]["power_above_healthy_db"] > 10.0
    fix = P.grade_incident(incident, ["cell=2;d_power_db=+0.0;d_tilt_deg=-5.0", "", ""])
    assert fix["recovered"] is True and fix["degraded"] == [] and fix["collateral"] == []
    assert fix["overshoot"]["power_off_healthy_db"] == 0.0 and fix["overshoot"]["tilt_off_healthy_deg"] == 0.0
    assert P.grade_incident(incident, ["", "", ""])["recovered"] is False


@pytest.mark.skipif(not REAL_TWIN, reason="needs the in-process Maveric twin (ran-rl)")
def test_cells_from_twin_matches_faults_rows_and_state_path():
    from t3.evals.twin_backend import ReplayTwin

    t = ReplayTwin("nominal", 7002, "official")
    t.reset()
    t.apply([])
    rows = P.cells_from_twin(t)
    assert rows == F._cell_rows(t)
    from_state = P.cells_from_twin(t.state)
    assert [r["power_dbm"] for r in from_state] == [r["power_dbm"] for r in rows]
    assert all(r["coverage_share"] is None for r in from_state)
