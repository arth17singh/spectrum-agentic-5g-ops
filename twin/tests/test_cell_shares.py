"""Per-cell coverage_share / sla_share (for the Twin View)."""

import pytest

from twin.kpis import kpis
from twin.presets import VIAVI_LIKE, VIAVI_SCENARIO
from twin.replay import ReplayTwin


def stepped(scenario, seed, config=None):
    t = ReplayTwin(scenario, seed, config)
    t.reset()
    for i in range(4):
        t.apply([{"cell": i, "d_power_db": 2.0, "d_tilt_deg": -1.0}])
    return t


def weighted(k, key):
    cells = [c for c in k["per_cell"] if c["ue_count"] > 0]
    n = sum(c["ue_count"] for c in cells)
    return sum(c["ue_count"] * c[key] for c in cells) / n


@pytest.mark.parametrize(
    "config,scenario",
    [(None, "nominal"), (None, "congestion"), (None, "interference"), (VIAVI_LIKE, VIAVI_SCENARIO)],
)
def test_shares_in_range_and_average_to_network_ratios(config, scenario):
    for seed in range(5):
        t = stepped(scenario, seed, config)
        k = t.kpis()
        for c in k["per_cell"]:
            for key in ("coverage_share", "sla_share"):
                assert c[key] is None or 0.0 <= c[key] <= 1.0
        assert weighted(k, "coverage_share") == pytest.approx(k["coverage_ratio"], abs=1e-12)
        assert weighted(k, "sla_share") == pytest.approx(k["sla_ratio"], abs=1e-12)
        # headline numbers are unchanged by the shares; state-only kpis() has no per-UE data
        plain = kpis(t.state)
        assert {x: k[x] for x in k if x != "per_cell"} == {x: plain[x] for x in plain if x != "per_cell"}
        assert "coverage_share" not in plain["per_cell"][0]


def test_cell_with_no_ues_has_none_shares():
    t = ReplayTwin("outage_cell_2", 3)  # cell 0 off from step 6: serves nobody
    t.reset()
    for _ in range(7):
        t.apply([])
    k = t.kpis()
    assert k["per_cell"][0]["ue_count"] == 0
    assert k["per_cell"][0]["coverage_share"] is None and k["per_cell"][0]["sla_share"] is None
    assert all(c["coverage_share"] is not None for c in k["per_cell"][1:])
    assert weighted(k, "coverage_share") == pytest.approx(k["coverage_ratio"], abs=1e-12)


def test_counterfactual_carries_per_cell_views_without_touching_the_twin():
    t = stepped("nominal", 2)
    state_before = t.state
    change = [{"cell": 1, "d_power_db": 3.0, "d_tilt_deg": 0.0}]
    cf = t.counterfactual(change)
    assert t.state == state_before
    for key in ("after_kpis", "baseline_kpis"):
        k = cf[key]
        assert weighted(k, "coverage_share") == pytest.approx(k["coverage_ratio"], abs=1e-12)
        assert weighted(k, "sla_share") == pytest.approx(k["sla_ratio"], abs=1e-12)
    real = ReplayTwin.from_log("nominal", 2, None, t.action_log)
    real.apply(change)
    assert real.kpis()["per_cell"] == cf["after_kpis"]["per_cell"]
