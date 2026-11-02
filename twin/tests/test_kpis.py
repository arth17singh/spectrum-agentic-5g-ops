import json
import math

import pytest

from twin.kpis import headline, kpis
from twin.presets import VIAVI_LIKE, VIAVI_SCENARIO
from twin.replay import ReplayTwin

KEYS = {"coverage_ratio", "sla_ratio", "sinr_p10_db", "sinr_mean_db", "rsrp_p10_dbm", "per_cell"}


def stepped(scenario, seed, config=None):
    t = ReplayTwin(scenario, seed, config)
    t.reset()
    for i in range(4):
        t.apply([{"cell": i, "d_power_db": 2.0, "d_tilt_deg": -1.0}])
    return t


def test_keys_ranges_and_per_cell():
    t = stepped("nominal", 3)
    s = t.state
    k = kpis(s)
    assert KEYS <= set(k)
    assert 0 <= k["coverage_ratio"] <= 1 and 0 <= k["sla_ratio"] <= 1
    assert k["coverage_ratio"] == s["coverage_ratio"] and k["sla_ratio"] == s["sla_ratio"]  # twin fields
    assert -30 <= k["sinr_p10_db"] <= 40 and -30 <= k["sinr_mean_db"] <= 40
    assert -140 <= k["rsrp_p10_dbm"] <= -40
    assert k["n_ues"] == 200 and len(k["per_cell"]) == 4
    assert [c["cell"] for c in k["per_cell"]] == [0, 1, 2, 3]
    for key in ("power_dbm", "tilt_deg", "ue_count", "sinr_mean_db", "sinr_p10_db", "alarm"):
        assert key in k["per_cell"][0]
    json.dumps(k)  # JSON-safe
    assert set(headline(s)) == {"coverage_ratio", "sla_ratio", "sinr_p10_db", "sinr_mean_db"}


def test_empty_and_alarmed_cells_do_not_poison_pooled_values():
    t = ReplayTwin("outage_cell_2", 3)
    t.reset()
    for _ in range(7):
        t.apply([])
    s = t.state
    assert s["ue_count"][0] == 0 and s["sinr_p10_db"][0] == -30.0  # floor value reported by the twin
    k = kpis(s)
    assert k["sinr_p10_db"] > -30 and math.isfinite(k["rsrp_p10_dbm"])


@pytest.mark.parametrize("config,scenario", [(None, "nominal"), (None, "congestion"), (None, "interference"), (VIAVI_LIKE, VIAVI_SCENARIO)])
def test_pooled_values_vs_exact(config, scenario):
    errs_sinr, errs_rsrp = [], []
    for seed in range(6):
        t = stepped(scenario, seed, config)
        k, e = kpis(t.state), t.exact_kpis()
        assert k["sinr_mean_db"] == pytest.approx(e["sinr_mean_db"], abs=1e-9)  # exact by construction
        errs_sinr.append(k["sinr_p10_db"] - e["sinr_p10_db"])
        errs_rsrp.append(k["rsrp_p10_dbm"] - e["rsrp_p10_dbm"])
    # documented accuracy of the p10 estimate (kpis.py): SINR within ~1.2 dB, RSRP within ~6 dB
    assert max(abs(x) for x in errs_sinr) < 1.5
    assert max(abs(x) for x in errs_rsrp) < 6.5


def test_p10_moves_with_the_network():
    """The estimate has to react to a real change: boosting every cell's tilt (more downtilt) shifts SINR."""
    a = ReplayTwin("nominal", 5)
    a.reset()
    b = ReplayTwin("nominal", 5)
    b.reset()
    for _ in range(3):
        a.apply([])
        b.apply([{"cell": c, "d_tilt_deg": 4.0} for c in range(4)])
    assert kpis(a.state)["sinr_p10_db"] != kpis(b.state)["sinr_p10_db"]
