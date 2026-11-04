"""Live Fix's twin-side helper: side effects of a held change on the PREFLIGHT copy, and the percell op."""
import importlib.util

import pytest

from t3.evals import confidence
from t3.evals import faults as F
from t3.evals import livefix_worker as W
from t3.evals.tests.test_percell import HEALTHY, OBSERVED, row

REAL_TWIN = importlib.util.find_spec("twin.replay") is not None
INVERSE = "cell=2;d_power_db=+0.0;d_tilt_deg=-5.0"


@pytest.mark.skipif(not REAL_TWIN, reason="needs the Maveric twin (ran-rl)")
def test_the_observed_broad_boost_is_predicted_to_hurt_neighbours_and_the_inverse_is_not():
    inc = F.make_incident("tilt_drift", 2, preset="official")
    boost = W.side_effects(inc, inc["start_log"], OBSERVED)
    by = {c["cell"]: c for c in boost["cells"]}
    assert boost["recovered_with"] is False and boost["overshoot_with"]["power_above_healthy_db"] > 10
    assert by[3]["worse"] and by[3]["health_with"] == "red"        # red because of the change: the card holds it
    assert by[0]["worse"] and by[0]["health_with"] == "amber"
    assert by[2]["health_with"] == "green" and not by[2]["worse"]
    fix = W.side_effects(inc, inc["start_log"], INVERSE)
    assert fix["recovered_with"] is True and not any(c["worse"] for c in fix["cells"])
    assert {c["health_with"] for c in fix["cells"]} == {"green"}
    for se in (boost, fix):
        assert se["twin"]["seed_offset"] == confidence.PREFLIGHT_SEED_OFFSET and se["version"] == "percell-v1"
        assert set(se["cells"][0]["with"]) >= {"power_dbm", "tilt_deg", "coverage_share", "sla_share", "sinr_p10_db"}


@pytest.mark.skipif(not REAL_TWIN, reason="needs the Maveric twin (ran-rl)")
def test_side_effects_never_use_the_graders_seed():
    seen = []

    class Spy:
        def __init__(self, scenario, seed, preset):
            seen.append(seed)
            from t3.evals.twin_backend import ReplayTwin
            self._t = ReplayTwin(scenario, seed, preset)

        def __getattr__(self, name):
            return getattr(self._t, name)

    inc = F.make_incident("pa_fault", 1, preset="official")
    W.side_effects(inc, inc["start_log"], "cell=1;d_power_db=+6.0;d_tilt_deg=+0.0", twin_cls=Spy)
    assert seen and set(seen) == {int(inc["seed"]) + confidence.PREFLIGHT_SEED_OFFSET}


def test_side_effects_refuse_a_log_past_the_grade_step():
    inc = {"scenario": "nominal", "seed": 1, "preset": "official", "commissioning": [], "fault": []}
    with pytest.raises(ValueError):
        W.side_effects(inc, [""] * 50, "")


def test_percell_op_is_percell_verdict():
    after = [row(0, sla=0.5), row(1), row(2), row(3)]
    v = W.handle("percell", {"cells_after": after, "cells_healthy": HEALTHY, "cells_before": HEALTHY,
                             "fault_cells": [2]})
    assert v["version"] == "percell-v1" and v["recovered"] is False and v["collateral"] == [0]
