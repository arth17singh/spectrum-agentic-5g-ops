import copy
import json

import pytest

from twin.presets import VIAVI_LIKE, VIAVI_SCENARIO
from twin.replay import ReplayMismatch, ReplayTwin


def changes(n):
    return [
        [{"cell": i % 4, "d_power_db": (i % 5) - 2.0, "d_tilt_deg": 1.0 - (i % 3)}]
        if i % 4 != 3
        else []
        for i in range(n)
    ]


def run(scenario="nominal", seed=7, config=None, n=10, **kw):
    t = ReplayTwin(scenario, seed, config, **kw)
    t.reset()
    for c in changes(n):
        t.apply(c)
    return t


def test_10_step_live_run_equals_from_log_replay():
    live = run()
    assert len(live.action_log) == 10
    replay = ReplayTwin.from_log("nominal", 7, None, live.action_log)
    assert replay.state == live.state  # exact, every float
    assert replay.action_log == live.action_log
    assert replay.step == 10


def test_replay_through_json_roundtrip_and_record():
    live = run("congestion", 3)
    rec = json.loads(json.dumps(live.to_record()))
    again = ReplayTwin.from_record(rec)
    assert again.state == live.state
    assert rec["config"] == live.config


@pytest.mark.parametrize("config,scenario", [(None, "outage_cell_2"), (VIAVI_LIKE, VIAVI_SCENARIO)])
def test_replay_exact_for_presets_and_outage(config, scenario):
    live = run(scenario, 11, config, n=12)
    assert ReplayTwin.from_log(scenario, 11, config, live.action_log).state == live.state


def test_reuse_models_does_not_change_results_or_leak_between_instances():
    a = run(reuse_models=False)
    b = run(reuse_models=True)
    c = run(reuse_models=True)
    assert a.state == b.state == c.state
    # two live instances sharing models stay independent when interleaved
    x, y = ReplayTwin("nominal", 1), ReplayTwin("nominal", 1)
    x.reset(), y.reset()
    x.apply([{"cell": 0, "d_power_db": 5.0, "d_tilt_deg": 0.0}])
    y.apply([])
    assert x.state != y.state
    assert y.state == ReplayTwin.from_log("nominal", 1, None, y.action_log).state


def test_different_seed_or_changes_differ_same_everything_equal():
    assert run(seed=1).state != run(seed=2).state
    assert run(seed=1).state == run(seed=1).state


def test_safety_filter_clamps_and_logs():
    t = ReplayTwin("nominal", 4)
    t.reset()
    p0, tilt0 = t.state["power_dbm"][1], t.state["tilt_deg"][1]
    s = t.apply([{"cell": 1, "d_power_db": 50.0, "d_tilt_deg": -50.0}])
    entry = t.action_log[0]
    # SDK bounds: power delta <= +8, tilt delta within +-5 and tilt >= 4 absolute
    assert s["power_dbm"][1] == p0 + 8.0
    assert s["tilt_deg"][1] == max(tilt0 - 5.0, 4.0)
    assert entry["clamped"] is True
    assert entry["requested"]["power_delta_db"][1] == 50.0
    assert entry["requested"]["tilt_delta_deg"][1] == -50.0
    assert entry["filtered"]["power_delta_db"][1] == 8.0
    assert entry["filtered"]["tilt_delta_deg"][1] == s["tilt_deg"][1] - tilt0
    assert "power_delta_clip:1" in entry["rules"] and "tilt_delta_clip:1" in entry["rules"]
    # the in-range change is not flagged
    t.apply([{"cell": 2, "d_power_db": 1.0, "d_tilt_deg": 0.5}])
    assert t.action_log[1]["clamped"] is False and t.action_log[1]["rules"] == []


def test_alarmed_cell_is_frozen():
    t = ReplayTwin("outage_cell_2", 3)  # cell 0 alarmed from step 6
    t.reset()
    for _ in range(7):
        t.apply([])
    assert t.state["alarm"][0] is True
    before = t.state["power_dbm"][0]
    t.apply([{"cell": 0, "d_power_db": 3.0, "d_tilt_deg": 1.0}])
    assert t.state["power_dbm"][0] == before
    assert "alarm_freeze:0" in t.action_log[-1]["rules"]


def test_non_finite_request_is_neutralised_and_replays():
    t = ReplayTwin("nominal", 2)
    t.reset()
    t.apply([{"cell": 0, "d_power_db": float("nan"), "d_tilt_deg": float("inf")}])
    e = t.action_log[0]
    assert e["filtered"]["power_delta_db"][0] == 0.0
    assert any(r.startswith("nonfinite_power:0") for r in e["rules"])
    log = json.loads(json.dumps(copy.deepcopy(t.action_log)))  # strings "nan"/"inf" survive JSON
    assert ReplayTwin.from_log("nominal", 2, None, log).state == t.state


def test_duplicate_cells_add_and_bad_input_rejected():
    a, b = ReplayTwin("nominal", 9), ReplayTwin("nominal", 9)
    a.reset(), b.reset()
    a.apply([{"cell": 1, "d_power_db": 1.0}, {"cell": 1, "d_power_db": 2.0}])
    b.apply([{"cell": 1, "d_power_db": 3.0}])
    assert a.state == b.state
    for bad in ([{"cell": 4}], [{"cell": -1}], [{"cell": 1.5}], [{"d_power_db": 1}], [{"cell": 0, "x": 1}], "no"):
        with pytest.raises((ValueError, TypeError)):
            a.apply(bad)
    assert a.step == 1  # rejected changes leave no trace in the log


def test_apply_before_reset_and_reset_clears_log():
    t = ReplayTwin("nominal", 1)
    with pytest.raises(RuntimeError):
        t.apply([])
    with pytest.raises(RuntimeError):
        _ = t.state
    t.reset()
    t.apply([])
    first = t.state
    t.reset()
    assert t.action_log == [] and t.step == 0
    t.apply([])
    assert t.state == first


def test_tampered_log_is_detected():
    live = run(n=6)
    log = copy.deepcopy(live.action_log)
    log[2]["change"][0]["d_power_db"] += 1.0  # a different request than the one that was measured
    with pytest.raises(ReplayMismatch):
        ReplayTwin.from_log("nominal", 7, None, log)
    log = copy.deepcopy(live.action_log)
    log[4]["kpis"]["coverage_ratio"] += 0.01
    with pytest.raises(ReplayMismatch):
        ReplayTwin.from_log("nominal", 7, None, log)
    ReplayTwin.from_log("nominal", 7, None, log, verify=False)  # opt-out works


def test_unknown_scenario_raises():
    with pytest.raises(ValueError):
        ReplayTwin("nope", 1).reset()
