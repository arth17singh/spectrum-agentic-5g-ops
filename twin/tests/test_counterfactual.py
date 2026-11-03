import pytest

from twin._change import canon_change, parse_change
from twin.kpis import headline
from twin.replay import ReplayTwin

CHANGE = [{"cell": 1, "d_power_db": 2.0, "d_tilt_deg": -1.0}]


def at_step(n, scenario="nominal", seed=7, **kw):
    t = ReplayTwin(scenario, seed, **kw)
    t.reset()
    for i in range(n):
        t.apply([{"cell": i % 4, "d_power_db": 1.0, "d_tilt_deg": 0.5}])
    return t


@pytest.mark.parametrize("zero", [[], "", [{"cell": 2, "d_power_db": 0.0, "d_tilt_deg": 0.0}], "cell=0;d_power_db=+0.0;d_tilt_deg=+0.0"])
def test_zero_change_gives_zero_delta(zero):
    t = at_step(4)
    delta = t.counterfactual_delta(zero)
    assert set(delta) == {"coverage_ratio", "sla_ratio", "sinr_p10_db", "sinr_mean_db"}
    assert all(v == 0.0 for v in delta.values())


def test_counterfactual_equals_independent_replay():
    live = at_step(5, "congestion", 3)
    cf = live.counterfactual(CHANGE)
    # independent replay of the log, then each future as a real step
    with_change = ReplayTwin.from_log("congestion", 3, None, live.action_log)
    with_change.apply(CHANGE)
    no_change = ReplayTwin.from_log("congestion", 3, None, live.action_log)
    no_change.apply([])
    assert cf["after"] == headline(with_change.state)
    assert cf["baseline"] == headline(no_change.state) == live.noop_kpis()
    assert cf["delta"] == {k: cf["after"][k] - cf["baseline"][k] for k in cf["delta"]}
    assert any(v != 0.0 for v in cf["delta"].values())  # the change did something
    assert cf["clamped"] is False


def test_counterfactual_does_not_mutate_live_twin():
    live = at_step(3)
    state, log, step = live.state, [dict(e) for e in live.action_log], live.step
    live.counterfactual(CHANGE)
    live.noop_kpis()
    assert live.state == state and live.action_log == log and live.step == step
    # the future after the probes is the same as for a twin that was never probed
    fresh = at_step(3)
    assert live.apply(CHANGE) == fresh.apply(CHANGE)
    assert live.action_log == fresh.action_log


def test_counterfactual_flags_clipped_change_and_works_on_viavi_like():
    t = at_step(2)
    cf = t.counterfactual([{"cell": 0, "d_power_db": 50.0}])
    assert cf["clamped"] is True and "power_delta_clip:0" in cf["rules"]
    assert cf["filtered"]["power_delta_db"][0] == 8.0
    v = ReplayTwin("viavi_stationary", 1, "viavi-like")
    v.reset()
    assert all(x == 0.0 for x in v.counterfactual_delta([]).values())
    assert v.counterfactual_delta("cell=0;d_power_db=+3.0;d_tilt_deg=+0.0")["sinr_mean_db"] != 0.0


def test_string_form_equals_list_form():
    a, b = at_step(2), at_step(2)
    items = [{"cell": 3, "d_power_db": -1.5, "d_tilt_deg": 2.0}, {"cell": 1, "d_power_db": 2.0, "d_tilt_deg": -1.0}]
    text = "cell=3;d_power_db=-1.5;d_tilt_deg=+2.0|cell=1;d_power_db=+2.0;d_tilt_deg=-1.0"
    assert a.apply(items) == b.apply(text)
    assert a.action_log[-1]["change"] == b.action_log[-1]["change"]


def test_canonical_roundtrip_and_rejects():
    items = [{"cell": 3, "d_power_db": -1.5, "d_tilt_deg": 2.0}, {"cell": 1, "d_power_db": 0.0, "d_tilt_deg": -0.04}]
    text = canon_change(items)
    assert text == "cell=1;d_power_db=+0.0;d_tilt_deg=+0.0|cell=3;d_power_db=-1.5;d_tilt_deg=+2.0"
    assert canon_change(text) == text
    assert parse_change(text)[1] == {"cell": 3, "d_power_db": -1.5, "d_tilt_deg": 2.0}
    assert parse_change("") == []
    for bad in ("cell=1", "cell=a;d_power_db=+1.0;d_tilt_deg=+0.0", "d_power_db=+1.0;cell=1;d_tilt_deg=+0.0",
                "cell=1;d_power_db=+1.0;d_tilt_deg=+0.0;x=1", "cell=9;d_power_db=+1.0;d_tilt_deg=+0.0"):
        with pytest.raises(ValueError):
            t = ReplayTwin("nominal", 1)
            t.reset()
            t.apply(bad)


def test_shared_twin_is_exact_and_lease_is_enforced():
    live = at_step(6)
    plain = ReplayTwin.from_log("nominal", 7, None, live.action_log)
    s1 = ReplayTwin.from_log("nominal", 7, None, live.action_log, shared=True)
    assert s1.state == plain.state == live.state
    s2 = ReplayTwin.from_log("nominal", 7, None, live.action_log[:3], shared=True)  # same underlying twin
    assert s2.state == ReplayTwin.from_log("nominal", 7, None, live.action_log[:3]).state
    with pytest.raises(RuntimeError):
        s1.apply([])
    with pytest.raises(RuntimeError):
        _ = s1.state
    # a cached call after the superseded one is exact again; counterfactual works on the shared twin
    s3 = ReplayTwin.cached("nominal", 7)
    s3.reset()
    s3.apply(CHANGE)
    cf = s3.counterfactual(CHANGE)
    ref = ReplayTwin("nominal", 7)
    ref.reset()
    ref.apply(CHANGE)
    assert cf == ref.counterfactual(CHANGE)
