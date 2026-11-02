"""The stand-in twin (only where ran_rl is installed): replay is exact and instances share nothing mutable."""

import pytest

pytest.importorskip("ran_rl.twin.maveric.twin")

from t3.trust.standin_twin import StandinReplayTwin, kpis  # noqa: E402

CH = [[{"cell": 0, "d_power_db": 2.0, "d_tilt_deg": 1.0}], [], [{"cell": 2, "d_power_db": -3.0, "d_tilt_deg": -1.0}]]


@pytest.mark.parametrize("config", ["official", "viavi-like"])
def test_replay_is_exact_and_isolated(config):
    a = StandinReplayTwin("nominal", 5, config)
    a.reset()
    other = StandinReplayTwin("nominal", 5, config)  # interleaved instance must not disturb `a`
    other.reset()
    for c in CH:
        sa = a.apply(c)
        other.apply([{"cell": 1, "d_power_db": 4.0, "d_tilt_deg": 2.0}])
    b = StandinReplayTwin.from_log("nominal", 5, config, a.action_log)
    assert kpis(b.observe()) == kpis(sa)
    assert sa["power_dbm"][0] == pytest.approx((40.0 if config == "viavi-like" else 23.0) + 2.0)


def test_fork_equals_from_log_and_is_independent():
    a = StandinReplayTwin("nominal", 9, "official")
    a.reset()
    for c in CH:
        a.apply(c)
    f = a.fork()
    r = StandinReplayTwin.from_log("nominal", 9, "official", a.action_log)
    assert kpis(f.apply([])) == kpis(r.apply([]))
    before = kpis(a.observe())
    f.apply([{"cell": 3, "d_power_db": 5.0, "d_tilt_deg": 3.0}])
    assert kpis(a.observe()) == before and len(a.action_log) == len(CH)
    after = kpis(a.apply([]))
    assert after == kpis(StandinReplayTwin.from_log("nominal", 9, "official", a.action_log[:-1]).apply([]))


def test_viavi_like_is_stationary():
    t = StandinReplayTwin("nominal", 1, "viavi-like")
    s0 = kpis(t.reset())
    s1 = kpis(t.apply([]))
    assert s0["coverage_ratio"] == s1["coverage_ratio"] and s0["sinr_p10_db"] == s1["sinr_p10_db"]
