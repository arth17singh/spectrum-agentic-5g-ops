from t3.trust.calibrate import propose, replay_rule
from t3.trust.gate import TrustGate


def seq(pairs):
    return [{"e": e, "direction_agree": None, "score": s} for e, s in pairs]


def test_proposal_never_pauses_healthy_and_catches_degraded(thresholds):
    g = TrustGate(None, thresholds)
    healthy = [seq([(0.3, 0.8)] * 10 + [(1.4, 0.3)] + [(0.3, 0.8)] * 5), seq([(0.5, 0.7)] * 16)]
    degraded = [seq([(2.5, 0.1)] * 16), seq([(0.4, 0.7), (2.0, 0.2), (2.2, 0.1)] + [(2.5, 0.1)] * 13)]
    out = propose(g, healthy, degraded)["best"]
    assert out["healthy_paused_episodes"] == 0
    assert out["degraded_paused_within_max"] == 2 and out["degraded_paused_at"] == [3, 3]
    assert out["tau"] >= 1.4 or out["pause_below"] < 0.3  # the lone healthy miss cannot trip both rules


def test_replay_rule_matches_gate(thresholds):
    g = TrustGate(None, thresholds)
    assert replay_rule(g, seq([(0.1, 0.9)] * 3 + [(5, 0.0), (5, 0.0)])) == 5
    assert replay_rule(g, seq([(0.1, 0.9)] * 20)) is None
