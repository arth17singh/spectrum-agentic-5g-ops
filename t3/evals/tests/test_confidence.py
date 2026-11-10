"""Confidence score and calibration unit, offline with a fake twin and a fake predictor (fast)."""
from __future__ import annotations

import json

import pytest

from t3.evals import calibration as cal
from t3.evals import confidence as cf

TARGETS = {"episode": {"fault_step": 2, "operator_steps": 3, "grade_step": 5},
           "graded_kpis": ["coverage_ratio", "sla_ratio", "sinr_p10_db"],
           "tolerance": {"coverage_ratio": 0.02, "sla_ratio": 0.02, "sinr_p10_db": 0.5},
           "material_gap": {"coverage_ratio": 0.01, "sla_ratio": 0.01, "sinr_p10_db": 0.25}}
INC = {"incident_id": "inc-x", "scenario": "nominal", "seed": 100, "preset": "official",
       "healthy_kpis_by_step": [None] * 5 + [{"coverage_ratio": 0.9, "sla_ratio": 0.9, "sinr_p10_db": 1.0}],
       "unfixed_kpis_by_step": [None] * 5 + [{"coverage_ratio": 0.8, "sla_ratio": 0.8, "sinr_p10_db": 0.0}]}
FAULT = "cell=0;d_power_db=+0.0;d_tilt_deg=+5.0"


class FakeTwin:
    """KPIs fall with cell 0's tilt above 0 (the fault) and recover when it comes back down."""
    seeds: list[int] = []

    def __init__(self, scenario, seed, preset):
        FakeTwin.seeds.append(seed)
        self.tilt0 = 0.0

    def reset(self):
        return self.state

    @property
    def state(self):
        hurt = max(0.0, self.tilt0) / 5.0
        return {"coverage_ratio": 0.9 - 0.1 * hurt, "sla_ratio": 0.9 - 0.1 * hurt,
                "sinr_p10_db": 1.0 - 1.0 * hurt, "sinr_mean_db": 5.0}

    def apply(self, change):
        for c in change:
            if c["cell"] == 0:
                self.tilt0 += c["d_tilt_deg"]
        return self.state


class FakePredictor:
    model_id, version = "fake-federated", "v0"

    def __init__(self, d):
        self.d = d

    def predict(self, state, change):
        return dict(self.d)


@pytest.fixture(autouse=True)
def identity_graded(monkeypatch):
    monkeypatch.setattr(cf, "graded", lambda s: dict(s))
    monkeypatch.setattr(cal, "graded", lambda s: dict(s))
    FakeTwin.seeds = []


def test_combine_renormalises_missing_signals():
    c, used = cf.combine({"preflight": 1.0, "federated": 0.0, "agreement": None, "lesson": None})
    assert used == pytest.approx({"preflight": 0.5 / 0.7, "federated": 0.2 / 0.7})
    assert c == pytest.approx(0.5 / 0.7)
    with pytest.raises(ValueError):
        cf.combine({"preflight": None})


def test_tier_thresholds():
    assert cf.tier(0.9, True) == "one-person"
    assert cf.tier(0.85, True) == "one-person"
    assert cf.tier(0.6, True) == "two-person"
    assert cf.tier(0.49, True) == "hold"
    assert cf.tier(0.99, False) == "hold"
    assert cf.tier(None, True) == "hold"


def test_signal_mapping_is_fixed():
    assert cf.signal_of(1.0) == pytest.approx(0.5)
    assert cf.z_of({"coverage_ratio": 1.0, "sla_ratio": 0.0, "sinr_p10_db": 0.0}, TARGETS["tolerance"]) \
        == pytest.approx(cf.Z_CLIP / 3)


def test_good_fix_scores_high_and_preflight_uses_a_mismatched_seed():
    out = cf.score_proposal(INC, ["", FAULT], "cell=0;d_power_db=+0.0;d_tilt_deg=-5.0", targets=TARGETS,
                            predictor=FakePredictor({"coverage_ratio": 0.1, "sla_ratio": 0.1, "sinr_p10_db": 1.0,
                                                     "sinr_mean_db": 0.0}),
                            twin_factory=FakeTwin, agreement=1.0, lesson=None)
    assert out["preflight_ok"] is True
    assert out["confidence"] > 0.85 and out["tier"] == "one-person"
    assert 100 + cf.PREFLIGHT_SEED_OFFSET in FakeTwin.seeds        # preflight: not the grader's seed
    assert out["breakdown"]["preflight"]["twin"]["seed"] != INC["seed"]
    assert set(out["weights_used"]) == {"preflight", "federated", "agreement"}
    json.dumps(out)                                                  # JSON-serialisable
    assert out["explain"].startswith("Confidence ")


def test_harmful_change_is_held():
    out = cf.score_proposal(INC, ["", FAULT], "cell=0;d_power_db=+0.0;d_tilt_deg=+3.0", targets=TARGETS,
                            twin_factory=FakeTwin, use_federated=False)
    assert out["preflight_ok"] is False and out["tier"] == "hold"
    assert out["breakdown"]["preflight"]["harm"]
    assert "harm" in out["explain"]


def test_no_change_and_bad_inputs():
    out = cf.score_proposal(INC, ["", FAULT], "", targets=TARGETS, twin_factory=FakeTwin)
    assert out["confidence"] is None and out["tier"] == "hold"
    with pytest.raises(ValueError):
        cf.score_proposal(INC, ["", FAULT], "cell=0;d_power_db=+0.0;d_tilt_deg=-5.0", targets=TARGETS,
                          twin_factory=FakeTwin, use_federated=False, agreement=1.5)
    with pytest.raises(ValueError):
        cf.preflight(INC, ["", FAULT, "", "", ""], "cell=0;d_power_db=+0.0;d_tilt_deg=-5.0", TARGETS,
                     twin_factory=FakeTwin)


# ---------- calibration ----------
def test_brier_ece_reliability():
    p, y = [0.9, 0.9, 0.1, 0.5], [1, 0, 0, 1]
    assert cal.brier(p, y) == pytest.approx((0.01 + 0.81 + 0.01 + 0.25) / 4)
    bins = cal.reliability(p, y)
    assert [b["n"] for b in bins] == [1, 0, 1, 0, 2]
    assert bins[4]["success_rate"] == 0.5
    assert cal.ece(p, y) == pytest.approx(0.25 * 0.1 + 0.25 * 0.5 + 0.5 * 0.4)
    assert cal.reliability([1.0], [1])[-1]["n"] == 1                  # 1.0 lands in the last bin
    assert cal.ece([0.8] * 5, [1, 1, 1, 1, 0]) == pytest.approx(0.0)


def test_outcome_uses_the_graders_own_twin():
    good = cal.outcome(INC, ["", FAULT], "cell=0;d_power_db=+0.0;d_tilt_deg=-5.0", TARGETS, twin_factory=FakeTwin)
    bad = cal.outcome(INC, ["", FAULT], "cell=1;d_power_db=+1.0;d_tilt_deg=+0.0", TARGETS, twin_factory=FakeTwin)
    assert good["success"] is True and bad["success"] is False
    assert FakeTwin.seeds == [100, 100]


def test_evaluate_skips_no_change_rows():
    rows = [{"incident_id": "inc-x", "step": 0, "case_id": "c0", "start_action_log": ["", FAULT],
             "change_text": "cell=0;d_power_db=+0.0;d_tilt_deg=-5.0"},
            {"incident_id": "inc-x", "step": 1, "case_id": "c1", "start_action_log": ["", FAULT, ""],
             "change_text": ""}]
    scorer = lambda inc, log, ch, targets: {"confidence": 0.8, "tier": "two-person", "preflight_ok": True,  # noqa: E731
                                            "breakdown": {"preflight": {"signal": 0.8}, "lesson": None}}
    grader = lambda inc, log, ch, targets: {"success": True, "recovery": 1.0}  # noqa: E731
    out = cal.evaluate(rows, {"inc-x": INC}, TARGETS, scorer=scorer, grader=grader)
    s = out["summary"]
    assert s["n"] == 1 and s["skipped"] == 1
    assert s["brier"] == pytest.approx(0.04) and s["ece"] == pytest.approx(0.2)
    assert out["items"][0]["signals"] == {"preflight": 0.8, "lesson": None}
