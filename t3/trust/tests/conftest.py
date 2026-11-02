"""Shared fakes for the trust tests: deterministic, no ran_rl, no AWS."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

KEYS = ("coverage_ratio", "sla_ratio", "sinr_p10_db", "sinr_mean_db")
# true per-dB / per-degree response of the fake network
RESP_P = {"coverage_ratio": 0.02, "sla_ratio": -0.01, "sinr_p10_db": -0.3, "sinr_mean_db": 0.1}
RESP_T = {"coverage_ratio": -0.03, "sla_ratio": 0.015, "sinr_p10_db": 0.4, "sinr_mean_db": 0.2}


class FakeTwin:
    """Deterministic 4-cell twin: KPIs = base + response(settings) + seeded per-step drift.
    config "viavi-like" halves the response (a harder rung); an alarmed cell 3 freezes (clips)."""

    kind = "fake"
    version = "fake-1"
    n_cells = 4

    def __init__(self, scenario, seed, config=None):
        self.scenario, self.seed, self.config = scenario, seed, config or "official"
        self.gain = 0.5 if self.config == "viavi-like" else 1.0
        self.action_log = []
        self._state = None

    def _drift(self, step):
        x = ((self.seed * 7919 + step * 104729) % 1000) / 1000.0 - 0.5
        return {k: 0.01 * x for k in KEYS}

    def _make(self, power, tilt, step):
        d = self._drift(step)
        k = {key: 0.5 + d[key] + self.gain * sum(RESP_P[key] * (p - 23) + RESP_T[key] * (t - 6)
                                                for p, t in zip(power, tilt)) for key in KEYS}
        return {"power_dbm": list(power), "tilt_deg": list(tilt), "ue_count": [50] * 4,
                "alarm": [False, False, False, self.config == "alarm"], "kpis": k}

    def reset(self):
        self.action_log = []
        self._state = self._make([23.0] * 4, [6.0] * 4, 0)
        return self._state

    def apply(self, change):
        p, t = list(self._state["power_dbm"]), list(self._state["tilt_deg"])
        for c in change:
            i = int(c["cell"])
            if self._state["alarm"][i]:
                continue  # frozen: the change is clipped
            p[i] += float(c.get("d_power_db", 0.0))
            t[i] += float(c.get("d_tilt_deg", 0.0))
        self.action_log.append([dict(c) for c in change])
        self._state = self._make(p, t, len(self.action_log))
        return self._state

    @classmethod
    def from_log(cls, scenario, seed, config, action_log):
        tw = cls(scenario, seed, config)
        tw.reset()
        for c in action_log:
            tw.apply(c)
        return tw


def fake_kpis(state):
    return dict(state["kpis"])


class FakeCFTwin(FakeTwin):
    """FakeTwin with ReplayTwin's ``counterfactual`` (the real path in stepping and ladder)."""

    def counterfactual(self, change):
        def trial(ch):
            tw = FakeTwin.from_log(self.scenario, self.seed, self.config, self.action_log)
            before = tw._state
            after = tw.apply(ch)
            return after, before

        after, before = trial(change)
        base, _ = trial([])
        clamped = any(before["alarm"][int(c["cell"])] for c in change)
        a, b = fake_kpis(after), fake_kpis(base)
        return {"baseline": b, "after": a, "delta": {k: a[k] - b[k] for k in KEYS}, "clamped": clamped}


class FakePredictor:
    """Returns KPI DELTAS like t2.predict: the true official-rung response times ``sign``
    (``-1`` = degraded); a zero change predicts exactly 0."""

    def __init__(self, version, sign=1.0, model_id="fake-resp"):
        self.model_id, self.version, self.sign = model_id, version, sign

    def predict(self, state, change):
        out = {k: 0.0 for k in KEYS}
        for c in change or []:
            for k in KEYS:
                out[k] += self.sign * (RESP_P[k] * c.get("d_power_db", 0.0) + RESP_T[k] * c.get("d_tilt_deg", 0.0))
        return out


CHANGES = [[{"cell": i % 3, "d_power_db": (2.0 if (i // 3) % 2 == 0 else -2.0), "d_tilt_deg": (1.0 if i % 2 else -1.0)}]
           for i in range(12)]


@pytest.fixture
def thresholds():
    return {
        "keys": ["coverage_ratio", "sla_ratio", "sinr_p10_db"],
        "scales": {"coverage_ratio": 0.05, "sla_ratio": 0.03, "sinr_p10_db": 0.5},
        "direction_deadband": {"coverage_ratio": 0.01, "sla_ratio": 0.01, "sinr_p10_db": 0.1},
        "weights": {"error": 0.5, "direction": 0.5},
        "tau": 1.0, "recent_window": 3, "recent_needed": 2,
        "alpha": 0.3, "warmup": 3, "watch_below": 0.6, "pause_below": 0.45,
        "max_changes_to_pause": 5,
    }
