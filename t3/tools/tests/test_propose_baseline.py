"""ran___propose with a policy that reads the pre-incident baseline (remediation-v2, t1/rl/policy.py "masked_delta").

The contract: ran___seed_case records ``baseline_step`` (1: the state after commissioning, for every seeded incident,
including the harness's later operator steps); ran___propose replays the case once, capturing the baseline and the
current state, each with the twin's per-cell shares, and calls ``propose(state, baseline=...)``; a case without a
baseline step is refused, never crashed. A policy without ``needs_baseline`` (ppo_sm_50k) is served exactly as
before: the real-twin test replays answers recorded at the deployed commit and compares them byte for byte, and
the parity test replays what the GATE's v2 operator proposed on dev incidents (fixtures/gen_*.py say how).
"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from t3.tools import app, store
from t3.tools.tests.test_ran_tools import World

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = Path(__file__).with_name("fixtures")
LOG2 = ["cell=1;d_power_db=+1.0;d_tilt_deg=+0.0", "cell=2;d_power_db=-3.0;d_tilt_deg=+0.0"]


@pytest.fixture(autouse=True)
def _reset():
    yield
    app.set_deps(None)


# ------------------------------------------------------------------ the step constant and seed_case
def test_incident_fault_step_is_the_declared_episode():
    episode = json.loads((ROOT / "t3/evals/targets.json").read_text())["episode"]
    assert app.INCIDENT_FAULT_STEP == episode["fault_step"] and app.BASELINE_STEP == episode["fault_step"] - 1


def _seed(w, case, log, **kw):
    return w.call("ran___seed_case", case_id=case, scenario="nominal", seed=7, preset="official",
                  start_action_log_json=json.dumps(log), **kw)


def test_seed_case_records_the_baseline_step_for_incidents_and_later_operator_steps():
    w = World()
    assert _seed(w, "case-b0000001-i00s0", LOG2)["seeded"] is True
    assert w.case_doc("case-b0000001-i00s0")["baseline_step"] == 1
    # a later operator step: the start log also carries the earlier proposals; the baseline stays step 1
    assert _seed(w, "case-b0000001-i00s2", LOG2 + [[], "cell=0;d_power_db=+1.0;d_tilt_deg=+0.0"])["seeded"]
    assert w.case_doc("case-b0000001-i00s2")["baseline_step"] == 1
    # shorter than the fault step: no incident, no baseline
    assert _seed(w, "case-b0000001-i01s0", LOG2[:1])["seeded"] is True
    assert "baseline_step" not in w.case_doc("case-b0000001-i01s0")
    # an explicit baseline step (direct calls only; the Gateway schema does not carry it) is checked
    assert _seed(w, "case-b0000001-i02s0", LOG2, baseline_step=0)["seeded"] is True
    assert w.case_doc("case-b0000001-i02s0")["baseline_step"] == 0
    assert "baseline_step" in _seed(w, "case-b0000001-i03s0", LOG2, baseline_step=2)["error"]
    assert "baseline_step" in _seed(w, "case-b0000001-i04s0", LOG2, baseline_step="x")["error"]


def test_a_policy_without_needs_baseline_answers_as_before():
    w = World()
    _seed(w, "case-b0000002-i00s0", LOG2)
    out = w.call("ran___propose", case_id="case-b0000002-i00s0")
    assert out["change"] == "cell=2;d_power_db=+0.0;d_tilt_deg=+2.0" and out["item_id"] == "step-2"
    assert "baseline_step" not in out and "error" not in out


# ------------------------------------------------------------------ a toy twin with per-cell shares
class ShareTwin:
    """W2's ReplayTwin surface the baseline path uses: reset, apply (logs filtered), state and kpis()['per_cell'] with
    power_dbm and the coverage/sla shares (None where a cell serves no UEs)."""

    def __init__(self, scenario, seed, config):
        self.config = {"preset": config} if isinstance(config, str) else config
        self.action_log = []

    def reset(self):
        self.state = {"step": 0, "power_dbm": [40.0] * 4, "tilt_deg": [6.0] * 4}
        self.action_log = []
        return copy.deepcopy(self.state)

    def apply(self, change):
        for c in change:
            self.state["power_dbm"][c["cell"]] += float(c["d_power_db"])
            self.state["tilt_deg"][c["cell"]] += float(c["d_tilt_deg"])
        self.state["step"] += 1
        self.action_log.append({"step": len(self.action_log), "change": copy.deepcopy(change),
                                "filtered": {"n": len(change)}, "rules": [], "clamped": False})
        return copy.deepcopy(self.state)

    def kpis(self):
        s = self.state["step"]
        return {"coverage_ratio": 0.9, "sla_ratio": 0.9,
                "per_cell": [{"cell": i, "power_dbm": self.state["power_dbm"][i], "tilt_deg": self.state["tilt_deg"][i],
                              "coverage_share": None if (i == 3 and s >= 2) else 0.5 + 0.01 * s,
                              "sla_share": None if (i == 3 and s >= 2) else 0.4 + 0.01 * s}
                             for i in range(4)]}


class BaselinePolicy:
    model_id, version, needs_baseline = "t1-remediation-bc-delta", "v2-test", True

    def __init__(self):
        self.calls = []

    def propose(self, state, baseline=None):
        self.calls.append((copy.deepcopy(state), copy.deepcopy(baseline)))
        return [{"cell": 2, "d_power_db": 3.0, "d_tilt_deg": 0.0}] if state["step"] == 2 else []


def test_a_baseline_policy_gets_the_step_1_state_and_both_share_lists():
    pol = BaselinePolicy()
    w = World(policy=pol, twin=ShareTwin)
    _seed(w, "case-b0000003-i00s0", LOG2)
    out = w.call("ran___propose", case_id="case-b0000003-i00s0")
    assert out["change"] == "cell=2;d_power_db=+3.0;d_tilt_deg=+0.0" and out["baseline_step"] == 1
    assert out["policy"] == {"model_id": "t1-remediation-bc-delta", "version": "v2-test"}
    state, base = pol.calls[-1]
    assert state["step"] == 2 and base["step"] == 1
    assert base["power_dbm"] == [40.0, 41.0, 40.0, 40.0]           # after commissioning, before the fault
    assert state["coverage_share"][:3] == pytest.approx([0.52] * 3) and state["coverage_share"][3] is None
    assert base["sla_share"] == pytest.approx([0.41] * 4)
    # a later operator step seeded with the earlier proposal: the baseline is still step 1; a no-op is an answer
    _seed(w, "case-b0000003-i00s1", LOG2 + [out["change"]])
    out2 = w.call("ran___propose", case_id="case-b0000003-i00s1")
    assert out2["no_change"] is True and out2["baseline_step"] == 1 and pol.calls[-1][1]["step"] == 1


def test_a_baseline_policy_refuses_a_case_without_a_baseline_step():
    pol = BaselinePolicy()
    w = World(policy=pol, twin=ShareTwin)
    out = w.call("ran___propose", case_id="case-b0000004")          # a chat case: never seeded at an incident
    assert "no recorded pre-incident step" in out["error"] and pol.calls == []
    _seed(w, "case-b0000004-i01s0", LOG2[:1])                        # seeded, but before any fault step
    assert "no recorded pre-incident step" in w.call("ran___propose", case_id="case-b0000004-i01s0")["error"]


def test_a_twin_without_per_cell_shares_is_refused_for_a_baseline_policy():
    class NoShares(ShareTwin):
        kpis = None

    w = World(policy=BaselinePolicy(), twin=NoShares)
    _seed(w, "case-b0000005-i00s0", LOG2)
    assert "per-cell" in w.call("ran___propose", case_id="case-b0000005-i00s0")["error"]


# ------------------------------------------------------------------ the real twin and the real policies
REAL = all(importlib.util.find_spec(m) for m in ("ran_rl", "stable_baselines3", "torch"))
real = pytest.mark.skipif(not REAL, reason="needs ran_rl, stable_baselines3 and torch (the twin and the PPO models)")


def _real_tools(tmp_path, artifact: str) -> app.Tools:
    from t1.rl.policy import load
    from twin.kpis import kpis as twin_kpis
    from twin.replay import ReplayTwin

    pol = load(ROOT / "t1/rl/artifacts" / artifact)
    return app.Tools(app.Deps(store=store.from_env({"T3_STORE": f"local:{tmp_path}"}),
                              twin_factory=lambda s, seed, cfg: ReplayTwin(s, seed, cfg), kpis=twin_kpis,
                              policy=lambda: pol, synthetic=True, now=lambda: "2026-11-04T00:00:00.000Z"))


@real
def test_ppo_sm_50k_answers_are_byte_identical_to_the_deployed_commit(tmp_path):
    golden = json.loads((FIXTURES / "ppo_propose_golden.json").read_text())
    assert golden["generated_at_commit"].startswith("614fc3b373de")
    tools = _real_tools(tmp_path, "ppo_sm_50k.zip")
    for i, case in enumerate(golden["cases"]):
        p0 = case["answers"][0]
        log1 = case["start_log"] + ([p0["change"]] if p0.get("change") else [[]])
        for s, log in enumerate((case["start_log"], log1)):
            cid = f"case-golden-{i:02d}s{s}"
            tools.seed_case({"case_id": cid, "scenario": case["scenario"], "seed": case["seed"],
                             "preset": case["preset"], "start_action_log_json": json.dumps(log)})
            assert tools.propose({"case_id": cid}) == case["answers"][s], (case["incident_id"], s)


@real
def test_v2_proposes_what_the_gate_operator_proposed(tmp_path):
    golden = json.loads((FIXTURES / "v2_gate_golden.json").read_text())
    import hashlib

    assert hashlib.sha256((ROOT / "t1/rl/artifacts/remediation_v2.zip").read_bytes()).hexdigest() == \
        golden["artifact_sha256"]
    tools = _real_tools(tmp_path, "remediation_v2.zip")
    for i, case in enumerate(golden["cases"]):
        log = list(case["start_log"])
        for s, want in enumerate(case["step_texts"]):
            cid = f"case-gate-{i:02d}s{s}"
            tools.seed_case({"case_id": cid, "scenario": case["scenario"], "seed": case["seed"],
                             "preset": case["preset"], "start_action_log_json": json.dumps(log)})
            out = tools.propose({"case_id": cid})
            assert out["baseline_step"] == 1 and out["policy"]["model_id"] == "t1-remediation-bc-delta"
            got = out["change"] if out.get("change") else ""
            assert got == want, (case["incident_id"], s, got, want)
            log.append(want if want else [])


@real
def test_v2_refuses_a_chat_case_instead_of_crashing(tmp_path):
    tools = _real_tools(tmp_path, "remediation_v2.zip")
    with pytest.raises(app.ToolError, match="no recorded pre-incident step"):
        tools.propose({"case_id": "case-chat0001"})
