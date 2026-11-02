import json

import numpy as np
import pytest

from t3.evals import incidents as I
from t3.evals import run as R
from t3.evals.graders import load_targets
from t3.evals.operators import NoAction, Oracle, PPO, RuleBaseline, StubPolicy, make
from t3.evals.swarm import DryRunSwarmClient, SwarmOperator, task_text
from t3.evals.twin_backend import ReplayTwin

T = load_targets()
INCS, DIGEST = I.load_test()
INC = INCS[0]


def _valid_change(ch, n=4):
    if ch is None:
        return True
    assert isinstance(ch, list)
    for item in ch:
        assert set(item) <= {"cell", "d_power_db", "d_tilt_deg"}
        assert 0 <= item["cell"] < n
        float(item.get("d_power_db", 0.0)), float(item.get("d_tilt_deg", 0.0))
    return True


@pytest.mark.parametrize("name", ["no_action", "rule_baseline", "config_revert", "ppo", "swarm"])
def test_operator_interface_conformance(name):
    op = make(name)
    for attr in ("name", "model_id", "synthetic", "act"):
        assert hasattr(op, attr)
    item = R.run_incident(INC, op, T)
    for k in ("item_id", "input_sha256", "output_sha256", "grader", "backend"):
        assert item[k]
    assert isinstance(item["passed"], bool)
    assert -1.0 <= item["score"] <= 1.0
    assert len(item["extra"]["change_texts"]) == T["episode"]["operator_steps"]
    json.dumps(item, allow_nan=False)


def test_operator_act_returns_valid_changes_on_a_live_state():
    t = ReplayTwin(INC["scenario"], INC["seed"], INC["preset"])
    t.reset()
    state = t.apply(INC["fault"])
    for op in (NoAction(), RuleBaseline(t.config["safety"]), PPO(StubPolicy())):
        assert _valid_change(op.act(state, []))


def test_no_action_lands_exactly_on_the_unfixed_counterfactual():
    item = R.run_incident(INC, NoAction(), T)
    assert item["score"] == pytest.approx(0.0, abs=1e-6)
    for k in T["graded_kpis"]:
        assert item["extra"]["final_kpis"][k] == pytest.approx(INC["unfixed_kpis_by_step"][-1][k], rel=1e-6, abs=1e-8)


def test_rule_baseline_is_the_sdk_controller():
    from ran_rl.env.config import SafetyBounds
    from ran_rl.env.observation import scale_action
    from ran_rl.evaluation.baseline import RuleBasedController

    t = ReplayTwin(INC["scenario"], INC["seed"], INC["preset"])
    t.reset()
    state = t.apply(INC["fault"])
    ch = RuleBaseline(t.config["safety"]).act(state, [])
    raw = scale_action(RuleBasedController(SafetyBounds()).act(state), SafetyBounds())
    vec = np.zeros(8)
    for i in ch:
        vec[i["cell"]] += i["d_power_db"]
        vec[4 + i["cell"]] += i["d_tilt_deg"]
    assert np.allclose(vec, raw)


def test_ppo_stub_is_labelled_synthetic():
    op = make("ppo")
    assert op.synthetic is True and "stub" in op.model_id


def test_oracle_with_the_inverse_in_its_grid_recovers():
    inc = next(i for i in INCS if len(i["fault"]) == 1)
    f = inc["fault"][0]
    inverse = [{"cell": f["cell"], "d_power_db": -f["d_power_db"], "d_tilt_deg": -f["d_tilt_deg"]}]
    wrong = [{"cell": (f["cell"] + 1) % 4, "d_power_db": -3.0, "d_tilt_deg": 0.0}]
    item = R.run_incident(inc, Oracle(grid=[wrong, inverse]), T)
    assert item["score"] > 0.5
    assert item["extra"]["changes_used"] >= 1


def test_swarm_adapter_speaks_canonical_strings_and_records_cases():
    client = DryRunSwarmClient()
    op = SwarmOperator(client, run_id="t3.evals.test")
    item = R.run_incident(INC, op, T, run_id="t3.evals.test")
    steps = item["extra"]["operator_extra"]["swarm_steps"]
    assert item["extra"]["operator_extra"]["case_id"].startswith("case-dryrun-")
    assert all(s["change_text"] == "" or s["change_text"].startswith("cell=") for s in steps)
    assert op.synthetic is True
    assert "run_id=t3.evals.test" in task_text(INC, "t3.evals.test")
    assert "cell" not in task_text(INC, "x").split("dB.")[0].lower()  # symptoms only, no fault cell


def test_swarm_refused_step_is_no_change():
    class Refusing:
        synthetic = True

        def open_case(self, case):
            assert case["start_action_log"] == [INC["commissioning"], INC["fault"]]
            return "case-x"

        def step(self, case_id, state, history):
            return {"change_text": "cell=0;d_power_db=+6.0;d_tilt_deg=+0.0", "applied": False}

    item = R.run_incident(INC, SwarmOperator(Refusing()), T)
    assert item["extra"]["changes_used"] == 0
    assert item["score"] == pytest.approx(0.0, abs=1e-6)


def test_end_to_end_run_writes_declared_items_result_and_gain(tmp_path):
    base = R.run("no_action", tmp_path, [], allow_dirty=True, run_id="t-noact")
    cand = R.run("swarm", tmp_path, ["t-noact"], allow_dirty=True, run_id="t-swarm")
    decl = json.loads((tmp_path / "declared" / "t-swarm.json").read_text())
    assert decl["test_set_sha256"] == DIGEST and decl["n_items"] == len(INCS)
    assert decl["synthetic"] is True and decl["baseline_run_id"] == "t-noact"
    items = [json.loads(x) for x in (tmp_path / "records" / "t-swarm" / "item.jsonl").read_text().splitlines()]
    assert len(items) == len(INCS) and all(i["run_id"] == "t-swarm" for i in items)
    res = json.loads((tmp_path / "results" / "t-swarm.json").read_text())
    g = res["details"]["gain"]
    assert set(g) == {"value", "ci_low", "ci_high", "baseline_run_id", "paired_n"}
    assert g["paired_n"] == len(INCS) and g["baseline_run_id"] == "t-noact"
    assert g["ci_low"] <= g["value"] <= g["ci_high"]
    assert g["value"] == pytest.approx(cand["details"]["mean_recovery"] - base["details"]["mean_recovery"])
    with pytest.raises(FileExistsError):
        R.run("no_action", tmp_path, [], allow_dirty=True, run_id="t-noact")


def test_swarm_measured_kpis_that_disagree_with_our_twin_are_recorded():
    class Lying:
        synthetic = True

        def open_case(self, case):
            return "case-y"

        def step(self, case_id, state, history):
            return {"change_text": "", "applied": False, "measured_kpis": {"coverage_ratio": 2.0}}

    item = R.run_incident(INC, SwarmOperator(Lying()), T)
    assert len(item["extra"]["kpi_mismatch"]) == T["episode"]["operator_steps"]
    ok = R.run_incident(INC, SwarmOperator(DryRunSwarmClient()), T)
    assert "kpi_mismatch" not in ok["extra"]


ITEM_FIELDS = {"schema_version", "kind", "run_id", "team", "created_at", "aws_time", "aws_time_source", "synthetic",
               "item_id", "input_sha256", "output_sha256", "score", "passed", "grader", "backend", "latency_ms",
               "cost_usd", "extra"}


def test_records_only_use_contract_fields(tmp_path):
    import re

    R.run("no_action", tmp_path, [], allow_dirty=True, run_id="t-c1")
    decl = json.loads((tmp_path / "declared" / "t-c1.json").read_text())
    assert set(decl) <= R.DECLARED_FIELDS
    assert set(decl["twin"]) <= {"kind", "version", "scenario", "seed", "config"}
    assert decl["twin"]["kind"] in ("maveric-inprocess", "maveric-hosted", "viavi", "none")
    assert re.match(r"^[a-z]{2}(-gov|-iso[a-z]?)?-[a-z]+-\d$", decl["region"])
    assert re.match(r"^[0-9a-f]{40}$", decl["code_sha"])
    items = [json.loads(x) for x in (tmp_path / "records" / "t-c1" / "item.jsonl").read_text().splitlines()]
    assert all(set(i) <= ITEM_FIELDS for i in items)
    res = json.loads((tmp_path / "results" / "t-c1.json").read_text())
    assert res["metric"] == res["target"]["metric"]
    assert res["passed"] == (res["value"] <= res["target"]["value"])
