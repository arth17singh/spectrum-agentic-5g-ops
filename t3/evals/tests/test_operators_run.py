import json

import numpy as np
import pytest

from t3.evals import run as R
from t3.evals.graders import load_targets
from t3.evals.operators import NoAction, Oracle, PPO, RuleBaseline, StubPolicy, make
from t3.evals.swarm import DryRunSwarmClient, SwarmOperator, task_text
from t3.evals.tests._local import local_suite
from t3.evals.twin_backend import ReplayTwin

T = load_targets()
SUITE, INCS, DIGEST = local_suite()  # built on this machine (see _local.py)
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
    ox = item["extra"]["operator_extra"]
    assert item["extra"]["mode"] == "proposal" and ox["mode"] == "proposal"
    assert len(ox["case_ids"]) == T["episode"]["operator_steps"]  # one front-door case per step
    assert all(c.startswith("case-dryrun-") for c in ox["case_ids"])
    assert all(s["change_text"] == "" or s["change_text"].startswith("cell=") for s in ox["swarm_steps"])
    assert op.synthetic is True
    assert "run_id=t3.evals.test" in task_text(INC, "t3.evals.test")
    assert "cell" not in task_text(INC, "x").split("dB.")[0].lower()  # symptoms only, no fault cell


class Recorder:
    """Proposes +6 dB on cell 0 every time; never applies (apply is refused pending approval)."""

    synthetic = True

    def __init__(self):
        self.opened = []

    def open_case(self, case):
        self.opened.append(case)
        return f"case-{len(self.opened)}"

    def step(self, case_id, state, history):
        return {"change_text": "cell=0;d_power_db=+6.0;d_tilt_deg=+0.0", "applied": False, "refused": True,
                "approval_id": "chg-abc"}


def test_full_mode_refused_step_is_no_change():
    rec = Recorder()
    item = R.run_incident(INC, SwarmOperator(rec, mode="full"), T)
    assert rec.opened[0]["start_action_log"] == [INC["commissioning"], INC["fault"]]
    assert len(rec.opened) == 1
    assert item["extra"]["mode"] == "full"
    assert item["extra"]["changes_used"] == 0
    assert item["score"] == pytest.approx(0.0, abs=1e-6)


def test_proposal_mode_grades_the_refused_proposal_on_our_twin():
    rec = Recorder()
    item = R.run_incident(INC, SwarmOperator(rec, mode="proposal", proposal_steps=2), T)
    assert item["extra"]["mode"] == "proposal"
    assert item["extra"]["changes_used"] == 2  # third step is past proposal_steps: no case, no change
    assert len(rec.opened) == 2
    # each case starts at the harness state: commissioning, fault, then the earlier proposals
    assert rec.opened[0]["start_action_log"] == [INC["commissioning"], INC["fault"]]
    assert rec.opened[1]["start_action_log"][-1] == [{"cell": 0, "d_power_db": 6.0, "d_tilt_deg": 0.0}]
    assert all(s["refused"] and s["approval_id"] == "chg-abc" for s in item["extra"]["operator_extra"]["swarm_steps"])


def test_end_to_end_run_writes_declared_items_result_and_gain(tmp_path):
    base = R.run("no_action", tmp_path, [], allow_dirty=True, suite=SUITE, run_id="t-noact")
    cand = R.run("config_revert", tmp_path, ["t-noact"], allow_dirty=True, suite=SUITE, run_id="t-cfg")
    decl = json.loads((tmp_path / "declared" / "t-cfg.json").read_text())
    assert decl["test_set_sha256"] == DIGEST and decl["n_items"] == len(INCS)
    assert decl["baseline_run_id"] == "t-noact" and "subset" not in decl
    items = [json.loads(x) for x in (tmp_path / "records" / "t-cfg" / "item.jsonl").read_text().splitlines()]
    assert len(items) == len(INCS) and all(i["run_id"] == "t-cfg" for i in items)
    res = json.loads((tmp_path / "results" / "t-cfg.json").read_text())
    g = res["details"]["gain"]
    assert set(g) == {"value", "ci_low", "ci_high", "baseline_run_id", "paired_n"}
    assert g["paired_n"] == len(INCS) and g["baseline_run_id"] == "t-noact"
    assert g["ci_low"] <= g["value"] <= g["ci_high"]
    assert g["value"] == pytest.approx(cand["details"]["mean_recovery"] - base["details"]["mean_recovery"])
    with pytest.raises(FileExistsError):
        R.run("no_action", tmp_path, [], allow_dirty=True, suite=SUITE, run_id="t-noact")

    # the swarm unit: masked subset, proposal mode, gain over config_revert, secondary vs rule_baseline
    R.run("rule_baseline", tmp_path, ["t-noact"], allow_dirty=True, suite=SUITE, run_id="t-rule")
    with pytest.raises(SystemExit):
        R.run("swarm", tmp_path, ["t-noact"], allow_dirty=True, suite=SUITE, run_id="t-sw0", subset="masked")
    R.run("swarm", tmp_path, ["t-cfg", "t-rule"], allow_dirty=True, suite=SUITE, run_id="t-sw", subset="masked")
    masked = [i for i in INCS if i["tier"] == "masked"]
    d = json.loads((tmp_path / "declared" / "t-sw.json").read_text())
    assert d["subset"] == "tier=masked" and d["n_items"] == len(masked)
    assert d["twin"]["config"]["eval"]["mode"] == "proposal"
    assert len(d["twin"]["config"]["eval"]["targets_sha256"]) == 64
    assert d["target"]["metric"] == "gain_ci_low_vs_config_revert_masked"
    items = [json.loads(x) for x in (tmp_path / "records" / "t-sw" / "item.jsonl").read_text().splitlines()]
    assert {i["item_id"] for i in items} == {i["incident_id"] for i in masked}
    assert all(i["extra"]["mode"] == "proposal" for i in items)
    res = json.loads((tmp_path / "results" / "t-sw.json").read_text())
    pg = res["details"]["primary_gain"]
    assert pg["baseline_run_id"] == "t-cfg" and pg["tier"] == "masked" and pg["paired_n"] == len(masked)
    # the fake swarm IS config_revert, so the gain is exactly zero and the unit fails
    assert pg["value"] == pytest.approx(0.0, abs=1e-9) and res["passed"] is False
    assert res["value"] == pytest.approx(pg["ci_low"])
    sec = res["details"]["secondary"]
    assert sec["rule_baseline"] == "t-rule" and isinstance(sec["wrong_cell_not_worse_than_rule"], bool)


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

    R.run("no_action", tmp_path, [], allow_dirty=True, suite=SUITE, run_id="t-c1")
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
