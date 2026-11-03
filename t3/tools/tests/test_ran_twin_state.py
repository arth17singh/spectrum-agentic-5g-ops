"""Per-cell twin state in the change record (Twin View): before / after / baseline snapshots,
twin.preset, and aws_time from the Lambda's own clock."""
from __future__ import annotations

import copy
import json

import pytest

from fakes import FakeTwin, FakeTwinCF, fake_kpis
from t3.tools import app
from test_ran_tools import APPLY, CASE, CHANGE, World

FIELDS = {("ChangeRecord", "twin_state"): True, ("ChangeTwin", "preset"): True}


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setattr(app, "_DECLARED", dict(FIELDS))   # the shape W1 is adding (optional, additive)
    yield
    app.set_deps(None)


def applied_record(twin=FakeTwinCF, steps_before=0):
    w = World(twin=twin, run_id="r-twin")
    for i in range(steps_before):
        args = dict(APPLY, item_id=f"pre-{i}", change="cell=0;d_power_db=+1.0;d_tilt_deg=+0.0")
        w.approved(args)
        assert w.call("ran___apply_change", **args)["applied"]
    obs_before = w.call("ran___observe", case_id=CASE)
    w.call("ran___predict", case_id=CASE, item_id="step-x", change=CHANGE)
    args = dict(APPLY, item_id="step-x")
    w.approved(args)
    assert w.call("ran___apply_change", **args)["applied"]
    obs_after = w.call("ran___observe", case_id=CASE)
    return w, w.records()[-1], obs_before, obs_after


def cells(kpis):
    return app.snapshot(0, kpis)["per_cell"]


def with_shares(k):
    k = copy.deepcopy(k)
    for c in k["per_cell"]:
        c["coverage_share"] = round(0.2 + 0.1 * c["cell"], 3)
        c["sla_share"] = None if c["cell"] == 3 else 0.25   # a cell with no UEs: None
    return k


class FakeTwinShares(FakeTwinCF):
    """Like W2 2977a7e: kpis() with per-cell shares, and counterfactual() returning after_kpis and
    baseline_kpis (full kpis dicts) next to the headline numbers."""

    def kpis(self):
        return with_shares(fake_kpis(self.state))

    def counterfactual(self, change):
        out = super().counterfactual(change)
        saved_state, saved_log = copy.deepcopy(self.state), copy.deepcopy(self.action_log)
        after = with_shares(fake_kpis(self.apply(change)))
        self.state, self.action_log = copy.deepcopy(saved_state), copy.deepcopy(saved_log)
        base = with_shares(fake_kpis(self.apply([])))
        self.state, self.action_log = saved_state, saved_log
        return {**out, "after_kpis": after, "baseline_kpis": base}


@pytest.mark.parametrize("twin", [FakeTwinShares, FakeTwinCF, FakeTwin])
@pytest.mark.parametrize("steps_before", [0, 2])
def test_before_after_baseline_snapshots(twin, steps_before):
    w, rec, obs_before, obs_after = applied_record(twin, steps_before)
    ts = rec["twin_state"]
    n = steps_before
    assert rec["twin"]["step"] == n
    # before: the state at twin.step, exactly what ran___observe showed at that step
    assert ts["before"]["step"] == n and ts["before"]["per_cell"] == cells(obs_before["kpis"])
    assert ts["before"]["coverage_ratio"] == obs_before["kpis"]["coverage_ratio"]
    # after: the change plus one step, the state ran___observe shows next
    assert ts["after"]["step"] == n + 1 and ts["after"]["per_cell"] == cells(obs_after["kpis"])
    assert ts["after"]["coverage_ratio"] == rec["measured_kpis"]["coverage_ratio"]
    # baseline: the same step advanced with no change (the counterfactual)
    assert ts["baseline"]["step"] == n + 1
    assert ts["baseline"]["coverage_ratio"] == rec["baseline_kpis"]["coverage_ratio"]
    replay = twin("nominal", 0, "official")
    replay.reset()
    for entry in w.case_doc()["action_log"][:n]:
        replay.apply(entry["change"])
    base_kpis = fake_kpis(replay.apply([]))
    assert ts["baseline"]["per_cell"] == cells(with_shares(base_kpis) if twin is FakeTwinShares else base_kpis)
    if twin is FakeTwinShares:   # shares pass through; None where the twin cannot tell
        assert ts["after"]["per_cell"][3]["sla_share"] is None and ts["after"]["per_cell"][1]["coverage_share"] == 0.3
        assert obs_after["kpis"]["per_cell"][1]["coverage_share"] == 0.3   # observe shows them too
    else:                        # no per-UE data: the keys are present and None
        assert all(c["coverage_share"] is None and c["sla_share"] is None for c in ts["after"]["per_cell"])
    # the changed cell differs between after and baseline; the others do not
    after = {c["cell"]: c for c in ts["after"]["per_cell"]}
    base = {c["cell"]: c for c in ts["baseline"]["per_cell"]}
    assert after[2]["tilt_deg"] == base[2]["tilt_deg"] + 2.0 and after[0] == base[0]


def test_twin_preset_and_small_record():
    _, rec, _, _ = applied_record()
    assert rec["twin"]["preset"] == "official"
    assert len(rec["twin_state"]["before"]["per_cell"]) == 4
    assert len(json.dumps(rec)) < 8000             # 4 cells x 3 snapshots stays small
    text = json.dumps(rec).lower()
    for secret_like in ("token", "http://", "https://", "adapter"):
        assert secret_like not in text


def test_snapshot_keeps_only_contract_cell_keys():
    kpis = {"coverage_ratio": 0.6, "sla_ratio": 0.9, "per_cell": [
        {"cell": 0.0, "power_dbm": 23.0, "tilt_deg": 6.0, "ue_count": 22.0, "rsrp_mean_dbm": -88.0,
         "rsrp_p10_dbm": -92.0, "sinr_mean_db": 9.3, "sinr_p10_db": 2.7, "alarm": False, "asleep": False,
         "coverage_share": 0.25, "sla_share": 0.3, "sinr_p10_db_bounds": [1, 2], "p10_method": "pooled",
         "rsrp_mean_dbm_extra": 1, "sinr_mean_db_none": None, "rsrp_p10_dbm_x": 0}]}
    snap = app.snapshot(3, kpis)
    assert snap == {"step": 3, "coverage_ratio": 0.6, "sla_ratio": 0.9, "per_cell": [
        {"cell": 0, "power_dbm": 23.0, "tilt_deg": 6.0, "ue_count": 22, "coverage_share": 0.25, "sla_share": 0.3,
         "sinr_mean_db": 9.3, "sinr_p10_db": 2.7, "rsrp_mean_dbm": -88.0, "rsrp_p10_dbm": -92.0,
         "alarm": False, "asleep": False}]}
    assert isinstance(snap["per_cell"][0]["cell"], int)
    hosted = {"coverage_ratio": 0.6, "sla_ratio": 0.9,
              "per_cell": [{"cell": 1, "power_dbm": 23.0, "tilt_deg": 6.0, "ue_count": 0}]}
    assert app.snapshot(0, hosted)["per_cell"] == [{"cell": 1, "power_dbm": 23.0, "tilt_deg": 6.0, "ue_count": 0,
                                                    "coverage_share": None, "sla_share": None}]


def test_aws_time_from_the_lambda_clock():
    w = World(run_id="r-clock")
    w.deps.aws_clock = lambda: ("2026-11-03T01:02:03.456Z", "lambda:clock")
    w.approved(APPLY)
    w.call("ran___apply_change", **APPLY)
    rec = w.records()[0]
    assert rec["aws_time"] == "2026-11-03T01:02:03.456Z" and rec["aws_time_source"] == "lambda:clock"


def test_lambda_clock_only_inside_lambda():
    assert app.lambda_clock({})() == (None, None)
    t, src = app.lambda_clock({"AWS_LAMBDA_FUNCTION_NAME": "team13-ran-tools"})()
    assert src == "lambda:clock" and t.endswith("Z") and "T" in t


def test_fields_are_left_out_until_the_contract_declares_them(monkeypatch):
    monkeypatch.setattr(app, "_DECLARED", {k: False for k in FIELDS})
    _, rec, _, _ = applied_record()
    assert "twin_state" not in rec and "preset" not in rec["twin"]


def test_real_shaped_record_validates_against_team13_contracts(monkeypatch):
    contracts = pytest.importorskip("team13_contracts")
    if "twin_state" not in contracts.ChangeRecord.model_fields:
        pytest.skip("this team13_contracts predates twin_state (W1 commit pending)")
    monkeypatch.setattr(app, "_DECLARED", {})   # use what the contract really declares
    for twin in (FakeTwinShares, FakeTwin):
        _, rec, _, _ = applied_record(twin)
        assert "twin_state" in rec and rec["twin"]["preset"] == "official"
        contracts.parse_record(copy.deepcopy(rec))
    # a real-shaped record: W2's per-cell keys, a dark cell with -inf RSRP, numpy-style 0/1 flags
    dark = {"coverage_ratio": 0.5, "sla_ratio": 0.8, "rsrp_p10_dbm": -110.0, "per_cell": [
        {"cell": 0, "power_dbm": 23.0, "tilt_deg": 6.0, "ue_count": 30, "rsrp_mean_dbm": -90.0,
         "rsrp_p10_dbm": -101.5, "sinr_mean_db": 8.1, "sinr_p10_db": 1.2, "alarm": 0, "asleep": 0,
         "coverage_share": 0.8, "sla_share": 0.9},
        {"cell": 1, "power_dbm": 20.0, "tilt_deg": 6.0, "ue_count": 0, "rsrp_mean_dbm": float("-inf"),
         "rsrp_p10_dbm": float("-inf"), "sinr_mean_db": float("nan"), "sinr_p10_db": float("nan"), "alarm": 1,
         "asleep": 0, "coverage_share": None, "sla_share": None}]}
    snap = app.snapshot(4, dark)
    assert snap["per_cell"][1]["rsrp_mean_dbm"] is None and snap["per_cell"][1]["sinr_p10_db"] is None
    assert snap["per_cell"][1]["alarm"] is True and snap["per_cell"][0]["asleep"] is False
    rec = dict(rec, twin_state={"before": snap, "after": app.snapshot(5, dark), "baseline": app.snapshot(5, dark)})
    contracts.parse_record(copy.deepcopy(rec))
    assert "rsrp_mean_dbm\": null" in json.dumps(rec["twin_state"])
