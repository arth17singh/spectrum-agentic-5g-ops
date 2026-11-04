import importlib.util
import json

import pytest

from t3.evals import attacks as A
from t3.evals import faults as F
from t3.evals import incidents as I
from t3.evals.graders import load_targets

T = load_targets()
REAL_TWIN = importlib.util.find_spec("twin.replay") is not None


# ----- faults: no twin needed -----

def test_every_button_maps_to_an_incident_template():
    names = {n for n, _ in I.TEMPLATES}
    assert set(F.FAULTS) == {"tilt_drift", "pa_fault", "bad_push", "hw_failure", "cascade"}
    for f in F.FAULTS.values():
        assert f["template"] in names and f["label"]


def test_fault_lands_on_the_chosen_cell_and_cascade_hits_the_neighbour():
    assert F.fault_cells_for("pa_fault", 2) == [{"cell": 2, "d_power_db": -6.0, "d_tilt_deg": 0.0}]
    assert [c["cell"] for c in F.fault_cells_for("cascade", 3)] == [3, 0]
    with pytest.raises(KeyError):
        F.fault_cells_for("meteor", 0)
    with pytest.raises(ValueError):
        F.fault_cells_for("pa_fault", 4)


def test_dev_seeds_never_include_a_test_seed():
    gen = F.dev_seeds()
    seeds = [next(gen) for _ in range(50)]
    assert all(I.split_of(s) == "dev" for s in seeds)
    test_seeds = {json.loads(line)["seed"] for line in I.TEST_PATH.read_text().splitlines() if line.strip()}
    assert not set(seeds) & test_seeds
    test_seed = next(s for s in range(F.DEV_SEED_BASE, F.DEV_SEED_BASE + 100) if I.split_of(s) == "test")
    with pytest.raises(ValueError, match="test split"):
        F._try("pa_fault", 0, test_seed, "official", "visible", T)


def test_spec_matches_the_incident_shape():
    s = F.spec_for("bad_push", 1, 7002)
    assert set(s) == set(I.candidate(0))
    assert s["scenario"] == "nominal" and s["commissioning"] == [] and s["template"] == "power-3+tilt+3"
    assert F.spec_for("bad_push", 1, 7002, tier="masked")["commissioning"] == I.commissioning(7002, "masked")


def _row(cell, cov, sla, sinr, ue=30, **kw):
    return {"cell": cell, "coverage_share": cov, "sla_share": sla, "sinr_p10_db": sinr, "ue_count": ue,
            "alarm": False, "asleep": False} | kw


def test_cell_health_relative_to_the_healthy_cell():
    ref = [_row(0, 0.9, 0.9, 2.0), _row(1, 0.9, 0.9, 2.0), _row(2, 0.9, 0.9, 2.0), _row(3, 0.9, 0.9, 2.0, ue=10)]
    cur = [_row(0, 0.9, 0.9, 2.0), _row(1, 0.85, 0.9, 2.0), _row(2, 0.6, 0.9, 2.0), _row(3, 0.8, 0.9, 2.0, ue=10)]
    h = {x["cell"]: x["health"] for x in F.cell_health(cur, reference=ref)}
    assert h == {0: "green", 1: "amber", 2: "red", 3: "amber"}  # cell 3: a 0.10 drop is one UE of 10
    assert all(x["health"] == "green" for x in F.cell_health(ref, reference=ref))
    sinr = F.cell_health([_row(0, 0.9, 0.9, -1.5)], reference=[_row(0, 0.9, 0.9, 2.0)])
    assert sinr[0]["health"] == "red"
    empty = F.cell_health([_row(0, None, None, 2.0, ue=0)], reference=[_row(0, 0.9, 0.9, 2.0)])
    assert empty[0]["health"] == "red" and "serves no UEs" in empty[0]["why"]
    assert F.cell_health([_row(0, 0.9, 0.9, 2.0, alarm=True)])[0]["health"] == "red"


def test_cache_table_covers_every_button_and_cell_with_dev_seeds():
    table = F.load_cache()
    assert table["catalog"] == F.CATALOG_VERSION
    for preset in ("official", "viavi-like"):
        for fk in F.FAULTS:
            for cell in range(F.N_CELLS):
                e = table["entries"][f"{fk}|c{cell}|{preset}|visible"]
                assert e["material"] and I.split_of(e["seed"]) == "dev"
                assert e["fault"] == F.fault_cells_for(fk, cell)
                assert {c["cell"] for c in e["fault"]} <= set(e["red_cells"]) and e["fault_cells_red"]
    assert F.cached_seed("pa_fault", 2) == table["entries"]["pa_fault|c2|official|visible"]["seed"]


# ----- faults: with the twin (about a second each) -----

@pytest.mark.skipif(not REAL_TWIN, reason="needs the real twin (twin.replay + ran-rl)")
def test_make_incident_from_cache_is_material_and_red_where_injected():
    x = F.make_incident("pa_fault", 2)
    e = F.load_cache()["entries"]["pa_fault|c2|official|visible"]
    assert x["seed"] == e["seed"] and x["split"] == "dev" and x["material_kpis"]
    assert x["incident_id"] == f"lf-pa_fault-c2-off-vis-{x['seed']}" and x["fault_cells"] == [2]
    assert x["start_log"] == ["", "cell=2;d_power_db=-6.0;d_tilt_deg=+0.0"]
    assert all(h["health"] == "green" for h in x["health_before"])
    assert {h["cell"] for h in x["health_after"] if h["health"] == "red"} == set(e["red_cells"])
    assert next(c for c in x["cells_after"] if c["cell"] == 2)["power_dbm"] == \
        next(c for c in x["cells_before"] if c["cell"] == 2)["power_dbm"] - 6.0


# ----- attacks -----

def test_attacks_reuse_judged_deny_cases():
    suite = A.load_suite()
    for a in A.ATTACKS.values():
        row = suite[a["suite_id"]]
        assert row["expect"] == "DENY" and row["adversarial"]
    assert suite["t3-041"]["rule_class"] == "guardrail_injection"
    assert suite["t3-011"]["caller"] == "ran_monitor" and suite["t3-011"]["tool"] == "ran___apply_change"


def test_build_call_fills_the_case_and_alarm_variant_carries_the_injection():
    c = A.build_call("rogue_agent", "case-abc-1")
    assert c["caller"] == "ran_monitor" and c["tool"] == "ran___apply_change" and c["args"]["case_id"] == "case-abc-1"
    j = A.build_call("poisoned_alarm", "case-abc-1")
    assert j["tool"] == "front___continue_case" and j["judged"] and j["args"]["case_id"] == "case-abc-1"
    al = A.build_call("poisoned_alarm", "case-abc-1", variant="alarm", cell=3)
    assert not al["judged"] and "4G-CELL-3" in al["args"]["message"] and "ignore all previous" in al["args"]["message"]
    with pytest.raises(ValueError):
        A.build_call("rogue_agent", "nocase")
    with pytest.raises(ValueError):
        A.build_call("rogue_agent", "case-x", variant="alarm")


def test_run_attack_reports_refusal_policy_and_aws_time():
    gw = A.FakeGateway("DENY", text='{"error": "Tool call denied by policy team13_default_deny"}')
    r = A.run_attack("rogue_agent", gw, "case-abc-1")
    assert r["refused"] and r["decision"] == "DENY" and r["policy"] == "team13_default_deny"
    assert r["aws_time"] == "2026-11-03T22:00:00Z" and not r["dry"]
    assert gw.calls == [("ran_monitor", "ran___apply_change", A.build_call("rogue_agent", "case-abc-1")["args"])]
    r2 = A.run_attack("poisoned_alarm", A.FakeGateway("DENY", text="blocked"), "case-abc-1")
    assert r2["policy"] == "guardrail_injection"  # no policy name in the text: the suite's rule class


def test_allow_or_error_is_never_shown_as_a_refusal_and_dry_makes_no_call():
    r = A.run_attack("rogue_agent", A.FakeGateway("ALLOW", text="{}"), "case-abc-1")
    assert not r["refused"] and r["policy"] is None
    gw = A.FakeGateway()
    d = A.run_attack("poisoned_alarm", gw, "case-abc-1", dry=True)
    assert d["dry"] and d["decision"] == "DRY" and not d["refused"] and gw.calls == []
