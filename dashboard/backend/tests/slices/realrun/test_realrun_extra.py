import json
from pathlib import Path

from spectrum.slices.realrun import extra, logic

T = {"metric": "pass_rate", "op": ">=", "unit": "ratio", "value": 0.5}


def w(root: Path, rel: str, obj) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj))


def unit_run(root, run, unit, value, checked, **det):
    w(root, f"declared/{run}.json", {"run_id": run, "unit": unit, "model_id": "m", "target": T, "declared_at": "2026-11-03T00:00:00Z"})
    w(root, f"results/{run}.json", {"run_id": run, "unit": unit, "metric": "pass_rate", "value": value, "passed": value >= 0.5,
                                     "checked_at": checked, "details": det})


def test_superseded_run_is_not_picked_and_is_listed(tmp_path):
    unit_run(tmp_path, "t1.knowledge-x-1", "t1.knowledge", 0.02, "2026-11-03T02:00:00Z")
    unit_run(tmp_path, "t1.knowledge-x-2", "t1.knowledge", 0.36, "2026-11-03T01:00:00Z")
    w(tmp_path, "superseded.json", {"superseded": [{"run_ids": ["t1.knowledge-x-1"], "reason": "Harness defect: parser",
                                                    "superseded_by": {"run_ids": ["t1.knowledge-x-2"]}}]})
    st = logic.Store(tmp_path)
    sup = extra.superseded(st)
    cell = next(u for u in logic.track1(st, st.run_ids(), sup)["units"] if u["unit"] == "t1.knowledge")["cells"]["base"]
    assert cell["run_id"] == "t1.knowledge-x-2"
    assert cell["superseded_runs"][0]["run_id"] == "t1.knowledge-x-1"


def test_judged_slots_and_t3_live_and_t2(tmp_path):
    unit_run(tmp_path, "t3.deny-1", "t3.deny", 1.0, "2026-11-03T01:00:00Z", by_expect={"DENY": {"matched": 2, "n": 2}})
    w(tmp_path, "judging.json", {"slots": [{"slot": 1, "name": "deny", "status": "filled", "run_id": "t3.deny-1"},
                                           {"slot": 2, "name": "allow", "status": "pending", "run_id": None, "waiting_for": "x"}],
                                 "edge_cases": []})
    seg = lambda a, b, c: {"region_A": {"mean_score": a, "n": 1}, "region_B": {"mean_score": b, "n": 1}, "region_C": {"mean_score": c, "n": 1}}  # noqa: E731
    for run, unit, ps, score in (("t2.local_only.A.disjoint-1", "t2.local_only.A.disjoint", seg(0.4, 0.9, 0.9), 0.7),
                                 ("t2.central.disjoint-1", "t2.central.disjoint", seg(0.3, 0.3, 0.5), 0.5),
                                 ("t2.fl_federated.disjoint-1", "t2.fl_federated.disjoint", seg(0.45, 0.4, 0.6), 0.6)):
        unit_run(tmp_path, run, unit, 1.0, "2026-11-03T01:00:00Z", per_segment=ps, mean_score=score, rounds_completed=20)
    st = logic.Store(tmp_path)
    j = extra.judged(st)
    assert j["filled"] == 1 and j["slots"][0]["runs"][0]["state"] == "pass" and j["slots"][1]["runs"] == []
    live = extra.track3_live(st, st.run_ids())
    assert live[0]["unit"] == "t3.deny" and "by_expect: DENY 2/2" in live[0]["facts"]
    sp = extra.track2_consortium(st, st.run_ids())["splits"][0]
    a = next(r for r in sp["per_silo"] if r["silo"] == "A")
    assert abs(a["federated_minus_local"] - 0.05) < 1e-9
    assert abs(sp["distance_to_central"]["value"] - 0.1) < 1e-9
    assert sp["federated"][0]["facts"]["rounds"]["value"] == 20 and sp["federated"][0]["facts"]["clients"] is None


def test_failed_check_or_incomplete_never_reads_pass(tmp_path):
    unit_run(tmp_path, "t3.deny-a", "t3.deny", 1.0, "2026-11-03T01:00:00Z", complete=False)
    unit_run(tmp_path, "t3.deny-b", "t3.deny", 1.0, "2026-11-03T01:00:00Z")
    w(tmp_path, ".sync/checks/t3.deny-b.json", {"exit_code": 1, "summary": {"parts": {
        "e": {"state": "FAIL", "lines": ["FAIL: declared n_items 27, found 22 item records"]}}}})
    unit_run(tmp_path, "t3.deny-c", "t3.deny", 1.0, "2026-11-03T01:00:00Z")
    st = logic.Store(tmp_path)
    live = {x["run_id"]: x for x in extra.track3_live(st, st.run_ids())}
    assert live["t3.deny-a"]["state"] == "incomplete"
    assert live["t3.deny-b"]["state"] == "check failed" and "27" in live["t3.deny-b"]["why"][0]
    assert live["t3.deny-c"]["state"] == "pass"
    assert logic.run_summary(st, "t3.deny-b")["result"]["recomputed"] == "check failed"
