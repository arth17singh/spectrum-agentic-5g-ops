import json
from pathlib import Path

from spectrum.slices.realrun import extra, logic

T = {"metric": "n_items", "op": ">=", "unit": "rounds", "value": 1}
CONV = {"metric": "scaled_mae", "lower_is_better": True, "rounds": [2, 3, 4],
        "curve": {"A": [0.6, 0.5, 0.49], "B": [0.5, 0.45, 0.43], "C": [0.9, 0.8, 0.76], "mean": [0.7, 0.6, 0.56]},
        "reference": {"local_only": {"A": 0.44, "B": 0.38, "C": 0.66, "run_ids": {}}, "central": {"value": 0.56},
                      "fedavg_local": {"value": 0.64}},
        "final_round_check": {"round": 4, "matches_published": True, "published_run_id": "t2.fl_federated.disjoint-x",
                              "max_abs_diff": 0.0},
        "session": {"session_id_short": "5fb29b2d", "rounds_completed": 20, "clients": 3, "secure_aggregation": True, "dp": "none"}}


def w(root: Path, rel: str, obj) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj))


def test_no_curve_until_a_run_exists(tmp_path):
    st = logic.Store(tmp_path)
    assert extra.latest_unit(st, st.run_ids(), "t2.fl_convergence") is None


def test_latest_convergence_run_passes_details_through(tmp_path):
    for run, at in (("t2.fl_convergence-1", "2026-11-03T06:00:00Z"), ("t2.fl_convergence-2", "2026-11-03T07:00:00Z")):
        w(tmp_path, f"declared/{run}.json", {"run_id": run, "unit": "t2.fl_convergence", "target": T})
        w(tmp_path, f"results/{run}.json", {"run_id": run, "unit": "t2.fl_convergence", "metric": "n_items", "value": 3,
                                             "passed": True, "checked_at": at, "details": CONV})
    st = logic.Store(tmp_path)
    c = extra.latest_unit(st, st.run_ids(), "t2.fl_convergence")
    assert c["run_id"] == "t2.fl_convergence-2" and c["details"]["session"]["clients"] == 3 and c["state"] == "pass"
