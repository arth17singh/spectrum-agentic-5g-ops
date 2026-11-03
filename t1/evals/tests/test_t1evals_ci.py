"""Gains use paired-bootstrap-v1 and tools/check_unit.py part e replays them from the item records."""

from __future__ import annotations

import json

import pytest

from t1.evals import run as R

np = pytest.importorskip("numpy")
check_unit = pytest.importorskip("tools.check_unit")

CI_FIELDS = ("method", "seed", "resamples", "level", "pair_key", "score_field", "baseline_run_id",
             "baseline_model_id", "cluster_field", "subset", "numpy_version", "value", "ci_low", "ci_high", "paired_n")


def _args(tmp_path, *extra):
    return R.parser().parse_args(["--evidence-dir", str(tmp_path), "--client", "fake", *extra])


def _part_e(tmp_path, run_id):
    ev = check_unit.LocalEvidence(tmp_path, allow_mtime=True)
    ck = check_unit.Checker(run_id, ev, None, None, poll_s=0.0)
    ck.load()
    ck.decl_time = ck.ev.time(ck.decl_rel)
    ck.part_e()
    return ck.parts["e"]


def _pair(tmp_path, unit, *tuned_extra):
    base = R.run(_args(tmp_path, "--unit", unit, "--run-id", f"t1.{unit}-b"))
    tuned = R.run(_args(tmp_path, "--unit", unit, "--fake-profile", "tuned", "--run-id", f"t1.{unit}-t",
                        "--baseline-run-id", base["run_id"], *tuned_extra))
    return base, tuned


def test_gain_records_the_full_v1_spec(tmp_path):
    _, tuned = _pair(tmp_path, "safety")
    g = tuned["details"]["gain"]
    assert all(k in g for k in CI_FIELDS)
    assert g["method"] == "paired-bootstrap-v1" and g["pair_key"] == "item_id" and g["score_field"] == "passed"
    assert g["cluster_field"] is None and g["subset"] is None and g["level"] == 0.95 and g["resamples"] == 10000
    assert g["baseline_model_id"] == "fake:base" and g["numpy_version"] == np.__version__
    assert g["ci_low"] <= g["value"] <= g["ci_high"] and g["paired_n"] == tuned["n_items"]


def test_check_unit_part_e_passes_on_a_ci_low_target(tmp_path):
    """The plan's tuned claim: the gain interval's lower bound, declared as the target."""
    _, tuned = _pair(tmp_path, "interface", "--target-metric", "gain_ci_low_vs_base", "--target-op", ">",
                     "--target-value", "0", "--target-unit", "ratio")
    assert tuned["value"] == tuned["details"]["gain"]["ci_low"]
    e = _part_e(tmp_path, tuned["run_id"])
    assert e.state == "PASS", e.lines
    assert any("gain_ci_low_vs_base = " in line and "paired-bootstrap-v1" in line for line in e.lines)


def test_check_unit_replays_a_gain_beside_a_plain_target(tmp_path):
    _, tuned = _pair(tmp_path, "safety", "--target-metric", "pass_rate", "--target-op", ">=", "--target-value", "0.95",
                     "--target-unit", "ratio")
    e = _part_e(tmp_path, tuned["run_id"])
    assert e.state == "PASS", e.lines
    assert any("details.gain interval" in line and "replayed" in line for line in e.lines)


def test_knowledge_group_gains_are_v1_with_a_subset(tmp_path):
    _, tuned = _pair(tmp_path, "knowledge", "--bootstrap-resamples", "500")
    for name, grp in tuned["details"]["groups"].items():
        g = grp["gain"]
        assert g["method"] == "paired-bootstrap-v1" and g["subset"] == {"field": "extra.group", "equals": name}
        assert g["paired_n"] == grp["n"]


def test_a_tampered_bound_fails_part_e(tmp_path):
    import json

    _, tuned = _pair(tmp_path, "interface", "--target-metric", "gain_ci_low_vs_base", "--target-op", ">",
                     "--target-value", "0", "--target-unit", "ratio")
    p = tmp_path / "results" / f"{tuned['run_id']}.json"
    res = json.loads(p.read_text())
    res["details"]["gain"]["ci_low"] += 1e-6
    res["value"] += 1e-6
    p.write_text(json.dumps(res))
    assert _part_e(tmp_path, tuned["run_id"]).state == "FAIL"


def test_ci_target_needs_a_baseline(tmp_path):
    with pytest.raises(SystemExit, match="baseline"):
        R.main(["--unit", "interface", "--client", "fake", "--target-metric", "gain_ci_low_vs_base",
                "--target-op", ">", "--target-value", "0", "--print-declaration"])
    with pytest.raises(SystemExit):
        R.parser().parse_args(["--unit", "interface", "--target-metric", "gain_lower"])


def test_latency_gain_is_a_note_not_v1(tmp_path):
    _, tuned = _pair(tmp_path, "price_latency")
    d = tuned["details"]
    assert "gain" not in d and "not replayed by check_unit" in d["latency_gain_note"]["note"]
    assert _part_e(tmp_path, tuned["run_id"]).state == "PASS"


def test_v1_matches_check_unit_on_known_values():
    this = [{"item_id": f"i{k}", "passed": True, "score": 1.0} for k in range(10)]
    base = [{"item_id": f"i{k}", "passed": k < 5, "score": 0.0} for k in range(10)]
    g = R.bootstrap_v1(this, base, "passed", resamples=500)
    d = [1.0 - (1.0 if k < 5 else 0.0) for k in sorted(range(10), key=lambda k: f"i{k}")]
    v, lo, hi, _ = check_unit.paired_bootstrap_v1(d, [f"i{k}" for k in sorted(range(10), key=lambda k: f"i{k}")],
                                                  R.CI_SEED, 500, 0.95)
    assert (g["value"], g["ci_low"], g["ci_high"]) == (v, lo, hi) and g["value"] == 0.5


@pytest.mark.parametrize("unit,bound", [("execution", 0.0), ("safety", 0.0), ("interface", -0.02), ("knowledge", -0.03)])
def test_tuned_default_target_and_secondary_checks_are_declared(unit, bound, capsys):
    if unit == "execution":
        pytest.importorskip("ran_rl")
    R.main(["--unit", unit, "--client", "fake", "--fake-profile", "tuned", "--baseline-run-id", "t1.x-base",
            "--print-declaration"])
    d = json.loads(capsys.readouterr().out)
    assert d["target"]["metric"] == "gain_ci_low_vs_base" and d["target"]["op"] == ">" and d["target"]["value"] == bound
    checks = d["decoding"]["secondary_checks"]
    assert checks == R.SECONDARY_TUNED[unit] and all({"name", "metric", "op", "value"} <= set(c) for c in checks)


def test_price_latency_keeps_p95_target_with_secondaries(capsys):
    R.main(["--unit", "price_latency", "--client", "fake", "--baseline-run-id", "t1.x-base", "--print-declaration"])
    d = json.loads(capsys.readouterr().out)
    assert d["target"] == {"metric": "p95_latency_ms", "op": "<=", "value": 3000.0, "unit": "ms"}
    assert [c["metric"] for c in d["decoding"]["secondary_checks"]] == ["p95_ratio_to_base", "cost_per_1k_usd"]
    R.main(["--unit", "price_latency", "--client", "fake", "--print-declaration"])  # base run
    assert [c["metric"] for c in json.loads(capsys.readouterr().out)["decoding"]["secondary_checks"]] == ["cost_per_1k_usd"]


def test_secondary_checks_are_evaluated_in_the_result(tmp_path):
    _, tuned = _pair(tmp_path, "safety")
    sc = {c["metric"]: c for c in tuned["details"]["secondary_checks"]}
    assert set(sc) == {"pass_rate", "gain_value", "over_refusal_rate"}
    assert sc["gain_value"]["measured"] == tuned["details"]["gain"]["value"]
    assert all(c["verdict"] in ("PASS", "FAIL") for c in sc.values())
    _, pl = _pair(tmp_path / "pl", "price_latency")
    pc = {c["metric"]: c for c in pl["details"]["secondary_checks"]}
    assert pc["p95_ratio_to_base"]["measured"] > 0 and pc["cost_per_1k_usd"]["verdict"] in ("PASS", "FAIL")
