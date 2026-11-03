import copy
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from scorecard_fixtures import GOLDEN_STATES, FakeReader, load  # noqa: E402
from spectrum.slices.scorecard.logic import (  # noqa: E402
    build_row,
    build_scorecard,
    compare,
    show_record_command,
)


def golden_card():
    r = FakeReader()
    return build_scorecard([(i, r.declared(i), r.result(i)) for i in r.runs()])


def by_run(card):
    return {u["run_id"]: u for t in card["tracks"] for u in t["units"]}


# ------------------------------------------------------------------ golden (see scorecard_fixtures.py for the arithmetic)

def test_golden_states_exact():
    rows = by_run(golden_card())
    assert {k: v["state"] for k, v in rows.items()} == GOLDEN_STATES


def test_golden_values_exact():
    rows = by_run(golden_card())
    acc = rows["syn-t1-acc"]
    assert acc["measured"] == 17 / 20 == 0.85
    assert acc["target"] == {"metric": "accuracy", "op": ">=", "value": 0.85, "unit": "fraction"}
    assert acc["gain"] == {"value": 1.7, "ci_low": 1.4, "ci_high": 2.0, "baseline_run_id": "syn-t1-acc-base", "paired_n": 20}
    assert acc["baseline_run_id"] == "syn-t1-acc-base"
    assert acc["n_items"] == 20 and acc["problems"] == []

    deny = rows["syn-t3-deny"]
    assert deny["measured"] == 39 / 40 == 0.975
    assert deny["gain"] == {"value": 1.3, "ci_low": 1.1, "ci_high": 1.5, "baseline_run_id": "syn-t3-deny-base", "paired_n": 40}

    pause = rows["syn-t3-pause"]
    assert pause["measured"] == 75 and pause["state"] == "fail" and pause["recomputed"] == "fail"
    assert pause["stored_passed"] is False and pause["gain"] is None and pause["gain_missing"] is False

    trace = rows["syn-t3-trace"]
    assert trace["state"] == "not checked"
    assert trace["measured"] is None and trace["checked_at"] is None and trace["checker"] is None and trace["n_items"] is None
    assert trace["declared_n_items"] == 12

    ret = rows["syn-t3-retention"]
    assert ret["measured"] == 0.45 and ret["stored_passed"] is True and ret["recomputed"] == "fail"
    assert [p["code"] for p in ret["problems"]] == ["stored_passed_disagrees"]


def test_golden_track_counts_exact():
    card = golden_card()
    assert [t["track"] for t in card["tracks"]] == ["track-1", "track-3"]
    counts = {t["track"]: {k: t[k] for k in ("declared", "passed", "failed", "not_checked", "inconsistent")} for t in card["tracks"]}
    assert counts == {
        "track-1": {"declared": 1, "passed": 1, "failed": 0, "not_checked": 0, "inconsistent": 0},
        "track-3": {"declared": 4, "passed": 1, "failed": 1, "not_checked": 1, "inconsistent": 1},
    }
    assert card["totals"] == {"declared": 5, "passed": 2, "failed": 1, "not_checked": 1, "inconsistent": 1}
    assert card["synthetic_rows"] == 5 and card["empty"] is False


def test_golden_orphan_listed_not_counted():
    card = golden_card()
    assert [o["run_id"] for o in card["orphans"]] == ["syn-orphan"]
    assert "syn-orphan" not in by_run(card)
    assert card["totals"]["declared"] == 5


def test_fixtures_carry_contract_fields_and_are_synthetic():
    for run in ("syn-t1-acc", "syn-t3-deny", "syn-t3-pause", "syn-t3-retention"):
        d, r = load("declared", run), load("results", run)
        assert d["synthetic"] is True and r["synthetic"] is True
        assert set(d) - {"synthetic", "baseline_run_id"} == {
            "unit", "track", "target", "n_items", "test_set_sha256", "model_id", "grader", "code_sha", "region", "declared_at"}
        assert set(r) - {"synthetic"} == {"unit", "metric", "value", "target", "passed", "n_items", "checked_at", "checker", "details"}
        assert set(d["target"]) == {"metric", "op", "value", "unit"}


# ------------------------------------------------------------------ the stored boolean is never trusted

def pair(run="syn-t3-deny"):
    return copy.deepcopy(load("declared", run)), copy.deepcopy(load("results", run))


def test_stored_passed_flipped_to_false_is_inconsistent_not_fail():
    d, r = pair()
    r["passed"] = False
    row = build_row("x", d, r)
    assert row["state"] == "inconsistent" and row["recomputed"] == "pass" and row["stored_passed"] is False


def test_measured_changed_but_stored_passed_left_is_inconsistent():
    d, r = pair()
    r["value"] = 0.5  # fails >= 0.95, stored passed still true
    assert build_row("x", d, r)["state"] == "inconsistent"


def test_consistent_flip_of_measured_and_passed_gives_fail():
    d, r = pair()
    r["value"], r["passed"] = 0.5, False
    row = build_row("x", d, r)
    assert row["state"] == "fail" and row["problems"] == []


@pytest.mark.parametrize("bad", ["true", 1, None])
def test_stored_passed_not_a_boolean_is_inconsistent(bad):
    d, r = pair()
    r["passed"] = bad
    row = build_row("x", d, r)
    assert row["state"] == "inconsistent"
    assert row["problems"][0]["code"] == "stored_passed_invalid"


def test_target_copied_into_result_must_equal_declared_target():
    d, r = pair()
    r["target"]["value"] = 0.5  # loosened after the fact; measured 0.975 passes both
    row = build_row("x", d, r)
    assert row["state"] == "inconsistent"
    assert "target_changed" in [p["code"] for p in row["problems"]]
    assert row["target"]["value"] == 0.95  # the declared target is what is shown


def test_decision_uses_declared_target_not_result_target():
    d, r = pair()
    r["value"], r["passed"] = 0.9, True
    r["target"]["value"] = 0.8  # result claims 0.9 >= 0.8; declaration says >= 0.95
    row = build_row("x", d, r)
    assert row["recomputed"] == "fail" and row["state"] == "inconsistent"


def test_scored_item_count_must_equal_declared():
    d, r = pair()
    r["n_items"] = 38
    row = build_row("x", d, r)
    assert row["state"] == "inconsistent" and "n_items_mismatch" in [p["code"] for p in row["problems"]]


def test_checked_before_declared_is_inconsistent():
    d, r = pair()
    r["checked_at"] = "2026-11-02T09:59:59Z"
    assert "checked_before_declared" in [p["code"] for p in build_row("x", d, r)["problems"]]


def test_metric_mismatch_is_inconsistent():
    d, r = pair()
    r["metric"] = "something_else"
    assert "metric_mismatch" in [p["code"] for p in build_row("x", d, r)["problems"]]


@pytest.mark.parametrize("v", [float("nan"), float("inf"), True, "0.97", None])
def test_unusable_measured_value_is_inconsistent(v):
    d, r = pair()
    r["value"] = v
    row = build_row("x", d, r)
    assert row["state"] == "inconsistent" and row["measured"] is None and row["recomputed"] is None


def test_invalid_declared_target_is_inconsistent_even_without_result():
    d, _ = pair()
    d["target"]["op"] = "~="
    row = build_row("x", d, None)
    assert row["state"] == "inconsistent" and row["problems"][0]["code"] == "target_invalid"
    d["target"] = None
    assert build_row("x", d, None)["target"] is None


# ------------------------------------------------------------------ gain

def test_gain_outside_its_interval_is_inconsistent():
    d, r = pair()
    r["details"]["gain"]["value"] = 2.0  # interval is [1.1, 1.5]
    assert "gain_outside_interval" in [p["code"] for p in build_row("x", d, r)["problems"]]


def test_gain_baseline_must_be_the_declared_baseline():
    d, r = pair()
    r["details"]["gain"]["baseline_run_id"] = "some-other-run"
    assert "baseline_mismatch" in [p["code"] for p in build_row("x", d, r)["problems"]]


def test_declared_baseline_without_gain_is_flagged_missing_not_failed():
    d, r = pair()
    del r["details"]["gain"]
    row = build_row("x", d, r)
    assert row["gain"] is None and row["gain_missing"] is True and row["state"] == "pass"


def test_unit_without_baseline_has_no_gain_and_is_not_missing():
    d, r = pair("syn-t3-pause")
    row = build_row("x", d, r)
    assert row["gain"] is None and row["gain_missing"] is False


def test_gain_not_an_object():
    d, r = pair()
    r["details"]["gain"] = 1.3
    assert "gain_malformed" in [p["code"] for p in build_row("x", d, r)["problems"]]


# ------------------------------------------------------------------ comparison

@pytest.mark.parametrize("m,op,v,want", [
    (1, ">=", 1, True), (0.99, ">=", 1, False),
    (1, "<=", 1, True), (1.01, "<=", 1, False),
    (1, ">", 1, False), (1.01, ">", 1, True),
    (1, "<", 1, False), (0.99, "<", 1, True),
    (1, "==", 1, True), (0.1 + 0.2, "==", 0.3, True), (1.001, "==", 1, False),
])
def test_compare_operators_and_boundaries(m, op, v, want):
    assert compare(m, op, v) is want


def test_compare_unknown_operator_raises():
    with pytest.raises(ValueError):
        compare(1, "!=", 1)


def test_not_checked_when_result_missing_and_no_trust_in_inputs():
    d, _ = pair()
    d_before = copy.deepcopy(d)
    row = build_row("x", d, None)
    assert row["state"] == "not checked" and row["measured"] is None
    assert d == d_before  # inputs untouched


# ------------------------------------------------------------------ whole scorecard

def test_empty_scorecard_is_honest():
    card = build_scorecard([])
    assert card["empty"] is True and card["tracks"] == [] and card["totals"]["declared"] == 0


def test_garbage_records_do_not_crash():
    card = build_scorecard([("a", "not a dict", ["nor this"]), ("b", None, None), ("c", {}, None)])
    rows = by_run(card)
    assert list(rows) == ["c"] and rows["c"]["state"] == "inconsistent" and rows["c"]["track"] == "unknown"


def test_tracks_sorted_numerically_then_other():
    mk = lambda t: {"unit": "u", "track": t, "target": {"metric": "m", "op": ">=", "value": 1, "unit": ""}, "n_items": 1}
    card = build_scorecard([("a", mk("track-10"), None), ("b", mk("zeta"), None), ("c", mk("track-2"), None)])
    assert [t["track"] for t in card["tracks"]] == ["track-2", "track-10", "zeta"]


def test_synthetic_flag_from_either_record():
    d, r = pair()
    del d["synthetic"]
    assert build_row("x", d, r)["synthetic"] is True  # result is synthetic
    del r["synthetic"]
    assert build_row("x", d, r)["synthetic"] is False


def test_show_record_command_is_copyable_and_safe():
    assert show_record_command("syn-t3-deny") == "python3 tools/show_record.py --run-id syn-t3-deny"
    assert show_record_command("a b; rm -rf /") == "python3 tools/show_record.py --run-id 'a b; rm -rf /'"


def test_math_is_finite_in_output():
    assert all(math.isfinite(u["measured"]) for u in by_run(golden_card()).values() if u["measured"] is not None)
