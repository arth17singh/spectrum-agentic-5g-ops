"""Trace logic over synthetic fixtures (all records carry synthetic: true)."""

import copy
import json
from pathlib import Path

from spectrum.slices.trace import logic

FIX = Path(__file__).parent / "fixtures"


def load(name):
    d = json.loads((FIX / f"{name}.json").read_text())
    return d["decision"], d["change"], d["control"]


def timeline(name, case):
    d, c, k = load(name)
    return logic.build_timeline(case, d, c, k)


def test_fixtures_are_marked_synthetic():
    for name in ("complete", "deny", "paused", "no_approval"):
        d, c, k = load(name)
        assert all(r["synthetic"] is True and r["team"] == "team13" for r in d + c + k)


def test_complete_case_passes_the_declared_sequence():
    tl = timeline("complete", "case-complete")
    chk = logic.check_sequence(tl)
    assert chk["passed"] is True and chk["order_ok"] is True and chk["missing"] == []
    assert [e["step"] for e in tl] == ["case_opened", "ask_agent", "tool_call", "tool_call", "approval", "approval", "change_applied", "observe"]
    assert [e["category"] for e in tl if e["step"] == "approval"] == ["approval", "allow"]
    applied = tl[6]
    d = applied["detail"]
    assert applied["kind"] == "change" and d["predicted_delta"]["coverage_ratio"] == 0.05
    assert d["measured_delta"]["coverage_ratio"] == 0.04 and d["scored"] is True and d["probe"] is False
    assert d["change_text"] == "cell=2;d_power_db=+0.0;d_tilt_deg=-2.0" and d["log_line_seen"] is False
    assert d["baseline_kpis"]["coverage_ratio"] == 0.80 and d["item_id"] == "step-0"
    assert applied["approval_id"] == "chg-aaaaaaaaaaaa"


def test_timeline_entry_fields():
    e = timeline("complete", "case-complete")[3]
    assert set(e) >= {"time", "source", "agent", "tool", "decision", "deciding_rule", "reason", "guardrail_scores", "approval_id", "detail"}
    assert e["guardrail_scores"] == {"prompt_attack": 0.01, "pii": 0.0}
    assert e["time"] == "2026-11-02T20:01:12Z"


def test_unscored_clipped_change_is_carried_through():
    d = timeline("deny", "case-deny")[-2]["detail"]
    assert d["scored"] is False and d["unscored_reason"] == "clipped"


def test_probe_change_has_no_prediction_and_log_line_is_deduplicated():
    tl = timeline("probe", "case-probe")
    applied = [e for e in tl if e["step"] == "change_applied"]
    assert len(applied) == 1 and applied[0]["kind"] == "change"  # the log line is folded into the change record
    d = applied[0]["detail"]
    assert d["probe"] is True and "predicted_delta" not in d and d["log_line_seen"] is True
    assert logic.check_sequence(tl)["passed"] is True


def test_log_line_alone_is_a_change_applied_event():
    d, c, k = load("probe")
    tl = logic.build_timeline("case-probe", d, [], k)
    applied = [e for e in tl if e["step"] == "change_applied"]
    assert len(applied) == 1 and applied[0]["kind"] == "log" and applied[0]["source"] == "/aws/lambda/team13-ran-tools"
    assert applied[0]["detail"]["change_text"] == "cell=2;d_power_db=+0.0;d_tilt_deg=-2.0"
    assert logic.check_sequence(tl)["missing"] == []
    row = logic.build_cases(d, [], k)[0]
    assert row["applied"] == 1 and row["allows"] == 5  # the log line is not a Cedar decision


def test_deny_case_has_red_entry_and_still_passes():
    tl = timeline("deny", "case-deny")
    denies = [e for e in tl if e["category"] == "deny"]
    assert len(denies) == 1 and denies[0]["agent"] == "specialist"
    assert logic.check_sequence(tl)["passed"] is True


def test_paused_case_is_flagged_and_incomplete():
    tl = timeline("paused", "case-paused")
    assert [e["category"] for e in tl if e["category"] == "pause"] == ["pause"]
    ctl = [e for e in tl if e["kind"] == "control"]
    assert len(ctl) == 1 and ctl[0]["decision"] == "PAUSE" and ctl[0]["agent"] == "trust-gate"
    chk = logic.check_sequence(tl)
    assert chk["passed"] is False
    assert set(chk["missing"]) == {"tool_call", "approval", "change_applied", "observe"}
    assert any("paused" in n for n in chk["notes"])


def test_missing_approval_is_flagged():
    tl = timeline("no_approval", "case-no-approval")
    chk = logic.check_sequence(tl)
    assert chk["passed"] is False and chk["missing"] == ["approval"]
    assert any("no ALLOWed apply_change" in n for n in chk["notes"])


def test_out_of_order_is_flagged():
    d, c, k = load("complete")
    c = copy.deepcopy(c)
    c[0]["aws_time"] = "2026-11-02T20:01:05Z"  # the change was measured before it was approved
    chk = logic.check_sequence(logic.build_timeline("case-complete", d, c, k))
    assert chk["order_ok"] is False and chk["passed"] is False and chk["missing"] == []


def test_denied_steps_do_not_count_and_approval_request_is_noted():
    d, c, k = load("complete")
    d = [r for r in d if not (r["tool"].endswith("apply_change") and r["decision"] == "ALLOW")]
    chk = logic.check_sequence(logic.build_timeline("case-complete", d, c, k))
    assert "approval" in chk["missing"]
    assert chk["steps"][3]["present"] is False


def test_aws_time_is_preferred_over_own_time():
    d, c, k = load("complete")
    d = copy.deepcopy(d)
    d[0]["aws_time"] = "2026-11-02T20:00:30Z"
    assert logic.build_timeline("case-complete", d, c, k)[0]["time"] == "2026-11-02T20:00:30Z"
    d[0]["aws_time"] = None
    assert logic.build_timeline("case-complete", d, c, k)[0]["time"] == d[0]["time"]


def test_build_cases_lists_each_case_with_counts():
    d, c, k = [], [], []
    for n in ("complete", "deny", "paused", "no_approval"):
        a, b, x = load(n)
        d += a
        c += b
        k += x
    rows = {r["case_id"]: r for r in logic.build_cases(d, c, k)}
    assert set(rows) == {"case-complete", "case-deny", "case-paused", "case-no-approval"}
    r = rows["case-complete"]
    assert (r["run_id"], r["n_events"], r["denies"], r["allows"], r["pauses"], r["applied"]) == ("run-synthetic-1", 8, 1, 6, 0, 1)
    assert r["first_time"] == "2026-11-02T20:01:00Z" and r["last_time"] == "2026-11-02T20:01:50Z"
    assert rows["case-deny"]["denies"] == 2
    assert rows["case-paused"]["pauses"] == 1 and rows["case-paused"]["check_passed"] is False
    assert rows["case-no-approval"]["check_passed"] is False and rows["case-complete"]["check_passed"] is True
    assert all(r["synthetic"] for r in rows.values())
    # newest first
    assert [r["case_id"] for r in logic.build_cases(d, c, k)][0] == "case-no-approval"


def test_build_case_and_command_quoting():
    d, c, k = load("complete")
    out = logic.build_case("case-complete", d, c, k)
    assert out["show_record"] == "python3 tools/show_record.py --run-id run-synthetic-1 --item case-complete"
    assert out["run_id"] == "run-synthetic-1" and out["check"]["passed"] is True and out["other_runs"] == []
    assert logic.build_case("nope", d, c, k) is None
    d2 = copy.deepcopy(d)
    for r in d2:
        r["case_id"] = "x; rm -rf /"
    assert "'x; rm -rf /'" in logic.build_case("x; rm -rf /", d2, [], [])["show_record"]


def test_same_case_id_in_two_runs_is_disambiguated():
    d, c, k = load("complete")
    d2 = copy.deepcopy(d)
    for r in d2:
        r["run_id"] = "run-synthetic-2"
        r["aws_time"] = r["time"] = r["time"].replace("20:01", "21:01")
    allrec = d + d2
    out = logic.build_case("case-complete", allrec, c, k)
    assert out["run_id"] == "run-synthetic-2" and out["other_runs"] == ["run-synthetic-1"]
    assert logic.build_case("case-complete", allrec, c, k, run_id="run-synthetic-1")["run_id"] == "run-synthetic-1"
    assert len(logic.build_cases(allrec, c, k)) == 2


def test_empty_inputs():
    assert logic.build_cases([], [], []) == []
    assert logic.build_timeline("c", [], [], []) == []
    assert logic.check_sequence([])["missing"] == list(logic.REQUIRED)
