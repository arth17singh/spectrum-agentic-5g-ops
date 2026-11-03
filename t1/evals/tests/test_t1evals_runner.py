"""Running under tools/sm: the replay-check / re-freeze job, TwinDrift messages, declarations that
match the launcher's, no git needed in the job, outputs and baselines via EVIDENCE_S3."""

from __future__ import annotations

import json

import pytest

from t1.evals import freeze  # first: puts contracts/ on sys.path when team13_contracts is not installed
from t1.evals import replay_job as J
from t1.evals import run as R
from team13_contracts import validate_record  # noqa: E402
from tools import evlib  # noqa: E402

from t1.evals.twin_access import on_freezing_platform  # noqa: E402

# Exact twin values replay only on the platform the execution items were frozen on
# (items/execution.platform.json); elsewhere `python -m t1.evals.replay_job check` measures the drift.
FROZEN_HERE = pytest.mark.skipif(not on_freezing_platform(),
                                 reason="execution items were frozen on another platform (see items/execution.platform.json)")



def _few(monkeypatch, n=3):
    lines = freeze.read_lines("execution")[:n]
    real = freeze.read_lines
    monkeypatch.setattr(freeze, "read_lines", lambda unit: lines if unit == "execution" else real(unit))
    return lines


def _validate_run(root, run_id):
    store = evlib.LocalStore(root)
    recs = [evlib.read_declared(run_id, store=store), evlib.read_result(run_id, store=store),
            *evlib.read_records(run_id, "item", store=store)]
    for r in recs:
        validate_record(r)
        evlib.validate(r)
    return recs


def test_replay_check_job_writes_valid_records(tmp_path, monkeypatch):
    """Runs anywhere (valid records, the platform reported); exact agreement only where frozen."""
    pytest.importorskip("ran_rl")
    _few(monkeypatch)
    assert J.main(["check", "--run-id", "t1.rc-test", "--evidence-dir", str(tmp_path), "--code-sha", "c" * 40]) == 0
    decl, res, *items = _validate_run(tmp_path, "t1.rc-test")
    assert len(items) == 3 and set(items[0]["extra"]["frozen"]) == set(J.KEYS) and res["details"]["platform"]["machine"]
    assert res["details"]["n_drifted"] == sum(not it["passed"] for it in items)
    if on_freezing_platform():
        assert all(it["passed"] and it["score"] == 0.0 for it in items)
        assert res["passed"] is True and res["value"] == 1.0 and res["details"]["n_drifted"] == 0


@FROZEN_HERE
def test_replay_check_reports_drift(tmp_path, monkeypatch):
    pytest.importorskip("ran_rl")
    lines = _few(monkeypatch, 2)
    it = json.loads(lines[0])
    it["expected"]["rule_gain"] += 0.004
    bad = [json.dumps(it, sort_keys=True, separators=(",", ":"), ensure_ascii=False), lines[1]]
    monkeypatch.setattr(freeze, "read_lines", lambda unit: bad)
    J.main(["check", "--run-id", "t1.rc-drift", "--evidence-dir", str(tmp_path), "--code-sha", "c" * 40])
    _, res, *items = _validate_run(tmp_path, "t1.rc-drift")
    assert res["passed"] is False and res["details"]["n_drifted"] == 1
    assert res["details"]["max_abs_diff_by_value"]["rule_gain"] == pytest.approx(0.004)
    assert "refreeze" in res["details"]["verdict"]


@FROZEN_HERE
def test_refreeze_is_a_fixed_point_on_the_freezing_platform(tmp_path, monkeypatch):
    pytest.importorskip("ran_rl")
    lines = _few(monkeypatch, 2)
    out = tmp_path / "out"
    J.main(["refreeze", "--run-id", "t1.rf-test", "--evidence-dir", str(tmp_path / "ev"), "--output-dir", str(out),
            "--code-sha", "c" * 40])
    assert (out / "execution.jsonl").read_text().splitlines() == lines
    summary = json.loads((out / "refreeze-summary.json").read_text())
    assert summary["n_changed"] == 0 and summary["hashes"]["execution"]["n_items"] == 2
    _validate_run(tmp_path / "ev", "t1.rf-test")


@FROZEN_HERE
def test_twin_drift_names_the_value_and_the_difference():
    pytest.importorskip("ran_rl")
    from t1.evals.graders import grade_execution
    from t1.evals.twin_access import TwinDrift

    it = freeze.load_items("execution")[0]
    exp = it["expected"]
    tampered = dict(it, expected=dict(exp, rule_gain=exp["rule_gain"] + 0.0025))
    with pytest.raises(TwinDrift) as e:
        grade_execution(tampered, json.dumps({"decision": "apply", "change": exp["oracle_change"]}))
    msg = str(e.value)
    assert "rule_gain" in msg and "noop_kpi" not in msg and "2.500e-03" in msg and it["item_id"] in msg
    assert "replay_job check" in msg


@pytest.mark.parametrize("unit", ["interface", "execution", "knowledge", "price_latency"])
def test_print_declaration_matches_what_the_run_declares(tmp_path, unit, capsys):
    """The launcher stores --print-declaration's fields; tools/sm/entry.py refuses the upload if the
    module's own declaration differs on a judged field. Check with the entry's own comparison."""
    if unit == "execution":
        pytest.importorskip("ran_rl")
    from tools.sm.entry import judged_diff

    args = ["--unit", unit, "--client", "fake", "--limit", "3", "--evidence-dir", str(tmp_path)]
    assert R.main([*args, "--print-declaration"]) == 0
    printed = json.loads(capsys.readouterr().out)
    printed.update(run_id="t1.decl", code_sha="d" * 40)
    R.run(R.parser().parse_args([*args, "--run-id", "t1.decl", "--code-sha", "d" * 40]))
    mine = evlib.read_declared("t1.decl", store=evlib.LocalStore(tmp_path))
    assert judged_diff(mine, printed) == []


def test_replay_job_declaration_matches(tmp_path, monkeypatch, capsys):
    pytest.importorskip("ran_rl")
    from tools.sm.entry import judged_diff

    _few(monkeypatch, 1)
    J.main(["check", "--print-declaration"])
    printed = json.loads(capsys.readouterr().out)
    printed.update(code_sha="e" * 40)
    J.main(["check", "--run-id", "t1.rc-decl", "--evidence-dir", str(tmp_path), "--code-sha", "e" * 40])
    assert judged_diff(evlib.read_declared("t1.rc-decl", store=evlib.LocalStore(tmp_path)), printed) == []


def test_no_git_in_a_job(monkeypatch):
    """The job has no .git: CODE_SHA (set by the runner) wins and git is never asked."""
    monkeypatch.setenv("CODE_SHA", "f" * 40)
    monkeypatch.setattr(R, "code_sha", lambda: (_ for _ in ()).throw(AssertionError("git called")))
    assert R.resolve_code_sha(None) == "f" * 40
    monkeypatch.setattr(R, "code_dirty", lambda: (_ for _ in ()).throw(FileNotFoundError("git")))
    assert R.safe_dirty() is None


def test_outputs_and_baseline_use_s3_side_store_only_with_evidence_s3(tmp_path, monkeypatch):
    from t1.evals.evidence import Evidence

    monkeypatch.delenv("EVIDENCE_S3", raising=False)
    assert Evidence(str(tmp_path)).side is None
    monkeypatch.setenv("EVIDENCE_S3", "s3://example-bucket/evidence/")
    ev = Evidence(str(tmp_path))
    assert isinstance(ev.side, evlib.S3Store) and ev.side.bucket == "example-bucket"  # no AWS call is made here


def test_price_latency_default_target_is_plan_v51():
    assert R.DEFAULT_TARGET["price_latency"] == {"metric": "p95_latency_ms", "op": "<=", "value": 3000.0, "unit": "ms"}
