"""run_unit: declare -> run -> collect -> result, locally (the check step needs AWS, so it is not run here)."""

import json
import sys
import textwrap
from pathlib import Path

import pytest

import evlib
import run_unit

TOOLS = Path(__file__).resolve().parents[1]


@pytest.fixture
def setup(tmp_path, monkeypatch):
    ev = tmp_path / "ev"
    monkeypatch.setenv("EVIDENCE_DIR", str(ev))
    test_set = tmp_path / "suite.jsonl"
    test_set.write_text("".join(json.dumps({"id": f"case-{i}"}) + "\n" for i in range(4)))
    cmd = tmp_path / "suite.py"
    cmd.write_text(textwrap.dedent(f"""
        import os, sys
        sys.path.insert(0, {str(TOOLS)!r})
        import evlib
        rid = os.environ["EVIDENCE_RUN_ID"]
        decl = evlib.read_declared(rid)            # the declaration exists before the command runs
        assert decl and decl["declared_at"]
        fail = os.environ.get("FAIL_ITEMS", "")
        evlib.append_records(rid, "item", [
            {{"item_id": f"case-{{i}}", "input_sha256": "a" * 64, "output_sha256": "b" * 64, "grader": "g",
              "backend": "fake", "passed": str(i) not in fail, "latency_ms": 100 + i,
              "aws_time": "2026-11-02T18:10:00Z", "aws_time_source": "cloudwatch:/team13/checkpoint"}}
            for i in range(4)])
    """))
    monkeypatch.setattr(run_unit.evlib, "code_dirty", lambda repo=None: False)
    return ev, test_set, cmd


def args(test_set, cmd, *extra):
    return ["--unit", "t3.deny", "--track", "track-3", "--metric", "pass_rate", "--op", ">=", "--value", "0.75",
            "--test-set", str(test_set), "--model-id", "m", "--grader", "g", "--gap-s", "0", "--synthetic",
            "--no-check", *extra, "--", sys.executable, str(cmd)]


def test_declare_run_collect_result(setup, capsys):
    ev, test_set, cmd = setup
    assert run_unit.main(args(test_set, cmd, "--run-id", "t3.deny-ru-1")) == 0
    store = evlib.LocalStore(ev)
    d = evlib.read_declared("t3.deny-ru-1", store)
    assert d["n_items"] == 4 and d["test_set_sha256"] == evlib.sha256_file(test_set) and d["synthetic"] is True
    assert d["code_sha"] == evlib.code_sha()
    r = evlib.read_result("t3.deny-ru-1", store)
    assert r["value"] == 1.0 and r["passed"] is True and r["checker"]["script"] == "tools/run_unit.py"
    assert r["details"]["command_rc"] == 0 and "command_s" in r["details"]["timing_s"]
    assert evlib.parse_time(d["declared_at"]) <= evlib.parse_time(r["details"]["command_started"])
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["records"] == {"item": 4} and "total_s" in summary["timing_s"]


def test_baseline_then_delta_target(setup, monkeypatch):
    ev, test_set, cmd = setup
    monkeypatch.setenv("FAIL_ITEMS", "12")  # the baseline command fails items 1 and 2 -> 0.5
    a = args(test_set, cmd, "--run-id", "t3.gain-1", "--baseline-cmd", f"{sys.executable} {cmd}")
    a[a.index("--value"):a.index("--value") + 2] = ["--delta-over-baseline", "0.25"]
    run_unit.main(a)
    store = evlib.LocalStore(ev)
    assert evlib.read_result("t3.gain-1-base", store)["value"] == 0.5
    d = evlib.read_declared("t3.gain-1", store)
    assert d["baseline_run_id"] == "t3.gain-1-base" and d["target"]["value"] == 0.75


def test_refuses_dirty_tree_and_rerun_of_same_run_id(setup, monkeypatch):
    ev, test_set, cmd = setup
    monkeypatch.setattr(run_unit.evlib, "code_dirty", lambda repo=None: True)
    assert run_unit.main(args(test_set, cmd, "--run-id", "t3.x-1")) == 2
    assert not (ev / "declared").exists()
    monkeypatch.setattr(run_unit.evlib, "code_dirty", lambda repo=None: False)
    assert run_unit.main(args(test_set, cmd, "--run-id", "t3.x-1")) == 0
    with pytest.raises(FileExistsError):  # a declaration is never rewritten
        run_unit.main(args(test_set, cmd, "--run-id", "t3.x-1"))


def test_check_command_is_a_separate_process():
    c = run_unit.check_command("r1", ["--s3", "s3://b/evidence/"])
    assert c[0] == sys.executable and c[1].endswith("tools/check_unit.py") and c[2:] == ["r1", "--s3", "s3://b/evidence/"]
