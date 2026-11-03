"""judging_json.py: fills run_ids and reported_results from result records, touching nothing else."""
from __future__ import annotations

import json
from pathlib import Path

import judging_json as jj


def result(run_id, *, passed=True, synthetic=False, created="2026-11-02T20:00:00Z", gain=None):
    rec = {
        "schema_version": "1", "kind": "result", "run_id": run_id, "team": "team13", "created_at": created,
        "synthetic": synthetic, "unit": "t3.deny", "metric": "ratio", "value": 1.0,
        "target": {"metric": "ratio", "op": "==", "value": 1, "unit": "ratio"},
        "passed": passed, "n_items": 27, "checked_at": created, "details": {},
    }
    if gain:
        rec["details"]["gain"] = gain
    return rec


def store(tmp_path: Path, recs: list[dict]) -> Path:
    d = tmp_path / "evidence" / "results"
    d.mkdir(parents=True)
    for r in recs:
        (d / f"{r['run_id']}.json").write_text(json.dumps(r))
    return tmp_path / "evidence"


def skeleton(tmp_path: Path) -> Path:
    p = tmp_path / "judging.json"
    p.write_text(json.dumps({"primary_track": "track-3", "team_code": "team13", "run_ids": [],
                             "run_ids_comment": ["slot 1"], "reported_results": [], "rebuild_command": "x"}))
    return p


def run(tmp_path, ev, *extra):
    return jj.main(["--dir", str(ev), "--judging", str(tmp_path / "judging.json"), *extra])


def test_fills_only_the_two_fields(tmp_path):
    ev = store(tmp_path, [result("b-run", created="2026-11-02T21:00:00Z"), result("a-run", gain={"value": 0.5, "ci_low": 0.2, "ci_high": 0.7, "baseline_run_id": "x", "extra": 1})])
    j = skeleton(tmp_path)
    assert run(tmp_path, ev) == 0
    doc = json.loads(j.read_text())
    assert doc["run_ids"] == ["a-run", "b-run"]  # by created_at, then id
    assert [r["run_id"] for r in doc["reported_results"]] == ["a-run", "b-run"]
    assert doc["reported_results"][0]["gain"] == {"value": 0.5, "ci_low": 0.2, "ci_high": 0.7, "baseline_run_id": "x"}
    assert doc["primary_track"] == "track-3" and doc["run_ids_comment"] == ["slot 1"] and doc["rebuild_command"] == "x"


def test_dry_run_writes_nothing(tmp_path, capsys):
    ev = store(tmp_path, [result("a-run")])
    j = skeleton(tmp_path)
    before = j.read_text()
    assert run(tmp_path, ev, "--dry-run") == 0
    assert j.read_text() == before
    assert "dry run: nothing written" in capsys.readouterr().out


def test_synthetic_superseded_exclude_include_and_order(tmp_path):
    ev = store(tmp_path, [result("s", synthetic=True), result("old"), result("x"), result("y"), result("z")])
    sup = tmp_path / "superseded.json"
    sup.write_text(json.dumps({"superseded": [{"run_ids": ["old"]}]}))
    order = tmp_path / "order.txt"
    order.write_text("z\ny\n")
    j = skeleton(tmp_path)
    assert run(tmp_path, ev, "--superseded", str(sup), "--exclude", "x", "--order-file", str(order)) == 0
    assert json.loads(j.read_text())["run_ids"] == ["z", "y"]
    assert run(tmp_path, ev, "--allow-synthetic", "--include", "s", "y") == 0
    assert sorted(json.loads(j.read_text())["run_ids"]) == ["s", "y"]


def test_more_than_twenty_is_refused_and_nothing_written(tmp_path, capsys):
    ev = store(tmp_path, [result(f"r{i:02d}") for i in range(21)])
    j = skeleton(tmp_path)
    before = j.read_text()
    assert run(tmp_path, ev) == 1
    assert j.read_text() == before
    assert "limit is 20" in capsys.readouterr().err
    assert run(tmp_path, ev, "--exclude", "r00") == 0
    assert len(json.loads(j.read_text())["run_ids"]) == 20


def test_bad_files_block_the_write_unless_skipped(tmp_path):
    ev = store(tmp_path, [result("good")])
    (ev / "results" / "junk.json").write_text("{not json")
    (ev / "results" / "wrong.json").write_text(json.dumps(result("other-name")))
    j = skeleton(tmp_path)
    before = j.read_text()
    assert run(tmp_path, ev) == 1 and j.read_text() == before
    assert run(tmp_path, ev, "--skip-bad") == 0
    assert json.loads(j.read_text())["run_ids"] == ["good"]


def test_failed_results_are_reported_as_failed(tmp_path):
    ev = store(tmp_path, [result("bad", passed=False)])
    skeleton(tmp_path)
    assert run(tmp_path, ev) == 0
    assert json.loads((tmp_path / "judging.json").read_text())["reported_results"][0]["passed"] is False


def test_missing_store_is_exit_2(tmp_path):
    skeleton(tmp_path)
    assert run(tmp_path, tmp_path / "nowhere") == 2


def test_s3_path_needs_a_valid_uri(tmp_path):
    skeleton(tmp_path)
    assert jj.main(["--s3", "not-a-uri", "--judging", str(tmp_path / "judging.json")]) == 2


def test_slots_report(tmp_path, capsys):
    j = tmp_path / "judging.json"
    doc = {"run_ids": ["a", "e1", "e2"],
           "slots": [{"slot": 1, "name": "x", "status": "filled", "run_id": "a"},
                     {"slot": 2, "name": "y", "status": "pending", "run_id": None, "waiting_for": "z", "base_run_id": "b"}],
           "edge_cases": [{"slot": "E1", "name": "w", "status": "filled", "run_ids": ["e1", "e2"]}]}
    j.write_text(json.dumps(doc))
    assert jj.main(["--slots-report", "--judging", str(j)]) == 0
    out = capsys.readouterr().out
    assert "1 of 2 slots filled; 1 of 1 edge cases filled; 3 run id(s) listed" in out
    doc["run_ids"] = ["a", "e1"]  # an edge-case run is missing from run_ids
    j.write_text(json.dumps(doc))
    assert jj.main(["--slots-report", "--judging", str(j)]) == 1
    doc["run_ids"] = ["a", "e1", "e2", "stray"]
    j.write_text(json.dumps(doc))
    assert jj.main(["--slots-report", "--judging", str(j)]) == 1
    doc["run_ids"] = ["a", "e1", "e2"]
    doc["slots"][1].update(status="filled")  # filled without naming a run
    j.write_text(json.dumps(doc))
    assert jj.main(["--slots-report", "--judging", str(j)]) == 1
