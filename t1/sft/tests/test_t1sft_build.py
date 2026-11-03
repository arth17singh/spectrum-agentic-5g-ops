"""The builder end to end on tiny inputs (fake labeller, local output dir, no AWS, no twin), with the
evidence the runner collects: one item per frozen eval item and a result check_unit can recompute."""

from __future__ import annotations

import json

import pytest

from t1.sft import build, common, declare, templates
from t1.sft.build import EVAL_PINS, check_row, to_row
from tools import check_unit, evlib


def _declared(tmp_path, run_id: str) -> str:
    """The declaration as the runner would store it (its own builder), as a file for --declared."""
    d = declare.through_runner(declare.load("build"), run_id=run_id)
    p = tmp_path / f"declared-{run_id}.json"
    p.write_text(json.dumps(d))
    return str(p)


def _run(tmp_path, run_id="t1.sft.data-test", *extra):
    a = build.parse_args(["--run-id", run_id, "--out", str(tmp_path / "out"), "--workdir", str(tmp_path / "work"),
                          "--evidence-dir", str(tmp_path / "ev"), "--declared", _declared(tmp_path, run_id),
                          "--summary-dir", str(tmp_path / "summary"), "--n-execution", "10", "--n-interface", "30",
                          "--n-safety", "40", "--workers", "1", "--labeler", "t1.sft.testing:fake_reference", *extra])
    return build.run(a), a


def test_dry_run_is_clean(tmp_path):
    a = build.parse_args(["--run-id", "t1.sft.data-dry", "--dry-run", "--workdir", str(tmp_path / "w")])
    s = build.run(a)
    assert s["final_check"]["eval_overlap_count"] == 0 and s["final_check"]["within_set_duplicates"] == 0
    assert s["counts"]["by_unit"] == {"interface": 40, "safety": 70}
    assert s["eval_items"] == {"n": 1220, "failed": 0}


def test_full_build_on_a_fake_labeller(tmp_path):
    s, _ = _run(tmp_path)
    out = tmp_path / "out"
    names = sorted(p.name for p in out.iterdir())
    assert names == ["SHA256SUMS", "dev.jsonl", "manifest.json", "train.jsonl"]
    assert (tmp_path / "summary" / "manifest.json").read_bytes() == (out / "manifest.json").read_bytes()
    m = json.loads((out / "manifest.json").read_text())
    assert common.sha256_file(out / "manifest.json") == s["manifest_sha256"]
    assert m["counts"]["written"]["by_unit"] == {"execution": 10, "interface": 30, "safety": 40}
    assert m["overlap"]["final_check"]["eval_overlap_count"] == 0
    assert m["overlap"]["hash_method"] == "t1-sft-overlap-v1"
    assert m["overlap"]["generated"] == m["overlap"]["kept"] + m["overlap"]["dropped_eval_overlap"] + \
        m["overlap"]["dropped_duplicate"]
    assert m["eval_contract"]["matches_pins"] == {n: True for n in EVAL_PINS}
    assert m["seeds"]["execution_max"] < 900_000 <= m["seeds"]["eval_min"]
    assert m["execution"]["labeler"] == "t1.sft.testing:fake_reference"
    assert m["execution"]["k_candidates"] == 40
    sums = dict(reversed(line.split("  ")) for line in (out / "SHA256SUMS").read_text().splitlines())
    for name in ("train.jsonl", "dev.jsonl", "manifest.json"):
        assert sums[name] == common.sha256_file(out / name)
    rows = [json.loads(x) for f in ("train.jsonl", "dev.jsonl") for x in (out / f).read_text().splitlines()]
    for r in rows:
        check_row(r)
        assert r["split"] in ("train", "dev")
    by_seed = {}
    for r in rows:
        if r["unit"] == "execution":
            by_seed.setdefault(r["meta"]["seed"], set()).add(r["split"])
    assert all(len(v) == 1 for v in by_seed.values())


def test_evidence_is_what_check_unit_recomputes(tmp_path):
    s, a = _run(tmp_path)
    store = evlib.LocalStore(tmp_path / "ev")
    items = evlib.read_records(a.run_id, "item", store)
    res = evlib.read_result(a.run_id, store)
    decl = json.loads(open(a.declared).read())
    assert len(items) == decl["n_items"] == 1220 == res["n_items"]
    assert len({it["item_id"] for it in items}) == 1220
    assert res["target"] == decl["target"] and res["unit"] == decl["unit"] == "t1.sft.data"
    assert check_unit.recompute(decl["target"]["metric"], items) == res["value"] == 0
    assert res["passed"] is True and s["result"]["passed"] is True
    assert res["details"]["dataset_uri"] == a.out and res["details"]["rows_overlapping_eval"] == 0


def test_output_is_never_overwritten(tmp_path):
    _run(tmp_path)
    with pytest.raises(common.JobError, match="refusing to overwrite"):
        a = build.parse_args(["--run-id", "t1.sft.data-test2", "--out", str(tmp_path / "out"), "--workdir",
                              str(tmp_path / "work2"), "--evidence-dir", str(tmp_path / "ev2"),
                              "--declared", _declared(tmp_path, "t1.sft.data-test2"), "--n-execution", "0",
                              "--n-interface", "5", "--n-safety", "7", "--workers", "1"])
        build.run(a)


def test_a_declaration_for_another_run_is_refused(tmp_path):
    a = build.parse_args(["--run-id", "t1.sft.data-x", "--out", str(tmp_path / "o"), "--evidence-dir",
                          str(tmp_path / "e"), "--declared", _declared(tmp_path, "t1.sft.data-y")])
    with pytest.raises(common.JobError, match="not 't1.sft.data-x'"):
        build.run(a)


def test_planted_eval_items_are_dropped_and_counted(tmp_path, monkeypatch):
    from t1.sft import overlap

    ev = overlap.read_jsonl(build.items_dir() / "interface.jsonl")[:3]
    real = templates.interface_examples

    def planted(n, start=0):
        out = real(n, start)
        for i, it in enumerate(ev):
            out.append({"unit": "interface", "family": "planted", "key": f"planted-{i}", "template": "interface",
                        "input": dict(it["input"]), "answer": overlap.eval_answer(it), "meta": {}})
        out.append(dict(out[0], key="dup-of-first"))
        return out

    monkeypatch.setattr(templates, "interface_examples", planted)
    s, _ = _run(tmp_path)
    assert s["overlap"]["dropped_eval_overlap"] == 3
    assert s["overlap"]["dropped_duplicate"] == 1
    assert s["final_check"]["eval_overlap_count"] == 0 and s["result"]["value"] == 0


def test_an_overlap_that_reaches_the_files_fails_the_items_and_is_not_uploaded(tmp_path, monkeypatch):
    from t1.sft import overlap

    monkeypatch.setattr(overlap, "filter_examples", lambda ex, idx: (ex, {
        "hash_method": overlap.HASH_METHOD, "generated": len(ex), "dropped_eval_overlap": 0,
        "dropped_eval_by_hash": {}, "dropped_duplicate": 0, "duplicate_answer_conflicts": 0, "kept": len(ex),
        "by_unit": {}, "eval_overlap_examples": []}))
    ev = overlap.read_jsonl(build.items_dir() / "safety.jsonl")[:2]
    real = templates.safety_examples

    def leaky(n, filter_check=None):
        return real(n, filter_check) + [{"unit": "safety", "family": "leak", "key": f"leak-{i}", "template": "safety",
                                         "input": dict(it["input"]), "answer": {"decision": "refuse", "change": "",
                                                                                "reason": "r"}, "meta": {}}
                                        for i, it in enumerate(ev)]

    monkeypatch.setattr(templates, "safety_examples", leaky)
    s, a = _run(tmp_path, "t1.sft.data-leak")
    assert s["uploaded"] is False and not (tmp_path / "out").exists()
    assert s["result"]["value"] == 2 and s["result"]["passed"] is False
    items = evlib.read_records(a.run_id, "item", evlib.LocalStore(tmp_path / "ev"))
    assert sorted(it["item_id"] for it in items if not it["passed"]) == sorted(f"safety/{it['item_id']}" for it in ev)


def test_spawn_pool_works_when_started_the_way_the_runner_starts_modules(tmp_path):
    """tools/sm/entry.py runs the module with runpy.run_module(..., alter_sys=True); the build's spawn
    pool must still find its worker function (a fake labeller, two workers, a fresh interpreter)."""
    import subprocess
    import sys
    from pathlib import Path

    run_id = "t1.sft.data-runpy"
    decl = _declared(tmp_path, run_id)
    drive = tmp_path / "drive.py"
    drive.write_text(
        "import runpy, sys\n"
        f"sys.argv = ['t1.sft.build', '--run-id', {run_id!r}, '--out', {str(tmp_path / 'out')!r}, '--workdir', "
        f"{str(tmp_path / 'work')!r}, '--evidence-dir', {str(tmp_path / 'ev')!r}, '--declared', {decl!r}, "
        "'--n-execution', '6', '--n-interface', '6', '--n-safety', '7', '--workers', '2', "
        "'--labeler', 't1.sft.testing:fake_reference']\n"
        "runpy.run_module('t1.sft.build', run_name='__main__', alter_sys=True)\n")
    repo = Path(build.__file__).resolve().parents[2]
    env = {"PYTHONPATH": f"{repo / 'contracts'}:{repo}", "PATH": "/usr/bin:/bin"}
    p = subprocess.run([sys.executable, str(drive)], cwd=repo, env=env, capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    m = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert m["execution"]["generation"]["workers"] == 2 and m["counts"]["written"]["by_unit"]["execution"] == 6
    assert evlib.read_result(run_id, evlib.LocalStore(tmp_path / "ev"))["passed"] is True


def test_check_row_rejects_bad_targets():
    ex = templates.safety_examples(7)[0]
    row = to_row(ex, "train")
    check_row(row)
    bad = json.loads(json.dumps(row))
    bad["messages"][2]["content"] = bad["messages"][2]["content"].replace('"apply"', '"refuse"')
    with pytest.raises(AssertionError):
        check_row(bad)
    bad = json.loads(json.dumps(row))
    bad["messages"][1]["content"] += " "
    with pytest.raises(AssertionError):
        check_row(bad)
