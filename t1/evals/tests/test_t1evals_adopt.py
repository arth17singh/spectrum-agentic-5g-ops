"""Adopting a re-freeze job's output: guards, superseded hashes, platform record, MANIFEST."""

from __future__ import annotations

import json
import shutil

import pytest

from t1.evals import adopt_refreeze as A
from t1.evals import freeze
from t1.evals.common import dumps, lines_sha256


@pytest.fixture()
def items_copy(tmp_path, monkeypatch):
    d = tmp_path / "items"
    shutil.copytree(freeze.ITEMS_DIR, d)
    monkeypatch.setattr(freeze, "ITEMS_DIR", d)
    monkeypatch.setattr(freeze, "MANIFEST", d / "MANIFEST.json")
    monkeypatch.setattr(freeze, "SUPERSEDED_FILE", d / "superseded.json")
    monkeypatch.setattr(freeze, "PLATFORM_FILE", d / "execution.platform.json")
    return d


def _job_output(tmp_path, items_dir, mutate=True, run_id="t1.execution.refreeze-x86-1"):
    src = tmp_path / "job"
    src.mkdir()
    ex = [json.loads(x) for x in (items_dir / "execution.jsonl").read_text().splitlines()]
    if mutate:
        ex[9]["expected"]["rule_gain"] += 1 / 60  # one UE flips on the other platform
    ex_lines = [dumps(it) for it in ex]
    (src / "execution.jsonl").write_text("".join(x + "\n" for x in ex_lines))
    pl_lines = (items_dir / "price_latency.jsonl").read_text().splitlines()
    (src / "price_latency.jsonl").write_text("".join(x + "\n" for x in pl_lines))
    old = {u: lines_sha256((items_dir / f"{u}.jsonl").read_text().splitlines()) for u in A.UNITS}
    summary = {"run_id": run_id, "platform": {"system": "Linux", "machine": "x86_64", "python": "3.12.10",
                                              "versions": {"torch": "2.9.0+cu130", "ran-rl": "1.2.9"}},
               "hashes": {"execution": {"sha256": lines_sha256(ex_lines), "n_items": len(ex_lines),
                                        "previous_sha256": old["execution"]},
                          "price_latency": {"sha256": lines_sha256(pl_lines), "n_items": len(pl_lines),
                                            "previous_sha256": old["price_latency"]}},
               "n_changed": 1, "n_state_text_changed": 0, "n_changes_changed": 0, "n_not_kept_by_filter_here": 0}
    (src / "refreeze-summary.json").write_text(json.dumps(summary))
    return src, old


def test_adopt_records_superseded_platform_and_manifest(tmp_path, items_copy):
    src, old = _job_output(tmp_path, items_copy)
    out = A.adopt(src, "t1.execution.refreeze-x86-1")
    assert out["changed"] == ["execution"]  # price_latency unchanged -> not superseded
    sup = freeze.load_superseded()
    assert sup[-1]["run_id"] == "t1.execution.refreeze-x86-1"
    assert sup[-1]["units"]["execution"]["sha256"] == old["execution"]
    assert freeze.frozen_on()["frozen_on"]["machine"] == "x86_64"
    assert freeze.frozen_on()["previous"][0]["frozen_on"]["machine"] == "arm64"
    assert freeze.verify() == []
    m = json.loads(freeze.MANIFEST.read_text())
    assert m["units"]["execution"]["sha256"] == out["new"]["execution"]["sha256"] != old["execution"]
    assert m["units"]["execution"]["frozen_on"]["frozen_on"]["machine"] == "x86_64"


def test_adopt_refuses_a_stale_or_tampered_output(tmp_path, items_copy):
    src, _ = _job_output(tmp_path, items_copy)
    with pytest.raises(SystemExit, match="not"):
        A.adopt(src, "another-run")
    s = json.loads((src / "refreeze-summary.json").read_text())
    s["hashes"]["execution"]["previous_sha256"] = "0" * 64
    (src / "refreeze-summary.json").write_text(json.dumps(s))
    with pytest.raises(SystemExit, match="started from"):
        A.adopt(src, "t1.execution.refreeze-x86-1")


def test_adopt_refuses_identity_changes(tmp_path, items_copy):
    src, _ = _job_output(tmp_path, items_copy)
    ex = [json.loads(x) for x in (src / "execution.jsonl").read_text().splitlines()]
    ex[0]["meta"]["seed"] += 1
    lines = [dumps(it) for it in ex]
    (src / "execution.jsonl").write_text("".join(x + "\n" for x in lines))
    s = json.loads((src / "refreeze-summary.json").read_text())
    s["hashes"]["execution"]["sha256"] = lines_sha256(lines)
    (src / "refreeze-summary.json").write_text(json.dumps(s))
    with pytest.raises(SystemExit, match="identity"):
        A.adopt(src, "t1.execution.refreeze-x86-1")


def test_adopt_nothing_when_identical(tmp_path, items_copy):
    src, _ = _job_output(tmp_path, items_copy, mutate=False)
    assert A.adopt(src, "t1.execution.refreeze-x86-1")["changed"] == []
    assert freeze.verify() == []
