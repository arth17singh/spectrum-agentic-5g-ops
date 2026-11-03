"""The overlap re-check of an already-built dataset against the eval items at the current commit
(tiny dataset built here with the fake labeller; no AWS, no twin)."""

from __future__ import annotations

import json
import shutil

import pytest

from t1.sft import build, common, declare, overlap, recheck
from tools import check_unit, evlib

DATA_RUN = "t1.sft.data-built"


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    """A dataset written by the real builder (fake labeller), as the build job writes it."""
    tmp = tmp_path_factory.mktemp("built")
    d = declare.through_runner(declare.load("build"), run_id=DATA_RUN)
    (tmp / "decl.json").write_text(json.dumps(d))
    a = build.parse_args(["--run-id", DATA_RUN, "--out", str(tmp / "data"), "--workdir", str(tmp / "work"),
                          "--evidence-dir", str(tmp / "ev"), "--declared", str(tmp / "decl.json"),
                          "--n-execution", "5", "--n-interface", "20", "--n-safety", "30", "--workers", "1",
                          "--labeler", "t1.sft.testing:fake_reference"])
    build.run(a)
    return tmp / "data"


def _declared(tmp_path, run_id, data_dir, **patch):
    d = declare.through_runner(declare.recheck_declaration(
        DATA_RUN, (data_dir / "SHA256SUMS").read_bytes(), (data_dir / "manifest.json").read_bytes()), run_id=run_id)
    for k, v in patch.items():
        d["decoding"]["dataset"][k] = v
    p = tmp_path / f"{run_id}.json"
    p.write_text(json.dumps(d))
    return str(p)


def _args(tmp_path, run_id, data_dir, *extra, **patch):
    return recheck.parse_args(["--run-id", run_id, "--data", str(data_dir), "--data-run-id", DATA_RUN,
                               "--evidence-dir", str(tmp_path / "ev"), "--declared",
                               _declared(tmp_path, run_id, data_dir, **patch), "--summary-dir", str(tmp_path / "s"),
                               *extra])


def test_a_clean_dataset_passes_with_one_item_per_eval_item(tmp_path, dataset):
    s = recheck.run(_args(tmp_path, "t1.sft.data-recheck-ok", dataset))
    store = evlib.LocalStore(tmp_path / "ev")
    items = evlib.read_records("t1.sft.data-recheck-ok", "item", store)
    res = evlib.read_result("t1.sft.data-recheck-ok", store)
    assert len(items) == res["n_items"] == 1220 and res["unit"] == "t1.sft.data-recheck"
    assert check_unit.recompute("fail_count", items) == res["value"] == 0 and res["passed"] is True
    assert res["details"]["rows_overlapping_eval"] == 0 and res["details"]["dataset_run_id"] == DATA_RUN
    assert res["details"]["eval_items_manifest_sha256"] == build.EVAL_PINS["MANIFEST.json"]
    assert s["result"]["passed"] is True and (tmp_path / "s" / "recheck_summary.json").is_file()


def test_rows_that_match_eval_items_fail_exactly_those_items(tmp_path, dataset):
    leaky = tmp_path / "leaky"
    shutil.copytree(dataset, leaky)
    ev = overlap.read_jsonl(build.items_dir() / "execution.jsonl")[:2]
    rows = overlap.read_jsonl(leaky / "train.jsonl")
    for i, it in enumerate(ev):  # an execution row whose state is an eval item's (another seed, another note)
        rows.append({"id": f"leak-{i}", "unit": "execution", "family": "leak", "split": "train",
                     "template": "execution", "input": {"state": it["input"]["state"], "note": "n"},
                     "answer": {"decision": "apply", "change": it["expected"]["oracle_change"], "reason": "r"},
                     "messages": [], "meta": {"scenario": "nominal", "seed": 1 + i, "prefix": []}})
    build.write_jsonl(leaky / "train.jsonl", rows)
    (leaky / "SHA256SUMS").unlink()
    common.write_sha256sums(leaky)
    recheck.run(_args(tmp_path, "t1.sft.data-recheck-leak", leaky))
    store = evlib.LocalStore(tmp_path / "ev")
    res = evlib.read_result("t1.sft.data-recheck-leak", store)
    assert res["passed"] is False and res["details"]["rows_overlapping_eval"] == 2
    # the two execution items fail, and so does any price_latency item that reuses one of their prompts
    states = {it["input"]["state"] for it in ev}
    reused = [f"price_latency/{p['item_id']}" for p in overlap.read_jsonl(build.items_dir() / "price_latency.jsonl")
              if p["template"] == "execution" and p["input"]["state"] in states]
    want = sorted([f"execution/{it['item_id']}" for it in ev] + reused)
    items = evlib.read_records("t1.sft.data-recheck-leak", "item", store)
    assert sorted(i["item_id"] for i in items if not i["passed"]) == want and res["value"] == len(want)


def test_a_dataset_other_than_the_declared_one_is_refused(tmp_path, dataset):
    with pytest.raises(common.JobError, match="not the declared"):
        recheck.run(_args(tmp_path, "t1.sft.data-recheck-sums", dataset, sha256sums_sha256="0" * 64))
    other = tmp_path / "other"
    shutil.copytree(dataset, other)
    (other / "dev.jsonl").write_text("")
    with pytest.raises(common.JobError, match="SHA256SUMS"):
        recheck.run(_args(tmp_path, "t1.sft.data-recheck-tamper", other))
    a = _args(tmp_path, "t1.sft.data-recheck-runid", dataset)
    a.data_run_id = "t1.sft.data-someone-else"
    with pytest.raises(common.JobError, match="manifest is for run"):
        recheck.run(a)
    assert not (tmp_path / "ev" / "results").exists()


def test_eval_items_off_the_pins_are_refused(tmp_path, dataset, monkeypatch):
    monkeypatch.setitem(build.EVAL_PINS, "execution.jsonl", "0" * 64)
    with pytest.raises(common.JobError, match="EVAL_PINS"):
        recheck.run(_args(tmp_path, "t1.sft.data-recheck-pins", dataset))


def test_recheck_declaration_names_the_dataset(dataset):
    d = declare.recheck_declaration(DATA_RUN, (dataset / "SHA256SUMS").read_bytes(),
                                    (dataset / "manifest.json").read_bytes())
    assert d["decoding"]["dataset"]["sha256sums_sha256"] == common.sha256_file(dataset / "SHA256SUMS")
    assert d["target"] == {"metric": "fail_count", "op": "==", "value": 0, "unit": "count"} and d["n_items"] == 1220
    with pytest.raises(common.JobError):
        declare.recheck_declaration("t1.sft.data-other", (dataset / "SHA256SUMS").read_bytes(),
                                    (dataset / "manifest.json").read_bytes())
