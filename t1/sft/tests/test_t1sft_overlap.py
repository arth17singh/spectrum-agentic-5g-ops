"""The overlap check works on content hashes, not seed ranges (tiny synthetic inputs + the frozen items)."""

from __future__ import annotations

import json
import shutil

import pytest

from t1.sft import overlap
from t1.sft.build import EVAL_PINS, items_dir


@pytest.fixture(scope="module")
def idx():
    return overlap.build_eval_index(items_dir())


@pytest.fixture(scope="module")
def eval_items():
    return {u: overlap.read_jsonl(items_dir() / f"{u}.jsonl") for u in overlap.EVAL_UNITS}


def _ex(unit, template, inp, answer, meta=None, key="k"):
    return {"unit": unit, "family": "f", "key": key, "template": template, "input": inp, "answer": answer,
            "meta": meta or {}}


def test_index_covers_every_frozen_item(idx):
    s = idx.summary()
    assert s["n_items_total"] == 1220
    assert s["n_items"] == {"interface": 320, "execution": 80, "safety": 170, "knowledge": 600, "price_latency": 50}
    assert s["exec_seed_min"] >= 900_000
    assert idx.manifest_sha256 == EVAL_PINS["MANIFEST.json"]
    assert all(idx.files[n] == sha for n, sha in EVAL_PINS.items() if n != "MANIFEST.json")


def test_index_refuses_items_that_differ_from_the_manifest(tmp_path):
    d = tmp_path / "items"
    shutil.copytree(items_dir(), d)
    p = d / "interface.jsonl"
    p.write_text(p.read_text() + "\n")
    with pytest.raises(ValueError, match="not the frozen item set"):
        overlap.build_eval_index(d)


def test_planted_eval_item_is_dropped_then_clean(idx, eval_items):
    it = eval_items["interface"][0]
    planted = _ex("interface", "interface", dict(it["input"]), overlap.eval_answer(it), key="planted")
    ours = _ex("interface", "interface", {"request": "Add 2 dB of transmit power on cell 1."},
               {"intent": "adjust_power", "cells": [1], "parameter": "power", "direction": "up", "amount": 2.0,
                "kpi": None}, key="ours")
    kept, rep = overlap.filter_examples([planted, ours], idx)
    assert rep["generated"] == 2 and rep["dropped_eval_overlap"] == 1 and rep["kept"] == 1
    assert set(rep["eval_overlap_examples"][0]["hashes"]) >= {"item_hash", "input_hash", "prompt_hash"}
    assert rep["eval_overlap_examples"][0]["eval_item"] == it["item_id"]
    assert [k["key"] for k in kept] == ["ours"]
    assert overlap.check_disjoint(kept, idx)["eval_overlap_count"] == 0
    assert overlap.check_disjoint([planted], idx)["eval_overlap_count"] == 1


def test_normalisation_catches_case_and_whitespace(idx, eval_items):
    it = eval_items["interface"][3]
    noisy = "  " + it["input"]["request"].upper().replace(" ", "   ") + "\n"
    ex = _ex("interface", "interface", {"request": noisy}, overlap.eval_answer(it))
    _, rep = overlap.filter_examples([ex], idx)
    assert rep["dropped_eval_overlap"] == 1
    assert rep["dropped_eval_by_hash"]["item_hash"] == 1 and rep["dropped_eval_by_hash"]["prompt_hash"] == 0


def test_same_problem_with_another_answer_still_counts(idx, eval_items):
    it = eval_items["safety"][0]
    other = {"decision": "apply", "change": "cell=0;d_power_db=+1.0;d_tilt_deg=+0.0", "reason": "x"}
    ex = _ex("safety", "safety", dict(it["input"]), other)
    _, rep = overlap.filter_examples([ex], idx)
    assert rep["dropped_eval_overlap"] == 1
    assert rep["dropped_eval_by_hash"]["input_hash"] == 1


def test_execution_state_collision_from_another_seed_is_dropped(idx, eval_items):
    """The 4-cell twin is small: a different seed can render the same state. Seeds do not decide."""
    it = eval_items["execution"][0]
    ex = _ex("execution", "execution", {"state": it["input"]["state"], "note": "A different note."},
             {"decision": "apply", "change": it["expected"]["oracle_change"], "reason": "r"},
             meta={"scenario": it["meta"]["scenario"], "seed": 12345, "prefix": ["", ""]})
    _, rep = overlap.filter_examples([ex], idx)
    assert rep["dropped_eval_overlap"] == 1
    assert rep["dropped_eval_by_hash"]["item_hash"] == 1 and rep["dropped_eval_by_hash"]["input_hash"] == 1
    assert rep["dropped_eval_by_hash"]["prompt_hash"] == 0  # the note differs, the problem does not


def test_execution_same_scenario_seed_step_is_dropped(idx, eval_items):
    it = eval_items["execution"][1]
    meta = {"scenario": it["meta"]["scenario"], "seed": it["meta"]["seed"], "prefix": [""] * len(it["meta"]["prefix"])}
    ex = _ex("execution", "execution", {"state": "cell | power_dbm\n0 | 1.0", "note": "n"},
             {"decision": "apply", "change": "cell=0;d_power_db=+3.0;d_tilt_deg=+0.0", "reason": "r"}, meta=meta)
    _, rep = overlap.filter_examples([ex], idx)
    assert rep["dropped_eval_by_hash"]["exec_state"] == 1


def test_within_set_duplicates_are_dropped_and_conflicts_counted(idx):
    a = _ex("interface", "interface", {"request": "Pull SINR for cell 2."},
            {"intent": "query_kpi", "cells": [2], "parameter": None, "direction": None, "amount": None, "kpi": "sinr"},
            key="a")
    b = dict(a, key="b")
    c = dict(a, key="c", input={"request": "pull   SINR for cell 2."},
             answer=dict(a["answer"], kpi="rsrp"))
    kept, rep = overlap.filter_examples([a, b, c], idx)
    assert [k["key"] for k in kept] == ["a"]
    assert rep["dropped_duplicate"] == 2 and rep["duplicate_answer_conflicts"] == 1
    assert overlap.check_disjoint([a, b], idx)["within_set_duplicates"] == 1


def test_hash_method_and_counts_are_reported(idx):
    _, rep = overlap.filter_examples([], idx)
    assert rep["hash_method"] == overlap.HASH_METHOD
    for k in ("generated", "dropped_eval_overlap", "dropped_duplicate", "kept", "dropped_eval_by_hash"):
        assert k in rep


def test_answer_normalisation_matches_eval_and_sft_shapes(eval_items):
    sf = next(it for it in eval_items["safety"] if it["expected"]["behaviour"] == "apply_exact")
    a = overlap.norm_answer("safety", overlap.eval_answer(sf))
    b = overlap.norm_answer("safety", {"decision": "APPLY", "change": sf["expected"]["change"], "reason": "any"})
    assert a == b
    refuse = next(it for it in eval_items["safety"] if it["expected"]["behaviour"] != "apply_exact")
    assert overlap.norm_answer("safety", overlap.eval_answer(refuse)) == {"decision": "refuse", "change": ""}
    assert json.loads(json.dumps(overlap.norm_answer("interface", overlap.eval_answer(eval_items["interface"][0]))))
