"""Interface and safety examples: labels the eval's own parser/graders and the SDK filter accept."""

from __future__ import annotations

import pytest

from t1.evals import canon as C
from t1.evals.graders import grade_interface, parse_ops
from t1.evals.parse import first_json_object
from t1.sft import overlap, templates
from t1.sft.build import items_dir


@pytest.fixture(scope="module")
def filter_check():
    from t1.evals.twin_access import filter_check as fc

    return fc


@pytest.fixture(scope="module")
def interface():
    return templates.interface_examples(400)


@pytest.fixture(scope="module")
def safety(filter_check):
    return templates.safety_examples(400, filter_check=filter_check)


def test_interface_labels_grade_and_cover_every_intent(interface):
    assert len({e["input"]["request"] for e in interface}) == len(interface) == 400
    intents = {e["answer"]["intent"] for e in interface}
    assert intents == {"adjust_power", "adjust_tilt", "query_kpi", "no_action", "out_of_scope"}
    for e in interface:
        text = templates.answer_text(e["answer"])
        assert first_json_object(text) == e["answer"]
        assert grade_interface({"expected": e["answer"]}, text)["passed"]
        assert list(e["answer"]) == ["intent", "cells", "parameter", "direction", "amount", "kpi"]
        assert e["answer"]["cells"] == sorted(e["answer"]["cells"])
        if e["answer"]["intent"].startswith("adjust_"):
            assert e["answer"]["amount"] > 0 and e["answer"]["direction"] in ("up", "down")


def test_interface_is_deterministic(interface):
    assert templates.interface_examples(400) == interface


def test_safety_labels_pass_the_sdk_filter(safety, filter_check):
    fams = {e["family"] for e in safety}
    assert fams == set(templates.SAFETY_SHARES)
    for e in safety:
        text = templates.answer_text(e["answer"])
        parsed, err = parse_ops(text)
        assert err is None and parsed["decision"] == e["answer"]["decision"]
        if e["answer"]["decision"] == "refuse":
            assert e["answer"]["change"] == ""
        else:
            assert parsed["change_text"] == C.canon(e["answer"]["change"])
            assert e["meta"]["check"]["verdict"] == "clean"
        if e["family"] in ("over_step", "over_abs", "alarm"):
            assert e["answer"]["decision"] == "refuse" and e["meta"]["check"]["sdk_filter_rules"]


def test_safety_applies_are_the_largest_share(safety):
    applies = sum(e["answer"]["decision"] == "apply" for e in safety)
    assert applies / len(safety) >= 0.5  # over-refusal on benign items is a declared target


def test_injected_instructions_are_never_followed(safety):
    for e in (e for e in safety if e["family"].startswith("injection")):
        fb = e["meta"]["forbidden"]
        key = "d_power_db" if fb["parameter"] == "power" else "d_tilt_deg"
        touched = [p for p in C.parse(e["answer"]["change"]) if p["cell"] == fb["cell"] and p[key] != 0] \
            if e["answer"]["change"] else []
        assert not touched


def test_no_request_sentence_is_an_eval_sentence(interface, safety):
    ev = set()
    for u in ("interface", "safety"):
        ev |= {overlap.norm_text(it["input"]["request"]) for it in overlap.read_jsonl(items_dir() / f"{u}.jsonl")}
    ours = {overlap.norm_text(e["input"]["request"]) for e in interface + safety}
    assert not ours & ev


def test_safety_counts_sum_to_n():
    for n in (7, 70, 1200):
        assert sum(templates.safety_counts(n).values()) == n
