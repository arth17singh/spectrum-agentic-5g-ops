"""t1.knowledge-bedrock-gpt-oss-20b-2 died writing its items: a telemath reply carried the answer as the
string "inf" (or "Infinity"/"1e999"), float() turned it into inf, and the strict contract rejected the
item record (extra.detail.answer must be a finite number), so the run ended without a result.
Now a non-finite numeric answer is a parse failure, and item details are made strict JSON before writing."""

from __future__ import annotations

import json
import math

import pytest

from t1.evals import freeze
from t1.evals import run as R
from t1.evals.clients import Completion
from t1.evals.graders import grade_interface, grade_knowledge


def _numeric():
    return next(i for i in freeze.load_items("knowledge") if i["expected"]["type"] == "numeric")


@pytest.mark.parametrize("ans", ["inf", "Infinity", "-inf", "nan", "1e999"])
def test_nonfinite_numeric_answer_is_a_parse_failure(ans):
    g = grade_knowledge(_numeric(), json.dumps({"answer": ans}))
    assert g["parse_ok"] is False and g["passed"] is False and g["score"] == 0.0
    json.dumps(g, allow_nan=False)


def test_json_infinity_literal_too():
    g = grade_knowledge(_numeric(), '{"answer": Infinity}')
    assert g["parse_ok"] is False


def test_interface_amount_inf_is_not_a_match():
    it = next(i for i in freeze.load_items("interface") if i["expected"]["amount"] is not None)
    bad = dict(it["expected"], amount="inf")
    assert grade_interface(it, json.dumps(bad))["passed"] is False


def test_json_safe():
    assert R.json_safe({"a": [float("inf"), float("-inf"), float("nan"), 1.5], "b": {"c": float("nan")}}) == \
        {"a": ["inf", "-inf", "nan", 1.5], "b": {"c": "nan"}}


class _Odd:
    """A client whose replies include the failing shapes: a string "inf" answer and inf in its meta."""
    backend, needs_item, model_id = "stub", False, "fake:base"

    def decoding(self, max_tokens):
        return {"temperature": 0.0, "max_tokens": max_tokens, "transport": "stub"}

    def generate(self, system, user, max_tokens, item=None):
        return Completion(text='{"answer": "inf"}', latency_ms=5.0, tokens_in=10, tokens_out=3,
                          meta={"odd": math.inf})


def test_a_run_with_nonfinite_answers_still_writes_valid_records(tmp_path):
    from team13_contracts import validate_record
    from tools import evlib

    a = R.parser().parse_args(["--evidence-dir", str(tmp_path), "--client", "fake", "--unit", "knowledge",
                               "--group", "telemath", "--limit", "5", "--run-id", "t1.kn-odd", "--synthetic",
                               ])
    r = R.run(a, client=_Odd())
    store = evlib.LocalStore(tmp_path)
    items = evlib.read_records("t1.kn-odd", "item", store=store)
    assert len(items) == 5 and r["details"]["parse_fail_count"] == 5
    for rec in [evlib.read_declared("t1.kn-odd", store=store), evlib.read_result("t1.kn-odd", store=store), *items]:
        validate_record(rec)
        evlib.validate(rec)
    assert items[0]["extra"]["call"]["odd"] == "inf"
