"""Knowledge parser/grader v3: a plain-text answer at the START of the reply counts when there is no
JSON answer (the base OTel model replies "B" or "B. <choice>"). Samples are real base-run outputs
(t1.knowledge-otel-base-1). Other units keep t1-parse-v1 / t1-graders-v2."""

from __future__ import annotations

import json

import pytest

from t1.evals import freeze
from t1.evals import run as R
from t1.evals.graders import grade_knowledge, grader_version
from t1.evals.parse import PARSER_VERSION, leading, parser_version


def _kn(t):
    return next(i for i in freeze.load_items("knowledge") if i["expected"]["type"] == t)


@pytest.mark.parametrize("text,want", [
    ("A. Immutable and auto-executable", "A"), ("A", "A"), ("(C)", "C"), ("Answer: D", "D"), ("B\nbecause", "B"),
    ("<think>x</think>\nE.", "E"),
])
def test_leading_letter(text, want):
    assert leading(text, "mcq") == want


@pytest.mark.parametrize("text", ["A network operator would", "As per 3GPP", "I do not have enough information",
                                  "The answer is B", "e.g. B", "a. lowercase", ""])
def test_leading_letter_rejects_prose(text):
    assert leading(text, "mcq") is None


def test_leading_rca_and_label():
    assert leading("C4: Non-colocated co-frequency neighboring cells", "rca") == "C4"
    assert leading("CT1", "label") == "CT1" and leading("RAN_AH1", "label") == "RAN_AH1"
    assert leading('"SA2"', "label") == "SA2" and leading("Classified as SA2", "label") is None


def test_mcq_plain_answers_score():
    it = _kn("mcq")
    a = it["expected"]["answer"]
    other = "B" if a == "A" else "A"
    g = grade_knowledge(it, f"{a}. some choice text")
    assert g["passed"] and g["parse_ok"] and g["detail"]["answer_form"] == "plain"
    assert grade_knowledge(it, json.dumps({"answer": a}))["detail"]["answer_form"] == "json"
    assert grade_knowledge(it, f"{other}")["passed"] is False
    # a JSON answer wins over a leading letter; prose without a leading answer is still a parse failure
    assert grade_knowledge(it, f'{other}. {{"answer": "{a}"}}')["passed"] is True
    assert grade_knowledge(it, "I do not have enough information to answer.")["parse_ok"] is False


def test_rca_answer_with_explanation():
    it = _kn("rca")
    a = it["expected"]["answer"]
    assert grade_knowledge(it, json.dumps({"answer": f"{a}: the reason"}))["passed"] is True
    assert grade_knowledge(it, f"{a}: the reason")["passed"] is True


def test_numeric_unchanged_needs_json():
    it = _kn("numeric")
    assert grade_knowledge(it, f"The value is {it['expected']['answer']}")["parse_ok"] is False


def test_versions_change_for_knowledge_only():
    assert parser_version("knowledge") != PARSER_VERSION and grader_version("knowledge") == "t1-graders-v3"
    for u in ("interface", "execution", "safety", "price_latency"):
        assert parser_version(u) == PARSER_VERSION and grader_version(u) == "t1-graders-v2"
    assert freeze.verify() == []


def test_module_declaration_kept_and_used_for_the_baseline_check(tmp_path):
    base = R.run(R.parser().parse_args(["--evidence-dir", str(tmp_path), "--client", "fake", "--unit", "interface",
                                        "--limit", "4", "--run-id", "t1.if-b"]))
    mod = json.loads((tmp_path / "outputs" / base["run_id"] / "module-declaration.json").read_text())
    assert mod["decoding"]["temperature"] == 0.0 and mod["run_id"] == "t1.if-b"
    # simulate the launcher's stored declaration: decoding holds only the secondary checks
    d = tmp_path / "declared" / "t1.if-b.json"
    decl = json.loads(d.read_text())
    decl["decoding"] = {"secondary_checks": []}
    d.write_text(json.dumps(decl))
    mod["decoding"]["temperature"] = 0.7  # the base actually ran with another temperature
    (tmp_path / "outputs" / base["run_id"] / "module-declaration.json").write_text(json.dumps(mod))
    with pytest.raises(SystemExit, match="temperature"):
        R.run(R.parser().parse_args(["--evidence-dir", str(tmp_path), "--client", "fake", "--unit", "interface",
                                     "--limit", "4", "--run-id", "t1.if-t", "--baseline-run-id", "t1.if-b"]))
