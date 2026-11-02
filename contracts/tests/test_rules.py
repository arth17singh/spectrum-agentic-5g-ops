"""Field rules, including the Python-only cross-field checks JSON Schema cannot express."""

import copy
import json

import pytest
from pydantic import ValidationError

from pathlib import Path

CONTRACTS = Path(__file__).resolve().parents[1]  # no `from conftest import`: works with --import-mode=importlib
FIXTURES = CONTRACTS / "fixtures"
from team13_contracts import parse_record


def load(rel: str) -> dict:
    return json.loads((FIXTURES / rel).read_text())


def bad(raw: dict) -> None:
    with pytest.raises(ValidationError):
        parse_record(raw)


CHANGE = "change/valid-healthy-predictor.json"
RESULT = "result/valid-pass.json"


@pytest.mark.parametrize("value", [True, "0.5", None, float("nan"), float("inf")])
def test_numbers_are_json_numbers_only(value):
    raw = load(RESULT)
    raw["value"] = value
    bad(raw)


@pytest.mark.parametrize(
    "ts",
    [
        "2026-11-02T20:40:00+00:00",  # must be Z
        "2026-11-02T20:40:00",  # no zone
        "2026-11-02T12:40:00-08:00",  # not UTC
        "2026-02-30T20:40:00Z",  # not a real date
        "2026-11-02T25:00:00Z",  # not a real hour
        "1730580000",  # epoch
    ],
)
def test_timestamps_are_iso_utc_z(ts):
    raw = load(RESULT)
    raw["checked_at"] = ts
    bad(raw)


@pytest.mark.parametrize("ts", ["2026-11-02T20:40:00Z", "2026-11-02T20:40:00.5Z", "2026-11-02T20:40:00.123456Z"])
def test_timestamp_precisions_accepted(ts):
    raw = load(RESULT)
    raw["checked_at"] = ts
    parse_record(raw)


def test_aws_time_needs_source():
    raw = load(RESULT)
    raw["aws_time"] = "2026-11-02T20:40:00Z"
    raw["aws_time_source"] = None
    bad(raw)


@pytest.mark.parametrize(
    "rel",
    [
        "declared/valid-t3-deny.json",
        "item/valid-deny-case.json",
        CHANGE,
        "control/valid-pause-by-trust-gate.json",
        "job/valid-sagemaker-training.json",
        "model/valid-ppo-approved.json",
        RESULT,
    ],
)
def test_closed_kinds_reject_unknown_keys(rel):
    raw = load(rel)
    raw["surprise"] = 1
    bad(raw)


def test_decision_v0_allows_unknown_keys_but_checks_known_ones():
    raw = load("decision/valid-deny-apply.json")
    raw["span_attributes"] = {"gen_ai.system": "aws.bedrock"}
    parse_record(raw)
    raw["source"] = "stdout"
    bad(raw)


def test_free_form_dicts_accept_any_json():
    raw = load("item/valid-deny-case.json")
    raw["extra"] = {"nested": {"list": [1, "a", None, {"k": 2.5}]}, "flag": False}
    parse_record(raw)


def test_change_error_must_equal_abs_difference():
    raw = load(CHANGE)
    raw["error"]["sinr_p10_db"] = 0.9
    bad(raw)


def test_change_cells_unique_and_items_closed():
    raw = load(CHANGE)
    raw["change"].append(copy.deepcopy(raw["change"][0]))
    bad(raw)
    raw = load(CHANGE)
    raw["change"] = []
    bad(raw)
    raw = load(CHANGE)
    raw["change"][0]["cell"] = -1
    bad(raw)
    raw = load(CHANGE)
    raw["change"][0]["cell"] = 1.0
    bad(raw)


@pytest.mark.parametrize("kind", ["unknown-twin", "Maveric", ""])
def test_twin_kind_enum(kind):
    raw = load(CHANGE)
    raw["twin"]["kind"] = kind
    bad(raw)


@pytest.mark.parametrize("op", [">=", "<=", "==", ">", "<"])
def test_result_passed_must_match_op(op):
    raw = load(RESULT)
    raw["target"]["op"] = op
    raw["value"] = 1
    raw["target"]["value"] = 1
    raw["passed"] = op in (">=", "<=", "==")
    parse_record(raw)
    raw["passed"] = not raw["passed"]
    bad(raw)


def test_result_metric_must_match_target():
    raw = load(RESULT)
    raw["metric"] = "something_else"
    bad(raw)


def test_job_end_not_before_start():
    raw = load("job/valid-sagemaker-training.json")
    raw["aws_end"] = "2026-11-02T16:00:00.000Z"
    bad(raw)


@pytest.mark.parametrize("by", ["person:op-blue", "trust-gate", "person:a.b_c-1"])
def test_control_by_ok(by):
    raw = load("control/valid-resume-by-person.json")
    raw["by"] = by
    parse_record(raw)


def test_validate_record_is_exported_and_raises():
    from team13_contracts import ResultRecord, validate_record

    assert isinstance(validate_record(load(RESULT)), ResultRecord)
    raw = load(RESULT)
    raw["value"] = "1"
    with pytest.raises(ValidationError):
        validate_record(raw)


def test_declared_n_cases_and_subset_optional():
    raw = load("declared/valid-t3-trust-pause.json")
    assert "n_cases" not in raw and "subset" not in raw
    m = parse_record(raw)
    assert m.n_cases is None and m.subset is None
    raw["n_cases"] = 0
    parse_record(raw)
    for bad_n in (-1, 1.0, True, "3"):
        raw["n_cases"] = bad_n
        bad(raw)


def test_one_frozen_suite_backs_two_units():
    deny = parse_record(load("declared/valid-t3-deny.json"))
    allow = parse_record(load("declared/valid-t3-allow-same-suite.json"))
    assert deny.test_set_sha256 == allow.test_set_sha256
    assert (deny.subset, allow.subset) == ("expect=DENY", "expect=ALLOW")
    assert deny.n_items != allow.n_items and deny.unit != allow.unit


@pytest.mark.parametrize("run_id", ["../x", "a/b", ".hidden", "", "x" * 200])
def test_run_id_is_a_safe_path_segment(run_id):
    raw = load(RESULT)
    raw["run_id"] = run_id
    bad(raw)
