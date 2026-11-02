"""The canonical change string (amendment 1, A2) and the amended change record (A5, A14, A15)."""

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from team13_contracts import parse_record
from team13_contracts.change import approval_key, canon, parse

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


# --------------------------------------------------------------------------
# canon / parse
# --------------------------------------------------------------------------


def test_single_cell_format():
    assert canon([{"cell": 0, "d_power_db": 1, "d_tilt_deg": -0.5}]) == "cell=0;d_power_db=+1.0;d_tilt_deg=-0.5"


@pytest.mark.parametrize(
    "value,text",
    [
        (0, "+0.0"),
        (0.0, "+0.0"),
        (-0.0, "+0.0"),
        (-0.04, "+0.0"),  # rounds to -0.0, written +0.0
        (0.04, "+0.0"),
        (1.25, "+1.2"),  # round-half-even on an exact binary half
        (-1.75, "-1.8"),
        (2.96, "+3.0"),
        (12, "+12.0"),
        (-3, "-3.0"),
    ],
)
def test_signs_rounding_and_zero(value, text):
    out = canon([{"cell": 1, "d_power_db": value, "d_tilt_deg": value}])
    assert out == f"cell=1;d_power_db={text};d_tilt_deg={text}"


def test_multi_cell_sorted_and_joined():
    out = canon(
        [
            {"cell": 3, "d_power_db": -1, "d_tilt_deg": 0.5},
            {"cell": 0, "d_power_db": 2, "d_tilt_deg": 0},
            {"cell": 10, "d_power_db": 0.1, "d_tilt_deg": -0.1},
        ]
    )
    assert out == (
        "cell=0;d_power_db=+2.0;d_tilt_deg=+0.0"
        "|cell=3;d_power_db=-1.0;d_tilt_deg=+0.5"
        "|cell=10;d_power_db=+0.1;d_tilt_deg=-0.1"
    )


def test_round_trip_is_stable():
    raw = [{"cell": 2, "d_power_db": -1.234, "d_tilt_deg": 0.06}, {"cell": 1, "d_power_db": -0.0, "d_tilt_deg": 3}]
    text = canon(raw)
    parsed = parse(text)
    assert parsed == [
        {"cell": 1, "d_power_db": 0.0, "d_tilt_deg": 3.0},
        {"cell": 2, "d_power_db": -1.2, "d_tilt_deg": 0.1},
    ]
    assert canon(parsed) == text
    assert canon(text) == text
    assert parse(canon(parse(text))) == parsed


def test_canon_accepts_models():
    from team13_contracts import CellChange

    assert canon([CellChange(cell=4, d_power_db=1.5, d_tilt_deg=-2)]) == "cell=4;d_power_db=+1.5;d_tilt_deg=-2.0"


@pytest.mark.parametrize(
    "text",
    [
        "",
        "cell=0;d_power_db=+1.0",  # missing tilt
        "cell=0;d_power_db=1.0;d_tilt_deg=+0.0",  # no sign
        "cell=0;d_power_db=+1;d_tilt_deg=+0.0",  # no decimal
        "cell=0;d_power_db=+1.00;d_tilt_deg=+0.0",  # two decimals
        "cell=0;d_power_db=-0.0;d_tilt_deg=+0.0",  # zero must be +0.0
        "cell=0;d_power_db=+01.0;d_tilt_deg=+0.0",  # leading zero
        "cell=01;d_power_db=+1.0;d_tilt_deg=+0.0",  # leading zero cell
        "cell=-1;d_power_db=+1.0;d_tilt_deg=+0.0",  # negative cell
        "cell=1;d_tilt_deg=+0.0;d_power_db=+1.0",  # wrong order of fields
        "cell=1; d_power_db=+1.0;d_tilt_deg=+0.0",  # whitespace
        "cell=1;d_power_db=+1.0;d_tilt_deg=+0.0;",  # trailing separator
        "cell=1;d_power_db=+1.0;d_tilt_deg=+0.0|",  # empty item
        "cell=2;d_power_db=+1.0;d_tilt_deg=+0.0|cell=1;d_power_db=+1.0;d_tilt_deg=+0.0",  # unsorted
        "cell=1;d_power_db=+1.0;d_tilt_deg=+0.0|cell=1;d_power_db=+2.0;d_tilt_deg=+0.0",  # duplicate
        "cell=1,d_power_db=+1.0,d_tilt_deg=+0.0",  # wrong separator
        "cell=1;d_power_db=+nan;d_tilt_deg=+0.0",
        "CELL=1;d_power_db=+1.0;d_tilt_deg=+0.0",
    ],
)
def test_parse_rejects_malformed(text):
    with pytest.raises(ValueError):
        parse(text)
    with pytest.raises(ValueError):
        canon(text)


@pytest.mark.parametrize(
    "items",
    [
        [],
        [{"cell": 0, "d_power_db": 1}],
        [{"cell": -1, "d_power_db": 1, "d_tilt_deg": 0}],
        [{"cell": 1.0, "d_power_db": 1, "d_tilt_deg": 0}],
        [{"cell": True, "d_power_db": 1, "d_tilt_deg": 0}],
        [{"cell": 0, "d_power_db": "1", "d_tilt_deg": 0}],
        [{"cell": 0, "d_power_db": float("nan"), "d_tilt_deg": 0}],
        [{"cell": 0, "d_power_db": 1, "d_tilt_deg": float("inf")}],
        [{"cell": 0, "d_power_db": 1, "d_tilt_deg": 0, "extra": 1}],
        [{"cell": 0, "d_power_db": 1, "d_tilt_deg": 0}, {"cell": 0, "d_power_db": 2, "d_tilt_deg": 0}],
    ],
)
def test_canon_rejects_bad_lists(items):
    with pytest.raises(ValueError):
        canon(items)


def test_approval_key_matches_kit_formula():
    import hashlib

    text = "cell=0;d_power_db=+1.0;d_tilt_deg=+0.0"
    want = "chg-" + hashlib.sha256(json.dumps(["case-1", "step-0", text]).encode()).hexdigest()[:12]
    assert approval_key("case-1", "step-0", text) == want


# --------------------------------------------------------------------------
# The amended change record
# --------------------------------------------------------------------------


def load(name: str) -> dict:
    return json.loads((FIXTURES / "change" / name).read_text())


def bad(raw: dict) -> None:
    with pytest.raises(ValidationError):
        parse_record(raw)


SCORED = "valid-healthy-predictor.json"
PROBE = "valid-probe-no-model.json"
CLIPPED = "valid-clipped-unscored.json"


def test_change_text_must_be_canon_of_change():
    raw = load(SCORED)
    raw["change_text"] = raw["change_text"].replace("d_tilt_deg=+1.0", "d_tilt_deg=+2.0")
    bad(raw)


def test_change_text_must_be_sorted():
    raw = load(SCORED)
    a, b = raw["change_text"].split("|")
    raw["change_text"] = f"{b}|{a}"  # still matches the schema pattern
    bad(raw)


def test_change_list_must_hold_rounded_values():
    raw = load(SCORED)
    raw["change"][0]["d_power_db"] = -2.04  # canon() still gives -2.0, but the list is not parse(change_text)
    bad(raw)


def test_measured_delta_is_measured_minus_baseline():
    raw = load(SCORED)
    raw["measured_delta"]["sinr_mean_db"] += 0.01
    bad(raw)


def test_error_is_on_deltas_and_exact():
    raw = load(SCORED)
    raw["error"]["coverage_ratio"] = round(raw["error"]["coverage_ratio"] + 0.001, 6)
    bad(raw)


def test_error_requires_predicted_delta():
    raw = load(SCORED)
    raw["probe"], raw["scored"], raw["unscored_reason"] = True, False, "probe"
    raw["predicted_delta"] = None
    bad(raw)


def test_unscored_requires_reason_and_scored_forbids_it():
    raw = load(CLIPPED)
    raw["unscored_reason"] = None
    bad(raw)
    raw = load(SCORED)
    raw["unscored_reason"] = "clipped"
    bad(raw)


@pytest.mark.parametrize("field", ["predictor", "predicted_delta", "error", "direction_agree"])
def test_non_probe_needs_prediction_fields(field):
    raw = load(SCORED)
    raw[field] = None
    bad(raw)
    raw = load(CLIPPED)  # non-probe, even when unscored
    raw[field] = None
    bad(raw)


def test_probe_may_omit_prediction_fields():
    raw = load(PROBE)
    for f in ("predictor", "predicted_delta", "error", "direction_agree"):
        raw.pop(f, None)
    m = parse_record(raw)
    assert m.probe and not m.scored


def test_scored_probe_needs_prediction_fields():
    raw = load(PROBE)
    raw["scored"], raw["unscored_reason"] = True, None
    bad(raw)


def test_fixture_change_ids_follow_kit_formula():
    for p in sorted((FIXTURES / "change").glob("valid-*.json")):
        raw = json.loads(p.read_text())
        assert raw["change_id"] == approval_key(raw["case_id"], raw["item_id"], raw["change_text"]), p.name


def test_delta_ratio_bounds():
    raw = copy.deepcopy(load(SCORED))
    raw["predicted_delta"]["coverage_ratio"] = 1.5
    bad(raw)
