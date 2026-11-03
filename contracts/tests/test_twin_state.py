"""Per-cell twin state on change records (Twin View) and ChangeTwin.preset. All optional and additive."""

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from team13_contracts import ChangeRecord, dump_record, parse_record

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
FULL = "valid-with-twin-state.json"
OLD = "valid-healthy-predictor.json"


def load(name: str) -> dict:
    return json.loads((FIXTURES / "change" / name).read_text())


def bad(raw: dict) -> None:
    with pytest.raises(ValidationError):
        parse_record(raw)


def test_full_twin_state_validates_and_round_trips():
    raw = load(FULL)
    m = parse_record(raw)
    assert isinstance(m, ChangeRecord)
    ts = m.twin_state
    assert ts is not None and ts.baseline is not None
    assert [c.cell for c in ts.before.per_cell] == [0, 1, 2, 3]
    assert ts.after.step == ts.before.step + 1
    assert m.twin.preset == "official"
    assert dump_record(parse_record(dump_record(m))) == dump_record(m)


def test_old_record_without_twin_state_or_preset_still_validates():
    raw = load(OLD)
    assert "twin_state" not in raw and "preset" not in raw["twin"]
    m = parse_record(raw)
    assert m.twin_state is None and m.twin.preset is None


def test_baseline_is_optional():
    raw = load(FULL)
    raw["twin_state"]["baseline"] = None
    parse_record(raw)
    del raw["twin_state"]["baseline"]
    parse_record(raw)


def test_cell_optional_fields_may_be_null_or_absent():
    raw = load(FULL)
    c = raw["twin_state"]["before"]["per_cell"][0]
    for k in ("coverage_share", "sla_share", "sinr_mean_db", "sinr_p10_db", "rsrp_mean_dbm", "rsrp_p10_dbm"):
        c[k] = None
    del c["alarm"], c["asleep"]
    parse_record(raw)


@pytest.mark.parametrize("field", ["coverage_share", "sla_share"])
@pytest.mark.parametrize("value", [1.2, -0.1])
def test_share_outside_unit_interval_rejected(field, value):
    raw = load(FULL)
    raw["twin_state"]["after"]["per_cell"][1][field] = value
    bad(raw)


@pytest.mark.parametrize("field", ["coverage_ratio", "sla_ratio"])
def test_snapshot_ratio_above_one_rejected(field):
    raw = load(FULL)
    raw["twin_state"]["before"][field] = 1.01
    bad(raw)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s["before"]["per_cell"].clear(),  # 0 cells
        lambda s: s["before"]["per_cell"].extend(
            [dict(s["before"]["per_cell"][0], cell=i) for i in range(4, 65)]
        ),  # 65 cells
        lambda s: s["after"]["per_cell"].append(copy.deepcopy(s["after"]["per_cell"][0])),  # duplicate cell
        lambda s: s["after"]["per_cell"][0].update(ue_count=-1),
        lambda s: s["after"]["per_cell"][0].update(ue_count=3.5),
        lambda s: s["after"]["per_cell"][0].update(alarm="MAJOR"),  # alarm is a bool in ran_rl
        lambda s: s["after"]["per_cell"][0].update(rsrp_p10_dbm=float("-inf")),  # non-finite goes as null
        lambda s: s["after"]["per_cell"][0].update(power_dbm=None),  # required
        lambda s: s["after"]["per_cell"][0].update(unknown=1),  # closed object
        lambda s: s.pop("after"),  # after is required
        lambda s: s["before"].update(step=-1),
    ],
)
def test_malformed_twin_state_rejected(mutate):
    raw = load(FULL)
    mutate(raw["twin_state"])
    bad(raw)


def test_64_cells_allowed():
    raw = load(FULL)
    pc = raw["twin_state"]["before"]["per_cell"]
    pc.extend(dict(pc[0], cell=i) for i in range(4, 64))
    parse_record(raw)


@pytest.mark.parametrize("preset", ["", 7])
def test_preset_must_be_non_empty_string(preset):
    raw = load(FULL)
    raw["twin"]["preset"] = preset
    bad(raw)


def test_aws_time_source_lambda_clock():
    raw = load(OLD)
    raw["aws_time"] = "2026-11-02T20:20:31.000Z"
    raw["aws_time_source"] = "lambda:clock"
    parse_record(raw)
