"""canonical_json / sha256_of / parse_record / dump_record behaviour."""

import hashlib
import json
from datetime import datetime, timedelta, timezone

import pytest

from pathlib import Path

CONTRACTS = Path(__file__).resolve().parents[1]  # no `from conftest import`: works with --import-mode=importlib
FIXTURES = CONTRACTS / "fixtures"
from team13_contracts import (
    UnknownKind,
    canonical_json,
    dump_record,
    is_valid,
    parse_record,
    sha256_of,
    utc_iso,
)


def test_canonical_json_sorted_compact_utf8():
    obj = {"b": 1, "a": {"y": [1, 2.5, None, True], "x": "café ✓"}}
    text = canonical_json(obj)
    assert text == '{"a":{"x":"café ✓","y":[1,2.5,null,true]},"b":1}'


def test_canonical_json_independent_of_key_order():
    a = {"z": 1, "a": 2, "m": {"q": 1, "b": 2}}
    b = {"m": {"b": 2, "q": 1}, "a": 2, "z": 1}
    assert canonical_json(a) == canonical_json(b)
    assert sha256_of(a) == sha256_of(b)


def test_canonical_json_stable_across_calls_and_golden():
    obj = {"run_id": "t3-run-001", "value": 0.85, "n": 24, "ok": False}
    assert canonical_json(obj) == canonical_json(json.loads(canonical_json(obj)))
    assert canonical_json(obj) == '{"n":24,"ok":false,"run_id":"t3-run-001","value":0.85}'
    assert sha256_of(obj) == hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def test_canonical_json_keeps_int_and_float_distinct():
    assert canonical_json({"v": 1}) == '{"v":1}'
    assert canonical_json({"v": 1.0}) == '{"v":1.0}'


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_canonical_json_refuses_nan_and_infinity(bad):
    with pytest.raises(ValueError):
        canonical_json({"v": bad})


def test_sha256_of_model_equals_sha256_of_dump():
    raw = json.loads((FIXTURES / "change" / "valid-healthy-predictor.json").read_text())
    m = parse_record(raw)
    assert sha256_of(m) == sha256_of(dump_record(m))


def test_dump_record_is_json_safe_and_complete():
    raw = json.loads((FIXTURES / "item" / "valid-t2-holdout.json").read_text())
    del raw["aws_time"], raw["aws_time_source"], raw["extra"]
    d = dump_record(parse_record(raw))
    assert d["aws_time"] is None and d["aws_time_source"] is None and d["extra"] == {}
    assert d["schema_version"] == "1" and d["team"] == "team13" and d["kind"] == "item"
    json.dumps(d, allow_nan=False)


def test_decision_extra_keys_survive_dump():
    raw = json.loads((FIXTURES / "decision" / "valid-allow-with-extra-key.json").read_text())
    d = dump_record(parse_record(raw))
    assert d["gateway_target"] == "team13-ran-tools"


def test_parse_record_unknown_and_missing_kind():
    with pytest.raises(UnknownKind):
        parse_record({"kind": "trace"})
    with pytest.raises(UnknownKind):
        parse_record({})
    with pytest.raises(TypeError):
        parse_record([1, 2])  # type: ignore[arg-type]
    assert is_valid({"kind": "nope"}) is False


def test_utc_iso_format_round_trips():
    ts = utc_iso(datetime(2026, 11, 2, 21, 30, 5, 123456, tzinfo=timezone(timedelta(hours=-8))))
    assert ts == "2026-11-03T05:30:05.123Z"
    with pytest.raises(ValueError):
        utc_iso(datetime(2026, 11, 2))
    assert utc_iso().endswith("Z")
