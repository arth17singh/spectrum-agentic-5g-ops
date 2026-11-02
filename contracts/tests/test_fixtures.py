"""Every valid fixture passes pydantic and its JSON Schema; every invalid one fails both."""

import json

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from conftest import FIXTURES, SCHEMA_DIR
from team13_contracts import KINDS, UnknownKind, canonical_json, dump_record, parse_record

REASONS = json.loads((FIXTURES / "invalid-reasons.json").read_text())


def _validator(kind: str) -> Draft202012Validator:
    schema = json.loads((SCHEMA_DIR / f"{kind}.schema.json").read_text())
    return Draft202012Validator(schema)


def _fixtures(prefix: str):
    out = []
    for kind in KINDS:
        for p in sorted((FIXTURES / kind).glob(f"{prefix}-*.json")):
            out.append(pytest.param(kind, p, id=f"{kind}/{p.name}"))
    return out


VALID = _fixtures("valid")
INVALID = _fixtures("invalid")


def test_every_kind_has_enough_fixtures():
    for kind in KINDS:
        assert len(list((FIXTURES / kind).glob("valid-*.json"))) >= 2, kind
        assert len(list((FIXTURES / kind).glob("invalid-*.json"))) >= 1, kind
    assert {p.parent.name for p in FIXTURES.glob("*/*.json")} == set(KINDS)


@pytest.mark.parametrize("kind", KINDS)
def test_schema_is_valid_draft_2020_12(kind):
    schema = json.loads((SCHEMA_DIR / f"{kind}.schema.json").read_text())
    Draft202012Validator.check_schema(schema)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"


@pytest.mark.parametrize("kind,path", VALID)
def test_valid_fixture_passes_pydantic_and_schema(kind, path):
    raw = json.loads(path.read_text())
    assert raw["kind"] == kind
    assert raw["synthetic"] is True, "fixtures must be marked synthetic"
    model = parse_record(raw)
    assert model.kind == kind
    errors = sorted(_validator(kind).iter_errors(raw), key=str)
    assert not errors, [e.message for e in errors]


@pytest.mark.parametrize("kind,path", VALID)
def test_valid_fixture_round_trips(kind, path):
    raw = json.loads(path.read_text())
    once = dump_record(parse_record(raw))
    twice = dump_record(parse_record(once))
    assert canonical_json(once) == canonical_json(twice)
    # The dumped form still satisfies the published schema.
    assert not list(_validator(kind).iter_errors(once))
    # Numbers stay numbers with their int/float type.
    assert json.loads(canonical_json(once)) == once


@pytest.mark.parametrize("kind,path", INVALID)
def test_invalid_fixture_fails_pydantic_and_schema(kind, path):
    rel = f"{kind}/{path.name}"
    assert rel in REASONS, f"document why {rel} is invalid in fixtures/invalid-reasons.json"
    raw = json.loads(path.read_text())
    with pytest.raises((ValidationError, UnknownKind)):
        parse_record(raw)
    assert list(_validator(kind).iter_errors(raw)), f"{rel} passes the JSON Schema: {REASONS[rel]}"


def test_reasons_list_no_missing_files():
    for rel in REASONS:
        assert (FIXTURES / rel).is_file(), rel
