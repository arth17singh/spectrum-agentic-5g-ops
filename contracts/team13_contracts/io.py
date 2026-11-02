"""Parse, dump and hash records.

- `parse_record(d)` validates a dict and returns the model for its `kind`.
- `dump_record(m)` returns a JSON-safe dict: every field, nulls included, numbers as
  JSON numbers (ints stay ints). The canonical form of a stored record is
  `dump_record(parse_record(raw))`.
- `canonical_json(obj)` is sorted keys, no whitespace, UTF-8 (no \\u escapes), and
  refuses NaN/Infinity. It is the Python reference only; the dashboard ledger's
  `canonical.ts` uses a different, float-as-string rule and is not compatible.
- `sha256_of(obj)` is the hex SHA-256 of `canonical_json(obj)` encoded as UTF-8.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from pydantic import BaseModel, ValidationError

from .records import RECORD_MODELS, Envelope


class UnknownKind(ValueError):
    pass


def parse_record(d: Mapping[str, Any]) -> Envelope:
    """Validate one record dict and return its model. Raises UnknownKind or pydantic.ValidationError."""
    if not isinstance(d, Mapping):
        raise TypeError(f"a record must be a JSON object, got {type(d).__name__}")
    kind = d.get("kind")
    model = RECORD_MODELS.get(kind) if isinstance(kind, str) else None
    if model is None:
        raise UnknownKind(f"unknown record kind {kind!r}; expected one of {sorted(RECORD_MODELS)}")
    return model.model_validate(dict(d))


def dump_record(m: Envelope) -> dict[str, Any]:
    """JSON-safe dict of a record model (all fields, nulls kept)."""
    if not isinstance(m, Envelope):
        raise TypeError(f"expected a record model, got {type(m).__name__}")
    return m.model_dump(mode="json")


def canonical_json(obj: Any) -> str:
    """Canonical JSON text: sorted keys, no whitespace, UTF-8, no NaN/Infinity."""
    if isinstance(obj, BaseModel):
        obj = obj.model_dump(mode="json")
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_of(obj: Any) -> str:
    """Hex SHA-256 of canonical_json(obj)."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def validate_record(rec: Mapping[str, Any]) -> Envelope:
    """Validate one record dict; return the parsed model or raise (UnknownKind, ValidationError, TypeError).

    Alias of parse_record, under the name the evidence library calls.
    """
    return parse_record(rec)


def is_valid(d: Mapping[str, Any]) -> bool:
    try:
        parse_record(d)
    except (UnknownKind, ValidationError, TypeError):
        return False
    return True
