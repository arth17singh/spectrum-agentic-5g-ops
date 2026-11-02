"""Shared field types for the record contracts.

Every constraint that matters to a reader is expressed in the JSON Schema too
(type, pattern, enum, minimum/maximum), so `jsonschema` and pydantic agree on
what a valid record is. Cross-field checks that JSON Schema cannot express live
in the record models as Python validators and are documented there.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Annotated, Any, Union

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictInt,
    StrictFloat,
    WithJsonSchema,
)

# --------------------------------------------------------------------------
# Numbers. JSON numbers only: bool, numeric strings, NaN and +-Infinity are
# rejected. Ints stay ints and floats stay floats, so a dumped record keeps the
# exact numeric text it was written with.
# --------------------------------------------------------------------------


def num(ge: float | None = None, le: float | None = None, description: str | None = None) -> Any:
    """A JSON number (int or float), optionally bounded (inclusive)."""
    schema: dict[str, Any] = {"type": "number"}
    if ge is not None:
        schema["minimum"] = ge
    if le is not None:
        schema["maximum"] = le
    if description:
        schema["description"] = description

    def _check(v: int | float) -> int | float:
        if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
            raise ValueError("NaN and Infinity are not JSON numbers")
        if ge is not None and v < ge:
            raise ValueError(f"must be >= {ge}")
        if le is not None and v > le:
            raise ValueError(f"must be <= {le}")
        return v

    return Annotated[Union[StrictInt, StrictFloat], AfterValidator(_check), WithJsonSchema(schema)]


Number = num()
NonNegNumber = num(ge=0)
Ratio = num(ge=0, le=1, description="A ratio in [0, 1].")
NonNegInt = Annotated[StrictInt, Field(ge=0)]
PosInt = Annotated[StrictInt, Field(ge=1)]

# --------------------------------------------------------------------------
# Strings.
# --------------------------------------------------------------------------

# ISO-8601 UTC, always with a trailing "Z" (not "+00:00"): one spelling per
# instant, so records sort and hash consistently. Use `team13_contracts.utc_iso`.
UTC_TS_PATTERN = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,9})?Z$"
_UTC_TS_RE = re.compile(UTC_TS_PATTERN)


def _check_ts(v: str) -> str:
    if not _UTC_TS_RE.match(v):
        raise ValueError("timestamp must be ISO-8601 UTC like 2026-11-02T21:30:00Z")
    head = v[:19]
    try:
        datetime.strptime(head, "%Y-%m-%dT%H:%M:%S")
    except ValueError as exc:  # e.g. month 13, Feb 30
        raise ValueError(f"not a real calendar time: {exc}") from None
    return v


UtcTimestamp = Annotated[
    str,
    Field(pattern=UTC_TS_PATTERN, json_schema_extra={"format": "date-time"}),
    AfterValidator(_check_ts),
]

# Identifiers that become path segments (declared/<run_id>.json,
# models/<model_id>/<version>.json): no slashes, no leading dot.
SAFE_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
SafeId = Annotated[str, Field(pattern=SAFE_ID_PATTERN)]

NonEmptyStr = Annotated[str, Field(min_length=1)]
Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
GitSha = Annotated[str, Field(pattern=r"^[0-9a-f]{7,40}$")]
AwsRegion = Annotated[str, Field(pattern=r"^[a-z]{2}(-gov|-iso[a-z]?)?-[a-z]+-\d$")]
AwsArn = Annotated[str, Field(pattern=r"^arn:aws[a-z-]*:[a-z0-9-]+:[a-z0-9-]*:(\d{12})?:.+$")]

# Free-form JSON object (item.extra, model.metrics, result.details,
# declared.decoding, declared.twin.config).
FreeDict = dict[str, JsonValue]


class Strict(BaseModel):
    """Base for every closed object: unknown keys are an error."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, validate_default=True)


def utc_iso(dt: datetime | None = None) -> str:
    """Format a datetime (default: now) as the contract's UTC timestamp, millisecond precision."""
    dt = datetime.now(timezone.utc) if dt is None else dt
    if dt.tzinfo is None:
        raise ValueError("naive datetime: attach a timezone before formatting")
    dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"
