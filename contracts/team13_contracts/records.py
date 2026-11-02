"""The eight evidence record kinds and their shared envelope (schema_version "1").

Field list: BUILD_CONTRACT.md, "Record contracts". Every record is one JSON object
with the envelope fields plus its kind's fields. Closed objects reject unknown keys,
except `decision` (v0: extra keys allowed until a real AgentCore span is captured)
and the free-form dicts (item.extra, model.metrics, result.details,
declared.decoding, declared.twin.config).

Python-only cross-field checks (JSON Schema cannot express them; noted per model):
- envelope: aws_time set => aws_time_source set.
- change: error.<key> == |predicted.<key> - measured.<key>| (abs tol 1e-6);
  cells in `change` are unique.
- result: target.metric == metric; passed == (value <op> target.value).
- job: aws_end >= aws_start when both are set.
"""

from __future__ import annotations

import operator
from typing import Any, ClassVar, Literal, Optional

from pydantic import ConfigDict, Field, StrictBool, model_validator

from .types import (
    AwsArn,
    AwsRegion,
    FreeDict,
    GitSha,
    NonEmptyStr,
    NonNegInt,
    NonNegNumber,
    Number,
    PosInt,
    Ratio,
    SafeId,
    Sha256Hex,
    Strict,
    UtcTimestamp,
)

SCHEMA_VERSION = "1"
TEAM = "team13"

TwinKind = Literal["maveric-inprocess", "maveric-hosted", "viavi", "none"]
Op = Literal[">=", "<=", "==", ">", "<"]

OPS = {
    ">=": operator.ge,
    "<=": operator.le,
    "==": operator.eq,
    ">": operator.gt,
    "<": operator.lt,
}

# Fields with a fixed default that must still appear in every stored record.
_ALWAYS_REQUIRED = ("schema_version", "kind", "team")


def _require_envelope(schema: dict[str, Any]) -> None:
    req = schema.setdefault("required", [])
    for name in reversed(_ALWAYS_REQUIRED):
        if name not in req:
            req.insert(0, name)


# --------------------------------------------------------------------------
# Envelope
# --------------------------------------------------------------------------


class Envelope(Strict):
    """Fields carried by every record."""

    model_config = ConfigDict(json_schema_extra=lambda s, _cls: _require_envelope(s))

    KIND: ClassVar[str] = ""

    schema_version: Literal["1"] = SCHEMA_VERSION
    run_id: SafeId = Field(description="The run this record belongs to; safe as a path segment.")
    team: Literal["team13"] = TEAM
    created_at: UtcTimestamp = Field(description="When our code wrote the record (UTC, trailing Z).")
    aws_time: Optional[UtcTimestamp] = Field(
        default=None, description="Time taken from an AWS source; null when there is none."
    )
    aws_time_source: Optional[NonEmptyStr] = Field(
        default=None,
        description="Where aws_time came from, e.g. cloudwatch:/team13/checkpoint, s3:last_modified, sagemaker:CreationTime.",
    )
    synthetic: StrictBool = Field(description="True for fixtures, dry runs and anything not measured on a real system.")

    @model_validator(mode="after")
    def _aws_time_has_source(self) -> "Envelope":
        if self.aws_time is not None and self.aws_time_source is None:
            raise ValueError("aws_time is set but aws_time_source is missing")
        return self


# --------------------------------------------------------------------------
# Shared sub-objects
# --------------------------------------------------------------------------


class Target(Strict):
    metric: NonEmptyStr
    op: Op
    value: Number
    unit: NonEmptyStr = Field(description="Unit of value, e.g. ratio, ms, count, usd_per_1k.")


class DeclaredTwin(Strict):
    kind: TwinKind
    version: NonEmptyStr
    scenario: NonEmptyStr
    seed: NonNegInt
    config: Optional[FreeDict] = None


class ChangeTwin(Strict):
    kind: TwinKind
    version: NonEmptyStr
    scenario: NonEmptyStr
    seed: NonNegInt
    step: NonNegInt = Field(description="Twin step at which the change was applied.")


class CellChange(Strict):
    cell: NonNegInt = Field(description="Cell index in the twin's per-cell arrays (0-based).")
    d_power_db: Number
    d_tilt_deg: Number


class ModelRef(Strict):
    model_id: SafeId
    version: SafeId


class Kpis(Strict):
    coverage_ratio: Ratio
    sla_ratio: Ratio
    sinr_p10_db: Number
    sinr_mean_db: Number


KPI_KEYS = ("coverage_ratio", "sla_ratio", "sinr_p10_db", "sinr_mean_db")


class KpiError(Strict):
    """Absolute error per KPI, plus one summary number the trust gate scores."""

    coverage_ratio: NonNegNumber
    sla_ratio: NonNegNumber
    sinr_p10_db: NonNegNumber
    sinr_mean_db: NonNegNumber
    summary: NonNegNumber


class LogPointer(Strict):
    log_group: NonEmptyStr
    log_stream: NonEmptyStr
    event_id: NonEmptyStr


class Checker(Strict):
    script: NonEmptyStr
    code_sha: GitSha


# --------------------------------------------------------------------------
# The eight kinds
# --------------------------------------------------------------------------


class DeclaredRecord(Envelope):
    """What a run will be judged on, written before the run. Stored at declared/<run_id>.json."""

    KIND: ClassVar[str] = "declared"
    kind: Literal["declared"] = "declared"
    unit: NonEmptyStr = Field(description="Unit of evidence, e.g. t3.deny.")
    track: Literal["track-1", "track-2", "track-3", "bridge"]
    target: Target
    baseline_run_id: Optional[SafeId] = None
    n_items: PosInt
    test_set_sha256: Sha256Hex
    prompt_sha256: Optional[Sha256Hex] = None
    parser: Optional[NonEmptyStr] = None
    decoding: Optional[FreeDict] = None
    model_id: NonEmptyStr = Field(description="Model under test, e.g. a Bedrock model id or our model_id.")
    grader: NonEmptyStr
    code_sha: GitSha
    region: AwsRegion
    twin: Optional[DeclaredTwin] = None
    declared_at: UtcTimestamp


class ItemRecord(Envelope):
    """One graded test item. Stored in records/<run_id>/item.jsonl."""

    KIND: ClassVar[str] = "item"
    kind: Literal["item"] = "item"
    item_id: NonEmptyStr
    input_sha256: Sha256Hex
    output_sha256: Sha256Hex
    score: Optional[Number] = None
    passed: Optional[StrictBool] = None
    grader: NonEmptyStr
    backend: NonEmptyStr
    latency_ms: Optional[NonNegNumber] = None
    cost_usd: Optional[NonNegNumber] = None
    extra: FreeDict = Field(default_factory=dict)


class DecisionRecord(Envelope):
    """One checkpoint (Cedar) decision. v0: extra keys allowed until a real span is captured.

    Stored in records/<run_id>/decision.jsonl.
    """

    model_config = ConfigDict(extra="allow")

    KIND: ClassVar[str] = "decision"
    kind: Literal["decision"] = "decision"
    case_id: NonEmptyStr
    trace_id: Optional[NonEmptyStr] = None
    span_id: Optional[NonEmptyStr] = None
    time: UtcTimestamp
    agent: NonEmptyStr
    tool: NonEmptyStr
    decision: Literal["ALLOW", "DENY"]
    deciding_rule: Optional[NonEmptyStr] = None
    reason: Optional[str] = None
    guardrail_scores: Optional[FreeDict] = None
    approval_id: Optional[NonEmptyStr] = None
    source: Literal["aws/spans", "/team13/checkpoint"]
    log_pointer: Optional[LogPointer] = None


class ChangeRecord(Envelope):
    """One applied change with predicted vs measured KPIs. Stored in records/<run_id>/change.jsonl."""

    KIND: ClassVar[str] = "change"
    kind: Literal["change"] = "change"
    case_id: NonEmptyStr
    change_id: str = Field(
        pattern=r"^chg-[0-9a-f]{12}$",
        description='Approval id: "chg-" + sha256(json.dumps([case_id, item_id, change]))[:12].',
    )
    twin: ChangeTwin
    change: list[CellChange] = Field(min_length=1)
    predictor: Optional[ModelRef] = None
    policy: Optional[ModelRef] = None
    predicted: Kpis
    measured: Kpis
    error: KpiError
    direction_agree: StrictBool
    trust_before: Optional[Number] = None
    trust_after: Optional[Number] = None

    @model_validator(mode="after")
    def _consistent(self) -> "ChangeRecord":
        cells = [c.cell for c in self.change]
        if len(set(cells)) != len(cells):
            raise ValueError("change lists the same cell twice")
        for k in KPI_KEYS:
            want = abs(getattr(self.predicted, k) - getattr(self.measured, k))
            got = getattr(self.error, k)
            if abs(got - want) > 1e-6:
                raise ValueError(f"error.{k}={got} but |predicted - measured| = {want}")
        return self


class ControlRecord(Envelope):
    """A pause or resume of a model's tool. Stored in records/<run_id>/control.jsonl."""

    KIND: ClassVar[str] = "control"
    kind: Literal["control"] = "control"
    model_id: SafeId
    version: SafeId
    action: Literal["pause", "resume"]
    by: str = Field(pattern=r"^(person:[A-Za-z0-9_.-]+|trust-gate)$")
    reason: NonEmptyStr
    policy_name: NonEmptyStr = Field(description="The Cedar policy (forbid) that enforces the pause.")
    time: UtcTimestamp


class JobRecord(Envelope):
    """One AWS job and its cost. A run may span several jobs. Stored in records/<run_id>/job.jsonl."""

    KIND: ClassVar[str] = "job"
    kind: Literal["job"] = "job"
    job_name: NonEmptyStr
    arn: AwsArn
    job_kind: Literal["training", "processing", "endpoint", "lambda"]
    region: AwsRegion
    status: NonEmptyStr = Field(description="Status as AWS reports it, e.g. Completed, Failed, InService.")
    aws_start: UtcTimestamp
    aws_end: Optional[UtcTimestamp] = Field(default=None, description="Null while the job is still running.")
    instance_type: Optional[NonEmptyStr] = Field(default=None, description="Null for Lambda.")
    billable_s: NonNegNumber
    est_cost_usd: NonNegNumber

    @model_validator(mode="after")
    def _ordered(self) -> "JobRecord":
        # Same Z-suffixed format on both sides, so string order is time order
        # once fractions are padded; compare the second-resolution prefix.
        if self.aws_end is not None and self.aws_end[:19] < self.aws_start[:19]:
            raise ValueError("aws_end is before aws_start")
        return self


class ModelRecord(Envelope):
    """A model version and its artifact. Stored at models/<model_id>/<version>.json."""

    KIND: ClassVar[str] = "model"
    kind: Literal["model"] = "model"
    model_id: SafeId
    version: SafeId
    model_kind: Literal["ppo", "t2-response", "llm-lora"]
    artifact_uri: NonEmptyStr
    artifact_sha256: Sha256Hex
    status: Literal["candidate", "approved", "paused", "withdrawn"]
    trained_on: NonEmptyStr = Field(description="What it was trained on: the training run_id, or a twin/dataset id.")
    metrics: FreeDict = Field(default_factory=dict)


class ResultRecord(Envelope):
    """A unit's checked outcome against its declaration. Stored at results/<run_id>.json."""

    KIND: ClassVar[str] = "result"
    kind: Literal["result"] = "result"
    unit: NonEmptyStr
    metric: NonEmptyStr
    value: Number
    target: Target = Field(description="Copy of the declared target.")
    passed: StrictBool
    n_items: NonNegInt
    checked_at: UtcTimestamp
    checker: Checker
    details: FreeDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _consistent(self) -> "ResultRecord":
        if self.target.metric != self.metric:
            raise ValueError(f"metric {self.metric!r} does not match target.metric {self.target.metric!r}")
        want = OPS[self.target.op](self.value, self.target.value)
        if self.passed != want:
            raise ValueError(f"passed={self.passed} but {self.value} {self.target.op} {self.target.value} is {want}")
        return self


RECORD_MODELS: dict[str, type[Envelope]] = {
    m.KIND: m
    for m in (
        DeclaredRecord,
        ItemRecord,
        DecisionRecord,
        ChangeRecord,
        ControlRecord,
        JobRecord,
        ModelRecord,
        ResultRecord,
    )
}
KINDS: tuple[str, ...] = tuple(RECORD_MODELS)
