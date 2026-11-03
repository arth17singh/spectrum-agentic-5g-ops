"""The eight evidence record kinds and their shared envelope (schema_version "1").

Field list: BUILD_CONTRACT.md, "Record contracts". Every record is one JSON object
with the envelope fields plus its kind's fields. Closed objects reject unknown keys,
except `decision` (v0: extra keys allowed until a real AgentCore span is captured)
and the free-form dicts (item.extra, model.metrics, result.details,
declared.decoding, declared.twin.config).

Python-only cross-field checks (JSON Schema cannot express them; noted per model):
- envelope: aws_time set => aws_time_source set.
- change (amendment 1): see ChangeRecord's docstring (canonical change_text,
  deltas, error on deltas, scored/unscored_reason, probe rule).
- result: target.metric == metric; passed == (value <op> target.value).
- job: aws_end >= aws_start when both are set.
"""

from __future__ import annotations

import operator
from typing import Any, ClassVar, Literal, Optional

from pydantic import ConfigDict, Field, StrictBool, model_validator

from .change import CHANGE_TEXT_PATTERN
from .change import canon as canon_change
from .change import parse as parse_change
from .types import (
    AwsArn,
    AwsRegion,
    FreeDict,
    GitSha,
    NonEmptyStr,
    NonNegInt,
    NonNegNumber,
    Number,
    num,
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
        description=(
            "Where aws_time came from, as <source>:<detail> with source one of lambda, sagemaker, cloudwatch, s3, "
            "unknown; e.g. lambda:clock (a Lambda's own clock), cloudwatch:/team13/checkpoint, s3:last_modified, "
            "sagemaker:CreationTime."
        ),
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
    preset: Optional[NonEmptyStr] = Field(
        default=None, description='Twin config preset name, e.g. "official" or "viavi-like" (tells R1 from R2).'
    )


Share = num(ge=0, le=1, description="A share in [0, 1].")


class CellState(Strict):
    """One cell's state in a twin snapshot. Non-finite twin values (e.g. -inf RSRP of a dark cell) go as null."""

    cell: NonNegInt = Field(description="Cell index (0-based).")
    power_dbm: Number
    tilt_deg: Number
    ue_count: NonNegInt
    coverage_share: Optional[Share] = None
    sla_share: Optional[Share] = None
    sinr_mean_db: Optional[Number] = None
    sinr_p10_db: Optional[Number] = None
    rsrp_mean_dbm: Optional[Number] = None
    rsrp_p10_dbm: Optional[Number] = None
    alarm: Optional[StrictBool] = Field(default=None, description="The twin's per-cell alarm (bool in ran_rl).")
    asleep: Optional[StrictBool] = None


class TwinSnapshot(Strict):
    step: NonNegInt
    coverage_ratio: Ratio
    sla_ratio: Ratio
    per_cell: list[CellState] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def _unique_cells(self) -> "TwinSnapshot":
        cells = [c.cell for c in self.per_cell]
        if len(set(cells)) != len(cells):
            raise ValueError(f"per_cell lists a cell twice: {cells}")
        return self


class TwinState(Strict):
    """Per-cell twin state around a change, for the Twin View."""

    before: TwinSnapshot = Field(description="State the agents observed at twin.step.")
    after: TwinSnapshot = Field(description="State after the change plus one step.")
    baseline: Optional[TwinSnapshot] = Field(
        default=None, description="No-change counterfactual at the same step as after."
    )


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


DeltaRatio = num(ge=-1, le=1, description="Change of a ratio, in [-1, 1].")


class KpiDelta(Strict):
    """KPI change against the no-change counterfactual at the same step (A5)."""

    coverage_ratio: DeltaRatio
    sla_ratio: DeltaRatio
    sinr_p10_db: Number
    sinr_mean_db: Number


class KpiError(Strict):
    """Absolute error per KPI delta, plus one summary number the trust gate scores."""

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
    n_cases: Optional[NonNegInt] = Field(default=None, description="How many cases the run opens (suite).")
    test_set_sha256: Sha256Hex
    subset: Optional[NonEmptyStr] = Field(
        default=None,
        description='Deterministic filter over the frozen test set, e.g. "expect=DENY"; n_items counts the subset.',
    )
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
    """One graded test item. Stored as records/<run_id>/item/<key>.json or in records/<run_id>/item.jsonl."""

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

    Stored as records/<run_id>/decision/<key>.json or in records/<run_id>/decision.jsonl.
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
    """One applied change: measured KPI delta vs the no-change counterfactual, and the predicted delta.

    Amendment 1 (A2, A5, A14, A15). Stored as records/<run_id>/change/<change_id>.json
    (one object per record, Lambdas) or appended to records/<run_id>/change.jsonl (batch).

    Python-only rules:
    - change_text == canon(change) and change == parse(change_text) (store the rounded list);
    - measured_delta == measured_kpis - baseline_kpis (abs tol 1e-6);
    - error.<kpi> == |predicted_delta - measured_delta| (abs tol 1e-6); error needs predicted_delta;
    - scored: false requires unscored_reason (e.g. "clipped"); scored: true forbids it;
    - scored: true requires predictor, predicted_delta, error and direction_agree; scored: false
      makes them optional (reasons: "probe", "clipped", "no_prediction" when ran___predict is paused).
    """

    KIND: ClassVar[str] = "change"
    kind: Literal["change"] = "change"
    case_id: NonEmptyStr
    change_id: str = Field(
        pattern=r"^chg-[0-9a-f]{12}$",
        description='Approval id: "chg-" + sha256(json.dumps([case_id, item_id, change_text]))[:12].',
    )
    item_id: NonEmptyStr = Field(description='Step id from ran___propose, e.g. "step-3".')
    change: list[CellChange] = Field(min_length=1, description="parse(change_text): the change as applied.")
    change_text: str = Field(pattern=CHANGE_TEXT_PATTERN, description="Canonical change string (A2).")
    twin: ChangeTwin
    baseline_kpis: Kpis = Field(description="KPIs of the no-change replay at the same step.")
    measured_kpis: Kpis = Field(description="KPIs after the change at the same step.")
    measured_delta: KpiDelta = Field(description="measured_kpis - baseline_kpis.")
    predicted_delta: Optional[KpiDelta] = None
    predictor: Optional[ModelRef] = None
    policy: Optional[ModelRef] = None
    error: Optional[KpiError] = Field(default=None, description="Per-KPI |predicted_delta - measured_delta|, unrounded.")
    direction_agree: Optional[StrictBool] = None
    scored: StrictBool = Field(description="False when the change must not count toward trust (e.g. clipped).")
    unscored_reason: Optional[NonEmptyStr] = Field(default=None, description='Required when scored is false: "probe", "clipped" or "no_prediction".')
    probe: StrictBool = Field(description="True for a fixed change applied with no model (A14).")
    trust_before: Optional[Number] = None
    trust_after: Optional[Number] = None
    approved_by: Optional[NonEmptyStr] = Field(
        default=None,
        description="Approver as the kit's approvals table stores it (decided_by): a pseudonymous id, never a real name.",
    )
    approved_at: Optional[UtcTimestamp] = Field(
        default=None, description="When the approval was decided (decided_at), UTC."
    )
    twin_state: Optional[TwinState] = Field(
        default=None, description="Per-cell twin state before and after the change, and the no-change baseline."
    )

    @model_validator(mode="after")
    def _consistent(self) -> "ChangeRecord":
        cells = [c.cell for c in self.change]
        if len(set(cells)) != len(cells):
            raise ValueError("change lists the same cell twice")
        as_list = [c.model_dump() for c in self.change]
        try:
            text = canon_change(as_list)
            parsed = parse_change(self.change_text)
        except ValueError as exc:
            raise ValueError(f"change/change_text: {exc}") from None
        if text != self.change_text:
            raise ValueError(f"change_text {self.change_text!r} != canon(change) {text!r}")
        if parsed != as_list:
            raise ValueError("change must equal parse(change_text): store the rounded values")
        for k in KPI_KEYS:
            want = getattr(self.measured_kpis, k) - getattr(self.baseline_kpis, k)
            got = getattr(self.measured_delta, k)
            if abs(got - want) > 1e-6:
                raise ValueError(f"measured_delta.{k}={got} but measured - baseline = {want}")
        if self.error is not None:
            if self.predicted_delta is None:
                raise ValueError("error is set but predicted_delta is missing")
            for k in KPI_KEYS:
                want = abs(getattr(self.predicted_delta, k) - getattr(self.measured_delta, k))
                got = getattr(self.error, k)
                if abs(got - want) > 1e-6:
                    raise ValueError(f"error.{k}={got} but |predicted_delta - measured_delta| = {want}")
        if not self.scored and self.unscored_reason is None:
            raise ValueError("scored is false but unscored_reason is missing")
        if self.scored and self.unscored_reason is not None:
            raise ValueError("unscored_reason is set but scored is true")
        if self.scored:
            missing = [
                n for n in ("predictor", "predicted_delta", "error", "direction_agree") if getattr(self, n) is None
            ]
            if missing:
                raise ValueError(f"scored change is missing {', '.join(missing)}")
        return self


class ControlRecord(Envelope):
    """A pause or resume of a model's tool. Stored as records/<run_id>/control/<key>.json or in records/<run_id>/control.jsonl."""

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
    """One AWS job and its cost. A run may span several jobs. Stored as records/<run_id>/job/<key>.json or in records/<run_id>/job.jsonl."""

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
