"""Which time on a record was produced by AWS, and which was written by our own code.

The rule of the project: a time is shown as "AWS time" only if AWS produced it. A runner or Lambda clock is never
AWS time. `resolve(rec)` returns the AWS time of one record, the source it came from, and the record's own
("runner") clock separately:

  cloudwatch        the record's `aws_time` is a CloudWatch log-event time (aws_time_source `cloudwatch:<log group>`:
                    the AgentCore/CloudWatch policy-decision log, the checkpoint log, the spans log)
  policy_createdAt  `aws_time` is the `createdAt` of the AgentCore policy (aws_time_source `agentcore:policy.createdAt`)
  gateway_date      `aws_time` is the Date header of the AgentCore Gateway response (`gateway:http-date`)
  cloudwatch_attest the CloudWatch time of the log line in which the SageMaker job that wrote the record attests the
                    sha256 of its file (the evidence sync keeps these in <root>/.sync/attest/<run>.json). Used only
                    when that entry's state is "match" and the sha256 is the one of the file on disk. AWS recorded it.
  s3_stored         the S3 LastModified of the record's own object, which S3 assigns when it stores the object. It is
                    when the record was STORED, not when the event happened; it is never given to a record read from
                    a .jsonl batch file (the file's time bounds many records, it is not theirs).
  none              AWS produced no time for this record

Anything else in `aws_time_source` (`lambda:clock`, null, an unknown label) is NOT accepted as AWS time: the value
is reported as the runner clock, and the record falls back to its S3 stored time or to `none`. Unknown labels are
never guessed from.

`STORED_KEY` is the private field the Trace and Twin View loaders set on a shallow copy of a record to carry its S3
stored time in. It never appears in a response: the views build their output from named fields only.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable

Rec = dict[str, Any]

STORED_KEY = "_stored_at"      # private carriers set on a shallow copy of a record; no response outputs them
ATTEST_KEY = "_attested_at"
# In priority order: the CloudWatch decision log, the job attestation, the policy / Gateway stamps, S3 storage.
SOURCES = ("cloudwatch", "cloudwatch_attest", "policy_createdAt", "gateway_date", "s3_stored", "none")
# Sources where AWS recorded the EVENT (a log event, a job's log line, a policy creation, a Gateway response).
# s3_stored is only "when the object was stored", so it never makes a record "real" on its own.
EVENT_SOURCES = ("cloudwatch", "cloudwatch_attest", "policy_createdAt", "gateway_date")
LABELS = {"cloudwatch": "CloudWatch", "cloudwatch_attest": "job attestation", "policy_createdAt": "policy createdAt",
          "gateway_date": "gateway Date", "s3_stored": "S3 stored", "none": "none"}


def parse(ts: Any) -> datetime | None:
    if not isinstance(ts, str) or not ts:
        return None
    try:
        d = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def classify(label: Any) -> str | None:
    """The source class of a record's `aws_time_source` label, or None when the label is not an AWS source."""
    s = label if isinstance(label, str) else ""
    if s.startswith("cloudwatch:"):
        return "cloudwatch"
    if s == "agentcore:policy.createdAt":
        return "policy_createdAt"
    if s == "gateway:http-date":
        return "gateway_date"
    if s.startswith("s3:"):
        return "s3_stored"
    return None


def _iso(v: Any) -> str | None:
    return v if isinstance(v, str) and parse(v) else None


def resolve(rec: Rec) -> Rec:
    """{aws_time, aws_time_source, runner_time} for one record. Pure; the carried S3 and attestation times come from
    rec[STORED_KEY] and rec[ATTEST_KEY]. Priority: CloudWatch decision log, job attestation, policy createdAt or
    Gateway Date, S3 stored."""
    own = _iso(rec.get("aws_time"))
    src = classify(rec.get("aws_time_source")) if own else None
    attested, stored = _iso(rec.get(ATTEST_KEY)), _iso(rec.get(STORED_KEY))
    if own and src == "cloudwatch":
        aws, source = own, "cloudwatch"
    elif attested:
        aws, source = attested, "cloudwatch_attest"
    elif own and src:
        aws, source = own, src
    elif stored:
        aws, source = stored, "s3_stored"
    else:
        aws, source = None, "none"
    # the record's own clock: what our code wrote. An `aws_time` that AWS did not produce (lambda:clock) is one too.
    accepted = own if (own and src) else None
    runner = None
    for cand in (rec.get("created_at"), rec.get("time"), own if not src else None):
        if isinstance(cand, str) and parse(cand) and cand != accepted:
            runner = cand
            break
    return {"aws_time": aws, "aws_time_source": source, "runner_time": runner}


def with_times(rec: Rec, times: Rec | None) -> Rec:
    """A shallow copy of rec carrying the file's AWS times (reader meta {"stored", "attested"}), or rec itself."""
    if not times or not (times.get("stored") or times.get("attested")):
        return rec
    out = dict(rec)
    if times.get("stored"):
        out[STORED_KEY] = times["stored"]
    if times.get("attested"):
        out[ATTEST_KEY] = times["attested"]
    return out


def read_records(reader: Any, kind: str) -> list[Rec]:
    """reader.all_records(kind) with each record carrying its file's AWS times. A reader without
    all_records_times (a test double) just gives the records."""
    fn = getattr(reader, "all_records_times", None)
    if fn is None:
        return list(reader.all_records(kind))
    return [with_times(rec, t) for rec, t in fn(kind)]


VIAVI_REASON = "ran outside AWS (through the team's browser session): no AWS-recorded time exists"


def no_event_reason(rec: Rec) -> str | None:
    """Why a record has no AWS-recorded event time (None when it has one). Said plainly, never hidden."""
    t = resolve(rec)
    if t["aws_time_source"] in EVENT_SOURCES:
        return None
    tw = rec.get("twin") if isinstance(rec.get("twin"), dict) else {}
    if "viavi_operate" in str(rec.get("run_id") or "") or tw.get("kind") == "viavi":
        return VIAVI_REASON
    if str(rec.get("aws_time_source") or "").startswith("lambda:"):
        return "stamped by the Lambda's own clock; no CloudWatch time for it is mirrored here" + (
            " (S3 stored time shown)" if t["aws_time_source"] == "s3_stored" else "")
    if t["aws_time_source"] == "s3_stored":
        return "no AWS-recorded event time found; only the S3 stored time"
    return "no AWS time found for this record"


def reasons(recs: Iterable[Rec]) -> list[Rec]:
    """[{reason, n}] for the non-synthetic rows without an AWS-recorded event time, most frequent first. The rows
    are the views' own (they carry `no_event_reason`, set from the record by `no_event_reason`)."""
    c: dict[str, int] = {}
    for r in recs:
        if r.get("synthetic") is not True and not r.get("real") and (why := r.get("no_event_reason")):
            c[why] = c.get(why, 0) + 1
    return [{"reason": k, "n": v} for k, v in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))]


def is_event_time(rec: Rec) -> bool:
    """AWS recorded the event itself (not only the storing of the record)."""
    return resolve(rec)["aws_time_source"] in EVENT_SOURCES


def summary(items: Iterable[Rec]) -> Rec:
    """"AWS time on k of n (cloudwatch a, S3 stored b, ...)" over rows that carry `aws_time_source`."""
    counts = {s: 0 for s in SOURCES}
    n = 0
    for it in items:
        n += 1
        s = it.get("aws_time_source")
        counts[s if s in counts else "none"] += 1
    k = n - counts["none"]
    parts = [f"{LABELS[s]} {counts[s]}" for s in SOURCES if s != "none" and counts[s]]
    text = f"AWS time on {k} of {n}" + (f" ({', '.join(parts)})" if parts else "")
    return {"n": n, "k": k, "by_source": counts, "text": text}
