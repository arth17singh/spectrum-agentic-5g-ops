"""Which time on a record was produced by AWS, and which was written by our own code.

The rule of the project: a time is shown as "AWS time" only if AWS produced it. A runner or Lambda clock is never
AWS time. `resolve(rec)` returns the AWS time of one record, the source it came from, and the record's own
("runner") clock separately:

  cloudwatch        the record's `aws_time` is a CloudWatch log-event time (aws_time_source `cloudwatch:<log group>`:
                    the AgentCore/CloudWatch policy-decision log, the checkpoint log, the spans log)
  policy_createdAt  `aws_time` is the `createdAt` of the AgentCore policy (aws_time_source `agentcore:policy.createdAt`)
  gateway_date      `aws_time` is the Date header of the AgentCore Gateway response (`gateway:http-date`)
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

STORED_KEY = "_stored_at"
SOURCES = ("cloudwatch", "s3_stored", "policy_createdAt", "gateway_date", "none")
# Sources where AWS recorded the EVENT (a log event, a policy creation, a Gateway response). s3_stored is only
# "when the object was stored", so it never makes a record "real" on its own.
EVENT_SOURCES = ("cloudwatch", "policy_createdAt", "gateway_date")
LABELS = {"cloudwatch": "CloudWatch", "s3_stored": "S3 stored", "policy_createdAt": "policy createdAt",
          "gateway_date": "gateway Date", "none": "none"}


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


def resolve(rec: Rec) -> Rec:
    """{aws_time, aws_time_source, runner_time} for one record. Pure; the S3 time comes from rec[STORED_KEY]."""
    own = rec.get("aws_time") if isinstance(rec.get("aws_time"), str) and parse(rec.get("aws_time")) else None
    src = classify(rec.get("aws_time_source")) if own else None
    stored = rec.get(STORED_KEY) if isinstance(rec.get(STORED_KEY), str) and parse(rec.get(STORED_KEY)) else None
    if own and src:
        aws, source = own, src
    elif stored:
        aws, source = stored, "s3_stored"
    else:
        aws, source = None, "none"
    # the record's own clock: what our code wrote. An `aws_time` that AWS did not produce (lambda:clock) is one too.
    runner = None
    for cand in (rec.get("created_at"), rec.get("time"), own if not src else None):
        if isinstance(cand, str) and parse(cand) and not (src and cand == own):
            runner = cand
            break
    return {"aws_time": aws, "aws_time_source": source, "runner_time": runner}


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
