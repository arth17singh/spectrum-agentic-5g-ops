"""Items and result of the DEV reward check t1.rl.reward-check-20261103T020244Z-4f92, from AWS's own records.

    python -m t1.rl.reward_check --profile fyuz            # dry run: prints the records, writes nothing
    python -m t1.rl.reward_check --profile fyuz --write    # the main session: writes 2 items + the result

The declaration (stored before the job) names two items:
  sagemaker-job-completed          DescribeTrainingJob says Completed (aws_time: TrainingEndTime);
  training-coverage-not-collapsed  the job's LAST `METRIC coverage_mean=<x>` line in /aws/sagemaker/TrainingJobs
                                   (streams <job>/...) is >= 0.82 (aws_time: that log event's timestamp; its
                                   ingestion time, stream and event id are recorded too).
Log events are de-duplicated by eventId (FilterLogEvents can return one twice) and the last is the latest
(timestamp, eventId). Messages are stored and printed with any URL, IP or host name masked.
Read-only AWS: S3 GetObject/ListObjectsV2, SageMaker DescribeTrainingJob, Logs FilterLogEvents. With
--write: records/<run>/item/<key>.json (one object per item) and results/<run>.json, never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "contracts"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from t1.rl.mask import Masker  # noqa: E402
from tools import evlib  # noqa: E402
from tools.sm import spec as smspec  # noqa: E402

RUN_ID = "t1.rl.reward-check-20261103T020244Z-4f92"
JOB = "ran-rl-team13-20261103-020251"
THRESHOLD = 0.82
LOG_GROUP = "/aws/sagemaker/TrainingJobs"
ITEM_JOB, ITEM_COVERAGE = "sagemaker-job-completed", "training-coverage-not-collapsed"
METRIC = "coverage_mean"
# the SDK launcher's metric definition shape; the job's own MetricDefinitions win when present
DEFAULT_REGEX = r"METRIC coverage_mean=([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)(?=\s|$)"
SCRIPT = "t1/rl/reward_check.py"


class Refusal(RuntimeError):
    pass


def sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def ms_iso(ms: int) -> str:
    return smspec.iso(datetime.fromtimestamp(ms / 1000.0, timezone.utc))


def metric_regex(d: dict, metric: str = METRIC) -> tuple[re.Pattern, str]:
    for m in (d.get("AlgorithmSpecification") or {}).get("MetricDefinitions") or []:
        if m.get("Name") == metric and m.get("Regex"):
            return re.compile(m["Regex"]), "DescribeTrainingJob.AlgorithmSpecification.MetricDefinitions"
    return re.compile(DEFAULT_REGEX.replace(METRIC, metric)), "default (the SDK launcher's metric definition shape)"


def last_metric(aws, job: str, region: str, d: dict, metric: str) -> dict | None:
    """The job's last `METRIC <metric>=<x>` log event: value, time, ingestion time, stream, event id; or None."""
    rx, src = metric_regex(d, metric)
    try:
        evs = metric_events(aws.client("logs", region), job, rx, metric)
    except Exception as e:  # noqa: BLE001 - no log group (yet): there is no such line
        if ((getattr(e, "response", None) or {}).get("Error") or {}).get("Code") != "ResourceNotFoundException":
            raise
        evs = []
    if not evs:
        return None
    e = evs[-1]
    return {"metric": metric, "value": e["value"], "timestamp": ms_iso(e["timestamp"]),
            "ingestion_time": ms_iso(e["ingestionTime"]) if e.get("ingestionTime") else None,
            "log_group": LOG_GROUP, "log_stream": e.get("logStreamName"), "event_id": e.get("eventId"),
            "n_events": len(evs), "regex_source": src}


def metric_events(logs, job: str, rx: re.Pattern, metric: str = METRIC) -> list[dict]:
    """Every `METRIC <metric>` event of the job, de-duplicated by eventId, oldest first."""
    seen: dict[str, dict] = {}
    kw = {"logGroupName": LOG_GROUP, "logStreamNamePrefix": f"{job}/", "filterPattern": f'"METRIC {metric}"'}
    while True:
        r = logs.filter_log_events(**kw)
        for e in r.get("events", []):
            m = rx.search(e.get("message", ""))
            if m:
                seen.setdefault(e["eventId"], {**e, "value": float(m.group(1))})
        tok = r.get("nextToken")
        if not tok or tok == kw.get("nextToken"):
            break
        kw["nextToken"] = tok
    return sorted(seen.values(), key=lambda e: (e["timestamp"], e["eventId"]))


def build(aws, run_id: str, job: str, region: str, bucket: str, threshold: float) -> tuple[dict, list[dict]]:
    s3, sm, logs = aws.client("s3"), aws.client("sagemaker", region), aws.client("logs", region)
    store = evlib.S3Store(f"s3://{bucket}/{smspec.EVIDENCE_PREFIX}", client=s3)
    decl = evlib.read_declared(run_id, store)
    if decl is None:
        raise Refusal(f"no declaration declared/{run_id}.json")
    items_decl = ((decl.get("twin") or {}).get("config") or {}).get("items") or {}
    if sorted(items_decl) != sorted((ITEM_JOB, ITEM_COVERAGE)) or decl.get("n_items") != 2:
        raise Refusal(f"the declaration's items are {sorted(items_decl)} (n_items {decl.get('n_items')}), "
                      f"not {ITEM_JOB} and {ITEM_COVERAGE}")
    if str(threshold) not in items_decl[ITEM_COVERAGE]:
        raise Refusal(f"the declared coverage item does not name the threshold {threshold}: "
                      f"{items_decl[ITEM_COVERAGE]!r}")
    d = sm.describe_training_job(TrainingJobName=job)
    status = d.get("TrainingJobStatus")
    if status in ("InProgress", "Stopping"):
        raise Refusal(f"job {job} is still {status}")
    end = smspec.iso(d.get("TrainingEndTime") or d.get("LastModifiedTime"))
    created = d.get("CreationTime")
    declared = evlib.parse_time(decl["declared_at"])
    if created is not None and created.astimezone(timezone.utc) <= declared:
        raise Refusal(f"job {job} was created {smspec.iso(created)}, not after the declaration ({decl['declared_at']})")
    grader = decl["grader"]
    synthetic = bool(decl.get("synthetic", False))
    job_facts = {"job_name": job, "arn": d.get("TrainingJobArn"), "status": status,
                 "secondary_status": d.get("SecondaryStatus"), "creation_time": smspec.iso(created),
                 "training_end_time": end, "billable_s": d.get("BillableTimeInSeconds"),
                 "instance_type": (d.get("ResourceConfig") or {}).get("InstanceType"),
                 "spot": bool(d.get("EnableManagedSpotTraining")), "failure_reason": d.get("FailureReason")}
    passed1 = status == "Completed"
    item1 = {"item_id": ITEM_JOB, "synthetic": synthetic,
             "input_sha256": sha({"item_id": ITEM_JOB, "job_name": job, "check": "TrainingJobStatus == Completed"}),
             "output_sha256": sha({"status": status}), "passed": passed1, "score": 1.0 if passed1 else 0.0,
             "grader": grader, "backend": "sagemaker:DescribeTrainingJob",
             "aws_time": end, "aws_time_source": "sagemaker:TrainingEndTime",
             "extra": {"requirement": items_decl[ITEM_JOB], **job_facts,
                       "counted_as": "passed" if passed1 else "not passed"}}
    rx, rx_src = metric_regex(d)
    evs = metric_events(logs, job, rx)
    masker = Masker()
    if evs:
        last = evs[-1]
        value = last["value"]
        passed2 = value >= threshold
        extra2 = {"requirement": items_decl[ITEM_COVERAGE], "value": value, "threshold": threshold, "op": ">=",
                  "log_group": LOG_GROUP, "log_stream": last.get("logStreamName"), "event_id": last.get("eventId"),
                  "timestamp": ms_iso(last["timestamp"]), "timestamp_ms": last["timestamp"],
                  "ingestion_time": ms_iso(last["ingestionTime"]) if last.get("ingestionTime") else None,
                  "ingestion_time_ms": last.get("ingestionTime"), "message": masker.mask(last["message"]).strip(),
                  "n_metric_events": len(evs), "metric_regex_source": rx_src,
                  "series": [e["value"] for e in evs], "counted_as": "passed" if passed2 else "not passed"}
        aws_time, src = ms_iso(last["timestamp"]), f"cloudwatch:{LOG_GROUP}"
    else:
        value, passed2 = None, False
        extra2 = {"requirement": items_decl[ITEM_COVERAGE], "value": None, "threshold": threshold,
                  "n_metric_events": 0, "reason": f"no METRIC {METRIC} line in the job's log: counted as not passed",
                  "counted_as": "not passed"}
        aws_time, src = end, "sagemaker:TrainingEndTime"
    item2 = {"item_id": ITEM_COVERAGE, "synthetic": synthetic,
             "input_sha256": sha({"item_id": ITEM_COVERAGE, "job_name": job, "log_group": LOG_GROUP,
                                  "metric": METRIC, "rule": f"last value >= {threshold}"}),
             "output_sha256": sha({"value": value, "event_id": extra2.get("event_id")}),
             "passed": passed2, "score": value, "grader": grader, "backend": f"cloudwatch:{LOG_GROUP}",
             "aws_time": aws_time, "aws_time_source": src, "extra": extra2}
    return decl, [item1, item2]


def main(argv: list[str] | None = None, aws=None, out=print) -> int:
    ap = argparse.ArgumentParser(prog="python -m t1.rl.reward_check", description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-id", default=RUN_ID)
    ap.add_argument("--job", default=JOB)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--write", action="store_true", help="write the items and the result (default: dry run)")
    ap.add_argument("--profile", default=os.environ.get("AWS_PROFILE"))
    ap.add_argument("--region", default=smspec.REGION)
    ap.add_argument("--bucket", help="data bucket (default: $TEAM13_DATA_BUCKET, else from the STS account)")
    a = ap.parse_args(argv)
    if aws is None:
        from tools.sm.run import Aws  # noqa: PLC0415

        aws = Aws(a.profile, a.region)
    a.bucket = a.bucket or smspec.data_bucket(
        None if os.environ.get(smspec.BUCKET_ENV) else aws.client("sts").get_caller_identity()["Account"])
    store = evlib.S3Store(f"s3://{a.bucket}/{smspec.EVIDENCE_PREFIX}", client=aws.client("s3"), region=a.region)
    try:
        decl, items = build(aws, a.run_id, a.job, a.region, a.bucket, a.threshold)
        if evlib.read_result(a.run_id, store) is not None or evlib.read_records(a.run_id, "item", store):
            raise Refusal(f"run {a.run_id} already has items or a result; evidence is written once")
        result = evlib.make_result(decl, items, SCRIPT, evlib.code_sha(), details={
            "items": {it["item_id"]: it["passed"] for it in items},
            "job_name": a.job, "coverage_threshold": a.threshold,
            "note": "DEV reward check, not a scored unit (see the declaration's purpose)"})
        for it in items:  # validate as stored, before anything is written
            evlib.validate({"schema_version": "1", "kind": "item", "team": "team13", "created_at": evlib.utc_now(),
                            "run_id": a.run_id, **it})
        evlib.validate({"schema_version": "1", "kind": "result", "team": "team13", "created_at": evlib.utc_now(),
                        "aws_time": None, "aws_time_source": None, **result})
    except (Refusal, evlib.EvidenceError) as e:
        out(f"refused: {e}")
        return 2
    for it in items:
        out(f"item {it['item_id']}: {'PASS' if it['passed'] else 'FAIL'}  aws_time {it['aws_time']} "
            f"({it['aws_time_source']})")
    out(json.dumps({"items": items, "result": result}, indent=2, sort_keys=True))
    out(f"result {result['metric']} = {result['value']} {result['target']['op']} {result['target']['value']} -> "
        f"{'PASSED' if result['passed'] else 'MISSED'}")
    if not a.write:
        out("DRY RUN: nothing written (pass --write)")
        return 0
    for uri in evlib.append_records(a.run_id, "item", items, store=store, layout="objects"):
        out(f"wrote {uri}")
    out(f"wrote {evlib.write_result(result, store)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
