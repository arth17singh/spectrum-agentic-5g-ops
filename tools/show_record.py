#!/usr/bin/env python3
"""show_record.py --run-id X --item Y: the judges' live test, read-only.

Prints, for one item of one run:
  1. the declaration (declared/<run_id>.json) and when S3 stored it;
  2. the item's record (records/<run_id>/<kind>.jsonl, the line whose id is Y) and when S3 stored that file;
  3. the record's aws_time and its claimed source;
  4. the matching AWS log event(s): a CloudWatch Logs Insights query, in every Region given, for events
     that mention both the run id and the item id, around the record's aws_time;
  5. the case -> run mapping from DynamoDB (GetItem), when the record names a case and a table is given;
  6. the run's result (results/<run_id>.json), if there is one.

AWS permissions used, and nothing else: s3:ListBucket + s3:GetObject (ListObjectsV2, GetObject),
logs:StartQuery + logs:GetQueryResults, dynamodb:GetItem. Regions: us-east-2 and us-west-2 by default.
Self-contained: it imports nothing from `spectrum`, `evlib` or `check_unit`.

Usage:
  show_record.py --run-id R --item case-0003 --s3 s3://bucket/evidence/     (or EVIDENCE_S3)
  show_record.py --run-id R --item case-0003 --offline tools/tests/fixtures/units/R
Exit codes: 0 found (and a matching log event when the record has an aws_time); 1 the record or a
matching log event was not found; 2 could not look (no access, bad arguments).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

REGIONS = ("us-east-2", "us-west-2")
LOG_GROUPS = ("/team13/checkpoint", "aws/spans", "/team13/bedrock-invocations", "/team13/runner",
              "/aws/lambda/team13-ran-tools", "/aws/sagemaker/TrainingJobs", "/aws/sagemaker/ProcessingJobs")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
SAFE_RE = re.compile(r"^[A-Za-z0-9._:@/+-]{1,200}$")  # ids safe to put inside a Logs Insights string
ID_FIELDS = ("item_id", "case_id", "change_id", "job_name", "model_id")


def parse_time(s) -> datetime:
    if isinstance(s, datetime):
        return s.astimezone(timezone.utc) if s.tzinfo else s.replace(tzinfo=timezone.utc)
    dt = datetime.fromisoformat(str(s).strip().replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError(f"time {s!r} has no UTC offset")
    return dt.astimezone(timezone.utc)


def insights_time(s: str) -> datetime:
    return datetime.fromisoformat(s.strip().replace(" ", "T")).replace(tzinfo=timezone.utc)


def iso(dt: datetime | None) -> str:
    return "-" if dt is None else dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def err_code(e: Exception) -> str:
    r = getattr(e, "response", None) or {}
    return str((r.get("Error") or {}).get("Code", "")) or e.__class__.__name__


class S3Reader:
    def __init__(self, client, uri: str):
        if not uri.startswith("s3://"):
            raise ValueError(f"expected s3://bucket/prefix/, got {uri!r}")
        self.bucket, _, prefix = uri[len("s3://"):].partition("/")
        self.prefix = prefix if (not prefix or prefix.endswith("/")) else prefix + "/"
        self.s3 = client

    def where(self, rel: str) -> str:
        return f"s3://{self.bucket}/{self.prefix}{rel}"

    def last_modified(self, rel: str) -> datetime | None:
        """ListObjectsV2 on the exact key (no HeadObject, so only List permission is needed)."""
        r = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=self.prefix + rel, MaxKeys=5)
        for o in r.get("Contents", []):
            if o["Key"] == self.prefix + rel:
                return parse_time(o["LastModified"])
        return None

    def list(self, prefix: str) -> list[str]:
        out = []
        for page in self.s3.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=self.prefix + prefix):
            out += [o["Key"][len(self.prefix):] for o in page.get("Contents", [])]
        return out

    def get(self, rel: str) -> bytes | None:
        try:
            return self.s3.get_object(Bucket=self.bucket, Key=self.prefix + rel)["Body"].read()
        except Exception as e:  # noqa: BLE001
            if err_code(e) in ("NoSuchKey", "404", "NotFound"):
                return None
            raise


class RealAws:
    def __init__(self):
        import boto3  # noqa: F401

        self._c: dict = {}

    def client(self, service: str, region: str | None = None):
        import boto3

        k = (service, region)
        if k not in self._c:
            self._c[k] = boto3.client(service, region_name=region or REGIONS[0])
        return self._c[k]


def insights(aws, region: str, group: str, query: str, start: datetime, end: datetime, poll_s: float,
             timeout_s: float = 120.0) -> list[dict] | None:
    logs = aws.client("logs", region)
    try:
        qid = logs.start_query(logGroupNames=[group], startTime=int(start.timestamp()),
                               endTime=int(math.ceil(end.timestamp())), queryString=query, limit=50)["queryId"]
    except Exception as e:  # noqa: BLE001
        if err_code(e) == "ResourceNotFoundException":
            return None
        raise
    t0 = time.monotonic()
    while True:
        r = logs.get_query_results(queryId=qid)
        if r.get("status") == "Complete":
            return [{c["field"]: c["value"] for c in row} for row in r.get("results", [])]
        if r.get("status") in ("Failed", "Cancelled", "Timeout", "Unknown"):
            raise RuntimeError(f"query in {group} ({region}) ended {r.get('status')}")
        if time.monotonic() - t0 > timeout_s:
            raise RuntimeError(f"query in {group} ({region}) did not finish in {timeout_s:.0f}s")
        time.sleep(poll_s)


def find_record(b: bytes, item: str, id_field: str | None) -> tuple[int, dict] | None:
    for n, line in enumerate(b.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if not isinstance(r, dict):
            continue
        fields = (id_field,) if id_field else ID_FIELDS
        if any(r.get(f) == item for f in fields):
            return n, r
    return None


def ddb_plain(av: dict):
    (t, v), = av.items()
    if t == "M":
        return {k: ddb_plain(x) for k, x in v.items()}
    if t == "L":
        return [ddb_plain(x) for x in v]
    if t == "N":
        return float(v) if any(c in v for c in ".eE") else int(v)
    return v


def show(run_id: str, item: str, reader: S3Reader, aws, kind: str = "item", id_field: str | None = None,
         regions=REGIONS, log_groups=LOG_GROUPS, around_min: float = 30.0, cases_table: str | None = None,
         poll_s: float = 1.0, out=sys.stdout) -> int:
    say = lambda s="": print(s, file=out)  # noqa: E731
    dump = lambda o: say(json.dumps(o, indent=2, sort_keys=True, ensure_ascii=False))  # noqa: E731
    code = 0

    say(f"== 1. declaration  {reader.where(f'declared/{run_id}.json')}")
    decl = reader.get(f"declared/{run_id}.json")
    if decl is None:
        say("   NOT FOUND")
        code = 1
    else:
        say(f"   S3 LastModified: {iso(reader.last_modified(f'declared/{run_id}.json'))}")
        try:
            dump(json.loads(decl))
        except ValueError:
            say("   (not valid JSON)")

    # Both record layouts (amendment A11): records/<run>/<kind>.jsonl and records/<run>/<kind>/<key>.json
    say()
    say(f"== 2. record  {reader.where(f'records/{run_id}/{kind}')}[.jsonl | /*.json]  "
        f"({id_field or '/'.join(ID_FIELDS)} == {item!r})")
    candidates = reader.list(f"records/{run_id}/{kind}")
    files = [r for r in candidates if r == f"records/{run_id}/{kind}.jsonl"]
    files += sorted(r for r in candidates if r.startswith(f"records/{run_id}/{kind}/") and r.endswith(".json")
                    and r.count("/") == 3)
    hits = []
    for rel in files:
        body = reader.get(rel)
        h = find_record(body, item, id_field) if body is not None else None
        if h:
            hits.append((rel, *h))
    if not hits:
        say("   NOT FOUND" + ("" if files else f" (no {kind} records for this run)"))
        return 1
    if len(hits) > 1:
        say(f"   WARNING: {len(hits)} records match; showing the first: " + ", ".join(h[0] for h in hits))
    rel, line_no, rec = hits[0]
    say(f"   {rel}" + (f" line {line_no}" if rel.endswith(".jsonl") else "")
        + f"; S3 LastModified: {iso(reader.last_modified(rel))}")
    dump(rec)

    say()
    say("== 3. AWS time")
    at = None
    if rec.get("aws_time"):
        try:
            at = parse_time(rec["aws_time"])
        except ValueError:
            pass
    say(f"   aws_time: {rec.get('aws_time')!r}   source: {rec.get('aws_time_source')!r}")
    if rec.get("run_id") != run_id:
        say(f"   WARNING: the record's run_id is {rec.get('run_id')!r}")
        code = 1

    say()
    say("== 4. matching log events (CloudWatch Logs Insights)")
    terms = [run_id] + ([item] if SAFE_RE.match(item) else [])
    query = " | ".join([f'filter @message like "{t}"' for t in terms]
                       + ["fields @timestamp, @ingestionTime, @log, @logStream, @message, @ptr",
                          "sort @timestamp asc", "limit 20"])
    ptr = rec.get("log_pointer") if isinstance(rec.get("log_pointer"), dict) else None
    groups = list(log_groups)
    if ptr and ptr.get("log_group") and ptr["log_group"] not in groups:
        groups.insert(0, ptr["log_group"])
    center = at or datetime.now(timezone.utc)
    start, end = center - timedelta(minutes=around_min), center + timedelta(minutes=around_min)
    say(f"   query: {query}")
    say(f"   window: {iso(start)} .. {iso(end)}; Regions: {', '.join(regions)}")
    found = []
    for region in regions:
        for g in groups:
            try:
                rows = insights(aws, region, g, query, start, end, poll_s)
            except Exception as e:  # noqa: BLE001
                say(f"   {g} ({region}): query failed ({err_code(e)})")
                continue
            if rows is None:
                continue
            for r in rows:
                if item not in r.get("@message", ""):
                    continue  # the item id was not safe to query by: filter here
                found.append((region, g, r))
    if not found:
        say("   no log event mentions both the run id and the item id in that window")
        if at is not None:
            code = 1
    for region, g, r in found:
        t = insights_time(r["@timestamp"])
        agree = "" if at is None else (f"   (aws_time {'MATCHES' if abs((t - at).total_seconds()) <= 1 else 'differs by %.1fs from' % (t - at).total_seconds()} this event)")
        say(f"   {r['@timestamp']} UTC  {g} ({region})  stream {r.get('@logStream', '')}{agree}")
        msg = r.get("@message", "")
        say(f"      {msg[:400]}{'...' if len(msg) > 400 else ''}")
    if at is not None and found and not any(abs((insights_time(r["@timestamp"]) - at).total_seconds()) <= 1
                                            for _, _, r in found):
        say("   WARNING: no matching event has the record's aws_time")
        code = 1

    extra = rec.get("extra") if isinstance(rec.get("extra"), dict) else {}
    case = rec.get("case_id") or extra.get("case_id")
    if case and cases_table:
        say()
        say(f"== 5. case -> run (DynamoDB {cases_table})")
        for region in regions:
            try:
                r = aws.client("dynamodb", region).get_item(TableName=cases_table, Key={"case_id": {"S": case}})
            except Exception as e:  # noqa: BLE001
                say(f"   {region}: GetItem failed ({err_code(e)})")
                continue
            if r.get("Item"):
                row = {k: ddb_plain(v) for k, v in r["Item"].items()}
                say(f"   {region}: {json.dumps(row, sort_keys=True)}")
                if row.get("run_id") != run_id:
                    say(f"   WARNING: the cases table maps {case} to run {row.get('run_id')!r}")
                    code = 1
                break
        else:
            say("   no row for this case")

    say()
    say(f"== 6. result  {reader.where(f'results/{run_id}.json')}")
    res = reader.get(f"results/{run_id}.json")
    if res is None:
        say("   none yet")
    else:
        try:
            r = json.loads(res)
            say(f"   {r.get('metric')} = {r.get('value')} target {r.get('target')} passed={r.get('passed')} "
                f"n_items={r.get('n_items')}; S3 LastModified {iso(reader.last_modified(f'results/{run_id}.json'))}")
        except ValueError:
            say("   (not valid JSON)")
    return code


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Show one evidence record and its AWS event (read-only).")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--item", required=True, help="item id (or case_id / change_id / job_name / model_id)")
    ap.add_argument("--kind", default="item", help="record kind file to look in (default item)")
    ap.add_argument("--id-field", help="match only this field (default: any of " + ", ".join(ID_FIELDS) + ")")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--s3", help="evidence prefix s3://bucket/evidence/ (default $EVIDENCE_S3)")
    src.add_argument("--offline", type=Path, metavar="FIXTURE", help="fixture directory: no AWS calls")
    ap.add_argument("--region", action="append", help=f"Region (repeatable; default {' '.join(REGIONS)})")
    ap.add_argument("--log-group", action="append", help="log group (repeatable; default $EVIDENCE_LOG_GROUPS or built-in list)")
    ap.add_argument("--around-min", type=float, default=30.0, help="search this many minutes either side of aws_time")
    ap.add_argument("--cases-table", default=os.environ.get("EVIDENCE_CASES_TABLE"),
                    help="DynamoDB table mapping case_id -> run_id (default $EVIDENCE_CASES_TABLE)")
    a = ap.parse_args(argv)
    if not RUN_ID_RE.match(a.run_id) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", a.kind):
        print("bad --run-id or --kind", file=sys.stderr)
        return 2
    regions = a.region or list(REGIONS)
    groups = a.log_group or [g for g in os.environ.get("EVIDENCE_LOG_GROUPS", "").split(",") if g.strip()] or list(LOG_GROUPS)
    try:
        if a.offline:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from evfake import FakeAws

            aws, uri = FakeAws.from_fixture(a.offline)
            reader, poll = S3Reader(aws.client("s3"), uri), 0.0
        else:
            uri = a.s3 or os.environ.get("EVIDENCE_S3", "").strip()
            if not uri:
                print("no evidence location: pass --s3 or --offline (or set EVIDENCE_S3)", file=sys.stderr)
                return 2
            aws = RealAws()
            reader, poll = S3Reader(aws.client("s3", regions[0]), uri), 1.0
        return show(a.run_id, a.item, reader, aws, a.kind, a.id_field, regions, groups, a.around_min,
                    a.cases_table, poll)
    except ImportError:
        print("boto3 is not installed (use --offline for fixtures)", file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001
        print(f"show_record: could not look: {e.__class__.__name__}: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
