"""In-memory stand-ins for the AWS APIs the evidence tools touch: S3, CloudWatch Logs (Insights),
SageMaker describe-*, DynamoDB GetItem. Used by the offline modes of check_unit/show_record and by tests.
No network, no boto3 needed. Every call is recorded in FakeAws.calls as (service, operation, region).

An offline fixture is a directory:
  aws.json     what "AWS" holds: S3 LastModified per key, SageMaker jobs, log events, DynamoDB items
  evidence/    the evidence tree, served as if it were s3://<bucket>/<prefix>
  ledger/      optional: ledger.jsonl (+ public_keys.json) of the evidence ledger

aws.json:
  {"s3": {"bucket": "fixture-bucket", "prefix": "evidence/", "last_modified": {"<rel path>": "<ISO time>"}},
   "sagemaker": [{"region": "us-east-2", "kind": "training|processing|endpoint", "name": "...",
                  "CreationTime": "...", "TrainingEndTime": "...", "LastModifiedTime": "...", ...}],
   "logs": [{"region": "us-east-2", "log_group": "/team13/checkpoint", "log_stream": "...",
             "timestamp": "<ISO>", "ingestion_time": "<ISO, optional>", "message": "...", "event_id": "..."}],
   "dynamodb": [{"region": "us-east-2", "table": "...", "item": {"case_id": "...", "run_id": "..."}}]}
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path

READ_ONLY_OPS = {"GetObject", "ListObjectsV2", "HeadObject", "StartQuery", "GetQueryResults", "GetItem",
                 "DescribeTrainingJob", "DescribeProcessingJob", "DescribeEndpoint"}


class FakeClientError(Exception):
    """Shaped like botocore.exceptions.ClientError: .response['Error']['Code']."""

    def __init__(self, code: str, message: str, operation: str, status: int = 400):
        super().__init__(f"An error occurred ({code}) when calling the {operation} operation: {message}")
        self.response = {"Error": {"Code": code, "Message": message},
                         "ResponseMetadata": {"HTTPStatusCode": status}}
        self.operation_name = operation


def _t(s: str | datetime) -> datetime:
    if isinstance(s, datetime):
        return s.astimezone(timezone.utc)
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)


class _Paginator:
    def __init__(self, fn, page_size: int):
        self.fn, self.page_size = fn, page_size

    def paginate(self, **kw):
        token = None
        while True:
            page = self.fn(MaxKeys=self.page_size, ContinuationToken=token, **kw)
            yield page
            token = page.get("NextContinuationToken")
            if not page.get("IsTruncated"):
                return


class FakeS3:
    def __init__(self, aws: "FakeAws", page_size: int = 1000):
        self.aws = aws
        self.objects: dict[tuple[str, str], dict] = {}  # (bucket, key) -> {body, last_modified, etag}
        self.page_size = page_size
        self.writes = 0

    def _rec(self, op: str) -> None:
        self.aws.calls.append(("s3", op, "global"))

    def put(self, bucket: str, key: str, body: bytes, last_modified: datetime | str) -> None:
        self.objects[(bucket, key)] = {"body": body, "last_modified": _t(last_modified),
                                       "etag": '"' + hashlib.md5(body).hexdigest() + '"'}  # noqa: S324 (S3's own ETag)

    def put_object(self, Bucket, Key, Body, IfNoneMatch=None, IfMatch=None, ContentType=None, **_):
        self._rec("PutObject")
        cur = self.objects.get((Bucket, Key))
        if IfNoneMatch == "*" and cur is not None:
            raise FakeClientError("PreconditionFailed", "At least one of the pre-conditions you specified did not hold",
                                  "PutObject", 412)
        if IfMatch is not None and (cur is None or cur["etag"] != IfMatch):
            if cur is None:
                raise FakeClientError("NoSuchKey", "The specified key does not exist.", "PutObject", 404)
            raise FakeClientError("PreconditionFailed", "At least one of the pre-conditions you specified did not hold",
                                  "PutObject", 412)
        body = Body if isinstance(Body, bytes) else Body.encode()
        self.writes += 1
        self.put(Bucket, Key, body, self.aws.now().replace(microsecond=0))  # S3 LastModified is whole seconds
        return {"ETag": self.objects[(Bucket, Key)]["etag"]}

    def get_object(self, Bucket, Key, Range=None, **_):
        self._rec("GetObject")
        o = self.objects.get((Bucket, Key))
        if o is None:
            raise FakeClientError("NoSuchKey", "The specified key does not exist.", "GetObject", 404)
        body = o["body"]
        if Range:
            m = re.fullmatch(r"bytes=(\d+)-(\d+)", Range)
            body = body[int(m.group(1)): int(m.group(2)) + 1]
        return {"Body": io.BytesIO(body), "ETag": o["etag"], "LastModified": o["last_modified"],
                "ContentLength": len(body)}

    def head_object(self, Bucket, Key, **_):
        self._rec("HeadObject")
        o = self.objects.get((Bucket, Key))
        if o is None:
            raise FakeClientError("404", "Not Found", "HeadObject", 404)
        return {"ETag": o["etag"], "LastModified": o["last_modified"], "ContentLength": len(o["body"])}

    def list_objects_v2(self, Bucket, Prefix="", MaxKeys=1000, ContinuationToken=None, **_):
        self._rec("ListObjectsV2")
        keys = sorted(k for (b, k) in self.objects if b == Bucket and k.startswith(Prefix))
        start = int(ContinuationToken or 0)
        chunk = keys[start:start + MaxKeys]
        page = {"KeyCount": len(chunk), "IsTruncated": start + MaxKeys < len(keys),
                "Contents": [{"Key": k, "LastModified": self.objects[(Bucket, k)]["last_modified"],
                              "ETag": self.objects[(Bucket, k)]["etag"],
                              "Size": len(self.objects[(Bucket, k)]["body"])} for k in chunk]}
        if page["IsTruncated"]:
            page["NextContinuationToken"] = str(start + MaxKeys)
        if not chunk:
            page.pop("Contents")
        return page

    def get_paginator(self, name: str):
        assert name == "list_objects_v2", name
        return _Paginator(self.list_objects_v2, self.page_size)


_LIKE = re.compile(r'filter\s+@message\s+like\s+"([^"\\]*)"')
_STREAM = re.compile(r'filter\s+@logStream\s*=\s*"([^"\\]*)"')


def _ts(dt: datetime) -> str:
    """Logs Insights returns @timestamp as 'YYYY-MM-DD HH:MM:SS.mmm' (UTC)."""
    return dt.strftime("%Y-%m-%d %H:%M:%S.") + f"{dt.microsecond // 1000:03d}"


class FakeLogs:
    def __init__(self, aws: "FakeAws", region: str):
        self.aws, self.region = aws, region
        self.queries: dict[str, dict] = {}

    def _rec(self, op: str) -> None:
        self.aws.calls.append(("logs", op, self.region))

    def start_query(self, startTime, endTime, queryString, logGroupName=None, logGroupNames=None, limit=None, **_):
        self._rec("StartQuery")
        groups = list(logGroupNames or []) + ([logGroupName] if logGroupName else [])
        known = {e["log_group"] for e in self.aws.events if e.get("region") == self.region}
        known |= set(self.aws.log_groups.get(self.region, []))
        for g in groups:
            if g not in known:
                raise FakeClientError("ResourceNotFoundException", f"Log group '{g}' does not exist", "StartQuery")
        qid = f"q-{len(self.aws.calls):06d}"
        self.queries[qid] = {"groups": groups, "start": startTime, "end": endTime, "q": queryString,
                             "limit": limit or 1000, "polled": 0}
        self.aws.query_log.append({"region": self.region, **self.queries[qid]})
        return {"queryId": qid}

    def get_query_results(self, queryId, **_):
        self._rec("GetQueryResults")
        q = self.queries[queryId]
        q["polled"] += 1
        if q["polled"] == 1:
            return {"status": "Running", "results": []}
        likes = _LIKE.findall(q["q"])
        streams = _STREAM.findall(q["q"])
        hits = []
        for e in self.aws.events:
            t = _t(e["timestamp"])
            if (e.get("region") != self.region or e["log_group"] not in q["groups"]
                    or not (q["start"] <= t.timestamp() <= q["end"])):
                continue
            if all(s in e["message"] for s in likes) and all(e.get("log_stream") == s for s in streams):
                hits.append((t, e))
        hits.sort(key=lambda x: x[0])
        if "stats " in q["q"]:
            if not hits:
                return {"status": "Complete", "results": []}
            row = [{"field": "first", "value": _ts(hits[0][0])}, {"field": "last", "value": _ts(hits[-1][0])},
                   {"field": "n", "value": str(len(hits))}]
            return {"status": "Complete", "results": [row]}
        rows = []
        for t, e in hits[: q["limit"]]:
            rows.append([{"field": "@timestamp", "value": _ts(t)},
                         {"field": "@ingestionTime", "value": _ts(_t(e.get("ingestion_time") or e["timestamp"]))},
                         {"field": "@log", "value": e["log_group"]},
                         {"field": "@logStream", "value": e.get("log_stream", "")},
                         {"field": "@message", "value": e["message"]},
                         {"field": "@ptr", "value": e.get("event_id", "")}])
        return {"status": "Complete", "results": rows}


class FakeSageMaker:
    def __init__(self, aws: "FakeAws", region: str):
        self.aws, self.region = aws, region

    def _find(self, kind: str, name: str, op: str) -> dict:
        self.aws.calls.append(("sagemaker", op, self.region))
        for j in self.aws.jobs:
            if j.get("region") == self.region and j["kind"] == kind and j["name"] == name:
                return {k: (_t(v) if k.endswith("Time") else v) for k, v in j.items()
                        if k not in ("region", "kind", "name")}
        raise FakeClientError("ValidationException", f"Requested resource not found: {name}", op)

    def describe_training_job(self, TrainingJobName, **_):
        return {"TrainingJobName": TrainingJobName, **self._find("training", TrainingJobName, "DescribeTrainingJob")}

    def describe_processing_job(self, ProcessingJobName, **_):
        return {"ProcessingJobName": ProcessingJobName,
                **self._find("processing", ProcessingJobName, "DescribeProcessingJob")}

    def describe_endpoint(self, EndpointName, **_):
        return {"EndpointName": EndpointName, **self._find("endpoint", EndpointName, "DescribeEndpoint")}


def _ddb(v):
    if isinstance(v, bool):
        return {"BOOL": v}
    if isinstance(v, (int, float)):
        return {"N": str(v)}
    if isinstance(v, dict):
        return {"M": {k: _ddb(x) for k, x in v.items()}}
    if isinstance(v, list):
        return {"L": [_ddb(x) for x in v]}
    return {"S": str(v)}


class FakeDynamoDB:
    def __init__(self, aws: "FakeAws", region: str):
        self.aws, self.region = aws, region

    def get_item(self, TableName, Key, **_):
        self.aws.calls.append(("dynamodb", "GetItem", self.region))
        want = {k: list(v.values())[0] for k, v in Key.items()}
        for row in self.aws.ddb:
            if row.get("region") == self.region and row["table"] == TableName and all(
                    str(row["item"].get(k)) == str(v) for k, v in want.items()):
                return {"Item": {k: _ddb(v) for k, v in row["item"].items()}}
        return {}


class FakeAws:
    """client(service, region) -> fake client. One S3 for all regions (S3 is global for reads)."""

    def __init__(self, events=None, jobs=None, ddb=None, log_groups=None, now: str | datetime | None = None):
        self.calls: list[tuple[str, str, str]] = []
        self.query_log: list[dict] = []
        self.events = list(events or [])
        self.jobs = list(jobs or [])
        self.ddb = list(ddb or [])
        self.log_groups: dict[str, list[str]] = dict(log_groups or {})
        self._now = _t(now) if now else None
        self.s3 = FakeS3(self)
        self._clients: dict[tuple[str, str], object] = {}

    def now(self) -> datetime:
        return self._now or datetime.now(timezone.utc)

    def set_now(self, t: str | datetime) -> None:
        self._now = _t(t)

    def client(self, service: str, region: str | None = None):
        if service == "s3":
            return self.s3
        region = region or "us-east-2"
        key = (service, region)
        if key not in self._clients:
            cls = {"logs": FakeLogs, "sagemaker": FakeSageMaker, "dynamodb": FakeDynamoDB}[service]
            self._clients[key] = cls(self, region)
        return self._clients[key]

    def ops(self) -> set[str]:
        return {op for _, op, _ in self.calls}

    @classmethod
    def from_fixture(cls, fixture: str | Path) -> tuple["FakeAws", str]:
        """(fake, s3 uri of the evidence prefix). Every evidence file must have a LastModified in aws.json."""
        fixture = Path(fixture)
        spec = json.loads((fixture / "aws.json").read_text())
        s3 = spec.get("s3", {})
        bucket, prefix = s3.get("bucket", "fixture-bucket"), s3.get("prefix", "evidence/")
        aws = cls(events=spec.get("logs"), jobs=spec.get("sagemaker"), ddb=spec.get("dynamodb"),
                  log_groups=spec.get("log_groups"), now=spec.get("now"))
        lm = s3.get("last_modified", {})
        root = fixture / "evidence"
        if root.is_dir():
            for p in sorted(root.rglob("*")):
                if p.is_file():
                    rel = p.relative_to(root).as_posix()
                    if rel not in lm:
                        raise ValueError(f"aws.json has no s3.last_modified for {rel}")
                    aws.s3.put(bucket, prefix + rel, p.read_bytes(), lm[rel])
        return aws, f"s3://{bucket}/{prefix}"
