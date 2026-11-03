"""In-memory stand-ins for DynamoDB, CloudWatch Logs and STS. Used by the tests and by the fixture mode
(KITAPPROVALS_FIXTURE=<rows.json>), which is labelled FIXTURE everywhere and never touches AWS.

They implement only the calls this slice makes, with the semantics that matter: the conditional update
(ConditionalCheckFailedException with ALL_OLD), Scan pagination, reserved-word refusal, a log stream that
must exist before PutLogEvents, and switchable failures.
"""

from __future__ import annotations

import copy
import json
import re
import threading
import time
from typing import Any

from botocore.exceptions import ClientError

RESERVED = {"agent", "status"}  # DynamoDB reserved words this table uses (review F16)


def _err(code: str, op: str, extra: dict | None = None) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": code}, **(extra or {})}, op)


def _ser(v: Any) -> dict:
    from boto3.dynamodb.types import TypeSerializer
    from decimal import Decimal

    if isinstance(v, float):
        v = Decimal(str(v))
    return TypeSerializer().serialize(v)


def _de(av: dict) -> Any:
    from boto3.dynamodb.types import TypeDeserializer

    return TypeDeserializer().deserialize(av)


def _check_reserved(*exprs: str | None) -> None:
    for e in exprs:
        for word in re.findall(r"(?<![#:\w])([A-Za-z_][A-Za-z0-9_]*)", e or ""):
            if word.lower() in RESERVED:
                raise _err("ValidationException", "Expression", {"Error": {"Code": "ValidationException",
                           "Message": f"Attribute name is a reserved keyword; reserved keyword: {word}"}})


class FakeDdb:
    def __init__(self, items: list[dict] | None = None, page_size: int = 2):
        self.items: dict[str, dict] = {}  # pk -> python values
        for it in items or []:
            self.items[it["pk"]] = dict(it)
        self.page_size = page_size
        self.calls: list[tuple[str, dict]] = []
        self.fail: dict[str, BaseException] = {}
        self.before_update = None  # hook to simulate a race
        self.lock = threading.Lock()

    def _maybe_fail(self, op: str) -> None:
        if op in self.fail:
            raise self.fail[op]

    def scan(self, **kw):
        self.calls.append(("Scan", kw))
        self._maybe_fail("Scan")
        _check_reserved(kw.get("FilterExpression"), kw.get("ProjectionExpression"))
        keys = sorted(self.items)
        start = kw.get("ExclusiveStartKey")
        if start:
            keys = [k for k in keys if k > _de(start["pk"])]
        page, rest = keys[: self.page_size], keys[self.page_size:]
        prefix = _de(kw["ExpressionAttributeValues"][":chg"])
        out = [self.items[k] for k in page if k.startswith(prefix) and "status" in self.items[k]]
        resp = {"Items": [{a: _ser(v) for a, v in i.items()} for i in out]}
        if rest:
            resp["LastEvaluatedKey"] = {"pk": _ser(page[-1])}
        return resp

    def get_item(self, **kw):
        self.calls.append(("GetItem", kw))
        self._maybe_fail("GetItem")
        item = self.items.get(_de(kw["Key"]["pk"]))
        return {"Item": {a: _ser(v) for a, v in item.items()}} if item else {}

    def update_item(self, **kw):
        self.calls.append(("UpdateItem", kw))
        self._maybe_fail("UpdateItem")
        _check_reserved(kw.get("UpdateExpression"), kw.get("ConditionExpression"))
        if self.before_update:
            hook, self.before_update = self.before_update, None
            hook(self)
        with self.lock:
            pk = _de(kw["Key"]["pk"])
            vals = {k: _de(v) for k, v in kw["ExpressionAttributeValues"].items()}
            item = self.items.get(pk)
            ok = item is not None and item.get("status") == vals[":pending"] and item.get("change") == vals[":change"]
            if not ok:
                extra = {"Item": {a: _ser(v) for a, v in item.items()}} if item and kw.get(
                    "ReturnValuesOnConditionCheckFailure") == "ALL_OLD" else {}
                raise _err("ConditionalCheckFailedException", "UpdateItem", extra)
            item.update({"status": vals[":s"], "decided_by": vals[":by"], "decided_at": vals[":t"],
                         "decision_reason": vals[":why"]})
            return {"Attributes": {a: _ser(v) for a, v in item.items()}}


class FakeLogs:
    def __init__(self, streams: set[str] | None = None):
        self.streams = set(streams or ())
        self.events: list[dict] = []  # {"stream", "timestamp", "message"}
        self.calls: list[tuple[str, dict]] = []
        self.fail: dict[str, BaseException] = {}
        self.land_then_fail = False  # PutLogEvents writes, then raises (a timeout after sending)

    def create_log_stream(self, **kw):
        self.calls.append(("CreateLogStream", kw))
        if "CreateLogStream" in self.fail:
            raise self.fail["CreateLogStream"]
        if kw["logStreamName"] in self.streams:
            raise _err("ResourceAlreadyExistsException", "CreateLogStream")
        self.streams.add(kw["logStreamName"])
        return {}

    def put_log_events(self, **kw):
        self.calls.append(("PutLogEvents", kw))
        if "PutLogEvents" in self.fail:
            raise self.fail["PutLogEvents"]
        if kw["logStreamName"] not in self.streams:
            raise _err("ResourceNotFoundException", "PutLogEvents")
        for e in kw["logEvents"]:
            self.events.append({"stream": kw["logStreamName"], **e})
        if self.land_then_fail:
            from botocore.exceptions import ReadTimeoutError

            self.land_then_fail = False
            raise ReadTimeoutError(endpoint_url="https://logs.fake")
        return {}

    def filter_log_events(self, **kw):
        self.calls.append(("FilterLogEvents", kw))
        if "FilterLogEvents" in self.fail:
            raise self.fail["FilterLogEvents"]
        term = kw.get("filterPattern", "").strip('"')
        hits = [e for e in self.events if e["stream"] in kw.get("logStreamNames", [e["stream"]])
                and term in e["message"] and e["timestamp"] >= kw.get("startTime", 0)]
        return {"events": [{"message": e["message"], "timestamp": e["timestamp"]} for e in hits]}


class FakeSts:
    def __init__(self, arn: str = "arn:aws:sts::111122223333:assumed-role/AWSReservedSSO_bb26-Participant_x/marker@example.com"):
        self.arn = arn
        self.fail: BaseException | None = None
        self.calls = 0

    def get_caller_identity(self):
        self.calls += 1
        if self.fail:
            raise self.fail
        return {"Arn": self.arn, "Account": self.arn.split(":")[4], "UserId": "AROAFAKE:" + self.arn.rsplit("/", 1)[-1]}


def fixture(path: str) -> tuple[FakeDdb, FakeLogs, FakeSts]:
    """Fake services seeded from a JSON file: {"items": [...]}; created_at may be negative = seconds ago."""
    with open(path) as fh:
        data = json.load(fh)
    now = int(time.time())
    items = []
    for it in copy.deepcopy(data.get("items", [])):
        for k in ("created_at", "decided_at", "applied_at"):
            if isinstance(it.get(k), int) and it[k] <= 0:
                it[k] = now + it[k]
        if isinstance(it.get("doc"), (dict, list)):
            it["doc"] = json.dumps(it["doc"])
        items.append(it)
    return FakeDdb(items, page_size=50), FakeLogs({"control"}), FakeSts()
