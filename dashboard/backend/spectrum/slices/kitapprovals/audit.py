"""The kit's control record for a decision (KIT control_record, L970-979), plus a fail-closed outbox.

Exactly the kit: CreateLogStream "control" (ResourceAlreadyExists is fine), then one PutLogEvents with
{"ts", "by": <caller ARN>, "record": "approval", "decision", "approval_id", "case_id", "change", "reason"}.
One additive field, decision_id = "<approval_id>:<decided_at>", lets a replay check whether the line already
landed. The event timestamp is the decision time, never the replay time (t3.suite.trace orders by it).

If the line cannot be written the decision still stands (the kit has the same order: update, then log), the
response says "not_audited", and the exact event goes to <state dir>/audit-outbox.jsonl (0600). While that
file is not empty, new decisions are refused. A replay first searches the stream for decision_id and drops
the entry if it is there, so a line is never written twice. After an ambiguous failure (a timeout after
sending), the entry waits 60 s first, so CloudWatch has ingested a line that did land.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

from . import aws as awsmod
from .pin import state_dir

log = logging.getLogger("spectrum.slices.kitapprovals")
_lock = threading.Lock()
AMBIGUOUS_WAIT_S = 60


def outbox_path() -> Path:
    return state_dir() / "audit-outbox.jsonl"


def _read() -> list[dict]:
    try:
        lines = outbox_path().read_text().splitlines()
    except OSError:
        return []
    out = []
    for ln in lines:
        try:
            out.append(json.loads(ln))
        except ValueError:
            continue
    return out


def _write(entries: list[dict]) -> None:
    p = outbox_path()
    p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not entries:
        p.unlink(missing_ok=True)
        return
    tmp = p.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write("".join(json.dumps(e) + "\n" for e in entries))
    os.replace(tmp, p)


def pending_count() -> int:
    with _lock:
        return len(_read())


def control_message(by_arn: str, t: float, decision: str, approval_id: str, case_id: Any, change: Any,
                    reason: str) -> dict:
    """The kit's control-record message, field for field (KIT L975-977), plus decision_id."""
    return {"ts": t, "by": by_arn, "record": "approval", "decision": decision, "approval_id": approval_id,
            "case_id": case_id, "change": change, "reason": reason,
            "decision_id": f"{approval_id}:{int(t)}"}


def _ensure_stream(logs: Any) -> None:
    from botocore.exceptions import ClientError

    try:
        logs.create_log_stream(logGroupName=awsmod.AUDIT_LOG, logStreamName=awsmod.STREAM)
    except ClientError as exc:
        if awsmod.error_code(exc) != "ResourceAlreadyExistsException":
            raise


def _put(logs: Any, event: dict) -> None:
    _ensure_stream(logs)
    logs.put_log_events(logGroupName=awsmod.AUDIT_LOG, logStreamName=awsmod.STREAM,
                        logEvents=[{"timestamp": event["timestamp"], "message": event["message"]}])


def _landed(logs: Any, entry: dict) -> bool:
    kw = dict(logGroupName=awsmod.AUDIT_LOG, logStreamNames=[awsmod.STREAM],
              filterPattern=f'"{entry["decision_id"]}"',
              startTime=entry["timestamp"] - 1000, endTime=entry["timestamp"] + 1000)
    for _ in range(20):
        resp = logs.filter_log_events(**kw)
        if any(entry["decision_id"] in (e.get("message") or "") for e in resp.get("events", [])):
            return True
        if not resp.get("nextToken"):
            return False
        kw["nextToken"] = resp["nextToken"]
    raise RuntimeError("FilterLogEvents did not finish")


def write(logs: Any, message: dict, now=time.time) -> str:
    """'written' or 'not_audited' (queued in the outbox)."""
    event = {"decision_id": message["decision_id"], "timestamp": int(message["ts"] * 1000),
             "message": json.dumps(message)}
    try:
        _put(logs, event)
        return "written"
    except Exception as exc:  # noqa: BLE001 - the decision stands; the line is kept, not lost
        log.warning("kitapprovals: control line not written (%s); queued", awsmod.error_code(exc))
        event["not_before"] = now() + (AMBIGUOUS_WAIT_S if awsmod.maybe_landed(exc) else 0)
        with _lock:
            _write(_read() + [event])
        return "not_audited"


def drain(logs: Any, now=time.time) -> int:
    """Replay what is due, never twice. Returns how many entries remain."""
    with _lock:
        entries = _read()
        keep = []
        for e in entries:
            if e.get("not_before", 0) > now():
                keep.append(e)
                continue
            try:
                if not _landed(logs, e):
                    _put(logs, e)
            except Exception as exc:  # noqa: BLE001
                log.warning("kitapprovals: outbox replay failed (%s)", awsmod.error_code(exc))
                if awsmod.maybe_landed(exc):
                    e["not_before"] = now() + AMBIGUOUS_WAIT_S
                keep.append(e)
        _write(keep)
        return len(keep)
