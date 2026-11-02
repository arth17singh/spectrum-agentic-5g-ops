"""Read CloudWatch log rows the same way live (Logs Insights) and offline (exported JSON).

A row is a dict of Logs Insights fields: @timestamp, @message, @log, @logStream, @ptr. Offline files may be
  - the get_query_results answer: {"results": [[{"field": "@timestamp", "value": "..."}, ...], ...]}
  - a list of such rows, or a list of plain dicts with the same field names
  - a filter_log_events / get_log_events answer: {"events": [{"timestamp": ms, "message": "...", ...}]}
  - JSON lines of any of the dict shapes above.
So the first real capture can be saved and replayed by the tests with the same parser.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from t3.suite.common import to_iso

CHECKPOINT_LOG = "/team13/checkpoint"
SPANS_LOG = "aws/spans"
BEDROCK_LOG = "/team13/bedrock-invocations"
TOOLS_LOG = "/aws/lambda/team13-ran-tools"   # our tools print change_applied lines here (amendment A11)


def _row_from_cells(cells) -> dict:
    return {c["field"]: c.get("value") for c in cells if isinstance(c, dict) and "field" in c}


def _row_from_event(ev: dict, group=None) -> dict:
    return {"@timestamp": ev.get("timestamp"), "@message": ev.get("message"),
            "@logStream": ev.get("logStreamName"), "@log": ev.get("logGroupName") or group,
            "@ptr": ev.get("eventId")}


def normalise(data, group=None) -> list[dict]:
    if isinstance(data, dict):
        if "results" in data:
            return normalise(data["results"], group)
        if "events" in data:
            return [_row_from_event(e, group) for e in data["events"]]
        if "@message" in data or "@timestamp" in data:
            return [dict(data)]
        if "message" in data and "timestamp" in data:
            return [_row_from_event(data, group)]
        return []
    rows = []
    for entry in data or []:
        if isinstance(entry, list):
            rows.append(_row_from_cells(entry))
        else:
            rows.extend(normalise(entry, group))
    for r in rows:
        if group and not r.get("@log"):
            r["@log"] = group
    return rows


def load_rows(path, group=None) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except ValueError:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    return normalise(data, group)


def message_json(row: dict):
    msg = row.get("@message")
    if isinstance(msg, dict):
        return msg
    if not isinstance(msg, str):
        return None
    s = msg.strip()
    if not s.startswith("{"):  # a Lambda line may carry a prefix before the JSON
        i = s.find("{")
        if i < 0:
            return None
        s = s[i:]
    try:
        return json.loads(s)
    except ValueError:
        return None


def row_time(row: dict) -> str | None:
    return to_iso(row.get("@timestamp"))


def pointer(row: dict, group=None) -> dict:
    return {"log_group": row.get("@log") or group, "log_stream": row.get("@logStream"), "event_id": row.get("@ptr")}


def insights(logs, groups, query, start_s: float, end_s: float, sleep=time.sleep, timeout_s=120.0) -> list[dict]:
    """Run one Logs Insights query (read only) and return its rows."""
    qid = logs.start_query(logGroupNames=list(groups), startTime=int(start_s), endTime=int(end_s) + 1,
                           queryString=query, limit=10000)["queryId"]
    waited = 0.0
    while True:
        res = logs.get_query_results(queryId=qid)
        if res["status"] not in ("Scheduled", "Running"):
            break
        if waited >= timeout_s:
            raise TimeoutError(f"Logs Insights query {qid} still {res['status']} after {timeout_s} s")
        sleep(1)
        waited += 1
    if res["status"] != "Complete":
        raise RuntimeError(f"Logs Insights query {qid} ended {res['status']}")
    rows = normalise(res, groups[0] if len(groups) == 1 else None)
    if len(rows) >= 10000:
        raise RuntimeError("Logs Insights returned 10000 rows: narrow the time window, the result is partial")
    return rows


Q_CHECKPOINT = ("fields @timestamp, @message, @log, @logStream, @ptr | filter @message like /checkpoint|case_opened|"
                "change_applied|approval|model_control|tool_error/ | sort @timestamp asc | limit 10000")
Q_SPANS = ("fields @timestamp, @message, @log, @logStream, @ptr | filter @message like /authorization_decision/ "
           "| sort @timestamp asc | limit 10000")


def q_case(case_id: str) -> str:
    return (f"fields @timestamp, @message, @log, @logStream, @ptr | filter @message like \"{case_id}\" "
            "| sort @timestamp asc | limit 10000")
