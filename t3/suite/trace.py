"""Rebuild one case's trace from AWS logs and check it against the declared expected sequence.

Sources (Logs Insights, read only; or exported JSON offline): /team13/checkpoint (case_opened, every Gateway
call at request and response, approvals, pauses), /aws/lambda/team13-ran-tools (our tools' change_applied
lines, amendment A11), aws/spans (Cedar decisions that mention the
case) and /team13/bedrock-invocations (the agents' Bedrock calls, found by the case id in the prompt).
Events are ordered by the CloudWatch event timestamp.

Expected sequence (declared before the case starts, hashed into test_set_sha256):
  case_opened -> ask_agent (allowed) -> tool call by a specialist (allowed) -> approval (by a person)
  -> change_applied (same approval id) -> observe (allowed, after the change)

  drive a live case:  python -m t3.suite.trace drive --run-id RUN --request "..." [--live]
  rebuild + check:    python -m t3.suite.trace check --case case-1a2b3c4d [--run-id RUN]
  offline:            python -m t3.suite.trace check --case ... --checkpoint-json cp.json \
                          [--tools-json t.json] [--spans-json s.json] [--bedrock-json b.json] [--no-write]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from t3.suite import common, logsrc
from t3.suite.check import parse_span
from t3.suite.common import RUNNING, envelope, epoch, sha256_json, to_iso, utc_now_iso

SCRIPT = "t3/suite/trace.py"
UNIT = "t3.trace"
EXPECTED = ["case_opened", "ask_agent", "specialist_tool", "approval", "change_applied", "observe"]
SPECIALISTS = ("ran_monitor", "ran_optimizer", "ran_explainer")
FRONT = ("front___start_case", "front___continue_case", "front___case_status")


def classify(cp_rows, span_rows=(), bedrock_rows=(), case_id=None, tool_rows=()) -> list[dict]:
    events = []
    pending = {}  # (agent, tool) -> request times, to give each response its start time
    tagged = [(r, logsrc.CHECKPOINT_LOG) for r in cp_rows] + [(r, logsrc.TOOLS_LOG) for r in tool_rows]
    tagged.sort(key=lambda rg: epoch(rg[0]["@timestamp"]))  # AWS time order before pairing
    for order, (row, group) in enumerate(tagged):
        msg = logsrc.message_json(row) or {}
        if case_id and case_id not in json.dumps(msg) and case_id not in str(row.get("@message", "")):
            continue
        t = logsrc.row_time(row)
        rec = msg.get("record")
        if group == logsrc.TOOLS_LOG and rec != "change_applied":
            continue  # from the tools' log only the change_applied lines count
        base = {"time": t, "t": epoch(row["@timestamp"]), "order": order, "source": group,
                "pointer": logsrc.pointer(row, group)}
        if rec == "case_opened":
            events.append({**base, "kind": "case_opened", "detail": {"requested_by": msg.get("requested_by")}})
        elif rec == "checkpoint" and msg.get("method") in (None, "tools/call") and msg.get("tool"):
            key = (msg.get("agent"), msg.get("tool"))
            if msg.get("phase") == "request":
                pending.setdefault(key, []).append(base["t"])
                continue
            start = pending.get(key, []).pop(0) if pending.get(key) else base["t"]
            tool = msg["tool"]
            ev = {**base, "time": to_iso(start), "end_time": base["time"], "t": start, "end": base["t"],
                  "agent": msg.get("agent"), "tool": tool,
                  "decision": "DENY" if msg.get("refused") else "ALLOW",
                  "detail": {"approval_id": msg.get("approval_id"), "approval": msg.get("approval")}}
            if tool.endswith("___ask_agent"):
                ev["kind"] = "ask_agent"
                ev["detail"]["to"] = (msg.get("arguments") or {}).get("agent")
            elif tool.endswith("___apply_change"):
                ev["kind"] = "apply_call"
            elif tool in FRONT:
                ev["kind"] = "front_door"
            else:
                ev["kind"] = "tool_call"
            events.append(ev)
        elif rec == "approval":
            events.append({**base, "kind": "approval", "decision": msg.get("decision"),
                           "detail": {"approval_id": msg.get("approval_id"), "by": msg.get("by")}})
        elif rec == "change_applied":
            events.append({**base, "kind": "change_applied",
                           "detail": {"approval_id": msg.get("approval_id"), "item_id": msg.get("item_id"),
                                      "run_id": msg.get("run_id"), "change_text": msg.get("change_text")}})
        elif rec in ("model_control", "tool_error"):
            events.append({**base, "kind": rec, "detail": {k: msg.get(k) for k in ("action", "policy", "error")}})
    for order, row in enumerate(span_rows):
        sp = parse_span(row)
        if sp and (not case_id or case_id in str(row.get("@message", ""))):
            events.append({"time": sp["time"], "t": sp["t"], "order": order, "source": logsrc.SPANS_LOG,
                           "kind": "cedar_decision", "decision": sp["decision"], "tool": sp["tool"],
                           "detail": {"policies": sp["policies"], "trace_id": sp["trace_id"]},
                           "pointer": sp["pointer"]})
    for order, row in enumerate(bedrock_rows):
        if case_id and case_id not in str(row.get("@message", "")):
            continue
        msg = logsrc.message_json(row) or {}
        events.append({"time": logsrc.row_time(row), "t": epoch(row["@timestamp"]), "order": order,
                       "source": logsrc.BEDROCK_LOG, "kind": "bedrock_call",
                       "detail": {"model_id": msg.get("modelId")}, "pointer": logsrc.pointer(row, logsrc.BEDROCK_LOG)})
    rank = {logsrc.CHECKPOINT_LOG: 0, logsrc.TOOLS_LOG: 1, logsrc.SPANS_LOG: 2, logsrc.BEDROCK_LOG: 3}
    events.sort(key=lambda e: (e["t"], rank[e["source"]], e["order"]))
    return events


def check_sequence(events, expected=EXPECTED) -> dict:
    """Find each expected stage in order; a stage counts only after the previous stage's time."""
    found, after, approval_id = [], float("-inf"), None
    tests = {
        "case_opened": lambda e: e["kind"] == "case_opened",
        "ask_agent": lambda e: e["kind"] == "ask_agent" and e["decision"] == "ALLOW",
        "specialist_tool": lambda e: e["kind"] == "tool_call" and e["decision"] == "ALLOW"
        and e.get("agent") in SPECIALISTS,
        "approval": lambda e: e["kind"] == "approval" and e.get("decision") == "APPROVED",
        "change_applied": lambda e: e["kind"] == "change_applied"
        and (approval_id is None or e["detail"]["approval_id"] == approval_id),
        "observe": lambda e: e["kind"] == "tool_call" and e["decision"] == "ALLOW"
        and str(e.get("tool", "")).endswith("___observe"),
    }
    for stage in expected:
        hit = next((e for e in events if e["t"] >= after and tests[stage](e)), None)
        found.append({"stage": stage, "found": hit is not None, "time": hit["time"] if hit else None,
                      "agent": hit.get("agent") if hit else None, "tool": hit.get("tool") if hit else None})
        if hit is None:
            break
        after = hit["t"]
        if stage == "approval":
            approval_id = hit["detail"]["approval_id"]
    missing = [s for s in expected if s not in {f["stage"] for f in found if f["found"]}]
    refused_applies = sum(1 for e in events if e["kind"] == "apply_call" and e["decision"] == "DENY")
    return {"stages": found, "missing": missing, "passed": not missing, "approval_id": approval_id,
            "refused_apply_calls": refused_applies,
            "counts": {k: sum(1 for e in events if e["kind"] == k) for k in sorted({e["kind"] for e in events})}}


def rebuild(case_id, cp_rows, span_rows=(), bedrock_rows=(), expected=EXPECTED, tool_rows=()) -> dict:
    events = classify(cp_rows, span_rows, bedrock_rows, case_id, tool_rows)
    verdict = check_sequence(events, expected)
    return {"case_id": case_id, "expected": expected, **verdict,
            "events": [{k: v for k, v in e.items() if k not in ("t", "order", "end")} for e in events]}


def declaration(run_id, cfg, synthetic=False):
    return envelope("declared", run_id, synthetic=synthetic, unit=UNIT, track="track-3",
                    target={"metric": "sequence_complete", "op": "==", "value": 1.0, "unit": "ratio"},
                    n_items=1, n_cases=1, test_set_sha256=sha256_json({"expected": EXPECTED}),
                    model_id=cfg["model_id"],
                    grader=f"{SCRIPT}: the case's events from CloudWatch contain, in time order, " + " -> ".join(EXPECTED),
                    code_sha=common.code_sha(), region=cfg["region"],
                    twin={"kind": "maveric-inprocess", "version": "n/a", "scenario": "n/a", "seed": 0},
                    declared_at=utc_now_iso())


def drive(runner, request, *, status_timeout_s=600.0, poll_s=5.0, sleep=time.sleep, clock=time.time, log=print):
    """Declare the trace unit, start a real case through the front door, wait for the person's approval,
    say "Approved.", and wait for the lead to finish. Records every front-door call."""
    runner.ev.write_declared(declaration(runner.run_id, runner.cfg, runner.synthetic))
    case_id = runner.start_case(unit=UNIT, request_text=request)

    def wait_done(turn):
        deadline = clock() + status_timeout_s
        while True:
            sleep(poll_s)
            out = runner.gw.call("team_app", "front___case_status", {"case_id": case_id}, timeout=60)
            status = json.loads(out["text"]) if out["decision"] == "ALLOW" else {"state": "REFUSED"}
            if status.get("state") not in RUNNING or clock() >= deadline:
                runner._record(f"trace-status-{turn}", "team_app", "front___case_status", {"case_id": case_id},
                               out, {"turn": turn})
                return status

    status = wait_done(1)
    log(f"turn 1: {status.get('state')} {str(status.get('reply'))[:300]}")
    waiting = status.get("waiting_approvals") or []
    if not waiting:
        log("no change is waiting for approval: the lead did not call apply_change")
        return {"case_id": case_id, "approved": None, "status": status}
    approval_id = waiting[0]["approval_id"]
    decision = runner.wait_for_person(approval_id)
    if decision["status"] != "APPROVED":
        return {"case_id": case_id, "approved": decision, "status": status}
    msg = {"case_id": case_id, "message": "Approved."}
    out = runner.gw.call("team_app", "front___continue_case", msg, timeout=60)
    runner._record("trace-continue-2", "team_app", "front___continue_case", msg, out, {"turn": 2})
    status = wait_done(2)
    log(f"turn 2: {status.get('state')} {str(status.get('reply'))[:300]}")
    return {"case_id": case_id, "approved": decision, "status": status}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Rebuild and check one case's trace.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("drive")
    d.add_argument("--run-id", required=True)
    d.add_argument("--request", required=True)
    d.add_argument("--tokens-dir")
    d.add_argument("--live", action="store_true")
    c = sub.add_parser("check")
    c.add_argument("--case", required=True)
    c.add_argument("--run-id", help="write item + result records for this run")
    c.add_argument("--checkpoint-json")
    c.add_argument("--spans-json")
    c.add_argument("--bedrock-json")
    c.add_argument("--tools-json")
    c.add_argument("--hours", type=float, default=24.0, help="live: how far back to look")
    c.add_argument("--out", help="write the rebuilt trace JSON here")
    c.add_argument("--no-write", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "drive":
        if not a.live:
            print(f"DRY RUN: would declare {UNIT} for {a.run_id} (expected {' -> '.join(EXPECTED)}), start a case "
                  f"through the front door with run_id in the request, wait for a person to approve, say "
                  f"'Approved.', wait for the lead. Then: python -m t3.suite.trace check --case <case> "
                  f"--run-id {a.run_id}")
            return 0
        from t3.suite.runner import build
        out = drive(build(a.run_id, tokens_dir=a.tokens_dir, pause=False), a.request)
        print(json.dumps(out, indent=1, default=str))
        return 0
    if a.checkpoint_json:
        cp = logsrc.load_rows(a.checkpoint_json, logsrc.CHECKPOINT_LOG)
        sp = logsrc.load_rows(a.spans_json, logsrc.SPANS_LOG) if a.spans_json else []
        bd = logsrc.load_rows(a.bedrock_json, logsrc.BEDROCK_LOG) if a.bedrock_json else []
        tl = logsrc.load_rows(a.tools_json, logsrc.TOOLS_LOG) if a.tools_json else []
    else:
        import boto3
        logs = boto3.client("logs", region_name=os.environ.get("AWS_REGION", "us-east-2"))
        end = time.time()
        start = end - a.hours * 3600
        q = logsrc.q_case(a.case)
        cp = logsrc.insights(logs, [logsrc.CHECKPOINT_LOG], q, start, end)
        sp = logsrc.insights(logs, [logsrc.SPANS_LOG], q, start, end)
        bd = logsrc.insights(logs, [logsrc.BEDROCK_LOG], q, start, end)
        tl = logsrc.insights(logs, [logsrc.TOOLS_LOG], q, start, end)
    out = rebuild(a.case, cp, sp, bd, tool_rows=tl)
    for e in out["events"]:
        print(f"{e['time']}  {e['kind']:<15} {e.get('agent') or '':<14} {e.get('tool') or '':<24} "
              f"{e.get('decision') or ''}")
    for s in out["stages"]:
        print(f"{'FOUND ' if s['found'] else 'MISSING'} {s['stage']:<16} {s['time'] or ''}")
    print("PASS" if out["passed"] else f"FAIL: missing {', '.join(out['missing'])}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)
    if a.run_id and not a.no_write:
        ev = common.load_evlib()
        first = out["events"][0]["time"] if out["events"] else None
        ev.append_records(a.run_id, "item", [envelope(
            "item", a.run_id, aws_time=first, aws_time_source=f"cloudwatch:{logsrc.CHECKPOINT_LOG}",
            item_id=a.case, input_sha256=sha256_json({"case_id": a.case, "expected": EXPECTED}),
            output_sha256=sha256_json(out["events"]), score=1.0 if out["passed"] else 0.0, passed=out["passed"],
            grader=SCRIPT, backend="cloudwatch-logs-insights",
            extra={"view": "aws", "stages": out["stages"], "counts": out["counts"]})])
        ev.write_result(envelope(
            "result", a.run_id, unit=UNIT, metric="sequence_complete", value=1.0 if out["passed"] else 0.0,
            target={"metric": "sequence_complete", "op": "==", "value": 1.0, "unit": "ratio"},
            passed=out["passed"], n_items=1, checked_at=utc_now_iso(),
            checker={"script": SCRIPT, "code_sha": common.code_sha()},
            details={"case_id": a.case, "missing": out["missing"], "counts": out["counts"]}))
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
