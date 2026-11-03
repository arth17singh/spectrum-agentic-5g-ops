"""Grade a suite run from AWS's own decision records, not from what the runner saw.

For each item the runner sent (its `item` records, view "client"), find the interceptor's request record of
that call in /team13/checkpoint (agent, tool and the item's own case id, amendment A12), the response record
with the same case id (refused, error), and the Cedar authorization_decision span in aws/spans between them. The observed decision is DENY when the call was
refused, else ALLOW. An item passes when the observed decision equals the suite's `expect` and, where a span
was joined, the span agrees. Times come from the CloudWatch event timestamps.

Writes `decision` and `item` records (view "aws") and one `result`; exits 0 only when every item matches.

Each unit (deny, allow; t3/suite/units.py) is graded on its own run id against its own subset of the suite.

  live:    python -m t3.suite.check --run-id RUN [--unit deny|allow|all] [--spans required|optional|off]
  offline: python -m t3.suite.check --run-id RUN --runner-records item.jsonl \
               --checkpoint-json cp.json --spans-json spans.json [--no-write]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from t3.suite import common, logsrc, units
from t3.suite.common import envelope, epoch, sha256_text, to_iso, utc_now_iso
from t3.suite.suite_hash import DEFAULT_SUITE, canonical, load_suite, suite_sha256

SCRIPT = "t3/suite/check.py"
SKEW_S = 5.0          # our clock vs AWS's, allowed either way
SPAN_JOIN_S = 10.0    # a span belongs to the checkpoint record nearest in time, within this window
RULE_RE = re.compile(r"denied due to ([A-Za-z0-9_]+)")


# ---------- normalising what the logs say ----------
def norm(value):
    """Numbers as the Gateway may re-serialise them: 1.0 and 1 compare equal."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, list):
        return [norm(v) for v in value]
    if isinstance(value, dict):
        return {k: norm(v) for k, v in value.items()}
    return value


def args_key(args) -> str:
    return canonical(norm({k: v for k, v in (args or {}).items() if k not in ("approval", "special_context")}))


def deciding_rule(text: str) -> str | None:
    if not text:
        return None
    m = RULE_RE.search(text)
    if m:
        return m.group(1)
    if "No policy applies" in text or "denied by default" in text:
        return "default_deny"
    if "already applied" in text or "already rejected" in text:
        return "approval_already_used"
    if "needs a person's approval" in text:
        return "require_human_approval"
    return None


def checkpoint_calls(rows) -> list[dict]:
    """Every interceptor record of a tools/call, request and response phase, in AWS time order."""
    out = []
    for i, row in enumerate(rows):
        msg = logsrc.message_json(row)
        if not msg or msg.get("record") != "checkpoint" or msg.get("phase") not in ("request", "response"):
            continue
        if msg.get("method") not in (None, "tools/call"):
            continue
        err = json.dumps(msg.get("error")) if msg.get("error") else ""
        out.append({"i": i, "phase": msg["phase"], "time": logsrc.row_time(row), "t": epoch(row["@timestamp"]),
                    "agent": msg.get("agent"), "tool": msg.get("tool"), "case_id": msg.get("case_id"),
                    "args_key": args_key(msg.get("arguments")), "refused": bool(msg.get("refused")),
                    "decision": "DENY" if msg.get("refused") else "ALLOW", "error": err,
                    "rule": deciding_rule(err) or (("approval_" + str(msg.get("approval")).lower())
                                                   if msg.get("approval") in ("APPLIED", "REJECTED") else None),
                    "approval_id": msg.get("approval_id"), "approval": msg.get("approval"),
                    "pointer": logsrc.pointer(row, logsrc.CHECKPOINT_LOG), "raw": row.get("@message")})
    out.sort(key=lambda c: (c["t"], c["i"]))
    return out


def checkpoint_responses(rows) -> list[dict]:
    return [c for c in checkpoint_calls(rows) if c["phase"] == "response"]


def _flatten(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _flatten(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _flatten(v, f"{prefix}[{i}]")
    else:
        yield prefix, obj


def _decision_word(value) -> str | None:
    v = str(value or "").strip().upper()
    if v in ("ALLOW", "ALLOWED", "PERMIT", "PERMITTED"):
        return "ALLOW"
    if v in ("DENY", "DENIED", "FORBID", "FORBIDDEN"):
        return "DENY"
    return None


def parse_span(row) -> dict | None:
    """Tolerant reader of an aws/spans authorization_decision span: the exact attribute layout is
    confirmed on the first real capture, so look for the documented names anywhere in the span."""
    msg = logsrc.message_json(row)
    if not msg:
        return None
    flat = list(_flatten(msg))
    decision = reason = policies = tool = trace_id = span_id = None
    guardrail = {}
    for key, value in flat:
        low = key.lower()
        if low.endswith("authorization_decision"):
            decision = _decision_word(value) or decision
        elif low.endswith("authorization_reason"):
            reason = value
        elif "determining_policies" in low:
            policies = value if policies is None else f"{policies},{value}"
        elif "guardrail" in low and "score" in low:
            guardrail[key] = value
        elif low in ("traceid", "trace_id") or low.endswith(".traceid"):
            trace_id = value
        elif low in ("spanid", "span_id") or low.endswith(".spanid"):
            span_id = value
        if isinstance(value, str) and "___" in value and tool is None and re.fullmatch(r"[a-z0-9_-]+___[a-z0-9_]+", value):
            tool = value
    if decision is None:
        return None
    return {"time": logsrc.row_time(row), "t": epoch(row["@timestamp"]), "decision": decision, "reason": reason,
            "policies": policies, "tool": tool, "trace_id": trace_id, "span_id": span_id,
            "guardrail_scores": guardrail or None, "pointer": logsrc.pointer(row, logsrc.SPANS_LOG)}


# ---------- matching ----------
def runner_items(records) -> list[dict]:
    items = [r for r in records if r.get("kind", "item") in ("item", "call")
             and (r.get("extra") or {}).get("view") == "client" and not (r.get("extra") or {}).get("setup")]
    return sorted(items, key=lambda r: r["extra"]["seq"])


def _same_call(c, ex, key):
    """The checkpoint record of this item's call: same agent and tool, and the item's own case id (A12);
    items without a case id are told apart by their arguments."""
    if c["agent"] != ex["caller"] or c["tool"] != ex["tool"]:
        return False
    case = (ex.get("args") or {}).get("case_id")
    return c["case_id"] == case if case else c["args_key"] == key


def match(items, calls, spans, spans_mode="optional") -> list[dict]:
    """Amendment A12: runner item -> the interceptor's request record (agent, tool, case id, after the send)
    -> the response with the same case id; the Cedar span joins the request by time (and tool, when the
    span names it), between the request and the response."""
    requests = [c for c in calls if c["phase"] == "request"]
    responses = [c for c in calls if c["phase"] == "response"]
    used, used_sp, out = set(), set(), []
    for rec in items:
        ex = rec["extra"]
        res = {"item_id": rec["item_id"], "expect": ex.get("expect"), "rule_class": ex.get("rule_class"),
               "caller": ex["caller"], "tool": ex["tool"], "case_id": ex.get("case_id"),
               "input_sha256": rec.get("input_sha256"), "observed": None, "passed": False, "match": None,
               "span": None, "problems": []}
        if ex.get("skipped"):
            res["problems"].append(f"not run: {ex['skipped']}")
            out.append(res)
            continue
        sent = epoch(ex["sent_at"]) if ex.get("sent_at") else float("-inf")
        key = args_key(ex.get("args"))
        req = next((c for c in requests if c["i"] not in used and c["t"] >= sent - SKEW_S
                    and _same_call(c, ex, key)), None)
        start = req["t"] if req else sent - SKEW_S
        resp = next((c for c in responses if c["i"] not in used and c["t"] >= start
                     and _same_call(c, ex, key)), None)
        if resp is None:
            res["problems"].append("no checkpoint response record for this call")
            out.append(res)
            continue
        used.add(resp["i"])
        if req:
            used.add(req["i"])
        res.update(match="request+response" if req else "response-only", observed=resp["decision"],
                   aws_time=resp["time"], request_time=req["time"] if req else None,
                   deciding_rule=resp["rule"], approval_id=resp["approval_id"] or (req or {}).get("approval_id"),
                   log_pointer=resp["pointer"],
                   output_sha256=sha256_text(resp["raw"] if isinstance(resp["raw"], str) else json.dumps(resp["raw"])))
        if spans_mode != "off":
            lo, hi = start - 1.0, max(resp["t"], start + SPAN_JOIN_S) + 1.0
            near = [s for s in spans if id(s) not in used_sp and s["tool"] in (None, ex["tool"]) and lo <= s["t"] <= hi]
            if near:
                sp = min(near, key=lambda s: (s["tool"] is None, abs(s["t"] - start)))
                used_sp.add(id(sp))
                res["span"] = {k: sp[k] for k in ("time", "decision", "reason", "policies", "trace_id", "span_id",
                                                   "guardrail_scores", "pointer")}
                if sp["decision"] != resp["decision"]:
                    res["problems"].append(f"sources disagree: checkpoint {resp['decision']}, span {sp['decision']}")
            elif spans_mode == "required":
                res["problems"].append("no authorization_decision span between the request and the response")
        if resp["decision"] != res["expect"]:
            res["problems"].append(f"expected {res['expect']}, AWS recorded {resp['decision']}")
        res["passed"] = not res["problems"]
        out.append(res)
    return out


# ---------- "allowed" must mean "applied to our twin" ----------
APPLIED_SLACK_S = 120.0   # how long after the last call a change_applied line still belongs to the run


def tool_changes(rows) -> list[dict]:
    """change_applied lines written by OUR tools Lambda (/aws/lambda/team13-ran-tools, source team13-ran-tools).
    A line from anywhere else (the kit's own apply_change logs to /team13/checkpoint) does not count."""
    out = []
    for i, row in enumerate(rows):
        msg = logsrc.message_json(row)
        if not msg or msg.get("record") != "change_applied":
            continue
        group = str(row.get("@log") or "")
        if not (group.endswith("team13-ran-tools") and msg.get("source") == "team13-ran-tools"):
            continue
        out.append({"i": i, "t": epoch(row["@timestamp"]), "time": logsrc.row_time(row),
                    "case_id": msg.get("case_id"), "item_id": msg.get("item_id"),
                    "approval_id": msg.get("approval_id"), "change_text": msg.get("change_text"),
                    "run_id": msg.get("run_id"), "pointer": logsrc.pointer(row, logsrc.TOOLS_LOG)})
    return out


def run_window(items) -> tuple[float, float]:
    sent = [epoch(r["extra"]["sent_at"]) for r in items if r["extra"].get("sent_at")]
    ends = [epoch(r["extra"]["sent_at"]) + (r.get("latency_ms") or 0) / 1000 for r in items
            if r["extra"].get("sent_at")]
    if not sent:
        return float("inf"), float("-inf")
    return min(sent) - SKEW_S, max(ends) + APPLIED_SLACK_S


def require_applied(results, items, tool_rows) -> None:
    """An ALLOW on ran___apply_change passes only with a change_applied line from our tools Lambda for the same
    case, item and approval key (and change text), with its AWS time inside the run window and not before
    the call. Otherwise the approval may have been consumed without touching our twin."""
    changes = tool_changes(tool_rows)
    lo, hi = run_window(items)
    by_id = {r["item_id"]: r for r in items}
    used = set()
    for res in results:
        if not (str(res["tool"]).endswith("___apply_change") and res["expect"] == "ALLOW"
                and res.get("observed") == "ALLOW"):
            continue
        args = (by_id.get(res["item_id"]) or {}).get("extra", {}).get("args") or {}
        key = common.change_key(args)
        start = epoch(res["request_time"]) - SKEW_S if res.get("request_time") else lo
        hit = next((c for c in changes if c["i"] not in used and c["case_id"] == args.get("case_id")
                    and c["item_id"] == args.get("item_id") and c["approval_id"] == key
                    and c["change_text"] in (None, args.get("change")) and lo <= c["t"] <= hi and c["t"] >= start),
                   None)
        if hit is None:
            res["problems"].append("allowed but not applied: no change_applied line from team13-ran-tools for "
                                   f"{args.get('case_id')} {args.get('item_id')} {key} in the run window")
            res["passed"] = False
        else:
            used.add(hit["i"])
            res["applied"] = {k: hit[k] for k in ("time", "approval_id", "change_text", "run_id", "pointer")}


# ---------- records ----------
def _holds(value: float, target: dict) -> bool:
    return {">=": value >= target["value"], "<=": value <= target["value"], "==": value == target["value"],
            ">": value > target["value"], "<": value < target["value"]}[target["op"]]


def build_records(run_id, results, declared, n_suite, code_sha, synthetic=False, unit_key=None):
    decisions, items = [], []
    for r in results:
        if r.get("observed"):
            decisions.append(envelope(
                "decision", run_id, aws_time=r["aws_time"], aws_time_source=f"cloudwatch:{logsrc.CHECKPOINT_LOG}",
                synthetic=synthetic, case_id=r["case_id"] or "(missing)", trace_id=(r["span"] or {}).get("trace_id"),
                span_id=(r["span"] or {}).get("span_id"), time=r["aws_time"], agent=r["caller"], tool=r["tool"],
                decision=r["observed"], deciding_rule=r.get("deciding_rule") or (r["span"] or {}).get("policies"),
                reason=(r["span"] or {}).get("reason"), guardrail_scores=(r["span"] or {}).get("guardrail_scores"),
                approval_id=r.get("approval_id"), source=logsrc.CHECKPOINT_LOG, log_pointer=r.get("log_pointer"),
                item_id=r["item_id"]))
        items.append(envelope(
            "item", run_id, aws_time=r.get("aws_time"),
            aws_time_source=f"cloudwatch:{logsrc.CHECKPOINT_LOG}" if r.get("aws_time") else None,
            synthetic=synthetic, item_id=r["item_id"], input_sha256=r["input_sha256"],
            output_sha256=r.get("output_sha256") or sha256_text(""), score=1.0 if r["passed"] else 0.0,
            passed=r["passed"], grader=SCRIPT, backend="agentcore-gateway",
            extra={"view": "aws", "expect": r["expect"], "observed": r["observed"], "rule_class": r["rule_class"],
                   "deciding_rule": r.get("deciding_rule"), "match": r["match"], "span": r["span"],
                   "problems": r["problems"], "caller": r["caller"], "tool": r["tool"], "case_id": r["case_id"],
                   "applied": r.get("applied")}))
    n_pass = sum(r["passed"] for r in results)
    by_expect = {}
    for want in ("DENY", "ALLOW"):
        sub = [r for r in results if r["expect"] == want]
        by_expect[want] = {"n": len(sub), "matched": sum(r["passed"] for r in sub)}
    by_class = {}
    for r in results:
        c = by_class.setdefault(r["rule_class"], {"n": 0, "matched": 0})
        c["n"] += 1
        c["matched"] += int(r["passed"])
    value = n_pass / len(results) if results else 0.0
    target = (declared or {}).get("target") or {"metric": "pass_rate", "op": ">=", "value": 1.0, "unit": "ratio"}
    complete = len(results) == n_suite and (declared is None or declared.get("n_items") == n_suite)
    result = envelope(
        "result", run_id, synthetic=synthetic,
        unit=units.UNITS[unit_key]["unit"] if unit_key else (declared or {}).get("unit"),
        metric=target["metric"], value=value, target=target,
        # as tools/check_unit.py recomputes it from the item records (an incomplete run fails on n_items there)
        passed=bool(results) and _holds(value, target), n_items=len(results), checked_at=utc_now_iso(),
        checker={"script": SCRIPT, "code_sha": code_sha},
        details={"subset": units.UNITS[unit_key]["subset"] if unit_key else None, "by_expect": by_expect,
                 "by_rule_class": by_class, "unit_items": n_suite,
                 "complete": complete, "spans_joined": sum(1 for r in results if r["span"]),
                 "mismatches": [{"item_id": r["item_id"], "problems": r["problems"]} for r in results
                                if not r["passed"]]})
    return decisions, items, result


# ---------- inputs ----------
def evidence_dir(arg=None) -> Path:
    return Path(arg or os.environ.get("SPECTRUM_EVIDENCE_DIR") or Path(__file__).resolve().parents[2] / "evidence")


def read_jsonl(path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_records(run_id, kind, ev_dir) -> list[dict]:
    """Both storage layouts (amendment A11): records/<run>/<kind>.jsonl and records/<run>/<kind>/<key>.json."""
    base = Path(ev_dir) / "records" / run_id
    recs = read_jsonl(base / f"{kind}.jsonl")
    folder = base / kind
    if folder.is_dir():
        recs += [json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("*.json"))]
    return recs


def load_calls(run_id, path=None, ev_dir=None) -> list[dict]:
    """The runner's client-view call log: an explicit file, else this machine's copy (common.calls_path),
    else the copy stored with the evidence (outputs/<run_id>/calls.jsonl), else old runs' item records."""
    if path:
        return read_jsonl(path)
    local = common.calls_path(run_id)
    if local.exists():
        return read_jsonl(local)
    try:
        ev = common.load_evlib()
        body = ev.default_store().get(common.calls_rel(run_id))
        if body:
            return [json.loads(line) for line in body.decode().splitlines() if line.strip()]
    except Exception:  # noqa: BLE001 - no store configured: fall through
        pass
    return read_records(run_id, "item", ev_dir) if ev_dir else []


def _store_declared(run_id) -> dict | None:
    """The declaration in the configured evidence store (EVIDENCE_S3 / EVIDENCE_DIR), if any."""
    try:
        return common.load_evlib().read_declared(run_id)
    except Exception:  # noqa: BLE001 - no store configured
        return None


def read_declared(run_id, ev_dir) -> dict | None:
    p = Path(ev_dir) / "declared" / f"{run_id}.json"
    return json.loads(p.read_text()) if p.exists() else None


def check(run_id, records, cp_rows, span_rows, *, suite_path=DEFAULT_SUITE, declared=None, spans_mode="optional",
          evlib=None, synthetic=False, log=print, unit=None, tool_rows=None) -> int:
    """unit: "deny" or "allow"; when None it comes from the declaration's unit. tool_rows: the tools Lambda's
    log (/aws/lambda/team13-ran-tools); an allowed apply passes only with its change_applied line there."""
    suite = load_suite(suite_path)
    if declared and declared.get("test_set_sha256") != suite_sha256(suite):
        log("FAIL the declared suite hash differs from the suite file")
        return 1
    key = unit or units.BY_NAME.get((declared or {}).get("unit"))
    if key not in units.UNITS:
        log("FAIL no unit: pass --unit deny|allow or keep the run's declaration next to the records")
        return 1
    if declared and declared.get("unit") != units.UNITS[key]["unit"]:
        log(f"FAIL the run was declared as {declared.get('unit')}, not {units.UNITS[key]['unit']}")
        return 1
    subset = units.select(suite, key)
    wanted = {it["id"] for it in subset}
    items = runner_items(records)
    calls = checkpoint_calls(cp_rows)
    spans = [s for s in (parse_span(r) for r in span_rows) if s]
    results = match(items, calls, spans, spans_mode)
    require_applied(results, items, tool_rows or [])
    for r in results:  # an item outside the unit's subset cannot count for it
        if r["item_id"] not in wanted:
            r["problems"].append(f"not in the {units.UNITS[key]['unit']} subset")
            r["passed"] = False
    decisions, item_recs, result = build_records(run_id, results, declared, len(subset), common.code_sha(),
                                                 synthetic=synthetic, unit_key=key)
    for r in results:
        log(f"{'PASS' if r['passed'] else 'FAIL'}  {r['item_id']} {r['caller']:<13} {r['tool']:<22} "
            f"expect {r['expect']:<5} aws {str(r['observed']):<5} {r.get('deciding_rule') or ''} "
            f"{'; '.join(r['problems'])}")
    log(f"{result['unit']} {run_id}: {sum(r['passed'] for r in results)} of {len(results)} items match "
        f"({len(subset)} in the unit); spans joined: {result['details']['spans_joined']}")
    if evlib is not None:
        if decisions:
            evlib.append_records(run_id, "decision", decisions)
        evlib.append_records(run_id, "item", item_recs)
        evlib.write_result(result)
    return 0 if result["passed"] else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Grade a Track 3 suite run from AWS decision records.")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--unit", choices=("all", "deny", "allow"),
                    help="all = <run-id>-deny and <run-id>-allow; default: the run's declared unit")
    ap.add_argument("--suite", default=str(DEFAULT_SUITE))
    ap.add_argument("--evidence-dir")
    ap.add_argument("--runner-records", help="the runner's item.jsonl (default: evidence/records/<run>/item.jsonl)")
    ap.add_argument("--checkpoint-json", help="offline: exported /team13/checkpoint rows")
    ap.add_argument("--spans-json", help="offline: exported aws/spans rows")
    ap.add_argument("--tools-json", help="offline: exported /aws/lambda/team13-ran-tools rows")
    ap.add_argument("--spans", choices=("required", "optional", "off"), default="optional")
    ap.add_argument("--no-write", action="store_true", help="print only; write no records")
    ap.add_argument("--save-rows", help="live: save the queried rows here (a folder) for offline replay")
    ap.add_argument("--synthetic", action="store_true", help="mark the records synthetic (fixtures, dry runs)")
    a = ap.parse_args(argv)
    targets = units.run_ids(a.run_id, a.unit) if a.unit else [(a.run_id, None)]
    if a.runner_records and len(targets) > 1:
        ap.error("--runner-records names one run; use --unit deny or --unit allow with it")
    codes = [check_run(rid, key, a) for rid, key in targets]
    return 0 if all(c == 0 for c in codes) else 1


def check_run(run_id, key, a) -> int:
    ev_dir = evidence_dir(a.evidence_dir)
    records = load_calls(run_id, a.runner_records, ev_dir)
    items = runner_items(records)
    if not items:
        print(f"no runner records for {run_id}")
        return 1
    offline = bool(a.checkpoint_json)
    if offline:
        cp_rows = logsrc.load_rows(a.checkpoint_json, logsrc.CHECKPOINT_LOG)
        span_rows = logsrc.load_rows(a.spans_json, logsrc.SPANS_LOG) if a.spans_json else []
        tool_rows = logsrc.load_rows(a.tools_json, logsrc.TOOLS_LOG) if a.tools_json else []
    else:
        import boto3
        logs = boto3.client("logs", region_name=os.environ.get("AWS_REGION", "us-east-2"))
        times = [epoch(r["extra"]["sent_at"]) for r in items if r["extra"].get("sent_at")]
        start, end = min(times) - 120, max(times) + 600
        cp_rows = logsrc.insights(logs, [logsrc.CHECKPOINT_LOG], logsrc.Q_CHECKPOINT, start, end)
        span_rows = [] if a.spans == "off" else logsrc.insights(logs, [logsrc.SPANS_LOG], logsrc.Q_SPANS, start, end)
        tool_rows = logsrc.insights(logs, [logsrc.TOOLS_LOG], logsrc.Q_TOOLS, start, end)
        if a.save_rows:
            out = Path(a.save_rows, run_id)
            out.mkdir(parents=True, exist_ok=True)
            (out / "checkpoint.json").write_text(json.dumps(cp_rows, indent=1))
            (out / "spans.json").write_text(json.dumps(span_rows, indent=1))
            (out / "tools.json").write_text(json.dumps(tool_rows, indent=1))
    evlib = None if a.no_write else common.load_evlib()
    return check(run_id, records, cp_rows, span_rows, suite_path=a.suite,
                 declared=read_declared(run_id, ev_dir) or _store_declared(run_id), spans_mode=a.spans, evlib=evlib,
                 synthetic=a.synthetic, unit=key, tool_rows=tool_rows)


if __name__ == "__main__":
    sys.exit(main())
