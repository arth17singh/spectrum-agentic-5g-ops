"""Pure functions behind the Trace view: records in, case lists, timelines and the sequence check out.

Inputs are plain dicts shaped like the contract kinds `decision`, `control` and `change`. Nothing here
reads a file or calls AWS. Each case is rebuilt from the records alone and then compared with the
declared expected sequence:

    case_opened -> ask_agent* -> tool calls -> approval -> change_applied -> observe

Ordering uses the record's own time (`aws_time` when it has one, otherwise the time our own code wrote), as before.
Every event also carries, separately, `aws_time` + `aws_time_source` (cloudwatch | s3_stored | policy_createdAt |
gateway_date | none: see `_shared/awstime.py`) and `runner_time`, the clock of our own code. A runner or Lambda clock is
never reported as AWS time.
"""

from __future__ import annotations

import shlex
from datetime import datetime, timezone
from typing import Any, Iterable

from spectrum.slices._shared import awstime

Rec = dict[str, Any]

# The declared expected sequence, in order. `ask_agent` may be absent (the star in the declaration);
# every other step is required.
SEQUENCE = ("case_opened", "ask_agent", "tool_call", "approval", "change_applied", "observe")
REQUIRED = ("case_opened", "tool_call", "approval", "change_applied", "observe")

_OPENERS = {"start_case", "open_case"}


# ---------------------------------------------------------------------------------------------- helpers

def tool_name(tool: Any) -> str:
    """`work___apply_change` -> `apply_change`: the gateway prefixes the target name with three underscores."""
    t = str(tool or "")
    return t.split("___")[-1] if "___" in t else t


def _parse(ts: Any) -> datetime | None:
    if not isinstance(ts, str) or not ts:
        return None
    try:
        d = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def event_time(rec: Rec, kind: str) -> str | None:
    """AWS time when the record carries one, otherwise the record's own time field."""
    own = rec.get("time") if kind in ("decision", "control") else rec.get("created_at")
    return rec.get("aws_time") or own or rec.get("created_at")


def _prov(rec: Rec) -> Rec:
    """Provenance of one displayed record, plus its AWS time and its runner clock (kept apart). `real` needs both:
    not marked synthetic (fixtures and dry runs are), and a time AWS recorded for the event itself (CloudWatch, the
    policy's createdAt, the Gateway's Date). A stored time alone (S3 LastModified) shows when the record was stored,
    not that the event was real, so it is not enough; a Lambda or runner clock is never one."""
    synthetic = rec.get("synthetic") is True
    t = awstime.resolve(rec)
    return {"synthetic": synthetic, "real": (not synthetic) and t["aws_time_source"] in awstime.EVENT_SOURCES, **t}


def _is_pause(rec: Rec) -> bool:
    text = f"{rec.get('deciding_rule') or ''} {rec.get('reason') or ''}".lower()
    return rec.get("decision") == "DENY" and "pause" in text


def _step_of(rec: Rec) -> str:
    """Which step of the expected sequence a decision record belongs to."""
    name = tool_name(rec.get("tool"))
    if name in _OPENERS or name == "case_opened":
        return "case_opened"
    if name == "ask_agent":
        return "ask_agent"
    if name == "apply_change":
        return "approval"
    if "observe" in name:
        return "observe"
    if name in ("case_status", "continue_case"):
        return "status"
    return "tool_call"


# ---------------------------------------------------------------------------------------------- timeline

def _decision_event(rec: Rec) -> Rec:
    step = _step_of(rec)
    if _is_pause(rec):
        cat = "pause"
    elif rec.get("decision") == "DENY":
        cat = "deny"
    else:
        cat = "allow"
    if step == "approval" and cat == "deny" and rec.get("approval_id"):
        # apply_change is the human-approval gate: a DENY that names an approval id is a request waiting for a person
        cat = "approval"
    return {
        "time": event_time(rec, "decision"), "source": rec.get("source"), "agent": rec.get("agent"),
        "tool": rec.get("tool"), "decision": rec.get("decision"), "deciding_rule": rec.get("deciding_rule"),
        "reason": rec.get("reason"), "guardrail_scores": rec.get("guardrail_scores"),
        "approval_id": rec.get("approval_id"),
        "detail": {k: rec.get(k) for k in ("trace_id", "span_id", "log_pointer") if rec.get(k) is not None},
        "kind": "decision", "step": step, "category": cat, **_prov(rec),
    }


_CHANGE_FIELDS = ("change_id", "item_id", "change_text", "change", "twin", "predictor", "policy", "baseline_kpis",
                  "measured_kpis", "measured_delta", "predicted_delta", "error", "direction_agree", "scored",
                  "unscored_reason", "probe", "trust_before", "trust_after")


def _change_event(rec: Rec, log_seen: bool = False) -> Rec:
    detail = {k: rec.get(k) for k in _CHANGE_FIELDS if k in rec}
    detail["log_line_seen"] = log_seen
    return {
        "time": event_time(rec, "change"), "source": "change", "agent": None, "tool": "apply_change",
        "decision": None, "deciding_rule": None, "reason": None, "guardrail_scores": None,
        "approval_id": rec.get("change_id"), "detail": detail,
        "kind": "change", "step": "change_applied", "category": "applied", **_prov(rec),
    }


def _is_applied_log(rec: Rec) -> bool:
    """The tools also print `{"record": "change_applied", ...}` to /aws/lambda/team13-ran-tools."""
    return rec.get("record") == "change_applied"


def _applied_log_event(rec: Rec) -> Rec:
    detail = {k: rec.get(k) for k in ("change_text", "item_id", "log_pointer") if rec.get(k) is not None}
    return {
        "time": event_time(rec, "decision"), "source": rec.get("source") or "/aws/lambda/team13-ran-tools", "agent": rec.get("agent"),
        "tool": rec.get("tool") or "ran___apply_change", "decision": None, "deciding_rule": None, "reason": None,
        "guardrail_scores": None, "approval_id": rec.get("approval_id"), "detail": detail,
        "kind": "log", "step": "change_applied", "category": "applied", **_prov(rec),
    }


def _control_event(rec: Rec) -> Rec:
    detail = {k: rec.get(k) for k in ("model_id", "version", "policy_name", "by") if rec.get(k) is not None}
    return {
        "time": event_time(rec, "control"), "source": "control", "agent": rec.get("by"),
        "tool": rec.get("model_id"), "decision": str(rec.get("action") or "").upper() or None,
        "deciding_rule": rec.get("policy_name"), "reason": rec.get("reason"), "guardrail_scores": None,
        "approval_id": None, "detail": detail,
        "kind": "control", "step": "control", "category": "control", **_prov(rec),
        "record_id": f"{rec.get('run_id')}|{rec.get('model_id')}|{rec.get('version')}|{rec.get('action')}|{rec.get('time')}",
    }


def _sorted(events: list[Rec]) -> list[Rec]:
    """Order by time. Log times have one-second resolution in places, so events with the same time are put in
    the declared sequence order (an unordered tie is not a violation); an undated event goes after the dated ones."""
    far = datetime.max.replace(tzinfo=timezone.utc)
    rank = {s: i for i, s in enumerate(SEQUENCE)}
    return sorted(events, key=lambda e: (_parse(e["time"]) or far, rank.get(e["step"], len(SEQUENCE))))


def build_timeline(case_id: str, decisions: Iterable[Rec], changes: Iterable[Rec], controls: Iterable[Rec] = ()) -> list[Rec]:
    """Ordered timeline of one case. Controls (pause/resume of a model) have no case id, so those whose time
    falls inside the case's own window are shown, marked source `control`. A `change_applied` log line is an
    event of its own, unless a `change` record with the same approval id exists (then the record carries it)."""
    d_in = [r for r in decisions if r.get("case_id") == case_id]
    c_in = [r for r in changes if r.get("case_id") == case_id]
    logs = [r for r in d_in + c_in if _is_applied_log(r)]
    decs = [r for r in d_in if not _is_applied_log(r)]
    chgs = [r for r in c_in if not _is_applied_log(r)]
    have = {r.get("change_id") for r in chgs}
    seen = {r.get("approval_id") for r in logs}
    ev = [_decision_event(r) for r in decs]
    ev += [_change_event(r, r.get("change_id") in seen) for r in chgs]
    ev += [_applied_log_event(r) for r in logs if r.get("approval_id") not in have]
    ev = _sorted(ev)
    times = [t for t in (_parse(e["time"]) for e in ev) if t]
    if times:
        lo, hi = min(times), max(times)
        ev += [_control_event(r) for r in controls if (t := _parse(event_time(r, "control"))) and lo <= t <= hi]
        ev = _sorted(ev)
    return ev


# ---------------------------------------------------------------------------------------------- the check

def check_sequence(timeline: list[Rec]) -> Rec:
    """Compare a timeline with the declared expected sequence.

    A step counts only when its call was ALLOWed (a refused call did not happen). Approval needs an ALLOWed
    `apply_change` decision with an approval id (a DENY that names one is a request still waiting for a
    person). `change_applied` is the `change` record (or its log line). `observe` counts only after the change.
    Order is judged on the first time each present step occurs; steps must not run backwards.
    """
    first: dict[str, int] = {}
    for i, e in enumerate(timeline):
        step = e["step"]
        if e["kind"] == "decision" and e["decision"] != "ALLOW":
            continue  # a refused call did not happen
        if step == "approval" and not e.get("approval_id"):
            continue
        if step == "observe" and "change_applied" not in first:
            step = "tool_call"  # looking at the network before the change is an ordinary tool call
        if step in SEQUENCE and step not in first:
            first[step] = i
    missing = [s for s in REQUIRED if s not in first]
    notes: list[str] = []
    approvals = [e for e in timeline if e["step"] == "approval"]
    if "approval" in missing:
        if "change_applied" in first:
            notes.append("A change was applied but no ALLOWed apply_change decision carries an approval id.")
        elif any(e["decision"] == "DENY" and e.get("approval_id") for e in approvals):
            notes.append("An approval was requested and has not been granted.")
        else:
            notes.append("No approval step was recorded.")
    if any(_is_pause_event(e) for e in timeline):
        notes.append("A call was refused because the model's tool is paused.")
    present = [s for s in SEQUENCE if s in first]
    positions = [first[s] for s in present]
    order_ok = positions == sorted(positions)
    # equal-time events are placed by input order, so a step cannot be ahead of the one before it on a tie
    if order_ok and len(present) > 1:
        t = [_parse(timeline[first[s]]["time"]) for s in present]
        if all(t):
            order_ok = all(a <= b for a, b in zip(t, t[1:]))
    steps = [{"step": s, "present": s in first, "required": s in REQUIRED, "index": first.get(s)} for s in SEQUENCE]
    return {"passed": not missing and order_ok, "order_ok": order_ok, "missing": missing, "notes": notes,
            "expected": list(SEQUENCE), "steps": steps}


def _is_pause_event(e: Rec) -> bool:
    return e.get("category") == "pause"


# ---------------------------------------------------------------------------------------------- listing

# ---------------------------------------------------------------------------------------------- provenance

def provenance(events: Iterable[Rec]) -> Rec:
    """What the screen may claim about the records it displays. `state`:
      empty       nothing displayed
      real        every record has an AWS-recorded event time and none is marked synthetic
      synthetic   every record is marked synthetic (test fixtures, dry runs)
      mixed       some synthetic, some not
      unverified  none synthetic, but some lack an AWS-recorded event time (so not shown as real)
    `aws_time_summary` is "AWS time on k of n (cloudwatch a, S3 stored b, ...)" over the same records, by source.
    `note` is the text to show. `claim` ("Nothing here is simulated.") is non-empty only in the real state."""
    ev = list(events)
    n = len(ev)
    n_syn = sum(1 for e in ev if e.get("synthetic"))
    n_unv = sum(1 for e in ev if not e.get("synthetic") and not e.get("real"))
    if n == 0:
        state, label, note = "empty", "", ""
    elif n_syn == n:
        state, label, note = "synthetic", "SYNTHETIC", "Synthetic records (test fixtures): not a real run."
    elif n_syn == 0 and n_unv == 0:
        state, label, note = "real", "REAL", "Every record shown has an AWS-recorded event time and is not marked synthetic."
    elif n_syn == 0:
        state, label = "unverified", "UNVERIFIED"
        note = f"{n_unv} of {n} records have no AWS-recorded event time, so they are not shown as real."
    else:
        state, label = "mixed", "MIXED"
        note = f"{n_syn} of {n} records synthetic" + (f"; {n_unv} more have no AWS-recorded event time." if n_unv else ".")
    claim = "Nothing here is simulated." if state == "real" else ""
    return {"state": state, "label": label, "note": note, "claim": claim, "n_records": n, "n_synthetic": n_syn, "n_unverified": n_unv,
            "aws_time_summary": awstime.summary(ev)}


def _group(records: Iterable[Rec]) -> dict[tuple[str, str], list[Rec]]:
    out: dict[tuple[str, str], list[Rec]] = {}
    for r in records:
        cid = r.get("case_id")
        if isinstance(cid, str) and cid:
            out.setdefault((str(r.get("run_id") or ""), cid), []).append(r)
    return out


def summarize(run_id: str, case_id: str, timeline: list[Rec], check: Rec) -> Rec:
    times = [e["time"] for e in timeline if e["time"]]
    parsed = sorted(((_parse(t), t) for t in times if _parse(t)), key=lambda p: p[0])
    dec = [e for e in timeline if e["kind"] == "decision"]
    return {
        "case_id": case_id, "run_id": run_id or None,
        "first_time": parsed[0][1] if parsed else None, "last_time": parsed[-1][1] if parsed else None,
        "n_events": len(timeline),
        "denies": sum(1 for e in dec if e["decision"] == "DENY"),
        "allows": sum(1 for e in dec if e["decision"] == "ALLOW"),
        "pauses": sum(1 for e in dec if e["category"] == "pause"),
        "applied": sum(1 for e in timeline if e["step"] == "change_applied"),
        "check_passed": check["passed"],
        "synthetic": any(e.get("synthetic") for e in timeline),
        "provenance": provenance(timeline),
    }


def build_overview(decisions: list[Rec], changes: list[Rec], controls: list[Rec]) -> Rec:
    """One row per (run_id, case_id), newest first, plus the provenance of every record displayed across them
    (a control shown in several case windows is counted once). A case exists when it has at least one decision
    or change."""
    d, c = _group(decisions), _group(changes)
    rows, shown, ctl_seen = [], [], set()
    for key in sorted(set(d) | set(c)):
        run_id, case_id = key
        tl = build_timeline(case_id, d.get(key, []), c.get(key, []), [x for x in controls if (x.get("run_id") or "") == run_id or not x.get("run_id")])
        rows.append(summarize(run_id, case_id, tl, check_sequence(tl)))
        for e in tl:
            if e["kind"] == "control":
                if e["record_id"] in ctl_seen:
                    continue
                ctl_seen.add(e["record_id"])
            shown.append(e)
    far = ""
    rows.sort(key=lambda r: r["last_time"] or far, reverse=True)
    return {"cases": rows, "provenance": provenance(shown)}


def build_cases(decisions: list[Rec], changes: list[Rec], controls: list[Rec]) -> list[Rec]:
    return build_overview(decisions, changes, controls)["cases"]


def build_case(case_id: str, decisions: list[Rec], changes: list[Rec], controls: list[Rec],
               run_id: str | None = None) -> Rec | None:
    """Full detail for one case, or None when no record mentions it. With the same case id in several runs,
    the run with the latest record is used unless `run_id` names one; the others are listed in `other_runs`."""
    d, c = _group(decisions), _group(changes)
    keys = sorted(k for k in set(d) | set(c) if k[1] == case_id and (run_id is None or k[0] == run_id))
    if not keys:
        return None
    rows = {k: summarize(k[0], case_id, build_timeline(case_id, d.get(k, []), c.get(k, [])), {"passed": False}) for k in keys}
    key = max(keys, key=lambda k: rows[k]["last_time"] or "")
    ctl = [x for x in controls if (x.get("run_id") or "") == key[0] or not x.get("run_id")]
    tl = build_timeline(case_id, d.get(key, []), c.get(key, []), ctl)
    check = check_sequence(tl)
    row = summarize(key[0], case_id, tl, check)
    return {
        **row, "timeline": tl, "check": check,
        "other_runs": [k[0] for k in keys if k != key],
        "show_record": f"python3 tools/show_record.py --run-id {shlex.quote(key[0])} --item {shlex.quote(case_id)}",
    }
