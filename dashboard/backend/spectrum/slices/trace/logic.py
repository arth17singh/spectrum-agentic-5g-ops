"""Pure functions behind the Trace view: records in, case lists, timelines and the sequence check out.

Inputs are plain dicts shaped like the contract kinds `decision`, `control` and `change`. Nothing here
reads a file or calls AWS. Each case is rebuilt from the records alone and then compared with the
declared expected sequence:

    case_opened -> ask_agent* -> tool calls -> approval -> change_applied -> observe

Ordering uses the record's own time (`aws_time` when it has one, otherwise the time our own code wrote), as before.
Every event also carries, separately, `aws_time` + `aws_time_source` (cloudwatch | cloudwatch_attest | s3_stored | policy_createdAt |
gateway_date | none: see `_shared/awstime.py`) and `runner_time`, the clock of our own code. A runner or Lambda clock is
never reported as AWS time.
"""

from __future__ import annotations

import re
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
    return {"synthetic": synthetic, "real": (not synthetic) and t["aws_time_source"] in awstime.EVENT_SOURCES, **t,
            "no_event_reason": awstime.no_event_reason(rec)}


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


# ---------------------------------------------------------------------------------------------- per-unit expectation

_UNIT = re.compile(r"^(?P<unit>.+?)-\d{8}T\d{6}Z")
# case ids a suite uses for calls that carry no case id at all (the deny items that test the case-id rule)
NO_CASE_IDS = ("(missing)", "no-case")
NO_CASE_LABEL = "calls without a case id (deny items that test the case-id rule)"


def unit_of(run_id: str | None) -> str | None:
    """The suite unit a run id names: t3.deny-20261103T054914Z -> t3.deny."""
    m = _UNIT.match(run_id or "")
    return m.group("unit") if m else None


def case_label(case_id: str | None) -> str | None:
    return NO_CASE_LABEL if case_id in NO_CASE_IDS else None


def _hms(iso: str | None) -> str:
    return (iso or "")[11:23] + "Z"


def _ordered_by_gateway_window(timeline: list[Rec]) -> tuple[bool, str] | None:
    """The Gateway logs a call's policy decision when the request ENDS, so an ALLOW can be stamped after the tool's
    own log line although the decision came first. AWS records the request's start and end (the AgentCore
    InvokeTool span in aws/spans, read by the evidence sync). When the applied change's CloudWatch Lambda log time and
    the ALLOW both fall inside that window, they belong to one request and their order is not judged by the
    two stamps. Returns (order_ok after treating them as one request, the note to show), or None when no change
    carries such evidence (then the strict order stands)."""
    for ch in timeline:
        w = ch.get("gateway_window")
        if ch["step"] != "change_applied" or ch.get("aws_time_source") != "cloudwatch_lambda" or not isinstance(w, dict):
            continue
        lo, hi, at = _parse(w.get("start")), _parse(w.get("end")), _parse(ch.get("aws_time"))
        allow = next((e for e in timeline if e["step"] == "approval" and e["decision"] == "ALLOW"
                      and e.get("approval_id") and e["approval_id"] == ch.get("approval_id")), None)
        if not (lo and hi and at and allow and lo <= at <= hi):
            continue
        ta = _parse(allow["time"])
        if not (ta and lo <= ta <= hi):
            continue
        moved = [{**e, "time": allow["time"]} if e is ch else e for e in timeline]
        ok = check_sequence(_sorted(moved))["order_ok"]
        return ok, (f"The apply ({_hms(ch['aws_time'])}, Lambda log) falls inside the Gateway request "
                    f"{_hms(w.get('start'))} to {_hms(w.get('end'))}; the Gateway logs its decision at request end "
                    f"({_hms(allow['time'])}), so the two stamps are not ordered against each other.")
    return None


def check_for_unit(run_id: str | None, timeline: list[Rec]) -> Rec:
    """The check against the sequence expected for the case's OWN unit, not one flow for every unit:
      t3.deny   complete when a refusal (DENY) is recorded and no change was applied;
      t3.pause  complete when a refusal because the model is paused is recorded;
      t3.allow  complete when an ALLOWed call is recorded; a case that asks for approval and applies a change must
                also show the ALLOWed approval and the applied change, in order;
      t3.trust* (the in-process trust loop: no agent calls, no human approval) complete when a change was applied
                and measured;
    any other unit: the full declared sequence (`check_sequence`). The result has the shape of `check_sequence`'s,
    plus `unit`, `expected_label` and `by_unit`."""
    generic = check_sequence(timeline)
    unit = unit_of(run_id) or ""
    steps_by = {s["step"]: s for s in generic["steps"]}
    dec = [e for e in timeline if e["kind"] == "decision"]
    applied = "change_applied" in steps_by and steps_by["change_applied"]["present"]

    def build(label: str, expected: list[str], present: dict[str, bool], extra_ok: bool = True, notes: list[str] | None = None,
              order_ok: bool = True) -> Rec:
        missing = [s for s in expected if not present[s]]
        return {"passed": not missing and extra_ok and order_ok, "order_ok": order_ok, "missing": missing,
                "notes": notes or [], "expected": expected, "expected_label": label, "unit": unit, "by_unit": True,
                "steps": [{"step": s, "present": present[s], "required": True, "index": None} for s in expected]}

    if unit == "t3.deny":
        refused = any(e["decision"] == "DENY" for e in dec)
        return build("a refusal is recorded (DENY) and no change is applied", ["refusal"], {"refusal": refused}, not applied,
                     ["A change was applied in a deny case."] if applied else [])
    if unit == "t3.pause":
        return build("a refusal because the model is paused is recorded", ["paused_refusal"],
                     {"paused_refusal": any(_is_pause_event(e) for e in timeline)})
    if unit == "t3.allow":
        if applied or any(e["step"] == "approval" for e in timeline):
            need = ["approval", "change_applied"]
            order_ok, notes = generic["order_ok"], [n for n in generic["notes"] if "approval" in n.lower()]
            if not order_ok:
                fixed = _ordered_by_gateway_window(timeline)
                if fixed is not None:
                    order_ok, note = fixed
                    notes.append(note)
            return build("the approval is ALLOWed and the change is applied, in order", need,
                         {s: steps_by[s]["present"] for s in need}, order_ok=order_ok, notes=notes)
        return build("an ALLOWed call is recorded", ["allowed_call"], {"allowed_call": any(e["decision"] == "ALLOW" for e in dec)})
    if unit.startswith("t3.trust"):
        return build("a change is applied and measured (in-process loop: no agent calls, no human approval)", ["change_applied"],
                     {"change_applied": applied}, order_ok=generic["order_ok"])
    return {**generic, "unit": unit or None, "expected_label": "case opened, agents asked, tools called, approval, change applied, observe",
            "by_unit": False}


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
            "aws_time_summary": awstime.summary(ev), "unverified_reasons": awstime.reasons(ev)}


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
        "case_id": case_id, "run_id": run_id or None, "unit": unit_of(run_id), "case_label": case_label(case_id),
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
        rows.append(summarize(run_id, case_id, tl, check_for_unit(run_id, tl)))
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
    check = check_for_unit(key[0], tl)
    row = summarize(key[0], case_id, tl, check)
    return {
        **row, "timeline": tl, "check": check,
        "other_runs": [k[0] for k in keys if k != key],
        "show_record": f"python3 tools/show_record.py --run-id {shlex.quote(key[0])} --item {shlex.quote(case_id)}",
    }
