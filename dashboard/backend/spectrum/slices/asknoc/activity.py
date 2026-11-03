"""What the swarm did in one case, rebuilt from the kit's audit log lines (CloudWatch /team13/checkpoint).

Line shapes are the kit's own: the interceptor's `checkpoint` record at request and response
(checkpoint_kit.py L311-315, L335, L352, L359), `case_opened` (L153), the person's `approval` control
record (L1002) and `tool_error` (L275). Only response-phase lines are shown (one per call).
Only whitelisted fields leave this module: never `arguments` (task text, reasons), never `requested_by`
or `by` (people's identities), never the raw error (it is redacted and shortened).
"""

from __future__ import annotations

import re

from .aws import iso_ms, message_json
from .kit import deny_rule
from .redact import Redactor

NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,47}$")
TOOL = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,40}___[A-Za-z][A-Za-z0-9_]{0,40}$")
APPROVAL = re.compile(r"^chg-[0-9a-f]{12}$")
CASE = re.compile(r"^case-[0-9a-f]{8}$")
_CELL = r"cell=\d+;d_power_db=[+-]\d+\.\d;d_tilt_deg=[+-]\d+\.\d"
CANONICAL = re.compile(rf"^{_CELL}(?:\|{_CELL})*$")
APPROVAL_WORDS = ("PENDING", "APPROVED", "REJECTED", "APPLIED", "NOT_ALLOWED")


def _ok(rx: re.Pattern, v) -> str | None:
    return v if isinstance(v, str) and rx.match(v) else None


def parse_events(events: list[dict], case_id: str, redactor: Redactor) -> list[dict]:
    out = []
    for ev in sorted(events, key=lambda e: (e.get("timestamp") or 0, e.get("eventId") or "")):
        msg = message_json(ev.get("message"))
        if not msg:
            continue
        if msg.get("case_id") not in (None, "", case_id):
            continue  # a line about another case that merely mentions this id
        rec = msg.get("record")
        base = {"time": iso_ms(ev.get("timestamp") or 0)}
        if rec == "case_opened":
            who = msg.get("requested_by")
            out.append({**base, "kind": "case_opened",
                        "by": "front door" if who == "front door" else ("a person (Agent Builder chat)" if who else None)})
        elif rec == "checkpoint" and msg.get("phase") == "response" and msg.get("method") in (None, "tools/call"):
            tool = _ok(TOOL, msg.get("tool"))
            if not tool or tool == "front___case_status":
                continue
            args = msg.get("arguments") if isinstance(msg.get("arguments"), dict) else {}
            item = {**base, "agent": _ok(NAME, msg.get("agent")), "tool": tool,
                    "decision": "DENY" if msg.get("refused") else "ALLOW"}
            if tool.endswith("___ask_agent"):
                item["kind"] = "ask_agent"
                item["to"] = _ok(NAME, args.get("agent"))
            elif tool.endswith("___apply_change"):
                item["kind"] = "apply_call"
                item["approval_id"] = _ok(APPROVAL, msg.get("approval_id"))
                item["approval"] = msg.get("approval") if msg.get("approval") in APPROVAL_WORDS else None
                item["change"] = _ok(CANONICAL, args.get("change"))
            elif tool.startswith("front___"):
                item["kind"] = "front_door"
            else:
                item["kind"] = "tool_call"
            if msg.get("refused"):
                err = msg.get("error")
                text = err.get("message", "") if isinstance(err, dict) else str(err or "")
                item["rule"] = deny_rule(text)
            out.append(item)
        elif rec == "approval":
            out.append({**base, "kind": "approval", "by": "a person",
                        "decision": msg.get("decision") if msg.get("decision") in APPROVAL_WORDS else None,
                        "approval_id": _ok(APPROVAL, msg.get("approval_id"))})
        elif rec == "tool_error":
            out.append({**base, "kind": "tool_error", "tool": msg.get("tool") if isinstance(msg.get("tool"), str)
                        and NAME.match(msg.get("tool")) else None,
                        "error": redactor.text(str(msg.get("error") or ""))[:200]})
        elif rec == "model_control":
            out.append({**base, "kind": "model_control",
                        "action": msg.get("action") if msg.get("action") in ("pause", "resume") else None})
    return out


def find_send(events: list[dict], tag: str, tool: str) -> dict:
    """Did a send tagged `tag` reach the kit? Reads the interceptor's response line for `tool` (it logs the
    arguments, which carry the tag, L313-315) and, for a start, the case_opened line (the request verbatim,
    L153). Returns {"response": None|{"refused", "case_id"}, "opened_case": None|case_id}."""
    found: dict = {"response": None, "opened_case": None}
    for ev in events:
        msg = message_json(ev.get("message")) or {}
        if tag not in (ev.get("message") or ""):
            continue
        if msg.get("record") == "case_opened" and tag in str(msg.get("request", "")):
            found["opened_case"] = _ok(CASE, msg.get("case_id")) or found["opened_case"]
        elif (msg.get("record") == "checkpoint" and msg.get("phase") == "response" and msg.get("tool") == tool):
            found["response"] = {"refused": bool(msg.get("refused")), "case_id": _ok(CASE, msg.get("case_id")),
                                 "rule": deny_rule(str((msg.get("error") or {}).get("message", "")
                                                       if isinstance(msg.get("error"), dict) else msg.get("error") or ""))}
    return found
