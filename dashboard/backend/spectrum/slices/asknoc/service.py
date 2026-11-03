"""Ask the NOC: turn and case lifecycle, one poller per case, reconciliation of uncertain sends, health.

A TURN is one operator message (the start of a case, or a follow-up). Its `ui_state` is the one field the
page keys its copy on:

  queued -> sending -> running -> answered | awaiting_approval | needs_input | failed | lost
                    -> refused_policy | refused_auth | token_expired | kit_unreachable | kit_error
                    -> lead_unconfirmed        (the kit answered, but could not confirm noc_lead got it)
                    -> outcome_unknown -> running | refused_policy | lead_unconfirmed | not_received | unconfirmed
  any active state  -> abandoned               ("Stop following": local only, the kit has no cancel)
  queued at restart -> cancelled_by_restart    (nothing had left the laptop)

Rules that matter (see the plan and its review):
- Nothing is ever resent automatically: the kit has no idempotency. A turn is saved as `sending` before the
  request leaves; after a restart it becomes `outcome_unknown` and is reconciled, never retried.
- `kit_unreachable` only when nothing reached AWS (DNS, refused, TLS, connect timeout). A read timeout, a
  5xx or a 429 may have been acted on -> `outcome_unknown`.
- "Arrived" is proved by the interceptor's RESPONSE line for the call (it carries our tag in `arguments`
  and, for a start, the case id parsed from the answer, checkpoint_kit.py L313-315, L336-342), never by
  `case_opened` alone: open_case runs before the lead is reached (L241-243).
- A follow-up whose arrival was only inferred may finish only after a new running state was seen or the
  reply changed; the old task's reply is never shown as the new turn's answer (send_to_lead overwrites
  task-<case> only after A2A answers, L237).
- `waiting_approvals` is refreshed on every poll (the kit scans it whatever the task state, L257-258).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .activity import APPROVAL, CANONICAL, CASE, find_send, parse_events
from .aws import AwsReads, AwsUnavailable
from .config import KitConfig, NotConfigured
from .kit import CONTINUE, START, STATUS, KitClient, Outcome, deny_rule
from .redact import Redactor
from .store import Store

RUNNING = ("TASK_STATE_SUBMITTED", "TASK_STATE_WORKING")  # checkpoint_kit.py L124
FAILED_STATES = ("TASK_STATE_FAILED", "TASK_STATE_CANCELED", "TASK_STATE_REJECTED")
TERMINAL = {"answered", "awaiting_approval", "needs_input", "failed", "lost", "abandoned", "cancelled_by_restart",
            "refused_policy", "refused_auth", "token_expired", "kit_unreachable", "kit_error", "lead_unconfirmed",
            "not_received", "unconfirmed"}
ACTIVE = {"queued", "sending", "running", "outcome_unknown"}
A2A_TROUBLE = re.compile(r"A2A error|urlopen|timed out|timeout|Connection|HTTP Error [45]\d\d|Remote end|reset by peer|"
                         r"Name or service|SSL|CERTIFICATE|unreachable", re.I)
APPROVAL_TEXT = re.compile(r"\b(approved?|go ahead|proceed|apply it|rejected?)\b", re.I)
PROBE_CASE = "case-unknown0"  # the kit's own probe id (checkpoint_kit.py L952)
REPO = Path(__file__).resolve().parents[5]

# Measured on the real kit (the one approved live smoke run, 2 Nov 2026 PST / 3 Nov 05:01Z UTC).
MEASURED = {"case_id": "case-d59bd7e0", "kind": "read-only question", "seconds": 120,
            "started": "2026-11-03T05:01:23Z", "finished": "2026-11-03T05:03:23Z",
            "note": "start_case returned a case id at once; about 19 status polls (5 s apart) saw SUBMITTED/WORKING."}


def _env_float(env, name, default):
    try:
        return float(env.get(name) or default)
    except ValueError:
        return float(default)


@dataclass
class Timing:
    lead_timeout_s: float = 300.0       # Builder setting per t3/agents/BUILDER_CHECKLIST.md:34; not verified
    overdue_after_s: float = 330.0
    release_slot_after_s: float = 360.0  # a running turn stops blocking new starts after this
    lost_after_s: float = 900.0
    poll_delays: tuple = (3.0, 5.0, 8.0, 10.0)
    error_delays: tuple = (5.0, 10.0, 20.0, 40.0, 60.0)
    reconcile_window_s: float = 120.0
    reconcile_every_s: float = 15.0
    continue_confirm_s: float = 120.0
    unconfirmed_hold_s: float = 300.0   # no follow-up for this long after lead_unconfirmed
    send_timeout_s: float = 45.0
    status_timeout_s: float = 35.0
    health_timeout_s: float = 10.0
    probe_cache_s: float = 300.0
    aws_cache_s: float = 60.0
    activity_cache_s: float = 10.0

    @classmethod
    def from_env(cls, env=None) -> "Timing":
        env = os.environ if env is None else env
        lead = _env_float(env, "ASKNOC_LEAD_TIMEOUT_S", 300)
        return cls(lead_timeout_s=lead, overdue_after_s=lead + 30,
                   release_slot_after_s=_env_float(env, "ASKNOC_RELEASE_SLOT_S", 360),
                   lost_after_s=_env_float(env, "ASKNOC_LOST_S", 900))


def iso(t: float | None) -> str | None:
    if not t:
        return None
    return datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def sha256(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def parse_waiting(raw) -> list[dict]:
    """The kit's waiting_approvals [{approval_id, change: "<json string>"}] (L257-258, L296), whitelisted."""
    out = []
    for w in raw if isinstance(raw, list) else []:
        if not isinstance(w, dict) or not APPROVAL.match(str(w.get("approval_id", ""))):
            continue
        try:
            ch = json.loads(w.get("change") or "{}")
        except (TypeError, ValueError):
            ch = {}
        ch = ch if isinstance(ch, dict) else {}
        change = ch.get("change") if isinstance(ch.get("change"), str) else ""
        item = ch.get("item_id") if isinstance(ch.get("item_id"), str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,64}", ch["item_id"]) else None
        out.append({"approval_id": w["approval_id"], "item_id": item,
                    "change": change if CANONICAL.match(change) else None,
                    "canonical": bool(CANONICAL.match(change)),
                    "reason": str(ch.get("reason") or "")[:300]})
    return out


class RefusedRequest(Exception):
    def __init__(self, status: int, code: str, message: str, **extra):
        super().__init__(code)
        self.status, self.code, self.message, self.extra = status, code, message, extra


class AskNoc:
    def __init__(self, config: KitConfig, kit: KitClient, store: Store, aws: AwsReads, *, offline: bool = False,
                 timing: Timing | None = None, clock=time.time, sleep=asyncio.sleep, env=None):
        self.config, self.kit, self.store, self.aws = config, kit, store, aws
        self.offline, self.clock, self.sleep = offline, clock, sleep
        self.env = os.environ if env is None else env
        self.timing = timing or Timing.from_env(self.env)
        self.tasks: dict[str, asyncio.Task] = {}
        self.idem: OrderedDict[str, tuple[str, dict, float]] = OrderedDict()
        self.calls: dict[str, deque] = {}
        self._cache: dict[str, tuple[float, object]] = {}
        self._redactor: tuple[tuple, Redactor] | None = None

    # ------------------------------------------------------------------ small helpers
    def now(self) -> float:
        return self.clock()

    def redactor(self) -> Redactor:
        view = self.config.load()
        key = view.redact_values
        if self._redactor is None or self._redactor[0] != key:
            self._redactor = (key, Redactor(key))
        return self._redactor[1]

    def save(self) -> None:
        self.store.save()

    def spawn(self, name: str, coro) -> None:
        old = self.tasks.get(name)
        if old is not None and not old.done():
            coro.close()
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # no loop (unit tests drive the steps by hand)
            coro.close()
            return
        self.tasks[name] = loop.create_task(coro, name=f"asknoc:{name}")

    def limited(self, bucket: str, n: int, per_s: float) -> float | None:
        """Global sliding-window limit. Returns seconds to wait, or None when allowed (and counted)."""
        q = self.calls.setdefault(bucket, deque())
        now = self.now()
        while q and now - q[0] > per_s:
            q.popleft()
        if len(q) >= n:
            return round(per_s - (now - q[0]), 1)
        q.append(now)
        return None

    def reserve_key(self, key: str, body_hash: str) -> dict | None:
        """In-memory idempotency for 10 min (LRU of 500). Synchronous: the key is reserved before any await,
        so two requests with one key can't both send. Returns the earlier response for a replay."""
        now = self.now()
        for k in [k for k, v in self.idem.items() if now - v[2] > 600]:
            self.idem.pop(k, None)
        hit = self.idem.get(key)
        if hit:
            if hit[0] != body_hash:
                raise RefusedRequest(409, "idempotency_conflict", "This request key was already used for another message.")
            if hit[1] is None:
                raise RefusedRequest(409, "duplicate_in_progress", "This message is already being sent.")
            return {**hit[1], "replayed": True}
        self.idem[key] = (body_hash, None, now)
        while len(self.idem) > 500:
            self.idem.popitem(last=False)
        return None

    def release_key(self, key: str) -> None:
        hit = self.idem.get(key)
        if hit and hit[1] is None:
            self.idem.pop(key, None)

    def remember_key(self, key: str, body_hash: str, response: dict) -> None:
        self.idem[key] = (body_hash, response, self.now())

    def inflight(self) -> int:
        now = self.now()
        n = 0
        for t in self.store.turns.values():
            if t["ui_state"] in ("queued", "sending"):
                n += 1
            elif t["ui_state"] == "running" and now - (t.get("accepted_at") or t.get("sent_at") or now) < self.timing.release_slot_after_s:
                n += 1
        return n

    # ------------------------------------------------------------------ turns
    def _turn(self, kind: str, text: str, case_id: str | None) -> dict:
        tid = "t-" + uuid.uuid4().hex[:12]
        prev = None
        if case_id:
            ids = self.store.cases[case_id]["turn_ids"]
            prev = self.store.turns.get(ids[-1]) if ids else None
        turn = {"turn_id": tid, "kind": kind, "case_id": case_id, "text": text, "tag": f"asknoc-{tid}",
                "created_at": self.now(), "sent_at": None, "accepted_at": None, "terminal_at": None,
                "ui_state": "queued", "a2a_state": None, "gateway_date": None, "reply": None,
                "waiting_approvals": [], "failed_cause": None, "deny_rule": None, "detail": None,
                "category": None, "overdue": False, "status_unavailable": False, "status_errors": 0, "polls": 0,
                "seen_running": False, "confirming": False,
                "prev_reply_sha256": (prev or {}).get("reply_sha256"), "reply_sha256": None, "recorded": False}
        self.store.turns[tid] = turn
        if case_id:
            self.store.cases[case_id]["turn_ids"].append(tid)
        return turn

    def sent_text(self, turn: dict) -> str:
        return f"{turn['text']}\n\nref: {turn['tag']}"

    def _ensure_case(self, case_id: str, turn: dict) -> None:
        case = self.store.cases.get(case_id)
        if case is None:
            self.store.cases[case_id] = {"case_id": case_id, "started_at": turn.get("sent_at") or self.now(),
                                         "turn_ids": [turn["turn_id"]], "approvals_seen": []}
        elif turn["turn_id"] not in case["turn_ids"]:
            case["turn_ids"].append(turn["turn_id"])

    def _accept(self, turn: dict, case_id: str, state: str | None) -> None:
        turn["case_id"] = case_id
        turn["accepted_at"] = turn.get("accepted_at") or self.now()
        turn["a2a_state"] = state
        turn["seen_running"] = turn["seen_running"] or state in RUNNING
        turn["ui_state"] = "running"
        self._ensure_case(case_id, turn)
        self.save()
        self.spawn(f"poll:{turn['turn_id']}", self.poll_loop(turn["turn_id"]))

    def _finish(self, turn: dict, state: str, **fields) -> None:
        turn.update(fields)
        turn["ui_state"] = state
        turn["terminal_at"] = self.now()
        if turn.get("case_id") and CASE.match(turn["case_id"]):
            self._ensure_case(turn["case_id"], turn)
        self.save()
        try:
            self.store.log_turn({"turn_id": turn["turn_id"], "kind": turn["kind"], "case_id": turn.get("case_id"),
                                 "trace_id": turn.get("case_id"), "ui_state": state, "a2a_state": turn.get("a2a_state"),
                                 "failed_cause": turn.get("failed_cause"), "created_at": iso(turn["created_at"]),
                                 "sent_at": iso(turn.get("sent_at")), "accepted_at": iso(turn.get("accepted_at")),
                                 "terminal_at": iso(turn["terminal_at"]), "gateway_date": turn.get("gateway_date"),
                                 "operator_text_sha256": sha256(turn["text"]),
                                 "sent_text_sha256": sha256(self.sent_text(turn)),
                                 "reply_sha256": turn.get("reply_sha256"), "polls": turn.get("polls"),
                                 "waiting_approvals": [w["approval_id"] for w in turn.get("waiting_approvals") or []]})
        except OSError:
            pass  # the local history is a convenience; the kit's audit log is the record

    # ------------------------------------------------------------------ requests from the page
    def check_start(self) -> None:
        if self.offline:
            raise RefusedRequest(503, "offline", "Offline: the panel makes no calls to the kit.")
        view = self.config.load()
        if not view.configured:
            raise RefusedRequest(503, "kit_not_configured", view.problem or "The kit is not configured on this laptop.")
        if not view.real_gateway and self.env.get("ASKNOC_ALLOW_TEST_GATEWAY") != "1":
            raise RefusedRequest(503, "not_real_gateway", "The kit folder names a Gateway that is not an AWS "
                                 "AgentCore address; nothing is sent to it.")
        left = view.token_seconds_left(self.now())
        if left is not None and left <= 0:
            raise RefusedRequest(503, "token_expired", "The app's token has expired. Nothing was sent.")
        if self.inflight() >= int(self.env.get("ASKNOC_MAX_INFLIGHT") or 1):
            raise RefusedRequest(409, "case_in_flight", "noc_lead is already working on a message from this panel. "
                                 "Wait for it, or press Stop following on it.")

    def start(self, text: str) -> dict:
        self.check_start()
        turn = self._turn("start", text, None)
        self.save()
        self.spawn(f"send:{turn['turn_id']}", self.send_turn(turn["turn_id"]))
        return self.turn_view(turn)

    async def check_follow_up(self, case_id: str, text: str) -> None:
        case = self.store.cases.get(case_id)
        if case is None:
            raise RefusedRequest(404, "unknown_case", "This panel did not start that case.")
        last = self.store.turns.get(case["turn_ids"][-1]) if case["turn_ids"] else None
        if last and last["ui_state"] not in TERMINAL:
            raise RefusedRequest(409, "turn_in_progress", "Wait for noc_lead's answer first (or press Stop following).")
        if last and last["ui_state"] == "lead_unconfirmed" and self.now() - (last.get("terminal_at") or 0) < self.timing.unconfirmed_hold_s:
            raise RefusedRequest(409, "lead_may_be_working", "noc_lead may still be working on the last message. "
                                 "Don't send another for 5 minutes; watch the activity.")
        approvals = list(dict.fromkeys(case.get("approvals_seen") or []))
        wants_relay = bool(APPROVAL_TEXT.search(text))
        if not approvals:
            if wants_relay:
                raise RefusedRequest(409, "no_approval_known", "No approval is known for this case, so there is nothing "
                                     "to relay. Approvals are decided by a person, not by this panel.")
            return
        try:
            rows = {a: await asyncio.to_thread(self.aws.get_row, a) for a in approvals}
            suite = await self.suite_run()
        except AwsUnavailable as e:
            raise RefusedRequest(409, "approval_state_unknown", "This case has a change that needs a person's approval "
                                 f"({', '.join(approvals)}). {str(e).replace(' Chat still works.', '')} Without reading the approvals table the panel won't "
                                 "send messages that could make noc_lead apply it.")
        status = {a: (r or {}).get("status", "MISSING") for a, r in rows.items()}
        pending = [a for a, s in status.items() if s == "PENDING"]
        approved = [a for a, s in status.items() if s == "APPROVED"]
        if wants_relay and pending:
            raise RefusedRequest(409, "approval_still_pending", f"{', '.join(pending)} is still waiting for a person. "
                                 "The panel relays a decision only after the approvals table shows it.", approvals=status)
        if approved and suite.get("run_id") and suite.get("blocks"):
            raise RefusedRequest(409, "judged_run_active", f"{', '.join(approved)} is approved, and new changes are filed "
                                 f"under suite run {suite['run_id']} ({suite['judged_label']}). Relaying now would let "
                                 "noc_lead apply it into that run's records. Wait for the adhoc fix (W4) or for the lead.",
                                 approvals=status)

    async def follow_up(self, case_id: str, text: str) -> dict:
        self.check_start()
        await self.check_follow_up(case_id, text)
        case = self.store.cases[case_id]
        last = self.store.turns.get(case["turn_ids"][-1]) if case["turn_ids"] else None
        if last and last["ui_state"] not in TERMINAL:  # re-checked after the awaits above
            raise RefusedRequest(409, "turn_in_progress", "Wait for noc_lead's answer first.")
        turn = self._turn("continue", text, case_id)
        self.save()
        self.spawn(f"send:{turn['turn_id']}", self.send_turn(turn["turn_id"]))
        return self.turn_view(turn)

    def stop_following(self, turn_id: str) -> dict:
        turn = self.store.turns.get(turn_id)
        if turn is None:
            raise RefusedRequest(404, "unknown_turn", "No such message.")
        if turn["ui_state"] not in ACTIVE or turn["ui_state"] in ("queued", "sending"):
            raise RefusedRequest(409, "not_stoppable", "Only a message the panel is waiting on can be stopped.")
        self._finish(turn, "abandoned", detail="Stopped following on this laptop. The kit has no cancel: noc_lead "
                     "may still be working, and its activity stays visible.")
        task = self.tasks.get(f"poll:{turn_id}")
        if task is not None:
            task.cancel()
        return self.turn_view(turn)

    # ------------------------------------------------------------------ sending
    async def send_turn(self, turn_id: str) -> None:
        turn = self.store.turns[turn_id]
        turn["ui_state"] = "sending"
        turn["sent_at"] = self.now()
        self.save()  # before anything leaves the laptop
        if turn["kind"] == "start":
            tool, args = START, {"request": self.sent_text(turn)}
        else:
            tool, args = CONTINUE, {"case_id": turn["case_id"], "message": self.sent_text(turn)}
        try:
            out = await asyncio.to_thread(self.kit.call, tool, args, self.timing.send_timeout_s)
        except NotConfigured as e:
            self._finish(turn, "kit_error", detail=str(e))
            return
        self.apply_send(turn, out)

    def apply_send(self, turn: dict, out: Outcome) -> None:
        turn["gateway_date"] = out.gateway_date
        red = self.redactor()
        if out.kind == "ok":
            cid = out.data.get("case_id")
            state = out.data.get("state") if isinstance(out.data.get("state"), str) else None
            if isinstance(cid, str) and CASE.match(cid) and (turn["kind"] == "start" or cid == turn["case_id"]):
                if turn["kind"] == "continue":
                    turn["seen_running"] = state in RUNNING
                self._accept(turn, cid, state)
            else:
                self._finish(turn, "kit_error", detail="The kit answered without a case id.")
            return
        if out.kind == "before_send":
            self._finish(turn, "kit_unreachable", category=out.category)
        elif out.maybe_delivered:
            turn["ui_state"] = "outcome_unknown"
            turn["category"] = out.category or (f"HTTP {out.http_status}" if out.http_status else None)
            self.save()
            self.spawn(f"reconcile:{turn['turn_id']}", self.reconcile(turn["turn_id"]))
        elif out.kind == "denied":
            self._finish(turn, "refused_policy", deny_rule=deny_rule(out.text), detail=red.text(out.text)[:300])
        elif out.kind == "auth":
            left = self.config.load().token_seconds_left(self.now())
            self._finish(turn, "token_expired" if left is not None and left <= 0 else "refused_auth",
                         category=f"HTTP {out.http_status}")
        elif out.kind == "tool_error" and A2A_TROUBLE.search(out.text):
            self._finish(turn, "lead_unconfirmed", detail=red.text(out.text)[:300])
            if turn["kind"] == "start":  # find the case the kit opened before it tried to reach the lead
                self.spawn(f"reconcile:{turn['turn_id']}", self.reconcile(turn["turn_id"], find_case_only=True))
        else:
            self._finish(turn, "kit_error", detail=red.text(out.text)[:300] or out.kind)

    async def reconcile(self, turn_id: str, find_case_only: bool = False) -> None:
        turn = self.store.turns[turn_id]
        tool = START if turn["kind"] == "start" else CONTINUE
        sent = turn.get("sent_at") or turn["created_at"]
        start_ms = int((sent - 300) * 1000)  # absorb clock skew between this laptop and AWS
        deadline = sent + self.timing.reconcile_window_s
        while True:
            if not find_case_only and turn["ui_state"] != "outcome_unknown":
                return
            try:
                events = await asyncio.to_thread(self.aws.filter_events, turn["tag"], start_ms)
            except AwsUnavailable as e:
                if find_case_only:
                    return
                if turn["kind"] == "continue":  # the status flip can still prove it
                    turn["confirming"], turn["detail"] = True, str(e)
                    self._accept(turn, turn["case_id"], None)
                else:
                    self._finish(turn, "unconfirmed", detail=str(e))
                return
            found = find_send(events, turn["tag"], tool)
            if find_case_only:
                if found["opened_case"]:
                    turn["case_id"] = found["opened_case"]
                    self._ensure_case(found["opened_case"], turn)
                    self.save()
                    return
            else:
                r = found["response"]
                if r and r["refused"]:
                    self._finish(turn, "refused_policy", deny_rule=r.get("rule"))
                    return
                if r and turn["kind"] == "start" and r["case_id"]:
                    self._accept(turn, r["case_id"], None)
                    return
                if r and turn["kind"] == "start":
                    self._finish(turn, "lead_unconfirmed", case_id=found["opened_case"])
                    return
                if r:  # a follow-up arrived; wait for its own task to show up
                    turn["confirming"] = True
                    self._accept(turn, turn["case_id"], None)
                    return
            if self.now() >= deadline:
                if find_case_only:
                    return
                if found["opened_case"]:
                    self._finish(turn, "lead_unconfirmed", case_id=found["opened_case"])
                else:
                    self._finish(turn, "not_received")
                return
            await self.sleep(self.timing.reconcile_every_s)

    # ------------------------------------------------------------------ polling
    async def poll_loop(self, turn_id: str) -> None:
        ok_i = err_i = 0
        delay = self.timing.poll_delays[0]
        while True:
            await self.sleep(delay)
            turn = self.store.turns.get(turn_id)
            if not turn or turn["ui_state"] != "running":
                return
            try:
                out = await asyncio.to_thread(self.kit.call, STATUS, {"case_id": turn["case_id"]},
                                              self.timing.status_timeout_s)
            except NotConfigured:
                out = Outcome("before_send", category="not_configured")
            if turn["ui_state"] != "running":  # stopped while we waited
                return
            if self.apply_status(turn, out):
                err_i += 1
                delay = self.timing.error_delays[min(err_i - 1, len(self.timing.error_delays) - 1)]
            else:
                err_i = 0
                ok_i += 1
                delay = self.timing.poll_delays[min(ok_i, len(self.timing.poll_delays) - 1)]

    def apply_status(self, turn: dict, out: Outcome) -> bool:
        """Apply one case_status answer. Returns True when it was an error (the caller backs off)."""
        now = self.now()
        elapsed = now - (turn.get("accepted_at") or turn.get("sent_at") or now)
        turn["polls"] += 1
        turn["overdue"] = elapsed > self.timing.overdue_after_s
        red = self.redactor()
        transient = out.kind in ("before_send", "after_send", "server", "throttled", "unparseable", "rpc_error",
                                 "client_error", "is_error") or (out.kind == "tool_error" and A2A_TROUBLE.search(out.text)
                                                                 and "unknown case" not in out.text)
        if transient:
            turn["status_errors"] += 1
            turn["status_unavailable"] = turn["status_errors"] >= 5
            if elapsed > self.timing.lost_after_s:
                self._finish(turn, "lost")
            return True
        turn["status_errors"], turn["status_unavailable"] = 0, False
        if out.kind == "auth":
            left = self.config.load().token_seconds_left(now)
            self._finish(turn, "token_expired" if left is not None and left <= 0 else "refused_auth",
                         category=f"HTTP {out.http_status}")
            return False
        if out.kind == "tool_error" and "unknown case" in out.text:
            self._finish(turn, "failed", failed_cause="kit_lost_case")
            return False
        if out.kind != "ok":
            self._finish(turn, "kit_error", detail=red.text(out.text)[:300] or out.kind)
            return False
        data = out.data
        state = data.get("state") if isinstance(data.get("state"), str) else ""
        waiting = parse_waiting(data.get("waiting_approvals"))
        turn["waiting_approvals"] = waiting
        case = self.store.cases.get(turn["case_id"])
        if case is not None:
            seen = case.setdefault("approvals_seen", [])
            seen.extend(w["approval_id"] for w in waiting if w["approval_id"] not in seen)
        turn["a2a_state"] = state[:60]
        if state in RUNNING:
            turn["seen_running"] = True
            if elapsed > self.timing.lost_after_s:
                self._finish(turn, "lost")
            else:
                self.save()
            return False
        raw = data.get("reply") if isinstance(data.get("reply"), str) else ""
        digest = sha256(raw)
        if turn["kind"] == "continue" and not turn["seen_running"] and digest == turn.get("prev_reply_sha256"):
            # still the previous task: the kit had not swapped in this message's task
            if elapsed >= self.timing.continue_confirm_s:
                self._finish(turn, "lead_unconfirmed", detail="The kit still shows the previous answer.")
            return False
        text, n = red.clean(raw)
        turn["reply_sha256"] = digest
        turn["reply"] = {"text": text[:20000], "truncated": len(text) > 20000, "redactions": n,
                         "empty": not raw.strip(), "received_at": iso(now)}
        if waiting:
            self._finish(turn, "awaiting_approval")
        elif state == "TASK_STATE_COMPLETED":
            self._finish(turn, "answered")
        elif state == "TASK_STATE_INPUT_REQUIRED":
            self._finish(turn, "needs_input")
        elif state in FAILED_STATES:
            near = elapsed >= self.timing.lead_timeout_s - 15
            self._finish(turn, "failed", failed_cause="stopped_near_limit" if near else "not_reported")
        else:
            self._finish(turn, "failed", failed_cause=f"unrecognised_state:{state[:40] or 'none'}")
        return False

    # ------------------------------------------------------------------ restart
    def resume(self) -> None:
        """After a restart: never resend; reconcile what may have left; follow what is still running."""
        now = self.now()
        for turn in list(self.store.turns.values()):
            s = turn["ui_state"]
            if s == "queued":
                self._finish(turn, "cancelled_by_restart")
            elif s == "sending":
                turn["ui_state"] = "outcome_unknown"
                self.save()
                self.spawn(f"reconcile:{turn['turn_id']}", self.reconcile(turn["turn_id"]))
            elif s == "outcome_unknown":
                self.spawn(f"reconcile:{turn['turn_id']}", self.reconcile(turn["turn_id"]))
            elif s == "running":
                if now - (turn.get("accepted_at") or now) > self.timing.lost_after_s:
                    self._finish(turn, "lost")
                else:
                    self.spawn(f"poll:{turn['turn_id']}", self.poll_loop(turn["turn_id"]))
            else:
                turn["recorded"] = True

    # ------------------------------------------------------------------ AWS-backed reads
    def _cached(self, key: str, ttl: float):
        hit = self._cache.get(key)
        return hit[1] if hit and self.now() - hit[0] < ttl else None

    def _put(self, key: str, value):
        self._cache[key] = (self.now(), value)
        return value

    def judged_runs(self) -> set[str]:
        runs: set[str] = set()
        evid = Path(self.env.get("SPECTRUM_EVIDENCE_DIR") or REPO / "evidence")
        for p in (REPO / "judging.json", evid / "judging.json"):
            try:
                runs |= {r for r in json.loads(p.read_text()).get("run_ids") or [] if isinstance(r, str)}
            except (OSError, ValueError, AttributeError):
                continue
        return runs

    async def suite_run(self) -> dict:
        """The run the RAN tools file new cases under (t3-active, t3/tools/app.py:399). Raises AwsUnavailable."""
        hit = self._cached("suite", self.timing.aws_cache_s)
        if hit is not None:
            return hit
        row = await asyncio.to_thread(self.aws.get_row, "t3-active")
        if not row or not row.get("run_id"):
            return self._put("suite", {"run_id": None, "blocks": False})
        run = str(row["run_id"])[:128]
        judged = run in self.judged_runs()
        allowed = {r.strip() for r in (self.env.get("ASKNOC_NONJUDGED_RUNS") or "").split(",") if r.strip()}
        try:
            age = max(0.0, self.now() - float(row.get("updated_at")))
        except (TypeError, ValueError):
            age = None
        return self._put("suite", {
            "run_id": run, "age_s": age, "judged": judged,
            "judged_label": "listed for judging" if judged else "a suite run, treated as judged",
            "blocks": run not in allowed})

    async def activity(self, case_id: str) -> dict:
        case = self.store.cases.get(case_id)
        if case is None:
            raise RefusedRequest(404, "unknown_case", "This panel did not start that case.")
        if self.offline:
            return {"available": False, "message": "Offline: no AWS reads."}
        hit = self._cached(f"act:{case_id}", self.timing.activity_cache_s)
        if hit is not None:
            return hit
        start_ms = int((case["started_at"] - 300) * 1000)
        try:
            events = await asyncio.to_thread(self.aws.filter_events, case_id, start_ms)
        except AwsUnavailable as e:
            return self._put(f"act:{case_id}", {"available": False, "message": str(e)})
        return self._put(f"act:{case_id}", {
            "available": True, "label": "LIVE", "source": "the kit's audit log in CloudWatch, filtered by case id",
            "fetched_at": iso(self.now()), "lag_note": "Log lines can arrive 5 to 30 s after the call.",
            "events": parse_events(events, case_id, self.redactor())})

    async def health(self, loopback: bool) -> dict:
        base = {"checked_at": iso(self.now()), "measured": MEASURED,
                "timing": {"lead_timeout_s": self.timing.lead_timeout_s,
                           "lead_timeout_source": "Agent Builder setting listed in t3/agents/BUILDER_CHECKLIST.md:34 (not verified)",
                           "lost_after_s": self.timing.lost_after_s},
                "fallback": "If the kit is down, the lead can still talk to noc_lead in the Agent Builder chat on the "
                            "team laptop (ask the lead for the address). That chat is not followed by this panel."}
        if not loopback:
            return {**base, "mode": "loopback_only", "live_possible": False, "checks": []}
        if self.offline:
            return {**base, "mode": "offline", "live_possible": False,
                    "checks": [{"id": "offline", "state": "unknown", "detail": "Offline mode: no calls to the kit or AWS."}]}
        checks: list[dict] = []

        async def check(cid: str, fn):
            try:
                state, detail = await fn()
            except Exception as e:  # noqa: BLE001 - health always answers
                state, detail = "unknown", f"Check failed ({type(e).__name__})."
            checks.append({"id": cid, "state": state, "detail": detail})

        view = self.config.load()

        async def c_config():
            if not view.configured:
                return "fail", view.problem
            if view.stale:
                return "warn", "kit-state.json is being rewritten (a kit deploy?); using the last good values."
            return "pass", "Kit folder set; the Gateway address and the app token are readable."

        async def c_token():
            if not view.configured:
                return "unknown", "No token without the kit folder."
            left = view.token_seconds_left(self.now())
            if view.claims.get("role") != "app":
                return "fail", "The token in tokens/team_app.txt is not the app's token."
            if left is None:
                return "warn", "The token has no expiry claim."
            if left <= 0:
                return "fail", "The app token has expired. The lead must re-run the kit's deploy."
            return ("warn" if left < 86400 else "pass"), f"App token valid for {int(left // 3600)} h."

        async def c_gateway():
            if not view.configured:
                return "unknown", "Not checked."
            hit = self._cached("gw", self.timing.probe_cache_s)
            out = hit or self._put("gw", await asyncio.to_thread(self.kit.list_tools, self.timing.health_timeout_s))
            if out.kind == "ok":
                missing = [t for t in (START, CONTINUE, STATUS) if t not in out.data.get("tools", [])]
                if missing:
                    return "fail", f"The Gateway does not list {', '.join(missing)} for the app."
                return "pass", "The Gateway accepts the app token and lists start_case, continue_case and case_status."
            if out.kind == "auth":
                return "fail", f"The Gateway refused the app token (HTTP {out.http_status})."
            if out.kind == "before_send":
                return "fail", f"Can't reach the Gateway ({out.category})."
            return "fail", f"The Gateway answered unexpectedly ({out.kind})."

        async def c_front():
            if not view.configured:
                return "unknown", "Not checked."
            hit = self._cached("front", self.timing.probe_cache_s)
            out = hit or self._put("front", await asyncio.to_thread(self.kit.call, STATUS, {"case_id": PROBE_CASE},
                                                                    self.timing.health_timeout_s))
            if out.kind == "tool_error" and "unknown case" in out.text:
                return "pass", "case_status on the kit's probe id answered 'unknown case': Cedar allowed the app, and the kit's function and table answered."
            if out.kind == "denied":
                return "fail", "Cedar refused the app's case_status call."
            return "fail", f"The front door answered unexpectedly ({out.kind})."

        async def c_builder():
            return "unknown", (f"Can't be checked without starting a case. Measured: a {MEASURED['kind']} took about "
                               f"{MEASURED['seconds']} s on {MEASURED['started'][:10]} ({MEASURED['case_id']}).")

        async def c_suite():
            try:
                s = await self.suite_run()
            except AwsUnavailable as e:
                return "unknown", f"Can't read which suite run new cases are filed under. {e}"
            if not s.get("run_id"):
                return "pass", "No suite run is marked active: chat cases are filed as adhoc-<case>."
            age = f"set {int(s['age_s'] // 60)} min ago" if s.get("age_s") is not None else "age unknown"
            return "warn", (f"New cases are filed under suite run {s['run_id']} ({age}; {s['judged_label']}). "
                            "A chat case's tool calls land in that run; relaying an approval is blocked.")

        async def c_aws():
            hit = self._cached("aws", self.timing.aws_cache_s)
            if hit is None:
                try:
                    await asyncio.to_thread(self.aws.probe)
                    hit = self._put("aws", ("pass", "CloudWatch reads work: the swarm's activity can be shown."))
                except AwsUnavailable as e:
                    hit = self._put("aws", ("warn", str(e)))
            return hit

        for cid, fn in (("config", c_config), ("token", c_token), ("gateway", c_gateway), ("front_door", c_front),
                        ("agent_builder", c_builder), ("suite_run", c_suite), ("aws_reads", c_aws)):
            await check(cid, fn)
        by = {c["id"]: c["state"] for c in checks}
        if not view.configured:
            mode = "not_configured"
        elif any(by.get(k) == "fail" for k in ("config", "token", "gateway", "front_door")):
            mode = "down"
        elif any(c["state"] in ("warn", "unknown") for c in checks if c["id"] != "agent_builder"):
            mode = "degraded"
        else:
            mode = "ok"
        return {**base, "mode": mode, "live_possible": mode in ("ok", "degraded"), "checks": checks,
                "real_gateway": view.real_gateway if view.configured else None}

    # ------------------------------------------------------------------ views
    def label(self, turn: dict) -> str:
        view = self.config.load()
        if view.configured and not view.real_gateway:
            return "TEST"  # a local stand-in Gateway (ASKNOC_ALLOW_TEST_GATEWAY=1), never the real kit
        if turn.get("recorded") or (turn.get("terminal_at") and self.now() - turn["terminal_at"] > self.timing.lost_after_s):
            return "RECORDED"
        return "LIVE"

    def turn_view(self, turn: dict) -> dict:
        now = self.now()
        start = turn.get("accepted_at") or turn.get("sent_at")
        end = turn.get("terminal_at") or now
        keys = ("turn_id", "kind", "case_id", "text", "tag", "ui_state", "a2a_state", "gateway_date", "reply",
                "waiting_approvals", "failed_cause", "deny_rule", "detail", "category", "overdue",
                "status_unavailable", "polls")
        v = {k: turn.get(k) for k in keys}
        v.update(trace_id=turn.get("case_id"), created_at=iso(turn["created_at"]), sent_at=iso(turn.get("sent_at")),
                 accepted_at=iso(turn.get("accepted_at")), terminal_at=iso(turn.get("terminal_at")),
                 elapsed_s=round(end - start, 1) if start else None, label=self.label(turn),
                 terminal=turn["ui_state"] in TERMINAL,
                 can_stop=turn["ui_state"] in ("running", "outcome_unknown"))
        return v

    def case_view(self, case_id: str) -> dict:
        case = self.store.cases.get(case_id)
        if case is None:
            raise RefusedRequest(404, "unknown_case", "This panel did not start that case.")
        turns = [self.store.turns[t] for t in case["turn_ids"] if t in self.store.turns]
        last = turns[-1] if turns else None
        can, why = True, None
        if self.offline:
            can, why = False, "Offline."
        elif last and last["ui_state"] not in TERMINAL:
            can, why = False, "Waiting for noc_lead's answer."
        elif last and last["ui_state"] == "lead_unconfirmed" and now_minus(self, last) < self.timing.unconfirmed_hold_s:
            can, why = False, "noc_lead may still be working on the last message."
        return {"case_id": case_id, "trace_id": case_id, "trace_id_note": "The kit uses the case id as the trace id.",
                "started_at": iso(case["started_at"]), "label": self.label(last) if last else "LIVE",
                "ui_state": last["ui_state"] if last else None, "turns": [self.turn_view(t) for t in turns],
                "approvals_seen": case.get("approvals_seen") or [], "can_continue": can, "cannot_continue_reason": why,
                "trace_cli": f"python -m t3.suite.trace check --case {case_id} --no-write"}

    def cases(self) -> list[dict]:
        rows = []
        for cid, case in self.store.cases.items():
            first = self.store.turns.get(case["turn_ids"][0]) if case["turn_ids"] else None
            last = self.store.turns.get(case["turn_ids"][-1]) if case["turn_ids"] else None
            rows.append({"case_id": cid, "started_at": iso(case["started_at"]),
                         "first_text": (first or {}).get("text", "")[:80], "ui_state": (last or {}).get("ui_state"),
                         "label": self.label(last) if last else "LIVE", "_t": case["started_at"]})
        rows.sort(key=lambda r: r["_t"], reverse=True)
        for r in rows:
            r.pop("_t")
        return rows[:50]

    def pending_turns(self) -> list[dict]:
        """Starts that have no case id yet (queued, sending, outcome_unknown, or finished without one)."""
        rows = [t for t in self.store.turns.values() if t["kind"] == "start" and not t.get("case_id")]
        rows.sort(key=lambda t: t["created_at"], reverse=True)
        return [self.turn_view(t) for t in rows[:10]]


def now_minus(svc: AskNoc, turn: dict) -> float:
    return svc.now() - (turn.get("terminal_at") or 0)
