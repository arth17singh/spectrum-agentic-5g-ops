"""Offline fakes: a Gateway that decides like the intended Cedar rules and writes checkpoint and span rows,
DynamoDB, a policy client, and an evidence writer. No network, no AWS."""
from __future__ import annotations

import email.utils
import json
import re
import uuid

from t3.suite.common import change_key

SPECIALISTS = ("ran_monitor", "ran_optimizer", "ran_explainer")
PERMITS = {
    "noc_lead": {"swarm___ask_agent", "ran___observe", "ran___apply_change", "case___open_case"},
    "ran_monitor": {"ran___observe", "case___open_case"},
    "ran_optimizer": {"ran___propose", "ran___predict", "ran___preflight", "case___open_case"},
    "ran_explainer": {"nlm___ask", "case___open_case"},
    "team_app": {"front___start_case", "front___continue_case", "front___case_status"},
}
GOVERNED_PREFIXES = ("ran___", "nlm___", "swarm___")
INJECTION = re.compile(r"ignore all previous instructions", re.I)
TOOLS_GROUP = "/aws/lambda/team13-ran-tools"


class Clock:
    def __init__(self, start=1_793_650_000.0):
        self.now = start

    def __call__(self):
        return self.now

    def sleep(self, s):
        self.now += s


class FakeEvlib:
    def __init__(self):
        self.calls = []
        self.declared, self.records, self.results = [], {}, []

    def write_declared(self, d):
        self.calls.append(("declared", d["run_id"]))
        self.declared.append(d)
        return f"declared/{d['run_id']}.json"

    def append_records(self, run_id, kind, recs):
        self.calls.append(("records", kind))
        self.records.setdefault((run_id, kind), []).extend(recs)

    def write_result(self, r):
        self.calls.append(("result", r["run_id"]))
        self.results.append(r)


class FakeDDB:
    """Low-level client shape (TableName, Key with type descriptors)."""

    def __init__(self):
        self.tables = {}
        self.writes = []

    def get_item(self, TableName, Key, ConsistentRead=False):
        k = next(iter(Key.values()))["S"]
        item = self.tables.get(TableName, {}).get(k)
        return {"Item": {a: {"S": str(v)} for a, v in item.items()}} if item else {}

    def put_item(self, TableName, Item):
        self.writes.append((TableName, "put", next(iter(Item["pk"].values())) if "pk" in Item else None))
        key_attr = next(iter(Item))
        self.tables.setdefault(TableName, {})[Item[key_attr]["S"] if "S" in Item[key_attr] else Item[key_attr]["N"]] = {
            a: next(iter(v.values())) for a, v in Item.items()}

    def update_item(self, **kw):  # the runner must never call this
        self.writes.append((kw.get("TableName"), "update", None))
        raise AssertionError("update_item called")

    # helpers for the fake Gateway and the test's "person"
    def approvals(self):
        return self.tables.setdefault("team13-approvals", {})


class NotFound(Exception):
    pass


class FakePolicyClient:
    def __init__(self, clock, propagate_s=6.0):
        self.clock, self.propagate_s = clock, propagate_s
        self.policies = {}

    def create_policy(self, policyEngineId, name, validationMode, definition):
        pid = "pol-" + uuid.uuid4().hex[:8]
        self.policies[pid] = {"name": name, "created": self.clock(), "statement": definition["policy"]["statement"],
                              "deleted": False}
        return {"policyId": pid, "name": name, "status": "CREATING", "createdAt": self.clock()}

    def get_policy(self, policyEngineId, policyId):
        p = self.policies.get(policyId)
        if p is None or p["deleted"]:
            raise NotFound(f"ResourceNotFoundException: {policyId}")
        return {"policyId": policyId, "status": "ACTIVE" if self.clock() - p["created"] >= self.propagate_s
                else "CREATING"}

    def delete_policy(self, policyEngineId, policyId):
        self.policies[policyId]["deleted"] = True

    def forbids(self, tool):
        return any(not p["deleted"] and self.clock() - p["created"] >= self.propagate_s and f'"{tool}"' in p["statement"]
                   for p in self.policies.values())


class FakeGateway:
    """A post() for common.Gateway. Decides each call like the intended rules and logs like the kit."""

    def __init__(self, clock, ddb, policy=None, latency_s=0.4, flip=None):
        self.clock, self.ddb, self.policy, self.latency_s = clock, ddb, policy, latency_s
        self.flip = flip or set()      # tools whose decision we deliberately get wrong (for negative tests)
        self.posts = []                # (time, agent, tool)
        self.cp_rows, self.span_rows, self.tool_rows = [], [], []
        self.cases = []
        self.apply_noop = False   # True: the approval is consumed but our tools never apply (a bypass)

    def _row(self, rows, message, group):
        rows.append({"@timestamp": int(self.clock() * 1000), "@message": json.dumps(message), "@log": group,
                     "@logStream": "s1", "@ptr": f"ptr-{len(rows)}-{group}"})

    def decide(self, agent, tool, args):
        if tool not in PERMITS.get(agent, set()):
            return "DENY", "No policy applies to the request (denied by default)"
        if tool.startswith(GOVERNED_PREFIXES) and not str(args.get("case_id", "")).startswith("case-"):
            return "DENY", "Policy evaluation denied due to team13_require_case_id-x1"
        if tool == "swarm___ask_agent" and args.get("agent") not in SPECIALISTS:
            return "DENY", "No policy applies to the request (denied by default)"
        text = " ".join(str(args.get(k, "")) for k in ("task", "question", "request", "message"))
        if INJECTION.search(text):
            return "DENY", "No policy applies to the request (denied by default) guardrail PROMPT_INJECTION 0.98"
        if tool in ("ran___predict", "nlm___ask") and self.policy and self.policy.forbids(tool):
            return "DENY", f"Policy evaluation denied due to team13_ranpause_{tool.split('___')[1]}_1-x2"
        if tool == "ran___apply_change":
            if args.get("approval") != "APPROVED":
                return "DENY", "Policy evaluation denied due to team13_require_human_approval-x3"
        return "ALLOW", None

    def __call__(self, url, body, headers, timeout):
        req = json.loads(body)
        token = headers["Authorization"].split()[-1]
        agent = token.replace("tok-", "")
        tool, args = req["params"]["name"], dict(req["params"]["arguments"])
        self.posts.append((self.clock(), agent, tool))
        logged = {k: v for k, v in args.items() if k not in ("approval", "special_context")}
        cp = {"record": "checkpoint", "agent": agent, "method": "tools/call", "tool": tool,
              "case_id": args.get("case_id"), "arguments": logged}
        approval = None
        if tool == "ran___apply_change" and agent == "noc_lead":
            key = change_key(args)
            row = self.ddb.approvals().get(key)
            if row is None:
                self.ddb.approvals()[key] = {"pk": key, "status": "PENDING", "case_id": args.get("case_id", "")}
                row = self.ddb.approvals()[key]
            approval = row["status"]
            args["approval"] = approval
        self._row(self.cp_rows, {**cp, "phase": "request"}, "/team13/checkpoint")
        self.clock.sleep(self.latency_s / 2)
        decision, reason = self.decide(agent, tool, args)
        if tool in self.flip:
            decision = "ALLOW" if decision == "DENY" else "DENY"
        self._row(self.span_rows, {"traceId": "t-" + uuid.uuid4().hex[:6], "spanId": uuid.uuid4().hex[:6],
                                   "attributes": {"aws.agentcore.policy.authorization_decision": decision,
                                                  "aws.agentcore.policy.authorization_reason": reason or "permit",
                                                  "aws.agentcore.policy.determining_policies": reason or "permit",
                                                  "mcp.tool.name": tool}}, "aws/spans")
        self.clock.sleep(self.latency_s / 2)
        if decision == "DENY":
            if tool == "ran___apply_change" and "require_human_approval" in (reason or ""):
                key = change_key(args)
                status = self.ddb.approvals()[key]["status"]
                note = (f"approval {key} was already {status.lower()}" if status in ("APPLIED", "REJECTED")
                        else f"this change needs a person's approval. Approval id {key}, status {status}")
                answer = {"jsonrpc": "2.0", "id": 1, "result": {"isError": True, "content": [
                    {"type": "text", "text": "Refused by the checkpoint: " + note}]}}
                err = {"code": -32002, "message": reason}
                self._row(self.cp_rows, {**cp, "phase": "response", "refused": True, "error": err,
                                         "approval_id": key, "approval": status}, "/team13/checkpoint")
                return 200, self._headers(), json.dumps(answer)
            err = {"code": -32002, "message": reason}
            self._row(self.cp_rows, {**cp, "phase": "response", "refused": True, "error": err}, "/team13/checkpoint")
            return 200, self._headers(), json.dumps({"jsonrpc": "2.0", "id": 1, "error": err})
        payload = {"ok": True}
        if tool == "front___start_case":
            case = "case-" + uuid.uuid4().hex[:8]
            self.cases.append(case)
            payload = {"case_id": case, "state": "TASK_STATE_SUBMITTED"}
            cp["case_id"] = case
        if tool == "ran___apply_change":
            key = change_key(args)
            self.ddb.approvals()[key]["status"] = "APPLIED"
            payload = {"applied": True, "approval_id": key}
            if not self.apply_noop:  # our tools Lambda prints the change it applied (amendment A11)
                self._row(self.tool_rows, {"ts": self.clock(), "source": "team13-ran-tools",
                                           "record": "change_applied", "case_id": args.get("case_id"),
                                           "run_id": "from-case-row", "approval_id": key,
                                           "change_text": args.get("change"), "item_id": args.get("item_id")},
                          TOOLS_GROUP)
        if tool.startswith(("ran___", "nlm___")):
            self._row(self.tool_rows, {"ts": self.clock(), "source": "team13-ran-tools", "record": "ran_tool",
                                       "tool": tool, "case_id": args.get("case_id"), "ok": True}, TOOLS_GROUP)
        self._row(self.cp_rows, {**cp, "phase": "response", "refused": False, "error": None}, "/team13/checkpoint")
        return 200, self._headers(), "data: " + json.dumps({"jsonrpc": "2.0", "id": 1, "result": {
            "content": [{"type": "text", "text": json.dumps(payload)}]}}) + "\n\n"

    def _headers(self):
        return {"Date": email.utils.formatdate(self.clock(), usegmt=True), "Content-Type": "application/json"}


def tokens_env():
    return {f"T3_TOKEN_{a.upper()}": f"tok-{a}" for a in
            ("noc_lead", "ran_monitor", "ran_optimizer", "ran_explainer", "team_app")}
