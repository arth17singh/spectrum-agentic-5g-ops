"""Cedar forbid text that pauses one model version's tool, and a dry-run-by-default client.

Rule style follows the organisers' checkpoint kit (read, not vendored) and contract amendment 1
(A8): one Cedar statement per policy, created with ``validationMode=IGNORE_ALL_FINDINGS`` (the
validator calls every forbid "overly restrictive" by design) and
``definition={"policy": {"statement": ...}}``, exactly like the kit's ``pause nlm``:

    forbid (principal, action == AgentCore::Action::"ran___predict", resource == AgentCore::Gateway::"<arn>")
    when { !(context.input has model_version) || context.input.model_version == "<v>" };

The ``has`` guard matters: a forbid whose condition reads a missing attribute errors instead of
denying. With the guard, a call that omits ``model_version`` is refused too, so an agent cannot
reach the paused version by leaving the field out. A per-version tool (``version_field=None``)
gets the plain forbid with no condition.

Pause policies are named ``team13_ranpause_<model>_<version>_<n>`` (A8), never
``team13_pause_nlm*``: the kit's ``resume nlm`` deletes every policy with that prefix. The
kit's deploy only touches its own rule names, so a kit redeploy leaves these in place.

This module never builds a boto3 client. The lead injects one (``bedrock-agentcore-control``)
and passes ``dry_run=False`` with the real gateway ARN and engine id to make it live.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

LOOSE = "IGNORE_ALL_FINDINGS"
NAME_MAX = 48
NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
PAUSE_PREFIX = "team13_ranpause_"
# A1/A4: only ran___predict runs the predictor; its input names the version in model_version.
DEFAULT_TARGET = {"action": "ran___predict", "version_field": "model_version"}
# Syntactically valid placeholder used only in dry run; a live client refuses it.
DRY_RUN_GATEWAY = "arn:aws:bedrock-agentcore:us-east-2:000000000000:gateway/dry-run-placeholder"
DRY_RUN_ENGINE = "dry-run-engine"


@dataclass(frozen=True)
class PauseTarget:
    action: str                       # "ran___predict" (target___tool, as the kit names them)
    version_field: str | None = None  # input field that picks the version, or None for a per-version tool

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z0-9_-]+___[A-Za-z0-9_-]+", self.action):
            raise ValueError(f"tool action must look like target___tool, got {self.action!r}")
        if self.version_field is not None and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.version_field):
            raise ValueError(f"bad version field {self.version_field!r}")


def _cedar_string(s: str) -> str:
    """A Cedar string literal; refuses anything but plain id characters (no escaping games)."""
    if not re.fullmatch(r"[A-Za-z0-9_.:@/+-]+", s):
        raise ValueError(f"refusing to put {s!r} into a Cedar literal")
    return json.dumps(s)


def forbid_statement(gateway_arn: str, target: PauseTarget, version: str) -> str:
    """The Cedar forbid that pauses `version` on `target` (A8 text)."""
    head = (f'forbid (principal, action == AgentCore::Action::{_cedar_string(target.action)}, '
            f'resource == AgentCore::Gateway::{_cedar_string(gateway_arn)})')
    if target.version_field is None:
        return head + ";"
    f = target.version_field
    return (head + "\n"
            f"when {{ !(context.input has {f}) || context.input.{f} == {_cedar_string(version)} }};")


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_") or "x"


def policy_base(model_id: str, version: str) -> str:
    """Stable name prefix for one model version's pause policies (<= 37 chars, Cedar-name safe)."""
    base = f"{PAUSE_PREFIX}{_slug(model_id)}_{_slug(version)}"
    if len(base) > 37:
        digest = hashlib.sha256(f"{model_id}@{version}".encode()).hexdigest()[:10]
        base = f"{PAUSE_PREFIX}{digest}"
    return base


def policy_name(model_id: str, version: str, n: int) -> str:
    """A fresh name each time (names are unique in the account; deletion is asynchronous)."""
    name = f"{policy_base(model_id, version)}_{n}"
    if len(name) > NAME_MAX or not NAME_RE.match(name):
        raise ValueError(f"bad policy name {name!r}")
    return name


@dataclass
class PolicyClient:
    """Adds and removes the pause forbid through bedrock-agentcore-control. Dry run by default.

    In dry run nothing leaves the process: the client keeps the policies it *would* hold, so the
    gate's idempotence and the demo's "refused after pause" check behave as they would live, and
    every would-be API call is appended to ``calls`` for inspection.
    """

    gateway_arn: str = DRY_RUN_GATEWAY
    engine_id: str = DRY_RUN_ENGINE
    targets: dict[str, dict] = field(default_factory=dict)  # "model_id@version" or "model_id" -> PauseTarget kwargs
    ac: Any = None                                          # injected boto3 bedrock-agentcore-control client
    dry_run: bool = True
    clock: Callable[[], float] = time.time
    calls: list[dict] = field(default_factory=list)
    _dry_policies: dict[str, dict] = field(default_factory=dict)

    def __post_init__(self):
        if not self.dry_run:
            if self.ac is None:
                raise ValueError("a live PolicyClient needs an injected bedrock-agentcore-control client")
            if self.gateway_arn == DRY_RUN_GATEWAY or self.engine_id == DRY_RUN_ENGINE:
                raise ValueError("a live PolicyClient needs the real gateway ARN and policy engine id")

    def target_for(self, model_id: str, version: str) -> PauseTarget:
        spec = self.targets.get(f"{model_id}@{version}") or self.targets.get(model_id) or DEFAULT_TARGET
        return PauseTarget(**spec)

    def _list(self) -> list[dict]:
        if self.dry_run:
            return list(self._dry_policies.values())
        out, token = [], None
        while True:
            kw = {"policyEngineId": self.engine_id}
            if token:
                kw["nextToken"] = token
            page = self.ac.list_policies(**kw)
            out += page.get("policies", [])
            token = page.get("nextToken")
            if not token:
                return out

    def active_pauses(self, model_id: str, version: str) -> list[dict]:
        base = policy_base(model_id, version) + "_"
        return [p for p in self._list()
                if p.get("name", "").startswith(base) and "DELET" not in str(p.get("status", ""))]

    def is_paused(self, model_id: str, version: str) -> bool:
        return bool(self.active_pauses(model_id, version))

    def pause(self, model_id: str, version: str, reason: str) -> dict:
        existing = self.active_pauses(model_id, version)
        if existing:
            return {"policy_name": existing[0]["name"], "created": False, "dry_run": self.dry_run,
                    "statement": None}
        statement = forbid_statement(self.gateway_arn, self.target_for(model_id, version), version)
        name = policy_name(model_id, version, int(self.clock()))
        kwargs = {"policyEngineId": self.engine_id, "name": name, "validationMode": LOOSE,
                  "description": f"trust-gate pause of {model_id}@{version}: {reason}"[:4096],
                  "definition": {"policy": {"statement": statement}}}
        self.calls.append({"op": "create_policy", "kwargs": kwargs, "dry_run": self.dry_run})
        if self.dry_run:
            self._dry_policies[name] = {"name": name, "policyId": f"dry-{name}", "status": "ACTIVE",
                                        "statement": statement}
        else:
            created = self.ac.create_policy(**kwargs)
            self.ac.get_waiter("policy_active").wait(policyEngineId=self.engine_id,
                                                     policyId=created["policyId"])
        return {"policy_name": name, "created": True, "dry_run": self.dry_run, "statement": statement}

    def resume(self, model_id: str, version: str) -> dict:
        removed = []
        for p in self.active_pauses(model_id, version):
            kwargs = {"policyEngineId": self.engine_id, "policyId": p["policyId"]}
            self.calls.append({"op": "delete_policy", "kwargs": kwargs, "dry_run": self.dry_run})
            if self.dry_run:
                self._dry_policies.pop(p["name"], None)
            else:
                self.ac.delete_policy(**kwargs)
            removed.append(p["name"])
        return {"removed": removed, "dry_run": self.dry_run}
