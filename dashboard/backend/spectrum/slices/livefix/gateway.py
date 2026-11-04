"""Live Fix's door to the kit: Ask the NOC's KitClient (same settings, token handling and failure classes), with
one more tool allowed: ran___seed_case, which Cedar permits to the app's token alone and refuses to every agent
(t3/policies team13_ran_seed_app_only / _never_agents). Nothing else: the app can't approve or apply.
"""

from __future__ import annotations

import json
from typing import Any

from ..asknoc.kit import CONTINUE, STATUS, KitClient, Outcome

SEED = "ran___seed_case"
LIVEFIX_TOOLS = (SEED, CONTINUE, STATUS)


class GatewayFailed(RuntimeError):
    def __init__(self, tool: str, out: Outcome):
        super().__init__(f"{tool}: {out.kind}" + (f" ({out.category})" if out.category else ""))
        self.tool, self.out = tool, out


class LiveFixKit(KitClient):
    def call(self, tool: str, arguments: dict, timeout: float) -> Outcome:  # type: ignore[override]
        if tool not in LIVEFIX_TOOLS:
            raise ValueError(f"tool {tool!r} is not one Live Fix may call")
        return self._send("tools/call", {"name": tool, "arguments": arguments}, timeout)


def ok_or_raise(tool: str, out: Outcome) -> Outcome:
    if out.kind != "ok":
        raise GatewayFailed(tool, out)
    return out


def parse_waiting(status: dict[str, Any]) -> list[dict[str, Any]]:
    """The case's waiting approvals from front___case_status: [{approval_id, change_text, item_id, reason}]."""
    out = []
    for w in status.get("waiting_approvals") or []:
        if not isinstance(w, dict):
            continue
        raw = w.get("change")
        try:
            body = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except ValueError:
            body = {}
        if not isinstance(body, dict):
            body = {}
        out.append({"approval_id": w.get("approval_id"), "change_text": body.get("change"),
                    "item_id": body.get("item_id"), "reason": body.get("reason")})
    return out
