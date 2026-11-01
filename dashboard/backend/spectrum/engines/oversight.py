"""Human oversight: approval queue with SLA timers, escalation ladder, four-eyes for critical tier,
signed human decisions.

AWS mode: each approval is a Step Functions task token (waitForTaskToken); decisions call
SendTaskSuccess/SendTaskFailure; escalation pages via SNS / Systems Manager Incident Manager;
approvers authenticate with Cognito and sign with their KMS key.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from ..core import Bus, Clock, KeyRing, canonical, sha256_hex

if TYPE_CHECKING:
    from .ledger import Ledger

HUMANS = {
    "a.rivera": {"id": "a.rivera", "name": "A. Rivera", "role": "NOC engineer", "level": "L1"},
    "j.okafor": {"id": "j.okafor", "name": "J. Okafor", "role": "On-call network lead", "level": "L2"},
    "m.chen": {"id": "m.chen", "name": "M. Chen", "role": "Duty manager", "level": "L3"},
}
LADDER = [
    {"level": "L1", "who": "a.rivera", "sla_s": 120, "label": "NOC engineer"},
    {"level": "L2", "who": "j.okafor", "sla_s": 300, "label": "On-call lead"},
    {"level": "L3", "who": "m.chen", "sla_s": 600, "label": "Duty manager"},
]


class Oversight:
    def __init__(self, clock: Clock, bus: Bus, keys: KeyRing, ledger: "Ledger"):
        self.clock = clock
        self.bus = bus
        self.keys = keys
        self.ledger = ledger
        self.items: dict[str, dict] = {}
        self.callbacks: dict[str, Callable[[dict], None]] = {}
        self.counter = 311
        self.on_duty = "a.rivera"

    def create(self, title: str, tier: str, requested_by: str, incident: str | None, plan: dict,
               preflight: dict, reasons: list[dict], dissent: list[dict], model_opinions: list[dict],
               summary: str, on_decided: Callable[[dict], None] | None = None, sla_scale: float = 1.0) -> dict:
        aid = f"APR-{self.counter}"
        self.counter += 1
        item = {
            "id": aid, "title": title, "tier": tier, "requested_by": requested_by, "incident": incident,
            "summary": summary, "plan": plan, "preflight": preflight, "reasons": reasons, "dissent": dissent,
            "model_opinions": model_opinions, "created_at": self.clock.iso(), "created_t": self.clock.t,
            "level_index": 0, "level_started_t": self.clock.t, "status": "pending",
            "required": 2 if tier == "critical" else 1, "approvals": [], "history": [],
            "blast_radius_pct": preflight.get("blast_radius_pct"), "sla_scale": sla_scale,
        }
        self.items[aid] = item
        if on_decided:
            self.callbacks[aid] = on_decided
        rec = self.ledger.append("approval.requested", requested_by, "operate",
                                 {"approval": aid, "title": title, "tier": tier, "incident": incident,
                                  "blast_radius_pct": item["blast_radius_pct"],
                                  "policies": [x["policy"] for x in reasons]})
        item["history"].append({"ts": item["created_at"], "event": "requested", "by": requested_by,
                                "ledger_seq": rec["seq"]})
        self.bus.emit("approval.requested", "operate", f"{aid} needs a human: {title}", self.view(item), "warn",
                      requested_by, ledger={"seq": rec["seq"], "hash": rec["hash"]})
        return item

    def decide(self, aid: str, decision: str, by: str = "a.rivera", device: str = "console",
               reason: str = "", changes: dict | None = None) -> dict:
        item = self.items[aid]
        if item["status"] not in ("pending",):
            return item
        human = HUMANS.get(by, HUMANS["a.rivera"])
        if decision in ("approved", "approved_with_changes"):
            if any(a["by"] == human["id"] for a in item["approvals"]):
                return item  # same person cannot approve twice (four-eyes)
            payload = {"approval": aid, "decision": "approved", "by": human["id"], "device": device,
                       "changes": changes or {}, "reason": reason}
            digest = sha256_hex(canonical(payload))
            sig = self.keys.sign(f"kms:spectrum/human/{human['id']}", digest)
            rec = self.ledger.append("approval.decision", f"human:{human['id']}", "operate", payload,
                                     key_id=f"kms:spectrum/human/{human['id']}")
            item["approvals"].append({"by": human["id"], "name": human["name"], "at": self.clock.iso(),
                                      "device": device, "sig": sig, "ledger_seq": rec["seq"], "changes": changes or {}})
            item["history"].append({"ts": self.clock.iso(), "event": f"approved by {human['name']} ({device})",
                                    "by": human["id"], "ledger_seq": rec["seq"]})
            if len(item["approvals"]) >= item["required"]:
                item["status"] = "approved"
                item["decided_at"] = self.clock.iso()
                item["ttd_s"] = round(self.clock.t - item["created_t"], 1)
            self.bus.emit("approval.decision", "operate",
                          f"{aid} approved by {human['name']} on {device}"
                          + ("" if item["status"] == "approved" else f" ({len(item['approvals'])}/{item['required']})"),
                          self.view(item), "notice", f"human:{human['id']}",
                          ledger={"seq": rec["seq"], "hash": rec["hash"]})
        elif decision in ("rejected", "escalated"):
            payload = {"approval": aid, "decision": decision, "by": human["id"], "device": device, "reason": reason}
            rec = self.ledger.append("approval.decision", f"human:{human['id']}", "operate", payload,
                                     key_id=f"kms:spectrum/human/{human['id']}")
            item["history"].append({"ts": self.clock.iso(), "event": f"{decision} by {human['name']}: {reason}",
                                    "by": human["id"], "ledger_seq": rec["seq"]})
            if decision == "rejected":
                item["status"] = "rejected"
                item["decided_at"] = self.clock.iso()
            else:
                self._escalate(item, reason or "escalated by operator", manual=True)
            self.bus.emit("approval.decision", "operate", f"{aid} {decision} by {human['name']}", self.view(item),
                          "warn", f"human:{human['id']}", ledger={"seq": rec["seq"], "hash": rec["hash"]})
        if item["status"] in ("approved", "rejected"):
            cb = self.callbacks.pop(aid, None)
            if cb:
                cb(item)
        return item

    def _escalate(self, item: dict, reason: str, manual: bool = False) -> None:
        if item["level_index"] >= len(LADDER) - 1:
            return
        item["level_index"] += 1
        item["level_started_t"] = self.clock.t
        lvl = LADDER[item["level_index"]]
        rec = self.ledger.append("escalation", "oversight", "operate",
                                 {"approval": item["id"], "to": lvl["level"], "who": lvl["who"], "reason": reason})
        item["history"].append({"ts": self.clock.iso(), "event": f"escalated to {lvl['level']} "
                                f"({HUMANS[lvl['who']]['name']}): {reason}", "by": "oversight",
                                "ledger_seq": rec["seq"]})
        self.bus.emit("escalation", "operate", f"{item['id']} escalated to {lvl['level']} "
                      f"{HUMANS[lvl['who']]['name']}: {reason}", self.view(item), "critical", "oversight",
                      ledger={"seq": rec["seq"], "hash": rec["hash"]})

    def tick(self) -> None:
        for item in self.items.values():
            if item["status"] != "pending":
                continue
            sla = LADDER[item["level_index"]]["sla_s"] * item.get("sla_scale", 1.0)
            if self.clock.t - item["level_started_t"] > sla:
                if item["level_index"] < len(LADDER) - 1:
                    self._escalate(item, f"SLA {sla}s breached")
                else:
                    item["status"] = "expired"

    def view(self, item: dict) -> dict:
        lvl = LADDER[item["level_index"]]
        out = {k: v for k, v in item.items() if k != "created_t" and k != "level_started_t"}
        out["created_t"] = round(item["created_t"], 2)
        out["level_started_t"] = round(item["level_started_t"], 2)
        out["level"] = lvl["level"]
        out["assignee"] = HUMANS[lvl["who"]]
        sla = lvl["sla_s"] * item.get("sla_scale", 1.0)
        out["sla_deadline_t"] = round(item["level_started_t"] + sla, 2)
        out["sla_s"] = sla
        return out

    def snapshot(self) -> dict:
        items = sorted(self.items.values(), key=lambda x: x["created_t"], reverse=True)
        return {"items": [self.view(i) for i in items], "ladder": LADDER, "humans": HUMANS, "on_duty": self.on_duty}
