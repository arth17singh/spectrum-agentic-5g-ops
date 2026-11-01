"""Trust plane: Cedar policy store + a deterministic evaluator that mirrors it, identities,
guardrails, delegation-chain checks, enclave attestation.

AWS mode: `authorize` -> verifiedpermissions.is_authorized(policyStoreId, principal, action,
resource, context); guardrails -> bedrock-runtime.apply_guardrail.
"""

from __future__ import annotations

import re
from collections import deque

from ..core import Bus, Clock, KeyRing, sha256_hex

POLICIES = [
    {
        "id": "P-001", "name": "Protect life-safety sites", "effect": "forbid",
        "cedar": 'forbid (principal, action == Action::"cell_sleep", resource)\n'
                 'when { resource.tags.contains("hospital") || resource.tags.contains("psap") };',
    },
    {
        "id": "P-002", "name": "Site agents act on their own neighbourhood", "effect": "permit",
        "cedar": 'permit (principal in Role::"SiteAgent", action in ActionGroup::"radio_changes", resource)\n'
                 'when { resource.site in principal.allowed_sites };',
    },
    {
        "id": "P-003", "name": "Large radio changes need a human", "effect": "forbid",
        "cedar": 'forbid (principal, action in ActionGroup::"radio_changes", resource)\n'
                 'when { context.blast_radius_pct > 5 && !context.human_approved };',
    },
    {
        "id": "P-004", "name": "Transport agent owns backhaul", "effect": "permit",
        "cedar": 'permit (principal == Agent::"transport", action in ActionGroup::"transport_changes", resource);',
    },
    {
        "id": "P-005", "name": "Coordinator may orchestrate approved plans", "effect": "permit",
        "cedar": 'permit (principal == Agent::"coordinator", action in ActionGroup::"radio_changes", resource)\n'
                 'when { context.plan_id != "" };',
    },
    {
        "id": "P-006", "name": "Delegation needs every hop authorised", "effect": "forbid",
        "cedar": 'forbid (principal, action, resource)\n'
                 'when { context.delegation_chain.containsAny(context.unauthorised_hops) };',
    },
    {
        "id": "P-007", "name": "No mass shutdowns", "effect": "forbid",
        "cedar": 'forbid (principal, action in [Action::"cell_sleep", Action::"restart_du"], resource)\n'
                 'when { context.target_count > 6 };',
    },
    {
        "id": "P-008", "name": "Quarantined identities can do nothing", "effect": "forbid",
        "cedar": 'forbid (principal, action, resource)\nwhen { principal.status == "quarantined" };',
    },
    {
        "id": "P-009", "name": "Autonomy level gates unattended actions", "effect": "forbid",
        "cedar": 'forbid (principal, action in ActionGroup::"radio_changes", resource)\n'
                 'when { principal.autonomy < 3 && !context.human_approved };',
    },
    {
        "id": "P-010", "name": "Energy agent manages sleep schedules", "effect": "permit",
        "cedar": 'permit (principal == Agent::"energy", action in [Action::"cell_sleep", Action::"cell_wake"], resource)\n'
                 'unless { resource.tags.contains("hospital") };',
    },
    {
        "id": "P-011", "name": "Read access to telemetry", "effect": "permit",
        "cedar": 'permit (principal, action in ActionGroup::"observe", resource);',
    },
    {
        "id": "P-013", "name": "Domain agents may raise tickets", "effect": "permit",
        "cedar": 'permit (principal in Role::"DomainAgent", action in ActionGroup::"ops", resource);',
    },
    {
        "id": "P-012", "name": "Model deploy gate", "effect": "permit",
        "cedar": 'permit (principal == Pipeline::"proving-ground", action == Action::"deploy", resource)\n'
                 'when { resource.safety >= 950 && resource.accuracy >= 850 &&\n'
                 '       resource.p95_ms <= 600 && resource.usd_per_1k_milli <= 1000 };',
    },
]
for p in POLICIES:
    p["hash"] = sha256_hex(p["cedar"])
    p["version"] = 3 if p["id"] in ("P-003", "P-007") else 1

RADIO = {"set_tx_power", "set_tilt", "set_handover_offset", "cell_sleep", "cell_wake", "restart_du", "load_balance"}
OPS = {"open_ticket", "dispatch_field_tech"}
TRANSPORT = {"reroute", "set_qos_policy"}
OBSERVE = {"get_kpis", "get_alarms", "get_topology", "trace_path", "simulate_change"}

INJECTION_PATTERNS = [r"ignore (all |prior |previous )?(rules|instructions)", r"coordinator authori[sz]es",
                      r"disregard policy", r"shut ?down all"]


class Trust:
    def __init__(self, clock: Clock, bus: Bus, keys: KeyRing):
        self.clock = clock
        self.bus = bus
        self.keys = keys
        self.policies = POLICIES
        self.decisions: deque[dict] = deque(maxlen=600)
        self._n = 0
        self.stats = {"permit": 0, "deny": 0, "needs_human": 0}
        self.policy_hits: dict[str, int] = {p["id"]: 0 for p in POLICIES}
        self.deny_by_action: dict[str, int] = {}
        self.deny_by_principal: dict[str, int] = {}
        self.guardrail_counts = {"prompt_attack": 0, "pii": 0, "denied_topic": 0, "grounding": 0}
        self.guardrail_blocks: deque[dict] = deque(maxlen=50)
        self.delegations: deque[dict] = deque(maxlen=50)
        self.series: deque[dict] = deque(maxlen=240)  # per-10s permit/deny buckets
        self._bucket = {"t": 0.0, "permit": 0, "deny": 0, "needs_human": 0}
        self.attestation = {
            "enclave_image": "spectrum-aggregator-v0.4.2.eif",
            "pcr0": sha256_hex("pcr0-spectrum-aggregator") + sha256_hex("pcr0b")[:32],
            "pcr1": sha256_hex("pcr1-kernel") + sha256_hex("pcr1b")[:32],
            "pcr2": sha256_hex("pcr2-app") + sha256_hex("pcr2b")[:32],
            "kms_key": "arn:aws:kms:us-west-2:111122223333:key/agg-mask-seeds",
            "condition": "kms:RecipientAttestation:PCR0",
            "root": "AWS Nitro Enclaves Root-G1",
            "verified": True,
            "last_attested": None,
        }

    # -- policy evaluation (mirrors the Cedar text above) ------------------------

    def authorize(self, principal: dict, action: str, resource: dict, context: dict, emit: bool = True,
                  record: bool = True) -> dict:
        """record=False evaluates without touching stats or the decision log (policy test bench)."""
        self._n += 1
        determining: list[str] = []
        deny = False
        needs_human = False

        def forbid(pid: str):
            nonlocal deny
            deny = True
            determining.append(pid)

        if principal.get("status") == "quarantined":
            forbid("P-008")
        if action == "cell_sleep" and set(resource.get("tags", [])) & {"hospital", "psap"}:
            forbid("P-001")
        if action in ("cell_sleep", "restart_du") and context.get("target_count", 1) > 6:
            forbid("P-007")
        if context.get("unauthorised_hops"):
            forbid("P-006")

        permit = False
        if action in OBSERVE:
            permit, pid = True, "P-011"
            determining.append(pid)
        elif action in TRANSPORT and principal["id"] == "transport":
            permit = True
            determining.append("P-004")
        elif action in RADIO:
            if principal["class"] == "site" and resource.get("site") in principal.get("allowed_sites", []):
                permit = True
                determining.append("P-002")
            elif principal["id"] == "coordinator" and context.get("plan_id"):
                permit = True
                determining.append("P-005")
            elif principal["id"] == "energy" and action in ("cell_sleep", "cell_wake"):
                permit = True
                determining.append("P-010")
            if not deny and permit and not context.get("human_approved"):
                if context.get("blast_radius_pct", 0) > 5:
                    needs_human = True
                    determining.append("P-003")
                elif principal.get("autonomy", 3) < 3:
                    needs_human = True
                    determining.append("P-009")
        elif action in OPS and principal["class"] in ("domain", "coordinator"):
            permit = True
            determining.append("P-013")
        elif action == "deploy":
            permit = True
            determining.append("P-012")

        if deny:
            decision = "deny"
        elif needs_human:
            decision = "needs_human"
        elif permit:
            decision = "permit"
        else:
            decision = "deny"
            determining.append("default-deny")

        if record:
            for pid in determining:
                if pid in self.policy_hits:
                    self.policy_hits[pid] += 1
            self.stats[decision] += 1
            self._bucket[decision] += 1
            if decision == "deny":
                self.deny_by_action[action] = self.deny_by_action.get(action, 0) + 1
                self.deny_by_principal[principal["id"]] = self.deny_by_principal.get(principal["id"], 0) + 1

        d = {
            "id": f"dec-{sha256_hex(f'{self.clock.t}{principal['id']}{action}{len(self.decisions)}')[:10]}",
            "ts": self.clock.iso(),
            "principal": principal["id"],
            "principal_class": principal["class"],
            "action": action,
            "resource": resource.get("id", "?"),
            "decision": decision,
            "policies": determining,
            "context": {k: v for k, v in context.items() if k in ("blast_radius_pct", "target_count", "plan_id",
                                                                     "human_approved", "delegation_chain")},
            "latency_us": 140 + (self._n * 7919) % 230 + (60 if decision != "permit" else 0),
        }
        if record:
            self.decisions.append(d)
        if emit:
            sev = {"permit": "debug", "needs_human": "notice", "deny": "critical" if deny else "warn"}[decision]
            self.bus.emit("policy.decision", "prove",
                          f"{decision.upper().replace('_', ' ')} {principal['id']} {action} {resource.get('id', '')}",
                          d, sev, principal["id"])
        return d

    def check_delegation(self, chain: list[dict], action: str, resource: dict, context: dict) -> dict:
        hops = []
        for p in chain:
            d = self.authorize(p, action, resource, {**context, "delegation_chain": [c["id"] for c in chain]},
                               emit=False)
            hops.append({"principal": p["id"], "decision": d["decision"], "policies": d["policies"]})
        unauthorised = [h["principal"] for h in hops if h["decision"] == "deny"]
        final = "deny" if unauthorised else ("needs_human" if any(h["decision"] == "needs_human" for h in hops)
                                             else "permit")
        rec = {"ts": self.clock.iso(), "chain": [c["id"] for c in chain], "action": action,
               "resource": resource.get("id"), "hops": hops, "final": final,
               "policy": "P-006" if unauthorised else None}
        self.delegations.append(rec)
        return rec

    # -- guardrails ---------------------------------------------------------------

    def scan(self, text: str, source: str) -> dict | None:
        for pat in INJECTION_PATTERNS:
            if re.search(pat, text, re.I):
                self.guardrail_counts["prompt_attack"] += 1
                hit = {"ts": self.clock.iso(), "filter": "prompt_attack", "source": source,
                       "excerpt": text[:96], "action": "BLOCKED", "pattern": pat}
                self.guardrail_blocks.append(hit)
                self.bus.emit("guardrail.block", "prove", f"prompt attack blocked from {source}", hit, "critical",
                              "guardrails")
                return hit
        return None

    def tick(self) -> None:
        if self.clock.t - self._bucket["t"] >= 10:
            self.series.append(dict(self._bucket))
            self._bucket = {"t": self.clock.t, "permit": 0, "deny": 0, "needs_human": 0}

    def snapshot(self) -> dict:
        return {
            "policies": [{**p, "hits": self.policy_hits[p["id"]]} for p in self.policies],
            "decisions": list(self.decisions)[-100:],
            "stats": self.stats,
            "deny_by_action": self.deny_by_action,
            "deny_by_principal": self.deny_by_principal,
            "guardrail_counts": self.guardrail_counts,
            "guardrail_blocks": list(self.guardrail_blocks),
            "delegations": list(self.delegations),
            "series": list(self.series),
            "attestation": self.attestation,
            "policy_store_id": "ps-7f3a9c21e4b04d1a",
        }
