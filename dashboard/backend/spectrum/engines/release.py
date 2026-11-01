"""Release plane: repeatable multi-vendor deployment. Shadow -> canary 10% -> 50% -> 100% with guard
metrics and automatic rollback.

AWS mode: SageMaker endpoint variants / inference components for traffic shifting, Bedrock Custom
Model Import for the managed target, ECS services for vendor edge targets, CDK stacks per target.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..core import Bus, Clock, KeyRing, r, sha256_hex

if TYPE_CHECKING:
    from .ledger import Ledger

TARGETS = [
    {"id": "sagemaker", "name": "SageMaker endpoint", "detail": "us-west-2, ml.g6.xlarge x2", "kind": "managed",
     "stack": "SpectrumNlmEndpoint-usw2"},
    {"id": "bedrock-cmi", "name": "Bedrock Custom Model Import", "detail": "us-west-2, on-demand", "kind": "managed",
     "stack": "SpectrumBedrockImport-usw2"},
    {"id": "vendor-a", "name": "Vendor A EMS edge", "detail": "ECS on Outposts-style rack, JSON dialect",
     "kind": "edge", "stack": "SpectrumEdgeVendorA"},
    {"id": "vendor-b", "name": "Vendor B EMS edge", "detail": "ECS, CLI dialect", "kind": "edge",
     "stack": "SpectrumEdgeVendorB"},
]
STAGES = [
    {"id": "shadow", "label": "Shadow", "traffic": 0},
    {"id": "canary10", "label": "Canary 10%", "traffic": 10},
    {"id": "canary50", "label": "Canary 50%", "traffic": 50},
    {"id": "full", "label": "Full 100%", "traffic": 100},
]
GUARDS = {"error_rate_max": 1.0, "p95_max_ms": 600, "safety_incidents_max": 0, "agreement_min": 90.0}


class Release:
    def __init__(self, clock: Clock, bus: Bus, keys: KeyRing, ledger: "Ledger", seed: int):
        self.clock = clock
        self.bus = bus
        self.keys = keys
        self.ledger = ledger
        self.rng = np.random.default_rng(seed + 7)
        self.on_traffic = None  # callback(version, pct)
        self.prod = "nlm-v6"
        self.candidate: str | None = None
        self.stage_index = -1
        self.paused = False
        self.regression = False
        self.targets = []
        for t in TARGETS:
            self.targets.append({**t, "version": "nlm-v6", "candidate": None, "traffic": 0, "state": "serving",
                                 "p95_ms": 0, "error_rate": 0.0, "health": "healthy",
                                 "manifest": self._manifest(t, "nlm-v6")})
        self.agreement: list[dict] = []
        self.disagreements: list[dict] = []
        self.tally = {"better": 0, "worse": 0, "tie": 0}
        self.guards = {"error_rate": 0.2, "p95_ms": 410, "safety_incidents": 0, "agreement": 0.0}
        self.history: list[dict] = [
            {"version": "nlm-v5", "event": "promoted to 100%", "ts": "2026-10-29T14:02:11.000-07:00"},
            {"version": "nlm-v6", "event": "shadow", "ts": "2026-10-30T10:15:40.000-07:00"},
            {"version": "nlm-v6", "event": "canary 10%", "ts": "2026-10-30T11:02:03.000-07:00"},
            {"version": "nlm-v6", "event": "promoted to 100%", "ts": "2026-10-30T16:40:27.000-07:00"},
        ]
        self._acc = 0.0
        self._sh = 0.0

    def _manifest(self, t: dict, version: str) -> dict:
        m = {
            "target": t["id"], "version": version, "region": "us-west-2",
            "image": f"111122223333.dkr.ecr.us-west-2.amazonaws.com/spectrum-nlm@sha256:{sha256_hex('img' + version)}",
            "adapter_hash": sha256_hex(f"adapter-{version}"), "stack": t["stack"],
            "dialect": {"vendor-a": "json/v3", "vendor-b": "cli/v2"}.get(t["id"], "n/a"),
            "iac_hash": sha256_hex(f"cdk-{t['stack']}-{version}"),
        }
        m["signature"] = self.keys.sign("kms:spectrum/release", sha256_hex(str(sorted(m.items()))))
        return m

    def start_shadow(self, version: str) -> None:
        if version == self.prod or version == self.candidate:
            return
        self.candidate = version
        self.stage_index = 0
        self.agreement.clear()
        self.disagreements.clear()
        self.tally = {"better": 0, "worse": 0, "tie": 0}
        for t in self.targets:
            t["candidate"] = version
            t["traffic"] = 0
            t["state"] = "shadow"
        self._record(f"{version} deployed in shadow on all targets")

    def advance(self) -> None:
        if self.candidate is None or self.stage_index >= len(STAGES) - 1:
            return
        self.stage_index += 1
        st = STAGES[self.stage_index]
        for t in self.targets:
            t["traffic"] = st["traffic"]
            t["state"] = "canary" if st["traffic"] < 100 else "serving"
            if st["traffic"] == 100:
                t["version"] = self.candidate
                t["candidate"] = None
                t["manifest"] = self._manifest(t, self.candidate)
        if st["traffic"] == 100:
            self.prod = self.candidate
        self._record(f"{self.candidate} {st['label'].lower()}")
        if self.on_traffic:
            self.on_traffic(self.candidate, st["traffic"])
        if st["traffic"] == 100:
            self.candidate = None

    def rollback(self, reason: str = "manual rollback", auto: bool = False) -> None:
        if self.candidate is None:
            return
        v = self.candidate
        for t in self.targets:
            t["traffic"] = 0
            t["state"] = "serving"
            t["candidate"] = None
        self.stage_index = -1
        self.candidate = None
        self.regression = False
        self._record(f"{v} rolled back: {reason}", sev="critical" if auto else "warn")
        if self.on_traffic:
            self.on_traffic(self.prod, 100)

    def _record(self, msg: str, sev: str = "notice") -> None:
        rec = self.ledger.append("release.stage", "release-controller", "release",
                                 {"candidate": self.candidate, "prod": self.prod, "stage_index": self.stage_index,
                                  "event": msg}, key_id="kms:spectrum/release")
        self.history.append({"version": self.candidate or self.prod, "event": msg, "ts": self.clock.iso(),
                             "ledger_seq": rec["seq"]})
        self.bus.emit("release.stage", "release", msg, self.view(), sev, "release-controller",
                      ledger={"seq": rec["seq"], "hash": rec["hash"]})

    def tick(self, dt: float) -> None:
        self._acc += dt
        self._sh += dt
        if self._acc >= 1.0:
            self._acc = 0.0
            for t in self.targets:
                base = {"sagemaker": 405, "bedrock-cmi": 470, "vendor-a": 380, "vendor-b": 395}[t["id"]]
                t["p95_ms"] = int(base + self.rng.normal(0, 14) + (260 if self.regression and t["traffic"] else 0))
                t["error_rate"] = r(max(0, 0.18 + self.rng.normal(0, 0.05) + (1.6 if self.regression and t["traffic"]
                                                                               else 0)), 3)
                t["health"] = "degraded" if t["error_rate"] > 1 else "healthy"
            act = [t for t in self.targets if t["traffic"] > 0] or self.targets
            self.guards["error_rate"] = r(max(t["error_rate"] for t in act), 3)
            self.guards["p95_ms"] = max(t["p95_ms"] for t in act)
            if self.candidate and self.stage_index >= 1 and (self.guards["error_rate"] > GUARDS["error_rate_max"]):
                self.rollback(f"guard breached: error rate {self.guards['error_rate']}% > 1%", auto=True)
            payload = self.view()
            if self.candidate:
                payload = {**payload, "agreement": self.agreement[-150:], "disagreements": self.disagreements[:15]}
            self.bus.emit("release.metrics", "release", "", payload, "debug", "release-controller", record=False)
        if self.candidate and self._sh >= 2.0:
            self._sh = 0.0
            agree = self.rng.random() < 0.94
            if agree:
                self.tally["tie"] += 1
            else:
                better = self.rng.random() < 0.85
                self.tally["better" if better else "worse"] += 1
                inc = f"INC-{2041 + len(self.disagreements) * 3 % 7}"
                self.disagreements.insert(0, {
                    "ts": self.clock.iso(), "incident": inc, "site": f"s-{int(self.rng.integers(1, 41)):02d}",
                    "prod": "handover offset +1 dB" if better else "wake 1 cell",
                    "candidate": "load_balance 2 neighbours" if better else "handover offset +2 dB",
                    "winner": "candidate" if better else "production",
                    "twin_delta": f"{'-' if better else '+'}{self.rng.uniform(3, 11):.1f}% congestion"})
                self.disagreements = self.disagreements[:30]
            total = sum(self.tally.values())
            rate = 100 * (self.tally["tie"] + self.tally["better"]) / max(total, 1)
            self.guards["agreement"] = r(100 * self.tally["tie"] / max(total, 1), 1)
            self.agreement.append({"t": r(self.clock.t, 1), "agreement": self.guards["agreement"],
                                   "non_regression": r(rate, 1)})
            self.agreement = self.agreement[-300:]

    def view(self) -> dict:
        return {"prod": self.prod, "candidate": self.candidate, "stage_index": self.stage_index,
                "stages": STAGES, "targets": self.targets, "guards": self.guards, "guard_limits": GUARDS,
                "tally": self.tally, "paused": self.paused}

    def snapshot(self) -> dict:
        return {**self.view(), "agreement": self.agreement, "disagreements": self.disagreements,
                "history": self.history[-30:],
                "dialects": {
                    "intent": "increase tx power on cell s-14-1 by 2 dB",
                    "vendor-a": '{\n  "op": "SET",\n  "mo": "NRCellDU=s-14-1",\n  "attr": {"configuredMaxTxPower": "+2dB"}\n}',
                    "vendor-b": "configure cell s-14-1\n  tx-power offset +2.0\ncommit confirmed 300",
                    "valid": {"vendor-a": True, "vendor-b": True}}}
