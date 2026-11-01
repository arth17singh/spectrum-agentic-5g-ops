"""Observability: golden signals per plane, gen_ai token/cost accounting, latency heatmap,
OpenTelemetry-shaped spans, SLOs.

AWS mode: ADOT collector -> X-Ray (traces) and CloudWatch (EMF metrics, GenAI observability,
AgentCore observability); reads back with cloudwatch.get_metric_data.
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

from ..core import Bus, Clock, r, sha256_hex

# modelled list-price assumptions, USD per 1M tokens (in, out). Shown as "modelled" in the UI.
PRICES = {
    "claude-sonnet": (3.0, 15.0), "claude-haiku": (1.0, 5.0), "nova-pro": (0.8, 3.2), "nova-lite": (0.06, 0.24),
    "nlm-v6": (0.02, 0.02), "nlm-v7": (0.02, 0.02), "nlm-v8": (0.02, 0.02),
}
BUCKETS = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
PLANES = ["learn", "verify", "release", "operate", "prove"]


class Observability:
    def __init__(self, clock: Clock, bus: Bus, seed: int):
        self.clock = clock
        self.bus = bus
        self.rng = np.random.default_rng(seed + 13)
        self.frames: deque[dict] = deque(maxlen=600)
        self.heat: deque[dict] = deque(maxlen=180)
        self.calls: deque[dict] = deque(maxlen=4000)
        self.tool_lat: list[float] = []
        self.tool_err = 0
        self.tokens_total: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        self.cost_total: dict[str, float] = defaultdict(float)
        self.spans: dict[str, list[dict]] = {}
        self.traces: list[dict] = []
        self.incident_costs: list[dict] = [
            {"incident": "INC-2036", "usd": 0.31, "frontier_usd": 1.12, "mttr_s": 211},
            {"incident": "INC-2037", "usd": 0.44, "frontier_usd": 1.58, "mttr_s": 264},
            {"incident": "INC-2038", "usd": 0.27, "frontier_usd": 0.97, "mttr_s": 158},
            {"incident": "INC-2039", "usd": 0.52, "frontier_usd": 1.91, "mttr_s": 302},
            {"incident": "INC-2040", "usd": 0.38, "frontier_usd": 1.31, "mttr_s": 187},
        ]
        self.agent_health: dict[str, dict] = defaultdict(lambda: {"calls": 0, "tool_errors": 0, "denies": 0,
                                                                  "hitl": 0, "think_ms": 0.0, "tokens": 0})
        self.gpu: dict[str, float] = {}
        self._acc = 0.0
        self.counters = defaultdict(int)

    # -- recording ------------------------------------------------------------------

    def record_call(self, model: str, agent: str, agent_class: str, tin: int, tout: int, latency_ms: float,
                    trace_id: str | None = None) -> float:
        pin, pout = PRICES.get(model, (0.5, 1.5))
        usd = tin / 1e6 * pin + tout / 1e6 * pout
        self.tokens_total[model][0] += tin
        self.tokens_total[model][1] += tout
        self.cost_total[model] += usd
        self.calls.append({"t": self.clock.t, "model": model, "tin": tin, "tout": tout, "usd": usd,
                           "lat": latency_ms, "agent": agent})
        h = self.agent_health[agent_class]
        h["calls"] += 1
        h["think_ms"] = 0.9 * h["think_ms"] + 0.1 * latency_ms if h["think_ms"] else latency_ms
        h["tokens"] += tin + tout
        self.counters["operate"] += 1
        return usd

    def record_tool(self, latency_ms: float, error: bool = False) -> None:
        self.tool_lat.append(latency_ms)
        if error:
            self.tool_err += 1

    def note(self, agent_class: str, kind: str) -> None:
        self.agent_health[agent_class][kind] += 1

    def span(self, trace_id: str, name: str, plane: str, start: float, end: float, parent: str | None = None,
             attrs: dict | None = None, status: str = "ok") -> str:
        sid = sha256_hex(f"{trace_id}{name}{start}{len(self.spans.get(trace_id, []))}")[:16]
        sp = {"trace_id": trace_id, "span_id": sid, "parent": parent, "name": name, "plane": plane,
              "start": r(start, 3), "end": r(end, 3), "attrs": attrs or {}, "status": status}
        self.spans.setdefault(trace_id, []).append(sp)
        self.bus.emit("obs.span", "system", "", sp, "debug", "otel", record=False)
        return sid

    def new_trace(self, title: str, incident: str | None = None) -> str:
        tid = sha256_hex(f"trace{title}{self.clock.t}")[:32]
        tr = {"trace_id": tid, "title": title, "incident": incident, "start": r(self.clock.t, 3)}
        self.traces.append(tr)
        self.traces = self.traces[-30:]
        live = {t["trace_id"] for t in self.traces}
        self.spans = {k: v for k, v in self.spans.items() if k in live}
        self.bus.emit("obs.trace", "system", "", tr, "debug", "otel", record=False)
        return tid

    # -- frames -------------------------------------------------------------------

    def tick(self, dt: float, extra: dict) -> None:
        self._acc += dt
        if self._acc < 1.0:
            return
        self._acc = 0.0
        t = self.clock.t
        # latency heatmap column: gateway/tool calls + background traffic
        lat = self.tool_lat + list(self.rng.lognormal(np.log(38), 0.7, 40)) + \
            [c["lat"] for c in self.calls if c["t"] > t - 1]
        counts = [0] * len(BUCKETS)
        for v in lat:
            for i, b in enumerate(BUCKETS):
                if v <= b:
                    counts[i] += 1
                    break
            else:
                counts[-1] += 1
        p95 = float(np.percentile(lat, 95)) if lat else 0
        self.heat.append({"t": r(t, 1), "counts": counts, "p95": r(p95, 1)})
        self.tool_lat = []

        window = [c for c in self.calls if c["t"] > t - 60]
        tpm: dict[str, int] = defaultdict(int)
        for c in window:
            tpm[c["model"]] += c["tin"] + c["tout"]

        golden = {}
        for p in PLANES:
            base = extra.get("rates", {}).get(p, 1.0)
            golden[p] = {
                "rate": r(base * (1 + 0.08 * self.rng.standard_normal()), 2),
                "errors": r(max(0, extra.get("errors", {}).get(p, 0.2) + 0.05 * self.rng.standard_normal()), 3),
                "p95": r(extra.get("p95", {}).get(p, 120) * (1 + 0.06 * self.rng.standard_normal()), 1),
                "saturation": r(min(0.99, max(0.01, extra.get("sat", {}).get(p, 0.4) + 0.03 *
                                              self.rng.standard_normal())), 3),
            }
        self.gpu = extra.get("gpu", {})
        frame = {
            "t": r(t, 1), "golden": golden, "tokens_per_min": dict(tpm),
            "cost_total": {k: r(v, 4) for k, v in self.cost_total.items()},
            "tokens_total": {k: v for k, v in self.tokens_total.items()},
            "gpu": self.gpu, "heat": self.heat[-1],
            "agent_health": {k: {**v, "think_ms": r(v["think_ms"], 1)} for k, v in self.agent_health.items()},
            "incident_costs": self.incident_costs[-12:],
            "calls_per_min": len(window),
        }
        self.frames.append(frame)
        self.bus.emit("obs.metrics", "system", "", frame, "debug", "otel", record=False)

    def slos(self) -> list[dict]:
        return [
            {"id": "slo-mttr", "name": "Incident MTTR under 5 min", "target": 99.0, "current": 99.4,
             "budget_left": 0.62, "burn_1h": 0.8, "burn_6h": 0.4},
            {"id": "slo-tool", "name": "Tool calls succeed", "target": 99.5, "current": 99.82,
             "budget_left": 0.71, "burn_1h": 0.5, "burn_6h": 0.6},
            {"id": "slo-hitl", "name": "Approvals answered within SLA", "target": 98.0, "current": 98.9,
             "budget_left": 0.48, "burn_1h": 1.3, "burn_6h": 0.9},
            {"id": "slo-policy", "name": "Policy decision under 5 ms", "target": 99.9, "current": 99.97,
             "budget_left": 0.88, "burn_1h": 0.2, "burn_6h": 0.3},
            {"id": "slo-fl", "name": "FL round completes under 60 s", "target": 95.0, "current": 100.0,
             "budget_left": 1.0, "burn_1h": 0.0, "burn_6h": 0.0},
        ]

    def snapshot(self) -> dict:
        return {
            "frames": [{k: v for k, v in f.items() if k not in ("heat", "tokens_total", "cost_total")}
                       for f in list(self.frames)[-120:]], "heat": list(self.heat), "buckets": BUCKETS,
            "prices": {k: list(v) for k, v in PRICES.items()},
            "incident_costs": self.incident_costs, "agent_health": dict(self.agent_health),
            "traces": self.traces, "spans": self.spans, "slos": self.slos(),
            "cost_total": {k: r(v, 4) for k, v in self.cost_total.items()},
            "tokens_total": dict(self.tokens_total),
        }
