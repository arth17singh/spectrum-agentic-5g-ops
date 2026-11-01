"""Network digital twin: 40 sites x 3 cells, 5 aggregation hubs with fibre to the core and
microwave backup between neighbouring hubs. Produces KPIs every sim second.

AWS mode: same simulator runs on ECS/Fargate and publishes ticks to Kinesis.
"""

from __future__ import annotations

import math

import numpy as np

from ..core import Bus, Clock, r

COLS, ROWS = 8, 5
STADIUM = {"s-21", "s-22", "s-29", "s-30", "s-13", "s-14"}
HOSPITAL = {"s-06": "hospital", "s-35": "hospital", "s-27": "psap"}


class Twin:
    def __init__(self, clock: Clock, bus: Bus, seed: int):
        self.clock = clock
        self.bus = bus
        self.rng = np.random.default_rng(seed + 11)
        self.sites: list[dict] = []
        for row in range(ROWS):
            for col in range(COLS):
                i = row * COLS + col + 1
                sid = f"s-{i:02d}"
                hub = "H5" if (row >= 3 and col >= 6) else f"H{col // 2 + 1}"
                tags = []
                if sid in STADIUM:
                    tags.append("stadium")
                if sid in HOSPITAL:
                    tags.append(HOSPITAL[sid])
                self.sites.append({
                    "id": sid, "col": col, "row": row, "hub": hub, "tags": tags,
                    "vendor": "A" if (col + row) % 3 else "B",
                    "cells": [{"id": f"{sid}-{k}", "sleeping": False, "power_db": 0.0} for k in range(3)],
                    "base": 0.35 + 0.25 * self.rng.random(),
                    "surge": 0.0, "absorb": 0.0, "offload": 0.0,
                    "prb": 0.0, "users": 0, "thr": 0.0, "drop": 0.0, "kw": 0.0, "backhaul": "fibre",
                })
        # put three cells near the stadium to sleep (energy saving earlier in the evening)
        for sid, k in (("s-20", 2), ("s-23", 1), ("s-28", 0)):
            self.site(sid)["cells"][k]["sleeping"] = True
        self.hubs = [
            {"id": f"H{n}", "fibre": f"T{n}", "fibre_ok": True, "mw_to": f"H{n - 1 if n > 1 else 2}", "via_mw": False}
            for n in range(1, 6)
        ]
        self.links = [{"id": f"T{n}", "kind": "fibre", "from": f"H{n}", "to": "CORE", "capacity_gbps": 40,
                       "ok": True} for n in range(1, 6)]
        self.links += [{"id": f"MW{n}", "kind": "microwave", "from": f"H{n}", "to": f"H{n - 1 if n > 1 else 2}",
                        "capacity_gbps": 12, "ok": True, "active": False} for n in range(1, 6)]
        self.alarms: list[dict] = []
        self.health_series: list[dict] = []
        self.actions: list[dict] = []
        self._acc = 0.0
        self.compute()

    def site(self, sid: str) -> dict:
        return self.sites[int(sid[2:]) - 1]

    def hub(self, hid: str) -> dict:
        return next(h for h in self.hubs if h["id"] == hid)

    def neighbours(self, sid: str) -> list[str]:
        s = self.site(sid)
        out = []
        for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, -1 if s["col"] % 2 == 0 else 1),
                       (-1, -1 if s["col"] % 2 == 0 else 1)):
            c, rr = s["col"] + dc, s["row"] + dr
            if 0 <= c < COLS and 0 <= rr < ROWS:
                out.append(f"s-{rr * COLS + c + 1:02d}")
        return out

    # -- faults -------------------------------------------------------------------

    def inject(self, kind: str, target: str | None = None) -> dict:
        if kind == "fiber_cut":
            hub = target or "H3"
            h = self.hub(hub)
            h["fibre_ok"] = False
            link = next(lk for lk in self.links if lk["id"] == h["fibre"])
            link["ok"] = False
            for s in self.sites:
                if s["hub"] == hub:
                    s["backhaul"] = "down"
            alarm = {"id": f"ALM-{len(self.alarms) + 7101}", "kind": "LOS", "severity": "critical",
                     "target": link["id"], "text": f"Loss of signal on fibre {link['id']} ({hub} to core)",
                     "ts": self.clock.iso()}
        elif kind == "stadium_surge":
            for s in self.sites:
                if "stadium" in s["tags"]:
                    s["surge"] = 0.55
                elif any(n in STADIUM for n in self.neighbours(s["id"])):
                    s["surge"] = 0.25
            alarm = {"id": f"ALM-{len(self.alarms) + 7101}", "kind": "LOAD", "severity": "major",
                     "target": "stadium", "text": "Event let-out: 68,000 attendees leaving stadium district",
                     "ts": self.clock.iso()}
        elif kind == "power_outage":
            s = self.site(target or "s-33")
            for c in s["cells"]:
                c["sleeping"] = True
            alarm = {"id": f"ALM-{len(self.alarms) + 7101}", "kind": "POWER", "severity": "critical",
                     "target": s["id"], "text": f"Mains power lost at {s['id']}, on battery", "ts": self.clock.iso()}
        elif kind == "interference":
            s = self.site(target or "s-08")
            s["surge"] = 0.4
            alarm = {"id": f"ALM-{len(self.alarms) + 7101}", "kind": "INTERFERENCE", "severity": "major",
                     "target": s["id"], "text": f"Uplink interference rise +9 dB at {s['id']}", "ts": self.clock.iso()}
        else:
            raise ValueError(kind)
        self.alarms.append(alarm)
        self.bus.emit("twin.alarm", "operate", alarm["text"], alarm,
                      "critical" if alarm["severity"] == "critical" else "warn", "twin")
        self.compute()
        return alarm

    def apply(self, tool: str, params: dict, actor: str) -> dict:
        """Apply an (already authorised) action to the twin."""
        if tool == "reroute":
            h = self.hub(params["hub"])
            h["via_mw"] = True
            for lk in self.links:
                if lk["id"] == f"MW{h['id'][1:]}":
                    lk["active"] = True
            for s in self.sites:
                if s["hub"] == h["id"]:
                    s["backhaul"] = "microwave"
        elif tool == "cell_wake":
            for cid in params["cells"]:
                sid = cid.rsplit("-", 1)[0]
                self.site(sid)["cells"][int(cid.rsplit("-", 1)[1])]["sleeping"] = False
        elif tool == "cell_sleep":
            for cid in params["cells"]:
                sid = cid.rsplit("-", 1)[0]
                self.site(sid)["cells"][int(cid.rsplit("-", 1)[1])]["sleeping"] = True
        elif tool == "set_tx_power":
            for cid in params["cells"]:
                sid = cid.rsplit("-", 1)[0]
                s = self.site(sid)
                s["cells"][int(cid.rsplit("-", 1)[1])]["power_db"] += params.get("delta_db", 2)
        elif tool == "set_handover_offset":
            pass
        elif tool == "load_balance":
            # neighbours absorb a share of the congested sites' demand
            for sid in params["from_sites"]:
                self.site(sid)["offload"] = params.get("share", 0.35)
            for sid in params["to_sites"]:
                self.site(sid)["absorb"] = params.get("absorb", 0.18)
        act = {"ts": self.clock.iso(), "tool": tool, "params": params, "actor": actor}
        self.actions.append(act)
        self.bus.emit("twin.action", "operate", f"{actor} {tool}", act, "notice", actor)
        self.compute()
        return act

    def clear(self) -> None:
        for h in self.hubs:
            h["fibre_ok"], h["via_mw"] = True, False
        for lk in self.links:
            lk["ok"] = True
            lk["active"] = False
        for s in self.sites:
            s["surge"] = s["absorb"] = s["offload"] = 0.0
            s["backhaul"] = "fibre"

    # -- simulation ------------------------------------------------------------------

    def demand(self, s: dict) -> float:
        t = self.clock.t
        diurnal = 0.06 * math.sin(t / 900.0 + s["col"])
        return max(0.05, (s["base"] + diurnal + s["surge"] + s["absorb"]) * (1 - s["offload"]))

    def preflight(self, steps: list[dict]) -> dict:
        """Run a plan on a fork of the twin and report predicted KPI deltas."""
        before = self.kpis()
        saved = [(s["absorb"], s["offload"], [dict(c) for c in s["cells"]], s["backhaul"]) for s in self.sites]
        saved_h = [dict(h) for h in self.hubs]
        saved_l = [dict(lk) for lk in self.links]
        for st in steps:
            if st["tool"] in ("reroute", "cell_wake", "set_tx_power", "load_balance"):
                self.apply_silent(st["tool"], st["params"])
        self.compute()
        after = self.kpis()
        touched = sum(len(st["params"].get("cells", [])) for st in steps) + \
            sum(len(st["params"].get("to_sites", [])) for st in steps)
        for s, (a, o, cells, bh) in zip(self.sites, saved):
            s["absorb"], s["offload"], s["cells"], s["backhaul"] = a, o, cells, bh
        self.hubs = saved_h
        self.links = saved_l
        self.compute()
        return {"before": before, "after": after,
                "blast_radius_pct": r(100 * touched / (len(self.sites) * 3), 1), "cells_touched": touched}

    def apply_silent(self, tool: str, params: dict) -> None:
        emit, self.bus.muted = self.bus.muted, True
        actions_len = len(self.actions)
        try:
            self.apply(tool, params, "preflight")
        finally:
            self.bus.muted = emit
            del self.actions[actions_len:]
            if self.bus.history and self.bus.history[-1]["type"] == "twin.action":
                self.bus.history.pop()

    def compute(self) -> None:
        for s in self.sites:
            active = sum(1 for c in s["cells"] if not c["sleeping"])
            boost = sum(c["power_db"] for c in s["cells"] if not c["sleeping"]) * 0.04
            cap = max(active, 0.2) / 3 * (1 + boost)
            if s["backhaul"] == "down":
                cap *= 0.02
            elif s["backhaul"] == "microwave":
                cap *= 0.9
            d = self.demand(s)
            prb = min(100.0, 100 * d / (cap * 1.15))
            s["prb"] = prb
            s["users"] = int(d * 1400)
            s["thr"] = min(d, cap) * 820
            s["drop"] = max(0.0, (prb - 82) * 0.11) + (6.0 if s["backhaul"] == "down" else 0.0)
            s["kw"] = active * 3.1 + (3 - active) * 0.35 + boost * 2

    def kpis(self) -> dict:
        hot = [s for s in self.sites if "stadium" in s["tags"]]
        ok = [s for s in self.sites if s["prb"] < 85 and s["drop"] < 1 and s["backhaul"] != "down"]
        return {
            "health_pct": r(100 * len(ok) / len(self.sites), 1),
            "congestion_pct": r(sum(s["prb"] for s in hot) / max(len(hot), 1), 1),
            "dropped_pct": r(sum(s["drop"] for s in hot) / max(len(hot), 1), 2),
            "throughput_gbps": r(sum(s["thr"] for s in self.sites) / 1000, 2),
            "users": sum(s["users"] for s in self.sites),
            "energy_kw": r(sum(s["kw"] for s in self.sites), 1),
            "sleeping_cells": sum(1 for s in self.sites for c in s["cells"] if c["sleeping"]),
        }

    def tick(self, dt: float) -> None:
        self._acc += dt
        if self._acc < 1.0:
            return
        self._acc = 0.0
        for s in self.sites:
            s["base"] = float(np.clip(s["base"] + self.rng.normal(0, 0.004), 0.25, 0.7))
        self.compute()
        k = self.kpis()
        self.health_series.append({"t": r(self.clock.t, 1), **k})
        self.health_series = self.health_series[-900:]
        self.bus.emit("twin.tick", "operate", "", {"kpis": k, "sites": self.compact(),
                                                    "hubs": self.hubs, "links": self.links}, "debug", "twin",
                      record=False)

    def compact(self) -> list[list]:
        # [id, prb, users, thr, drop, kw, backhaul, sleeping_mask]
        out = []
        for s in self.sites:
            mask = sum((1 << i) for i, c in enumerate(s["cells"]) if c["sleeping"])
            out.append([s["id"], r(s["prb"], 1), s["users"], r(s["thr"], 0), r(s["drop"], 2), r(s["kw"], 1),
                        s["backhaul"], mask])
        return out

    def snapshot(self) -> dict:
        return {
            "layout": {"cols": COLS, "rows": ROWS},
            "sites": [{k: s[k] for k in ("id", "col", "row", "hub", "tags", "vendor", "cells")} for s in self.sites],
            "state": self.compact(),
            "hubs": self.hubs,
            "links": self.links,
            "alarms": self.alarms[-50:],
            "kpis": self.kpis(),
            "health_series": self.health_series[-600:],
            "actions": self.actions[-50:],
        }
