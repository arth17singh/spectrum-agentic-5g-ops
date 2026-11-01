"""Engine orchestrator: wires the sub-engines together, runs the simulation loop, pre-warms a
deterministic demo state, and drives the Demo Director beats."""

from __future__ import annotations

import asyncio
import time

import numpy as np

from .core import Bus, Clock, KeyRing, Scheduler, r
from .engines.federation import Federation
from .engines.ledger import Ledger
from .engines.observability import Observability
from .engines.oversight import Oversight
from .engines.proving import Proving
from .engines.release import Release
from .engines.swarm import Swarm
from .engines.trust import Trust
from .engines.twin import Twin

BEATS = [
    {"n": 1, "title": "Cold open", "route": "/", "spotlight": ["ribbon", "consortium"], "dur": 30,
     "note": "Every operator has outages and learns alone, because nobody can share data. Five operators fix that "
             "here."},
    {"n": 2, "title": "Federated round, live", "route": "/federation", "spotlight": ["round-timeline", "privacy",
                                                                                    "convergence"], "dur": 40,
     "note": "Round 13 is training right now on five operators' private data with DP-SGD. Watch epsilon: every "
             "operator has a hard privacy budget."},
    {"n": 3, "title": "What the aggregator sees", "route": "/federation", "spotlight": ["secagg"], "dur": 25,
     "note": "This is what our server sees from each operator: noise. The masks cancel only in the sum, exactly, "
             "with zero error."},
    {"n": 4, "title": "Poisoned update", "route": "/federation", "spotlight": ["update-norms", "fleet"], "dur": 25,
     "note": "Operator E is compromised and sends a poisoned update. The attested enclave rejects it with robust "
             "aggregation."},
    {"n": 5, "title": "Proving Ground gate", "route": "/proving", "spotlight": ["pipeline", "scorecard"], "dur": 35,
     "note": "The new checkpoint has to earn its way in: accuracy, safety red-team, latency, price, and signed "
             "private scores from each operator."},
    {"n": 6, "title": "Shadow, then canary", "route": "/release", "spotlight": ["rollout", "targets"], "dur": 25,
     "note": "It ships to four targets across two vendors from one signed manifest, shadow first, then 10% canary "
             "with guard metrics."},
    {"n": 7, "title": "Incident hits", "route": "/swarm", "spotlight": ["twin", "swarm"], "dur": 30,
     "note": "Saturday night. The concert just ended, and fibre T3 is cut. Watch the swarm wake up."},
    {"n": 8, "title": "Agents negotiate", "route": "/swarm", "spotlight": ["negotiation", "plan"], "dur": 30,
     "note": "Energy and Experience disagree. The coordinator brokers a compromise, and the twin predicts the "
             "outcome before anything touches the network."},
    {"n": 9, "title": "Human approval", "route": "/oversight", "spotlight": ["decision"], "dur": 25,
     "note": "Twelve percent of cells is above the 5% rule, so a human decides with the plan, the predicted impact, "
             "and the dissent."},
    {"n": 10, "title": "Prompt-injection attack", "route": "/swarm", "spotlight": ["swarm", "messages"], "dur": 40,
     "note": "Someone plants an instruction in a syslog line. The site agent is fooled, but policy blocks it, its "
             "messages are rejected, and it is quarantined."},
    {"n": 11, "title": "Provenance", "route": "/audit", "spotlight": ["provenance"], "dur": 30,
     "note": "Why did the network reroute? One click goes from the action to the approver, the model, its safety "
             "label, and the federated round that trained it."},
    {"n": 12, "title": "Tamper test", "route": "/audit", "spotlight": ["chain"], "dur": 25,
     "note": "Edit one record directly in the database, and verification breaks at exactly that record. The anchor "
             "in S3 Object Lock still holds the truth."},
    {"n": 13, "title": "Cost and traces", "route": "/observability", "spotlight": ["cost", "trace"], "dur": 25,
     "note": "Every agent step is traced. The fine-tuned model runs 40 site agents for a fraction of a frontier "
             "model's cost."},
    {"n": 14, "title": "Close the loop", "route": "/", "spotlight": ["ribbon"], "dur": 20,
     "note": "Tonight's incident becomes tomorrow's private training data at each operator. That is the loop."},
]


class Engine:
    def __init__(self, seed: int = 7, warm: bool = True):
        self.seed = seed
        self.build(warm)

    def build(self, warm: bool = True, hold: bool = False) -> None:
        self.clock = Clock()
        self.bus = Bus(self.clock)
        self.keys = KeyRing()
        self.sched = Scheduler(self.clock)
        self.ledger = Ledger(self.clock, self.bus, self.keys)
        self.trust = Trust(self.clock, self.bus, self.keys)
        self.twin = Twin(self.clock, self.bus, self.seed)
        self.obs = Observability(self.clock, self.bus, self.seed)
        self.oversight = Oversight(self.clock, self.bus, self.keys, self.ledger)
        self.fl = Federation(self.clock, self.bus, self.keys, self.ledger, self.seed)
        self.proving = Proving(self.clock, self.bus, self.keys, self.ledger, self.seed)
        self.release = Release(self.clock, self.bus, self.keys, self.ledger, self.seed)
        self.swarm = Swarm(self.clock, self.bus, self.keys, self.ledger, self.trust, self.twin, self.oversight,
                           self.obs, self.sched, self.seed)
        self.fl.on_promote = lambda ck: self.proving.candidate_from_checkpoint(ck)
        self.proving.on_gate = self._on_gate
        self.release.on_traffic = lambda v, pct: self.swarm.set_model_cohort(v, pct)
        self.director = {"beat": 0, "started_at": None, "auto": False, "beats": BEATS}
        self.replay = False
        if warm:
            self.warm(hold)

    def _on_gate(self, model_id: str, passed: bool) -> None:
        if passed:
            self.release.start_shadow(model_id)

    # ------------------------------------------------------------------ warm-up

    def warm(self, hold: bool = False) -> None:
        """Deterministically fast-forward to the demo's opening state (~2 s)."""
        t0 = time.perf_counter()
        bus = self.bus
        bus.muted = True
        rng = np.random.default_rng(self.seed + 99)
        self.clock.t = -7500.0
        # past approvals and an old eval/release trail
        for i, (title, by) in enumerate([("Reroute H2 over MW2 during maintenance", "a.rivera"),
                                         ("Restart DU at s-11", "a.rivera"),
                                         ("Raise tx power on 4 cells near s-08", "j.okafor"),
                                         ("Sleep 6 cells overnight (energy)", "a.rivera"),
                                         ("Rollback handover offsets s-19/s-20", "a.rivera"),
                                         ("Wake 2 cells for a stadium warm-up event", "a.rivera")]):
            self.clock.t += 900
            it = self.oversight.create(title=title, tier="high", requested_by="coordinator", incident=None,
                                       plan={"id": f"plan-old-{i}", "steps": []},
                                       preflight={"blast_radius_pct": 6.5 + i, "before": {}, "after": {}},
                                       reasons=[{"policy": "P-003", "text": "blast radius > 5%"}], dissent=[],
                                       model_opinions=[], summary=title)
            self.clock.t += 40 + 10 * i
            self.oversight.decide(it["id"], "approved", by=by)
        self.clock.t = -7400.0
        # bulk routine history (signed tool calls, A2A informs)
        while self.clock.t < -320:
            self.clock.t += float(rng.uniform(2.2, 4.6))
            self.swarm.routine(emit=False)
        # federation: run 12 rounds with the full simulator at 10 Hz
        self.clock.t = -320.0
        self.fl.start()
        dt = 0.1
        while self.fl.round < 12 or self.fl.phase != "done":
            self.clock.t += dt
            self.fl.tick(dt)
            self.proving.tick(dt)
            self.release.tick(dt)
            self.obs.tick(dt, self._obs_extra())
            if self.fl.round == 12 and self.fl.phase == "aggregate":
                self.fl.hold_at_boundary = True
            if self.clock.t > -2:
                break
        # last few seconds: everything ticks so live series are populated
        while self.clock.t < 0:
            self.clock.t += dt
            self.twin.tick(dt)
            self.swarm.tick(dt)
            self.obs.tick(dt, self._obs_extra())
        self.clock.t = 0.0
        if not hold:
            self.fl.hold_at_boundary = False
            self.fl.state = "running"
            self.fl._begin_round()
        else:
            self.fl.state = "ready"
        self._synth_old_trace()
        bus.muted = False
        bus.history.clear()
        bus.seq = 0
        self.warm_ms = round((time.perf_counter() - t0) * 1000)

    def _synth_old_trace(self) -> None:
        tr = self.obs.new_trace("incident.handle INC-2040", "INC-2040")
        base = -1500.0
        root = self.obs.span(tr, "incident.handle", "operate", base, base + 187, None, {"incident.id": "INC-2040"})
        spans = [("agent.sentinel.detect", 0, 2.1), ("gen_ai.chat claude-haiku", 0.3, 1.4),
                 ("agent.coordinator.plan", 2.2, 6.8), ("gen_ai.chat claude-sonnet", 2.4, 5.6),
                 ("a2a.send coordinator→site-11", 5.7, 5.72), ("gateway.tool_call restart_du", 6.9, 7.0),
                 ("avp.is_authorized", 6.91, 6.912), ("twin.simulate_change", 7.0, 7.9),
                 ("stepfunctions.wait_for_approval", 8.0, 61.0), ("gateway.tool_call restart_du", 61.2, 61.3),
                 ("twin.restart_du", 61.3, 140.0), ("agent.coordinator.verify", 140.2, 187.0)]
        tokens = {"gen_ai.chat claude-haiku": ("claude-haiku", 2140, 312), "gen_ai.chat claude-sonnet": ("claude-sonnet", 4630, 588)}
        for name, s, e in spans:
            plane = "prove" if name.startswith(("avp", "stepfunctions")) else "operate"
            attrs = {"note": "prior incident (pre-warmed)"}
            if name in tokens:
                model, tin, tout = tokens[name]
                attrs.update({"gen_ai.system": "aws.bedrock", "gen_ai.operation.name": "chat",
                              "gen_ai.request.model": model, "gen_ai.usage.input_tokens": tin,
                              "gen_ai.usage.output_tokens": tout, "aws.region": "us-west-2"})
            self.obs.span(tr, name, plane, base + s, base + e, root, attrs)

    def _obs_extra(self) -> dict:
        fl_running = self.fl.state == "running"
        return {
            "rates": {"learn": 24.0 if fl_running else 0.4, "verify": 6.0 if self.proving.active and
                      self.proving.active["state"] == "running" else 0.2, "release": 58.0, "operate": 3.5,
                      "prove": 9.0},
            "errors": {"learn": 0.0, "verify": 0.0, "release": self.release.guards["error_rate"], "operate": 0.18,
                       "prove": 0.0},
            "p95": {"learn": 410, "verify": 820, "release": self.release.guards["p95_ms"], "operate": 1900,
                    "prove": 4.2},
            "sat": {"learn": 0.86 if fl_running else 0.1, "verify": 0.3, "release": 0.42, "operate": 0.35,
                    "prove": 0.12},
            "gpu": {o["id"]: r(o["gpu"] if o["status"] == "training" else 0.05, 3) for o in self.fl.ops},
        }

    # ------------------------------------------------------------------ loop

    def tick(self, real_dt: float) -> None:
        if getattr(self, "building", False):
            return
        dt = self.clock.advance(real_dt)
        if dt <= 0:
            return
        self.sched.run_due()
        self.fl.tick(dt)
        self.proving.tick(dt)
        self.release.tick(dt)
        self.twin.tick(dt)
        self.swarm.tick(dt)
        self.trust.tick()
        self.oversight.tick()
        self.obs.tick(dt, self._obs_extra())
        self._clock_acc = getattr(self, "_clock_acc", 0.0) + dt
        if self._clock_acc >= 1.0:
            self._clock_acc = 0.0
            self.bus.emit("sim.clock", "system", "", self.clock_view(), "debug", "clock", record=False)

    async def run(self) -> None:
        last = time.perf_counter()
        while True:
            await asyncio.sleep(0.1)
            now = time.perf_counter()
            try:
                self.tick(now - last)
            except Exception as e:  # keep the demo alive, but say what broke
                self.bus.emit("system.error", "system", f"engine error: {e!r}", {}, "critical", "engine")
            last = now

    def clock_view(self) -> dict:
        return {"t": r(self.clock.t, 2), "iso": self.clock.iso(), "speed": self.clock.speed,
                "paused": self.clock.paused}

    # ------------------------------------------------------------------ director

    def reset(self) -> None:
        subs = self.bus.subscribers
        self.building = True
        try:
            self.build(warm=True, hold=True)
        finally:
            self.building = False
        self.bus.subscribers = subs
        self.bus.emit("system.reset", "system", "scenario reset to the pre-demo state", {}, "notice", "director")

    def beat(self, n: int) -> dict:
        b = BEATS[n - 1]
        self.director["beat"] = n
        self.director["started_at"] = self.clock.iso()
        action = {
            1: lambda: None,
            2: self._beat_round,
            3: lambda: setattr(self.fl, "reveal", True),
            4: lambda: self.fl.client("op-e", "poison"),
            5: self._beat_eval,
            6: self._beat_canary,
            7: self.swarm.start_incident,
            8: lambda: None,
            9: lambda: None,
            10: lambda: self.swarm.telemetry_injection("site-17"),
            11: lambda: None,
            12: lambda: self.ledger.tamper(),
            13: lambda: None,
            14: self._beat_loop,
        }[n]
        action()
        self.bus.emit("director.beat", "system", f"beat {n}: {b['title']}", {"beat": b, "director": self.director},
                      "info", "director")
        return b

    def _beat_round(self) -> None:
        if self.fl.state in ("ready", "paused", "idle"):
            self.fl.start()

    def _beat_eval(self) -> None:
        if not self.proving.pending_candidate or not self.proving.pending_candidate.get("pending"):
            ck = self.fl.checkpoints[-1] if self.fl.checkpoints else self.fl.save_checkpoint()
            if not ck["promoted"]:
                self.fl.promote(ck["id"])
        if not self.proving.active or self.proving.active["state"] != "running":
            self.proving.run()

    def _beat_canary(self) -> None:
        if self.release.candidate is None:
            cand = self.proving.pending_candidate
            if cand:
                self.release.start_shadow(cand["id"])
        if self.release.stage_index == 0:
            self.release.advance()

    def _beat_loop(self) -> None:
        inc = self.swarm.active
        rec = self.ledger.append("loop.enqueue", "coordinator", "learn",
                                 {"incident": inc["id"] if inc else "INC-2041",
                                  "destination": "op-a private incident store", "consent": "operator policy OP-A-7",
                                  "next_round": self.fl.round + 1})
        self.bus.emit("loop.enqueue", "learn", f"{inc['id'] if inc else 'INC-2041'} queued as private training data "
                      f"at Op-A for round {self.fl.round + 1}", {"incident": inc["id"] if inc else None},
                      "notice", "coordinator", ledger={"seq": rec["seq"], "hash": rec["hash"]})

    # ------------------------------------------------------------------ snapshot

    def snapshot(self) -> dict:
        return {
            "seq": self.bus.seq,
            "clock": self.clock_view(),
            "director": self.director,
            "federation": self.fl.snapshot(),
            "proving": self.proving.snapshot(),
            "release": self.release.snapshot(),
            "twin": self.twin.snapshot(),
            "swarm": self.swarm.snapshot(),
            "trust": self.trust.snapshot(),
            "ledger": self.ledger.snapshot(),
            "oversight": self.oversight.snapshot(),
            "observability": self.obs.snapshot(),
            "stream": list(self.bus.history)[-400:],
        }
