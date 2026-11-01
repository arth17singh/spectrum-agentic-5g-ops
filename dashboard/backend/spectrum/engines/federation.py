"""Learn plane: federated LoRA fine-tuning across five operators with DP-SGD, secure aggregation,
an attested enclave aggregator and an optional CKKS (homomorphic) path.

What is real in mock mode:
  * secure aggregation maths: pairwise masks over fixed-point integers mod 2^32 cancel EXACTLY
  * privacy accounting is an approximation (documented) that composes correctly when sigma changes
  * checkpoints carry real SHA-256 hashes and Ed25519 signatures through the ledger

AWS mode: one ECS task per operator (own task role, own S3 bucket) running a Flower client with
Opacus; aggregator in a Nitro Enclave; checkpoints to S3; epsilon reported by the Opacus RDP accountant.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from ..core import Bus, Clock, KeyRing, r, sha256_hex

if TYPE_CHECKING:
    from .ledger import Ledger

OPERATORS = [
    {"id": "op-a", "short": "Op-A", "name": "Cascadia Mobile", "vendor": "A", "region": "us-west-2",
     "samples": 18400, "speed": 1.00, "floor": 0.93},
    {"id": "op-b", "short": "Op-B", "name": "Northwind Telecom", "vendor": "B", "region": "ca-central-1",
     "samples": 12100, "speed": 0.92, "floor": 1.01},
    {"id": "op-c", "short": "Op-C", "name": "Meridian Networks", "vendor": "A", "region": "us-east-1",
     "samples": 9300, "speed": 1.08, "floor": 1.07},
    {"id": "op-d", "short": "Op-D", "name": "Halcyon Wireless", "vendor": "B", "region": "eu-west-1",
     "samples": 22600, "speed": 0.86, "floor": 0.89},
    {"id": "op-e", "short": "Op-E", "name": "Tessellate Fiber", "vendor": "A", "region": "ap-southeast-2",
     "samples": 7000, "speed": 1.04, "floor": 1.16},
]

STEPS_PER_ROUND = 240
BATCH = 64
DELTA = 1e-5
EPS_C = 1.71  # calibration constant for the mock accountant (see eps_increment)
ADAPTER_BYTES = 22_675_456  # LoRA r=16 on q,k,v,o of a ~1B model, fp16
MATRIX = 16
SCALE = 2 ** 16
MOD = 2 ** 32

PHASE_TIME = {"pull": 1.0, "train": 16.0, "secure": 1.4, "upload": 1.6, "aggregate": 2.6}


def eps_increment(q: float, steps: float, sigma: float) -> float:
    """Squared-epsilon increment. For the subsampled Gaussian mechanism epsilon grows ~ q*sqrt(T log(1/d))/sigma;
    accumulating eps^2 lets sigma change mid-run and still compose sensibly. Opacus' RDP accountant replaces
    this in AWS mode."""
    return (EPS_C * q / sigma) ** 2 * steps * math.log(1 / DELTA)


class Federation:
    def __init__(self, clock: Clock, bus: Bus, keys: KeyRing, ledger: "Ledger", seed: int):
        self.clock = clock
        self.bus = bus
        self.keys = keys
        self.ledger = ledger
        self.rng = np.random.default_rng(seed + 3)
        self.on_promote = None  # set by the engine: callback(checkpoint)
        self.config = {
            "rounds": 30, "local_epochs": 1, "lr": 2e-4, "clip_norm": 1.0, "noise_multiplier": 1.1,
            "batch_size": BATCH, "target_eps": 3.0, "delta": DELTA, "strategy": "trimmed_mean",
            "secagg_mode": "enclave", "auto_checkpoint_every": 2, "lora_rank": 16, "lora_alpha": 32,
            "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
        }
        self.run = {
            "id": "fl-2026-10-31-a", "base_model": "llama-3.2-1b-instruct",
            "base_hash": sha256_hex("llama-3.2-1b-instruct-weights"),
            "task": "network-ops instruction tuning (RCA + tool plans, vendor A/B dialects)",
            "started_at": None,
        }
        self.state = "idle"  # idle | running | paused | stopped | complete
        self.round = 0
        self.phase = "idle"
        self.phase_t = 0.0
        self.global_loss = 2.35
        self.global_acc = 0.31
        self.ops = []
        for o in OPERATORS:
            self.ops.append({
                **o, "q": BATCH / o["samples"], "status": "idle", "participating": True, "step": 0,
                "loss": 2.4, "clip_frac": 0.0, "eps_sq": 0.0, "eps": 0.0, "update_norm": 0.0,
                "gpu": 0.0, "bytes_up": 0, "train_time": PHASE_TIME["train"] / o["speed"],
                "local_only": 2.4, "warned": False, "poison_next": False, "commitment": sha256_hex(o["id"] + "-data"),
                "grad_hist": [], "phase": "idle",
            })
        self.rounds: list[dict] = []
        self.current: dict | None = None
        self.checkpoints: list[dict] = []
        self.annotations: list[dict] = []
        self.step_series: list[dict] = []  # sparse live loss series
        self.secagg: dict | None = None
        self.he = {
            "scheme": "CKKS", "poly_modulus_degree": 8192, "coeff_modulus_bits": [60, 40, 40, 60],
            "scale_bits": 40, "security_bits": 128, "slice_weights": 4096,
            "plain_kib": 16, "cipher_kib": 512, "noise_budget": [], "error": None, "timings_ms": {},
        }
        self.sovereignty = {"adapter_bytes": 0, "metric_bytes": 0, "raw_bytes": 0,
                            "per_op": {o["id"]: {"adapter": 0, "metrics": 0, "raw": 0} for o in OPERATORS}}
        self.reveal = False
        self._emit_acc = 0.0
        self._hist_acc = 0.0
        self.hold_at_boundary = False

    # ------------------------------------------------------------------ controls

    def start(self) -> None:
        if self.state in ("idle", "paused", "stopped", "ready"):
            if not self.run["started_at"]:
                self.run["started_at"] = self.clock.iso()
            self.state = "running"
            self.hold_at_boundary = False
            if self.phase in ("idle", "done"):
                self._begin_round()
            self.bus.emit("fl.run", "learn", f"training running, round {self.round}", self.view_run(), "notice",
                          "aggregator")

    def pause(self) -> None:
        if self.state == "running":
            self.state = "paused"
            self.bus.emit("fl.run", "learn", "training paused", self.view_run(), "notice", "aggregator")

    def stop(self) -> None:
        self.state = "stopped"
        self.phase = "done"
        self.ledger.append("fl.stop", "human:a.rivera", "learn", {"run": self.run["id"], "round": self.round},
                           key_id="kms:spectrum/human/a.rivera")
        self.bus.emit("fl.run", "learn", "training stopped by operator", self.view_run(), "warn", "human:a.rivera")

    def next_round(self) -> None:
        if self.phase == "train":
            for o in self.ops:
                if o["status"] == "training":
                    o["step"] = STEPS_PER_ROUND
            self.phase_t = 999

    def set_config(self, changes: dict, actor: str = "human:a.rivera") -> dict:
        before = {k: self.config[k] for k in changes if k in self.config}
        for k, v in changes.items():
            if k in self.config:
                self.config[k] = v
        label = ", ".join(f"{k} {before[k]} → {changes[k]}" for k in before)
        ann = {"round": self.round, "t": r(self.clock.t, 1), "label": label, "kind": "config"}
        self.annotations.append(ann)
        rec = self.ledger.append("fl.config", actor, "learn", {"run": self.run["id"], "before": before,
                                                               "after": {k: changes[k] for k in before}},
                                 key_id="kms:spectrum/human/a.rivera")
        self.bus.emit("fl.config", "learn", f"config changed: {label}", {"config": self.config, "annotation": ann},
                      "notice", actor, ledger={"seq": rec["seq"], "hash": rec["hash"]})
        return self.config

    def projected_eps(self, sigma: float, rounds: int | None = None) -> dict:
        rounds = rounds or self.config["rounds"]
        remaining = max(0, rounds - max(self.round, 0)) * STEPS_PER_ROUND
        out = {}
        for o in self.ops:
            out[o["id"]] = r(math.sqrt(o["eps_sq"] + eps_increment(o["q"], remaining, sigma)), 3)
        return out

    def client(self, op_id: str, action: str) -> None:
        o = next(x for x in self.ops if x["id"] == op_id)
        if action == "pause":
            o["participating"] = not o["participating"]
            msg = f"{o['short']} {'resumed' if o['participating'] else 'paused'} participation"
        elif action == "drop":
            o["status"] = "dropped"
            o["participating"] = False
            msg = f"{o['short']} dropped out; Shamir shares let the round complete"
        elif action == "poison":
            o["poison_next"] = True
            msg = f"{o['short']} compromised: next update will be poisoned"
        else:
            raise ValueError(action)
        self.bus.emit("fl.client", "learn", msg, {"op": op_id, "action": action}, "warn", op_id)

    def save_checkpoint(self, manual: bool = True, emit: bool = True) -> dict:
        n = len(self.checkpoints) + 1
        h = sha256_hex(f"adapter-{self.run['id']}-r{self.round}-{self.global_loss:.5f}")
        ck = {
            "id": f"ckpt-r{self.round:02d}" + ("" if not any(c["round"] == self.round for c in self.checkpoints)
                                              else f"-{n}"),
            "round": self.round, "val_loss": r(self.global_loss, 4), "accuracy": r(self.global_acc, 4),
            "eps_max": r(max(o["eps"] for o in self.ops), 3), "delta": DELTA,
            "size_bytes": ADAPTER_BYTES, "adapter_hash": h, "manual": manual,
            "s3_uri": f"s3://spectrum-fl-checkpoints/{self.run['id']}/r{self.round:02d}/adapter.safetensors",
            "created_at": self.clock.iso(), "promoted": False,
        }
        rec = self.ledger.append("fl.checkpoint", "aggregator", "learn",
                                 {k: ck[k] for k in ("id", "round", "val_loss", "accuracy", "eps_max", "adapter_hash",
                                                     "s3_uri")},
                                 key_id="kms:spectrum/aggregator", emit=emit)
        ck["ledger_seq"] = rec["seq"]
        ck["sig"] = rec["sig"]
        self.checkpoints.append(ck)
        self.annotations.append({"round": self.round, "t": r(self.clock.t, 1), "label": ck["id"], "kind": "checkpoint"})
        if emit:
            self.bus.emit("fl.checkpoint", "learn", f"checkpoint {ck['id']} saved and signed", ck, "notice",
                          "aggregator", ledger={"seq": rec["seq"], "hash": rec["hash"]})
        return ck

    def restore_checkpoint(self, ck_id: str) -> dict:
        ck = next(c for c in self.checkpoints if c["id"] == ck_id)
        self.global_loss = ck["val_loss"]
        self.global_acc = ck["accuracy"]
        self.annotations.append({"round": self.round, "t": r(self.clock.t, 1), "label": f"restored {ck_id}",
                                 "kind": "restore"})
        rec = self.ledger.append("fl.restore", "human:a.rivera", "learn", {"checkpoint": ck_id,
                                                                           "adapter_hash": ck["adapter_hash"]},
                                 key_id="kms:spectrum/human/a.rivera")
        self.bus.emit("fl.checkpoint", "learn", f"global model restored to {ck_id}", ck, "warn", "human:a.rivera",
                      ledger={"seq": rec["seq"], "hash": rec["hash"]})
        return ck

    def promote(self, ck_id: str | None = None) -> dict:
        ck = next(c for c in self.checkpoints if c["id"] == ck_id) if ck_id else self.checkpoints[-1]
        ck["promoted"] = True
        self.bus.emit("fl.promote", "learn", f"{ck['id']} promoted to the Proving Ground", ck, "notice", "aggregator")
        if self.on_promote:
            self.on_promote(ck)
        return ck

    # ------------------------------------------------------------------ simulation

    def _target_loss(self, rnd: int) -> float:
        sigma_pen = 0.09 * (self.config["noise_multiplier"] - 1.1)
        return 0.56 + sigma_pen + 1.72 * math.exp(-0.21 * rnd)

    def _begin_round(self) -> None:
        self.round += 1
        self.phase = "pull"
        self.phase_t = 0.0
        t0 = self.clock.t
        self.current = {"round": self.round, "t0": r(t0, 2), "phases": {o["id"]: [] for o in self.ops},
                        "aggregator": [], "participants": []}
        for o in self.ops:
            if o["status"] == "budget_exhausted":
                continue
            if o["participating"] and o["status"] != "dropped":
                o["status"] = "pulling"
                o["phase"] = "pull"
                o["step"] = 0
                o["loss"] = self.global_loss + 0.04 + 0.03 * self.rng.standard_normal()
                self.current["participants"].append(o["id"])
                self.current["phases"][o["id"]].append({"phase": "pull", "start": r(t0, 2)})
        self.bus.emit("fl.round.phase", "learn", f"round {self.round} started: {len(self.current['participants'])} "
                      f"operators pulling global adapter", {"round": self.round, "phase": "pull",
                                                            "current": self.current}, "info", "aggregator")

    def _mark(self, op_id: str, phase: str) -> None:
        segs = self.current["phases"][op_id]
        if segs and "end" not in segs[-1]:
            segs[-1]["end"] = r(self.clock.t, 2)
        if phase:
            segs.append({"phase": phase, "start": r(self.clock.t, 2)})

    def tick(self, dt: float) -> None:
        if self.state != "running" or not self.current:
            return
        self.phase_t += dt
        cfg = self.config
        parts = [o for o in self.ops if o["id"] in self.current["participants"]]

        if self.phase == "pull" and self.phase_t >= PHASE_TIME["pull"]:
            self.phase = "train"
            self.phase_t = 0.0
            for o in parts:
                o["status"] = "training"
                o["phase"] = "train"
                self._mark(o["id"], "train")
            self.bus.emit("fl.round.phase", "learn", f"round {self.round}: local DP-SGD training",
                          {"round": self.round, "phase": "train", "current": self.current}, "info", "aggregator")

        elif self.phase == "train":
            all_done = True
            for o in parts:
                if o["status"] == "dropped":
                    continue
                if o["status"] == "training":
                    rate = STEPS_PER_ROUND / o["train_time"]
                    new_step = min(STEPS_PER_ROUND, o["step"] + rate * dt)
                    d_steps = new_step - o["step"]
                    o["step"] = new_step
                    o["eps_sq"] += eps_increment(o["q"], d_steps, cfg["noise_multiplier"])
                    o["eps"] = math.sqrt(o["eps_sq"])
                    prog = new_step / STEPS_PER_ROUND
                    start = self.global_loss + 0.05
                    o["loss"] = max(0.3, start * (1 - 0.11 * prog) + 0.025 * self.rng.standard_normal())
                    o["gpu"] = float(np.clip(0.86 + 0.06 * self.rng.standard_normal(), 0.6, 0.99))
                    if o["step"] >= STEPS_PER_ROUND:
                        o["status"] = "securing"
                        o["phase"] = "secure"
                        o["gpu"] = 0.12
                        self._mark(o["id"], "secure")
                        o["_t"] = 0.0
                    else:
                        all_done = False
                    self._check_budget(o)
                elif o["status"] == "securing":
                    o["_t"] += dt
                    if o["_t"] >= PHASE_TIME["secure"] * (1.6 if cfg["secagg_mode"] == "he" else 1.0):
                        o["status"] = "uploading"
                        o["phase"] = "upload"
                        self._mark(o["id"], "upload")
                        o["_t"] = 0.0
                    all_done = False
                elif o["status"] == "uploading":
                    o["_t"] += dt
                    if o["_t"] >= PHASE_TIME["upload"]:
                        o["status"] = "waiting"
                        o["phase"] = "waiting"
                        o["bytes_up"] += ADAPTER_BYTES
                        self.sovereignty["adapter_bytes"] += ADAPTER_BYTES
                        self.sovereignty["metric_bytes"] += 6144
                        self.sovereignty["per_op"][o["id"]]["adapter"] += ADAPTER_BYTES
                        self.sovereignty["per_op"][o["id"]]["metrics"] += 6144
                        self._mark(o["id"], "")
                        self.bus.emit("fl.upload", "learn", f"{o['short']} uploaded masked adapter update "
                                      f"({ADAPTER_BYTES / 2**20:.1f} MiB)", {"op": o["id"], "round": self.round,
                                                                            "bytes": ADAPTER_BYTES}, "info", o["id"])
                    all_done = False
            self._emit_steps(dt, parts)
            if all_done or self.phase_t > 999:
                for o in parts:
                    if o["status"] in ("training", "securing", "uploading"):
                        o["status"] = "waiting"
                        self._mark(o["id"], "")
                self.phase = "aggregate"
                self.phase_t = 0.0
                self.current["aggregator"].append({"phase": "aggregate", "start": r(self.clock.t, 2)})
                self._compute_secagg(parts)
                self.bus.emit("fl.round.phase", "learn", f"round {self.round}: secure aggregation in enclave",
                              {"round": self.round, "phase": "aggregate", "current": self.current}, "info",
                              "aggregator")

        elif self.phase == "aggregate" and self.phase_t >= PHASE_TIME["aggregate"]:
            self._finish_round(parts)

    def _check_budget(self, o: dict) -> None:
        tgt = self.config["target_eps"]
        if o["eps"] >= 0.9 * tgt and not o["warned"]:
            o["warned"] = True
            self.bus.emit("fl.budget", "learn", f"{o['short']} at {100 * o['eps'] / tgt:.0f}% of privacy budget "
                          f"(ε {o['eps']:.2f} of {tgt})", {"op": o["id"], "eps": r(o["eps"]), "target": tgt},
                          "warn", o["id"])
        if o["eps"] >= tgt:
            o["status"] = "budget_exhausted"
            o["step"] = STEPS_PER_ROUND
            self.bus.emit("fl.budget", "learn", f"{o['short']} reached its ε budget and leaves the federation",
                          {"op": o["id"], "eps": r(o["eps"]), "target": tgt}, "critical", o["id"])

    def _emit_steps(self, dt: float, parts: list[dict]) -> None:
        self._emit_acc += dt
        self._hist_acc += dt
        if self._emit_acc >= 0.25:
            self._emit_acc = 0.0
            rows = []
            for o in parts:
                rows.append({"op": o["id"], "step": int(o["step"]), "loss": r(o["loss"], 4), "eps": r(o["eps"], 4),
                             "clip_frac": r(o["clip_frac"], 4), "gpu": r(o["gpu"], 3), "status": o["status"]})
            self.step_series.append({"t": r(self.clock.t, 2), "round": self.round,
                                     "loss": {x["op"]: x["loss"] for x in rows},
                                     "eps": {x["op"]: x["eps"] for x in rows}})
            self.step_series = self.step_series[-1200:]
            self.bus.emit("fl.step", "learn", "", {"round": self.round, "rows": rows}, "debug", "aggregator",
                          record=False)
        if self._hist_acc >= 1.0:
            self._hist_acc = 0.0
            hists = {}
            C = self.config["clip_norm"]
            for o in parts:
                if o["status"] != "training":
                    continue
                med = C * (0.5 + 0.62 * math.exp(-0.13 * self.round)) * (1 + 0.15 * (o["floor"] - 1))
                vals = self.rng.lognormal(math.log(med), 0.42, 512)
                o["clip_frac"] = float((vals > C).mean())
                counts, _ = np.histogram(vals, bins=40, range=(0, 3 * C))
                o["grad_hist"] = counts.tolist()
                hists[o["id"]] = {"counts": o["grad_hist"], "clip_frac": r(o["clip_frac"], 4), "median": r(med, 4)}
            if hists:
                self.bus.emit("fl.gradhist", "learn", "", {"round": self.round, "clip_norm": C, "range": [0, 3 * C],
                                                           "hists": hists}, "debug", "aggregator", record=False)

    def _compute_secagg(self, parts: list[dict]) -> None:
        """Real pairwise-mask secure aggregation over fixed-point ints mod 2^32."""
        n = len(parts)
        base = np.outer(np.sin(np.linspace(0, 3, MATRIX)), np.cos(np.linspace(0, 2.2, MATRIX))) * 0.08
        true = []
        for i, o in enumerate(parts):
            u = base * (1 + 0.2 * i) + self.rng.normal(0, 0.012, (MATRIX, MATRIX))
            norm = float(np.clip(1.0 * math.exp(-0.035 * self.round) * (1 + 0.07 * self.rng.standard_normal()), 0.4,
                                 2))
            if o["poison_next"]:
                u = -u * 9 + self.rng.normal(0, 0.05, (MATRIX, MATRIX))
                norm = 0  # set below relative to median
            o["update_norm"] = norm
            true.append(u)
        norms = [o["update_norm"] for o in parts if not o["poison_next"]]
        med = float(np.median(norms)) if norms else 1.0
        for o in parts:
            if o["poison_next"]:
                o["update_norm"] = med * 14.2
        q_true = [np.round(u * SCALE).astype(np.int64) % MOD for u in true]
        masks = {}
        for i in range(n):
            for j in range(i + 1, n):
                masks[(i, j)] = self.rng.integers(0, MOD, (MATRIX, MATRIX), dtype=np.int64)
        masked = []
        for i in range(n):
            m = q_true[i].copy()
            for j in range(n):
                if i < j:
                    m = (m + masks[(i, j)]) % MOD
                elif j < i:
                    m = (m - masks[(j, i)]) % MOD
            masked.append(m)
        sum_masked = sum(masked) % MOD
        sum_true = sum(q_true) % MOD
        max_err = int(np.abs(sum_masked - sum_true).max())

        def deq(a):
            a = np.where(a >= MOD // 2, a - MOD, a)
            return a.astype(np.float64) / SCALE

        agg = deq(sum_masked) / max(n, 1)
        # what the host sees: masked values look uniform; show them centred to [-1, 1]
        host = [((m.astype(np.float64) / MOD) * 2 - 1) for m in masked]
        self.secagg = {
            "round": self.round, "ops": [o["id"] for o in parts], "mode": self.config["secagg_mode"],
            "masked": [np.round(h, 3).tolist() for h in host],
            "true": [np.round(u, 4).tolist() for u in true],
            "aggregate": np.round(agg, 4).tolist(),
            "max_abs_error": max_err, "pairwise_masks": len(masks),
            "dropout_recovery": "Shamir t=3 of 5", "fixed_point": "2^16 scale, mod 2^32",
        }
        if self.config["secagg_mode"] == "he":
            self.he["noise_budget"] = [{"op": "encrypt", "bits": 97}, *[
                {"op": f"add {o['short']}", "bits": 97 - 2 * (k + 1)} for k, o in enumerate(parts)],
                {"op": "rescale", "bits": 57}, {"op": "decrypt", "bits": 57}]
        else:
            self.he["noise_budget"] = [{"op": "encrypt", "bits": 97}, *[
                {"op": f"add {o['short']}", "bits": 97 - 2 * (k + 1)} for k, o in enumerate(parts)],
                {"op": "rescale", "bits": 57}, {"op": "decrypt", "bits": 57}]
        self.he["error"] = float(f"{abs(self.rng.normal(3e-7, 8e-8)):.2e}")
        self.he["timings_ms"] = {"encrypt": int(41 + self.rng.integers(0, 9)), "aggregate": int(12 + self.rng.integers(0, 4)),
                                 "decrypt": int(9 + self.rng.integers(0, 3))}
        self.bus.emit("fl.secagg", "learn", f"round {self.round}: {len(masks)} pairwise masks cancelled, "
                      f"max |Σmasked − Σtrue| = {max_err}", self.secagg, "info", "aggregator")

    def _finish_round(self, parts: list[dict]) -> None:
        cfg = self.config
        norms = [o["update_norm"] for o in parts]
        med = float(np.median(norms)) if norms else 1.0
        mad = float(np.median([abs(x - med) for x in norms])) if norms else 0.1
        threshold = med + 4 * max(mad, 0.04)
        decisions = {}
        poisoned_in = False
        for o in parts:
            if o["update_norm"] > threshold:
                if cfg["secagg_mode"] == "enclave" and cfg["strategy"] != "fedavg":
                    decisions[o["id"]] = "rejected"
                else:
                    decisions[o["id"]] = "accepted"
                    poisoned_in = o["poison_next"]
                if o["poison_next"]:
                    sev = "critical"
                    verdict = decisions[o["id"]]
                    self.bus.emit("fl.poison", "learn",
                                  f"{o['short']} round {self.round}: update norm {o['update_norm'] / med:.1f}× median, "
                                  + ("rejected by robust aggregation in the enclave" if verdict == "rejected"
                                     else "server is blind in this mode, so the update was aggregated"),
                                  {"op": o["id"], "round": self.round, "ratio": r(o["update_norm"] / med, 2),
                                   "decision": verdict, "mode": cfg["secagg_mode"], "strategy": cfg["strategy"]},
                                  sev, "aggregator")
            else:
                decisions[o["id"]] = "accepted"
            o["poison_next"] = False

        target = self._target_loss(self.round)
        self.global_loss = target + 0.012 * self.rng.standard_normal() + (0.34 if poisoned_in else 0.0)
        self.global_acc = float(np.clip(0.885 - 0.58 * math.exp(-0.19 * self.round) + 0.004 *
                                        self.rng.standard_normal() - (0.09 if poisoned_in else 0), 0, 1))
        local_only = {}
        for o in self.ops:
            lo = o["floor"] + 1.5 * math.exp(-0.16 * self.round) + 0.015 * self.rng.standard_normal()
            o["local_only"] = lo
            local_only[o["id"]] = r(lo, 4)
        self.current["aggregator"][-1]["end"] = r(self.clock.t, 2)
        for o in parts:
            if o["status"] not in ("budget_exhausted", "dropped"):
                o["status"] = "idle"
                o["phase"] = "idle"
        rec_payload = {
            "run": self.run["id"], "round": self.round, "participants": [o["id"] for o in parts],
            "val_loss": r(self.global_loss, 4), "accuracy": r(self.global_acc, 4),
            "decisions": decisions, "strategy": cfg["strategy"], "secagg_mode": cfg["secagg_mode"],
            "eps": {o["id"]: r(o["eps"], 4) for o in self.ops}, "sigma": cfg["noise_multiplier"],
            "clip": cfg["clip_norm"],
            "commitments": {o["id"]: sha256_hex(f"{o['commitment']}-r{self.round}") for o in parts},
            "attestation_pcr0": "verified" if cfg["secagg_mode"] == "enclave" else "n/a",
        }
        rec = self.ledger.append("fl.round", "aggregator", "learn", rec_payload, key_id="kms:spectrum/aggregator",
                                 emit=not self.bus.muted)
        summary = {
            "round": self.round, "t0": self.current["t0"], "t1": r(self.clock.t, 2), "val_loss": r(self.global_loss, 4),
            "accuracy": r(self.global_acc, 4), "local_only": local_only,
            "eps": {o["id"]: r(o["eps"], 4) for o in self.ops},
            "update_norms": {o["id"]: r(o["update_norm"], 4) for o in parts},
            "threshold": r(threshold, 4), "median_norm": r(med, 4), "decisions": decisions,
            "phases": self.current["phases"], "aggregator": self.current["aggregator"],
            "participants": [o["id"] for o in parts], "ledger_seq": rec["seq"],
            "sigma": cfg["noise_multiplier"], "clip": cfg["clip_norm"],
        }
        self.rounds.append(summary)
        self.bus.emit("fl.round", "learn", f"round {self.round} aggregated: val loss {self.global_loss:.3f}, "
                      f"accuracy {100 * self.global_acc:.1f}%", summary, "notice", "aggregator",
                      ledger={"seq": rec["seq"], "hash": rec["hash"]})
        self.current = None
        self.phase = "done"
        if cfg["auto_checkpoint_every"] and self.round % cfg["auto_checkpoint_every"] == 0:
            self.save_checkpoint(manual=False, emit=not self.bus.muted)
        if self.round >= cfg["rounds"]:
            self.state = "complete"
            self.bus.emit("fl.run", "learn", "training complete", self.view_run(), "notice", "aggregator")
        elif self.hold_at_boundary:
            self.state = "ready"
            self.bus.emit("fl.run", "learn", f"round {self.round + 1} queued", self.view_run(), "info", "aggregator")
        else:
            self._begin_round()

    # ------------------------------------------------------------------ views

    def view_run(self) -> dict:
        return {"run": self.run, "state": self.state, "round": self.round, "phase": self.phase,
                "config": self.config, "rounds_total": self.config["rounds"]}

    def snapshot(self) -> dict:
        return {
            **self.view_run(),
            "operators": [{k: (r(v, 4) if isinstance(v, float) else v) for k, v in o.items()
                           if k not in ("eps_sq", "_t")} for o in self.ops],
            "rounds_log": self.rounds,
            "current": self.current,
            "checkpoints": self.checkpoints,
            "annotations": self.annotations[-40:],
            "step_series": self.step_series[-240:],
            "secagg": self.secagg,
            "he": self.he,
            "sovereignty": self.sovereignty,
            "attestation_required": self.config["secagg_mode"] == "enclave",
            "steps_per_round": STEPS_PER_ROUND,
            "global": {"val_loss": r(self.global_loss, 4), "accuracy": r(self.global_acc, 4)},
            "reveal": self.reveal,
        }
