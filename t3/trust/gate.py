"""TrustGate: a trust score per (model_id, version) of the Track 2 predictor, updated once per
applied, scored change (contract amendment 1, A5-A7).

Inputs are deltas against the no-change counterfactual at the same step (A5):
``predicted_delta`` (what ``ran___predict`` said) and ``measured_delta`` (twin after the change
minus the replay of the same step with no change).

Per change:

    e = mean over k in {coverage_ratio, sla_ratio, sinr_p10_db} of |pred_delta_k - meas_delta_k| / scale_k
    D = 1 if the direction agrees, 0 if not (direction_agreement on the deltas, dead band per key)
    s = w_err * (1 - min(1, e)) + w_dir * D       (s = 1 - min(1, e) when no key moved materially)

Trust is a bias-corrected EWMA of s (as in Adam): m_n = (1 - a) m_{n-1} + a s_n, m_0 = 0,
trust_n = m_n / (1 - (1 - a)^n): a weighted average of observed scores only (no prior).

Pause rule (A6), only once n >= warmup (3) scored changes:
    at least ``recent_needed`` (2) of the last ``recent_window`` (3) changes have e > tau,  OR
    trust < pause_below.
The pause happens once: the store's atomic ``claim_pause`` wins, the injected policy client adds
the Cedar forbid on ``ran___predict`` for that version, and ONE ``control`` record
(``by: "trust-gate"``) is written. ``resume`` is for a person only and starts a new probation
epoch (warm-up and the 2-of-3 window start over).

Skipped, never scored: probes (``probe: true``), ``scored: false`` records (e.g. clipped by the
safety filter), records without a ``predictor`` or without ``predicted_delta`` (A6: PPO and
other policies make no KPI prediction; a ``policy`` field alone is never scored).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Callable

from .records import direction_agreement, error_e, make_control_record, utc_now

THRESHOLDS_PATH = Path(__file__).with_name("thresholds.json")


def load_thresholds(path: str | Path = THRESHOLDS_PATH) -> tuple[dict, str]:
    """Return (thresholds, sha256 of the file's bytes). The hash goes into declarations."""
    raw = Path(path).read_bytes()
    th = json.loads(raw)
    validate_thresholds(th)
    return th, hashlib.sha256(raw).hexdigest()


def validate_thresholds(th: dict) -> None:
    keys = th["keys"]
    if not keys or set(keys) != set(th["scales"]) or set(keys) != set(th["direction_deadband"]):
        raise ValueError("thresholds: 'keys', 'scales' and 'direction_deadband' must name the same KPIs")
    if any(not (float(v) > 0) for v in th["scales"].values()):
        raise ValueError("thresholds: every scale must be > 0")
    if not 0 < float(th["alpha"]) <= 1:
        raise ValueError("thresholds: alpha must be in (0, 1]")
    if int(th["warmup"]) < 1:
        raise ValueError("thresholds: warmup must be >= 1")
    if not 1 <= int(th["recent_needed"]) <= int(th["recent_window"]):
        raise ValueError("thresholds: need 1 <= recent_needed <= recent_window")
    if not float(th["tau"]) > 0:
        raise ValueError("thresholds: tau must be > 0")
    w = th["weights"]
    if abs(float(w["error"]) + float(w["direction"]) - 1.0) > 1e-9 or min(w.values()) < 0:
        raise ValueError("thresholds: weights must be >= 0 and sum to 1")
    if not 0 < float(th["pause_below"]) <= float(th.get("watch_below", th["pause_below"])) <= 1:
        raise ValueError("thresholds: need 0 < pause_below <= watch_below <= 1")


def skip_reason(rec: dict) -> str | None:
    if rec.get("probe"):
        return "probe (no model)"
    if rec.get("scored") is False:
        return f"unscored: {rec.get('unscored_reason', 'unspecified')}"
    p = rec.get("predictor") or {}
    if not (p.get("model_id") and p.get("version")):
        return "no predictor on the record (only the Track 2 predictor is scored)"
    if not rec.get("predicted_delta") or not rec.get("measured_delta"):
        return "no predicted_delta / measured_delta"
    return None


class TrustGate:
    def __init__(self, store, thresholds: dict | str | Path | None = None, policy_client=None,
                 writer=None, clock: Callable[[], str] | None = None, *, synthetic: bool = False):
        if thresholds is None or isinstance(thresholds, (str, Path)):
            self.thresholds, self.thresholds_sha256 = load_thresholds(thresholds or THRESHOLDS_PATH)
        else:
            validate_thresholds(thresholds)
            self.thresholds = thresholds
            self.thresholds_sha256 = hashlib.sha256(
                json.dumps(thresholds, sort_keys=True).encode()).hexdigest()
        self.store = store
        self.policy_client = policy_client
        self.writer = writer
        self.clock = clock or utc_now
        self.synthetic = synthetic
        self.last_control_paths: list[str] = []
        th = self.thresholds
        self.keys = list(th["keys"])
        self.scales = {k: float(th["scales"][k]) for k in self.keys}
        self.deadband = {k: float(th["direction_deadband"][k]) for k in self.keys}

    # ------------------------------------------------------------------ scoring
    def score(self, predicted_delta: dict, measured_delta: dict) -> dict:
        th = self.thresholds
        e = error_e(predicted_delta, measured_delta, self.scales)
        d = direction_agreement(predicted_delta, measured_delta, self.deadband, self.keys)
        e_clip = min(1.0, e)
        if d is None:
            s = 1.0 - e_clip
        else:
            s = float(th["weights"]["error"]) * (1.0 - e_clip) + float(th["weights"]["direction"]) * (1.0 if d else 0.0)
        return {"e": e, "e_over_tau": e > float(th["tau"]), "direction_agree": d, "score": s}

    @staticmethod
    def ewma_step(m: float, n: int, s: float, alpha: float) -> tuple[float, int, float]:
        """One bias-corrected EWMA step: returns (m, n, trust)."""
        m = (1.0 - alpha) * m + alpha * s
        n += 1
        return m, n, m / (1.0 - (1.0 - alpha) ** n)

    def pause_trigger(self, n: int, trust: float, recent_e: list[float]) -> str | None:
        th = self.thresholds
        if n < int(th["warmup"]):
            return None
        window = recent_e[-int(th["recent_window"]):]
        over = sum(1 for e in window if e > float(th["tau"]))
        if over >= int(th["recent_needed"]):
            return f"{over} of the last {len(window)} changes had e > tau {float(th['tau']):g}"
        if trust < float(th["pause_below"]):
            return f"trust {trust:.3f} < pause line {float(th['pause_below']):g}"
        return None

    # ------------------------------------------------------------------ update
    def update(self, change_record: dict) -> dict:
        why = skip_reason(change_record)
        if why is not None:
            return {"trust_before": None, "trust_after": None, "action": None, "skipped": why}
        mid = str(change_record["predictor"]["model_id"])
        ver = str(change_record["predictor"]["version"])
        th = self.thresholds
        sc = self.score(change_record["predicted_delta"], change_record["measured_delta"])
        st = self.store.get(mid, ver)
        cid = change_record.get("change_id")
        if cid and any(h.get("change_id") == cid and h.get("epoch") == st["epoch"] for h in st["history"]):
            # a retried call (sync Lambda retry, resent apply): never score the same change twice
            prev = [h for h in st["history"] if h.get("change_id") == cid][-1]
            return {"model_id": mid, "version": ver, "trust_before": prev.get("trust_before"),
                    "trust_after": prev["trust"], "e": prev["e"], "score": prev["score"], "action": None,
                    "status": st["status"], "n": st["n"], "skipped": "duplicate change_id (already scored)"}
        before = st["trust"]
        st["m"], st["n"], st["trust"] = self.ewma_step(st["m"], st["n"], sc["score"], float(th["alpha"]))
        st["recent_e"] = (st.get("recent_e", []) + [sc["e"] if math.isfinite(sc["e"]) else 1e9])[-int(th["recent_window"]):]
        st["history"].append({"change_id": change_record.get("change_id"), "e": sc["e"] if math.isfinite(sc["e"]) else 1e9, "score": sc["score"],
                              "trust_before": before, "trust": st["trust"], "epoch": st["epoch"]})
        self.store.put(mid, ver, st)
        status = self.store.get(mid, ver)["status"]
        out = {"model_id": mid, "version": ver, "trust_before": before, "trust_after": st["trust"],
               "n": st["n"], "epoch": st["epoch"], **sc, "action": None, "status": status,
               "band": self._band(st["trust"], st["n"])}
        trigger = self.pause_trigger(st["n"], st["trust"], st["recent_e"])
        if trigger and status == "active":
            reason = (f"{trigger} after {st['n']} scored changes; last change "
                      f"{change_record.get('change_id')} e={sc['e']:.3f} "
                      f"direction_agree={sc['direction_agree']}")
            ctrl = self.pause(mid, ver, reason, run_id=change_record.get("run_id"))
            if ctrl is not None:
                out["action"] = "pause"
                out["control"] = ctrl
            out["status"] = "paused"
        return out

    def _band(self, trust: float | None, n: int) -> str:
        th = self.thresholds
        if trust is None or n < int(th["warmup"]):
            return "warm-up"
        if trust < float(th["pause_below"]):
            return "below-pause-line"
        if trust < float(th.get("watch_below", th["pause_below"])):
            return "watch"
        return "trusted"

    # ------------------------------------------------------------------ control
    def pause(self, model_id: str, version: str, reason: str, *, run_id: str | None = None) -> dict | None:
        """Pause one model version. Idempotent: returns None if it is already paused."""
        if not run_id:
            raise ValueError("pause needs the run_id its control record belongs to")
        if not self.store.claim_pause(model_id, version):
            return None
        try:
            res = self.policy_client.pause(model_id, version, reason)
        except Exception:
            self.store.release_pause(model_id, version)  # let a later change retry
            raise
        st = self.store.get(model_id, version)
        st["policy_name"] = res["policy_name"]
        self.store.put(model_id, version, st)
        rec = make_control_record(run_id=run_id, model_id=model_id, version=version, action="pause",
                                  by="trust-gate", reason=reason, policy_name=res["policy_name"],
                                  time=self.clock(),
                                  synthetic=self.synthetic or bool(res.get("dry_run")))
        self._write(run_id, rec)
        return rec

    def resume(self, model_id: str, version: str, *, person: str, reason: str, run_id: str) -> dict | None:
        """A person lifts the pause. Starts a new probation epoch. None if it was not paused."""
        pseudo = person.removeprefix("person:")
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", pseudo):
            raise ValueError("resume needs a person pseudonym like 'a.rivera' (letters, digits, . _ -)")
        if not reason or not reason.strip():
            raise ValueError("resume needs a reason")
        st = self.store.get(model_id, version)
        if st["status"] != "paused":
            return None
        res = self.policy_client.resume(model_id, version)
        if not self.store.release_pause(model_id, version):
            return None
        name = st.get("policy_name") or ",".join(res.get("removed", [])) or "none"
        st = self.store.get(model_id, version)
        st.update({"m": 0.0, "n": 0, "trust": None, "recent_e": [], "epoch": st["epoch"] + 1,
                   "policy_name": None})
        self.store.put(model_id, version, st)
        rec = make_control_record(run_id=run_id, model_id=model_id, version=version, action="resume",
                                  by=f"person:{pseudo}", reason=reason, policy_name=name,
                                  time=self.clock(),
                                  synthetic=self.synthetic or bool(res.get("dry_run")))
        self._write(run_id, rec)
        return rec

    def _write(self, run_id: str, rec: dict) -> None:
        self.last_control_paths = []
        if self.writer is None:
            return
        append = getattr(self.writer, "append_records", self.writer)
        self.last_control_paths = list(append(run_id, "control", [rec]) or [])

    def state(self, model_id: str, version: str) -> dict[str, Any]:
        return self.store.get(model_id, version)
