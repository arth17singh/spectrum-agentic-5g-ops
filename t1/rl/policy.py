"""Track 1 PPO policy, inference side. v0, local, not federated / not the official job.

    from t1.rl.policy import load
    pol = load("t1/rl/artifacts/ppo_v0.zip")
    pol.propose(state)       -> [{"cell": 2, "d_power_db": -1.0, "d_tilt_deg": +0.0}, ...]   # safety-filtered
    pol.propose_text(state)  -> "cell=2;d_power_db=-1.0;d_tilt_deg=+0.0"                    # canonical string
    pol.model_id, pol.version

`state` is the SDK twin state dict for a 4-cell cluster (power_dbm, tilt_deg, ue_count or user_share,
rsrp_mean_dbm, rsrp_p10_dbm, sinr_mean_db, sinr_p10_db, coverage_ratio, sla_ratio; `alarm` optional).
`hour`, `phase` and absolute user counts are not used (t1/rl/obs.py). A tool call is stateless, so the
policy always sees prev_action = 0, exactly as in training (it was trained with prev_action masked).

The deterministic PPO action is scaled by the SDK's `scale_action`, quantised TOWARD ZERO to 0.1 dB /
0.1 deg (so the canonical string carries exactly what is applied), then passed through the SDK's
ActionSafetyFilter (step limits, absolute bounds, alarm freeze) and quantised again. Cells whose
change is zero are left out. An all-zero proposal is the empty list / empty string (no-op).

Canonical change string (contract amendment 1, A2): `cell=<int>;d_power_db=<+x.x>;d_tilt_deg=<+x.x>`,
several cells joined with `|` and sorted by cell, cells 0-based, one decimal, explicit sign.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from t1.rl.obs import UE_COUNT_MAX, mask_obs

LABEL = "v0, local, not federated / not the official job"
Q = 0.1


def quantise_toward_zero(v: np.ndarray, q: float = Q) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return np.trunc(np.round(np.abs(v) / q, 6)) * q * np.sign(v) + 0.0


def canon_change(change: Sequence[Mapping[str, Any]]) -> str:
    """List of {cell, d_power_db, d_tilt_deg} -> canonical string (sorted by cell, zero-change cells dropped
    only if the caller already dropped them; a cell listed with +0.0/+0.0 is kept as given)."""
    def f(v: float) -> str:
        r = round(float(v), 1)
        return f"{(r if r != 0 else 0.0):+.1f}"

    return "|".join(f"cell={int(c['cell'])};d_power_db={f(c['d_power_db'])};d_tilt_deg={f(c['d_tilt_deg'])}"
                    for c in sorted(change, key=lambda c: int(c["cell"])))


def parse_change(text: str) -> list[dict[str, float]]:
    out = []
    for part in filter(None, text.split("|")):
        kv = dict(p.split("=") for p in part.split(";"))
        out.append({"cell": int(kv["cell"]), "d_power_db": float(kv["d_power_db"]), "d_tilt_deg": float(kv["d_tilt_deg"])})
    return out


class Policy:
    def __init__(self, model: Any, meta: Mapping[str, Any] | None = None):
        from ran_rl.env.config import EnvConfig, SafetyBounds
        from ran_rl.env.safety import ActionSafetyFilter

        self._model = model
        self._cfg = EnvConfig()
        self._bounds: SafetyBounds = self._cfg.safety
        self._filter = ActionSafetyFilter(self._bounds)
        meta = dict(meta or {})
        self.meta = meta
        self.model_id = str(meta.get("model_id", "t1-ppo-v0"))
        self.version = str(meta.get("version", "v0"))

    # ---- state handling ------------------------------------------------------------------
    @staticmethod
    def _full_state(state: Mapping[str, Any]) -> dict[str, Any]:
        s = dict(state)
        n = len(s["power_dbm"])
        if "ue_count" not in s:
            share = np.asarray(s["user_share"], dtype=float)
            s["ue_count"] = (share * UE_COUNT_MAX / max(share.sum(), 1e-9) / 2).tolist()
        s.setdefault("alarm", [False] * n)
        s.setdefault("hour", 0.0)
        s.setdefault("phase", 0.0)
        s.setdefault("asleep", [False] * n)
        return s

    def _normalised_action(self, state: Mapping[str, Any]) -> np.ndarray:
        from ran_rl.env.observation import normalize_observation

        s = self._full_state(state)
        n = len(s["power_dbm"])
        obs = mask_obs(normalize_observation(s, np.zeros(2 * n), self._cfg))
        action, _ = self._model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float64)

    def _filtered(self, state: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        from ran_rl.env.observation import scale_action

        s = self._full_state(state)
        n = len(s["power_dbm"])
        raw = quantise_toward_zero(scale_action(self._normalised_action(s)[: 2 * n], self._bounds))
        res = self._filter.apply(raw, s)
        f = quantise_toward_zero(res.filtered)
        return f[:n], f[n:]

    # ---- public API ----------------------------------------------------------------------
    def propose(self, state: Mapping[str, Any]) -> list[dict[str, float]]:
        dp, dt = self._filtered(state)
        return [{"cell": i, "d_power_db": float(dp[i]), "d_tilt_deg": float(dt[i])}
                for i in range(len(dp)) if dp[i] != 0 or dt[i] != 0]

    def propose_text(self, state: Mapping[str, Any]) -> str:
        return canon_change(self.propose(state))

    def action(self, state: Mapping[str, Any]) -> np.ndarray:
        """The proposal as a normalised env action in [-1, 1] (for running the policy in the SDK env)."""
        dp, dt = self._filtered(state)
        b = self._bounds
        pw = np.where(dp >= 0, dp / b.power_delta_max_db, dp / -b.power_delta_min_db)
        return np.concatenate([pw, dt / b.tilt_delta_max_deg])


def load(path: str | Path, **meta: Any) -> Policy:
    from stable_baselines3 import PPO

    path = Path(path)
    sidecar = path.with_suffix(".meta.json")
    m = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    m.update(meta)
    return Policy(PPO.load(str(path), device="cpu"), m)
