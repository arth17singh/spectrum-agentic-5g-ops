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

Optional post-processing, read from the meta sidecar (absent = off, so ppo_v0 and ppo_sm_50k behave
exactly as before): ``"postprocess": {"deadzone_power_db": x, "deadzone_tilt_deg": y}`` zeroes a power
(tilt) delta smaller than x dB (y deg) in magnitude, after scaling and before the safety filter. The
fault-aware remediation policy (t1/rl/remediation) uses it so that "no change" is an exact no-op.

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
        # "masked" (default, v0: t1/rl/obs.py) or "sdk": the SDK trainer's own observation (ran_rl
        # training/train.py, MultiInputPolicy on RANOptimizationEnv, no wrapper), as the SageMaker-trained
        # checkpoints were trained and evaluated; prev_action is zero (a stateless tool call), as at reset.
        self.obs_mode = str(meta.get("obs", "masked"))
        if self.obs_mode not in ("masked", "sdk"):
            raise ValueError(f"unknown observation mode {self.obs_mode!r} in the policy's meta")
        post = meta.get("postprocess") or {}
        if not isinstance(post, Mapping) or set(post) - {"deadzone_power_db", "deadzone_tilt_deg"}:
            raise ValueError(f"unknown postprocess in the policy's meta: {post!r}")
        self.deadzone = (float(post.get("deadzone_power_db", 0.0)), float(post.get("deadzone_tilt_deg", 0.0)))
        if min(self.deadzone) < 0:
            raise ValueError(f"a deadzone must not be negative: {self.deadzone}")

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

    def observation(self, state: Mapping[str, Any]) -> dict[str, np.ndarray]:
        """The observation the model sees for ``state`` (training code uses this same function)."""
        return observation_for(state, self.obs_mode, self._cfg)

    def _normalised_action(self, state: Mapping[str, Any]) -> np.ndarray:
        action, _ = self._model.predict(self.observation(state), deterministic=True)
        return np.asarray(action, dtype=np.float64)

    def _filtered(self, state: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        from ran_rl.env.observation import scale_action

        s = self._full_state(state)
        n = len(s["power_dbm"])
        raw = quantise_toward_zero(scale_action(self._normalised_action(s)[: 2 * n], self._bounds))
        dz_power, dz_tilt = self.deadzone
        if dz_power or dz_tilt:
            raw[:n] = np.where(np.abs(raw[:n]) < dz_power, 0.0, raw[:n])
            raw[n:] = np.where(np.abs(raw[n:]) < dz_tilt, 0.0, raw[n:])
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


def observation_for(state: Mapping[str, Any], mode: str = "masked", cfg: Any = None) -> dict[str, np.ndarray]:
    """The model observation of a raw twin state: the SDK's normalisation with prev_action = 0 (a stateless
    tool call), then the masked view when ``mode`` is "masked". Shared by inference and training."""
    from ran_rl.env.config import EnvConfig
    from ran_rl.env.observation import normalize_observation

    s = Policy._full_state(state)
    n = len(s["power_dbm"])
    obs = normalize_observation(s, np.zeros(2 * n), cfg if cfg is not None else EnvConfig())
    if mode == "masked":   # v0 was trained on the masked view; the SDK's own trainer was not
        obs = mask_obs(obs)
    elif mode != "sdk":
        raise ValueError(f"unknown observation mode {mode!r}")
    return obs


def load(path: str | Path, **meta: Any) -> Policy:
    from stable_baselines3 import PPO

    path = Path(path)
    sidecar = path.with_suffix(".meta.json")
    m = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    m.update(meta)
    return Policy(PPO.load(str(path), device="cpu"), m)
