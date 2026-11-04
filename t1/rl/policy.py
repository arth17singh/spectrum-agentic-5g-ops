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

Observation mode "masked_delta" (remediation-v2, t1/rl/remediation/v2.py): the masked view PLUS
``kpi_delta``, each KPI's change against the same network's pre-incident BASELINE state (the PM-counter history
an operator has): per cell ue share, rsrp mean and p10, sinr mean and p10 (SDK-normalised units), the network's
coverage and sla ratios, and per cell coverage share and sla share (the twin's per-cell counters,
``ReplayTwin.kpis()["per_cell"]``, carried in the state as ``coverage_share`` / ``sla_share`` lists); and
``cell_share``, the current per-cell coverage and sla shares. It never carries the baseline's CONFIGURATION:
power and tilt are not differenced and the baseline's power and tilt are never read, so a fault that does not
appear in the configuration record (a tilt drift, a failing amplifier) is seen only through its KPIs, never as
a config diff to revert. A policy in this mode needs ``baseline=`` on every call (propose, propose_text,
action, observation), and both states need the per-cell share lists.

THE NO-UE RULE (masked_delta only; ppo_sm_50k and v1 never pass through it). A cell SERVES UEs iff its
``ue_count`` is a finite number > 0. For a cell that serves none, the twin reports floors for its UE-defined
KPIs (rsrp mean/p10 -140 dBm, sinr mean/p10 -30 dB: ran_rl's _FLOOR_DBM / _FLOOR_DB) and None for its
coverage and sla shares; those values mean "undefined", not a measurement. So:

- ``kpi_delta``: a UE-defined KPI's delta for cell c (rsrp mean/p10, sinr mean/p10, coverage share, sla share)
  is 0.0 unless c serves UEs in BOTH the current state and the baseline and both values are finite; the ue-share
  delta is always defined (a cell that lost its users shows its lost share); a network ratio's delta is 0.0 if
  either ratio is non-finite.
- ``cell_share``: an undefined current share (no UEs, None or non-finite) is 0.0 before the 2x-1 mapping.
- ``serves_ues`` (8): the serving indicator, 1.0 or 0.0, for the current state's cells then the baseline's.
- Any other non-finite value v2 reads (None, NaN, +-inf) is undefined the same way: a non-finite UE-defined
  KPI zeroes that delta; a non-finite ue_count means the cell does not serve UEs. The v1 part of the observation
  (the SDK's normalised view of the current state) is computed after replacing a non-finite per-cell KPI with
  the twin's own floor, a non-finite ue_count with 0 and a non-finite network ratio with 0: what the twin reports
  for a cell without UEs; finite twin states pass unchanged.

The observation's length never changes: kpi_delta 30, cell_share 8, serves_ues 8. Training (v2.to_arrays) and
inference (Policy.propose, v2.BaselineAware) build it with this same function, observation_for.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from t1.rl.obs import UE_COUNT_MAX, mask_obs

LABEL = "v0, local, not federated / not the official job"
Q = 0.1
OBS_MODES = ("masked", "sdk", "masked_delta")
# The KPI keys differenced against the baseline in "masked_delta" (normalised SDK units; per cell except the two
# network ratios). power and tilt are deliberately absent: the baseline's configuration never enters.
DELTA_KEYS = ("ue_count", "rsrp_mean", "rsrp_p10", "sinr_mean", "sinr_p10", "coverage_ratio", "sla_ratio")
SHARE_KEYS = ("coverage_share", "sla_share")   # per-cell counters read from the raw state (twin.kpis per_cell)
CONFIG_KEYS = ("power", "tilt")
DELTA_CLIP = 2.0
# The no-UE rule (module docstring): UE-defined per-cell KPIs, normalised key -> raw key, and the twin's floor for
# each (ran_rl/twin/maveric/twin.py _FLOOR_DBM / _FLOOR_DB, the values it reports for a cell without UEs).
UE_DEFINED = {"rsrp_mean": "rsrp_mean_dbm", "rsrp_p10": "rsrp_p10_dbm", "sinr_mean": "sinr_mean_db",
              "sinr_p10": "sinr_p10_db"}
FLOORS = {"rsrp_mean_dbm": -140.0, "rsrp_p10_dbm": -140.0, "sinr_mean_db": -30.0, "sinr_p10_db": -30.0}
NETWORK_RATIOS = ("coverage_ratio", "sla_ratio")
KPI_DELTA_DIM = 30     # (ue share + 4 UE-defined) x 4 cells + 2 network ratios + 2 shares x 4 cells
CELL_SHARE_DIM = 8     # 2 shares x 4 cells
SERVES_UES_DIM = 8     # 4 cells now + 4 cells at the baseline


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
        if self.obs_mode not in OBS_MODES:
            raise ValueError(f"unknown observation mode {self.obs_mode!r} in the policy's meta")
        self.needs_baseline = self.obs_mode == "masked_delta"
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

    def observation(self, state: Mapping[str, Any], baseline: Mapping[str, Any] | None = None) -> dict[str, np.ndarray]:
        """The observation the model sees for ``state`` (training code uses this same function)."""
        if self.needs_baseline and baseline is None:
            raise ValueError(f"{self.model_id}@{self.version} reads KPI changes against the pre-incident baseline "
                             "(obs 'masked_delta'): pass baseline= (the network's state before the incident)")
        return observation_for(state, self.obs_mode, self._cfg, baseline=baseline if self.needs_baseline else None)

    def _normalised_action(self, state: Mapping[str, Any], baseline: Mapping[str, Any] | None = None) -> np.ndarray:
        action, _ = self._model.predict(self.observation(state, baseline), deterministic=True)
        return np.asarray(action, dtype=np.float64)

    def _filtered(self, state: Mapping[str, Any],
                  baseline: Mapping[str, Any] | None = None) -> tuple[np.ndarray, np.ndarray]:
        from ran_rl.env.observation import scale_action

        s = self._full_state(state)
        n = len(s["power_dbm"])
        raw = quantise_toward_zero(scale_action(self._normalised_action(s, baseline)[: 2 * n], self._bounds))
        dz_power, dz_tilt = self.deadzone
        if dz_power or dz_tilt:
            raw[:n] = np.where(np.abs(raw[:n]) < dz_power, 0.0, raw[:n])
            raw[n:] = np.where(np.abs(raw[n:]) < dz_tilt, 0.0, raw[n:])
        res = self._filter.apply(raw, s)
        f = quantise_toward_zero(res.filtered)
        return f[:n], f[n:]

    # ---- public API ----------------------------------------------------------------------
    def propose(self, state: Mapping[str, Any], baseline: Mapping[str, Any] | None = None) -> list[dict[str, float]]:
        dp, dt = self._filtered(state, baseline)
        return [{"cell": i, "d_power_db": float(dp[i]), "d_tilt_deg": float(dt[i])}
                for i in range(len(dp)) if dp[i] != 0 or dt[i] != 0]

    def propose_text(self, state: Mapping[str, Any], baseline: Mapping[str, Any] | None = None) -> str:
        return canon_change(self.propose(state, baseline))

    def action(self, state: Mapping[str, Any], baseline: Mapping[str, Any] | None = None) -> np.ndarray:
        """The proposal as a normalised env action in [-1, 1] (for running the policy in the SDK env)."""
        dp, dt = self._filtered(state, baseline)
        b = self._bounds
        pw = np.where(dp >= 0, dp / b.power_delta_max_db, dp / -b.power_delta_min_db)
        return np.concatenate([pw, dt / b.tilt_delta_max_deg])


def _normalised(state: Mapping[str, Any], cfg: Any) -> dict[str, np.ndarray]:
    from ran_rl.env.config import EnvConfig
    from ran_rl.env.observation import normalize_observation

    s = Policy._full_state(state)
    n = len(s["power_dbm"])
    return normalize_observation(s, np.zeros(2 * n), cfg if cfg is not None else EnvConfig())


def _num(v: Any) -> float:
    """A finite float, or NaN for anything else (None, NaN, +-inf, a bool, a non-number)."""
    if isinstance(v, (bool, np.bool_)) or v is None:
        return math.nan
    try:
        f = float(v)
    except (TypeError, ValueError):
        return math.nan
    return f if math.isfinite(f) else math.nan


def _vec(state: Mapping[str, Any], key: str) -> np.ndarray:
    return np.asarray([_num(v) for v in state[key]], dtype=np.float64)


def _scalar(state: Mapping[str, Any], key: str) -> float:
    v = state[key]
    return _num(np.ravel(v)[0] if isinstance(v, (list, tuple, np.ndarray)) else v)


def serves_ues(state: Mapping[str, Any]) -> np.ndarray:
    """Per cell: does it serve UEs (ue_count finite and > 0)? The no-UE rule's indicator."""
    s = Policy._full_state(state)
    ue = _vec(s, "ue_count")
    return np.where(np.isnan(ue), False, ue > 0)


def cell_shares(state: Mapping[str, Any]) -> np.ndarray:
    """The per-cell coverage and sla shares (SHARE_KEYS, cells in order) as one vector, NaN where undefined (no UEs,
    None or non-finite). Raises when the state does not carry the share lists at all."""
    missing = [k for k in SHARE_KEYS if k not in state]
    if missing:
        raise ValueError(f"observation mode 'masked_delta' needs the per-cell counters {missing} in the state "
                         "(t1/rl/remediation/v2.py with_shares: ReplayTwin.kpis()['per_cell'])")
    serve = np.tile(serves_ues(state), len(SHARE_KEYS))
    vals = np.concatenate([_vec(state, k) for k in SHARE_KEYS])
    return np.where(serve, vals, math.nan)


def sanitized(state: Mapping[str, Any]) -> dict[str, Any]:
    """The state the SDK normalisation sees in masked_delta: non-finite per-cell KPIs -> the twin's floors,
    non-finite ue_count -> 0, non-finite network ratios -> 0 (finite twin states are unchanged)."""
    s = Policy._full_state(state)
    for raw, floor in FLOORS.items():
        s[raw] = [floor if math.isnan(x) else x for x in _vec(s, raw)]
    s["ue_count"] = [0.0 if math.isnan(x) else x for x in _vec(s, "ue_count")]
    for k in NETWORK_RATIOS:
        x = _scalar(s, k)
        s[k] = 0.0 if math.isnan(x) else x
    return s


def kpi_delta(now: Mapping[str, np.ndarray], base: Mapping[str, np.ndarray],
              now_state: Mapping[str, Any], base_state: Mapping[str, Any]) -> np.ndarray:
    """``now - base`` for DELTA_KEYS (masked, normalised units) and the per-cell shares (SHARE_KEYS), under the no-UE
    rule (module docstring), concatenated and clipped to +-DELTA_CLIP: always KPI_DELTA_DIM long. The configuration
    (CONFIG_KEYS, power_dbm, tilt_deg) is never read from the baseline."""
    both = serves_ues(now_state) & serves_ues(base_state)
    parts = []
    for k in DELTA_KEYS:
        d = np.asarray(now[k], dtype=np.float64).ravel() - np.asarray(base[k], dtype=np.float64).ravel()
        if k in UE_DEFINED:
            raw = UE_DEFINED[k]
            ok = both & ~np.isnan(_vec(now_state, raw)) & ~np.isnan(_vec(base_state, raw))
            d = np.where(ok, d, 0.0)
        elif k in NETWORK_RATIOS:
            ok = not (math.isnan(_scalar(now_state, k)) or math.isnan(_scalar(base_state, k)))
            d = d if ok else np.zeros_like(d)
        parts.append(d)
    sd = cell_shares(now_state) - cell_shares(base_state)    # NaN wherever either side is undefined
    parts.append(np.where(np.isnan(sd), 0.0, sd))
    out = np.clip(np.concatenate(parts), -DELTA_CLIP, DELTA_CLIP).astype(np.float32)
    if out.shape != (KPI_DELTA_DIM,):
        raise ValueError(f"kpi_delta has shape {out.shape}, expected ({KPI_DELTA_DIM},): a 4-cell state is required")
    return out


def _kpi_only(state: Mapping[str, Any]) -> dict[str, Any]:
    """A baseline with its configuration neutralised: the SDK normalisation needs power and tilt fields, so they
    are set to constants; nothing derived from them is used (kpi_delta skips CONFIG_KEYS)."""
    s = dict(state)
    n = len(s["rsrp_mean_dbm"])
    s["power_dbm"] = [0.0] * n
    s["tilt_deg"] = [0.0] * n
    return s


def observation_for(state: Mapping[str, Any], mode: str = "masked", cfg: Any = None,
                    baseline: Mapping[str, Any] | None = None) -> dict[str, np.ndarray]:
    """The model observation of a raw twin state: the SDK's normalisation with prev_action = 0 (a stateless
    tool call), then the masked view when ``mode`` is "masked" or "masked_delta"; "masked_delta" adds
    ``kpi_delta`` against ``baseline`` (required there, ignored elsewhere). Shared by inference and training."""
    if mode == "masked_delta":
        if baseline is None:
            raise ValueError("observation mode 'masked_delta' needs the pre-incident baseline state")
        obs = mask_obs(_normalised(sanitized(state), cfg))
        base = mask_obs(_normalised(sanitized(_kpi_only(baseline)), cfg))
        obs["kpi_delta"] = kpi_delta(obs, base, state, baseline)
        shares = cell_shares(state)
        obs["cell_share"] = np.clip(2.0 * np.where(np.isnan(shares), 0.0, shares) - 1.0, -1.0, 1.0).astype(np.float32)
        obs["serves_ues"] = np.concatenate([serves_ues(state), serves_ues(baseline)]).astype(np.float32)
        return obs
    obs = _normalised(state, cfg)
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
