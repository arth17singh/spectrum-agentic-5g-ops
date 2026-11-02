"""STAND-IN predictors for the trust demo. Not W9's ``t2.predict`` models.

Same interface as ``t2.predict``: ``.predict(state, change) -> {coverage_ratio, sla_ratio,
sinr_p10_db, sinr_mean_db}``, ``.model_id``, ``.version``; numpy only.

``LinearDeltaPredictor`` predicts post-change KPIs as  before + drift + sign * W @ x(state, change)
where x holds, per cell, the power delta (dB), the tilt delta (deg), each delta times the cell's
current offset from the reference setting (23 dBm, 6 deg), each delta squared, and each delta
times the cell's UE share (linear in these features). ``fit_linear`` fits W by ridge least
squares on a few hundred random steps of the twin, against the change's counterfactual effect
(KPIs with the change minus KPIs with no change, from the same forked twin state), so UE
mobility does not leak into the fit; ``drift`` is the mean no-change one-step KPI drift.

The "healthy" model is the fit (sign +1); the "degraded" model is a sign-flipped copy
(sign -1): it predicts every effect backwards, the failure the gate must catch.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Callable

import numpy as np

OUT_KEYS = ("coverage_ratio", "sla_ratio", "sinr_p10_db", "sinr_mean_db")
PER_CELL = 8


def features(state: dict, change: list[dict], n_cells: int) -> np.ndarray:
    dp, dt = np.zeros(n_cells), np.zeros(n_cells)
    for c in change:
        i = int(c["cell"])
        dp[i] += float(c.get("d_power_db", 0.0))
        dt[i] += float(c.get("d_tilt_deg", 0.0))
    p = (np.asarray(state["power_dbm"], dtype=float) - 23.0) / 10.0
    t = (np.asarray(state["tilt_deg"], dtype=float) - 6.0) / 3.0
    ue = np.asarray(state["ue_count"], dtype=float)
    share = ue / max(1.0, ue.sum())
    cols = [dp, dt, dp * p, dt * t, dp * dp / 4.0, dt * dt / 4.0, dp * share, dt * share]
    return np.concatenate(cols)


class LinearDeltaPredictor:
    def __init__(self, W: np.ndarray, n_cells: int, kpis: Callable[[dict], dict], *, model_id: str,
                 version: str, sign: float = 1.0, drift=None):
        self.W = np.asarray(W, dtype=float)  # shape (len(OUT_KEYS), PER_CELL * n_cells)
        self.drift = np.zeros(len(OUT_KEYS)) if drift is None else np.asarray(drift, dtype=float)
        self.n_cells, self.kpis = n_cells, kpis
        self.model_id, self.version, self.sign = model_id, version, float(sign)

    def predict(self, state: dict, change: list[dict]) -> dict:
        before = self.kpis(state)
        delta = self.drift + self.sign * (self.W @ features(state, change, self.n_cells))
        out = {k: float(before[k] + d) for k, d in zip(OUT_KEYS, delta)}
        for k in ("coverage_ratio", "sla_ratio"):
            out[k] = min(1.0, max(0.0, out[k]))
        return out

    def predict_delta(self, state: dict, change: list[dict]) -> dict:
        """KPI change against the no-change counterfactual at the same step (A5)."""
        delta = self.sign * (self.W @ features(state, change, self.n_cells))
        return {k: float(d) for k, d in zip(OUT_KEYS, delta)}

    def flipped(self, version: str) -> "LinearDeltaPredictor":
        return LinearDeltaPredictor(self.W, self.n_cells, self.kpis, model_id=self.model_id,
                                    version=version, sign=-self.sign, drift=self.drift)

    def save(self, path: str | Path) -> None:
        np.savez(path, W=self.W, drift=self.drift, n_cells=self.n_cells, sign=self.sign,
                 model_id=self.model_id, version=self.version)


def random_change(rng: np.random.Generator, state: dict, n_cells: int) -> list[dict]:
    """1-2 cells, power +-1..4 dB, tilt +-0.5..3 deg, kept inside comfortable bounds."""
    cells = rng.choice(n_cells, size=int(rng.integers(1, 3)), replace=False)
    out = []
    for i in sorted(int(c) for c in cells):
        dp = float(rng.choice([-1, 1]) * rng.uniform(1, 4))
        dt = float(rng.choice([-1, 1]) * rng.uniform(0.5, 3))
        p, t = state["power_dbm"][i], state["tilt_deg"][i]
        if not 15 <= p + dp <= 31:
            dp = -dp
        if not 4 <= t + dt <= 12:
            dt = -dt
        out.append({"cell": i, "d_power_db": round(dp, 1), "d_tilt_deg": round(dt, 1)})
    return out


def fit_linear(twin_factory: Callable, kpis: Callable[[dict], dict], *, n_steps: int = 300,
               episode_len: int = 12, seed: int = 7, config="official", scenario: str = "nominal",
               ridge: float = 1e-2, model_id: str = "standin-linear", version: str = "v1"):
    """Fit W on random changes against counterfactual effects; returns (predictor, fit_info).

    The no-change branch is a deep copy of the stand-in twin taken before the change, so it
    shares the RNG state and UE positions (a fork, cheaper than replaying the log).
    """
    rng = np.random.default_rng(seed)
    X, Y, drift = [], [], []
    ep = 0
    n = None
    while len(X) < n_steps:
        tw = twin_factory(scenario, 10_000 + seed * 100 + ep, config)
        s = tw.reset()
        n = tw.n_cells
        for _ in range(episode_len):
            c = random_change(rng, s, n)
            ref = copy.deepcopy(tw)
            k0 = kpis(ref.apply([]))
            after = tw.apply(c)
            kb, ka = kpis(s), kpis(after)
            X.append(features(s, c, n))
            Y.append([ka[k] - k0[k] for k in OUT_KEYS])
            drift.append([k0[k] - kb[k] for k in OUT_KEYS])
            s = after
            if len(X) >= n_steps:
                break
        ep += 1
    X, Y = np.asarray(X), np.asarray(Y)
    A = X.T @ X + ridge * np.eye(X.shape[1])
    W = np.linalg.solve(A, X.T @ Y).T
    resid = Y - X @ W.T
    r2 = 1 - resid.var(axis=0) / Y.var(axis=0)
    info = {"n_steps": int(len(X)), "episodes": ep, "episode_len": episode_len, "seed": seed,
            "config": config if isinstance(config, str) else dict(config), "scenario": scenario,
            "target": "counterfactual effect (with change - no change, same forked state)",
            "r2_in_sample": {k: round(float(v), 4) for k, v in zip(OUT_KEYS, r2)},
            "resid_std": {k: round(float(v), 4) for k, v in zip(OUT_KEYS, resid.std(axis=0))},
            "target_std": {k: round(float(v), 4) for k, v in zip(OUT_KEYS, Y.std(axis=0))},
            "mean_drift": {k: round(float(v), 4) for k, v in zip(OUT_KEYS, np.mean(drift, axis=0))}}
    pred = LinearDeltaPredictor(W, n, kpis, model_id=model_id, version=version,
                                drift=np.mean(drift, axis=0))
    return pred, info
