"""Trust-gate thresholds v2: chosen on CALIBRATION data only, declared before any held-out run.

Why v2 (diagnosis on v1's calibration data, seeds 20000-20011, never on demo seeds):
- v1 chose tau as the smallest value that kept 12 calibration episodes within the false-pause
  budget: an in-sample edge. At v1's tau the healthy model exceeds tau on 9.7 % of changes, so a
  2-of-3 trigger somewhere in a 24-change run is likely out of sample.
- The terms that push healthy e over tau are sla_ratio and sinr_p10_db (mean terms 1.05 and 0.96 vs
  0.48 for coverage). The label-derived scales are tight relative to the model's real residuals for
  those two KPIs (healthy residual std / scale: coverage 0.35, sla 0.59, p10 SINR 0.51).
- Two-cell changes have 3.5x the exceedance rate of one-cell changes; no drift effect by step.
- The degraded model (tilt_reversed) is separated mostly by coverage (err/scale p90 2.52).

The A6 rule is unchanged (n >= 3; 2 of the last 3 with e > tau, or EWMA trust < pause line). v2
changes only scales, tau, alpha and the pause line, by this declared procedure:

1. Data: ``calibrate.collect`` on twin.replay (official, nominal), change seed 99, twin seeds
   20000-20039 (40 episodes x 24 changes; the first 12 episodes are v1's), healthy t2.predict and
   tilt_reversed. Seeds 2000-2019 (the v2 evaluation) are never touched here.
2. Two candidates:
   C1 "label scales"    : v1's scales (std of the t2 training labels).
   C2 "residual scales" : per KPI, the 90th percentile of the healthy model's |pred - meas| on
                          these calibration changes (e ~ 1 at the healthy model's p90 on every KPI).
3. For each candidate, block bootstrap of 24-change runs (4 blocks of 6 consecutive changes from
   random calibration episodes, 4000 runs, seed 7), the same resampled runs for both models.
   Grid alpha in {0.1, 0.15, 0.2, 0.3}, pause line in {0.05, ..., 0.70}, tau in {0.50, ..., 5.00}.
   Every grid point gets the estimated healthy false-pause probability per run (pf) and the
   estimated probability that the degraded model is paused by change 6 (pd). Score = estimated
   probability of passing the declared success criterion on 20 independent runs,
   P(Binom(20, pf) <= 2) * P(Binom(20, pd) >= 19). Pick the max; ties: lower pf, then larger tau.
4. Pick the candidate with the higher score (ties: lower pf; then C1).
5. Write thresholds_v2.json (v1's fixed fields + chosen scales/alpha/tau/pause line) and
   calibration_v2.json (data, both candidates' estimates, the choice).

Success criterion, declared with v2 and evaluated once on fresh twin seeds 2000-2019 with the
24-change demo script: healthy false pauses <= 2 of 20 AND degraded paused in >= 19 of 20 runs by
change 6.

    python -m t3.trust.calibrate_v2            # writes t3/trust/thresholds_v2.json, calibration_v2.json
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from .calibrate import ARTIFACT, collect
from .gate import THRESHOLDS_PATH, TrustGate, load_thresholds

HERE = Path(__file__).parent
THRESHOLDS_V2 = HERE / "thresholds_v2.json"
CALIBRATION_V2 = HERE / "calibration_v2.json"

RUN_LEN, BLOCK, N_RUNS, BOOT_SEED = 24, 6, 4000, 7
DETECT_BY = 6
SUCCESS = {"eval_twin_seeds": [2000, 2019], "script": "t3/trust/scripted_changes.json (24 changes)",
           "healthy_false_pauses_max": 2, "degraded_paused_by_change_6_min": 19, "runs": 20}


def score_arrays(gate: TrustGate, episodes, mk):
    e = np.array([[gate.score(r["predicted_delta"][mk], r["measured_delta"])["e"] for r in ep] for ep in episodes])
    s = np.array([[gate.score(r["predicted_delta"][mk], r["measured_delta"])["score"] for r in ep] for ep in episodes])
    return e, s


def bootstrap_index(n_eps, ep_len, rng):
    """(N_RUNS, RUN_LEN) pairs of (episode, position) built from blocks of consecutive changes."""
    nb = RUN_LEN // BLOCK
    eps = rng.integers(0, n_eps, size=(N_RUNS, nb))
    starts = rng.integers(0, ep_len - BLOCK + 1, size=(N_RUNS, nb))
    pos = (starts[:, :, None] + np.arange(BLOCK)[None, None, :]).reshape(N_RUNS, RUN_LEN)
    return np.repeat(eps, BLOCK, axis=1), pos


def first_pause(e, s, alpha, pause_below, tau, warmup, window, needed):
    """Vectorised A6 rule over runs: 1-based index of the first pause, or 0 for none."""
    n_runs, L = e.shape
    m = np.zeros(n_runs)
    out = np.zeros(n_runs, dtype=int)
    over = e > tau
    for t in range(L):
        m = (1 - alpha) * m + alpha * s[:, t]
        trust = m / (1 - (1 - alpha) ** (t + 1))
        if t + 1 < warmup:
            continue
        cnt = over[:, max(0, t - window + 1): t + 1].sum(axis=1)
        hit = ((cnt >= needed) | (trust < pause_below)) & (out == 0)
        out[hit] = t + 1
    return out


def binom_cdf(k: int, n: int, p: float) -> float:
    return float(sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1)))


def p_pass(hfalse: float, det: float) -> float:
    """Estimated probability that 20 independent runs meet the declared success criterion."""
    n = SUCCESS["runs"]
    ok_h = binom_cdf(SUCCESS["healthy_false_pauses_max"], n, hfalse)
    ok_d = 1.0 - binom_cdf(SUCCESS["degraded_paused_by_change_6_min"] - 1, n, det)
    return ok_h * ok_d


def choose(eh, sh, ed, sd, th):
    """Maximise the estimated P(pass) of the declared success criterion; ties: lower healthy false
    pause, then larger tau. Every grid point is evaluated (the objective is not monotone in tau)."""
    w, needed, warm = int(th["recent_window"]), int(th["recent_needed"]), int(th["warmup"])
    best, best_key, n = None, None, 0
    for alpha in (0.1, 0.15, 0.2, 0.3):
        for pb_i in range(1, 15):
            pb = round(pb_i * 0.05, 2)
            for t_i in range(10, 101):
                tau = round(t_i * 0.05, 2)
                hp = first_pause(eh, sh, alpha, pb, tau, warm, w, needed)
                hfalse = float((hp > 0).mean())
                if hfalse > 0.2:
                    continue  # cannot pass; skip the degraded replay
                dp = first_pause(ed, sd, alpha, pb, tau, warm, w, needed)
                det = float(((dp > 0) & (dp <= DETECT_BY)).mean())
                pp = p_pass(hfalse, det)
                n += 1
                key = (round(pp, 6), -hfalse, tau)
                if best_key is None or key > best_key:
                    best_key, best = key, {"alpha": alpha, "pause_below": pb, "tau": tau, "est_p_pass": pp,
                                           "est_healthy_false_pause": hfalse, "est_degraded_paused_by_6": det,
                                           "est_degraded_paused_any": float((dp > 0).mean())}
    return best, n


def main() -> int:
    from t2.predict import load, load_degraded
    from twin.kpis import headline
    from twin.replay import ReplayTwin

    from .demo import code_sha
    from .scripted import random_change

    t0 = time.time()
    th1, sha1 = load_thresholds(THRESHOLDS_PATH)
    healthy, degraded = load(ARTIFACT), load_degraded(ARTIFACT)
    hk, dk = healthy.model_id + "@" + healthy.version, degraded.model_id + "@" + degraded.version
    episodes, clipped, total = collect(ReplayTwin, headline, [healthy, degraded], n_changes=960, episode_len=RUN_LEN,
                                       seed=99, config="official", scenario="nominal", change_fn=random_change)
    keys = th1["keys"]
    flat = [r for ep in episodes for r in ep]
    resid_p90 = {k: round(float(np.percentile([abs(r["predicted_delta"][hk][k] - r["measured_delta"][k]) for r in flat], 90)), 4)
                 for k in keys}
    candidates = {"C1-label-scales": dict(th1["scales"]), "C2-residual-scales": resid_p90}
    ep_len = min(len(ep) for ep in episodes)
    episodes_eq = [ep[:ep_len] for ep in episodes]
    rng = np.random.default_rng(BOOT_SEED)
    ei, pi = bootstrap_index(len(episodes_eq), ep_len, rng)
    results = {}
    for name, scales in candidates.items():
        th = dict(th1, scales=scales)
        gate = TrustGate(store=None, thresholds=th)
        eh, sh = score_arrays(gate, episodes_eq, hk)
        ed, sd = score_arrays(gate, episodes_eq, dk)
        best, n_ok = choose(eh[ei, pi], sh[ei, pi], ed[ei, pi], sd[ei, pi], th)
        results[name] = {"scales": scales, "best": best, "settings_meeting_target": n_ok}
        print(name, scales, best)
    # sanity check of the estimator: v1 as declared, under the same bootstrap (calibration data only)
    g1 = TrustGate(store=None, thresholds=th1)
    eh, sh = score_arrays(g1, episodes_eq, hk)
    ed, sd = score_arrays(g1, episodes_eq, dk)
    args1 = (th1["alpha"], th1["pause_below"], th1["tau"], th1["warmup"], th1["recent_window"], th1["recent_needed"])
    hp1, dp1 = first_pause(eh[ei, pi], sh[ei, pi], *args1), first_pause(ed[ei, pi], sd[ei, pi], *args1)
    v1_est = {"est_healthy_false_pause": float((hp1 > 0).mean()),
              "est_degraded_paused_by_6": float(((dp1 > 0) & (dp1 <= DETECT_BY)).mean())}
    v1_est["est_p_pass"] = p_pass(v1_est["est_healthy_false_pause"], v1_est["est_degraded_paused_by_6"])
    print("v1 under the same bootstrap:", v1_est)
    order = sorted(results, key=lambda n: (results[n]["best"]["est_p_pass"] if results[n]["best"] else -1,
                                           -(results[n]["best"]["est_healthy_false_pause"] if results[n]["best"] else 1),
                                           n == "C1-label-scales"), reverse=True)
    pick = order[0]
    b = results[pick]["best"]
    th2 = dict(th1)
    th2.update({"schema": "t3.trust.thresholds/2", "version": "v2", "scales": results[pick]["scales"],
                "alpha": b["alpha"], "tau": b["tau"], "pause_below": b["pause_below"],
                "watch_below": round(b["pause_below"] + 0.15, 2)})
    th2.pop("calibration_false_pause_budget", None)
    th2["notes"] = {
        "rule": th1["notes"]["rule"],
        "procedure": "t3.trust.calibrate_v2 (module docstring): calibration seeds 20000-20039, block bootstrap, "
                     "max estimated P(pass) of the declared success criterion over 20 runs",
        "candidate": pick,
        "success_criterion": SUCCESS,
        "forecast": {k: round(v, 4) for k, v in b.items() if k.startswith("est_")},
        "supersedes": f"nothing yet: v1 (thresholds.json sha256 {sha1}) stays the default until the lead approves a switch",
    }
    THRESHOLDS_V2.write_text(json.dumps(th2, indent=2) + "\n")
    th2_loaded, sha2 = load_thresholds(THRESHOLDS_V2)
    rep = {"what": "trust-gate thresholds v2 calibration (calibration data only)", "code_sha": code_sha(),
           "healthy": hk, "degraded": dk, "twin_seeds": f"20000..{20000 + len(episodes) - 1}", "change_seed": 99,
           "changes_total": total, "changes_clamped_excluded": clipped, "episode_len_used": ep_len,
           "bootstrap": {"runs": N_RUNS, "run_len": RUN_LEN, "block": BLOCK, "seed": BOOT_SEED},
           "selection": "max estimated P(pass)", "detect_by": DETECT_BY,
           "v1_thresholds_sha256": sha1, "v1_under_same_bootstrap": v1_est, "candidates": results, "picked": pick,
           "thresholds_v2_sha256": sha2, "thresholds_v2": th2_loaded, "success_criterion": SUCCESS,
           "episodes": episodes, "elapsed_s": round(time.time() - t0, 1)}
    CALIBRATION_V2.write_text(json.dumps(rep, indent=1) + "\n")
    print(f"picked {pick}: {b}\nthresholds_v2.json sha256 {sha2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
