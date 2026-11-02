"""Offline calibration of the trust gate (amendment 1, A6): at least 200 replayed changes each
for a healthy and a degraded predictor, recorded.

Procedure (stand-in twin and predictors until W2/W9 merge; both are labelled in the output):
1. Fit the healthy stand-in predictor on fit seeds (10000+), official Maveric preset.
2. Replay random changes on held-out twin seeds (20000+), episodes of ``episode_len`` steps,
   until ``n_changes`` unclipped changes are collected. Each change gets its exact counterfactual
   delta (replay of the same step with no change). Clipped changes are counted and excluded.
3. Score each change for the healthy model and its sign-flipped copy with the declared scales:
   e, direction agreement, per-change score s.
4. Report e quantiles for both, the per-change rate e > tau and, replaying each episode through
   the full declared rule (warm-up, 2-of-3 over tau, EWMA line), the false-pause rate of the
   healthy model and the changes-to-pause of the degraded model.

The run also prints the proposal of PROPOSAL_RULE (below). thresholds.json is edited by hand
from it and committed BEFORE any demo run (declared first, then run); a second calibration run
with the declared values is the recorded one (t3/trust/calibration.json).

    python -m t3.trust.calibrate --out t3/trust/calibration.json [--n-changes 288 --episode-len 24]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from .gate import TrustGate, load_thresholds
from .stepping import predicted_delta, step_with_counterfactual


def collect(twin_factory, kpis, predictors, *, n_changes, episode_len, seed, config, scenario, change_fn):
    rng = np.random.default_rng(seed)
    episodes, clipped, total = [], 0, 0
    ep = 0
    while sum(len(e) for e in episodes) < n_changes:
        tw_seed = 20_000 + ep
        tw = twin_factory(scenario, tw_seed, config)
        state = tw.reset()
        rows = []
        for _ in range(episode_len):
            change = change_fn(rng, state, tw.n_cells if hasattr(tw, "n_cells") else 4)
            preds = {p.version: predicted_delta(p, state, change) for p in predictors}
            st = step_with_counterfactual(tw, twin_factory, kpis, change, scenario=scenario,
                                          seed=tw_seed, config=config, before_state=state)
            total += 1
            state = st["after"]
            if st["clipped"]:
                clipped += 1
                continue
            meas = {k: st["measured_kpis"][k] - st["baseline_kpis"][k] for k in st["measured_kpis"]
                    if k in ("coverage_ratio", "sla_ratio", "sinr_p10_db", "sinr_mean_db")}
            rows.append({"twin_seed": tw_seed, "change": change, "measured_delta": meas, "predicted_delta": preds})
        episodes.append(rows)
        ep += 1
    return episodes, clipped, total


def replay_rule(gate: TrustGate, scored: list[dict]) -> int | None:
    """Run the declared pause rule over a sequence of per-change scores; 1-based pause index or None."""
    th = gate.thresholds
    m, n, recent = 0.0, 0, []
    for i, sc in enumerate(scored, start=1):
        m, n, trust = gate.ewma_step(m, n, sc["score"], float(th["alpha"]))
        recent = (recent + [sc["e"]])[-int(th["recent_window"]):]
        if gate.pause_trigger(n, trust, recent):
            return i
    return None


PROPOSAL_RULE = ("pause_below = (lowest post-warm-up EWMA trust of the healthy model over all calibration "
                 "episodes) - 0.05, floored to 0.05; then tau = the smallest value on a 0.05 grid in "
                 "[0.5, 5] at which the full rule pauses the healthy model in none of the calibration "
                 "episodes")


def _with(gate: TrustGate, **over) -> TrustGate:
    th = dict(gate.thresholds)
    th.update(over)
    return TrustGate(store=None, thresholds=th)


def propose(gate: TrustGate, healthy_seqs, degraded_seqs) -> dict:
    th = gate.thresholds
    warm = int(th["warmup"])
    lows = []
    for seq in healthy_seqs:
        m, n = 0.0, 0
        for i, sc in enumerate(seq, start=1):
            m, n, t = gate.ewma_step(m, n, sc["score"], float(th["alpha"]))
            if i >= warm:
                lows.append(t)
    pause_below = math.floor((min(lows) - 0.05) / 0.05) * 0.05
    pause_below = round(max(0.05, pause_below), 2)
    tau = None
    for k in range(10, 101):
        cand = round(k * 0.05, 2)
        g = _with(gate, tau=cand, pause_below=pause_below, watch_below=max(pause_below, float(th["watch_below"])))
        if all(replay_rule(g, seq) is None for seq in healthy_seqs):
            tau = cand
            break
    out = {"pause_below": pause_below, "tau": tau, "healthy_lowest_trust_after_warmup": min(lows)}
    if tau is not None:
        g = _with(gate, tau=tau, pause_below=pause_below, watch_below=max(pause_below, float(th["watch_below"])))
        out["degraded_paused_at_with_proposal"] = [replay_rule(g, seq) for seq in degraded_seqs]
    return out


def calibrate(*, twin_factory, kpis, healthy, degraded, thresholds_path=None, n_changes=240,
              episode_len=12, seed=99, config="official", scenario="nominal", change_fn=None, fit_info=None):
    from .standin_predictors import random_change
    change_fn = change_fn or random_change
    th, th_sha = load_thresholds(thresholds_path) if thresholds_path else load_thresholds()
    gate = TrustGate(store=None, thresholds=th)
    t0 = time.time()
    episodes, clipped, total = collect(twin_factory, kpis, [healthy, degraded], n_changes=n_changes,
                                       episode_len=episode_len, seed=seed, config=config,
                                       scenario=scenario, change_fn=change_fn)
    per_model = {}
    flat_meas = np.array([[r["measured_delta"][k] for k in th["keys"]] for ep in episodes for r in ep])
    for p in (healthy, degraded):
        seqs = [[gate.score(r["predicted_delta"][p.version], r["measured_delta"]) for r in ep] for ep in episodes]
        es = np.array([s["e"] for seq in seqs for s in seq])
        ds = [s["direction_agree"] for seq in seqs for s in seq]
        ss = np.array([s["score"] for seq in seqs for s in seq])
        pauses = [replay_rule(gate, seq) for seq in seqs]
        per_model[p.version] = {
            "model_id": p.model_id, "version": p.version, "n_changes": int(es.size),
            "e_quantiles": {q: float(np.percentile(es, float(q))) for q in ("5", "10", "25", "50", "75", "90", "95", "97", "99")},
            "e_mean": float(es.mean()),
            "rate_e_over_tau": float((es > float(th["tau"])).mean()),
            "direction": {"agree": sum(d is True for d in ds), "disagree": sum(d is False for d in ds),
                          "unmoved": sum(d is None for d in ds)},
            "score_mean": float(ss.mean()),
            "episodes": len(seqs), "episode_len": episode_len,
            "episodes_paused": sum(x is not None for x in pauses),
            "paused_at": pauses,
            "per_change_e": [round(float(x), 6) for x in es],
        }
    proposal = propose(gate, [[gate.score(r["predicted_delta"][healthy.version], r["measured_delta"]) for r in ep] for ep in episodes],
                       [[gate.score(r["predicted_delta"][degraded.version], r["measured_delta"]) for r in ep] for ep in episodes])
    return {
        "what": "trust-gate calibration (amendment 1, A6)",
        "twin_impl": getattr(twin_factory, "__module__", "?") + "." + getattr(twin_factory, "__name__", "?"),
        "predictor_impl": type(healthy).__module__ + "." + type(healthy).__name__,
        "config": config, "scenario": scenario, "change_seed": seed, "twin_seeds": f"20000..{20000 + len(episodes) - 1}",
        "thresholds_sha256": th_sha, "thresholds": th,
        "changes_total": total, "changes_clipped_excluded": clipped,
        "measured_delta_std": {k: float(v) for k, v in zip(th["keys"], flat_meas.std(axis=0))},
        "measured_delta_abs_median": {k: float(v) for k, v in zip(th["keys"], np.median(np.abs(flat_meas), axis=0))},
        "proposal_rule": PROPOSAL_RULE,
        "proposal": proposal,
        "models": per_model, "fit": fit_info, "elapsed_s": round(time.time() - t0, 2),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-changes", type=int, default=288)
    ap.add_argument("--episode-len", type=int, default=24)
    ap.add_argument("--fit-steps", type=int, default=400)
    args = ap.parse_args(argv)
    from .standin_predictors import fit_linear
    from .standin_twin import StandinReplayTwin, kpis
    healthy, fit = fit_linear(StandinReplayTwin, kpis, n_steps=args.fit_steps, model_id="standin-linear", version="v1")
    degraded = healthy.flipped("v1-flipped")
    rep = calibrate(twin_factory=StandinReplayTwin, kpis=kpis, healthy=healthy, degraded=degraded,
                    n_changes=args.n_changes, episode_len=args.episode_len, fit_info=fit)
    Path(args.out).write_text(json.dumps(rep, indent=1) + "\n")
    th = rep["thresholds"]
    print(f"changes: {rep['changes_total']} replayed, {rep['changes_clipped_excluded']} clipped and excluded")
    print("measured delta std:", {k: round(v, 4) for k, v in rep["measured_delta_std"].items()})
    for v, m in rep["models"].items():
        q = m["e_quantiles"]
        print(f"{v:<12} n={m['n_changes']}  e p50 {q['50']:.3f} p90 {q['90']:.3f} p97 {q['97']:.3f}  "
              f"P(e>tau={th['tau']}) {m['rate_e_over_tau']:.3f}  dir {m['direction']}  "
              f"episodes paused {m['episodes_paused']}/{m['episodes']}  at {m['paused_at']}")
    print(f"proposal ({rep['proposal_rule']}):\n  {rep['proposal']}\ndeclared: tau {th['tau']}, pause_below {th['pause_below']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
