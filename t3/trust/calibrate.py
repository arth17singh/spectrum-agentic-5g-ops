"""Calibration of the trust gate (amendment 1, A6): at least 200 replayed changes each for the
healthy and the degraded Track 2 predictor, on the real twin, recorded.

Declared procedure (this file is the only way thresholds.json changes):

1. Scales: the standard deviation of each scored KPI delta label in the predictor's TRAINING rows
   (``t2/data/t2_dataset_v0.npz``, split == 0), so e = 1 means an error the size of a typical effect.
2. Replay: ``twin.replay.ReplayTwin``, official preset, scenario "nominal", twin seeds 20000+ (disjoint
   from the t2 dataset's 10000-10699 and the demo/ladder seeds), episodes of ``episode_len`` random
   changes (``scripted.random_change``, change seed 99) until ``n_changes`` unclamped changes exist.
   Each change: measured delta = ``counterfactual`` (exact), predicted delta = ``t2.predict``
   (healthy ``load``, degraded ``load_degraded`` = fault "tilt_reversed"). Clamped changes are
   counted and excluded (A5).
3. Proposal (PROPOSAL_RULE): alpha, pause_below and tau from a grid, with warm-up 3 and the 2-of-3
   window fixed by A6, dead bands and weights as declared.
4. ``--apply`` writes scales, alpha, tau, pause_below into thresholds.json, then re-scores the stored
   changes under the written file and records its sha256 in the output.

Everything per change (both predictions and the measurement) is stored, so the proposal can be
recomputed offline (``repropose``).

    python -m t3.trust.calibrate --out t3/trust/calibration.json --apply
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from .gate import THRESHOLDS_PATH, TrustGate, load_thresholds
from .stepping import predicted_delta, step_with_counterfactual

DATASET = "t2/data/t2_dataset_v0.npz"
ARTIFACT = "t2/artifacts/t2_response_v0.npz"
PROPOSAL_RULE = ("joint grid over alpha in {0.1, 0.15, 0.2, 0.3} and pause_below in {0.05, 0.10, ..., 0.70}; "
                 "for each pair take the smallest tau in {0.50, 0.55, ..., 5.00} at which the full rule pauses "
                 "the healthy model in at most `calibration_false_pause_budget` (declared) calibration episodes. "
                 "Among those, pick the most degraded episodes paused within max_changes_to_pause, then the "
                 "smallest mean changes-to-pause (not paused counts as length + 1), then the fewest healthy "
                 "pauses, then the largest healthy margin (lowest healthy post-warm-up trust minus pause_below)")


def label_scales(dataset: str | Path = DATASET, keys=("coverage_ratio", "sla_ratio", "sinr_p10_db")) -> dict:
    with np.load(dataset, allow_pickle=False) as z:
        names = [str(n) for n in z["label_names"]]
        y = z["y"][z["split"] == 0]
    return {k: round(float(np.asarray(y[:, names.index(k)], dtype=np.float64).std()), 4) for k in keys}


def collect(twin_factory, kpis, predictors, *, n_changes, episode_len, seed, config, scenario, change_fn,
            twin_seed0=20_000):
    rng = np.random.default_rng(seed)
    episodes, clipped, total, ep = [], 0, 0, 0
    while sum(len(e) for e in episodes) < n_changes:
        tw_seed = twin_seed0 + ep
        tw = twin_factory(scenario, tw_seed, config)
        state = tw.reset()
        rows = []
        for _ in range(episode_len):
            change = change_fn(rng, state)
            preds = {p.model_id + "@" + p.version: predicted_delta(p, state, change) for p in predictors}
            st = step_with_counterfactual(tw, twin_factory, kpis, change, scenario=scenario, seed=tw_seed,
                                          config=config, before_state=state)
            total += 1
            state = st["after"]
            if st["clipped"]:
                clipped += 1
                continue
            meas = {k: st["measured_kpis"][k] - st["baseline_kpis"][k] for k in st["baseline_kpis"]}
            rows.append({"twin_seed": tw_seed, "change": change, "measured_delta": meas, "predicted_delta": preds})
        episodes.append(rows)
        ep += 1
    return episodes, clipped, total


def replay_rule(gate: TrustGate, scored: list[dict]) -> int | None:
    """Run the declared pause rule over per-change scores; 1-based pause index or None."""
    th = gate.thresholds
    m, n, recent = 0.0, 0, []
    for i, sc in enumerate(scored, start=1):
        m, n, trust = gate.ewma_step(m, n, sc["score"], float(th["alpha"]))
        recent = (recent + [sc["e"]])[-int(th["recent_window"]):]
        if gate.pause_trigger(n, trust, recent):
            return i
    return None


def _with(gate: TrustGate, **over) -> TrustGate:
    th = dict(gate.thresholds)
    th.update(over)
    th["watch_below"] = max(float(th["pause_below"]), float(th.get("watch_below", th["pause_below"])))
    return TrustGate(store=None, thresholds=th)


def _lowest_trust(seqs, alpha, warm):
    lows = []
    for seq in seqs:
        m, n = 0.0, 0
        for i, sc in enumerate(seq, start=1):
            m, n, t = TrustGate.ewma_step(m, n, sc["score"], alpha)
            if i >= warm:
                lows.append(t)
    return min(lows)


def propose(gate: TrustGate, healthy_seqs, degraded_seqs, budget: int | None = None) -> dict:
    th = gate.thresholds
    max_n = int(th["max_changes_to_pause"])
    budget = int(th.get("calibration_false_pause_budget", 0) if budget is None else budget)
    best, best_key, n_ok = None, None, 0
    for alpha in (0.1, 0.15, 0.2, 0.3):
        low = _lowest_trust(healthy_seqs, alpha, int(th["warmup"]))
        for pb_i in range(1, 15):
            pb = round(pb_i * 0.05, 2)
            for t_i in range(10, 101):
                tau = round(t_i * 0.05, 2)
                g = _with(gate, alpha=alpha, pause_below=pb, tau=tau)
                h_at = [replay_rule(g, seq) for seq in healthy_seqs]
                h_paused = sum(x is not None for x in h_at)
                if h_paused > budget:
                    continue
                n_ok += 1
                at = [replay_rule(g, seq) for seq in degraded_seqs]
                within = sum(1 for x in at if x is not None and x <= max_n)
                mean_at = float(np.mean([x if x is not None else len(s) + 1 for x, s in zip(at, degraded_seqs)]))
                key = (within, -mean_at, -h_paused, low - pb)
                if best_key is None or key > best_key:
                    best_key = key
                    best = {"alpha": alpha, "pause_below": pb, "tau": tau,
                            "healthy_lowest_trust_after_warmup": low, "healthy_paused_episodes": h_paused,
                            "healthy_paused_at": h_at, "false_pause_budget": budget,
                            "degraded_paused_at": at, "degraded_paused_within_max": within,
                            "degraded_episodes": len(degraded_seqs), "max_changes_to_pause": max_n}
                break  # a larger tau only weakens detection at this (alpha, pause_below)
    return {"best": best, "settings_within_budget": n_ok}


def score_models(gate: TrustGate, episodes, model_keys) -> dict:
    out = {}
    for mk in model_keys:
        seqs = [[gate.score(r["predicted_delta"][mk], r["measured_delta"]) for r in ep] for ep in episodes]
        es = np.array([s["e"] for seq in seqs for s in seq])
        ds = [s["direction_agree"] for seq in seqs for s in seq]
        pauses = [replay_rule(gate, seq) for seq in seqs]
        out[mk] = {
            "n_changes": int(es.size),
            "e_quantiles": {q: float(np.percentile(es, float(q))) for q in ("5", "25", "50", "75", "90", "95", "97", "99")},
            "e_mean": float(es.mean()),
            "rate_e_over_tau": float((es > float(gate.thresholds["tau"])).mean()),
            "direction": {"agree": sum(d is True for d in ds), "disagree": sum(d is False for d in ds),
                          "no_verdict": sum(d is None for d in ds)},
            "score_mean": float(np.mean([s["score"] for seq in seqs for s in seq])),
            "episodes": len(seqs), "episodes_paused": sum(x is not None for x in pauses), "paused_at": pauses,
            "sequences": [[{"e": s["e"], "direction_agree": s["direction_agree"], "score": s["score"]} for s in seq]
                          for seq in seqs],
        }
    return out


def apply_proposal(best: dict, scales: dict, path: Path = THRESHOLDS_PATH) -> str:
    th = json.loads(path.read_text())
    th["scales"] = {k: scales[k] for k in th["keys"]}
    th["alpha"], th["tau"], th["pause_below"] = best["alpha"], best["tau"], best["pause_below"]
    th["watch_below"] = round(max(best["pause_below"] + 0.15, best["pause_below"]), 2)
    notes = th.setdefault("notes", {})
    notes["scales"] = f"std of the training-split delta labels of the Track 2 predictor ({DATASET}); e = 1 is an error the size of a typical effect"
    notes["alpha_tau_pause_below"] = ("written by t3.trust.calibrate --apply (PROPOSAL_RULE) from replayed changes on twin.replay "
                                      "(official, nominal, seeds 20000+) with t2.predict healthy and tilt_reversed; in-sample, "
                                      "so the held-out check is the demo seeds; see t3/trust/calibration.json")
    notes.pop("recalibrate", None)
    path.write_text(json.dumps(th, indent=2) + "\n")
    return load_thresholds(path)[1]


def repropose(path: str | Path, budget: int | None = None) -> dict:
    rep = json.loads(Path(path).read_text())
    gate = TrustGate(store=None, thresholds=rep["thresholds_used_for_proposal"])
    h, d = rep["healthy"], rep["degraded"]
    seqs = score_models(gate, rep["episodes"], [h, d])
    return propose(gate, seqs[h]["sequences"], seqs[d]["sequences"], budget)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-changes", type=int, default=288)
    ap.add_argument("--episode-len", type=int, default=24)
    ap.add_argument("--apply", action="store_true", help="write the proposal into thresholds.json")
    args = ap.parse_args(argv)

    from t2.predict import load, load_degraded
    from twin.kpis import headline
    from twin.replay import ReplayTwin

    from .demo import code_sha
    from .scripted import random_change

    t0 = time.time()
    healthy, degraded = load(ARTIFACT), load_degraded(ARTIFACT)
    hk, dk = healthy.model_id + "@" + healthy.version, degraded.model_id + "@" + degraded.version
    scales = label_scales()
    th0, sha0 = load_thresholds()
    th_prop = dict(th0, scales={k: scales[k] for k in th0["keys"]})
    episodes, clipped, total = collect(ReplayTwin, headline, [healthy, degraded], n_changes=args.n_changes,
                                       episode_len=args.episode_len, seed=99, config="official",
                                       scenario="nominal", change_fn=random_change)
    gate = TrustGate(store=None, thresholds=th_prop)
    pre = score_models(gate, episodes, [hk, dk])
    proposal = propose(gate, pre[hk]["sequences"], pre[dk]["sequences"])
    written_sha = None
    final_th, final_sha = th_prop, None
    if args.apply and proposal["best"]:
        written_sha = apply_proposal(proposal["best"], scales)
        final_th, final_sha = load_thresholds()
    final = score_models(TrustGate(store=None, thresholds=final_th), episodes, [hk, dk])
    flat = np.array([[r["measured_delta"][k] for k in th0["keys"]] for ep in episodes for r in ep])
    rep = {
        "what": "trust-gate calibration (amendment 1, A6), real twin and real Track 2 predictor",
        "twin_impl": "twin.replay.ReplayTwin (official preset, scenario nominal)",
        "predictor_impl": f"t2.predict ({ARTIFACT}); degraded = load_degraded (fault tilt_reversed)",
        "healthy": hk, "degraded": dk, "code_sha": code_sha(),
        "change_seed": 99, "twin_seeds": f"20000..{20000 + len(episodes) - 1}",
        "episode_len": args.episode_len, "changes_total": total, "changes_clamped_excluded": clipped,
        "scales_rule": f"std of the training-split delta labels in {DATASET}", "scales": scales,
        "thresholds_before_sha256": sha0, "thresholds_used_for_proposal": th_prop,
        "proposal_rule": PROPOSAL_RULE, "proposal": proposal,
        "applied": bool(args.apply and proposal["best"]),
        "thresholds_after_sha256": final_sha, "thresholds_after": final_th if final_sha else None,
        "measured_delta_std": {k: float(v) for k, v in zip(th0["keys"], flat.std(axis=0))},
        "models_under_final_thresholds": {k: {kk: vv for kk, vv in v.items() if kk != "sequences"} for k, v in final.items()},
        "episodes": episodes, "elapsed_s": round(time.time() - t0, 1),
    }
    Path(args.out).write_text(json.dumps(rep, indent=1) + "\n")
    print(f"changes: {total} replayed, {clipped} clamped and excluded; scales {scales}")
    for mk, m in final.items():
        q = m["e_quantiles"]
        print(f"{mk:<52} n={m['n_changes']} e p50 {q['50']:.3f} p90 {q['90']:.3f}  P(e>tau) {m['rate_e_over_tau']:.3f}  "
              f"dir {m['direction']}  paused {m['episodes_paused']}/{m['episodes']} at {m['paused_at']}")
    print(f"proposal: {proposal['best']}")
    print(f"applied: {rep['applied']}  thresholds.json sha256 {final_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
