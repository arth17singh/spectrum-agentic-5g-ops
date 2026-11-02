"""Scripted trust-gate demo. No agents: a fixed list of changes is applied to a twin, every
predictor's predicted KPI delta is compared with the measured delta, and the gate decides.

Interfaces (all injected, so W2/W9/W6 modules plug in at merge):

* ``twin_factory(scenario, seed, config)`` -> twin with ``.reset()``, ``.apply(change)`` and
  ``.action_log``; ``twin_factory.from_log(scenario, seed, config, action_log)`` (W2's
  ``twin.replay.ReplayTwin`` class itself fits).
* ``kpis(state)`` -> ``{coverage_ratio, sla_ratio, sinr_p10_db, sinr_mean_db, ...}`` (W2).
* predictors with ``.predict(state, change)`` (or ``.predict_delta``), ``.model_id``, ``.version``.
* ``writer.append_records(run_id, kind, records)`` (W6).
* ``policy_client`` with ``pause / resume / is_paused`` (``t3.trust.policy.PolicyClient``).

Deltas, not levels (A5): measured_delta = KPIs after the change - KPIs of the replay of the same
step with no change (``baseline_kpis``); predicted_delta = the predictor's own change-vs-no-change.
A change the safety filter clipped is written with ``scored: false, unscored_reason: "clipped"``.

Each predictor runs its own case (its own twin episode, the same scripted changes). Once a
version is paused, calls to it are refused, as the Cedar forbid on ``ran___predict`` would refuse
them; the demo counts those refusals. With the default dry-run policy client the forbid is
simulated in process, and every record is marked ``synthetic: true``.

Per model the demo also writes one ``result`` record (unit ``t3.trust_gate``) whose
``details.thresholds`` holds the declared gate settings and the thresholds.json sha256 (the
Trust view reads them there).

CLI (stand-in twin and predictors; writes only under --out):
    python -m t3.trust.demo --out /path/to/dir [--seeds 10]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from .gate import TrustGate
from .policy import PolicyClient
from .records import make_change_record, make_result_record
from .stepping import predicted_delta, step_with_counterfactual, twin_desc
from .store import MemoryTrustStore


def thresholds_summary(gate: TrustGate) -> dict:
    th = gate.thresholds
    return {"tau": th["tau"], "scales": th["scales"], "pause_below": th["pause_below"],
            "watch_below": th.get("watch_below"), "alpha": th["alpha"], "n_min": th["warmup"],
            "window": {"needed": th["recent_needed"], "of_last": th["recent_window"]},
            "direction_deadband": th["direction_deadband"], "weights": th["weights"],
            "keys": th["keys"], "sha256": gate.thresholds_sha256}


def run_case(*, gate: TrustGate, twin_factory, kpis, predictor, changes, scenario, seed, config,
             run_id, case_id, writer, synthetic=True, policy_client=None) -> dict:
    pc = policy_client or gate.policy_client
    mid, ver = predictor.model_id, predictor.version
    tw = twin_factory(scenario, seed, config)
    state = tw.reset()
    rows, paused_at, refused = [], None, 0
    for i, change in enumerate(changes, start=1):
        if pc.is_paused(mid, ver):
            refused += 1          # the forbid refuses ran___predict; the change waits for a person
            rows.append({"i": i, "refused": True})
            continue
        pdelta = predicted_delta(predictor, state, change)
        st = step_with_counterfactual(tw, twin_factory, kpis, change, scenario=scenario, seed=seed,
                                      config=config, before_state=state)
        state = st["after"]
        rec = make_change_record(
            run_id=run_id, case_id=case_id, item_id=f"step-{i - 1}", change=change,
            twin=twin_desc(tw, scenario, seed), baseline_kpis=st["baseline_kpis"],
            measured_kpis=st["measured_kpis"], predicted_delta=pdelta,
            predictor={"model_id": mid, "version": ver}, scales=gate.scales, deadband=gate.deadband,
            scored=not st["clipped"], unscored_reason="clipped" if st["clipped"] else None,
            synthetic=synthetic)
        res = gate.update(rec)
        rec["trust_before"], rec["trust_after"] = res["trust_before"], res["trust_after"]
        writer.append_records(run_id, "change", [rec])
        if res.get("skipped"):
            rows.append({"i": i, "refused": False, "skipped": res["skipped"]})
            continue
        rows.append({"i": i, "refused": False, "e": res["e"], "direction_agree": res["direction_agree"],
                     "score": res["score"], "trust": res["trust_after"], "band": res["band"],
                     "action": res["action"]})
        if res["action"] == "pause":
            paused_at = i
    scored = [r for r in rows if "trust" in r]
    trust = [r["trust"] for r in scored]
    warm = int(gate.thresholds["warmup"])
    return {"case_id": case_id, "model_id": mid, "version": ver, "changes": len(changes),
            "scored": len(scored), "clipped": sum(1 for r in rows if r.get("skipped", "").startswith("unscored")),
            "paused_at_change": paused_at, "refused_after_pause": refused,
            "final_trust": trust[-1] if trust else None,
            "min_trust_after_warmup": min(trust[warm - 1:], default=None),
            "e_over_tau": sum(1 for r in scored if r["e"] > float(gate.thresholds["tau"])),
            "trust_trajectory": [round(t, 4) for t in trust],
            "e_trajectory": [round(r["e"], 4) for r in scored], "rows": rows}


def run_demo(*, twin_factory, kpis, predictors, changes, writer, run_id, scenario="nominal",
             seed=0, config="official", store=None, policy_client=None, thresholds=None,
             synthetic=True, code_sha=None) -> dict:
    pc = policy_client or PolicyClient()
    gate = TrustGate(store or MemoryTrustStore(), thresholds, pc, writer, synthetic=synthetic)
    cases = []
    for p in predictors:
        case_id = f"case-{run_id[:8]}-{p.version}"
        c = run_case(gate=gate, twin_factory=twin_factory, kpis=kpis, predictor=p, changes=changes,
                     scenario=scenario, seed=seed, config=config, run_id=run_id, case_id=case_id,
                     writer=writer, synthetic=synthetic, policy_client=pc)
        cases.append(c)
        writer.append_records(run_id, "result", [make_result_record(
            run_id=run_id, unit="t3.trust_gate", metric="paused_at_change", value=c["paused_at_change"],
            target=None, passed=None, n_items=c["scored"],
            checker={"script": "t3/trust/demo.py", "code_sha": code_sha},
            details={"model_id": p.model_id, "version": p.version, "thresholds": thresholds_summary(gate),
                     "final_trust": c["final_trust"], "min_trust_after_warmup": c["min_trust_after_warmup"],
                     "refused_after_pause": c["refused_after_pause"], "trust_trajectory": c["trust_trajectory"]},
            synthetic=synthetic)])
    return {"run_id": run_id, "thresholds_sha256": gate.thresholds_sha256,
            "thresholds": gate.thresholds, "scenario": scenario, "seed": seed, "config": config,
            "policy_calls": pc.calls, "cases": cases}


def format_table(summary: dict) -> str:
    th = summary["thresholds"]
    lines = [f"run {summary['run_id']}  twin seed {summary['seed']}  preset {summary['config']}  "
             f"tau {th['tau']} (2 of last 3)  pause line {th['pause_below']}  n_min {th['warmup']}  alpha {th['alpha']}",
             f"{'model@version':<30}{'scored':>7}{'e>tau':>7}{'paused at':>11}{'refused':>9}{'final':>8}{'min*':>8}"]
    for c in summary["cases"]:
        fin = "-" if c["final_trust"] is None else f"{c['final_trust']:.3f}"
        mn = "-" if c["min_trust_after_warmup"] is None else f"{c['min_trust_after_warmup']:.3f}"
        lines.append(f"{c['model_id'] + '@' + c['version']:<30}{c['scored']:>7}{c['e_over_tau']:>7}"
                     f"{str(c['paused_at_change'] or 'never'):>11}{c['refused_after_pause']:>9}{fin:>8}{mn:>8}")
        lines.append("   trust: " + " ".join(f"{t:.2f}" for t in c["trust_trajectory"]))
        lines.append("   e:     " + " ".join(f"{t:.2f}" for t in c["e_trajectory"]))
    lines.append("   (* lowest trust once n_min is reached)")
    return "\n".join(lines)


def _code_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).parent, check=True).stdout.strip()
    except Exception:
        return None


class _Null:
    def append_records(self, *a):
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output directory (records/, demo_summary.json)")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--seed", type=int, default=0, help="twin seed of the headline run")
    ap.add_argument("--seeds", type=int, default=0, help="also repeat over this many twin seeds (robustness)")
    ap.add_argument("--fit-steps", type=int, default=400)
    ap.add_argument("--config", default="official")
    args = ap.parse_args(argv)

    from .local_writer import LocalRecordWriter
    from .scripted import load_script
    from .standin_predictors import fit_linear
    from .standin_twin import STANDIN_LABEL, StandinReplayTwin, kpis

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or time.strftime("t3trust-%Y%m%dT%H%M%SZ", time.gmtime())
    changes, script_sha = load_script()
    t0 = time.time()
    code_sha = _code_sha()
    healthy, fit = fit_linear(StandinReplayTwin, kpis, n_steps=args.fit_steps,
                              model_id="standin-linear", version="v1")
    degraded = healthy.flipped("v1-flipped")
    healthy.save(out / "standin-linear-v1.npz")
    writer = LocalRecordWriter(out)
    summary = run_demo(twin_factory=StandinReplayTwin, kpis=kpis, predictors=[healthy, degraded],
                       changes=changes, writer=writer, run_id=run_id, seed=args.seed,
                       config=args.config, code_sha=code_sha)
    print(format_table(summary))
    robustness = None
    max_n = summary["thresholds"]["max_changes_to_pause"]
    if args.seeds:
        rows = []
        for s in range(args.seeds):
            sm = run_demo(twin_factory=StandinReplayTwin, kpis=kpis, predictors=[healthy, degraded],
                          changes=changes, writer=_Null(), run_id=f"{run_id}-s{s}", seed=1000 + s,
                          config=args.config)
            h, d = sm["cases"]
            rows.append({"seed": 1000 + s, "healthy_paused_at": h["paused_at_change"],
                         "healthy_min_trust": h["min_trust_after_warmup"],
                         "degraded_paused_at": d["paused_at_change"]})
        dp = [r["degraded_paused_at"] for r in rows]
        robustness = {"seeds": args.seeds, "twin_seeds": f"1000..{999 + args.seeds}",
                      "healthy_false_pauses": sum(r["healthy_paused_at"] is not None for r in rows),
                      "degraded_paused": sum(x is not None for x in dp),
                      "degraded_paused_within_declared": sum(x is not None and x <= max_n for x in dp),
                      "degraded_paused_at": dp, "rows": rows}
        print(f"\nrobustness over {args.seeds} other twin seeds: healthy paused "
              f"{robustness['healthy_false_pauses']}x; degraded paused {robustness['degraded_paused']}x, "
              f"within {max_n} changes {robustness['degraded_paused_within_declared']}x; at {dp}")
    summary.update({"twin_impl": STANDIN_LABEL,
                    "predictor_impl": "t3.trust.standin_predictors (stand-in for t2.predict)",
                    "policy_client": "dry-run (no AWS call; forbid simulated in process)",
                    "fit": fit, "script_sha256": script_sha, "code_sha": code_sha,
                    "elapsed_s": round(time.time() - t0, 2), "robustness": robustness})
    (out / "demo_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    ok_h = summary["cases"][0]["paused_at_change"] is None
    pd = summary["cases"][1]["paused_at_change"]
    ok_d = pd is not None and pd <= max_n
    print(f"\nhealthy never paused: {ok_h};  degraded paused within {max_n} changes: {ok_d} (at {pd})")
    print(f"wrote {out / 'demo_summary.json'} and {out / 'records' / run_id}")
    return 0 if ok_h and ok_d else 1


if __name__ == "__main__":
    sys.exit(main())
