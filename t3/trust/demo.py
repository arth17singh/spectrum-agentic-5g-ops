"""Scripted trust-gate demo. No agents: a fixed list of changes is applied to the twin, every
predictor's predicted KPI delta is compared with the measured delta, and the gate decides.

Real modules: ``twin.replay.ReplayTwin`` (official preset), ``twin.kpis.headline``, ``t2.predict``
(healthy ``load`` and degraded ``load_degraded`` = fault "tilt_reversed"), the evlib-shaped writer
``t3.trust.evio.EvidenceWriter`` and the dry-run ``PolicyClient``. ``run_demo`` takes them all
injected, so tests use fakes.

Deltas, not levels (A5): measured_delta = KPIs after the change - KPIs of the same step with no
change (``ReplayTwin.counterfactual``); predicted_delta = ``t2.predict`` output. A change the safety
filter clamped is written with ``scored: false, unscored_reason: "clipped"``.

Each predictor runs its own case (its own twin episode, the same scripted changes). Once a version
is paused, calls to it are refused, as the Cedar forbid on ``ran___predict`` would refuse them; the
demo counts those refusals. The policy client is a dry run (forbid simulated in process), so every
record is ``synthetic: true``.

Per model one ``result`` (unit ``t3.trust_gate``, run_id ``<run_id>.<model_id>``) carries
``details.thresholds`` (the declared settings and the thresholds.json sha256) for the Trust view.

    python -m t3.trust.demo --out /path/to/evidence-dir [--seeds 10]
"""

from __future__ import annotations

import argparse
import json
import re
import secrets
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


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s)


def gate_result(*, run_id, case: dict, expect_pause: bool, gate: TrustGate, code_sha, synthetic, n_changes) -> dict:
    """One `result` per model (evlib: numeric value, boolean passed, its own run_id)."""
    max_n = int(gate.thresholds["max_changes_to_pause"])
    at = case["paused_at_change"]
    if expect_pause:
        metric, value = "changes_to_pause", float(at if at is not None else n_changes + 1)
        target = {"metric": metric, "op": "<=", "value": max_n, "unit": "changes"}
        passed = at is not None and at <= max_n
    else:
        metric, value = "pauses", float(0 if at is None else 1)
        target = {"metric": metric, "op": "==", "value": 0, "unit": "count"}
        passed = at is None
    return make_result_record(
        run_id=f"{run_id}.{_slug(case['model_id'] + '@' + case['version'])}", unit="t3.trust_gate", metric=metric, value=value,
        target=target, passed=passed, n_items=case["scored"],
        checker={"script": "t3/trust/demo.py", "code_sha": code_sha or "unknown"},
        details={"model_id": case["model_id"], "version": case["version"],
                 "role": "degraded" if expect_pause else "healthy", "source_run_id": run_id,
                 "case_id": case["case_id"], "thresholds": thresholds_summary(gate),
                 "paused_at_change": at, "final_trust": case["final_trust"],
                 "min_trust_after_warmup": case["min_trust_after_warmup"],
                 "refused_after_pause": case["refused_after_pause"], "trust_trajectory": case["trust_trajectory"]},
        synthetic=synthetic)


def run_demo(*, twin_factory, kpis, predictors, changes, writer, run_id, scenario="nominal",
             seed=0, config="official", store=None, policy_client=None, thresholds=None,
             synthetic=True, code_sha=None, expect_pause=None) -> dict:
    """``expect_pause[i]``: predictor i is the deliberately degraded one (default: the last)."""
    pc = policy_client or PolicyClient()
    gate = TrustGate(store or MemoryTrustStore(), thresholds, pc, writer, synthetic=synthetic)
    expect_pause = expect_pause or [i == len(predictors) - 1 for i in range(len(predictors))]
    cases, results = [], []
    for idx, p in enumerate(predictors):
        case_id = f"case-{run_id[:8]}-{idx + 1:02d}"
        c = run_case(gate=gate, twin_factory=twin_factory, kpis=kpis, predictor=p, changes=changes,
                     scenario=scenario, seed=seed, config=config, run_id=run_id, case_id=case_id,
                     writer=writer, synthetic=synthetic, policy_client=pc)
        cases.append(c)
        res = gate_result(run_id=run_id, case=c, expect_pause=expect_pause[idx], gate=gate,
                          code_sha=code_sha, synthetic=synthetic, n_changes=len(changes))
        if hasattr(writer, "write_result"):
            writer.write_result(res)
        results.append(res)
    return {"run_id": run_id, "thresholds_sha256": gate.thresholds_sha256,
            "thresholds": gate.thresholds, "scenario": scenario, "seed": seed, "config": config,
            "policy_calls": pc.calls, "cases": cases, "results": results}


def format_table(summary: dict) -> str:
    th = summary["thresholds"]
    lines = [f"run {summary['run_id']}  twin seed {summary['seed']}  preset {summary['config']}  "
             f"tau {th['tau']} (2 of last 3)  pause line {th['pause_below']}  n_min {th['warmup']}  alpha {th['alpha']}",
             f"{'model@version':<48}{'scored':>7}{'e>tau':>7}{'paused at':>11}{'refused':>9}{'final':>8}{'min*':>8}"]
    for c in summary["cases"]:
        fin = "-" if c["final_trust"] is None else f"{c['final_trust']:.3f}"
        mn = "-" if c["min_trust_after_warmup"] is None else f"{c['min_trust_after_warmup']:.3f}"
        lines.append(f"{c['model_id'] + '@' + c['version']:<48}{c['scored']:>7}{c['e_over_tau']:>7}"
                     f"{str(c['paused_at_change'] or 'never'):>11}{c['refused_after_pause']:>9}{fin:>8}{mn:>8}")
        lines.append("   trust: " + " ".join(f"{t:.2f}" for t in c["trust_trajectory"]))
        lines.append("   e:     " + " ".join(f"{t:.2f}" for t in c["e_trajectory"]))
    lines.append("   (* lowest trust once n_min is reached)")
    return "\n".join(lines)


def code_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).parent, check=True).stdout.strip()
    except Exception:
        return None


class _Null:
    def append_records(self, *a):
        return []


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="evidence root (records/, results/) and demo_summary.json")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--seed", type=int, default=0, help="twin seed of the headline run")
    ap.add_argument("--seeds", type=int, default=0, help="also repeat over this many twin seeds (robustness)")
    ap.add_argument("--config", default="official")
    ap.add_argument("--scenario", default="nominal")
    args = ap.parse_args(argv)

    from t2.predict import load, load_degraded
    from twin.kpis import headline
    from twin.replay import ReplayTwin

    from .calibrate import ARTIFACT
    from .evio import EvidenceWriter
    from .scripted import load_script

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or time.strftime("t3.trust-%Y%m%dT%H%M%SZ-", time.gmtime()) + secrets.token_hex(2)
    changes, script_sha = load_script()
    t0 = time.time()
    sha = code_sha()
    healthy, degraded = load(ARTIFACT), load_degraded(ARTIFACT)
    writer = EvidenceWriter(out)
    summary = run_demo(twin_factory=ReplayTwin, kpis=headline, predictors=[healthy, degraded],
                       changes=changes, writer=writer, run_id=run_id, seed=args.seed,
                       config=args.config, scenario=args.scenario, code_sha=sha)
    print(format_table(summary))
    robustness = None
    max_n = summary["thresholds"]["max_changes_to_pause"]
    if args.seeds:
        rows = []
        for s in range(args.seeds):
            sm = run_demo(twin_factory=ReplayTwin, kpis=headline, predictors=[healthy, degraded],
                          changes=changes, writer=_Null(), run_id=f"{run_id}-s{s}", seed=1000 + s,
                          config=args.config, scenario=args.scenario)
            h, d = sm["cases"]
            rows.append({"seed": 1000 + s, "healthy_paused_at": h["paused_at_change"],
                         "healthy_min_trust": h["min_trust_after_warmup"],
                         "degraded_paused_at": d["paused_at_change"]})
        dp = [r["degraded_paused_at"] for r in rows]
        robustness = {"seeds": args.seeds, "twin_seeds": f"1000..{999 + args.seeds}",
                      "healthy_false_pauses": sum(r["healthy_paused_at"] is not None for r in rows),
                      "healthy_paused_at": [r["healthy_paused_at"] for r in rows],
                      "degraded_paused": sum(x is not None for x in dp),
                      "degraded_paused_within_declared": sum(x is not None and x <= max_n for x in dp),
                      "degraded_paused_at": dp, "rows": rows}
        print(f"\nrobustness over {args.seeds} other twin seeds: healthy paused "
              f"{robustness['healthy_false_pauses']}x at {robustness['healthy_paused_at']}; degraded paused "
              f"{robustness['degraded_paused']}x, within {max_n} changes "
              f"{robustness['degraded_paused_within_declared']}x; at {dp}")
    summary.update({"twin_impl": "twin.replay.ReplayTwin", "predictor_impl": f"t2.predict {ARTIFACT} (+ tilt_reversed)",
                    "writer": writer.backend, "policy_client": "dry-run (no AWS call; forbid simulated in process)",
                    "script_sha256": script_sha, "code_sha": sha,
                    "elapsed_s": round(time.time() - t0, 2), "robustness": robustness})
    (out / "demo_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    ok_h = summary["cases"][0]["paused_at_change"] is None
    pd = summary["cases"][1]["paused_at_change"]
    ok_d = pd is not None and pd <= max_n
    print(f"\nhealthy never paused: {ok_h};  degraded paused within {max_n} changes: {ok_d} (at {pd})")
    print(f"wrote {out / 'demo_summary.json'} and records/results under {out}")
    return 0 if ok_h and ok_d else 1


if __name__ == "__main__":
    sys.exit(main())
