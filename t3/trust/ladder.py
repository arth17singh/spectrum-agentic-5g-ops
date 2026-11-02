"""The pilot ladder: the same scripted situations on two or three rungs, each model's gain over a
"no change" baseline on each rung, and retention with a bootstrap interval.

Rungs (labels are fixed by contract amendment 1 and are the only allowed values):

    name "maveric"    label "MAVERIC"                         official Maveric preset
    name "rehearsal"  label "REHEARSAL (Viavi-like Maveric)"  Maveric at viavi-like settings (NOT Viavi)
    name "viavi"      label "VIAVI"                           the organisers' Viavi adapter (when access exists)

Retention is meaningful only against Maveric at viavi-like settings (amendment 1): it is
``gain(VIAVI) / gain(REHEARSAL)`` and is **null when there is no VIAVI rung**. Without Viavi the
report still gives every rung's gain, and a separately named ``settings_shift`` ratio
(REHEARSAL / MAVERIC) in the local report only, never called retention.

Gain, per case (twin seed): at each of ``steps`` steps the model chooses among candidates
{scripted change, its mirror (negated deltas), no change}: a predictor picks the candidate with
the best predicted objective (no change if nothing beats 0); a policy (``.propose(state)``)
proposes directly. The step gain is objective(measured KPIs after the pick) - objective(KPIs of
the replay of the same step with no change), exact on a deterministic twin. Case gain = mean
step gain. Cases are the same seeds and scripted changes on every rung, so gains are paired.
A twin without ``from_log`` (an HTTP adapter) gets an unpaired baseline: a separate no-change
episode with the same seed; the rung says ``pairing: "unpaired"``.

Objective (declared here, hashed into the report): coverage_ratio + sla_ratio + 0.05 * sinr_p10_db.

Output: one ``result`` record per model, ``unit: "t3.ladder"``, ``details = {rungs: [{name, label,
twin_kind, n_cases, gain, gain_ci}], retention, retention_ci, model_id, version}`` (shape fixed by
the coordinator), plus a fuller local JSON report.

    python -m t3.trust.ladder --out /path/to/dir [--cases 10 --steps 12]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from .records import make_result_record
from .retention import bootstrap_mean_ci, bootstrap_retention, retention
from .stepping import baseline_twin, predicted_delta

LABELS = {"maveric": "MAVERIC", "rehearsal": "REHEARSAL (Viavi-like Maveric)", "viavi": "VIAVI"}
OBJECTIVE = {"coverage_ratio": 1.0, "sla_ratio": 1.0, "sinr_p10_db": 0.05}
OBJECTIVE_SHA = hashlib.sha256(json.dumps(OBJECTIVE, sort_keys=True).encode()).hexdigest()


def objective(k: dict) -> float:
    return sum(w * float(k[name]) for name, w in OBJECTIVE.items())


def mirror(change: list[dict]) -> list[dict]:
    return [{"cell": c["cell"], "d_power_db": -float(c.get("d_power_db", 0.0)),
             "d_tilt_deg": -float(c.get("d_tilt_deg", 0.0))} for c in change]


def choose(model, state: dict, change: list[dict]) -> list[dict]:
    if hasattr(model, "propose"):
        return model.propose(state)
    best, best_val = [], 0.0
    for cand in (change, mirror(change)):
        d = predicted_delta(model, state, cand)
        val = sum(w * d[k] for k, w in OBJECTIVE.items())
        if val > best_val:
            best, best_val = cand, val
    return best


def case_gain(model, twin_factory, kpis, *, scenario, seed, config, changes) -> dict:
    paired = hasattr(twin_factory, "from_log")
    tw = twin_factory(scenario, seed, config)
    state = tw.reset()
    base_tw = None
    if not paired:
        base_tw = twin_factory(scenario, seed, config)
        base_tw.reset()
    gains, picks = [], []
    for change in changes:
        pick = choose(model, state, change)
        if paired:
            ref = baseline_twin(tw, twin_factory, scenario, seed, config)
            base = kpis(ref.apply([]))
        else:
            base = kpis(base_tw.apply([]))
        state = tw.apply(pick)
        gains.append(objective(kpis(state)) - objective(base))
        picks.append("none" if not pick else ("scripted" if pick == change else "mirror"))
    return {"seed": seed, "gain": float(np.mean(gains)), "step_gains": gains,
            "picks": {p: picks.count(p) for p in ("scripted", "mirror", "none")},
            "pairing": "paired" if paired else "unpaired"}


def run_rung(model, rung: dict, kpis, *, seeds, changes, scenario="nominal") -> dict:
    cases = [case_gain(model, rung["factory"], kpis, scenario=scenario, seed=s, config=rung["config"],
                       changes=changes) for s in seeds]
    g = [c["gain"] for c in cases]
    ci = bootstrap_mean_ci(g)
    return {"name": rung["name"], "label": LABELS[rung["name"]], "twin_kind": rung["twin_kind"],
            "config": rung["config"], "n_cases": len(cases), "gain": float(np.mean(g)),
            "gain_ci": list(ci) if ci else None, "pairing": cases[0]["pairing"] if cases else None,
            "cases": cases}


def ladder(model, rungs: list[dict], kpis, *, seeds, changes, scenario="nominal") -> dict:
    names = [r["name"] for r in rungs]
    if any(n not in LABELS for n in names) or len(set(names)) != len(names):
        raise ValueError(f"rung names must be distinct and from {sorted(LABELS)}")
    results = {r["name"]: run_rung(model, r, kpis, seeds=seeds, changes=changes, scenario=scenario) for r in rungs}
    ret, ret_ci, boot = None, None, None
    if "viavi" in results and "rehearsal" in results:
        boot = bootstrap_retention([c["gain"] for c in results["rehearsal"]["cases"]],
                                   [c["gain"] for c in results["viavi"]["cases"]])
        ret, ret_ci = boot["point"], boot["ci"]
    shift = None
    if "maveric" in results and "rehearsal" in results:
        b = bootstrap_retention([c["gain"] for c in results["maveric"]["cases"]],
                                [c["gain"] for c in results["rehearsal"]["cases"]])
        shift = {"what": "settings shift on Maveric (REHEARSAL gain / MAVERIC gain). NOT retention: "
                         "no Viavi involved", "point": b["point"], "ci": b["ci"], "rule": b["rule"]}
    details = {
        "rungs": [{k: results[n][k] for k in ("name", "label", "twin_kind", "n_cases", "gain", "gain_ci")}
                  for n in names],
        "retention": ret, "retention_ci": ret_ci,
        "model_id": model.model_id, "version": model.version,
    }
    return {"details": details, "full": results, "retention_bootstrap": boot, "settings_shift": shift,
            "has_real_viavi": "viavi" in results}


def ladder_result_record(lad: dict, *, run_id: str, n_items: int, code_sha=None, synthetic=True) -> dict:
    return make_result_record(run_id=run_id, unit="t3.ladder", metric="retention",
                              value=lad["details"]["retention"], target=None, passed=None, n_items=n_items,
                              checker={"script": "t3/trust/ladder.py", "code_sha": code_sha},
                              details=lad["details"], synthetic=synthetic)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--cases", type=int, default=10)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--fit-steps", type=int, default=400)
    args = ap.parse_args(argv)

    from .demo import _code_sha
    from .local_writer import LocalRecordWriter
    from .scripted import load_script
    from .standin_predictors import fit_linear
    from .standin_twin import STANDIN_LABEL, StandinReplayTwin, kpis

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or time.strftime("t3ladder-%Y%m%dT%H%M%SZ", time.gmtime())
    changes, script_sha = load_script()
    changes = changes[: args.steps]
    t0 = time.time()
    healthy, fit = fit_linear(StandinReplayTwin, kpis, n_steps=args.fit_steps, model_id="standin-linear", version="v1")
    degraded = healthy.flipped("v1-flipped")
    rungs = [{"name": "maveric", "factory": StandinReplayTwin, "config": "official", "twin_kind": "maveric-inprocess"},
             {"name": "rehearsal", "factory": StandinReplayTwin, "config": "viavi-like", "twin_kind": "maveric-inprocess"}]
    seeds = list(range(3000, 3000 + args.cases))
    writer = LocalRecordWriter(out)
    code_sha = _code_sha()
    report = {"run_id": run_id, "objective": OBJECTIVE, "objective_sha256": OBJECTIVE_SHA,
              "script_sha256": script_sha, "steps": len(changes), "seeds": seeds,
              "twin_impl": STANDIN_LABEL, "predictor_impl": "t3.trust.standin_predictors (stand-in for t2.predict)",
              "banner": "No VIAVI rung: retention is null by rule. REHEARSAL is Maveric at viavi-like settings, NOT Viavi.",
              "fit": fit, "models": []}
    for m in (healthy, degraded):
        lad = ladder(m, rungs, kpis, seeds=seeds, changes=changes)
        writer.append_records(run_id, "result", [ladder_result_record(lad, run_id=run_id, n_items=len(seeds) * len(rungs),
                                                                      code_sha=code_sha)])
        report["models"].append(lad)
        print(f"\n{m.model_id}@{m.version}")
        for r in lad["details"]["rungs"]:
            lo, hi = r["gain_ci"]
            print(f"  {r['label']:<32} {r['twin_kind']:<18} n={r['n_cases']:<3} gain {r['gain']:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
        print(f"  retention (VIAVI / REHEARSAL): {lad['details']['retention']}  (null: no VIAVI rung)")
        s = lad["settings_shift"]
        ci = "None" if s["ci"] is None else f"[{s['ci'][0]:.2f}, {s['ci'][1]:.2f}]"
        pt = "None" if s["point"] is None else f"{s['point']:.3f}"
        print(f"  settings shift, NOT retention (REHEARSAL / MAVERIC): {pt}  CI {ci}")
    report["elapsed_s"] = round(time.time() - t0, 2)
    (out / "ladder_report.json").write_text(json.dumps(report, indent=1, default=str))
    print(f"\nwrote {out / 'ladder_report.json'} and {out / 'records' / run_id / 'result.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
