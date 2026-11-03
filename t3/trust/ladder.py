"""The pilot ladder: the same scripted situations on each rung, each model's gain over a "no change"
baseline on each rung, and retention with a bootstrap interval.

Rungs (labels fixed by contract amendment 1; the only allowed values):

    name "maveric"    label "MAVERIC"                         official Maveric preset, scenario "nominal"
    name "rehearsal"  label "REHEARSAL (Viavi-like Maveric)"  Maveric "viavi-like" preset, scenario "viavi_stationary" (NOT Viavi)
    name "viavi"      label "VIAVI"                           the organisers' Viavi adapter (when access exists)

Retention = gain(VIAVI) / gain(REHEARSAL) (confirmed by the lead). It is **null when there is no VIAVI
rung**, and ``details.retention_null_reason`` says why. The REHEARSAL / MAVERIC ratio appears only in
the local report as ``settings_shift``, never called retention.

Gain, per case (twin seed): at each step the model chooses among {scripted change, its mirror
(negated deltas), no change}: a predictor picks the candidate with the best predicted objective (no
change if nothing beats 0); a policy (``.propose(state)``) proposes directly. Step gain =
objective(KPIs after the pick) - objective(KPIs of the same step with no change), exact on the replay
twin (``counterfactual``). Case gain = mean step gain. Same seeds and changes on every rung (paired).
A twin without ``counterfactual``/``from_log`` (an HTTP adapter) gets an unpaired baseline: a separate
no-change episode with the same seed (``pairing: "unpaired"``). Not built: A5's Viavi noise-floor
filter from no-op probes.

Steps default to 5 (the Viavi episode length) so every rung, including a future VIAVI rung, runs the
same script. Objective (declared, hashed): coverage_ratio + sla_ratio + 0.05 * sinr_p10_db.

Output: one ``result`` per model, ``unit: "t3.ladder"``, run_id ``<run_id>.<model_id>``, with
``details = {rungs: [{name, label, twin_kind, n_cases, gain, gain_ci}], retention, retention_ci,
model_id, version, retention_null_reason?}``. evlib needs a numeric value: ``metric`` is
"retention" when it exists, else "gain_rehearsal" (target > 0).

    python -m t3.trust.ladder --out /path/to/evidence-dir [--cases 20 --steps 5]
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import secrets
import sys
import time
from pathlib import Path

import numpy as np

from .records import make_result_record
from .retention import bootstrap_mean_ci, bootstrap_retention
from .stepping import predicted_delta

LABELS = {"maveric": "MAVERIC", "rehearsal": "REHEARSAL (Viavi-like Maveric)", "viavi": "VIAVI"}
OBJECTIVE = {"coverage_ratio": 1.0, "sla_ratio": 1.0, "sinr_p10_db": 0.05}
OBJECTIVE_SHA = hashlib.sha256(json.dumps(OBJECTIVE, sort_keys=True).encode()).hexdigest()
RETENTION_TARGET = 0.5  # declared: at least half the rehearsal gain should survive on Viavi
NO_VIAVI = "no VIAVI rung (no Viavi access yet): retention = gain(VIAVI) / gain(REHEARSAL) is undefined"


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
    tw = twin_factory(scenario, seed, config)
    state = tw.reset()
    paired = hasattr(tw, "counterfactual") or hasattr(twin_factory, "from_log")
    base_tw = None
    if not paired:
        base_tw = twin_factory(scenario, seed, config)
        base_tw.reset()
    gains, picks = [], []
    for change in changes:
        pick = choose(model, state, change)
        if hasattr(tw, "counterfactual"):
            base = tw.counterfactual([])["baseline"]
        elif paired:
            base = kpis(twin_factory.from_log(scenario, seed, config, copy.deepcopy(tw.action_log)).apply([]))
        else:
            base = kpis(base_tw.apply([]))
        state = tw.apply(pick)
        gains.append(objective(kpis(state)) - objective(base))
        picks.append("none" if not pick else ("scripted" if pick == change else "mirror"))
    return {"seed": seed, "gain": float(np.mean(gains)), "step_gains": gains,
            "picks": {p: picks.count(p) for p in ("scripted", "mirror", "none")},
            "pairing": "paired" if paired else "unpaired"}


def run_rung(model, rung: dict, kpis, *, seeds, changes) -> dict:
    cases = [case_gain(model, rung["factory"], kpis, scenario=rung["scenario"], seed=s, config=rung["config"],
                       changes=changes) for s in seeds]
    g = [c["gain"] for c in cases]
    ci = bootstrap_mean_ci(g)
    return {"name": rung["name"], "label": LABELS[rung["name"]], "twin_kind": rung["twin_kind"],
            "config": rung["config"], "scenario": rung["scenario"], "n_cases": len(cases),
            "gain": float(np.mean(g)), "gain_ci": list(ci) if ci else None,
            "pairing": cases[0]["pairing"] if cases else None, "cases": cases}


def ladder(model, rungs: list[dict], kpis, *, seeds, changes) -> dict:
    names = [r["name"] for r in rungs]
    if any(n not in LABELS for n in names) or len(set(names)) != len(names):
        raise ValueError(f"rung names must be distinct and from {sorted(LABELS)}")
    results = {r["name"]: run_rung(model, r, kpis, seeds=seeds, changes=changes) for r in rungs}
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
    if ret is None:
        details["retention_null_reason"] = (NO_VIAVI if "viavi" not in results else
                                            (boot or {}).get("rule", "rehearsal rung missing"))
    return {"details": details, "full": results, "retention_bootstrap": boot, "settings_shift": shift,
            "has_real_viavi": "viavi" in results}


def ladder_result_record(lad: dict, *, run_id: str, n_items: int, code_sha=None, synthetic=True) -> dict:
    d = lad["details"]
    if d["retention"] is not None:
        metric, value = "retention", float(d["retention"])
        target = {"metric": "retention", "op": ">=", "value": RETENTION_TARGET, "unit": ""}
        passed = value >= RETENTION_TARGET
    else:
        reh = next((r for r in d["rungs"] if r["name"] == "rehearsal"), d["rungs"][-1])
        metric, value = "gain_" + reh["name"], float(reh["gain"])
        target = {"metric": metric, "op": ">", "value": 0, "unit": "objective"}
        passed = value > 0
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", d["model_id"] + "@" + d["version"])
    return make_result_record(run_id=f"{run_id}.{slug}", unit="t3.ladder", metric=metric, value=value,
                              target=target, passed=passed, n_items=n_items,
                              checker={"script": "t3/trust/ladder.py", "code_sha": code_sha or "unknown"},
                              details=d, synthetic=synthetic)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--cases", type=int, default=20)
    ap.add_argument("--steps", type=int, default=5)
    args = ap.parse_args(argv)

    from t2.predict import load, load_degraded
    from twin.kpis import headline
    from twin.presets import VIAVI_SCENARIO
    from twin.replay import ReplayTwin

    from .calibrate import ARTIFACT
    from .demo import code_sha
    from .evio import EvidenceWriter
    from .scripted import load_script

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or time.strftime("t3.ladder-%Y%m%dT%H%M%SZ-", time.gmtime()) + secrets.token_hex(2)
    changes, script_sha = load_script()
    changes = changes[: args.steps]
    t0 = time.time()
    rungs = [{"name": "maveric", "factory": ReplayTwin, "config": "official", "scenario": "nominal",
              "twin_kind": "maveric-inprocess"},
             {"name": "rehearsal", "factory": ReplayTwin, "config": "viavi-like", "scenario": VIAVI_SCENARIO,
              "twin_kind": "maveric-inprocess"}]
    seeds = list(range(3000, 3000 + args.cases))
    writer = EvidenceWriter(out)
    sha = code_sha()
    report = {"run_id": run_id, "objective": OBJECTIVE, "objective_sha256": OBJECTIVE_SHA,
              "script_sha256": script_sha, "steps": len(changes), "seeds": seeds,
              "twin_impl": "twin.replay.ReplayTwin", "predictor_impl": f"t2.predict {ARTIFACT} (+ tilt_reversed)",
              "banner": "No VIAVI rung: retention is null by rule. REHEARSAL is Maveric at viavi-like settings, NOT Viavi.",
              "code_sha": sha, "models": []}
    for m in (load(ARTIFACT), load_degraded(ARTIFACT)):
        lad = ladder(m, rungs, headline, seeds=seeds, changes=changes)
        writer.write_result(ladder_result_record(lad, run_id=run_id, n_items=len(seeds) * len(rungs), code_sha=sha))
        report["models"].append(lad)
        print(f"\n{m.model_id}@{m.version}")
        for r in lad["details"]["rungs"]:
            lo, hi = r["gain_ci"]
            print(f"  {r['label']:<32} {r['twin_kind']:<18} n={r['n_cases']:<3} gain {r['gain']:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
        print(f"  retention (VIAVI / REHEARSAL): {lad['details']['retention']}  ({lad['details'].get('retention_null_reason', '')})")
        s = lad["settings_shift"]
        ci = "None" if s["ci"] is None else f"[{s['ci'][0]:.2f}, {s['ci'][1]:.2f}]"
        pt = "None" if s["point"] is None else f"{s['point']:.3f}"
        print(f"  settings shift, NOT retention (REHEARSAL / MAVERIC): {pt}  CI {ci}")
    report["elapsed_s"] = round(time.time() - t0, 2)
    (out / "ladder_report.json").write_text(json.dumps(report, indent=1, default=str))
    print(f"\nwrote {out / 'ladder_report.json'} and results/{run_id}.*.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
