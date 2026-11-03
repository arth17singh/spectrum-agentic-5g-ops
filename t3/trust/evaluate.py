"""One-shot evaluation of declared threshold sets on fresh twin seeds (run once, after declaration).

For each thresholds version and each twin seed: the 24-change demo script on twin.replay (official,
nominal) with the healthy t2 predictor and tilt_reversed, the dry-run policy client, records and
results written through the evlib-shaped writer. Then the declared success criterion of v2 is
checked for every version:

    healthy false pauses <= 2 of 20  AND  degraded paused in >= 19 of 20 runs by change 6

    python -m t3.trust.evaluate --out <evidence dir> --summary t3/trust/runs/v2_eval.json \
        --versions v2 v1 --seeds 2000 2019
"""

from __future__ import annotations

import argparse
import json
import secrets
import time
from pathlib import Path

from .calibrate_v2 import SUCCESS
from .gate import load_thresholds, thresholds_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--versions", nargs="+", default=["v2", "v1"])
    ap.add_argument("--seeds", nargs=2, type=int, default=[2000, 2019])
    args = ap.parse_args(argv)

    from t2.predict import load, load_degraded
    from twin.kpis import headline
    from twin.replay import ReplayTwin

    from .calibrate import ARTIFACT
    from .demo import code_sha, run_demo
    from .evio import EvidenceWriter
    from .scripted import load_script

    changes, script_sha = load_script()
    healthy, degraded = load(ARTIFACT), load_degraded(ARTIFACT)
    writer = EvidenceWriter(args.out)
    sha = code_sha()
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    seeds = list(range(args.seeds[0], args.seeds[1] + 1))
    out = {"what": "one-shot evaluation on fresh twin seeds", "code_sha": sha, "started_at": stamp,
           "twin": "twin.replay.ReplayTwin official/nominal", "script_sha256": script_sha, "seeds": args.seeds,
           "criterion": SUCCESS, "versions": {}}
    for v in args.versions:
        th, th_sha = load_thresholds(thresholds_path(v))
        rows = []
        for s in seeds:
            run_id = f"t3.trust-eval-{v}-{stamp}-{s}-{secrets.token_hex(2)}"
            sm = run_demo(twin_factory=ReplayTwin, kpis=headline, predictors=[healthy, degraded], changes=changes,
                          writer=writer, run_id=run_id, seed=s, config="official", scenario="nominal",
                          thresholds=thresholds_path(v), code_sha=sha)
            h, d = sm["cases"]
            rows.append({"seed": s, "run_id": run_id, "healthy_paused_at": h["paused_at_change"],
                         "healthy_min_trust": h["min_trust_after_warmup"], "degraded_paused_at": d["paused_at_change"]})
        hp = [r["healthy_paused_at"] for r in rows]
        dp = [r["degraded_paused_at"] for r in rows]
        n_false = sum(x is not None for x in hp)
        n_det6 = sum(x is not None and x <= 6 for x in dp)
        passed = n_false <= SUCCESS["healthy_false_pauses_max"] and n_det6 >= SUCCESS["degraded_paused_by_change_6_min"]
        out["versions"][v] = {"thresholds_sha256": th_sha, "alpha": th["alpha"], "tau": th["tau"],
                              "pause_below": th["pause_below"], "scales": th["scales"],
                              "healthy_false_pauses": n_false, "healthy_paused_at": hp,
                              "degraded_paused_by_6": n_det6, "degraded_paused_any": sum(x is not None for x in dp),
                              "degraded_paused_at": dp, "runs": len(rows), "criterion_passed": passed, "rows": rows}
        print(f"{v} (sha {th_sha[:12]}): healthy false pauses {n_false}/{len(rows)} at {hp}; degraded paused by "
              f"change 6 {n_det6}/{len(rows)} at {dp}; criterion passed: {passed}")
    out["finished_at"] = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    Path(args.summary).write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
