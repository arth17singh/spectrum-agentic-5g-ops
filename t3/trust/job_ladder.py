"""Job entry: the pilot ladder, R1 (MAVERIC, official) and R2 (REHEARSAL, Maveric at viavi-like settings),
one evidence run per model. See t3/trust/jobs.py for the record layout.

Declared before it runs: healthy R1 mean gain > 0 (and its bootstrap 95 % CI lower bound > 0); degraded R1
mean gain < 0. The R2 gain and the settings shift (R2 / R1, NOT retention) are reported as measurements.
Retention stays null: there is no VIAVI rung.

    RUN_ID=<run_id> EVIDENCE_S3=s3://<bucket>/evidence/ python -m t3.trust.job_ladder --model healthy
    python -m t3.trust.job_ladder --model degraded --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from . import jobs
from .job_gate_eval import twin_version


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["healthy", "degraded"], required=True)
    ap.add_argument("--run-id", default=None, help="default: $RUN_ID or $EVIDENCE_RUN_ID")
    ap.add_argument("--dry-run", action="store_true", help="first two seeds, in memory; writes nothing")
    ap.add_argument("--print-declaration", action="store_true")
    ap.add_argument("--declare-first", action="store_true")
    a = ap.parse_args(argv)

    testset, testset_sha = jobs.load_testset(jobs.LADDER_TESTSET)
    run_id = a.run_id or jobs.env_run_id() or (
        f"dryrun-t3.ladder-{a.model}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}" if a.dry_run else None)
    if not run_id:
        sys.exit("set RUN_ID (or EVIDENCE_RUN_ID) or pass --run-id")
    sha = jobs.code_sha_or_unknown()
    decl = jobs.ladder_declaration(run_id=run_id, model=a.model, code_sha=sha,
                                   region=os.environ.get("AWS_REGION", "us-east-2"), testset=testset,
                                   testset_sha=testset_sha, twin_version=twin_version())
    if a.print_declaration:
        print(json.dumps(decl, indent=1))
        return 0
    if a.declare_first and not a.dry_run:
        jobs.declare_first(decl)

    from t2.predict import load, load_degraded
    from twin.kpis import headline
    from twin.replay import ReplayTwin

    from .scripted import load_script

    changes, script_sha = load_script()
    if script_sha != testset["script_sha256"]:
        sys.exit("demo script differs from the declared test set; refusing to run")
    predictor = (load if a.model == "healthy" else load_degraded)(testset["artifact"])
    rungs = [{"name": r["name"], "factory": ReplayTwin, "config": r["config"], "scenario": r["scenario"],
              "twin_kind": "maveric-inprocess"} for r in (testset["rungs"]["R1"], testset["rungs"]["R2"])]
    seeds = testset["seeds"][:2] if a.dry_run else testset["seeds"]
    out = jobs.run_ladder_job(model=a.model, run_id=run_id, seeds=seeds, rungs=rungs, kpis=headline,
                              predictor=predictor, changes=changes[: testset["steps"]], testset_sha=testset_sha,
                              code_sha=sha)
    return jobs.finish(run_id=run_id, declaration=decl, out=out, dry_run=a.dry_run, declare_first_done=a.declare_first)


if __name__ == "__main__":
    raise SystemExit(main())
