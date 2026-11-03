"""Job entry: trust-gate evaluation on the declared fresh seeds 2000-2019, one evidence run per
(thresholds version, model). See t3/trust/jobs.py for the record layout and the re-measurement note.

    RUN_ID=<run_id> EVIDENCE_S3=s3://<bucket>/evidence/ python -m t3.trust.job_gate_eval --version v1 --model healthy
    python -m t3.trust.job_gate_eval --version v2 --model degraded --dry-run
    python -m t3.trust.job_gate_eval --version v1 --model healthy --print-declaration --run-id <id>

The four runs behind the headline numbers: (v1, healthy) false pauses, (v2, healthy) false pauses,
(v1, degraded) and (v2, degraded) caught by change 6.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from . import jobs


def twin_version() -> str | None:
    try:
        from twin.presets import config_hash, resolve_config
        from twin.replay import sdk_version

        return f"ran-rl {sdk_version()} config {config_hash(resolve_config('official'))[:12]} (official)"
    except Exception:  # noqa: BLE001 - the declaration still names the twin kind
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", choices=["v1", "v2"], required=True)
    ap.add_argument("--model", choices=["healthy", "degraded"], required=True)
    ap.add_argument("--run-id", default=None, help="default: $RUN_ID or $EVIDENCE_RUN_ID")
    ap.add_argument("--dry-run", action="store_true", help="first two seeds, in memory; writes nothing")
    ap.add_argument("--print-declaration", action="store_true", help="print the declaration and exit")
    ap.add_argument("--declare-first", action="store_true", help="write the declaration via evlib before running")
    a = ap.parse_args(argv)

    testset, testset_sha = jobs.load_testset(jobs.GATE_TESTSET)
    run_id = a.run_id or jobs.env_run_id() or (
        f"dryrun-t3.trust_gate-{a.version}-{a.model}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}" if a.dry_run else None)
    if not run_id:
        sys.exit("set RUN_ID (or EVIDENCE_RUN_ID) or pass --run-id")
    sha = jobs.code_sha_or_unknown()
    decl = jobs.gate_declaration(run_id=run_id, version=a.version, model=a.model, code_sha=sha,
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

    from .gate import load_thresholds, thresholds_path
    from .scripted import load_script

    if load_thresholds(thresholds_path(a.version))[1] != testset["thresholds"][a.version]:
        sys.exit(f"thresholds {a.version} differ from the declared test set; refusing to run")
    changes, script_sha = load_script()
    if script_sha != testset["script_sha256"]:
        sys.exit("demo script differs from the declared test set; refusing to run")
    predictor = (load if a.model == "healthy" else load_degraded)(testset["artifact"])
    seeds = testset["seeds"][:2] if a.dry_run else testset["seeds"]
    out = jobs.run_gate_eval(version=a.version, model=a.model, run_id=run_id, seeds=seeds, twin_factory=ReplayTwin,
                             kpis=headline, predictor=predictor, changes=changes,
                             thresholds_path=thresholds_path(a.version), testset_sha=testset_sha, code_sha=sha)
    return jobs.finish(run_id=run_id, declaration=decl, out=out, dry_run=a.dry_run, declare_first_done=a.declare_first)


if __name__ == "__main__":
    raise SystemExit(main())
