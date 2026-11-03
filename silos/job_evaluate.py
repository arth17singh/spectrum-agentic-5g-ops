"""Job entry: score a federated model (the FL service's final aggregate, or its DP variant) on the shared test set.

    python -m silos.job_evaluate --label federated --split disjoint --weights s3://<transfer bucket>/.../<final aggregation file>
    python -m silos.job_evaluate --label dp        --split disjoint --weights <local path>
    add --dry-run  or  --print-declaration

Same scoring as the baselines (silos.scorerun), numpy only: no torch, no FL-service package. The weights file is a
safetensors file with the exact tensor names of silos.nn_core.SHAPES; anything else is refused. One invocation = one
evidence run with a `model` record (artifact URI and SHA-256 of the exact bytes scored).

Environment: RUN_ID, CODE_SHA, EVIDENCE_S3, AWS_REGION; DATA_DIR or DATA_COPY_S3. Optional context recorded in the
declaration: --session-id, --iteration-file (the name of the final _aggregation_ file), --epsilon.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import evio, nn_core as nn, scorerun as sr

UNIT_OF = {"federated": "t2.fl_federated", "dp": "t2.fl_dp"}
NOTE = {
    "federated": "Final aggregate of the FL service session (protocol without DP) scored on the shared test set of the simulated silos. A measurement; the comparisons with the local-only models and the central reference are separate declared runs (silos.job_compare).",
    "dp": "Final aggregate of the FL service session with differential privacy (record-level DP-SGD per simulated silo) scored on the shared test set. A measurement; reported relative to the non-DP federated model in a separate run.",
}


def load_weights(src: str) -> bytes:
    if src.startswith("s3://"):
        import boto3

        b, _, k = src[5:].partition("/")
        return boto3.client("s3", region_name=evio.region()).get_object(Bucket=b, Key=k)["Body"].read()
    return Path(src).read_bytes()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", choices=["federated", "dp"], required=True)
    ap.add_argument("--split", choices=list(sr.SPLITS), default="disjoint")
    ap.add_argument("--weights", help="path or s3:// URI of the final aggregate (safetensors)")
    ap.add_argument("--session-id", default=None)
    ap.add_argument("--iteration-file", default=None, help="file name of the highest _aggregation_ file")
    ap.add_argument("--epsilon", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--print-declaration", action="store_true")
    ap.add_argument("--declare-first", action="store_true")
    a = ap.parse_args(argv)
    run_id = evio.env_run_id() or ("dryrun-" + UNIT_OF[a.label].replace(".", "-") if (a.dry_run or a.print_declaration) else None)
    if not run_id:
        sys.exit("set RUN_ID (or EVIDENCE_RUN_ID)")
    code_sha = evio.env_code_sha(required=not (a.dry_run or a.print_declaration))
    manifest = sr.load_manifest()
    model_id = f"t2-response-fl-{a.label}-{a.split}@v1"
    declared = sr.score_declaration(
        run_id=run_id, unit=f"{UNIT_OF[a.label]}.{a.split}", model_id=model_id, n_rows=sr.test_rows(manifest, a.split), test_sha=sr.test_sha(manifest, a.split),
        code_sha=code_sha, note=NOTE[a.label],
        extra={"split": a.split, "session_id": a.session_id, "final_aggregation_file": a.iteration_file, "epsilon": a.epsilon,
               "entry": f"python -m silos.job_evaluate --label {a.label} --split {a.split}"})
    if a.print_declaration:
        evio.print_json(declared)
        return 0
    if not a.weights:
        sys.exit("--weights is required")
    if a.declare_first and not a.dry_run:
        evio.declare_first(declared)

    blob = load_weights(a.weights)
    params = nn.safetensors_loads(blob)
    nn.check_params(params)  # exact names and shapes, or refuse
    root = sr.fetch([f"{a.split}/shared_test/test.csv"], manifest)
    test = nn.read_xy(root / a.split / "shared_test" / "test.csv")
    if a.dry_run:
        test = {k: v[:300] for k, v in test.items()}
    pred = nn.forward_np(params, test["x"])
    items = sr.score_items(test, pred, model_id)
    if a.dry_run:
        declared["target"]["value"] = declared["n_items"] = len(items)
    sha = nn.sha256_bytes(blob)
    summ = sr.summarize(test, pred)
    details = {**summ, "split": a.split, "artifact_uri": a.weights, "artifact_sha256": sha, "session_id": a.session_id,
               "final_aggregation_file": a.iteration_file, "epsilon": a.epsilon, "dry_run": a.dry_run}
    model_rec = {"model_id": model_id.split("@")[0], "version": "v1", "model_kind": "t2-response", "artifact_uri": a.weights, "artifact_sha256": sha,
                 "status": "candidate", "trained_on": f"FL service session {a.session_id or '?'} ({a.label}), simulated silos A+B+C, split {a.split}",
                 "metrics": {"mean_score": summ["mean_score"], "mean_score_noop": summ["mean_score_noop"]}}
    out = evio.finish(declared=declared, items=items, script="silos/job_evaluate.py", code_sha=code_sha, details=details, dry_run=a.dry_run, sink=evio.Sink(run_id))
    if not a.dry_run:
        evio.get_evlib().write_model({**model_rec, "run_id": run_id})
    r = out["result"]
    print(json.dumps({"run_id": run_id, "unit": r["unit"], "mean_score": summ["mean_score"], "no_op": summ["mean_score_noop"], "mae": summ["mae"],
                      "direction_accuracy": summ["direction_accuracy"], "n_items": r["n_items"], "passed": r["passed"]}, indent=1))
    return 0 if r["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
