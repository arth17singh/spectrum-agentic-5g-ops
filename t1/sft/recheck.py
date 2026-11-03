"""Re-check an ALREADY-BUILT SFT dataset for overlap with the eval items at the launch commit.

Why: dataset t1.sft.data-20261103T034500Z-7c2a was built and checked against the eval items pinned at
f1d3deb. W12 then re-froze execution.jsonl and price_latency.jsonl on x86 (482fe5f: same item ids, splits,
scenarios, seeds and prefixes; 43 items' state text differs in the last digit). This job re-runs the same
four-way check (t1/sft/overlap.py, t1-sft-overlap-v1) of the dataset's written rows against the NEW items.
It changes nothing: it reads the dataset and writes evidence only.

CPU job through the team runner (ml.m5.xlarge, a few minutes; no SDK). The dataset arrives as channel `data`:

    --entry "-m t1.sft.recheck --run-id {run_id} --data {channel:data} --data-run-id D
             --evidence-dir {evidence_dir} --summary-dir {output_dir}"

In order (any failed precondition stops the job before evidence is written):
  1. the declaration the launcher stored; the items' MANIFEST.json is its test set, and every item file
     matches t1/sft/build.py EVAL_PINS;
  2. the dataset matches its own SHA256SUMS, and that file's sha256 is the one the declaration names; its
     manifest.json is for --data-run-id;
  3. one item record per eval item (all 1,220, dev and test): passed when no dataset row matches it by
     normalised item hash, answer-free input hash, exact prompt hash or (scenario, seed, step). Declared
     target fail_count == 0, which check_unit recomputes. The row-side view (rows that hit any eval item,
     per hash, and duplicates inside the set) goes in the result details.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from t1.sft import common as K

UNIT = "t1.sft.data-recheck"


def run(a: argparse.Namespace) -> dict[str, Any]:
    from t1.sft import build, overlap

    t0 = time.time()
    run_id = K.check_run_id(a.run_id)
    code_sha = K.code_sha_from_env(a.code_sha)
    decl = K.stored_declaration(run_id, a.declared)
    items_dir = Path(a.items_dir or build.items_dir())
    K.check_test_set(decl, K.sha256_file(items_dir / "MANIFEST.json"), "t1/evals/items/MANIFEST.json")
    idx = overlap.build_eval_index(items_dir)  # every item file must match MANIFEST.json
    actual = {**idx.files, "MANIFEST.json": idx.manifest_sha256}
    off = sorted(n for n, sha in build.EVAL_PINS.items() if actual.get(n) != sha)
    if off:
        raise K.JobError(f"eval item files differ from EVAL_PINS ({build.EVAL_PINS_SOURCE}): {off}")

    work = Path(a.workdir or tempfile.mkdtemp(prefix="sft-recheck-"))
    data_dir, _ = K.materialise(a.data, work, "data", channel_env="SM_CHANNEL_DATA", region=a.region)
    sums_sha, listed = K.verify_sums(data_dir)
    want = ((decl.get("decoding") or {}).get("dataset") or {}).get("sha256sums_sha256")
    if want and want != sums_sha:
        raise K.JobError(f"dataset SHA256SUMS sha256 {sums_sha} is not the declared {want}")
    if not {"train.jsonl", "dev.jsonl", "manifest.json"} <= set(listed):
        raise K.JobError(f"dataset SHA256SUMS lists {sorted(listed)}; train.jsonl, dev.jsonl, manifest.json expected")
    manifest = json.loads((data_dir / "manifest.json").read_text())
    if manifest.get("run_id") != a.data_run_id:
        raise K.JobError(f"the dataset's manifest is for run {manifest.get('run_id')!r}, not {a.data_run_id!r}")
    K.log(f"dataset {a.data_run_id}: SHA256SUMS sha256 {sums_sha}; eval items {idx.manifest_sha256}")

    rows = overlap.read_jsonl(data_dir / "train.jsonl") + overlap.read_jsonl(data_dir / "dev.jsonl")
    row_check = overlap.check_disjoint(rows, idx)
    backend = f"{os.environ.get('SM_CURRENT_INSTANCE_TYPE', 'local')}; python hash sets over the dataset rows"
    items = overlap.eval_item_records(rows, idx, backend)
    failed = [it["item_id"] for it in items if not it["passed"]]
    details = {
        "dataset_run_id": a.data_run_id, "dataset_sha256sums_sha256": sums_sha,
        "dataset_manifest_sha256": listed.get("manifest.json"), "rows_checked": row_check["rows_checked"],
        "rows_overlapping_eval": row_check["eval_overlap_count"], "rows_hits_by_hash": row_check["hits_by_hash"],
        "within_set_duplicates": row_check["within_set_duplicates"], "max_execution_seed": row_check["max_execution_seed"],
        "hash_method": overlap.HASH_METHOD, "eval_items_manifest_sha256": idx.manifest_sha256,
        "eval_item_files_sha256": idx.files, "eval_pins_source": build.EVAL_PINS_SOURCE,
        "built_against": (manifest.get("eval_contract") or {}).get("files_sha256"),
        "eval_items_overlapped": failed[:50],
    }
    res = K.write_evidence(a.evidence_dir, decl, items, "t1/sft/recheck.py", code_sha, details)
    K.log(f"result: {res['metric']} = {res['value']} over {res['n_items']} eval items; rows hitting the eval "
          f"{row_check['eval_overlap_count']} of {row_check['rows_checked']} -> {'PASSED' if res['passed'] else 'MISSED'}")
    summary = {"kind": "t1-sft-recheck-summary", "run_id": run_id, "code_sha": code_sha, "details": details,
               "result": {k: res[k] for k in ("metric", "value", "passed", "n_items")},
               "seconds": round(time.time() - t0, 1)}
    if a.summary_dir:
        K.write_json(Path(a.summary_dir) / "recheck_summary.json", summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="python -m t1.sft.recheck", description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--data", help="the dataset: {channel:data}, a dir, or an s3:// prefix (else SM_CHANNEL_DATA)")
    ap.add_argument("--data-run-id", required=True, help="the dataset's build run id (its manifest must say so)")
    ap.add_argument("--evidence-dir", required=True, help="local evlib store for items + result ({evidence_dir})")
    ap.add_argument("--declared", help="the run's declaration as a JSON file (default: read from EVIDENCE_S3)")
    ap.add_argument("--summary-dir", help="also write recheck_summary.json here ({output_dir})")
    ap.add_argument("--items-dir", help="eval items (default t1/evals/items of the code being run)")
    ap.add_argument("--code-sha")
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", K.DEFAULT_REGION))
    ap.add_argument("--workdir")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    try:
        out = run(a)
    except K.JobError as e:
        K.log(f"FAILED: {e}")
        return 2
    print(K.dumps_doc(out["result"]), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
