"""Job entry: pair two score runs row by row and record the comparison (the Track 2 declarations).

    python -m silos.job_compare --kind beats   --x <fed run> --y <local B run> --split disjoint --name federated_vs_B
    python -m silos.job_compare --kind measure --x <fed run> --y <dp run>      --split disjoint --name dp_cost
    python -m silos.job_compare --kind measure --x <central run> --y <fed run> --split disjoint --name distance_to_central
    add --dry-run  or  --print-declaration

diff_i = score_y_i - score_x_i, where score is the per-row scaled MAE (lower is better): a positive diff means
model X is better than model Y on that row. Both score runs are read from the evidence store, matched on item_id,
and refused unless they scored the same shared test (same test_set_sha256) and the same rows.

  beats    declared target: mean_score of the diffs > 0 (check_unit recomputes it), AND the 95 % paired-bootstrap
           interval lower bound > 0. The bootstrap resamples whole clusters (seeds), because rows of one cluster
           are correlated; the row-level interval is reported too but overstates precision. The verdict in the
           result details is the AND of both conditions.
  measure  declared target: n_items >= rows (a report, no pass bar); the mean diff and its interval are in the details.

Environment: RUN_ID, CODE_SHA, EVIDENCE_S3 (or EVIDENCE_DIR), AWS_REGION.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys

import numpy as np

from . import evio, nn_core as nn, scorerun as sr

N_BOOT = 2000


def read_scores(run_id: str, store=None) -> tuple[dict, dict]:
    ev = evio.get_evlib()
    decl = ev.read_declared(run_id, store)
    if not decl:
        raise SystemExit(f"no declaration for run {run_id!r}")
    items = {it["item_id"]: it for it in ev.read_records(run_id, "item", store)}
    if not items:
        raise SystemExit(f"run {run_id!r} has no item records")
    return decl, items


def declaration_for(kind: str, name: str, split: str, run_id: str, code_sha: str, xd: dict, yd: dict, n_rows: int, x_run: str, y_run: str) -> dict:
    if kind == "beats":
        target = {"metric": "mean_score", "op": ">", "value": 0, "unit": "scaled-MAE units (y minus x; positive = x better)"}
        also = {"declared_also": "95 % paired-bootstrap interval lower bound of the mean difference > 0 (clusters resampled; details.ci_lower_gt0). The verdict in the result details is both conditions.",
                "bootstrap": {"n_boot": N_BOOT, "seed": sr.EVAL_SEED, "unit": "cluster (seed)"}}
    else:
        target = {"metric": "n_items", "op": ">=", "value": n_rows, "unit": "paired shared-test rows compared (a report; no pass bar)"}
        also = {}
    return evio.declaration(
        run_id=run_id, unit=f"t2.compare.{name}.{split}", target=target, n_items=n_rows, test_set_sha256=xd["test_set_sha256"],
        model_id=f"{xd['model_id']} vs {yd['model_id']}", grader="silos.nn_core per-row scaled MAE, paired by item_id", code_sha=code_sha,
        extra={"x_run": x_run, "y_run": y_run, "comparison": kind, "split": split, "score_scale": nn.SCORE_SCALE.tolist(), **also,
               "entry": f"python -m silos.job_compare --kind {kind} --name {name} --split {split}"})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=["beats", "measure"], required=True)
    ap.add_argument("--x", required=True, help="run id of model X (the one declared better, for 'beats')")
    ap.add_argument("--y", required=True, help="run id of model Y")
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", choices=list(sr.SPLITS), default="disjoint")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--print-declaration", action="store_true")
    ap.add_argument("--declare-only", action="store_true", help="write the declaration via evlib and exit (the runner does this BEFORE launching the job)")
    ap.add_argument("--declare-first", action="store_true")
    a = ap.parse_args(argv)
    run_id = evio.env_run_id() or ("dryrun-t2-compare-" + a.name if (a.dry_run or a.print_declaration) else None)
    if not run_id:
        sys.exit("set RUN_ID (or EVIDENCE_RUN_ID)")
    code_sha = evio.env_code_sha(required=not (a.dry_run or a.print_declaration))
    xd, xi = read_scores(a.x)
    yd, yi = read_scores(a.y)
    if xd["test_set_sha256"] != yd["test_set_sha256"]:
        sys.exit("the two runs scored different test sets (test_set_sha256 differs); refusing to compare")
    if set(xi) != set(yi):
        sys.exit(f"the two runs scored different rows ({len(set(xi) ^ set(yi))} differ); refusing to compare")
    ids = sorted(xi)
    declared = declaration_for(a.kind, a.name, a.split, run_id, code_sha, xd, yd, len(ids), a.x, a.y)
    if a.print_declaration:
        evio.print_json(declared)
        return 0
    if a.declare_only:
        evio.get_evlib().write_declared(declared)
        print(f"declared {declared['run_id']} ({declared['unit']})")
        return 0
    if a.declare_first and not a.dry_run:
        evio.declare_first(declared)

    xs = np.array([xi[i]["score"] for i in ids]); ys = np.array([yi[i]["score"] for i in ids])
    seeds = np.array([xi[i]["seed"] for i in ids]); segs = np.array([xi[i]["segment"] for i in ids])
    diff = ys - xs
    boot = nn.paired_bootstrap(diff, seeds, n_boot=N_BOOT, seed=sr.EVAL_SEED)
    items = []
    for k, i in enumerate(ids):
        items.append(evio.item(
            item_id=i, input_sha256=hashlib.sha256((xi[i]["input_sha256"] + yi[i]["input_sha256"]).encode()).hexdigest(),
            output_sha256=hashlib.sha256(f"{xs[k]!r}|{ys[k]!r}".encode()).hexdigest(), grader=declared["grader"],
            backend="numpy on CPU, from two score runs", score=float(diff[k]), x_score=float(xs[k]), y_score=float(ys[k]),
            segment=str(segs[k]), seed=int(seeds[k])))
    per_seg = {s: {"n": int((segs == s).sum()), "mean_diff": float(diff[segs == s].mean())} for s in sorted(set(segs))}
    details = {"bootstrap": boot, "ci_lower_gt0": bool(boot["ci_lo"] > 0), "mean_diff": boot["mean"], "x_mean_score": float(xs.mean()),
               "y_mean_score": float(ys.mean()), "x_wins_fraction": float((diff > 0).mean()), "per_segment": per_seg,
               "x_run": a.x, "y_run": a.y, "x_model": xd["model_id"], "y_model": yd["model_id"], "kind": a.kind, "dry_run": a.dry_run}
    if a.kind == "beats":
        details["verdict"] = bool(boot["mean"] > 0 and boot["ci_lo"] > 0)
    out = evio.finish(declared=declared, items=items, script="silos/job_compare.py", code_sha=code_sha, details=details, dry_run=a.dry_run, sink=evio.Sink(run_id))
    r = out["result"]
    print(json.dumps({"run_id": run_id, "unit": r["unit"], "mean_diff": boot["mean"], "ci95": [boot["ci_lo"], boot["ci_hi"]], "ci_lower_gt0": details["ci_lower_gt0"],
                      "verdict": details.get("verdict"), "passed": r["passed"], "n_items": r["n_items"]}, indent=1))
    return 0 if (r["passed"] and details.get("verdict", True)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
