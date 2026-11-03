"""Job entry: pair two score runs row by row and record the comparison (the Track 2 declarations).

    python -m silos.job_compare --kind beats   --x <fed run> --y <local B run> --split disjoint --name federated_vs_B
    python -m silos.job_compare --kind measure --x <fed run> --y <dp run>      --split disjoint --name dp_cost
    python -m silos.job_compare --kind measure --x <central run> --y <fed run> --split disjoint --name distance_to_central
    add --dry-run  or  --print-declaration

diff_i = score_y_i - score_x_i, where score is the per-row scaled MAE (lower is better): a positive diff means
model X is better than model Y on that row. Both score runs are read from the evidence store, matched on item_id,
and refused unless they scored the same shared test (same test_set_sha256) and the same rows.

  beats    declared target: score_gain_ci_low > 0, i.e. the lower bound of the 95 % paired-bootstrap interval of
           the mean difference. The bootstrap resamples whole clusters (seeds), because rows of one cluster are
           correlated; the row-level interval is reported too (details.bootstrap_detail) but overstates precision.
           details.gain records the full replay spec (method paired-bootstrap-v1, seed, resamples, level, pair_key,
           score_field, baseline_run_id null because the items already hold the differences, cluster_field extra.seed,
           numpy_version, value, ci_low, ci_high, paired_n) so tools/check_unit.py part e recomputes the bound from
           the item records; the same spec is recorded for a `measure` run, where the interval is only a note.
  measure  declared target: n_items >= rows (a report, no pass bar); the mean diff and its interval are in the details.

Exit code: 0 whenever the comparison ran and its result was written, INCLUDING a missed `beats` target (passed=false in the
result record, printed as TARGET MISSED); non-zero only for a real error. The job runner attests the evidence of exit-0 jobs only.

Environment: RUN_ID, CODE_SHA, EVIDENCE_S3 (or EVIDENCE_DIR), AWS_REGION.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

from . import evio, nn_core as nn, scorerun as sr

N_BOOT = 2000


def read_scores(run_id: str, store=None) -> tuple[dict, dict]:
    ev = evio.get_evlib()
    decl = ev.read_declared(run_id, store)
    if not decl:
        raise SystemExit(f"no declaration for run {run_id!r}")
    items = {it["item_id"]: it for it in evio.read_records_fast(run_id, "item", store)}
    if not items:
        raise SystemExit(f"run {run_id!r} has no item records")
    return decl, items


def declaration_for(kind: str, name: str, split: str, run_id: str, code_sha: str, xd: dict, yd: dict, n_rows: int, x_run: str, y_run: str) -> dict:
    if kind == "beats":
        target = {"metric": "score_gain_ci_low", "op": ">", "value": 0,
                  "unit": "scaled-MAE units: lower bound of the 95 % paired-bootstrap interval of the mean per-row difference (y minus x; positive = x better)"}
        also = {"claim": "the lower bound of the 95 % paired-bootstrap interval (clusters = seeds resampled) of the mean per-row score difference is above 0; the target IS that bound, replayed by check_unit from the item records with the spec in details.gain",
                "bootstrap": {"method": "paired-bootstrap-v1", "resamples": N_BOOT, "seed": sr.EVAL_SEED, "level": 0.95, "cluster_field": "extra.seed"}}
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
    ap.add_argument("--print-runner-declaration", action="store_true", help="print the declaration in the form `python -m tools.sm.run --declaration` accepts")
    ap.add_argument("--declare-only", action="store_true", help="write the declaration via evlib and exit (the runner does this BEFORE launching the job)")
    ap.add_argument("--declare-first", action="store_true")
    a = ap.parse_args(argv)
    run_id = evio.env_run_id() or ("dryrun-t2-compare-" + a.name if (a.dry_run or a.print_declaration or a.print_runner_declaration) else None)
    if not run_id:
        sys.exit("set RUN_ID (or EVIDENCE_RUN_ID)")
    code_sha = evio.env_code_sha(required=not (a.dry_run or a.print_declaration or a.print_runner_declaration))
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
    if a.print_runner_declaration:
        evio.print_json(evio.runner_declaration(declared))
        return 0
    if a.declare_only:
        evio.get_evlib().write_declared(declared)
        print(f"declared {declared['run_id']} ({declared['unit']})")
        return 0
    if not a.dry_run:
        declared = evio.reconcile(declared)  # inside a runner job: the stored declaration must agree with this code
    if a.declare_first and not a.dry_run:
        evio.declare_first(declared)

    xs = np.array([xi[i]["score"] for i in ids]); ys = np.array([yi[i]["score"] for i in ids])
    seeds = np.array([xi[i]["extra"]["seed"] for i in ids]); segs = np.array([xi[i]["extra"]["segment"] for i in ids])
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
    gain = {"method": "paired-bootstrap-v1", "seed": sr.EVAL_SEED, "resamples": N_BOOT, "level": 0.95, "pair_key": "item_id",
            "score_field": "score", "baseline_run_id": None, "cluster_field": "extra.seed", "subset": None, "numpy_version": np.__version__,
            "value": boot["mean"], "ci_low": boot["ci_lo"], "ci_high": boot["ci_hi"], "paired_n": len(ids)}
    details = {"gain": gain, "bootstrap_detail": boot, "ci_lower_gt0": bool(boot["ci_lo"] > 0), "mean_diff": boot["mean"], "x_mean_score": float(xs.mean()),
               "y_mean_score": float(ys.mean()), "x_wins_fraction": float((diff > 0).mean()), "per_segment": per_seg,
               "x_run": a.x, "y_run": a.y, "x_model": xd["model_id"], "y_model": yd["model_id"], "kind": a.kind, "dry_run": a.dry_run}
    if a.kind == "beats":
        details["verdict"] = bool(boot["ci_lo"] > 0)  # the declared target: the interval's lower bound
    out = evio.finish(declared=declared, items=items, script="silos/job_compare.py", code_sha=code_sha, details=details, dry_run=a.dry_run,
                      sink=evio.Sink(run_id), value_override=boot["ci_lo"] if a.kind == "beats" else None)
    r = out["result"]
    print(json.dumps({"run_id": run_id, "unit": r["unit"], "mean_diff": boot["mean"], "ci95": [boot["ci_lo"], boot["ci_hi"]], "ci_lower_gt0": details["ci_lower_gt0"],
                      "verdict": details.get("verdict"), "passed": r["passed"], "n_items": r["n_items"]}, indent=1))
    if not r["passed"]:  # a declared target that is MISSED is a result, not an error: it is written and attested like any other
        print(f"TARGET MISSED: {r['metric']} = {r['value']:.6g} is not {r['target']['op']} {r['target']['value']} (the result record says passed=false)", file=sys.stderr)
    return 0  # only a real error (an exception, a refused input) exits non-zero; the job runner attests the evidence of exit-0 jobs only


if __name__ == "__main__":
    raise SystemExit(main())
