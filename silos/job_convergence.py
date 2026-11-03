"""Job entry: the measured convergence curve of a federated session (Track 2): error against round.

    python -m silos.job_convergence --split disjoint --rounds 2:21 --rounds-dir <dir of the per-round aggregates> \\
        --published <the published final aggregate> --published-run <the t2.fl_federated run> \\
        --local-only A=<run> --local-only B=<run> --local-only C=<run> --central <run> --fedavg <run>
    add --dry-run  or  --print-declaration  or  --print-runner-declaration

What it does, for each round k in the range:
  * reads the FL service's global aggregate of that round (the file `...-<client>-<k>-global.safetensor`, one copy per client)
    and refuses to call the round consistent unless every client's copy holds identical tensors (the files differ only in
    their per-client header metadata, which is cross-checked against the file name: round index, global-not-local, session);
  * loads it exactly the way silos.job_evaluate loads the published final model (nn.safetensors_loads + nn.check_params; the
    publish step is a byte copy, so there is no other conversion) and scores it with the very same code (silos.scorerun:
    nn.forward_np on the shared test set of the split, scorerun.summarize);
  * the error of silo S in round k is the mean per-row scaled MAE on S's own test segment (silo A = segment region_A, ...),
    the `per_segment.region_S.mean_score` of the evaluation run; one item record per (round, silo) carries it with n.

The last round is checked against the published model (the evidence run of silos.job_evaluate): the tensors must be identical and
the recomputed per-segment scores must equal the ones stored in that run's result. The reference lines (local-only, central,
local FedAvg) are QUOTED from their evidence results, never recomputed.

Declared target: pass_count == rounds x silos. An item passes when its score is finite on a non-empty segment from a round whose
client copies held identical tensors, and, for the last round, only when the check against the published model holds; so check_unit,
which recomputes pass_count from the item records, proves both the curve and the final-round consistency.

Environment: RUN_ID, CODE_SHA, EVIDENCE_S3 (or EVIDENCE_DIR), AWS_REGION; DATA_DIR or DATA_COPY_S3 (--data-dir for the runner).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import struct
import sys
from pathlib import Path

import numpy as np

from . import evio, nn_core as nn, scorerun as sr

UNIT = "t2.fl_convergence"
SILO_SEGMENT = {"A": "region_A", "B": "region_B", "C": "region_C"}
SILOS = tuple(SILO_SEGMENT)
SCORE_TOL = 1e-6  # the last round's recomputed scores vs the stored ones of the published run; the observed difference is recorded
# <prefix>-<session>-<member>-<client>-<iteration>-global.safetensor(s): the service's own file name, matched by pattern
ROUND_FILE = re.compile(r"^(?P<prefix>[A-Za-z0-9]+)-(?P<session>[A-Za-z0-9]+)-(?P<member>[A-Za-z0-9]+)-(?P<client>[A-Za-z0-9]+)-(?P<iter>\d+)-global\.safetensors?$")
NOTE = ("Error against round of a federated session: the global aggregate of every round of the FL service session, scored with the "
        "scorer of the published final model on each simulated silo's own test segment of the shared test set. A measurement "
        "(there is no pass bar on the error); the declared target only proves that every point of the curve exists, comes from "
        "client copies with identical tensors, and that the last round is the published model.")
DEFINITIONS = {
    "metric": "scaled_mae: mean over test rows of mean over the 4 labels of |prediction - truth| / score_scale (silos.nn_core.row_scores); lower is better",
    "silo_segment": SILO_SEGMENT,
    "mean": "unweighted mean of A, B and C (the three segments have equal n, so it equals the score on the three regions pooled)",
    "round": "the iteration index in the service's file name; iteration 1 is the initial weights (not scored), and the aggregate with index k is the result of aggregating the clients' iteration k-1 updates",
    "reference": "each reference value is the model's score on the silo's own test segment, quoted from its evidence result (local_only.S = the model trained on silo S alone, scored on region_S)",
}


def parse_rounds(text: str) -> tuple[int, int]:
    m = re.fullmatch(r"(\d+):(\d+)", text.strip())
    if not m or int(m.group(1)) > int(m.group(2)) or int(m.group(1)) < 1:
        raise SystemExit(f"--rounds must look like 2:21 (first:last iteration index), got {text!r}")
    return int(m.group(1)), int(m.group(2))


def discover(root: str | Path) -> dict:
    """Find the per-round aggregates anywhere under `root` (the service nests them in directories); parse names, never trust paths."""
    by_round: dict[int, dict[str, Path]] = {}
    sessions, members, clients = set(), set(), set()
    for p in sorted(Path(root).rglob("*")):
        m = ROUND_FILE.match(p.name)
        if not (m and p.is_file()):
            continue
        k, c = int(m["iter"]), m["client"]
        if c in by_round.setdefault(k, {}):
            raise SystemExit(f"two files for round {k} of client {c}: {by_round[k][c]} and {p}")
        by_round[k][c] = p
        sessions.add(m["session"]); members.add(m["member"]); clients.add(c)
    if not by_round:
        raise SystemExit(f"no '...-<client>-<round>-global.safetensor' file under {root}")
    if len(sessions) != 1:
        raise SystemExit(f"the aggregates belong to {len(sessions)} sessions ({sorted(sessions)}); expected exactly one")
    return {"by_round": by_round, "session_id": next(iter(sessions)), "members": sorted(members), "clients": sorted(clients)}


def read_run(ev, store, run_id: str, test_sha: str) -> tuple[dict, dict]:
    decl, res = ev.read_declared(run_id, store), ev.read_result(run_id, store)
    if not decl or not res:
        raise SystemExit(f"run {run_id!r} has no declaration or no result in the evidence store")
    if decl.get("test_set_sha256") != test_sha:
        raise SystemExit(f"run {run_id!r} scored another test set (test_set_sha256 differs from this declaration's); refusing to quote it")
    if not res.get("passed", False):
        raise SystemExit(f"run {run_id!r} did not pass its own declaration; refusing to quote it")
    return decl, res


def segment_scores(res: dict) -> dict[str, float]:
    ps = (res.get("details") or {}).get("per_segment") or {}
    out = {}
    for silo, seg in SILO_SEGMENT.items():
        if seg not in ps or not isinstance(ps[seg].get("mean_score"), (int, float)):
            raise SystemExit(f"run {res.get('run_id')!r} has no per_segment.{seg}.mean_score")
        out[silo] = float(ps[seg]["mean_score"])
    return out


def dp_of(pub_decl: dict) -> str | None:
    """'none' only if the published run's declaration is the no-DP evaluation (unit t2.fl_federated) with epsilon declared null."""
    det = ((pub_decl.get("twin") or {}).get("config") or {}).get("details") or {}
    if str(pub_decl.get("unit", "")).startswith("t2.fl_federated.") and "epsilon" in det and det["epsilon"] is None:
        return "none"
    return None


def with_mean(scores: dict[str, float]) -> dict[str, float]:
    return {**{s: scores[s] for s in SILOS}, "mean": float(np.mean([scores[s] for s in SILOS]))}


def build_reference(ev, store, test_sha: str, local: dict[str, str], central: str, fedavg: str, published: dict) -> dict:
    if set(local) != set(SILOS):
        raise SystemExit(f"--local-only needs one run per silo {list(SILOS)}, got {sorted(local)}")
    matrix = {}
    for s, rid in sorted(local.items()):
        decl, res = read_run(ev, store, rid, test_sha)
        if (res.get("details") or {}).get("silo") not in (None, s):
            raise SystemExit(f"run {rid!r} is silo {res['details'].get('silo')!r}, not {s!r}")
        matrix[s] = segment_scores(res)
    ref = {"local_only": {**with_mean({s: matrix[s][s] for s in SILOS}), "run_ids": dict(sorted(local.items())),
                          "cross_segment": {s: {t: matrix[s][t] for t in SILOS} for s in SILOS}}}
    for key, rid in (("central", central), ("fedavg_local", fedavg)):
        _, res = read_run(ev, store, rid, test_sha)
        ref[key] = {**with_mean(segment_scores(res)), "run_ids": {"all": rid}}
    noop = {s: float(published["details"]["per_segment"][SILO_SEGMENT[s]]["mean_score_noop"]) for s in SILOS}
    ref["no_op"] = {**with_mean(noop), "run_ids": {"all": published["run_id"]}}
    return ref


def declaration_for(run_id: str, code_sha: str, split: str, first: int, last: int, manifest: dict, a_refs: dict) -> dict:
    n = (last - first + 1) * len(SILOS)
    return evio.declaration(
        run_id=run_id, unit=f"{UNIT}.{split}",
        target={"metric": "pass_count", "op": "==", "value": n,
                "unit": "points of the curve (round x silo) scored from a round whose client copies hold identical tensors; the last round counts only if its tensors equal the published model's and it reproduces its stored per-segment scores"},
        n_items=n, test_set_sha256=sr.test_sha(manifest, split), model_id=f"t2-response-fl-rounds-{split}@v1", grader=sr.GRADER, code_sha=code_sha,
        extra={"note": NOTE, "split": split, "rounds": [first, last], "silos": list(SILOS), "definitions": DEFINITIONS, "score_scale": nn.SCORE_SCALE.tolist(),
               "published_run": a_refs["published_run"], "reference_runs": a_refs["reference_runs"],
               "entry": f"python -m silos.job_convergence --split {split}"})


def score_params(params: dict, test: dict) -> dict:
    """Exactly job_evaluate's scoring of one parameter set: forward pass, then scorerun.summarize."""
    return sr.summarize(test, nn.forward_np(params, test["x"]))


def per_silo(summ: dict) -> dict[str, dict]:
    return {s: summ["per_segment"][SILO_SEGMENT[s]] for s in SILOS}


def header_metadata(blob: bytes) -> dict:
    """The safetensors header's __metadata__ (a dict of strings), empty if there is none."""
    (n,) = struct.unpack("<Q", blob[:8])
    return json.loads(blob[8 : 8 + n]).get("__metadata__") or {}


def check_metadata(blob: bytes, k: int, session: str, path) -> None:
    """Cross-check the service's per-file metadata (keys matched by suffix) against the name the file was found under."""
    for key, v in header_metadata(blob).items():
        if key.endswith("_iteration") and str(v) != str(k):
            raise SystemExit(f"{path}: its metadata says iteration {v} but its name says {k}")
        if key.endswith("_session_id") and v != session:
            raise SystemExit(f"{path}: its metadata names another session")
        if key.endswith("_is_local_model") and str(v).lower() == "true":
            raise SystemExit(f"{path}: its metadata says it is a client's LOCAL model, not a global aggregate")


def tensors_sha256(params: dict) -> str:
    """sha256 of the canonical (metadata-free, deterministic) safetensors encoding of the tensors."""
    return nn.sha256_bytes(nn.safetensors_dumps(params))


def load_params(blob: bytes) -> dict:
    params = nn.safetensors_loads(blob)
    nn.check_params(params)  # exact names and shapes, or refuse
    return params


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=list(sr.SPLITS), default="disjoint")
    ap.add_argument("--rounds", required=True, help="first:last iteration index of the aggregates to score, e.g. 2:21")
    ap.add_argument("--rounds-dir", help="directory holding the per-round aggregates (searched recursively; the runner passes {channel:<name>}/rounds)")
    ap.add_argument("--published", help="the published final aggregate (the file silos.job_evaluate scored)")
    ap.add_argument("--published-run", required=True, help="run id of the silos.job_evaluate run that scored the published aggregate")
    ap.add_argument("--local-only", action="append", default=[], metavar="S=RUN", help="the local-only run of silo S (A, B and C)")
    ap.add_argument("--central", required=True)
    ap.add_argument("--fedavg", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--print-declaration", action="store_true")
    ap.add_argument("--print-runner-declaration", action="store_true", help="print the declaration in the form `python -m tools.sm.run --declaration` accepts")
    ap.add_argument("--data-dir", default=None, help="local data tree (with its manifest.json); the runner passes {channel:data}")
    ap.add_argument("--declare-only", action="store_true", help="write the declaration via evlib and exit (the runner does this BEFORE launching the job)")
    ap.add_argument("--declare-first", action="store_true")
    a = ap.parse_args(argv)
    if a.data_dir:
        os.environ["DATA_DIR"] = a.data_dir
    printing = a.dry_run or a.print_declaration or a.print_runner_declaration
    run_id = evio.env_run_id() or ("dryrun-t2-fl-convergence" if printing else None)
    if not run_id:
        sys.exit("set RUN_ID (or EVIDENCE_RUN_ID)")
    code_sha = evio.env_code_sha(required=not printing)
    first, last = parse_rounds(a.rounds)
    local = {}
    for kv in a.local_only:
        s, _, rid = kv.partition("=")
        if s not in SILO_SEGMENT or not rid or s in local:
            raise SystemExit(f"--local-only expects A=<run>, B=<run>, C=<run> once each, got {kv!r}")
        local[s] = rid
    if set(local) != set(SILOS):
        raise SystemExit("--local-only is needed for A, B and C")
    manifest = sr.load_manifest()
    declared = declaration_for(run_id, code_sha, a.split, first, last, manifest, {
        "published_run": a.published_run, "reference_runs": {"local_only": dict(sorted(local.items())), "central": a.central, "fedavg_local": a.fedavg}})
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
    if not (a.rounds_dir and a.published):
        sys.exit("--rounds-dir and --published are required")
    if not a.dry_run:
        declared = evio.reconcile(declared)  # inside a runner job: the stored declaration must agree with this code
    if a.declare_first and not a.dry_run:
        evio.declare_first(declared)

    ev = evio.get_evlib()
    store = ev.default_store()
    test_sha = sr.test_sha(manifest, a.split)
    pub_decl, pub_res = read_run(ev, store, a.published_run, test_sha)
    reference = build_reference(ev, store, test_sha, local, a.central, a.fedavg, pub_res)

    found = discover(a.rounds_dir)
    wanted = list(range(first, last + 1))
    missing = [k for k in wanted if k not in found["by_round"]]
    if missing:
        sys.exit(f"missing rounds {missing}: the aggregates found are {sorted(found['by_round'])}")
    n_clients = len(found["clients"])
    short = {k: sorted(found["by_round"][k]) for k in wanted if len(found["by_round"][k]) != n_clients}
    if short:
        sys.exit(f"rounds with fewer client copies than the {n_clients} clients seen: {short}")

    root = sr.fetch([f"{a.split}/shared_test/test.csv"], manifest)
    test = nn.read_xy(root / a.split / "shared_test" / "test.csv")
    if a.dry_run:
        test = {k: v[:300] for k, v in test.items()}

    # the published model, scored by the same function; its bytes must be the file the published run recorded
    pub_blob = Path(a.published).read_bytes()
    pub_sha = nn.sha256_bytes(pub_blob)
    recorded = (pub_res.get("details") or {}).get("artifact_sha256")
    if recorded and recorded != pub_sha:
        sys.exit(f"--published has sha256 {pub_sha[:12]} but run {a.published_run} scored {recorded[:12]}; wrong file")
    pub_params = load_params(pub_blob)

    rounds_out, curve, shared, items, details_rounds = [], {s: [] for s in SILOS}, [], [], []
    last_blob = last_params = None
    for k in wanted:
        copies = {c: p.read_bytes() for c, p in sorted(found["by_round"][k].items())}
        loaded = {}
        for c, b in copies.items():
            check_metadata(b, k, found["session_id"], found["by_round"][k][c])
            loaded[c] = load_params(b)
        tsha = {c: tensors_sha256(p) for c, p in loaded.items()}
        identical = len(set(tsha.values())) == 1
        use = sorted(copies)[0]
        blob, params, sha = copies[use], loaded[use], tsha[use]
        summ = score_params(params, test)
        seg = per_silo(summ)
        rounds_out.append(k)
        shared.append(summ["mean_score"])
        details_rounds.append({"round": k, "tensors_sha256": sha, "clients_identical": identical, "files": len(copies),
                               "files_byte_identical": len({nn.sha256_bytes(b) for b in copies.values()}) == 1})
        for s in SILOS:
            curve[s].append(seg[s]["mean_score"])
            items.append((k, s, seg[s], sha, identical))
        if k == last:
            last_blob, last_params, last_summ = blob, params, summ
    curve["mean"] = [float(np.mean([curve[s][i] for s in SILOS])) for i in range(len(rounds_out))]

    # final-round check against the published model: tensors first, then the scorer
    max_abs_diff = max(float(np.abs(last_params[n].astype(np.float64) - pub_params[n].astype(np.float64)).max()) for n in nn.SHAPES)
    stored = segment_scores(pub_res)
    recomputed = {s: per_silo(last_summ)[s]["mean_score"] for s in SILOS}
    score_diff = max([abs(recomputed[s] - stored[s]) for s in SILOS] + [abs(last_summ["mean_score"] - float((pub_res["details"] or {}).get("mean_score", math.nan)))])
    sha_equal = tensors_sha256(last_params) == tensors_sha256(pub_params)
    # a dry run scores a slice of the test set, which cannot reproduce the stored full-set scores: only the tensors are compared then
    score_ok = True if a.dry_run else bool(score_diff <= SCORE_TOL)
    matches = bool(sha_equal and max_abs_diff == 0.0 and score_ok)
    final_check = {"round": last, "matches_published": matches, "published_run_id": a.published_run, "max_abs_diff": max_abs_diff,
                   "tensors_equal": sha_equal, "round_tensors_sha256": tensors_sha256(last_params), "published_tensors_sha256": tensors_sha256(pub_params),
                   "file_sha256_equal": nn.sha256_bytes(last_blob) == pub_sha, "published_file_sha256": pub_sha,
                   "score_max_abs_diff": None if a.dry_run else score_diff, "score_tolerance": SCORE_TOL,
                   "compared": "per-segment region_A/B/C mean_score and the shared-test mean_score of the published run's result"}

    out_items = []
    for k, s, segm, sha, identical in items:
        ok = bool(identical and segm["n"] > 0 and math.isfinite(segm["mean_score"]) and (matches if k == last else True))
        out_items.append(evio.item(
            item_id=f"round{k:02d}:{s}", input_sha256=test_sha, output_sha256=sha, grader=sr.GRADER, backend=sr.BACKEND,
            score=float(segm["mean_score"]), passed=ok, round=k, silo=s, segment=SILO_SEGMENT[s], mae=float(segm["mean_score"]), n=int(segm["n"]),
            mae_by_label=segm["mae"], clients_identical=bool(identical)))

    session = {
        "session_id_short": found["session_id"][:8], "rounds_completed": len(found["by_round"]), "clients": n_clients,
        "secure_aggregation": None,
        "dp": dp_of(pub_decl),
        "sources": {
            "session_id_short": "the file names of the aggregates", "rounds_completed": "distinct round indices among the aggregates (the last round is counted, the initial weights are not)",
            "clients": "distinct client ids among the aggregates' file names",
            "secure_aggregation": "null: nothing in the aggregates or in the evidence records says how the updates were combined, so it is not asserted here",
            "dp": f"the declaration of {a.published_run}: unit {pub_decl.get('unit')}, epsilon null",
        },
    }
    details = {
        "metric": "scaled_mae", "lower_is_better": True, "rounds": rounds_out,
        "curve": curve, "reference": reference, "final_round_check": final_check, "session": session,
        "split": a.split, "silos": list(SILOS), "segments": SILO_SEGMENT, "n_per_silo": {s: int(per_silo(last_summ)[s]["n"]) for s in SILOS},
        "shared_test_mean": shared, "per_round": details_rounds, "definitions": DEFINITIONS, "dry_run": a.dry_run,
    }
    out = evio.finish(declared=declared, items=out_items, script="silos/job_convergence.py", code_sha=code_sha, details=details, dry_run=a.dry_run, sink=evio.Sink(run_id))
    r = out["result"]
    print(json.dumps({"run_id": run_id, "unit": r["unit"], "rounds": [first, last], "mean_first": curve["mean"][0], "mean_last": curve["mean"][-1],
                      "final_round_matches_published": matches, "n_items": r["n_items"], "pass_count": r["value"], "passed": r["passed"]}, indent=1))
    return 0 if r["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
