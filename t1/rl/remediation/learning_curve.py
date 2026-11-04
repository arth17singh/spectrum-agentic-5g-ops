"""Learning curve for the remediation policy: is it data-limited? One declared, descriptive evidence unit.

Four models trained on 10%, 25%, 50% and 100% of remediation-v1's training incidents, each evaluated on the
frozen held-out set (t1/rl/remediation/suite/heldout-v2.jsonl, all 1,008 incidents) with the evaluation's
own measures (t1/rl/remediation/evaluate.py play: network recovery, percell-v1, diagnosis, change size).

Procedure: behaviour-cloning ROUND 0 only (bc.fit with bc.HYPER's round-0 settings and seed) on the
twin-labelled states of the chosen train incidents; validation on every val incident's twin-labelled states.
remediation-v1 itself adds two DAgger rounds. They are left out here on purpose and declared: DAgger states
depend on the policy being trained (so they confound data size with the learner's own mistakes), and the full
procedure at four sizes (data regeneration and two rollout rounds each) does not fit the 20-minute budget.
The 100% point is therefore v1's round-0 model; the job checks its weights against the training job's
rounds/round0.zip.

Data: dataset.npz from v1's training job output (sha256 checked against
t1/rl/artifacts/remediation_v1.provenance.json). Train incidents are ordered by sha256("lc-v1:<seed>") and
each fraction takes the first ceil(f x n) of them, so the subsets are nested (10% in 25% in 50% in 100%).

Decision rule (declared before the run): item score = percell_recovered(100%) - percell_recovered(50%) on
each held-out incident; target percell_gain_100_vs_50_ci_low > 0 (paired-bootstrap-v1, check_unit part e
replays it). PASS = data-limited, so the next generation is MORE DATA; MISS = not data-limited, so the next
generation is BETTER INPUTS (what the policy observes).

    python -m t1.rl.remediation.learning_curve --print-declaration > /tmp/rem-lc-decl.json
    python -m tools.sm.run --run-id t1.rl.remediation-learning-curve-<UTC> --declaration /tmp/rem-lc-decl.json \\
      --instance-type ml.m5.4xlarge --max-run 3600 --requirement "stable-baselines3==2.9.2" \\
      --requirement "gymnasium==1.4.0" --requirement "pydantic>=2.7,<3" \\
      --inputs train=s3://<bucket>/jobs/t1.rl.remediation-train-20261104T073837Z/output/<job>/output/output.tar.gz \\
      --entry "-m t1.rl.remediation.learning_curve --run-id {run_id} --evidence-dir {evidence_dir} \\
               --output-dir {output_dir} --train-output {channel:train} --workers 15"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from t1.rl.remediation import bc
from t1.rl.remediation import data as D

UNIT = "t1.rl.remediation-learning-curve"
FRACTIONS = (0.10, 0.25, 0.50, 1.00)
PROCEDURE = "bc-round-0"
ORDER_SALT = "lc-v1"
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PROVENANCE = REPO / "t1" / "rl" / "artifacts" / "remediation_v1.provenance.json"
TRAIN_RUN = "t1.rl.remediation-train-20261104T073837Z"
TRAIN_JOB = "t1-rl-remediation-train-20261104T073837Z-664093"
TRAIN_OUTPUT_S3 = (f"s3://bb26-p-team13-data-483598626742-use2/jobs/{TRAIN_RUN}/output/{TRAIN_JOB}/output/"
                   "output.tar.gz")
METRIC = "percell_gain_100_vs_50_ci_low"
DECISION = {"rule": "item score = percell_recovered(100%) - percell_recovered(50%) on each held-out incident; "
                    "PASS iff its 95% paired-bootstrap lower bound > 0",
            "if_pass": "data-limited: the next generation gets MORE DATA (more twin-labelled incidents)",
            "if_miss": "not data-limited: the next generation gets BETTER INPUTS (what the policy observes, e.g. "
                       "the change since the last step), not more of the same data"}
SMOKE_MAX = 6
GRADER = ("t1/rl/remediation/evaluate.py play (t3/evals/graders.py network level + t3/evals/percell.py "
          "percell-v1 per cell) on every model; item score = percell_recovered(100%) - percell_recovered(50%)")


def pct(f: float) -> int:
    return int(round(f * 100))


def dataset_sha256() -> str:
    return json.loads(PROVENANCE.read_text())["dataset_sha256"]


def find_dataset(train_output: Path, workdir: Path) -> tuple[Path, Path | None]:
    """dataset.npz (and rounds/round0.zip) from the training job's output: a directory or its output.tar.gz."""
    train_output = Path(train_output)
    tars = [train_output] if train_output.is_file() else sorted(train_output.glob("*.tar.gz"))
    root = train_output
    if tars:
        root = workdir / "train-output"
        root.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tars[0]) as t:
            try:
                t.extractall(root, filter="data")
            except TypeError:  # a Python without extraction filters: refuse absolute or parent paths ourselves
                bad = [m.name for m in t.getmembers() if m.name.startswith("/") or ".." in Path(m.name).parts]
                if bad:
                    raise SystemExit(f"refusing to extract {bad[:3]} from {tars[0]}")
                t.extractall(root)
    ds = next(iter(sorted(root.rglob("dataset.npz"))), None)
    if ds is None:
        raise SystemExit(f"no dataset.npz under {train_output}")
    r0 = next(iter(sorted(root.rglob("round0.zip"))), None)
    return ds, r0


# --------------------------------------------------------------------------- data subsets

def load_arrays(ds: Path, expect_sha: str | None) -> dict[str, Any]:
    got = hashlib.sha256(ds.read_bytes()).hexdigest()
    if expect_sha is not None and got != expect_sha:
        raise SystemExit(f"{ds} sha256 {got} is not v1's training dataset {expect_sha}")
    z = np.load(ds, allow_pickle=False)
    return {"obs": {k[4:]: z[k] for k in z.files if k.startswith("obs_")}, "target": z["target"],
            "seed": z["seed"], "role": z["role"].astype(str), "kind": z["kind"].astype(str), "sha256": got}


def twin_labelled(a: dict[str, Any], role: str) -> np.ndarray:
    return (a["role"] == role) & np.isin(a["kind"], np.array(D.SAMPLE_KINDS))


def ordered_train_seeds(a: dict[str, Any]) -> list[int]:
    seeds = sorted({int(s) for s in a["seed"][twin_labelled(a, "train")]})
    return sorted(seeds, key=lambda s: hashlib.sha256(f"{ORDER_SALT}:{s}".encode()).hexdigest())


def subset(a: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
    idx = np.flatnonzero(mask)  # original order kept: the 100% subset is exactly v1's round-0 train arrays
    return {"obs": {k: v[idx] for k, v in a["obs"].items()}, "target": a["target"][idx]}


def fraction_masks(a: dict[str, Any], fractions: tuple[float, ...]) -> dict[float, dict[str, Any]]:
    order = ordered_train_seeds(a)
    base = twin_labelled(a, "train")
    out = {}
    for f in fractions:
        keep = set(order[: max(1, math.ceil(f * len(order)))])
        m = base & np.isin(a["seed"], np.array(sorted(keep)))
        out[f] = {"mask": m, "n_incidents": len(keep), "n_samples": int(m.sum())}
    return out


# --------------------------------------------------------------------------- training

def save_model(model: Any, out_dir: Path, f: float, code_sha: str, prov: dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"remediation_lc{pct(f):03d}"
    zp = out_dir / f"{name}.zip"
    model.save(str(zp))
    meta = {"model_id": "t1-remediation-bc-lc", "version": f"lc{pct(f):03d}-{code_sha[:7]}", "obs": bc.OBS_MODE,
            "postprocess": dict(bc.POSTPROCESS),
            "label": f"learning curve: behaviour cloning round 0 only on {pct(f)}% of remediation-v1's train "
                     "incidents (twin-labelled states); ppo_sm_50k's SB3 PPO actor architecture; deterministic"}
    zp.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    zp.with_suffix(".provenance.json").write_text(json.dumps({**prov, "fraction": f, "artifact": zp.name,
                                                              "sha256": hashlib.sha256(zp.read_bytes()).hexdigest(),
                                                              "meta": meta}, indent=1, sort_keys=True) + "\n")
    return zp


def actor_diff(model: Any, other_zip: Path) -> float:
    """Largest absolute difference between the actor weights of ``model`` and a saved model."""
    from stable_baselines3 import PPO

    o = PPO.load(str(other_zip), device="cpu").policy.state_dict()
    s = model.policy.state_dict()
    keys = [k for k in s if k.startswith(("mlp_extractor.policy_net", "action_net"))]
    return float(max(float((s[k] - o[k]).abs().max()) for k in keys))


def train_all(a: dict[str, Any], fractions: tuple[float, ...], hyper: dict[str, Any], out_dir: Path,
              code_sha: str, round0: Path | None, log: Any = print) -> list[dict[str, Any]]:
    from t3.evals.twin_backend import single_thread

    single_thread()
    val = subset(a, twin_labelled(a, "val"))
    masks = fraction_masks(a, fractions)
    prov = {"unit": UNIT, "procedure": PROCEDURE, "dataset_sha256": a["sha256"], "code_sha": code_sha,
            "hyper": hyper, "order_salt": ORDER_SALT, "train_run": TRAIN_RUN}
    out = []
    for f in fractions:
        tr = subset(a, masks[f]["mask"])
        model = bc.new_model(hyper["seed"])
        rec = bc.fit(model, tr, val, epochs=hyper["epochs"], patience=hyper["patience"], lr=hyper["lr"],
                     batch=hyper["batch"], seed=hyper["seed"], log=log)
        zp = save_model(model, out_dir, f, code_sha, prov)
        row = {"fraction": f, "n_incidents": masks[f]["n_incidents"], "n_samples": masks[f]["n_samples"],
               "n_val_samples": int(len(val["target"])), "artifact": str(zp),
               "sha256": hashlib.sha256(zp.read_bytes()).hexdigest(),
               "fit": {k: v for k, v in rec.items() if k != "history"}}
        if f == 1.0 and round0 is not None:
            row["vs_v1_round0_max_abs_diff"] = actor_diff(model, round0)
        log(f"[lc] {pct(f)}%: {row['n_incidents']} incidents, {row['n_samples']} samples, "
            f"best val mse {rec['best_val_mse']:.5f}, {rec['epochs_run']} epochs")
        out.append(row)
    return out


# --------------------------------------------------------------------------- evaluation

def _play(args: tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]) -> dict[str, Any]:
    from t1.rl.remediation import evaluate as E

    incident, specs, targets = args
    return E._work(("heldout", incident, specs, targets))


def item_for(r: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    from t3.evals import incidents as inc
    from t3.evals.twin_backend import BACKEND

    i, ops = r["incident"], r["operators"]
    pc = {k: float(ops[k]["percell_recovered"]) for k in keys}
    full, half = keys[-1], keys[FRACTIONS.index(0.50)] if len(keys) == len(FRACTIONS) else keys[0]
    keep = ("percell_recovered", "recovery", "success", "diagnosis_hit", "cells_touched", "total_abs_power_db",
            "power_above_healthy_db", "percell_degraded_cells", "change_texts")
    return {"item_id": f"heldout/{i['incident_id']}",
            "input_sha256": hashlib.sha256(inc.canonical_line(i).encode()).hexdigest(),
            "output_sha256": hashlib.sha256(inc.canonical_line({k: ops[k]["change_texts"] for k in keys})
                                            .encode()).hexdigest(),
            "score": pc[full] - pc[half], "passed": bool(ops[full]["percell_recovered"]), "grader": GRADER,
            "backend": BACKEND, "latency_ms": float(ops[full]["latency_ms"]),
            "extra": {"suite": "heldout", "incident": {k: i.get(k) for k in ("incident_id", "seed", "preset",
                                                                              "template", "fault_cells", "stratum")},
                      "diff": {f"percell_100_minus_{k}": pc[full] - pc[k] for k in keys},
                      "models": {k: {m: ops[k][m] for m in keep} for k in keys}}}


def curve_summary(rows: list[dict[str, Any]], keys: list[str], boot: dict[str, Any]) -> dict[str, Any]:
    from t3.evals.stats import paired_bootstrap

    ids = [r["incident"]["incident_id"] for r in rows]
    out = {}
    for k in keys:
        os_ = [r["operators"][k] for r in rows]

        def mean(m: str) -> float:
            return float(np.mean([float(o[m]) for o in os_]))

        full = {i: float(r["operators"][keys[-1]]["percell_recovered"]) for i, r in zip(ids, rows)}
        mine = {i: float(r["operators"][k]["percell_recovered"]) for i, r in zip(ids, rows)}
        out[k] = {"n": len(os_), "percell_recovered": mean("percell_recovered"), "recovery": mean("recovery"),
                  "success_rate": mean("success"), "diagnosis_hit": mean("diagnosis_hit"),
                  "cells_touched": mean("cells_touched"), "total_abs_power_db": mean("total_abs_power_db"),
                  "power_above_healthy_db": mean("power_above_healthy_db"),
                  "percell_100_minus_this": paired_bootstrap(full, mine, boot["resamples"], boot["seed"], boot["ci"])
                  if k != keys[-1] else None}
    return out


# --------------------------------------------------------------------------- declaration and main

def declaration(code_sha: str, *, n_items: int, heldout_sha: str, fractions: tuple[float, ...] = FRACTIONS,
                smoke: bool = False, run_id: str | None = None, instance: str = "ml.m5.4xlarge") -> dict[str, Any]:
    from t3.evals.twin_backend import BACKEND, twin_version

    d: dict[str, Any] = {
        "unit": UNIT, "track": "track-1",
        "target": {"metric": METRIC, "op": ">", "value": 0.0, "unit": "recovered"},
        "n_items": n_items, "test_set_sha256": heldout_sha,
        "model_id": f"t1-remediation-bc-lc@{code_sha[:7]}", "grader": GRADER, "synthetic": bool(smoke),
        "subset": "one item per held-out incident (t1/rl/remediation/suite/heldout-v2.jsonl)",
        "twin": {"kind": "maveric-inprocess", "version": twin_version(), "scenario": "remediation-learning-curve",
                 "seed": bc.SEED,
                 "config": {"backend": BACKEND, "descriptive": True,
                            "decision_rule": DECISION,
                            "fractions": list(fractions), "procedure": PROCEDURE,
                            "procedure_note": ("behaviour cloning round 0 only (bc.fit, bc.HYPER round-0 settings and "
                                               "seed) on the twin-labelled states of the chosen train incidents; "
                                               "remediation-v1 adds two DAgger rounds, left out here: DAgger states "
                                               "depend on the learner, and the full procedure at four sizes does not "
                                               "fit the 20-minute budget. The 100% point is v1's round-0 model "
                                               "(checked against the training job's rounds/round0.zip)."),
                            "subsets": f"train incidents ordered by sha256('{ORDER_SALT}:<seed>'); fraction f takes "
                                       "the first ceil(f x n): nested",
                            "validation": "every val incident's twin-labelled states (the same for every fraction)",
                            "hyper": {k: bc.HYPER[k] for k in ("lr", "batch", "epochs", "patience", "seed")},
                            "training_data": {"run_id": TRAIN_RUN, "job": TRAIN_JOB, "s3": TRAIN_OUTPUT_S3,
                                              "file": "dataset.npz", "sha256": dataset_sha256()},
                            "heldout": {"file": "t1/rl/remediation/suite/heldout-v2.jsonl", "sha256": heldout_sha},
                            "measures": "evaluate.play on every model: network recovery and success, percell-v1, "
                                        "diagnosis_hit, cells touched, |power| and power above healthy",
                            "executed": f"SageMaker training job started by tools/sm/run.py on {instance} (x86_64, "
                                        "the platform the held-out set was frozen on)",
                            "smoke": bool(smoke)}}}
    if run_id:
        d["run_id"] = run_id
    return d


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id")
    ap.add_argument("--evidence-dir", type=Path)
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--train-output", type=Path, help="the v1 training job's output (dir or output.tar.gz)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--instance", default="ml.m5.4xlarge", help="recorded in the declaration")
    ap.add_argument("--print-declaration", action="store_true")
    ap.add_argument("--declare", action="store_true", help="also write the declaration (local runs only)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--heldout-file", type=Path, help="smoke: another held-out file (with its .sha256 and meta)")
    ap.add_argument("--limit-heldout", type=int, help=f"smoke: first K held-out incidents (<= {SMOKE_MAX})")
    ap.add_argument("--epochs", type=int, help="smoke: fewer epochs")
    ap.add_argument("--no-dataset-check", action="store_true", help="smoke: a dataset other than v1's")
    a = ap.parse_args(argv)

    from t1.rl.remediation import evaluate as E
    from t3.evals.evshim import Evidence, code_sha
    from t3.evals.graders import load_targets
    from t3.evals.run import single_thread

    single_thread()  # the main process trains and replays too; pinned before any twin or torch work
    if (a.heldout_file or a.limit_heldout or a.epochs or a.no_dataset_check) and not a.smoke:
        ap.error("--heldout-file, --limit-heldout, --epochs and --no-dataset-check are for --smoke runs only")
    if a.smoke and (a.limit_heldout is None or a.limit_heldout > SMOKE_MAX):
        ap.error(f"a smoke needs --limit-heldout <= {SMOKE_MAX}")
    sha = code_sha()
    ho, ho_sha, _ = E.heldout_set(a.heldout_file)
    ho = ho[: a.limit_heldout] if a.limit_heldout else ho
    if a.print_declaration:
        print(json.dumps(declaration(sha, n_items=len(ho), heldout_sha=ho_sha, smoke=a.smoke, instance=a.instance),
                         indent=2, sort_keys=True))
        return 0
    if not (a.run_id and a.evidence_dir and a.output_dir and a.train_output):
        ap.error("--run-id, --evidence-dir, --output-dir and --train-output are required to run")
    targets = load_targets()
    boot = targets["bootstrap"]
    ev = Evidence(a.evidence_dir)
    decl = declaration(sha, n_items=len(ho), heldout_sha=ho_sha, smoke=a.smoke, run_id=a.run_id, instance=a.instance)
    if a.declare:
        ev.write_declared({**decl, "code_sha": sha, "region": "us-east-2"})
    t0 = time.time()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    import tempfile

    ds, round0 = find_dataset(a.train_output, Path(tempfile.mkdtemp(prefix="lc-train-")))  # not in the outputs
    arrays = load_arrays(ds, None if a.no_dataset_check else dataset_sha256())
    hyper = {k: bc.HYPER[k] for k in ("lr", "batch", "epochs", "patience", "seed")}
    if a.epochs:
        hyper["epochs"] = a.epochs
    models = train_all(arrays, FRACTIONS, hyper, a.output_dir / "models", sha, round0)
    train_s = round(time.time() - t0, 1)
    keys = [f"lc{pct(m['fraction']):03d}" for m in models]
    specs = {k: {"kind": "ppo", "path": m["artifact"]} for k, m in zip(keys, models)}
    rebuild = {"rebuild": False}
    here = E.platform_tag()
    frozen_on = (E.heldout_set(a.heldout_file)[2] or {}).get("platform")
    if frozen_on != here:  # a smoke on another platform: rebuild the frozen trajectories here (keep filter not re-applied)
        ho = [E.rebuild(i, targets) for i in ho]
        rebuild = {"rebuild": True, "frozen_on": frozen_on, "here": here}
    check = E.self_check({"heldout": ho}, targets)  # fail fast before the pool: the first 3 replay here
    print(f"[lc] fail-fast self-check passed: {check['replayed']}", flush=True)
    work = [(i, specs, targets) for i in ho]
    if a.workers > 1:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            rows = list(pool.map(_play, work, chunksize=4))
    else:
        rows = [_play(w) for w in work]
    items = [item_for(r, keys) for r in rows]
    for it in items:
        it["synthetic"] = bool(a.smoke)
    ev.append_records(a.run_id, "item", items)
    from t3.evals.stats import ci_spec, paired_bootstrap

    vals = {it["item_id"]: float(it["score"]) for it in items}
    g = paired_bootstrap(vals, {k: 0.0 for k in vals}, boot["resamples"], boot["seed"], boot["ci"])
    gain = {**ci_spec(boot, None, f"t1-remediation-bc-lc@lc050-{sha[:7]}", "score", None), **g,
            "metric": "percell_recovered", "note": "items hold percell_recovered(100%) - percell_recovered(50%); "
                                                   "baseline_run_id null"}
    value = float(g["ci_low"])
    passed = value > 0.0
    curve = curve_summary(rows, keys, boot)
    from t1.rl.remediation.job_train import finite

    result = {
        "run_id": a.run_id, "unit": UNIT, "metric": METRIC, "value": value, "target": decl["target"],
        "passed": passed, "n_items": len(items), "synthetic": bool(a.smoke),
        "checker": {"script": "t1/rl/remediation/learning_curve.py", "code_sha": sha},
        "details": {"gain": gain, "decision": DECISION["if_pass"] if passed else DECISION["if_miss"],
                    "decision_rule": DECISION, "curve": curve, "models": models, "procedure": PROCEDURE,
                    "dataset_sha256": arrays["sha256"], "heldout_sha256": ho_sha, "rebuild": rebuild,
                    "self_check": check, "host": E._host(), "platform": here, "train_s": train_s,
                    "wall_s": round(time.time() - t0, 1)}}
    ev.write_result(finite(result))
    (a.output_dir / "learning-curve.json").write_text(json.dumps(finite(result["details"]), indent=1,
                                                                 sort_keys=True) + "\n")
    print(json.dumps(finite({"run_id": a.run_id, "metric": METRIC, "value": value, "passed": passed,
                             "decision": result["details"]["decision"],
                             "curve": {k: {m: c[m] for m in ("percell_recovered", "recovery", "diagnosis_hit",
                                                             "cells_touched")} for k, c in curve.items()},
                             "models": [{k: m.get(k) for k in ("fraction", "n_incidents", "n_samples",
                                                               "vs_v1_round0_max_abs_diff")} for m in models],
                             "wall_s": result["details"]["wall_s"]}), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
