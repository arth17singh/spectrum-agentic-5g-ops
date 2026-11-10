"""Data + training of remediation-v1, as one evidence unit (a SageMaker job via tools/sm/run.py).

    python -m t1.rl.remediation.job_train --print-declaration > /tmp/rem-train-decl.json
    python -m tools.sm.run --run-id t1.rl.remediation-train-<UTC> --declaration /tmp/rem-train-decl.json \\
      --instance-type ml.m5.xlarge --max-run 5400 --requirement "stable-baselines3==2.9.2" \\
      --requirement "gymnasium==1.4.0" --requirement "pydantic>=2.7,<3" \\
      --entry "-m t1.rl.remediation.job_train --run-id {run_id} --evidence-dir {evidence_dir} \\
               --output-dir {output_dir} --model-dir {model_dir} --workers 4" [--dry-run]

Local smoke (at most 50 incidents, labelled smoke, never a claim):
    python -m t1.rl.remediation.job_train --run-id smoke-<x> --evidence-dir /tmp/e --output-dir /tmp/o \\
      --model-dir /tmp/m --limit 32 --declare --smoke

Records: one ``item`` per candidate (input = canonical spec, output = canonical outcome: kept or the drop
reason, role, sample count, the verified inverse) and a ``result`` with n_items == the candidate count.
Artifacts: {model_dir}/remediation_v1.zip + .meta.json + .provenance.json (SageMaker's model.tar.gz);
{output_dir}/dataset.npz (observations, targets, sample info), training.json, data-summary.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from t1.rl.remediation import bc
from t1.rl.remediation import data as D

UNIT = "t1.rl.remediation-train"
CANDIDATES_REL = "t1/rl/remediation/candidates.jsonl"
SMOKE_MAX = 50


def version_for(code_sha: str) -> str:
    return f"v1-{code_sha[:7]}"


def finite(obj: Any) -> Any:
    """JSON-safe copy: numpy scalars to Python, NaN and infinities to None (evidence refuses NaN)."""
    import math

    if isinstance(obj, dict):
        return {str(k): finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [finite(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def declaration(specs: list[dict[str, Any]], code_sha: str, run_id: str | None = None,
                smoke: bool = False) -> dict[str, Any]:
    from t3.evals.twin_backend import BACKEND, twin_version

    data = D.candidates_bytes(specs)
    roles: dict[str, int] = {}
    for s in specs:
        roles[s["role"]] = roles.get(s["role"], 0) + 1
    d: dict[str, Any] = {
        "unit": UNIT, "track": "track-1",
        "target": {"metric": "n_items", "op": "==", "value": len(specs), "unit": "count"},
        "n_items": len(specs),
        "test_set_sha256": hashlib.sha256(data).hexdigest(),
        "model_id": f"{bc.MODEL_ID}@{version_for(code_sha)}",
        "grader": D.GRADER,
        "synthetic": bool(smoke),
        "twin": {"kind": "maveric-inprocess", "version": twin_version(), "scenario": "remediation-train",
                 "seed": min(int(s["seed"]) for s in specs),
                 "config": {"backend": BACKEND, "candidates": CANDIDATES_REL, "data_version": D.DATA_VERSION,
                            "seed_range": [min(int(s["seed"]) for s in specs), max(int(s["seed"]) for s in specs)],
                            "roles": roles, "role_salt": D.ROLE_SALT,
                            "labels": "the way back to the commissioned configuration (twin replay), one step's "
                                      "limits; incidents kept only if the inverse verifiably recovers them",
                            "sample_kinds": list(D.SAMPLE_KINDS),
                            "training": {**bc.HYPER, "obs": bc.OBS_MODE, "postprocess": bc.POSTPROCESS,
                                         "architecture": "ppo_sm_50k's spaces and policy kwargs (MultiInputPolicy "
                                                         "64-64 tanh, log_std_init -0.5); actor mean regression",
                                         "holdout": "role=holdout candidates are never trained on (evaluation's "
                                                    "dev holdout)"},
                            "executed": "SageMaker training job started by tools/sm/run.py (x86_64 CPU image); "
                                        "in-process twin (twin/, W2), no network twin",
                            "evidence_writer": "tools/evlib.py: a local store in the job, uploaded and attested "
                                               "by tools/sm/entry.py",
                            "smoke": bool(smoke)}},
    }
    if run_id:
        d["run_id"] = run_id
    return d


def item_for(r: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    from t3.evals import incidents as inc
    from t3.evals.twin_backend import BACKEND

    outcome = {"kept": r["kept"], "reason": r["reason"], "role": r["role"], "n_samples": len(r["samples"]),
               "inverse": r.get("inverse"), "incident_id": r.get("incident_id")}
    return {"item_id": f"cand-{spec['seed']}",
            "input_sha256": hashlib.sha256(inc.canonical_line(spec).encode()).hexdigest(),
            "output_sha256": hashlib.sha256(inc.canonical_line(outcome).encode()).hexdigest(),
            "score": None, "passed": None, "grader": D.GRADER, "backend": BACKEND,
            "latency_ms": float(r.get("latency_ms", 0.0)),
            "extra": {**outcome, "index": spec["index"], "template": spec["template"], "preset": spec["preset"],
                      "tier": spec["tier"], "fault_cells": sorted({f["cell"] for f in spec["fault"]}),
                      "recovery_of_inverse": r.get("recovery_of_inverse"), "alarm_on_fault": r.get("alarm_on_fault")}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id")
    ap.add_argument("--evidence-dir", type=Path)
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--model-dir", type=Path)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, help=f"first N candidates only (local smoke, N <= {SMOKE_MAX})")
    ap.add_argument("--smoke", action="store_true", help="label everything synthetic smoke (local tests)")
    ap.add_argument("--declare", action="store_true", help="also write the declaration (local runs; a job's "
                                                             "declaration is written by tools/sm/run.py)")
    ap.add_argument("--print-declaration", action="store_true")
    ap.add_argument("--hyper", default=None, help="JSON overrides of bc.HYPER (smoke only)")
    a = ap.parse_args(argv)

    from t3.evals.evshim import Evidence, code_sha
    from t3.evals.graders import load_targets
    from t3.evals.twin_backend import single_thread

    specs = D.load_candidates()
    if a.limit:
        if not a.smoke or a.limit > SMOKE_MAX:
            ap.error(f"--limit is for local smoke runs only (with --smoke, at most {SMOKE_MAX})")
        specs = specs[: a.limit]
    sha = code_sha()
    if a.print_declaration:
        print(json.dumps(declaration(specs, sha, smoke=a.smoke), indent=2, sort_keys=True))
        return 0
    if not (a.run_id and a.evidence_dir and a.output_dir and a.model_dir):
        ap.error("--run-id, --evidence-dir, --output-dir and --model-dir are required to run")
    if a.hyper and not a.smoke:
        ap.error("--hyper is for smoke runs only (the job trains with bc.HYPER as declared)")
    single_thread()
    ev = Evidence(a.evidence_dir)
    if a.declare:  # a local run; in a job tools/sm/run.py wrote it (with code_sha and region) before the job
        ev.write_declared({**declaration(specs, sha, a.run_id, a.smoke), "code_sha": sha, "region": "us-east-2"})
    t0 = time.time()
    targets = load_targets()
    rows = D.generate(specs, targets, workers=a.workers)
    gen_s = round(time.time() - t0, 1)
    print(f"[train] generated {len(rows)} candidates in {gen_s} s: {json.dumps(D.summary(rows)['by_reason'])}",
          flush=True)
    by_seed = {int(s["seed"]): s for s in specs}
    version = version_for(sha)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    workdir = a.output_dir / "rounds"
    workdir.mkdir(parents=True, exist_ok=True)
    out = bc.train(rows, by_seed, targets, version=version, workdir=workdir, workers=a.workers,
                   hyper=json.loads(a.hyper) if a.hyper else None)
    arrays = D.to_arrays(out["rows"], bc.OBS_MODE)  # the twin-labelled states plus every DAgger state
    ds = a.output_dir / "dataset.npz"
    np.savez_compressed(ds, target=arrays["target"], seed=np.array([i[0] for i in arrays["info"]]),
                        role=np.array([i[1] for i in arrays["info"]]), kind=np.array([i[2] for i in arrays["info"]]),
                        **{f"obs_{k}": v for k, v in arrays["obs"].items()})
    ds_sha = hashlib.sha256(ds.read_bytes()).hexdigest()
    summ = D.summary(rows)
    prov = {"run_id": a.run_id, "unit": UNIT, "code_sha": sha, "version": version, "smoke": a.smoke,
            "candidates_sha256": hashlib.sha256(D.candidates_bytes(specs)).hexdigest(), "n_candidates": len(specs),
            "dataset_sha256": ds_sha, "data": summ, "hyper": out["record"]["hyper"],
            "base_architecture": "t1/rl/artifacts/ppo_sm_50k.zip (spaces and policy kwargs only; weights not copied)"}
    art = bc.save(out["model"], a.model_dir, version, prov)
    (a.output_dir / "training.json").write_text(json.dumps(out["record"], indent=1, sort_keys=True, default=float) + "\n")
    (a.output_dir / "data-summary.json").write_text(json.dumps(summ, indent=1, sort_keys=True) + "\n")
    items = [finite(item_for(r, by_seed[int(r["seed"])])) for r in rows]
    for it in items:
        it["synthetic"] = bool(a.smoke)
    ev.append_records(a.run_id, "item", items)
    n = len(items)
    result = {
        "run_id": a.run_id, "unit": UNIT, "metric": "n_items", "value": n,
        "target": {"metric": "n_items", "op": "==", "value": len(specs), "unit": "count"},
        "passed": n == len(specs), "n_items": n, "synthetic": bool(a.smoke),
        "checker": {"script": "t1/rl/remediation/job_train.py", "code_sha": sha},
        "details": {"data": summ, "dataset_sha256": ds_sha, "artifact": {k: art[k] for k in ("sha256", "meta")},
                    "training": {"rounds": [{k: v for k, v in r.items() if k != "history_tail"}
                                            for r in out["record"]["rounds"]],
                                 "final_val_proposals": out["record"]["final_val_proposals"]},
                    "generation_s": gen_s, "wall_s": round(time.time() - t0, 1),
                    "note": "training diagnostics only (val split); the claimed evaluation is "
                            "t1/rl/remediation/evaluate.py"},
    }
    ev.write_result(finite(result))
    print(json.dumps({"run_id": a.run_id, "n_items": n, "kept": summ["kept"], "artifact_sha256": art["sha256"],
                      "version": version, "final_val_proposals": out["record"]["final_val_proposals"],
                      "wall_s": result["details"]["wall_s"]}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
