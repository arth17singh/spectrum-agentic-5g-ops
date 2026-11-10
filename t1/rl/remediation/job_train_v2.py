"""Data + training of remediation-v2 (t1/rl/remediation/v2.py) as one evidence unit (a SageMaker job).

    python -m t1.rl.remediation.job_train_v2 --print-declaration > /tmp/rem-train-v2-decl.json
    python -m tools.sm.run --run-id t1.rl.remediation-train-v2-<UTC> --declaration /tmp/rem-train-v2-decl.json \\
      --instance-type ml.m5.4xlarge --max-run 5400 --requirement "stable-baselines3==2.9.2" \\
      --requirement "gymnasium==1.4.0" --requirement "pydantic>=2.7,<3" \\
      --entry "-m t1.rl.remediation.job_train_v2 --run-id {run_id} --evidence-dir {evidence_dir} \\
               --output-dir {output_dir} --model-dir {model_dir} --workers 15" [--dry-run]

Local smoke (at most 50 incidents, labelled smoke, never a claim):
    python -m t1.rl.remediation.job_train_v2 --run-id smoke-<x> --evidence-dir /tmp/e --output-dir /tmp/o \\
      --model-dir /tmp/m --limit 24 --declare --smoke

The held-out gate is never opened here: t1/rl/remediation/guard.py is installed before anything else, in this
process and in every worker. Development uses dev data only (the train and val roles of candidates-v2.jsonl; the
deadzone is chosen on the val incidents). Records: one ``item`` per candidate and a ``result`` with n_items ==
the candidate count, as v1's training unit. Artifacts: {model_dir}/remediation_v2.zip + .meta.json +
.provenance.json; {output_dir}/dataset.npz, training.json, deadzone.json, data-summary.json.
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

from t1.rl.remediation import v2 as V

UNIT = "t1.rl.remediation-train-v2"
SMOKE_MAX = 50


def version_for(code_sha: str) -> str:
    return f"v2-{code_sha[:7]}"


def declaration(specs: list[dict[str, Any]], code_sha: str, run_id: str | None = None,
                smoke: bool = False) -> dict[str, Any]:
    from t1.rl.remediation import bc
    from t1.rl.remediation import data as D
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
        "model_id": f"{V.MODEL_ID}@{version_for(code_sha)}",
        "grader": D.GRADER,
        "synthetic": bool(smoke),
        "twin": {"kind": "maveric-inprocess", "version": twin_version(), "scenario": "remediation-train-v2",
                 "seed": min(int(s["seed"]) for s in specs),
                 "config": {"backend": BACKEND, "candidates": V.CANDIDATES_REL, "data_version": V.DATA_VERSION,
                            "seed_range": [min(int(s["seed"]) for s in specs), max(int(s["seed"]) for s in specs)],
                            "roles": roles, "role_salt": D.ROLE_SALT,
                            "inputs": "t1/rl/policy.py obs 'masked_delta': v1's masked observation + kpi_delta (30: KPI "
                                      "changes against the pre-incident baseline, fault_step - 1) + cell_share (8) + "
                                      "serves_ues (8); never the baseline's configuration (power/tilt): the "
                                      "config-history rule; cells without UEs under the no-UE rule (policy.py)",
                            "labels": "the residual way back to the commissioned configuration (data.label_for, twin "
                                      "replay), one step's limits; incidents kept only if the inverse verifiably "
                                      "recovers them",
                            "sample_kinds": list(D.SAMPLE_KINDS) + ["dagger-fault@4", "dagger-fault@5",
                                                                    "dagger-healthy@4", "dagger-healthy@5"],
                            "training": {**V.HYPER, "obs": V.OBS_MODE,
                                         "architecture": "ppo_sm_50k's policy kwargs on its observation space plus "
                                                         "kpi_delta (30), cell_share (8) and serves_ues (8); actor "
                                                         "mean regression",
                                         "holdout": "role=holdout candidates are never trained on"},
                            "deadzone": {"candidates": [list(d) for d in V.DEADZONES], "tolerance": V.DZ_TOLERANCE,
                                         "rule": "v2.py docstring (chosen on the val incidents, dev data only)"},
                            "development_rule": "dev data only; the held-out gate (t1/rl/remediation/suite/heldout-*) "
                                                "is never opened in this job (t1/rl/remediation/guard.py)",
                            "v1_training_unit": "t1.rl.remediation-train-20261104T073837Z",
                            "executed": "SageMaker training job started by tools/sm/run.py (x86_64 CPU image); "
                                        "in-process twin (twin/, W2), no network twin",
                            "evidence_writer": "tools/evlib.py: a local store in the job, uploaded and attested "
                                               "by tools/sm/entry.py",
                            "smoke": bool(smoke)}},
    }
    if bc.MODEL_ID == V.MODEL_ID:   # pragma: no cover - guards a copy-paste slip
        raise RuntimeError("v2 must not reuse v1's model id")
    if run_id:
        d["run_id"] = run_id
    return d


def main(argv: list[str] | None = None) -> int:
    V.protect()   # first: nothing in this process may open the held-out gate
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id")
    ap.add_argument("--evidence-dir", type=Path)
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--model-dir", type=Path)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, help=f"first N candidates only (local smoke, N <= {SMOKE_MAX})")
    ap.add_argument("--smoke", action="store_true", help="label everything synthetic smoke (local tests)")
    ap.add_argument("--declare", action="store_true", help="also write the declaration (local runs only)")
    ap.add_argument("--print-declaration", action="store_true")
    ap.add_argument("--hyper", default=None, help="JSON overrides of v2.HYPER (smoke only)")
    a = ap.parse_args(argv)

    from t1.rl.remediation import data as D
    from t1.rl.remediation.job_train import finite, item_for
    from t3.evals.evshim import Evidence, code_sha
    from t3.evals.graders import load_targets
    from t3.evals.twin_backend import single_thread

    specs = V.load_candidates()
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
        ap.error("--hyper is for smoke runs only (the job trains with v2.HYPER as declared)")
    single_thread()
    ev = Evidence(a.evidence_dir)
    if a.declare:
        ev.write_declared({**declaration(specs, sha, a.run_id, a.smoke), "code_sha": sha, "region": "us-east-2"})
    t0 = time.time()
    targets = load_targets()
    rows = V.generate(specs, targets, workers=a.workers)
    gen_s = round(time.time() - t0, 1)
    print(f"[v2] generated {len(rows)} candidates in {gen_s} s: {json.dumps(D.summary(rows)['by_reason'])}",
          flush=True)
    by_seed = {int(s["seed"]): s for s in specs}
    version = version_for(sha)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    workdir = a.output_dir / "rounds"
    workdir.mkdir(parents=True, exist_ok=True)
    out = V.train(rows, by_seed, targets, version=version, workdir=workdir, workers=a.workers,
                  hyper=json.loads(a.hyper) if a.hyper else None)
    arrays = V.to_arrays(out["rows"])
    ds = a.output_dir / "dataset.npz"
    np.savez_compressed(ds, target=arrays["target"], seed=np.array([i[0] for i in arrays["info"]]),
                        role=np.array([i[1] for i in arrays["info"]]), kind=np.array([i[2] for i in arrays["info"]]),
                        **{f"obs_{k}": v for k, v in arrays["obs"].items()})
    ds_sha = hashlib.sha256(ds.read_bytes()).hexdigest()
    summ = D.summary(rows)
    prov = {"run_id": a.run_id, "unit": UNIT, "code_sha": sha, "version": version, "smoke": a.smoke,
            "candidates_sha256": hashlib.sha256(D.candidates_bytes(specs)).hexdigest(), "n_candidates": len(specs),
            "dataset_sha256": ds_sha, "data": summ, "hyper": out["record"]["hyper"],
            "deadzone": out["record"]["deadzone"],
            "base_architecture": "t1/rl/artifacts/ppo_sm_50k.zip policy kwargs on an extended observation space "
                                 "(weights not copied)",
            "development_rule": "dev data only; the held-out gate was not opened (guard installed)"}
    art = V.save(out["model"], a.model_dir, version, out["deadzone"], prov)
    (a.output_dir / "training.json").write_text(json.dumps(out["record"], indent=1, sort_keys=True,
                                                           default=float) + "\n")
    (a.output_dir / "deadzone.json").write_text(json.dumps(finite({**out["record"]["deadzone"],
                                                                   "rows": out["deadzone_rows"]}),
                                                           indent=1, sort_keys=True) + "\n")
    (a.output_dir / "data-summary.json").write_text(json.dumps(summ, indent=1, sort_keys=True) + "\n")
    items = []
    for r in rows:
        it = finite(item_for(r, by_seed[int(r["seed"])]))
        it["synthetic"] = bool(a.smoke)
        items.append(it)
    ev.append_records(a.run_id, "item", items)
    n = len(items)
    result = {
        "run_id": a.run_id, "unit": UNIT, "metric": "n_items", "value": n,
        "target": {"metric": "n_items", "op": "==", "value": len(specs), "unit": "count"},
        "passed": n == len(specs), "n_items": n, "synthetic": bool(a.smoke),
        "checker": {"script": "t1/rl/remediation/job_train_v2.py", "code_sha": sha},
        "details": {"data": summ, "dataset_sha256": ds_sha, "artifact": {k: art[k] for k in ("sha256", "meta")},
                    "training": {"rounds": [{k: v for k, v in r.items() if k != "history_tail"}
                                            for r in out["record"]["rounds"]],
                                 "deadzone": out["record"]["deadzone"],
                                 "final_val_proposals": out["record"]["final_val_proposals"]},
                    "guard": {"installed": True, "what": "t1/rl/remediation/guard.py: the held-out gate refused "
                                                         "to every open() in this job"},
                    "generation_s": gen_s, "wall_s": round(time.time() - t0, 1),
                    "note": "training diagnostics only (dev val split); the claimed evaluation is "
                            "t1/rl/remediation/evaluate.py --candidate v2"},
    }
    ev.write_result(finite(result))
    print(json.dumps(finite({"run_id": a.run_id, "n_items": n, "kept": summ["kept"], "artifact_sha256": art["sha256"],
                             "version": version, "deadzone": out["record"]["deadzone"],
                             "final_val_proposals": out["record"]["final_val_proposals"],
                             "wall_s": result["details"]["wall_s"]}), indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
