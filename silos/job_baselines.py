"""Job entry: train a baseline model and score it on the ONE shared test set (CPU).

    python -m silos.job_baselines --method local   --silo A --split disjoint
    python -m silos.job_baselines --method central          --split disjoint     # pooled data: a reference, NOT private
    python -m silos.job_baselines --method fedavg           --split disjoint     # plain FedAvg in one process: "local FedAvg, not the FL service"
    add --dry-run (tiny: 1 epoch, first 500 training rows, nothing written)  or  --print-declaration

One invocation = one evidence run. Items: one per shared-test row (per-row scaled-MAE score and prediction);
result: a MEASUREMENT (target n_items >= rows scored, no pass bar), the MAE per label, per segment, the direction
accuracy and the no-op reference are in the result details. A `model` record points at the weights
(safetensors) written to $MODEL_DIR (default /opt/ml/model) and, when MODEL_OUT_S3 is set, to S3.

Environment: RUN_ID, CODE_SHA, EVIDENCE_S3, AWS_REGION; DATA_DIR or DATA_COPY_S3 (see silos.scorerun);
MODEL_DIR, MODEL_OUT_S3 (optional). Training uses torch (CPU).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from . import evio, nn_core as nn, scorerun as sr

UNIT_OF = {"local": "t2.local_only", "central": "t2.central", "fedavg": "t2.fedavg_local"}


def unit_name(method: str, silo: str | None, split: str) -> str:
    return f"{UNIT_OF[method]}.{silo}.{split}" if method == "local" else f"{UNIT_OF[method]}.{split}"


def model_id_for(method: str, silo: str | None, split: str) -> str:
    base = {"local": f"t2-response-local-{(silo or '').lower()}", "central": "t2-response-central", "fedavg": "t2-response-fedavg-local"}[method]
    return f"{base}-{split}@v1"


NOTE = {
    "local": "Local-only control: trained on ONE simulated silo's training file only. A measurement of that silo's model on the shared test set; no pass bar.",
    "central": "Central reference: trained on the three simulated silos' training files POOLED. It is not private and not federated; it is the reference the federated model is compared to.",
    "fedavg": "Fallback/preview: plain FedAvg of the three simulated silos run in one process (row-count weighted). It is NOT the FL service.",
}


def declaration_for(method: str, silo: str | None, split: str, run_id: str, code_sha: str, manifest: dict, epochs: int, seed: int) -> dict:
    return sr.score_declaration(
        run_id=run_id, unit=unit_name(method, silo, split), model_id=model_id_for(method, silo, split), n_rows=sr.test_rows(manifest, split),
        test_sha=sr.test_sha(manifest, split), code_sha=code_sha, note=NOTE[method],
        extra={"method": method, "silo": silo, "split": split, "epochs": epochs, "seed": seed,
               "architecture": "MLP 38-128-128-4, Huber loss, Adam 1e-3, batch 256", "entry": f"python -m silos.job_baselines --method {method}" + (f" --silo {silo}" if silo else "") + f" --split {split}"})


def train(method: str, silo: str | None, split: str, root: Path, epochs: int, seed: int, limit: int | None):
    import torch

    torch.set_num_threads(max(1, (os.cpu_count() or 2)))

    def load(s: str):
        d = nn.read_xy(root / split / s / "train.csv")
        if limit:
            d = {k: v[:limit] for k, v in d.items()}
        return d

    if method == "local":
        x, y = nn.tensors(load(silo))
        torch.manual_seed(seed)
        m = nn.torch_model()
        nn.train_epochs(m, x, y, epochs, seed)
        return m, {silo: len(x)}
    data = {s: load(s) for s in sr.SILOS}
    if method == "central":
        pooled = {k: np.concatenate([data[s][k] for s in sr.SILOS]) for k in ("x", "y")}
        x, y = nn.tensors(pooled)
        torch.manual_seed(seed)
        m = nn.torch_model()
        nn.train_epochs(m, x, y, epochs, seed)
        return m, {s: len(data[s]["x"]) for s in sr.SILOS}
    torch.manual_seed(seed)
    return nn.fedavg({s: nn.tensors(d) for s, d in data.items()}, epochs, seed), {s: len(data[s]["x"]) for s in sr.SILOS}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=["local", "central", "fedavg"], required=True)
    ap.add_argument("--silo", choices=list(sr.SILOS))
    ap.add_argument("--split", choices=list(sr.SPLITS), default="disjoint")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--print-declaration", action="store_true")
    ap.add_argument("--declare-first", action="store_true")
    a = ap.parse_args(argv)
    if (a.method == "local") != bool(a.silo):
        ap.error("--silo is required for --method local and not allowed otherwise")
    run_id = evio.env_run_id() or ("dryrun-" + unit_name(a.method, a.silo, a.split).replace(".", "-") if (a.dry_run or a.print_declaration) else None)
    if not run_id:
        sys.exit("set RUN_ID (or EVIDENCE_RUN_ID)")
    code_sha = evio.env_code_sha(required=not (a.dry_run or a.print_declaration))
    manifest = sr.load_manifest()
    epochs = 1 if a.dry_run else a.epochs
    declared = declaration_for(a.method, a.silo, a.split, run_id, code_sha, manifest, epochs, a.seed)
    if a.print_declaration:
        evio.print_json(declared)
        return 0
    if a.declare_first and not a.dry_run:
        evio.declare_first(declared)

    need = [f"{a.split}/shared_test/test.csv"] + [f"{a.split}/{s}/train.csv" for s in ([a.silo] if a.method == "local" else sr.SILOS)]
    root = sr.fetch(need, manifest)
    t0 = time.time()
    model, train_rows = train(a.method, a.silo, a.split, root, epochs, a.seed, 500 if a.dry_run else None)
    secs = time.time() - t0
    params = nn.params_from_torch(model)
    nn.check_params(params)
    test = nn.read_xy(root / a.split / "shared_test" / "test.csv")
    if a.dry_run:
        test = {k: v[:300] for k, v in test.items()}
    pred = nn.forward_np(params, test["x"])
    items = sr.score_items(test, pred, declared["model_id"])
    if a.dry_run:  # the dry run scores a slice, so its declared target is the slice size
        declared["target"]["value"] = declared["n_items"] = len(items)

    blob = nn.safetensors_dumps(params, {"model_id": declared["model_id"], "run_id": run_id})
    sha = nn.sha256_bytes(blob)
    mdir = Path(os.environ.get("MODEL_DIR", "/opt/ml/model" if Path("/opt/ml").exists() else "model_out"))
    uri = f"file://{mdir}/model.safetensors"
    if not a.dry_run:
        mdir.mkdir(parents=True, exist_ok=True)
        (mdir / "model.safetensors").write_bytes(blob)
        out_s3 = os.environ.get("MODEL_OUT_S3", "").strip()
        if out_s3:
            import boto3

            b, _, k = out_s3[5:].partition("/")
            key = f"{k.rstrip('/')}/{run_id}/model.safetensors"
            boto3.client("s3", region_name=evio.region()).put_object(Bucket=b, Key=key, Body=blob)
            uri = f"s3://{b}/{key}"
    summ = sr.summarize(test, pred)
    details = {**summ, "method": a.method, "silo": a.silo, "split": a.split, "epochs": epochs, "seed": a.seed, "train_rows": train_rows,
               "train_seconds": round(secs, 1), "artifact_sha256": sha, "dry_run": a.dry_run,
               "label": "local FedAvg, not the FL service" if a.method == "fedavg" else a.method}
    model_rec = {"model_id": declared["model_id"].split("@")[0], "version": "v1", "model_kind": "t2-response", "artifact_uri": uri, "artifact_sha256": sha,
                 "status": "candidate", "trained_on": f"{a.split}: " + ", ".join(f"{s}={n} rows" for s, n in train_rows.items()),
                 "metrics": {"mean_score": summ["mean_score"], "mean_score_noop": summ["mean_score_noop"]}}
    sink = evio.Sink(run_id)
    out = evio.finish(declared=declared, items=items, script="silos/job_baselines.py", code_sha=code_sha, details=details, dry_run=a.dry_run, sink=sink)
    if not a.dry_run:
        evio.get_evlib().write_model({**model_rec, "run_id": run_id})
    r = out["result"]
    print(json.dumps({"run_id": run_id, "unit": r["unit"], "mean_score": summ["mean_score"], "no_op": summ["mean_score_noop"], "mae": summ["mae"],
                      "direction_accuracy": summ["direction_accuracy"], "n_items": r["n_items"], "passed": r["passed"], "artifact": uri}, indent=1))
    return 0 if r["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
