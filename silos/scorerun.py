"""Shared by the score-producing jobs (baselines, FL-aggregate evaluation) and the comparison job.

A "score run" is one evidence run in which ONE model predicts every row of the ONE shared test set of a split.
Its item records carry the per-row scaled-MAE score and the prediction; a comparison run later pairs two score
runs row by row. Everything uses `silos.nn_core` (numpy), so the evaluation of a federated aggregate needs no
FL-service package and no torch.

Data access: DATA_DIR (a local tree laid out like silos.generate's output) or DATA_COPY_S3 (the evidence copy
written by silos.job_data, with its manifest.json). Files are verified against the manifest's SHA-256.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from . import evio, nn_core as nn

SPLITS = ("disjoint", "overlap")
SILOS = ("A", "B", "C")
SEGMENTS = ("viavi_like", "viavi_default_noise", "region_A", "region_B", "region_C")
GRADER = "silos.nn_core: per-row scaled MAE (declared scales) on the shared test set"
BACKEND = "numpy forward pass on CPU"
EVAL_SEED = 20261102  # fixed seed of the paired bootstrap, declared in the templates


def item_id(segment: str, seed: int, step: int) -> str:
    return f"{segment}:{int(seed)}:{int(step)}"


def committed_manifest() -> dict:
    return json.loads((Path(__file__).resolve().parent / "manifest.json").read_text())


def manifest_files(manifest: dict) -> dict[str, dict]:
    return {f["path"]: f for f in manifest["files"]}


def load_manifest(data_dir: str | None = None, data_s3: str | None = None) -> dict:
    """The manifest of the data tree in use (the AWS copy when DATA_COPY_S3 is set), else the committed one."""
    data_dir = data_dir or os.environ.get("DATA_DIR") or None
    data_s3 = data_s3 or os.environ.get("DATA_COPY_S3") or None
    if data_dir and (Path(data_dir) / "manifest.json").exists():
        return json.loads((Path(data_dir) / "manifest.json").read_text())
    if data_s3:
        import boto3

        b, _, k = data_s3[5:].partition("/")
        body = boto3.client("s3", region_name=evio.region()).get_object(Bucket=b, Key=f"{k.rstrip('/')}/manifest.json")["Body"].read()
        return json.loads(body)
    return committed_manifest()


def fetch(paths: list[str], manifest: dict, data_dir: str | None = None, data_s3: str | None = None) -> Path:
    """Return a local directory holding `paths`, verified against the manifest SHA-256."""
    data_dir = data_dir or os.environ.get("DATA_DIR") or None
    data_s3 = data_s3 or os.environ.get("DATA_COPY_S3") or None
    files = manifest_files(manifest)
    if data_dir:
        root = Path(data_dir)
    elif data_s3:
        import boto3

        root = Path(tempfile.mkdtemp(prefix="silo-data-"))
        s3 = boto3.client("s3", region_name=evio.region())
        b, _, k = data_s3[5:].partition("/")
        for p in paths:
            (root / p).parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(b, f"{k.rstrip('/')}/{p}", str(root / p))
    else:
        raise SystemExit("set DATA_DIR (local tree) or DATA_COPY_S3 (s3://bucket/t2-silos/<data run>/)")
    for p in paths:
        got = hashlib.sha256((root / p).read_bytes()).hexdigest()
        if got != files[p]["sha256"]:
            raise SystemExit(f"{p}: sha256 {got[:12]} differs from the manifest {files[p]['sha256'][:12]}; wrong data tree?")
    return root


def score_items(test: dict, pred: np.ndarray, label: str) -> list[dict]:
    sc = nn.row_scores(pred, test["y"])
    out = []
    for i in range(len(sc)):
        out.append(evio.item(
            item_id=item_id(test["segment"][i], test["seed"][i], test["step"][i]),
            input_sha256=hashlib.sha256(np.ascontiguousarray(test["x"][i]).tobytes()).hexdigest(),
            output_sha256=hashlib.sha256(np.ascontiguousarray(pred[i], dtype="<f8").tobytes()).hexdigest(),
            grader=GRADER, backend=BACKEND, score=float(sc[i]), segment=str(test["segment"][i]), seed=int(test["seed"][i]),
            pred=[float(v) for v in pred[i]], model=label))
    return out


def summarize(test: dict, pred: np.ndarray) -> dict:
    d = {"mean_score": float(nn.row_scores(pred, test["y"]).mean()),
         "mean_score_noop": float(nn.row_scores(np.zeros_like(test["y"]), test["y"]).mean()),
         "mae": nn.label_mae(pred, test["y"]), "direction_accuracy": nn.direction_accuracy(pred, test["y"]),
         "score_scale": nn.SCORE_SCALE.tolist(), "per_segment": {}}
    for seg in SEGMENTS:
        m = test["segment"] == seg
        if m.any():
            d["per_segment"][seg] = {"n": int(m.sum()), "mean_score": float(nn.row_scores(pred[m], test["y"][m]).mean()),
                                     "mean_score_noop": float(nn.row_scores(np.zeros_like(test["y"][m]), test["y"][m]).mean()),
                                     "mae": nn.label_mae(pred[m], test["y"][m])}
    return d


def score_declaration(*, run_id: str, unit: str, model_id: str, n_rows: int, test_sha: str, code_sha: str, note: str, extra: dict | None = None) -> dict:
    return evio.declaration(
        run_id=run_id, unit=unit, target={"metric": "n_items", "op": ">=", "value": n_rows, "unit": "shared-test rows scored (a measurement: the MAE is in the result details, there is no pass bar)"},
        n_items=n_rows, test_set_sha256=test_sha, model_id=model_id, grader=GRADER, code_sha=code_sha,
        extra={"note": note, "score_scale": nn.SCORE_SCALE.tolist(), **(extra or {})})


def test_rows(manifest: dict, split: str) -> int:
    return manifest_files(manifest)[f"{split}/shared_test/test.csv"]["rows"]


def test_sha(manifest: dict, split: str) -> str:
    return manifest_files(manifest)[f"{split}/shared_test/test.csv"]["sha256"]
