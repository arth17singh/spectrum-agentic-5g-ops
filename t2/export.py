"""Export a trained Track 2 checkpoint to a torch-free `.npz` (float32 weights).

    python -m t2.export --ckpt <ckpt.pt> --out t2/artifacts/t2_response_v0.npz \
        [--manifest <dataset manifest>] [--model-id t2-response-v0] [--version v0]

The npz holds W0,b0,W1,b1,... (x @ W + b, ReLU between layers), y_std (4), and `meta_json`.
Load it with `t2.predict.load`, which needs numpy only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from t2.features import CONTRACT_VERSION, FEATURE_NAMES, LABELS

LABEL = "v0, local, not federated / not the official job"


def state_dict_to_params(state_dict: dict[str, Any], y_std: np.ndarray) -> dict[str, np.ndarray]:
    params: dict[str, np.ndarray] = {}
    i = 0
    while f"layers.{i}.weight" in state_dict:
        params[f"W{i}"] = state_dict[f"layers.{i}.weight"].detach().cpu().numpy().T.astype(np.float32)
        params[f"b{i}"] = state_dict[f"layers.{i}.bias"].detach().cpu().numpy().astype(np.float32)
        i += 1
    params["y_std"] = np.asarray(y_std, dtype=np.float32)
    return params


def export_npz(state_dict: dict[str, Any], y_std: np.ndarray, out: str | Path, meta: dict[str, Any]) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    full = {"model_id": "t2-response-v0", "version": "v0", "label": LABEL, "contract_version": CONTRACT_VERSION,
            "feature_names": FEATURE_NAMES, "labels": list(LABELS),
            "target": "KPI delta (after the change minus the no-op counterfactual at the same step) divided by y_std; anchored: net(x) - net(x with no change)",
            "activation": "relu", **meta}
    np.savez(out, **state_dict_to_params(state_dict, y_std), meta_json=np.asarray(json.dumps(full)))
    return out


def main(argv: list[str] | None = None) -> int:
    import torch

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest")
    ap.add_argument("--model-id", default="t2-response-v0")
    ap.add_argument("--version", default="v0")
    a = ap.parse_args(argv)
    ck = torch.load(a.ckpt, weights_only=False)
    meta: dict[str, Any] = {"model_id": a.model_id, "version": a.version, "hidden": list(ck["hidden"]),
                            "dataset_sha256": ck.get("dataset_sha256"), "n_train": ck.get("n_train"),
                            "train_seed": ck.get("seed"), "epochs_run": ck.get("epochs_run")}
    if a.manifest:
        m = json.loads(Path(a.manifest).read_text())
        meta["dataset_manifest"] = {"dataset_sha256": m["dataset_sha256"], "n_rows": m["n_rows"],
                                    "generator_git_commit": m["generator"]["git_commit"]}
    try:
        meta["trainer_git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                                    check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        meta["trainer_git_commit"] = "unknown"
    p = export_npz(ck["state_dict"], ck["y_std"], a.out, meta)
    print(f"wrote {p} ({p.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
