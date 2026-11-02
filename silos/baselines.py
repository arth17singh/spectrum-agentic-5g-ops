"""A quick, honest preview: local-only vs federated averaging vs central, on both silo splits.

    python -m silos.baselines --data <dir> [--seeds 3] [--epochs 20] [--out silos/baselines_results.json]

NOT the FL service. "Federated" here is plain FedAvg run locally in one process over the three silos
(same model code, same data), weighted by each silo's row count, so the large silo dominates the
average. It tells us whether the federation can possibly win on this data before the service is used.

Models: one small MLP (38 -> 128 -> 128 -> 4, ReLU), inputs scaled to [-1, 1] with the fixed contract
ranges, labels multiplied by fixed factors (LABEL_SCALE) so the four losses are comparable, Huber loss,
Adam 1e-3, batch 256. Every method sees the same number of passes over each silo's data
(EPOCHS epochs; FedAvg = EPOCHS rounds of 1 local epoch each). Reference predictor: "no-op" = predict
a zero delta (what Maveric would say if the change did nothing).

Metrics on the shared held-out test (per segment): MAE per label in label units (ratios, dB); `skill` =
mean over labels of MAE / MAE(no-op), below 1 beats predicting no effect; direction accuracy = share
of rows with a clearly non-zero true delta (|delta| >= DEAD_BAND) whose predicted sign matches,
averaged over the four labels (the no-op predictor scores 0 by construction).
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from silos.features import INPUT_COLUMNS, LABELS, input_range

LABEL_SCALE = np.array([10.0, 10.0, 1.0, 1.0])  # coverage, sla, sinr_p10, sinr_mean deltas
DEAD_BAND = np.array([0.01, 0.01, 0.1, 0.1])  # a true delta smaller than this has no clear direction
SPLITS = ("disjoint", "overlap")
SILOS = ("A", "B", "C")
SEGMENTS = ("viavi_like", "viavi_default_noise", "region_A", "region_B", "region_C")
_LO = np.array([input_range(c)[0] for c in INPUT_COLUMNS])
_HI = np.array([input_range(c)[1] for c in INPUT_COLUMNS])


def encode(df: pd.DataFrame) -> torch.Tensor:
    x = 2.0 * (df[INPUT_COLUMNS].to_numpy(dtype=np.float64) - _LO) / (_HI - _LO) - 1.0
    return torch.tensor(x, dtype=torch.float32)


def targets(df: pd.DataFrame) -> torch.Tensor:
    return torch.tensor(df[list(LABELS)].to_numpy(dtype=np.float64) * LABEL_SCALE, dtype=torch.float32)


def make_model(seed: int) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(len(INPUT_COLUMNS), 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 4))


def train_epochs(model: nn.Module, x: torch.Tensor, y: torch.Tensor, epochs: int, seed: int, lr: float = 1e-3) -> None:
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.HuberLoss(delta=1.0)
    n = len(x)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, 256):
            idx = perm[i : i + 256]
            opt.zero_grad()
            loss_fn(model(x[idx]), y[idx]).backward()
            opt.step()


def fedavg(data: dict[str, tuple[torch.Tensor, torch.Tensor]], rounds: int, seed: int) -> nn.Module:
    """Plain FedAvg, run locally. Weighted by row count; one local epoch per round; optimiser state reset."""
    glob = make_model(seed)
    total = sum(len(x) for x, _ in data.values())
    for r in range(rounds):
        states, weights = [], []
        for k, (silo, (x, y)) in enumerate(sorted(data.items())):
            local = copy.deepcopy(glob)
            train_epochs(local, x, y, 1, seed * 1000 + r * 10 + k)
            states.append(local.state_dict())
            weights.append(len(x) / total)
        avg = {k: sum(w * s[k] for w, s in zip(weights, states)) for k in states[0]}
        glob.load_state_dict(avg)
    return glob


@torch.no_grad()
def evaluate(model: nn.Module | None, test: pd.DataFrame) -> dict[str, dict]:
    """Per segment: MAE per label (label units), skill vs the no-op predictor, direction accuracy."""
    out = {}
    for seg in SEGMENTS:
        d = test[test["segment"] == seg]
        y = d[list(LABELS)].to_numpy(dtype=np.float64)
        pred = np.zeros_like(y) if model is None else model(encode(d)).numpy().astype(np.float64) / LABEL_SCALE
        mae = np.abs(pred - y).mean(axis=0)
        mae0 = np.abs(y).mean(axis=0)
        acc = []
        for j in range(4):
            m = np.abs(y[:, j]) >= DEAD_BAND[j]
            acc.append(float((np.sign(pred[m, j]) == np.sign(y[m, j])).mean()) if m.any() else float("nan"))
        out[seg] = {
            "mae": {l: float(v) for l, v in zip(LABELS, mae)},
            "skill": float(np.mean(mae / np.maximum(mae0, 1e-12))),
            "direction_accuracy": float(np.nanmean(acc)),
            "direction_accuracy_by_label": {l: v for l, v in zip(LABELS, acc)},
            "n_rows": int(len(d)),
        }
    return out


def run(data_dir: Path, seeds: int, epochs: int) -> dict:
    results: dict = {"splits": {}}
    for split in SPLITS:
        tr = {s: pd.read_csv(data_dir / split / s / "train.csv") for s in SILOS}
        test = pd.read_csv(data_dir / split / "shared_test" / "test.csv")
        enc = {s: (encode(d), targets(d)) for s, d in tr.items()}
        pooled = (torch.cat([enc[s][0] for s in SILOS]), torch.cat([enc[s][1] for s in SILOS]))
        per_method: dict[str, list] = {}
        for seed in range(seeds):
            models: dict[str, nn.Module | None] = {"no-op (zero delta)": None}
            for s in SILOS:
                m = make_model(seed)
                train_epochs(m, *enc[s], epochs, seed)
                models[f"local-only {s}"] = m
            models["local FedAvg, not the FL service (row-weighted)"] = fedavg(enc, epochs, seed)
            m = make_model(seed)
            train_epochs(m, *pooled, epochs, seed)
            models["central (pooled data)"] = m
            for name, model in models.items():
                per_method.setdefault(name, []).append(evaluate(model, test))
        agg = {}
        for name, runs in per_method.items():
            agg[name] = {}
            for seg in SEGMENTS:
                def stat(f):
                    v = np.array([f(r[seg]) for r in runs])
                    return [float(v.mean()), float(v.std())]
                agg[name][seg] = {
                    "skill": stat(lambda r: r["skill"]),
                    "direction_accuracy": stat(lambda r: r["direction_accuracy"]),
                    "mae": {l: stat(lambda r, l=l: r["mae"][l]) for l in LABELS},
                    "n_rows": runs[0][seg]["n_rows"],
                }
        results["splits"][split] = agg
    results["config"] = {"seeds": seeds, "epochs": epochs, "label_scale": LABEL_SCALE.tolist(), "dead_band": DEAD_BAND.tolist(),
                         "model": "MLP 38-128-128-4", "label": "local FedAvg, not the FL service"}
    return results


def verdict(agg: dict) -> dict:
    """Does the federation beat every local-only model on each segment (skill, lower is better)?"""
    fed = "local FedAvg, not the FL service (row-weighted)"
    out = {}
    for seg in SEGMENTS:
        locals_ = {k: v[seg]["skill"][0] for k, v in agg.items() if k.startswith("local-only")}
        out[seg] = {
            "fedavg": agg[fed][seg]["skill"][0], "best_local": min(locals_.values()), "best_local_is": min(locals_, key=locals_.get),
            "fedavg_beats_every_local": agg[fed][seg]["skill"][0] < min(locals_.values()),
            "central": agg["central (pooled data)"][seg]["skill"][0],
        }
    return out


def print_tables(results: dict) -> None:
    for split, agg in results["splits"].items():
        print(f"\n=== split: {split}  (skill = mean MAE / MAE of the no-op, lower is better; dir = direction accuracy; mean over seeds) ===")
        print(f'{"method":52s}' + "".join(f"{s:>22s}" for s in SEGMENTS))
        for name, segs in agg.items():
            print(f"{name:52s}" + "".join(f'{segs[s]["skill"][0]:>12.3f} /{segs[s]["direction_accuracy"][0]:>6.2f}  ' for s in SEGMENTS))
        v = verdict(agg)
        print("FedAvg beats every local-only model:", {s: bool(x["fedavg_beats_every_local"]) for s, x in v.items()})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    torch.set_num_threads(4)
    t = time.time()
    res = run(a.data, a.seeds, a.epochs)
    res["verdict"] = {sp: verdict(agg) for sp, agg in res["splits"].items()}
    res["seconds"] = round(time.time() - t, 1)
    print_tables(res)
    if a.out:
        a.out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
