"""Train the Track 2 response model (small MLP, torch). v0, local, not federated / not the official job.

    python -m t2.train --data <dataset.npz> --out <ckpt.pt> [--epochs 200] [--seed 0]

Trains on the TRAIN seeds only (a hash-based 10% of them is held out as validation for early
stopping); the test seeds are never read here. The network predicts the standardised KPI DELTA
(after the change minus the no-op counterfactual at the same step), anchored: net(x) - net(x with no change).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from t2.features import col
from t2.model_torch import ResponseMLP, encoded, residual_targets

_CHANGE_COLS = [col(f, i) for f in ("d_power_db", "d_tilt_deg") for i in range(4)]


def _zero_change(x: np.ndarray) -> np.ndarray:
    z = np.array(x, copy=True)
    z[:, _CHANGE_COLS] = 0.0
    return z


def is_val_seed(seed: int) -> bool:
    return int(hashlib.sha256(f"val:{int(seed)}".encode()).hexdigest(), 16) % 10 == 0


def fit(x_tr: np.ndarray, y_tr: np.ndarray, x_va: np.ndarray, y_va: np.ndarray, *, hidden=(128, 128), anchored: bool = True,
        epochs: int = 200, lr: float = 2e-3, batch: int = 256, patience: int = 30, seed: int = 0,
        verbose: bool = False) -> dict[str, Any]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    r_tr, r_va = residual_targets(x_tr, y_tr), residual_targets(x_va, y_va)
    y_std = np.maximum(r_tr.std(axis=0), 1e-6)
    xt, yt = torch.tensor(encoded(x_tr)), torch.tensor((r_tr / y_std).astype(np.float32))
    xv, yv = torch.tensor(encoded(x_va)), torch.tensor((r_va / y_std).astype(np.float32))
    x0t, x0v = torch.tensor(encoded(_zero_change(x_tr))), torch.tensor(encoded(_zero_change(x_va)))
    net = ResponseMLP(tuple(hidden))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()
    best, best_state, bad, hist = float("inf"), None, 0, []
    for ep in range(epochs):
        net.train()
        perm = torch.tensor(rng.permutation(len(xt)))
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            out = net(xt[idx]) - net(x0t[idx]) if anchored else net(xt[idx])
            loss_fn(out, yt[idx]).backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            tr = loss_fn(net(xt) - net(x0t) if anchored else net(xt), yt).item()
            va = loss_fn(net(xv) - net(x0v) if anchored else net(xv), yv).item()
        hist.append({"epoch": ep, "train_mse": tr, "val_mse": va})
        if verbose and ep % 10 == 0:
            print(hist[-1], flush=True)
        if va < best - 1e-5:
            best, best_state, bad = va, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    return {"state_dict": {k: v.clone() for k, v in net.state_dict().items()}, "hidden": tuple(hidden),
            "y_std": y_std, "best_val_mse": best, "epochs_run": len(hist), "history": hist}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, nargs="+", default=[128, 128])
    a = ap.parse_args(argv)
    with np.load(a.data, allow_pickle=False) as z:
        x, y, seed, split = z["x"], z["y"], z["seed"], z["split"]
    train_mask = split == 0  # test rows (split == 1) are not touched
    val_mask = np.array([is_val_seed(s) for s in seed]) & train_mask
    tr_mask = train_mask & ~val_mask
    t0 = time.time()
    res = fit(x[tr_mask], y[tr_mask], x[val_mask], y[val_mask], hidden=tuple(a.hidden), epochs=a.epochs,
              seed=a.seed, verbose=True)
    res.update({"n_train": int(tr_mask.sum()), "n_val": int(val_mask.sum()), "seed": a.seed,
                "train_seconds": round(time.time() - t0, 1),
                "dataset_sha256": hashlib.sha256(Path(a.data).read_bytes()).hexdigest()})
    torch.save(res, a.out)
    print(json.dumps({k: v for k, v in res.items() if k not in ("state_dict", "history", "y_std")}, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
