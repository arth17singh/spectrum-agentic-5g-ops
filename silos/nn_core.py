"""Service-neutral core of the Track 2 network-response model: layout, data loading, model, scoring.

SELF-CONTAINED ON PURPOSE: this file imports only numpy (torch lazily) and the standard library, so
the same bytes can be copied next to the federated model code and run inside the FL service's training
container. `silos/tests` checks that its constants equal `silos/features.py`.

Contents
  * the 38 input columns and 4 delta labels (data contract v0.2 + amendment 1), fixed ranges, encoding;
  * CSV loading of a silo file (no pandas);
  * the model: MLP 38-128-128-4 (`ResponseNet` for torch, `forward_np` for numpy), initial weights;
  * training helpers (plain torch loops: local-only, central, local FedAvg);
  * scoring: per-row scaled MAE, per-label MAE, direction accuracy, paired bootstrap;
  * a minimal safetensors reader and writer (numpy only) so weights files can be made and read without
    the `safetensors` package.
"""

from __future__ import annotations

import csv
import hashlib
import json
import struct
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# ------------------------------------------------------------------------------------------------
# layout (must equal silos/features.py; tested)
# ------------------------------------------------------------------------------------------------
N_CELLS = 4
CELL_FIELDS = ("user_share", "rsrp_mean_dbm", "rsrp_p10_dbm", "sinr_mean_db", "sinr_p10_db", "power_dbm", "tilt_deg")
CLUSTER_FIELDS = ("coverage_ratio", "sla_ratio")
CHANGE_FIELDS = ("d_power_db", "d_tilt_deg")
INPUT_COLUMNS = ([f"{f}[{i}]" for f in CELL_FIELDS for i in range(N_CELLS)] + list(CLUSTER_FIELDS)
                 + [f"{f}[{i}]" for f in CHANGE_FIELDS for i in range(N_CELLS)])
LABELS = ("d_coverage_ratio", "d_sla_ratio", "d_sinr_p10_db", "d_sinr_mean_db")
N_INPUTS = len(INPUT_COLUMNS)  # 38
RANGES = {
    "user_share": (0.0, 1.0), "rsrp_mean_dbm": (-140.0, -40.0), "rsrp_p10_dbm": (-140.0, -40.0),
    "sinr_mean_db": (-30.0, 40.0), "sinr_p10_db": (-30.0, 40.0), "power_dbm": (10.0, 46.0),
    "tilt_deg": (4.0, 15.0), "coverage_ratio": (0.0, 1.0), "sla_ratio": (0.0, 1.0),
    "d_power_db": (-6.0, 8.0), "d_tilt_deg": (-5.0, 5.0),
}
_LO = np.array([RANGES[c.split("[")[0]][0] for c in INPUT_COLUMNS])
_HI = np.array([RANGES[c.split("[")[0]][1] for c in INPUT_COLUMNS])

# Training scale of the labels (so the four losses are comparable) and the DECLARED scoring scale:
# per-row score = mean over labels of |prediction - truth| / SCORE_SCALE. The scoring scale is roughly the
# error of predicting "no effect" on the Viavi-like slice, so a score near 1 means "no better than no-op".
LABEL_SCALE = np.array([10.0, 10.0, 1.0, 1.0])
SCORE_SCALE = np.array([0.02, 0.03, 0.5, 0.35])
DEAD_BAND = np.array([0.01, 0.01, 0.1, 0.1])

HIDDEN = (128, 128)
SHAPES = {  # exact names and shapes of the exchanged weights
    "fc1.weight": (HIDDEN[0], N_INPUTS), "fc1.bias": (HIDDEN[0],),
    "fc2.weight": (HIDDEN[1], HIDDEN[0]), "fc2.bias": (HIDDEN[1],),
    "out.weight": (len(LABELS), HIDDEN[1]), "out.bias": (len(LABELS),),
}
SILO_OF_INDEX = {1: "A", 2: "B", 3: "C"}  # CLIENT_INDEX is 1-based, like the service's own example


def encode(x_raw: np.ndarray) -> np.ndarray:
    """Raw input rows -> [-1, 1] with the fixed contract ranges (no data-dependent statistics)."""
    return 2.0 * (np.asarray(x_raw, dtype=np.float64) - _LO) / (_HI - _LO) - 1.0


# ------------------------------------------------------------------------------------------------
# data
# ------------------------------------------------------------------------------------------------
def read_xy(path: str | Path, segments: Iterable[str] | None = None) -> dict[str, np.ndarray]:
    """Read a silo CSV: raw inputs X (n,38), delta labels Y (n,4), and the metadata the evaluation needs.

    Returns {"x", "y", "segment", "seed", "step", "silo"} as numpy arrays. Only these columns are read;
    nothing else in the file can reach a model.
    """
    keep = set(segments) if segments is not None else None
    xs, ys, seg, seed, step, silo = [], [], [], [], [], []
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        idx = {c: i for i, c in enumerate(header)}
        xi = [idx[c] for c in INPUT_COLUMNS]
        yi = [idx[c] for c in LABELS]
        si, ei, ti, oi = idx["segment"], idx["seed"], idx["step"], idx["silo"]
        for row in r:
            if keep is not None and row[si] not in keep:
                continue
            xs.append([float(row[i]) for i in xi])
            ys.append([float(row[i]) for i in yi])
            seg.append(row[si]); seed.append(int(row[ei])); step.append(int(row[ti])); silo.append(row[oi])
    return {"x": np.asarray(xs, dtype=np.float64).reshape(-1, N_INPUTS), "y": np.asarray(ys, dtype=np.float64).reshape(-1, len(LABELS)),
            "segment": np.asarray(seg), "seed": np.asarray(seed, dtype=np.int64), "step": np.asarray(step, dtype=np.int64),
            "silo": np.asarray(silo)}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ------------------------------------------------------------------------------------------------
# model
# ------------------------------------------------------------------------------------------------
def init_params(seed: int = 0) -> dict[str, np.ndarray]:
    """PyTorch-default (Kaiming-uniform) initial weights as float32 numpy, exact names and shapes of SHAPES."""
    rng = np.random.default_rng(seed)
    out: dict[str, np.ndarray] = {}
    for name, shape in SHAPES.items():
        fan_in = shape[1] if name.endswith("weight") else SHAPES[name.replace("bias", "weight")][1]
        bound = 1.0 / np.sqrt(fan_in)
        out[name] = rng.uniform(-bound, bound, size=shape).astype(np.float32)
    return out


def check_params(params: dict[str, Any]) -> None:
    """Raise unless the weights have exactly the names and shapes the service will aggregate."""
    if set(params) != set(SHAPES):
        raise ValueError(f"weight names {sorted(params)} != {sorted(SHAPES)}")
    for k, shape in SHAPES.items():
        if tuple(np.shape(params[k])) != shape:
            raise ValueError(f"{k}: shape {tuple(np.shape(params[k]))} != {shape}")


def forward_np(params: dict[str, np.ndarray], x_raw: np.ndarray) -> np.ndarray:
    """Predicted KPI deltas (unscaled), numpy only."""
    h = encode(x_raw)
    h = np.maximum(h @ params["fc1.weight"].T.astype(np.float64) + params["fc1.bias"], 0.0)
    h = np.maximum(h @ params["fc2.weight"].T.astype(np.float64) + params["fc2.bias"], 0.0)
    return (h @ params["out.weight"].T.astype(np.float64) + params["out.bias"]) / LABEL_SCALE


def torch_model():
    """The same network as a torch module; state_dict keys equal SHAPES (no wrapper prefixes)."""
    import torch
    from torch import nn

    class ResponseNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(N_INPUTS, HIDDEN[0])
            self.fc2 = nn.Linear(HIDDEN[0], HIDDEN[1])
            self.out = nn.Linear(HIDDEN[1], len(LABELS))

        def forward(self, x):
            return self.out(torch.relu(self.fc2(torch.relu(self.fc1(x)))))

    return ResponseNet()


def params_from_torch(model) -> dict[str, np.ndarray]:
    return {k: v.detach().cpu().numpy().astype(np.float32) for k, v in model.state_dict().items()}


def load_params_into(model, params: dict[str, np.ndarray]) -> None:
    import torch

    check_params(params)
    model.load_state_dict({k: torch.tensor(np.asarray(v, dtype=np.float32)) for k, v in params.items()})


# ------------------------------------------------------------------------------------------------
# training helpers (torch, CPU)
# ------------------------------------------------------------------------------------------------
def tensors(d: dict[str, np.ndarray]):
    import torch

    return (torch.tensor(encode(d["x"]), dtype=torch.float32), torch.tensor(d["y"] * LABEL_SCALE, dtype=torch.float32))


def train_epochs(model, x, y, epochs: int, seed: int, lr: float = 1e-3, batch: int = 256) -> None:
    import torch
    from torch import nn

    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.HuberLoss(delta=1.0)
    n = len(x)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            opt.zero_grad()
            loss_fn(model(x[idx]), y[idx]).backward()
            opt.step()


def fedavg(data: dict[str, tuple], rounds: int, seed: int, init: dict[str, np.ndarray] | None = None):
    """Plain FedAvg run locally in one process (NOT the FL service). Row-count weighted, one local epoch per round."""
    import copy

    glob = torch_model()
    if init is not None:
        load_params_into(glob, init)
    total = sum(len(x) for x, _ in data.values())
    for r in range(rounds):
        states, weights = [], []
        for k, (silo, (x, y)) in enumerate(sorted(data.items())):
            local = copy.deepcopy(glob)
            train_epochs(local, x, y, 1, seed * 1000 + r * 10 + k)
            states.append(local.state_dict())
            weights.append(len(x) / total)
        glob.load_state_dict({k: sum(w * s[k] for w, s in zip(weights, states)) for k in states[0]})
    return glob


# ------------------------------------------------------------------------------------------------
# scoring
# ------------------------------------------------------------------------------------------------
def row_scores(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-row scaled MAE: mean over the four labels of |pred - truth| / SCORE_SCALE."""
    return (np.abs(np.asarray(pred) - np.asarray(y)) / SCORE_SCALE).mean(axis=1)


def label_mae(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {l: float(v) for l, v in zip(LABELS, np.abs(pred - y).mean(axis=0))}


def direction_accuracy(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Share of rows with a clearly non-zero true delta (|delta| >= DEAD_BAND) whose predicted sign matches."""
    out = {}
    for j, l in enumerate(LABELS):
        m = np.abs(y[:, j]) >= DEAD_BAND[j]
        out[l] = float((np.sign(pred[m, j]) == np.sign(y[m, j])).mean()) if m.any() else float("nan")
    return out


def paired_bootstrap(diff: np.ndarray, groups: np.ndarray, n_boot: int = 2000, seed: int = 0, level: float = 0.95) -> dict[str, Any]:
    """Paired bootstrap of the mean of per-row differences, resampling WHOLE GROUPS (clusters) with replacement,
    because rows of one cluster are correlated. Also reports the row-level interval (overstates precision)."""
    diff = np.asarray(diff, dtype=np.float64)
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    g_sum = np.bincount(inv, weights=diff, minlength=len(uniq))
    g_n = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
    idx = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    boots = g_sum[idx].sum(axis=1) / g_n[idx].sum(axis=1)
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(boots, [a, 1.0 - a])
    row = np.empty(n_boot)
    for c in range(0, n_boot, 200):
        k = min(200, n_boot - c)
        row[c : c + k] = diff[rng.integers(0, len(diff), size=(k, len(diff)))].mean(axis=1)
    rlo, rhi = np.quantile(row, [a, 1.0 - a])
    return {"mean": float(diff.mean()), "ci_lo": float(lo), "ci_hi": float(hi), "n_groups": int(len(uniq)), "n_rows": int(len(diff)),
            "level": level, "n_boot": n_boot, "seed": seed, "method": "paired bootstrap resampling clusters (seeds)",
            "row_level_ci": [float(rlo), float(rhi)]}


# ------------------------------------------------------------------------------------------------
# minimal safetensors (numpy only)
# ------------------------------------------------------------------------------------------------
_DT = {"F32": np.dtype("<f4"), "F64": np.dtype("<f8")}


def safetensors_dumps(tensors_: dict[str, np.ndarray], metadata: dict[str, str] | None = None) -> bytes:
    header: dict[str, Any] = {}
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    blobs, off = [], 0
    for name in sorted(tensors_):
        a = np.ascontiguousarray(tensors_[name])
        code = {np.dtype("float32"): "F32", np.dtype("float64"): "F64"}[a.dtype]
        raw = a.astype(_DT[code]).tobytes()
        header[name] = {"dtype": code, "shape": list(a.shape), "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    return struct.pack("<Q", len(hb)) + hb + b"".join(blobs)


def safetensors_loads(data: bytes) -> dict[str, np.ndarray]:
    (n,) = struct.unpack("<Q", data[:8])
    header = json.loads(data[8 : 8 + n])
    body = data[8 + n :]
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        b, e = meta["data_offsets"]
        native = np.float64 if meta["dtype"] == "F64" else np.float32
        out[name] = np.frombuffer(body[b:e], dtype=_DT[meta["dtype"]]).reshape(meta["shape"]).astype(native)
    return out
