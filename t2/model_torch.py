"""The torch side of the Track 2 response model (training only; inference is t2/predict.py)."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from t2.features import N_FEATURES, encode


class ResponseMLP(nn.Module):
    def __init__(self, hidden: tuple[int, ...] = (128, 128), n_out: int = 4):
        super().__init__()
        dims = (N_FEATURES, *hidden, n_out)
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: encoded features
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return x


def residual_targets(x_raw: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Training targets. The dataset labels are already KPI deltas vs the no-op counterfactual
    (contract amendment 1, A5), so this is the identity; x_raw is accepted for symmetry."""
    return np.asarray(y, dtype=np.float64)


def encoded(x_raw: np.ndarray) -> np.ndarray:
    return encode(x_raw).astype(np.float32)
