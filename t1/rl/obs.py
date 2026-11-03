"""Observation handling shared by PPO training and `Policy.propose` (v0, local, not federated).

The SDK observation (ran_rl.env.observation.normalize_observation) carries `hour`, `phase`,
`prev_action` and an absolute `ue_count`. A tool call to `propose` is stateless and its 4-cell
state (data contract v0.2) has none of those, so v0 trains and runs the policy on a MASKED view:

    hour, phase, prev_action   -> constant 0
    ue_count                   -> the cell's SHARE of users, mapped to [-1, 1] (2*share - 1)

Everything else (power, tilt, rsrp/sinr mean and p10, coverage, sla, alarms) is untouched.
The same `mask_obs` is applied in training (via `MaskedObs`) and at inference.
"""

from __future__ import annotations

from typing import Any, Mapping

import gymnasium as gym
import numpy as np

UE_COUNT_MAX = 200.0  # EnvConfig().observation.ue_count_max


def mask_obs(obs: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {k: np.array(v, copy=True) for k, v in obs.items()}
    counts = (out["ue_count"].astype(np.float64) + 1.0) / 2.0 * UE_COUNT_MAX
    total = counts.sum()
    share = counts / total if total > 0 else np.full_like(counts, 1.0 / len(counts))
    out["ue_count"] = (2.0 * share - 1.0).astype(np.float32)
    for k in ("hour", "phase", "prev_action"):
        out[k] = np.zeros_like(out[k])
    return out


class MaskedObs(gym.ObservationWrapper):
    """gym wrapper applying `mask_obs` to the SDK env's observation (spaces unchanged)."""

    def observation(self, observation: Any) -> dict[str, np.ndarray]:
        return mask_obs(observation)
