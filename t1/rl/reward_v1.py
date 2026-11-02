"""Reward v1 for the Track 1 official job: the SDK Default Reward plus a coverage term and a coverage floor.

v0, local, not federated. Copied by `t1/rl/official/run.py` to `team/rewards.py` of the SageMaker project, where
`config/reward.yaml` selects it as `team.rewards:coverage_aware_reward`.

Why: the Default Reward and the Scoring Objective do not pay for coverage, so PPO v0 raised its objective by
cutting power (coverage 0.42 against the rule baseline's 0.83). This reward adds
    + coverage * coverage_ratio
    - floor_weight * max(0, floor - coverage_ratio)
to the Default Reward, so giving coverage away costs more than the energy it saves. The score stays the fixed
Scoring Objective; only the Training Reward changes.
"""

from __future__ import annotations

from ran_rl import StepContext
from ran_rl.rewards.default import default_reward


def coverage_aware_reward(
    ctx: StepContext,
    coverage: float = 1.0,
    floor: float = 0.85,
    floor_weight: float = 4.0,
    sla: float = 1.0,
    spectral_efficiency: float = 0.5,
    energy: float = 0.3,
    balance: float = 0.2,
    violation: float = 1.0,
    churn: float = 0.1,
) -> tuple[float, dict[str, float]]:
    """Default Reward + `coverage` * coverage_ratio - `floor_weight` * shortfall below `floor`.

    Returns `(total, components)`: the Default Reward's six components plus `coverage_score` and
    `coverage_shortfall`.
    """
    total, components = default_reward(ctx, sla, spectral_efficiency, energy, balance, violation, churn)
    cov = float(ctx.state["coverage_ratio"])
    shortfall = max(0.0, floor - cov)
    return total + coverage * cov - floor_weight * shortfall, components | {
        "coverage_score": cov,
        "coverage_shortfall": shortfall,
    }
