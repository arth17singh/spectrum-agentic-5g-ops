"""Short PPO run on the in-process Maveric twin, through the SDK's own environment.

v0, local, not federated / not the official job.

    python -m t1.rl.train --out <dir> [--minutes 30] [--envs 8] [--seed 42]

Uses `ran_rl.make_env(twin="maveric")` (SDK Default Reward for the CCO use case, SDK safety
filter), wrapped only by `t1.rl.obs.MaskedObs` (see that module). Training draws a seeded
random scenario per episode from every scenario the CCO use case allows, including outage.
Stops after `--minutes` of wall clock, saves `<out>/ppo_v0.zip` and `<out>/train_log.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

LABEL = "v0, local, not federated / not the official job"
TRAIN_SCENARIOS = ["low_load", "nominal", "congestion", "interference", "mobility_stress", "outage"]


def make_one():
    import ran_rl
    from ran_rl.env.config import EnvConfig
    from stable_baselines3.common.monitor import Monitor

    from t1.rl.obs import MaskedObs

    cfg = EnvConfig(scenarios=TRAIN_SCENARIOS)
    return Monitor(MaskedObs(ran_rl.make_env(twin="maveric", env_config=cfg)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-timesteps", type=int, default=5_000_000)
    ap.add_argument("--n-steps", type=int, default=512, help="rollout length per env")
    a = ap.parse_args(argv)

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    torch.set_num_threads(2)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    budget_s = a.minutes * 60.0

    class TimeBox(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self.t0 = time.time()
            self.log: list[dict] = []
            self.stopped_by = "max_timesteps"

        def _on_rollout_end(self) -> None:
            ep = [e["r"] for e in self.model.ep_info_buffer]
            self.log.append({"timesteps": int(self.num_timesteps), "wall_s": round(time.time() - self.t0, 1),
                             "ep_rew_mean": float(sum(ep) / len(ep)) if ep else None})
            print(self.log[-1], flush=True)

        def _on_step(self) -> bool:
            if time.time() - self.t0 > budget_s:
                self.stopped_by = "time_box"
                return False
            return True

    venv = (SubprocVecEnv if a.envs > 1 else DummyVecEnv)([make_one] * a.envs)
    venv.seed(a.seed)
    model = PPO("MultiInputPolicy", venv, device="cpu", seed=a.seed, learning_rate=3e-4, gamma=0.99,
                gae_lambda=0.95, clip_range=0.2, n_steps=a.n_steps, batch_size=256,
                policy_kwargs={"log_std_init": -0.5}, verbose=0)
    cb = TimeBox()
    t0 = time.time()
    model.learn(a.max_timesteps, callback=cb)
    wall = time.time() - t0
    model.save(out / "ppo_v0.zip")
    venv.close()
    (out / "train_log.json").write_text(json.dumps({
        "label": LABEL, "algorithm": "PPO (stable-baselines3 MultiInputPolicy)", "n_envs": a.envs,
        "n_steps_per_env": a.n_steps, "batch_size": 256, "seed": a.seed, "log_std_init": -0.5,
        "timesteps": int(model.num_timesteps), "wall_seconds": round(wall, 1), "stopped_by": cb.stopped_by,
        "scenarios": TRAIN_SCENARIOS, "curve": cb.log}, indent=2) + "\n")
    print(f"saved {out / 'ppo_v0.zip'} after {model.num_timesteps} steps in {wall:.0f}s ({cb.stopped_by})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
