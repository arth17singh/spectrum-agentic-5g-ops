"""Build-time helper: train a tiny PPO for about one second on the SDK's local twin and save it.

Only proves that stable-baselines3 imports, trains, saves and (in the smoke handler) loads and
predicts inside the image. It is not a model anyone should use.

Usage: python train_tiny_ppo.py /path/to/tiny_ppo.zip
"""

import sys
import time


def main(out: str) -> None:
    import ran_rl
    from stable_baselines3 import PPO

    env = ran_rl.make_env(twin="local")
    model = PPO("MultiInputPolicy", env, n_steps=48, batch_size=48, n_epochs=2, seed=0, device="cpu")
    t0 = time.time()
    model.learn(total_timesteps=96)
    print(f"tiny PPO trained in {time.time() - t0:.2f}s -> {out}")
    model.save(out)


if __name__ == "__main__":
    main(sys.argv[1])
