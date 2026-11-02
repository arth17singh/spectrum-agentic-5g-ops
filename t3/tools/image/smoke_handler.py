"""Smoke handler for the heavy-tools image (CMD smoke.smoke_handler.handler).

Proves, inside the Lambda runtime on arm64, that torch, gpytorch, the shipped Maveric GP models,
the ran_rl SDK and stable-baselines3 all work. It deliberately does not import any of our own
packages (the real handler is t3.tools.app.handler).

Event (all optional): {"steps": 10, "seed": 1, "scenario": "nominal", "ppo": true}
  - builds the in-process Maveric twin, resets with seed/scenario;
  - applies two explicit changes (cell 0 power +4 dB, cell 0 tilt -2 deg; then cell 2 tilt +2 deg),
    then replays zero-change steps up to `steps` in total; each step goes through the SDK's safety
    filter and advances the twin one step, exactly what a stateless tool does on replay;
  - returns the final KPIs, timings and peak RSS.
"""

import importlib.metadata
import os
import resource
import time

_T_START = time.perf_counter()
import numpy as np  # noqa: E402

_T_NUMPY = time.perf_counter()
import torch  # noqa: E402

_T_TORCH = time.perf_counter()
import gpytorch  # noqa: E402
import ran_rl  # noqa: E402
import stable_baselines3  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

_T_IMPORTS = time.perf_counter()
_IMPORT_S = {
    "numpy": round(_T_NUMPY - _T_START, 2),
    "torch": round(_T_TORCH - _T_NUMPY, 2),
    "gpytorch_ran_rl_sb3": round(_T_IMPORTS - _T_TORCH, 2),
    "total": round(_T_IMPORTS - _T_START, 2),
}

_PPO_PATH = os.path.join(os.path.dirname(__file__), "tiny_ppo.zip")
_PPO = None
_ENV_CACHE = {}


def _peak_rss_mb() -> float:
    # Linux ru_maxrss is in KiB.
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def _kpis(state: dict) -> dict:
    return {
        "coverage_ratio": state["coverage_ratio"],
        "sla_ratio": state["sla_ratio"],
        "sinr_p10_db": float(np.mean(state["sinr_p10_db"])),
        "sinr_mean_db": float(np.mean(state["sinr_mean_db"])),
        "power_dbm": state["power_dbm"],
        "tilt_deg": state["tilt_deg"],
    }


def handler(event, context):
    global _PPO
    event = event or {}
    steps = int(event.get("steps", 10))
    seed = int(event.get("seed", 1))
    scenario = event.get("scenario", "nominal")
    cold = not _ENV_CACHE.get("warm")
    _ENV_CACHE["warm"] = True
    t0 = time.perf_counter()

    env = ran_rl.make_env(twin="maveric")
    t_build = time.perf_counter()
    obs, info = env.reset(seed=seed, options={"scenario": scenario})
    t_reset = time.perf_counter()
    start = _kpis(info["raw_state"])

    n = env.action_space.shape[0] // 2
    changes = []
    a1 = np.zeros(env.action_space.shape, dtype=np.float32)
    a1[0] = 0.5  # cell 0 power
    a1[n] = -0.5  # cell 0 tilt
    a2 = np.zeros(env.action_space.shape, dtype=np.float32)
    a2[n + 2] = 0.5  # cell 2 tilt
    changes = [a1, a2]
    state = info["raw_state"]
    for i in range(steps):
        action = changes[i] if i < len(changes) else np.zeros_like(a1)
        obs, _r, _te, _tr, step_info = env.step(action)
        state = step_info["raw_state"]
    t_steps = time.perf_counter()

    ppo_out = None
    if event.get("ppo", True):
        t_p0 = time.perf_counter()
        if _PPO is None:
            _PPO = PPO.load(_PPO_PATH, device="cpu")
        act, _ = _PPO.predict(obs, deterministic=True)
        ppo_out = {
            "action_len": int(len(act)),
            "action": [round(float(x), 4) for x in act],
            "load_and_predict_s": round(time.perf_counter() - t_p0, 3),
        }
    env.close()

    return {
        "ok": True,
        "cold_container": cold,
        "versions": {
            "torch": torch.__version__,
            "gpytorch": gpytorch.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "numpy": np.__version__,
            "ran_rl": importlib.metadata.version("ran-rl"),
            "torch_threads": torch.get_num_threads(),
        },
        "import_s": _IMPORT_S,
        "steps_applied": steps,
        "timing_s": {
            "build_twin": round(t_build - t0, 3),
            "reset": round(t_reset - t_build, 3),
            "steps": round(t_steps - t_reset, 3),
            "per_step": round((t_steps - t_reset) / max(steps, 1), 4),
            "total": round(time.perf_counter() - t0, 3),
        },
        "peak_rss_mb": _peak_rss_mb(),
        "start": start,
        "end": _kpis(state),
        "ppo": ppo_out,
    }
