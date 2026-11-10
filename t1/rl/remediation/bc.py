"""Behaviour cloning (+ DAgger) of the twin's label into the SAME SB3 PPO actor as ppo_sm_50k.

The model is a stable-baselines3 PPO with ppo_sm_50k's observation space, action space and policy
kwargs (MultiInputPolicy, a 64-64 tanh actor, log_std_init -0.5), so ``t1.rl.policy.load`` serves it
unchanged. Only the actor is trained, by regression of its mean (the deterministic action) onto
``data.unit_target(label)``; the value head and log_std are left as built and unused at inference.

DAgger: after each round the current policy is rolled out through ``t1.rl.policy.Policy`` (quantise,
deadzone, safety filter, exactly as served) on the train incidents, from the faulted and from the
healthy network; every state it reaches is labelled by the twin (the way back to the commissioned
configuration) and added. That teaches it to finish a partial fix and to undo its own wrong-cell changes.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from t1.rl.remediation import data as D

REPO = Path(__file__).resolve().parents[3]
BASE_ARTIFACT = REPO / "t1" / "rl" / "artifacts" / "ppo_sm_50k.zip"
MODEL_ID = "t1-remediation-bc"
OBS_MODE = "masked"
POSTPROCESS = {"deadzone_power_db": 1.0, "deadzone_tilt_deg": 1.0}
SEED = 20261104
HYPER = {"lr": 1e-3, "batch": 256, "epochs": 300, "patience": 30, "dagger_rounds": 2, "dagger_epochs": 150,
         "dagger_patience": 20, "seed": SEED, "loss": "mse on the actor mean (8 dims), targets data.unit_target"}


def _gym_env(obs_space: Any, act_space: Any):
    import gymnasium as gym

    class SpacesEnv(gym.Env):
        """Carries the spaces only: the model is never trained by interaction here."""

        def __init__(self) -> None:
            self.observation_space, self.action_space = obs_space, act_space

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return self.observation_space.sample(), {}

        def step(self, action):
            raise RuntimeError("SpacesEnv carries spaces only")

    return SpacesEnv()


def base_spaces(base: Path = BASE_ARTIFACT) -> tuple[Any, Any, dict[str, Any]]:
    from stable_baselines3 import PPO

    m = PPO.load(str(base), device="cpu")
    return m.observation_space, m.action_space, dict(m.policy_kwargs or {})


def new_model(seed: int = SEED, base: Path = BASE_ARTIFACT):
    """A fresh PPO with ppo_sm_50k's spaces and policy kwargs (its weights are NOT copied)."""
    import torch
    from stable_baselines3 import PPO

    torch.manual_seed(seed)
    obs_space, act_space, kwargs = base_spaces(base)
    return PPO("MultiInputPolicy", _gym_env(obs_space, act_space), policy_kwargs=kwargs, seed=seed, device="cpu",
               verbose=0)


def actor_mean(policy: Any, obs: dict[str, Any]):
    """The deterministic action before clipping: SB3's own path (preprocess, pi features, actor, action_net)."""
    from stable_baselines3.common.preprocessing import preprocess_obs

    pre = preprocess_obs(obs, policy.observation_space, normalize_images=policy.normalize_images)
    return policy.action_net(policy.mlp_extractor.forward_actor(policy.pi_features_extractor(pre)))


def _tensors(obs: dict[str, np.ndarray], idx: np.ndarray | None = None) -> dict[str, Any]:
    import torch

    return {k: torch.as_tensor(v if idx is None else v[idx]) for k, v in obs.items()}


def val_loss(policy: Any, obs: dict[str, np.ndarray], target: np.ndarray) -> float:
    import torch

    if len(target) == 0:
        return float("nan")
    with torch.no_grad():
        mean = actor_mean(policy, _tensors(obs))
        return float(torch.mean((mean - torch.as_tensor(target)) ** 2))


def fit(model: Any, train: dict[str, Any], val: dict[str, Any], *, epochs: int, patience: int, lr: float,
        batch: int, seed: int, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Minibatch regression of the actor mean; keeps the weights with the best validation loss."""
    import torch

    pol = model.policy
    params = list(pol.mlp_extractor.policy_net.parameters()) + list(pol.action_net.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    rng = np.random.default_rng(seed)
    obs, target = train["obs"], np.asarray(train["target"], dtype=np.float32)
    n = len(target)
    best, best_state, since, hist = float("inf"), None, 0, []
    t0 = time.time()
    for ep in range(epochs):
        order = rng.permutation(n)
        tot = 0.0
        for i in range(0, n, batch):
            idx = order[i:i + batch]
            mean = actor_mean(pol, _tensors(obs, idx))
            loss = torch.mean((mean - torch.as_tensor(target[idx])) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
        v = val_loss(pol, val["obs"], np.asarray(val["target"], dtype=np.float32)) if len(val["target"]) else tot / n
        hist.append({"epoch": ep, "train_mse": tot / n, "val_mse": v})
        if v < best - 1e-7:
            best, best_state, since = v, copy.deepcopy(pol.state_dict()), 0
        else:
            since += 1
        if ep % 25 == 0:
            log(f"[bc] epoch {ep} train_mse {tot / n:.5f} val_mse {v:.5f}")
        if since >= patience:
            break
    if best_state is not None:
        pol.load_state_dict(best_state)
    return {"epochs_run": len(hist), "best_val_mse": best, "final_train_mse": hist[-1]["train_mse"] if hist else None,
            "wall_s": round(time.time() - t0, 2), "history": hist}


def policy_of(model: Any, version: str) -> Any:
    """The model behind t1.rl.policy.Policy, with the meta it will ship with."""
    from t1.rl.policy import Policy

    return Policy(model, meta_for(version))


def meta_for(version: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    m = {"model_id": MODEL_ID, "version": version, "obs": OBS_MODE, "postprocess": dict(POSTPROCESS),
         "label": "remediation-v1: behaviour cloning (+DAgger) of the twin-verified way back to the commissioned "
                  "configuration on fresh dev-split incidents; ppo_sm_50k's SB3 PPO actor architecture; deterministic"}
    if extra:
        m.update(extra)
    return m


# --------------------------------------------------------------------------- DAgger rollouts

def _rollout_one(spec: dict[str, Any], targets: dict[str, Any], policy: Any) -> list[dict[str, Any]]:
    """The states the policy reaches (before its 2nd and 3rd decisions) from the faulted and healthy network."""
    comm = D.commissioned_config(spec)
    out = []
    for with_fault, tag in ((True, "fault"), (False, "healthy")):
        tw, s = D.run_script(spec, targets, with_fault=with_fault)
        for step in (3, 4):
            s = tw.apply(policy.propose(s))
            out.append({"kind": f"dagger-{tag}@{step + 1}", "state": D.jsonable_state(s),
                        "label": D.label_for(s, comm).tolist()})
    return out


def _rollout_work(args: tuple[list[dict[str, Any]], dict[str, Any], str, dict[str, Any]]) -> list[dict[str, Any]]:
    specs, targets, model_path, meta = args
    from stable_baselines3 import PPO

    from t1.rl.policy import Policy
    from t3.evals.twin_backend import single_thread

    single_thread()
    policy = Policy(PPO.load(model_path, device="cpu"), meta)
    return [{"seed": s["seed"], "role": s["role"], "kept": True, "samples": _rollout_one(s, targets, policy)}
            for s in specs]


def rollouts(specs: list[dict[str, Any]], targets: dict[str, Any], model_path: str, meta: dict[str, Any],
             workers: int = 1, chunk: int = 25) -> list[dict[str, Any]]:
    chunks = [specs[i:i + chunk] for i in range(0, len(specs), chunk)]
    if workers <= 1:
        parts = [_rollout_work((c, targets, model_path, meta)) for c in chunks]
    else:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers) as pool:
            parts = list(pool.map(_rollout_work, [(c, targets, model_path, meta) for c in chunks]))
    return [r for p in parts for r in p]


# --------------------------------------------------------------------------- diagnostics (training only)

def proposal_stats(policy: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """On labelled states: no-change accuracy on zero labels, cell accuracy and exactness on non-zero labels."""
    zero = zero_ok = nz = cell_ok = exact = 0
    for r in rows:
        for smp in r["samples"]:
            lab = np.asarray(smp["label"])
            n = len(lab) // 2
            prop = policy.propose(smp["state"])
            if not np.any(lab):
                zero += 1
                zero_ok += int(not prop)
                continue
            nz += 1
            mag = np.abs(lab[:n]) / 6.0 + np.abs(lab[n:]) / 5.0
            if prop:
                got = np.zeros(2 * n)
                for c in prop:
                    got[c["cell"]], got[n + c["cell"]] = c["d_power_db"], c["d_tilt_deg"]
                gmag = np.abs(got[:n]) / 6.0 + np.abs(got[n:]) / 5.0
                cell_ok += int(int(np.argmax(gmag)) == int(np.argmax(mag)))
                exact += int(np.allclose(got, lab, atol=0.15))
    return {"zero_label_states": zero, "no_change_rate_on_zero": zero_ok / zero if zero else None,
            "nonzero_label_states": nz, "largest_cell_matches": cell_ok / nz if nz else None,
            "within_0.15_of_label": exact / nz if nz else None}


def train(rows: list[dict[str, Any]], specs_by_seed: dict[int, dict[str, Any]], targets: dict[str, Any], *,
          version: str, workdir: Path, workers: int = 1, hyper: dict[str, Any] | None = None,
          log: Callable[[str], None] = print) -> dict[str, Any]:
    """Round 0 on the twin-labelled states, then DAgger rounds. Returns the model and a training record."""
    h = {**HYPER, **(hyper or {})}
    model = new_model(h["seed"])
    tr_rows = [r for r in rows if r["kept"] and r["role"] == "train"]
    va_rows = [r for r in rows if r["kept"] and r["role"] == "val"]
    record: dict[str, Any] = {"hyper": h, "rounds": []}

    def arrays(rs):
        a = D.to_arrays(rs, OBS_MODE)
        return {"obs": a["obs"], "target": a["target"]} if rs and a["target"].size else \
            {"obs": {}, "target": np.zeros((0, 8), dtype=np.float32)}

    for rnd in range(h["dagger_rounds"] + 1):
        train_a, val_a = arrays(tr_rows), arrays(va_rows)
        if not len(train_a["target"]):
            raise RuntimeError("no training samples (no kept train incidents)")
        log(f"[bc] round {rnd}: {len(train_a['target'])} train samples, {len(val_a['target'])} val samples")
        fit_rec = fit(model, train_a, val_a, epochs=h["epochs"] if rnd == 0 else h["dagger_epochs"],
                      patience=h["patience"] if rnd == 0 else h["dagger_patience"], lr=h["lr"], batch=h["batch"],
                      seed=h["seed"] + rnd, log=log)
        pol = policy_of(model, version)
        rec = {"round": rnd, "train_samples": int(len(train_a["target"])), "val_samples": int(len(val_a["target"])),
               "fit": {k: v for k, v in fit_rec.items() if k != "history"},
               "history_tail": fit_rec["history"][-3:], "val_proposals": proposal_stats(pol, va_rows)}
        record["rounds"].append(rec)
        if rnd == h["dagger_rounds"]:
            break
        path = workdir / f"round{rnd}.zip"
        model.save(str(path))
        meta = meta_for(version)
        new_tr = rollouts([specs_by_seed[r["seed"]] for r in tr_rows if not r.get("dagger")], targets, str(path), meta,
                          workers)
        new_va = rollouts([specs_by_seed[r["seed"]] for r in va_rows if not r.get("dagger")], targets, str(path), meta,
                          workers)
        for r in new_tr + new_va:
            r["dagger"] = rnd + 1
        tr_rows += new_tr
        va_rows += new_va
        rec["dagger_added"] = {"train_states": sum(len(r["samples"]) for r in new_tr),
                               "val_states": sum(len(r["samples"]) for r in new_va)}
    record["final_val_proposals"] = record["rounds"][-1]["val_proposals"]
    return {"model": model, "record": record, "rows": tr_rows + va_rows}


def save(model: Any, out_dir: Path, version: str, provenance: dict[str, Any], name: str = "remediation_v1") -> dict:
    """<name>.zip + <name>.meta.json (read by t1.rl.policy.load) + <name>.provenance.json."""
    import hashlib

    out_dir.mkdir(parents=True, exist_ok=True)
    zp = out_dir / f"{name}.zip"
    model.save(str(zp))
    meta = meta_for(version)
    (out_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    sha = hashlib.sha256(zp.read_bytes()).hexdigest()
    prov = {**provenance, "artifact": zp.name, "sha256": sha, "meta": meta}
    (out_dir / f"{name}.provenance.json").write_text(json.dumps(prov, indent=1, sort_keys=True) + "\n")
    return {"zip": str(zp), "sha256": sha, "meta": meta}
