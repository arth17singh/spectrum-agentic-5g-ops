"""Generate the Track 2 operator-silo datasets from the in-process Maveric twin.

    python -m silos.generate --out <dir> [--config silos/config.yaml] [--workers 12]
                             [--manifest silos/manifest.json]

Two splits are written, both with the same three silos and sizes:

  disjoint     silo A (large, moving users, 23-30 dBm starts, mostly tilt changes), B (mid, 30-37 dBm,
               mostly power changes) and C (small, stationary users, 43-46 dBm, mixed) each cover a
               separate power range;
  overlap      the CONTROL: every silo draws its start power from the full 23-46 dBm range.

Layout: <out>/<split>/<silo>/{train,test}.csv and <out>/<split>/shared_test/test.csv (the Viavi-like
point plus a slice of each silo's operating region, with a `segment` column).

Rows follow data contract v0.2 with amendment 1 labels (see silos/features.py): inputs are the
cluster state and the safety-filtered change, labels are KPI DELTAS against the no-op counterfactual
at the same step. The twin is driven directly (not through the Gym env) with the SDK's own
ActionSafetyFilter, and the no-op step is a replay of the same step from a snapshot of the twin's
state and random generator, so it is exact.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from silos.features import (
    BASE_KPIS,
    COLUMNS,
    CONTRACT_VERSION,
    INPUT_COLUMNS,
    LABELS,
    METADATA,
    N_CELLS,
    N_INPUTS,
    POWER_ABS,
    POWER_DELTA,
    TILT_ABS,
    TILT_DELTA,
    cluster_kpis,
    delta_labels,
    state_change_to_inputs,
)

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config.yaml"
DEFAULT_MANIFEST = HERE / "manifest.json"
STEP_SECONDS = 3600.0
KINDS = ("zero", "single_power", "single_tilt", "single_both", "multi")
POWER_STEPS = (-4.0, -2.0, -1.0, 1.0, 2.0, 4.0, 6.0, 8.0)
TILT_STEPS = (-5.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 5.0)
_TWIN_ATTRS = (
    "_power", "_tilt", "_alarm", "_asleep", "_ue", "_speed", "_heading", "_heading_mean",
    "_rng", "_step", "_hour", "_displacement_m",
)
SPLITS = ("disjoint", "overlap")


# ---------------------------------------------------------------------------------------------
# seeds, hashing
# ---------------------------------------------------------------------------------------------
def seed_hash(seed: int) -> str:
    return hashlib.sha256(str(int(seed)).encode()).hexdigest()


def split_of(seed: int) -> str:
    """'test' for ~20% of seeds by a hash of the seed alone (the same rule as t2/gen.py)."""
    return "test" if int(seed_hash(seed), 16) % 10 < 2 else "train"


def pick_seeds(base: int, n: int, role: str) -> list[int]:
    """The first ``n`` seeds at or above ``base`` whose hash puts them in ``role``."""
    out, s = [], base
    while len(out) < n:
        if split_of(s) == role:
            out.append(s)
        s += 1
    return out


def canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def config_sha256(cfg: dict[str, Any]) -> str:
    return hashlib.sha256(canonical(cfg)).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def git_state() -> tuple[str, bool]:
    """(HEAD commit, working tree of silos/ is dirty)."""
    try:
        head = subprocess.check_output(["git", "-C", str(HERE), "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(HERE), "status", "--porcelain", "--", str(HERE)], text=True
            ).strip()
        )
        return head, dirty
    except Exception:  # not a git checkout
        return "unknown", True


# ---------------------------------------------------------------------------------------------
# planning: which clusters (jobs) make which files
# ---------------------------------------------------------------------------------------------
def _job(split: str, silo: str, segment: str, role: str, seed: int, steps: int, cfg: dict[str, Any],
         **over: Any) -> dict[str, Any]:
    s = cfg["silos"].get(silo)
    job = {
        "split": split, "silo": silo, "segment": segment, "role": role, "seed": int(seed),
        "steps": int(steps), "episode_len": int(cfg["episode_len"]),
        "drift_margin_db": float(cfg["drift_margin_db"]), "start_jitter_db": float(cfg["start_jitter_db"]),
        "tilt_start_deg": list(cfg["tilt_start_deg"]),
    }
    if s is not None:
        job.update(
            n_ues=int(s["n_ues"]), mobility=cfg["mobility"][s["mobility"]], moving=s["mobility"] == "moving",
            noise_floor_dbm=float(s["noise_floor_dbm"]), mix=s["mix"],
            power_start_dbm=list(s["power_start_dbm"] if split == "disjoint" else cfg["overlap_power_start_dbm"]),
        )
    job.update(over)
    return job


def plan(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Every cluster to simulate. Viavi-like slices are split-independent (generated once)."""
    jobs: list[dict[str, Any]] = []
    t = cfg["test"]
    for split in SPLITS:
        for silo, s in cfg["silos"].items():
            for k, seed in enumerate(pick_seeds(s["seed_base"], s["clusters"], "train")):
                jobs.append(_job(split, silo, f"train_{silo}", "train", seed, cfg["steps_per_cluster"], cfg, cluster=k))
            for k, seed in enumerate(pick_seeds(s["seed_base"], t["clusters_per_slice"], "test")):
                jobs.append(_job(split, silo, f"region_{silo}", "test", seed, t["steps_per_cluster"], cfg, cluster=k))
    v = t["viavi_like"]
    for name, sl in v["slices"].items():
        for k, seed in enumerate(pick_seeds(v["seed_base"], t["clusters_per_slice"], "test")):
            jobs.append({
                "split": "shared", "silo": "viavi", "segment": name, "role": "test", "seed": int(seed),
                "steps": int(t["steps_per_cluster"]), "episode_len": int(v["episode_len"]),
                "drift_margin_db": 0.0, "start_jitter_db": 0.0, "tilt_start_deg": [v["tilt_deg"]] * 2,
                "n_ues": int(v["n_ues"]), "mobility": cfg["mobility"][v["mobility"]], "moving": v["mobility"] == "moving",
                "noise_floor_dbm": float(sl["noise_floor_dbm"]), "mix": v["mix"],
                "power_start_dbm": [v["power_dbm"]] * 2, "drift_dbm": list(v["drift_dbm"]), "cluster": k,
            })
    return jobs


# ---------------------------------------------------------------------------------------------
# one cluster
# ---------------------------------------------------------------------------------------------
def _rng_for(job: dict[str, Any]) -> np.random.Generator:
    key = f'{job["split"]}|{job["silo"]}|{job["segment"]}|{job["seed"]}'
    return np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))


def _snapshot(twin: Any) -> dict[str, Any]:
    return {a: copy.deepcopy(getattr(twin, a)) for a in _TWIN_ATTRS}


def _restore(twin: Any, snap: dict[str, Any]) -> None:
    for a in _TWIN_ATTRS:
        setattr(twin, a, copy.deepcopy(snap[a]))


def _sample_delta(rng: np.random.Generator, bounds: tuple[float, float], steps: tuple[float, ...]) -> float:
    if rng.random() < 0.5:
        return float(rng.choice(steps))
    return float(rng.uniform(*bounds))


def sample_change(rng: np.random.Generator, mix: dict[str, Any], power: np.ndarray, tilt: np.ndarray,
                  p_bounds: tuple[float, float]) -> tuple[np.ndarray, np.ndarray, str]:
    """A change within the SDK limits that keeps every cell's power in ``p_bounds`` and tilt in 4..15."""
    kind = str(rng.choice(KINDS, p=np.asarray(mix["kinds"], dtype=float) / sum(mix["kinds"])))
    dp, dt = np.zeros(N_CELLS), np.zeros(N_CELLS)
    if kind != "zero":
        if kind == "multi":
            cells = rng.choice(N_CELLS, size=int(rng.integers(2, N_CELLS + 1)), replace=False)
        else:
            cells = [int(rng.integers(N_CELLS))]
        for c in cells:
            if kind == "multi":
                do_p, do_t = rng.random() < mix["multi_power"], rng.random() < mix["multi_tilt"]
                if not (do_p or do_t):
                    do_t = mix["multi_tilt"] >= mix["multi_power"]
                    do_p = not do_t
            else:
                do_p = kind in ("single_power", "single_both")
                do_t = kind in ("single_tilt", "single_both")
            if do_p:
                dp[c] = _sample_delta(rng, POWER_DELTA, POWER_STEPS)
            if do_t:
                dt[c] = _sample_delta(rng, TILT_DELTA, TILT_STEPS)
    dp = np.clip(dp, np.maximum(POWER_DELTA[0], p_bounds[0] - power), np.minimum(POWER_DELTA[1], p_bounds[1] - power))
    dt = np.clip(dt, np.maximum(TILT_DELTA[0], TILT_ABS[0] - tilt), np.minimum(TILT_DELTA[1], TILT_ABS[1] - tilt))
    dp, dt = np.trunc(dp * 10.0) / 10.0, np.trunc(dt * 10.0) / 10.0  # one decimal, toward zero
    return dp, dt, change_kind(dp, dt)


def change_kind(dp: np.ndarray, dt: np.ndarray) -> str:
    cp, ct = dp != 0, dt != 0
    n = int((cp | ct).sum())
    if n == 0:
        return "zero"
    if n > 1:
        return "multi"
    return "single_both" if cp.any() and ct.any() else ("single_power" if cp.any() else "single_tilt")


def _build_twin(job: dict[str, Any]) -> Any:
    from ran_rl.twin.factory import MAVERIC_MODELS_DIR, MAVERIC_TOPOLOGY
    from ran_rl.twin.maveric.twin import MavericDigitalTwin, MavericTwinConfig
    from ran_rl.twin.scenarios import Scenario

    sc = Scenario(
        mobility={"alpha": job["mobility"]["alpha"], "velocity_m": job["mobility"]["velocity_m"]},
        ue_resample_fraction=job["mobility"]["ue_resample_fraction"],
    )
    return MavericDigitalTwin(
        MAVERIC_MODELS_DIR, MAVERIC_TOPOLOGY, MavericTwinConfig(n_ues=job["n_ues"]), scenarios={"silo": sc}
    )


def run_cluster(job: dict[str, Any]) -> dict[str, Any]:
    """Simulate one cluster; returns numeric rows and the per-row varying metadata."""
    from ran_rl.env.config import SafetyBounds
    from ran_rl.env.safety import ActionSafetyFilter

    import torch

    torch.set_num_threads(1)
    twin = _build_twin(job)
    filt = ActionSafetyFilter(SafetyBounds())
    rng = _rng_for(job)
    state = twin.reset("silo", job["seed"], noise_floor_dbm=job["noise_floor_dbm"])
    p_lo, p_hi = job["power_start_dbm"]
    if "drift_dbm" in job:
        p_bounds = tuple(job["drift_dbm"])
    else:
        m = job["drift_margin_db"]
        p_bounds = (max(POWER_ABS[0], p_lo - m), min(POWER_ABS[1], p_hi + m))
    t_lo, t_hi = job["tilt_start_deg"]
    n_ep = job["steps"] // job["episode_len"]
    if n_ep * job["episode_len"] != job["steps"]:
        raise ValueError("steps must be a multiple of episode_len")
    num = np.zeros((job["steps"], N_INPUTS + len(LABELS) + len(BASE_KPIS)))
    varying: list[tuple[int, int, float, float, str, int]] = []
    row = 0
    zeros = [0.0] * N_CELLS
    for ep in range(n_ep):
        # the operator re-plans: a start point drawn from the silo's operating region
        j = job["start_jitter_db"]
        p0 = rng.uniform(p_lo, p_hi)
        t0 = rng.uniform(t_lo, t_hi)
        power = np.clip(p0 + rng.uniform(-j, j, N_CELLS), p_lo, p_hi)
        tilt = np.clip(t0 + rng.uniform(-j, j, N_CELLS), TILT_ABS[0], TILT_ABS[1])
        twin._power[:] = power
        twin._tilt[:] = tilt
        state = twin.observe()
        for _ in range(job["episode_len"]):
            raw_dp, raw_dt, _kind = sample_change(
                rng, job["mix"], np.asarray(state["power_dbm"]), np.asarray(state["tilt_deg"]), p_bounds
            )
            res = filt.apply(np.concatenate([raw_dp, raw_dt]), state)
            fdp, fdt = res.filtered[:N_CELLS], res.filtered[N_CELLS:]
            clipped = int(not np.array_equal(res.filtered, np.concatenate([raw_dp, raw_dt])))
            x = state_change_to_inputs(state, fdp, fdt)
            kind = change_kind(fdp, fdt)
            snap = _snapshot(twin)
            twin.apply_action({"power_delta_db": fdp.tolist(), "tilt_delta_deg": fdt.tolist()})
            after = twin.advance(STEP_SECONDS)
            if kind == "zero":
                noop = after
            else:
                changed = _snapshot(twin)
                _restore(twin, snap)
                twin.apply_action({"power_delta_db": zeros, "tilt_delta_deg": zeros})
                noop = twin.advance(STEP_SECONDS)
                _restore(twin, changed)
            num[row, :N_INPUTS] = x
            num[row, N_INPUTS : N_INPUTS + 4] = delta_labels(after, noop)
            num[row, N_INPUTS + 4 :] = cluster_kpis(noop)
            varying.append((row, ep, float(np.mean(power)), float(np.mean(tilt)), kind, clipped))
            state = after
            row += 1
    return {"job": job, "num": num, "varying": varying}


# ---------------------------------------------------------------------------------------------
# assembling files
# ---------------------------------------------------------------------------------------------
def frame_for(result: dict[str, Any], run_id: str, commit: str, twin_version: str):
    import pandas as pd

    job, num, var = result["job"], result["num"], result["varying"]
    n = len(num)
    df = pd.DataFrame(num, columns=INPUT_COLUMNS + list(LABELS) + list(BASE_KPIS))
    const = {
        "run_id": run_id, "segment": job["segment"], "silo": job["silo"], "cluster": job["cluster"],
        "seed": job["seed"], "n_ues": job["n_ues"], "moving": int(job["moving"]),
        "noise_floor_dbm": job["noise_floor_dbm"], "source": "maveric-twin", "twin_version": twin_version,
        "backend": "in-process", "contract_version": CONTRACT_VERSION, "generator_commit": commit,
    }
    for k, v in const.items():
        df[k] = v
    df["step"] = np.arange(n)
    df["episode"] = [v[1] for v in var]
    df["start_power_dbm"] = [v[2] for v in var]
    df["start_tilt_deg"] = [v[3] for v in var]
    df["change_kind"] = [v[4] for v in var]
    df["clipped"] = [v[5] for v in var]
    return df[COLUMNS]


def write_csv(df, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n")


def operating_summary(df) -> dict[str, Any]:
    pw = df[[f"power_dbm[{i}]" for i in range(N_CELLS)]].to_numpy()
    tl = df[[f"tilt_deg[{i}]" for i in range(N_CELLS)]].to_numpy()
    dpw = df[[f"d_power_db[{i}]" for i in range(N_CELLS)]].to_numpy()
    dtl = df[[f"d_tilt_deg[{i}]" for i in range(N_CELLS)]].to_numpy()
    return {
        "power_dbm": {"min": float(pw.min()), "mean": float(pw.mean()), "max": float(pw.max())},
        "start_power_dbm": {"min": float(df["start_power_dbm"].min()), "max": float(df["start_power_dbm"].max())},
        "tilt_deg": {"min": float(tl.min()), "mean": float(tl.mean()), "max": float(tl.max())},
        "n_ues": sorted(int(v) for v in df["n_ues"].unique()),
        "moving": sorted(int(v) for v in df["moving"].unique()),
        "noise_floor_dbm": sorted(float(v) for v in df["noise_floor_dbm"].unique()),
        "change_kind_counts": {k: int(v) for k, v in df["change_kind"].value_counts().sort_index().items()},
        "zero_change_fraction": float((df["change_kind"] == "zero").mean()),
        "power_change_fraction": float((dpw != 0).any(axis=1).mean()),
        "tilt_change_fraction": float((dtl != 0).any(axis=1).mean()),
        "label_mean": {c: float(df[c].mean()) for c in LABELS},
        "label_std": {c: float(df[c].std()) for c in LABELS},
    }


def generate(cfg: dict[str, Any], out: Path, workers: int | None = None, manifest_path: Path | None = None,
             progress: bool = False) -> dict[str, Any]:
    """Run every job, write the CSV tree under ``out`` and (optionally) the manifest."""
    import pandas as pd

    t0 = time.time()
    commit, dirty = git_state()
    twin_version = f'ran-rl {importlib.metadata.version("ran-rl")}'
    cfg_hash = config_sha256(cfg)
    run_id = f'{cfg["run_label"]}-{cfg_hash[:8]}'
    jobs = plan(cfg)
    order = sorted(range(len(jobs)), key=lambda i: -jobs[i]["steps"] * jobs[i].get("n_ues", 100))
    results: list[Any] = [None] * len(jobs)
    workers = workers or max(1, min(len(jobs), (os.cpu_count() or 2) - 2))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {i: ex.submit(run_cluster, jobs[i]) for i in order}
        for n, (i, f) in enumerate(futs.items(), 1):
            results[i] = f.result()
            if progress:
                j = jobs[i]
                print(f'[{n}/{len(jobs)}] {j["split"]}/{j["silo"]}/{j["segment"]} seed={j["seed"]} '
                      f'({time.time() - t0:.0f}s)', flush=True)

    frames: dict[tuple[str, str, str], list] = {}
    for r in results:
        j = r["job"]
        frames.setdefault((j["split"], j["silo"], j["role"]), []).append(frame_for(r, run_id, commit, twin_version))
    files: list[dict[str, Any]] = []

    def emit(rel: str, df, split: str, silo: str, role: str) -> None:
        write_csv(df, out / rel)
        seeds = sorted(int(s) for s in df["seed"].unique())
        files.append({
            "path": rel, "split": split, "silo": silo, "role": role, "rows": int(len(df)),
            "sha256": file_sha256(out / rel), "clusters": len(seeds),
            "segments": sorted(df["segment"].unique().tolist()),
            "seeds": [{"seed": s, "sha256": seed_hash(s), "class": split_of(s)} for s in seeds],
            "operating_point": operating_summary(df),
        })

    for split in SPLITS:
        region = []
        for silo in cfg["silos"]:
            tr = pd.concat(frames[(split, silo, "train")], ignore_index=True)
            te = pd.concat(frames[(split, silo, "test")], ignore_index=True)
            emit(f"{split}/{silo}/train.csv", tr, split, silo, "train")
            emit(f"{split}/{silo}/test.csv", te, split, silo, "test")
            region.append(te)
        shared = pd.concat(frames[("shared", "viavi", "test")] + region, ignore_index=True)
        emit(f"{split}/shared_test/test.csv", shared, split, "shared", "shared_test")

    manifest = {
        "kind": "t2-silo-manifest",
        "label": "simulated operators on the in-process Maveric twin; NOT the FL service, NOT partner data",
        "contract_version": CONTRACT_VERSION,
        "amendment": "1 (labels are deltas against the no-op counterfactual at the same step)",
        "run_id": run_id,
        "generator_commit": commit,
        "generator_tree_dirty": dirty,
        "twin": twin_version + " (MavericDigitalTwin, shipped sim_tilt_sweep_4cell GP models)",
        "config_sha256": cfg_hash,
        "config": cfg,
        "split_rule": cfg["split_rule"],
        "columns": COLUMNS,
        "input_columns": INPUT_COLUMNS,
        "label_columns": list(LABELS),
        "metadata_columns_never_inputs": list(METADATA) + list(BASE_KPIS),
        "label_definition": {
            "d_*": "KPIs(after the change, advanced one step) - KPIs(same step replayed with no change)",
            "coverage_ratio, sla_ratio": "as reported by the twin",
            "sinr_mean_db": "user-share weighted mean of per-cell means (exact)",
            "sinr_p10_db": "user-share weighted mean of per-cell 10th percentiles (approximation of the pooled p10; same as t2/kpis.py)",
        },
        "files": files,
        "seconds": round(time.time() - t0, 1),
    }
    if manifest_path is not None:
        manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--config", default=DEFAULT_CONFIG, type=Path)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST, type=Path)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    m = generate(cfg, a.out, a.workers, a.manifest, progress=True)
    tot = sum(f["rows"] for f in m["files"] if f["role"] in ("train", "test"))
    print(f'wrote {len(m["files"])} files, {tot} rows (excluding shared_test copies) in {m["seconds"]} s; '
          f'commit {m["generator_commit"][:10]} dirty={m["generator_tree_dirty"]}')
    return 0


if __name__ == "__main__":
    sys.exit(main())
