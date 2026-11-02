"""Generate the Track 2 response-model dataset from the in-process Maveric twin.

v0, local, not federated / not the official job.

    python -m t2.gen --out <dir> [--episodes 700] [--steps 12] [--workers 8]

One row = (state at step t, safety-filtered change, cluster KPIs of the state after the change
is applied and the twin advances one step). Contract v0.2 (see t2/features.py).

Design points
 * The twin is driven directly (ran_rl twin + the SDK's ActionSafetyFilter), not through the Gym
   env, so a row's change is exactly what the twin applied.
 * Every step the generator also runs the ZERO-change counterfactual from the same twin snapshot
   (same RNG stream, so the same UE movement). The LABEL `y` is the DELTA
   KPIs(after the change, advanced one step) - KPIs(same step advanced with no change), per contract
   amendment 1 (A5); this cancels mobility drift. The levels are kept as metadata (`y_level`,
   `y_nochange`), never inputs.
 * Start power/tilt are varied (not a change, so the per-step limits do not apply), including
   an exact Viavi-like start (40 dBm, 6 deg) in ~10% of episodes.
 * Scenarios: nominal, low_load, congestion, interference, mobility_stress. `outage` is left out
   of v0: alarm is a forbidden input and the safety filter zeroes changes on an alarmed cell, so
   an alarm-blind model would see an unexplained no-op.
 * The split is by seed hash: sha256(str(seed)) % 10 < 2 is test. Whole episodes (all rows of a
   seed) go to one side, so a test seed never appears in training.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from t2.features import (
    CONTRACT_VERSION,
    FEATURE_NAMES,
    LABELS,
    N_CELLS,
    N_FEATURES,
    POWER_DELTA,
    TILT_DELTA,
    arrays_to_change,
    col,
    state_change_to_row,
)
from t2.kpis import kpis_vector

LABEL = "v0, local, not federated / not the official job"
SCENARIOS = ("nominal", "low_load", "congestion", "interference", "mobility_stress")
STEP_SECONDS = 3600.0
BASE_SEED = 10_000
VIAVI_LIKE = (40.0, 6.0)
CHANGE_KINDS = ("zero", "single_power", "single_tilt", "single_both", "multi")
START_KINDS = ("random", "viavi_like")
_ATTRS = ("_power", "_tilt", "_alarm", "_asleep", "_ue", "_speed", "_heading", "_heading_mean",
          "_rng", "_step", "_hour", "_displacement_m")


def split_of(seed: int) -> str:
    """'test' for ~20% of seeds, decided by a hash of the seed alone."""
    return "test" if int(hashlib.sha256(str(int(seed)).encode()).hexdigest(), 16) % 10 < 2 else "train"


def snapshot(twin: Any) -> dict[str, Any]:
    """Copy of the twin's mutable episode state (not its models)."""
    return {a: copy.deepcopy(getattr(twin, a)) for a in _ATTRS}


def restore(twin: Any, snap: dict[str, Any]) -> None:
    for a in _ATTRS:
        setattr(twin, a, copy.deepcopy(snap[a]))


def build_twin() -> Any:
    from ran_rl.env.config import EnvConfig, TwinConfig
    from ran_rl.twin.factory import build_twin as _build

    return _build(EnvConfig(twin=TwinConfig(backend="maveric")))


def _sample_delta(rng: np.random.Generator, lo: float, hi: float, steps: tuple[float, ...]) -> float:
    if rng.random() < 0.5:
        return float(rng.choice(steps))
    return float(rng.uniform(lo, hi))


def sample_raw_change(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """A raw (pre-filter) change of a random kind: single-cell power / tilt / both, or multi-cell."""
    kind = rng.choice(["single_power", "single_tilt", "single_both", "multi"], p=[0.2, 0.2, 0.15, 0.45])
    dp, dt = np.zeros(N_CELLS), np.zeros(N_CELLS)
    cells = rng.choice(N_CELLS, size=1 if kind != "multi" else int(rng.integers(2, N_CELLS + 1)), replace=False)
    for c in cells:
        if kind in ("single_power", "single_both", "multi"):
            dp[c] = _sample_delta(rng, *POWER_DELTA, (-4, -2, -1, 1, 2, 4, 6, 8))
        if kind in ("single_tilt", "single_both") or (kind == "multi" and rng.random() < 0.6):
            dt[c] = _sample_delta(rng, *TILT_DELTA, (-3, -2, -1, 1, 2, 3, 5, -5))
    return dp, dt


def change_kind(dp: np.ndarray, dt: np.ndarray) -> str:
    cp, ct = dp != 0, dt != 0
    n = int((cp | ct).sum())
    if n == 0:
        return "zero"
    if n > 1:
        return "multi"
    return "single_both" if cp.any() and ct.any() else ("single_power" if cp.any() else "single_tilt")


def _filtered(filt: Any, state: dict[str, Any], dp: np.ndarray, dt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    res = filt.apply(np.concatenate([dp, dt]), state)
    return res.filtered[:N_CELLS].copy(), res.filtered[N_CELLS:].copy()


def _step(twin: Any, dp: np.ndarray, dt: np.ndarray) -> dict[str, Any]:
    twin.apply_action({"power_delta_db": dp.tolist(), "tilt_delta_deg": dt.tolist()})
    return twin.advance(STEP_SECONDS)


def run_episode(seed: int, steps: int) -> dict[str, list]:
    """One seeded episode -> row columns. Deterministic in (seed, steps)."""
    from ran_rl.env.config import SafetyBounds
    from ran_rl.env.safety import ActionSafetyFilter

    twin = build_twin()
    filt = ActionSafetyFilter(SafetyBounds())
    rng = np.random.default_rng([seed, 7])  # generator randomness, apart from the twin's own seed
    scenario = SCENARIOS[int(rng.integers(len(SCENARIOS)))]
    state = twin.reset(scenario, seed)

    start_kind = "viavi_like" if rng.random() < 0.10 else "random"
    if start_kind == "viavi_like":
        power_t, tilt_t = np.full(N_CELLS, VIAVI_LIKE[0]), np.full(N_CELLS, VIAVI_LIKE[1])
    else:
        power_t = np.clip(rng.uniform(14, 44) + rng.normal(0, 2.5, N_CELLS), 10, 46)
        tilt_t = np.clip(rng.uniform(4.5, 12) + rng.normal(0, 1.2, N_CELLS), 4, 15)
    twin.apply_action({"power_delta_db": (power_t - np.asarray(state["power_dbm"])).tolist(),
                       "tilt_delta_deg": (tilt_t - np.asarray(state["tilt_deg"])).tolist()})
    state = twin.observe()  # a start configuration, not a change: no advance

    cols: dict[str, list] = {k: [] for k in ("x", "y", "y0", "seed", "t", "scenario", "start_kind", "kind",
                                              "start_power", "start_tilt")}
    for t in range(steps):
        x_state = state
        snap = snapshot(twin)
        # zero-change counterfactual (same snapshot, same RNG stream -> same UE movement)
        zdp, zdt = np.zeros(N_CELLS), np.zeros(N_CELLS)
        y0 = kpis_vector(_step(twin, zdp, zdt))
        candidates = []
        for _ in range(2):
            dp, dt = _filtered(filt, state, *sample_raw_change(rng))
            candidates.append((dp, dt))
        include_zero = rng.random() < 0.5
        outcomes = []
        for dp, dt in candidates[1:]:  # the second candidate is a pure counterfactual
            restore(twin, snap)
            outcomes.append(kpis_vector(_step(twin, dp, dt)))
        restore(twin, snap)
        dp, dt = candidates[0]
        next_state = _step(twin, dp, dt)  # the trajectory follows candidate 0
        outcomes.insert(0, kpis_vector(next_state))
        todo = [(candidates[0], outcomes[0]), (candidates[1], outcomes[1])]
        if include_zero:
            todo.append(((zdp, zdt), y0))
        for (cdp, cdt), y in todo:
            cols["x"].append(state_change_to_row(x_state, arrays_to_change(cdp, cdt), apply_safety=False))
            cols["y"].append(y)
            cols["y0"].append(y0)
            cols["seed"].append(seed)
            cols["t"].append(t)
            cols["scenario"].append(SCENARIOS.index(scenario))
            cols["start_kind"].append(START_KINDS.index(start_kind))
            cols["kind"].append(CHANGE_KINDS.index(change_kind(cdp, cdt)))
            cols["start_power"].append(float(power_t.mean()))
            cols["start_tilt"].append(float(tilt_t.mean()))
        state = next_state
    return cols


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def generate(episodes: int, steps: int, workers: int, base_seed: int = BASE_SEED) -> dict[str, np.ndarray]:
    seeds = [base_seed + i for i in range(episodes)]
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            parts = list(ex.map(run_episode, seeds, [steps] * len(seeds), chunksize=8))
    else:
        parts = [run_episode(s, steps) for s in seeds]
    out: dict[str, list] = {k: [] for k in parts[0]}
    for p in parts:  # seed order -> deterministic file
        for k, v in p.items():
            out[k].extend(v)
    arr = {
        "x": np.asarray(out["x"], dtype=np.float32),
        # LABELS (contract amendment 1, A5): KPI DELTA against the no-op counterfactual at the same step
        "y": (np.asarray(out["y"], dtype=np.float64) - np.asarray(out["y0"], dtype=np.float64)).astype(np.float32),
        "y_level": np.asarray(out["y"], dtype=np.float32),  # metadata: next-step KPI levels after the change
        "y_nochange": np.asarray(out["y0"], dtype=np.float32),  # metadata: levels with no change
        "seed": np.asarray(out["seed"], dtype=np.int64),
        "step": np.asarray(out["t"], dtype=np.int16),
        "scenario": np.asarray(out["scenario"], dtype=np.int8),
        "start_kind": np.asarray(out["start_kind"], dtype=np.int8),
        "change_kind": np.asarray(out["kind"], dtype=np.int8),
        "start_power_mean": np.asarray(out["start_power"], dtype=np.float32),
        "start_tilt_mean": np.asarray(out["start_tilt"], dtype=np.float32),
    }
    arr["split"] = np.asarray([0 if split_of(s) == "train" else 1 for s in arr["seed"]], dtype=np.int8)
    return arr


def save(arr: dict[str, np.ndarray], out_dir: Path, *, episodes: int, steps: int, wall_s: float) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "t2_dataset_v0.npz"
    np.savez_compressed(
        path, **arr,
        feature_names=np.asarray(FEATURE_NAMES), label_names=np.asarray(LABELS),
        scenario_names=np.asarray(SCENARIOS), change_kind_names=np.asarray(CHANGE_KINDS),
        start_kind_names=np.asarray(START_KINDS),
    )
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    seeds = np.unique(arr["seed"])
    test_seeds = [int(s) for s in seeds if split_of(int(s)) == "test"]
    manifest = {
        "kind": "t2-dataset-manifest",
        "label": LABEL,
        "contract_version": CONTRACT_VERSION,
        "dataset_file": path.name,
        "dataset_sha256": sha,
        "n_rows": int(len(arr["x"])),
        "n_rows_train": int((arr["split"] == 0).sum()),
        "n_rows_test": int((arr["split"] == 1).sum()),
        "n_episodes": episodes,
        "steps_per_episode": steps,
        "seeds": {"base": BASE_SEED, "count": episodes, "first": int(seeds.min()), "last": int(seeds.max()),
                  "n_train_seeds": len(seeds) - len(test_seeds), "test_seeds": test_seeds},
        "split_rule": "sha256(str(seed)) as hex integer % 10 < 2 -> test, else train (whole episodes)",
        "generator": {
            "module": "t2/gen.py",
            "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty_paths_t2": _git("status", "--porcelain", "--", "t2") or "clean",
            "ran_rl_version": _pkg_version("ran-rl"),
            "twin": "maveric-inprocess (MavericDigitalTwin, shipped sim_tilt_sweep_4cell models)",
            "noise_floor": "twin default (reset without noise_floor_dbm; CCO path)",
            "scenarios": list(SCENARIOS),
            "scenario_note": "outage left out of v0 (alarm is not an input; the safety filter freezes alarmed cells)",
            "step_seconds": STEP_SECONDS,
            "start_config": "random 14-44 dBm (+-2.5 per cell), tilt 4.5-12 (+-1.2 per cell); 10% exactly 40 dBm / 6 deg",
            "change_mix": "per step 2 candidates (single power / single tilt / single both / multi-cell, "
                          "half discrete steps and half uniform within -6..+8 dB and +-5 deg), then the SDK "
                          "ActionSafetyFilter; plus the zero-change row with probability 0.5",
            "kpi_definition": "t2/kpis.py (coverage and SLA as reported; sinr_mean = share-weighted mean; "
                              "sinr_p10 = share-weighted mean of per-cell p10, an approximation)",
            "wall_seconds": round(wall_s, 1),
        },
        "inputs": FEATURE_NAMES,
        "labels": list(LABELS),
        "label_kind": "delta vs the no-op counterfactual at the same step (amendment 1, A5)",
        "metadata_never_inputs": ["seed", "step", "scenario", "start_kind", "change_kind",
                                  "start_power_mean", "start_tilt_mean", "split", "y_level", "y_nochange"],
        "counts": {
            "by_scenario": {n: int((arr["scenario"] == i).sum()) for i, n in enumerate(SCENARIOS)},
            "by_change_kind": {n: int((arr["change_kind"] == i).sum()) for i, n in enumerate(CHANGE_KINDS)},
        },
    }
    (out_dir / "t2_dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def _pkg_version(name: str) -> str:
    from importlib.metadata import version

    try:
        return version(name)
    except Exception:  # noqa: BLE001
        return "unknown"


def load(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=700)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    t0 = time.time()
    arr = generate(a.episodes, a.steps, a.workers)
    path = save(arr, Path(a.out), episodes=a.episodes, steps=a.steps, wall_s=time.time() - t0)
    print(f"wrote {path}: {len(arr['x'])} rows ({int((arr['split'] == 0).sum())} train / "
          f"{int((arr['split'] == 1).sum())} test) in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
