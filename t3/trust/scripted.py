"""The scripted changes used by the trust demo and the ladder (fixed, declared, hashable).

``scripted_changes(k, seed)`` is deterministic and independent of the twin state: each step
moves 1-2 cells by power +-1..4 dB and tilt +-0.5..3 deg, and keeps every cell within
[-6, +6] dB and [-2, +5] deg of where it started by flipping a step's sign when it would leave
that box (from the 6 deg start, tilt stays in 4..11 deg, inside the SDK's safety bounds).
``scripted_changes.json`` is the committed copy for K=24, seed=2026; its sha256 goes into the
declaration of any run that uses it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

SCRIPT_PATH = Path(__file__).with_name("scripted_changes.json")
N_CELLS = 4


def scripted_changes(k: int = 24, seed: int = 2026, n_cells: int = N_CELLS) -> list[list[dict]]:
    rng = np.random.default_rng(seed)
    p_off, t_off = np.zeros(n_cells), np.zeros(n_cells)
    out = []
    for _ in range(k):
        cells = sorted(int(c) for c in rng.choice(n_cells, size=int(rng.integers(1, 3)), replace=False))
        step = []
        for i in cells:
            dp = float(rng.choice([-1, 1]) * rng.uniform(1, 4))
            dt = float(rng.choice([-1, 1]) * rng.uniform(0.5, 3))
            if not -6 <= p_off[i] + dp <= 6:
                dp = -dp
            if not -2 <= t_off[i] + dt <= 5:
                dt = -dt
            dp, dt = round(dp, 1), round(dt, 1)
            p_off[i] += dp
            t_off[i] += dt
            step.append({"cell": i, "d_power_db": dp, "d_tilt_deg": dt})
        out.append(step)
    return out


def random_change(rng: np.random.Generator, state: dict, n_cells: int = N_CELLS) -> list[dict]:
    """Calibration changes: 1-2 cells, power +-1..4 dB, tilt +-0.5..3 deg (one decimal), each
    flipped when it would leave 15..31 dBm or 4..12 deg (relative to a 23 dBm reference; on a
    40 dBm preset the safety filter clips and the change is recorded as unscored)."""
    cells = rng.choice(n_cells, size=int(rng.integers(1, 3)), replace=False)
    out = []
    for i in sorted(int(c) for c in cells):
        dp = float(rng.choice([-1, 1]) * rng.uniform(1, 4))
        dt = float(rng.choice([-1, 1]) * rng.uniform(0.5, 3))
        p, t = state["power_dbm"][i], state["tilt_deg"][i]
        if not 15 <= p + dp <= 31:
            dp = -dp
        if not 4 <= t + dt <= 12:
            dt = -dt
        out.append({"cell": i, "d_power_db": round(dp, 1), "d_tilt_deg": round(dt, 1)})
    return out


def load_script(path: str | Path = SCRIPT_PATH) -> tuple[list[list[dict]], str]:
    raw = Path(path).read_bytes()
    return json.loads(raw)["changes"], hashlib.sha256(raw).hexdigest()


def write_script(path: str | Path = SCRIPT_PATH, k: int = 24, seed: int = 2026) -> None:
    doc = {"generator": "t3.trust.scripted.scripted_changes", "k": k, "seed": seed,
           "changes": scripted_changes(k, seed)}
    Path(path).write_text(json.dumps(doc, indent=1) + "\n")


if __name__ == "__main__":
    write_script()
    print(SCRIPT_PATH, load_script()[1])
