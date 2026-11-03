"""The frozen incident suite: scripted misconfigurations injected into the in-process Maveric twin.

Each incident starts from a healthy reset (scenario, seed, preset), applies the healthy per-cell
commissioning as step 1 (none on the "visible" tier; seeded per-cell offsets on the "masked" tier, so
the configuration alone does not reveal the faulted cell), then one fault change as step 2 (through
the SDK safety filter, which must clip neither), and records the twin's KPIs on two
trajectories through the grade step: ``healthy`` (no fault, no-op steps) and ``unfixed`` (fault, then
no-op steps). Operators are later graded against those, at the same step.

Candidates come from a fixed template list x preset; each candidate has its own seed. Seeds are
split into dev / test by sha256 of the seed alone (no seed is in both). A candidate is dropped if
the fault leaves no_action within tolerance (the fault must actually break the network) or if the
safety filter clipped the fault. The test split is frozen to ``suite/incidents-test.jsonl`` (one
canonical JSON line per incident) and its sha256 to ``suite/incidents-test.sha256``.

    python -m t3.evals.incidents --write      # regenerate and freeze dev + test
    python -m t3.evals.incidents --check      # regenerate and compare with the frozen bytes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from t3.evals.graders import grade_kpis, load_targets
from t3.evals.twin_backend import (
    BACKEND,
    PRESET_SCENARIO,
    ReplayTwin,
    graded,
    twin_version,
)

SUITE_DIR = Path(__file__).with_name("suite")
TEST_PATH = SUITE_DIR / "incidents-test.jsonl"
DEV_PATH = SUITE_DIR / "incidents-dev.jsonl"
HASH_PATH = SUITE_DIR / "incidents-test.sha256"
META_PATH = SUITE_DIR / "incidents-meta.json"

SUITE_VERSION = "t3-incidents-v3"
# Earlier frozen test sets this one replaces (never reuse them for scoring).
SUPERSEDES = [
    {"suite": "t3-incidents-v1", "test_sha256": "c0846996a08357836efcc7af1b7ab285d63b8c3622a846a580661d7bef388f0a",
     "twin_backend": "maveric-inprocess-standin",
     "reason": "every cell started identical, so a config-diff heuristic solved it; replaced by the masked tier"},
    {"suite": "t3-incidents-v2", "test_sha256": "20f9013399da360b26214e11de15c9f9edbb8dcb9bcc4cf1d89fb7e194d3dd12",
     "twin_backend": "maveric-inprocess-standin",
     "reason": "frozen on t3/evals' stand-in twin; the twin backend changed to W2's twin.replay over ran-rl 1.2.9's "
               "Maveric twin (user-share-weighted pooled p10 KPIs), so the frozen KPIs no longer replay. Re-frozen "
               "once, before any swarm or model run used the suite."},
]
SEED_BASE = 4100
N_CANDIDATES = 84
SPLIT_SALT = "t3-evals-split-v1"
KPI_DECIMALS = 9  # rounding for the frozen file only; grading recomputes from the twin

# (name, [(cell offset, d_power_db, d_tilt_deg), ...]); the cell offset is added to the base cell.
# Every fault's exact inverse is inside the per-step safety limits (tilt +-5, power -6..+8) and in
# the oracle's candidate grid.
TEMPLATES: list[tuple[str, list[tuple[int, float, float]]]] = [
    ("tilt+5", [(0, 0.0, 5.0)]),
    ("power-6", [(0, -6.0, 0.0)]),
    ("tilt+4", [(0, 0.0, 4.0)]),
    ("power-4", [(0, -4.0, 0.0)]),
    ("power-3+tilt+3", [(0, -3.0, 3.0)]),
    ("power-6+tilt+5", [(0, -6.0, 5.0)]),
    ("two-cell:power-6|tilt+5", [(0, -6.0, 0.0), (1, 0.0, 5.0)]),
]
PRESETS = ("official", "viavi-like")


def canonical_line(obj: dict[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def split_of(seed: int) -> str:
    """'test' or 'dev', from sha256 of the seed only."""
    return "test" if hashlib.sha256(f"{SPLIT_SALT}:{seed}".encode()).digest()[0] % 2 == 0 else "dev"


TIERS = ("visible", "masked")
COMMISSION_POWER_DB = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
COMMISSION_TILT_DEG = (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0)


def commissioning(seed: int, tier: str, n_cells: int = 4) -> list[dict[str, Any]]:
    """The healthy per-cell configuration applied at step 1 (before the fault).

    "visible": none, every cell keeps the preset's identical power and tilt, so a faulted cell
    stands out in the configuration. "masked": seeded per-cell offsets, so the configuration alone
    no longer says which cell is wrong.
    """
    if tier == "visible":
        return []
    import numpy as np

    rng = np.random.default_rng([seed, 7])
    out = []
    for c in range(n_cells):
        dp = float(rng.choice(COMMISSION_POWER_DB))
        dt = float(rng.choice(COMMISSION_TILT_DEG))
        if dp or dt:
            out.append({"cell": c, "d_power_db": dp, "d_tilt_deg": dt})
    return out


def candidate(i: int) -> dict[str, Any]:
    """The i-th candidate spec (deterministic, no twin needed)."""
    seed = SEED_BASE + i
    preset = PRESETS[i % 2]
    tier = TIERS[(i // 2) % 2]
    name, parts = TEMPLATES[(i // 4) % len(TEMPLATES)]
    base_cell = (i // (4 * len(TEMPLATES))) % 4
    fault = [
        {"cell": (base_cell + off) % 4, "d_power_db": float(dp), "d_tilt_deg": float(dt)}
        for off, dp, dt in parts
    ]
    return {"seed": seed, "preset": preset, "scenario": PRESET_SCENARIO[preset], "tier": tier, "template": name,
            "commissioning": commissioning(seed, tier), "fault": fault}


def _r(d: dict[str, float]) -> dict[str, float]:
    return {k: round(v, KPI_DECIMALS) for k, v in d.items()}


def script(spec: dict[str, Any], targets: dict[str, Any], with_fault: bool) -> list[list[dict[str, Any]]]:
    """The changes applied at steps 1..fault_step: commissioning at step 1, the fault at fault_step."""
    ep = targets["episode"]
    steps: list[list[dict[str, Any]]] = [[] for _ in range(ep["fault_step"])]
    steps[0] = list(spec["commissioning"])
    if with_fault:
        steps[ep["fault_step"] - 1] = list(spec["fault"])
    return steps


def build(spec: dict[str, Any], targets: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """Run the healthy and faulted trajectories. Returns (incident or None, reason)."""
    ep = targets["episode"]
    n_steps, fs = ep["grade_step"], ep["fault_step"]
    if fs < 2:
        raise ValueError("fault_step must leave step 1 for commissioning")
    healthy = ReplayTwin(spec["scenario"], spec["seed"], spec["preset"])
    faulted = ReplayTwin(spec["scenario"], spec["seed"], spec["preset"])
    s0 = healthy.reset()
    faulted.reset()
    h_script, u_script = script(spec, targets, False), script(spec, targets, True)
    h_traj, u_traj = [graded(s0)], [graded(s0)]
    for step in range(1, n_steps + 1):
        h_traj.append(graded(healthy.apply(h_script[step - 1] if step <= fs else [])))
        u_traj.append(graded(faulted.apply(u_script[step - 1] if step <= fs else [])))
    for entry in faulted.action_log[:fs]:
        if entry["clamped"]:
            return None, f"commissioning or fault clipped by the safety filter: {entry['rules']}"
    g = grade_kpis(u_traj[-1], h_traj[-1], u_traj[-1], targets)
    if g["success"]:
        return None, "no_action already within tolerance at the grade step (fault not material)"
    iid = (f"inc-{'off' if spec['preset'] == 'official' else 'vvl'}-{spec['tier'][:3]}-{spec['seed']}-"
           f"{spec['template']}")
    s_fault = faulted.action_log[fs - 1]
    return {
        "suite": SUITE_VERSION,
        "incident_id": iid,
        "split": split_of(spec["seed"]),
        "scenario": spec["scenario"],
        "seed": spec["seed"],
        "preset": spec["preset"],
        "tier": spec["tier"],
        "template": spec["template"],
        "commissioning": spec["commissioning"],
        "fault": spec["fault"],
        "fault_cells": sorted({f["cell"] for f in spec["fault"]}),
        "fault_step": fs,
        "grade_step": n_steps,
        "fault_filtered": s_fault["filtered"],
        "pre_fault_kpis": _r(h_traj[fs - 1]),
        "post_fault_kpis": _r(u_traj[fs]),
        "healthy_kpis_by_step": [_r(k) for k in h_traj],
        "unfixed_kpis_by_step": [_r(k) for k in u_traj],
        "material_kpis": [k for k, p in g["per_kpi"].items() if p["material"]],
    }, "kept"


def generate(targets: dict[str, Any] | None = None) -> dict[str, Any]:
    targets = targets or load_targets()
    kept: dict[str, list[dict[str, Any]]] = {"dev": [], "test": []}
    dropped = []
    for i in range(N_CANDIDATES):
        spec = candidate(i)
        inc, why = build(spec, targets)
        if inc is None:
            dropped.append({"seed": spec["seed"], "preset": spec["preset"], "tier": spec["tier"],
                            "template": spec["template"],
                            "fault": spec["fault"], "split": split_of(spec["seed"]), "reason": why})
        else:
            kept[inc["split"]].append(inc)
    return {"dev": kept["dev"], "test": kept["test"], "dropped": dropped}


def to_bytes(incidents: list[dict[str, Any]]) -> bytes:
    return "".join(canonical_line(x) + "\n" for x in incidents).encode("utf-8")


def load_test(path: Path = TEST_PATH, verify_hash: bool = True) -> tuple[list[dict[str, Any]], str]:
    """The frozen test incidents and the sha256 of the file's bytes (checked against the sidecar)."""
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if verify_hash and path == TEST_PATH:
        want = HASH_PATH.read_text().split()[0]
        if digest != want:
            raise RuntimeError(f"{path} sha256 {digest} != frozen {want}: the test set changed")
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()], digest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="regenerate and freeze the suite files")
    ap.add_argument("--check", action="store_true", help="regenerate and compare with the frozen test file")
    a = ap.parse_args(argv)
    out = generate()
    test_bytes = to_bytes(out["test"])
    digest = hashlib.sha256(test_bytes).hexdigest()
    summary = {
        "test": len(out["test"]), "dev": len(out["dev"]), "dropped": len(out["dropped"]),
        "test_sha256": digest,
    }
    if a.write:
        SUITE_DIR.mkdir(exist_ok=True)
        TEST_PATH.write_bytes(test_bytes)
        DEV_PATH.write_bytes(to_bytes(out["dev"]))
        if BACKEND != "maveric-inprocess":
            raise SystemExit(f"refusing to freeze on twin backend {BACKEND!r}: the real twin (twin.replay) is required")
        HASH_PATH.write_text(f"{digest}  incidents-test.jsonl\n")
        META_PATH.write_text(json.dumps({
            "frozen_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "supersedes": SUPERSEDES,
            "suite": SUITE_VERSION, "candidates": N_CANDIDATES, "seed_base": SEED_BASE, "split_salt": SPLIT_SALT,
            "templates": [t[0] for t in TEMPLATES], "presets": list(PRESETS), "tiers": list(TIERS),
            "counts": {"test": len(out["test"]), "dev": len(out["dev"]), "dropped": len(out["dropped"])},
            "test_sha256": digest, "twin_backend": BACKEND, "twin_version": twin_version(),
            "python": sys.version.split()[0], "platform": f"{platform.system()}-{platform.machine()}",
            "note": "float bits can differ across CPU architectures; regenerate with --check on the same platform",
            "dropped": out["dropped"],
        }, indent=2, sort_keys=True) + "\n")
    if a.check:
        frozen = TEST_PATH.read_bytes()
        summary["matches_frozen"] = frozen == test_bytes
        print(json.dumps(summary))
        return 0 if frozen == test_bytes else 1
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
