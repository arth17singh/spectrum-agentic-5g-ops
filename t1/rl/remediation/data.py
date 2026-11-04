"""Twin-labelled remediation data. No model labels anything: every label comes from the twin.

Candidates: ``incidents.candidate``'s grid (preset, tier, template, faulted cell), but on fresh seeds and
over all four cells: candidate k uses the k-th seed s >= SEED_BASE with ``incidents.split_of(s) == "dev"``
that no other part of the repo uses (``exclusion_checks``). SEED_BASE is 1,000,000 because every range
below it is taken (t3/evals/confirm.py KNOWN_SEED_RANGES: Track 1 SFT hashes seeds into 0-899,999, the
Track 1 execution eval reserves 900,000-989,999, the silos use 100,000-499,999, the confirmation suite
990,003-990,271). Each candidate gets a role from sha256 of its seed: train (75%), val (10%, early
stopping) or holdout (15%, the evaluation's dev holdout; never trained on). The committed
``candidates.jsonl`` is the generator's input (one canonical spec per line), like t3's candidates file.

An incident is kept iff ``incidents.build`` keeps it (the fault breaks the network and nothing was
clipped) AND the twin verifies its label: the fault's exact inverse, applied at the first operator step,
recovers every graded KPI to within tolerance at the grade step, unclipped. Episode as the harness plays
it (t3/evals/targets.json): reset (0), commissioning (1), fault (2), three operator steps (3-5).

The label of ANY state is the way back to the commissioned configuration (the healthy per-cell power and
tilt after step 1), clipped to one step's limits:

    label = clip(commissioned - current)        power in [-6, +8] dB, tilt in [-5, +5] deg

so it is the fault's inverse on a faulted state, zero on a healthy or already-fixed state, the remainder
after a partial fix, and the undo of a wrong-cell change (DAgger states, bc.py). Samples per kept
incident, as the operator sees the state before each of its three steps:

    fault@2, fault@3, fault@4     the faulted network, left alone             -> the inverse
    fixed@3, fixed@4              after the inverse at step 3                 -> zero
    half@3                        after half the inverse at step 3            -> the other half
    healthy@2, healthy@3, healthy@4   the same network without the fault      -> zero
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from t3.evals import incidents as inc
from t3.evals.graders import grade_kpis, load_targets
from t3.evals.twin_backend import BACKEND, PRESET_SCENARIO, ReplayTwin, graded, single_thread

DATA_VERSION = "t1-remediation-data-v1"
SEED_BASE = 1_000_000
N_CANDIDATES = 2400
ROLE_SALT = "t1-remediation-roles-v1"
ROLE_HOLDOUT_LT, ROLE_VAL_LT = 38, 64   # first sha256 byte: < 38 holdout (14.8%), < 64 val (10.2%), else train
HERE = Path(__file__).resolve().parent
CANDIDATES_PATH = HERE / "candidates.jsonl"
META_PATH = HERE / "data-meta.json"
GRADER = ("t1/rl/remediation/data.py:v1 (twin replay: kept iff incidents.build keeps the candidate and the "
          "fault's exact inverse at the first operator step recovers every graded KPI to tolerance, unclipped)")
POWER_DELTA = (-6.0, 8.0)   # one step's limits (SDK SafetyBounds; the twin config says the same)
TILT_DELTA = 5.0
QUANT = 0.1
SAMPLE_KINDS = ("fault@2", "fault@3", "fault@4", "fixed@3", "fixed@4", "half@3", "healthy@2", "healthy@3",
                "healthy@4")


# --------------------------------------------------------------------------- candidates

def role_of(seed: int) -> str:
    b = hashlib.sha256(f"{ROLE_SALT}:{seed}".encode()).digest()[0]
    return "holdout" if b < ROLE_HOLDOUT_LT else "val" if b < ROLE_VAL_LT else "train"


def fresh_seeds(n: int, start: int = SEED_BASE, exclude: Iterable[int] = ()) -> list[int]:
    """The first n seeds >= start that ``incidents.split_of`` puts in dev and ``exclude`` does not hold.

    Refuses to walk into the remediation held-out block (heldout.SEED_LO..SEED_HI): training data must
    never come from the evaluation's seeds; pick a start beyond the block instead."""
    from t3.evals.lessons import HELDOUT_SEED_BLOCK

    ex = set(exclude)
    out, s = [], start
    while len(out) < n:
        if HELDOUT_SEED_BLOCK[0] <= s <= HELDOUT_SEED_BLOCK[1]:
            raise ValueError(f"seed {s} is in the remediation held-out block {HELDOUT_SEED_BLOCK}: training "
                             "candidates may not use it")
        if inc.split_of(s) == "dev" and s not in ex:
            out.append(s)
        s += 1
    return out


def spec_for(k: int, seed: int) -> dict[str, Any]:
    """Candidate k: incidents.candidate's grid (preset, tier, template, base cell) on ``seed``, all 4 cells."""
    preset = inc.PRESETS[k % 2]
    tier = inc.TIERS[(k // 2) % 2]
    name, parts = inc.TEMPLATES[(k // 4) % len(inc.TEMPLATES)]
    base_cell = (k // (4 * len(inc.TEMPLATES))) % 4
    fault = [{"cell": (base_cell + off) % 4, "d_power_db": float(dp), "d_tilt_deg": float(dt)} for off, dp, dt in parts]
    return {"index": k, "seed": seed, "role": role_of(seed), "preset": preset, "scenario": PRESET_SCENARIO[preset],
            "tier": tier, "template": name, "commissioning": inc.commissioning(seed, tier), "fault": fault}


def candidates(n: int = N_CANDIDATES) -> list[dict[str, Any]]:
    return [spec_for(k, s) for k, s in enumerate(fresh_seeds(n))]


def candidates_bytes(specs: list[dict[str, Any]]) -> bytes:
    return "".join(inc.canonical_line(s) + "\n" for s in specs).encode("utf-8")


def load_candidates(path: Path = CANDIDATES_PATH, check: bool = True) -> list[dict[str, Any]]:
    """The committed generator input; ``check`` refuses it unless the generator reproduces it byte for byte."""
    data = Path(path).read_bytes()
    specs = [json.loads(x) for x in data.decode("utf-8").splitlines() if x.strip()]
    if check and candidates_bytes(candidates(len(specs))) != data:
        raise RuntimeError(f"{path} does not match the generator (data.candidates): regenerate it and commit")
    return specs


# --------------------------------------------------------------------------- exclusion (freeze time)

def exclusion_checks(specs: list[dict[str, Any]]) -> dict[str, Any]:
    """The seeds against every seed set and range the repo uses, plus a literal scan of the repo."""
    from t3.evals import confirm as C

    from t3.evals.lessons import HELDOUT_SEED_BLOCK

    seeds = sorted(int(s["seed"]) for s in specs)
    sets = C.explicit_seed_sets()
    sets["confirm"] = {int(i["seed"]) for i in C.load_confirm()[0]}
    collisions = {k: sorted(set(seeds) & v) for k, v in sets.items()}
    lo, hi = seeds[0], seeds[-1]
    in_ranges = [r["what"] for r in C.KNOWN_SEED_RANGES if r["lo"] <= hi and lo <= r["hi"]]
    confirm_lo, confirm_hi = min(sets["confirm"]), max(sets["confirm"])
    if confirm_lo <= hi and lo <= confirm_hi:
        in_ranges.append("confirmation suite")
    if HELDOUT_SEED_BLOCK[0] <= hi and lo <= HELDOUT_SEED_BLOCK[1]:
        in_ranges.append("remediation held-out block")
    scan = C.literal_scan(seeds, exclude=(CANDIDATES_PATH, META_PATH))
    return {"seed_range": [lo, hi], "n_seeds": len(seeds),
            "known_ranges": C.KNOWN_SEED_RANGES + [{"what": "confirmation suite", "lo": confirm_lo, "hi": confirm_hi,
                                                    "source": "t3/evals/suite/incidents-confirm.jsonl"},
                                                   {"what": "remediation held-out block", "lo": HELDOUT_SEED_BLOCK[0],
                                                    "hi": HELDOUT_SEED_BLOCK[1],
                                                    "source": "t1/rl/remediation/heldout.py"}],
            "ranges_overlapping": in_ranges,
            "explicit_sets": {k: len(v) for k, v in sets.items()},
            "explicit_collisions": {k: v for k, v in collisions.items() if v},
            "literal_scan": {"files_scanned": scan["files_scanned"], "hits": scan["hits"][:20],
                             "n_hits": len(scan["hits"])},
            "split_of": {"dev": sum(1 for s in seeds if inc.split_of(s) == "dev"),
                         "test": sum(1 for s in seeds if inc.split_of(s) == "test")},
            "clean": not in_ranges and not any(collisions.values()) and not scan["hits"]
            and all(inc.split_of(s) == "dev" for s in seeds)}


def spec_hash(spec: dict[str, Any]) -> str:
    """What makes an incident: seed, preset, tier, template, commissioning and fault (not ids or roles)."""
    keep = {k: spec[k] for k in ("seed", "preset", "scenario", "tier", "template", "commissioning", "fault")}
    return hashlib.sha256(inc.canonical_line(keep).encode()).hexdigest()


def spec_overlap(specs: list[dict[str, Any]]) -> dict[str, int]:
    """Spec-hash overlap with the frozen test, dev, candidate, confirmation and remediation held-out files
    (expected: 0 each; the held-out file counts once it is frozen)."""
    from t3.evals import confirm as C
    from t3.evals.lessons import HELDOUT_PATH

    mine = {spec_hash(s) for s in specs}
    files = {"test": inc.TEST_PATH, "dev": inc.DEV_PATH, "candidates": inc.CANDIDATES_PATH, "confirm": C.CONFIRM_PATH,
             "heldout": HELDOUT_PATH}
    out = {}
    for name, p in files.items():
        if not Path(p).exists():
            continue
        rows = [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
        out[name] = len(mine & {spec_hash(r) for r in rows})
    return out


def write_candidates(n: int = N_CANDIDATES) -> dict[str, Any]:
    """Freeze the generator input and its exclusion record (local, before any job; both are committed)."""
    specs = candidates(n)
    data = candidates_bytes(specs)
    CANDIDATES_PATH.write_bytes(data)
    roles: dict[str, int] = {}
    for s in specs:
        roles[s["role"]] = roles.get(s["role"], 0) + 1
    meta = {"version": DATA_VERSION, "n_candidates": n, "seed_base": SEED_BASE, "role_salt": ROLE_SALT,
            "roles": roles, "candidates_sha256": hashlib.sha256(data).hexdigest(),
            "exclusion": exclusion_checks(specs), "spec_hash_overlap": spec_overlap(specs),
            "grid": "incidents.candidate's (preset, tier, template) grid with the base cell over all 4 cells "
                    "(period 112); commissioning from incidents.commissioning(seed, tier)",
            "why_not_200000": "the brief asked for 200000+, but 100,000-499,999 are the silo seeds and 0-899,999 "
                              "the Track 1 SFT seed hash range (t3/evals/confirm.py KNOWN_SEED_RANGES)"}
    META_PATH.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return meta


# --------------------------------------------------------------------------- labels

def quantise(v: np.ndarray, q: float = QUANT) -> np.ndarray:
    """Round to the 0.1 grid (labels are exact differences of 0.1-grid configurations)."""
    return np.round(np.asarray(v, dtype=np.float64) / q) * q + 0.0


def config_of(state: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(state["power_dbm"], dtype=np.float64), np.asarray(state["tilt_deg"], dtype=np.float64)


def label_for(state: dict[str, Any], commissioned: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """[d_power_db x n, d_tilt_deg x n]: the way back to the commissioned configuration, one step's worth."""
    p, t = config_of(state)
    dp = np.clip(quantise(commissioned[0] - p), *POWER_DELTA)
    dt = np.clip(quantise(commissioned[1] - t), -TILT_DELTA, TILT_DELTA)
    return np.concatenate([dp, dt])


def as_change(label: np.ndarray) -> list[dict[str, float]]:
    n = len(label) // 2
    return [{"cell": c, "d_power_db": float(label[c]), "d_tilt_deg": float(label[n + c])}
            for c in range(n) if label[c] != 0 or label[n + c] != 0]


def unit_target(label: np.ndarray, eps: float = 0.05) -> np.ndarray:
    """The normalised action whose scaled value is ``label`` (scale_action's inverse), nudged ``eps`` away
    from zero so that Policy's quantise-toward-zero lands on the label when the regression is close."""
    lab = np.asarray(label, dtype=np.float64)
    n = len(lab) // 2
    nudged = np.where(lab != 0, lab + eps * np.sign(lab), 0.0)
    pw = np.where(nudged[:n] >= 0, nudged[:n] / POWER_DELTA[1], nudged[:n] / -POWER_DELTA[0])
    return np.clip(np.concatenate([pw, nudged[n:] / TILT_DELTA]), -1.0, 1.0)


# --------------------------------------------------------------------------- one incident

def _twin(spec: dict[str, Any]) -> ReplayTwin:
    t = ReplayTwin(spec["scenario"], spec["seed"], spec["preset"])
    t.reset()
    return t


def run_script(spec: dict[str, Any], targets: dict[str, Any], with_fault: bool) -> tuple[ReplayTwin, dict[str, Any]]:
    tw = _twin(spec)
    state = None
    for ch in inc.script(spec, targets, with_fault=with_fault):
        state = tw.apply(ch)
    return tw, state


def commissioned_config(spec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    tw = _twin(spec)
    return config_of(tw.apply(list(spec["commissioning"])))


def incident_samples(spec: dict[str, Any], targets: dict[str, Any]) -> dict[str, Any]:
    """Build one candidate; if kept and verified, its labelled states (raw twin states + labels)."""
    built, why = inc.build(spec, targets)
    out: dict[str, Any] = {"seed": spec["seed"], "index": spec["index"], "role": spec["role"], "kept": False,
                           "reason": why, "samples": []}
    if built is None:
        return out
    ep = targets["episode"]
    gs = ep["grade_step"]
    comm = commissioned_config(spec)
    samples: list[tuple[str, dict[str, Any], np.ndarray]] = []
    # the faulted network, left alone
    tw, s = run_script(spec, targets, with_fault=True)
    inverse = label_for(s, comm)
    expected = np.zeros_like(inverse)
    n = len(inverse) // 2
    for f in spec["fault"]:
        expected[f["cell"]] -= f["d_power_db"]
        expected[n + f["cell"]] -= f["d_tilt_deg"]
    if not np.allclose(inverse, expected, atol=1e-9):
        out["reason"] = f"label {inverse.tolist()} is not the fault's inverse {expected.tolist()}"
        return out
    samples.append(("fault@2", s, inverse))
    for step in (3, 4):
        s = tw.apply([])
        samples.append((f"fault@{step}", s, label_for(s, comm)))
    # the inverse at the first operator step: the twin must verify it
    tw, _ = run_script(spec, targets, with_fault=True)
    s = tw.apply(as_change(inverse))
    if tw.action_log[-1]["clamped"]:
        out["reason"] = f"the inverse was clipped by the safety filter: {tw.action_log[-1]['rules']}"
        return out
    samples.append(("fixed@3", s, label_for(s, comm)))
    s = tw.apply([])
    samples.append(("fixed@4", s, label_for(s, comm)))
    s = tw.apply([])
    assert tw.step == gs, (tw.step, gs)
    g = grade_kpis(graded(s), built["healthy_kpis_by_step"][gs], built["unfixed_kpis_by_step"][gs], targets)
    if not g["success"]:
        out["reason"] = f"the inverse does not recover the incident (recovery {g['recovery']})"
        return out
    # half the inverse (toward zero on the 0.1 grid), then the remainder is the label
    half = np.trunc(np.round(inverse / 2 / QUANT, 6)) * QUANT + 0.0
    tw, _ = run_script(spec, targets, with_fault=True)
    s = tw.apply(as_change(half))
    samples.append(("half@3", s, label_for(s, comm)))
    # the same network without the fault
    tw, s = run_script(spec, targets, with_fault=False)
    samples.append(("healthy@2", s, label_for(s, comm)))
    for step in (3, 4):
        s = tw.apply([])
        samples.append((f"healthy@{step}", s, label_for(s, comm)))
    out.update(kept=True, reason="kept", incident_id=built["incident_id"], fault_cells=built["fault_cells"],
               inverse=inverse.tolist(), recovery_of_inverse=g["recovery"],
               commissioned={"power_dbm": comm[0].tolist(), "tilt_deg": comm[1].tolist()},
               alarm_on_fault=bool(np.any(samples[0][1].get("alarm", [False]))),
               samples=[{"kind": k, "state": jsonable_state(st), "label": lab.tolist()} for k, st, lab in samples])
    return out


def jsonable_state(state: dict[str, Any]) -> dict[str, Any]:
    """The fields the policy reads (t1.rl.policy.Policy._full_state), as plain JSON."""
    keep = ("power_dbm", "tilt_deg", "ue_count", "rsrp_mean_dbm", "rsrp_p10_dbm", "sinr_mean_db", "sinr_p10_db",
            "coverage_ratio", "sla_ratio", "alarm", "asleep", "hour", "phase")
    out = {}
    for k in keep:
        if k not in state:
            continue
        v = state[k]
        if isinstance(v, (list, tuple, np.ndarray)):
            out[k] = [bool(x) if isinstance(x, (bool, np.bool_)) else float(x) for x in v]
        elif isinstance(v, (bool, np.bool_)):
            out[k] = bool(v)
        else:
            out[k] = float(v)
    return out


# --------------------------------------------------------------------------- many incidents

def _work(args: tuple[list[dict[str, Any]], dict[str, Any]]) -> list[dict[str, Any]]:
    specs, targets = args
    single_thread()
    import time

    out = []
    for spec in specs:
        t0 = time.perf_counter()
        r = incident_samples(spec, targets)
        r["latency_ms"] = (time.perf_counter() - t0) * 1000.0
        out.append(r)
    return out


def generate(specs: list[dict[str, Any]], targets: dict[str, Any] | None = None, workers: int = 1,
             chunk: int = 25) -> list[dict[str, Any]]:
    """incident_samples for every spec, in spec order (deterministic; workers only split the list)."""
    targets = targets or load_targets()
    chunks = [specs[i:i + chunk] for i in range(0, len(specs), chunk)]
    if workers <= 1:
        res = [_work((c, targets)) for c in chunks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            res = list(pool.map(_work, [(c, targets) for c in chunks]))
    return [r for part in res for r in part]


def to_arrays(rows: list[dict[str, Any]], mode: str = "masked") -> dict[str, Any]:
    """Observations (t1.rl.policy.observation_for, the inference path) and normalised targets."""
    from t1.rl.policy import observation_for

    obs: dict[str, list[np.ndarray]] = {}
    tgt, info = [], []
    for r in rows:
        for smp in r["samples"]:
            o = observation_for(smp["state"], mode)
            for k, v in o.items():
                obs.setdefault(k, []).append(np.asarray(v))
            tgt.append(unit_target(np.asarray(smp["label"])))
            info.append((int(r["seed"]), r["role"], smp["kind"]))
    return {"obs": {k: np.stack(v) for k, v in obs.items()}, "target": np.asarray(tgt, dtype=np.float32),
            "info": info}


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_reason: dict[str, int] = {}
    by_role: dict[str, dict[str, int]] = {}
    for r in rows:
        key = r["reason"] if not r["kept"] else "kept"
        key = key.split(":")[0].split(" (")[0]
        by_reason[key] = by_reason.get(key, 0) + 1
        role = by_role.setdefault(r["role"], {"candidates": 0, "kept": 0, "samples": 0})
        role["candidates"] += 1
        role["kept"] += int(r["kept"])
        role["samples"] += len(r["samples"])
    return {"candidates": len(rows), "kept": sum(1 for r in rows if r["kept"]), "by_reason": by_reason,
            "by_role": by_role, "alarm_on_fault": sum(1 for r in rows if r.get("alarm_on_fault")),
            "backend": BACKEND}
