"""remediation-v2: the same twin-labelled behaviour cloning as v1, with BETTER INPUTS and a closed loop.

Why (v1's gate, t3.evals.remediation-policy-20261104T085949Z, and the learning curve
t1.rl.remediation-learning-curve-20261104T090005Z): v1 finds the faulted cell on 84% of the held-out faults but
restores every cell on 19%, proposes nothing on 12% and changes a healthy network on 26%; its learning curve is
flat after 50% of the data, so the declared next step is better inputs, not more data.

INPUTS (t1/rl/policy.py observation mode "masked_delta"): v1's masked observation of the current state, plus
``kpi_delta``: each KPI's change against the network's own pre-incident BASELINE (the PM-counter history an
operator has: per cell ue share, rsrp mean/p10, sinr mean/p10, coverage share, sla share; the network's
coverage and sla ratios), plus ``cell_share``, the current per-cell coverage and sla shares, plus ``serves_ues``,
which cells serve UEs now and at the baseline. The baseline is the twin state at the step before the last
configuration step that preceded the incident: ``fault_step - 1`` in the harness (t3/evals/targets.json
episode), i.e. right after commissioning. A cell that serves no UEs has no defined rsrp, sinr or shares (the twin
reports floors and None): THE NO-UE RULE in t1/rl/policy.py's docstring zeroes those deltas and shares and lets
``serves_ues`` carry the fact, identically in training (to_arrays) and inference (Policy, BaselineAware).

THE CONFIG-HISTORY RULE: the policy never sees the baseline's CONFIGURATION. Power and tilt are not differenced
and the baseline's power and tilt are never read (policy._kpi_only neutralises them before normalisation). A real
tilt drift or amplifier fault does not appear in the configuration record, and a config diff would make the
task a trivial revert. Tests: t1/rl/tests/test_remediation_v2.py (changing the baseline's power or tilt leaves
the observation identical; no observation key is a config difference).

CLOSED LOOP over the three operator steps: the label of any state is v1's (data.label_for): the RESIDUAL way back
to the commissioned configuration, clipped to one step's limits, so it is the fault's inverse on a faulted
state, the remainder after a partial fix, the undo of a wrong change, and zero on a healthy or fully fixed
state. Besides v1's sample kinds, DAgger rounds roll the current model out (deadzone off, so it learns to undo
its own small spurious changes) from the faulted and the healthy network and label every state it reaches.

DEVELOPMENT ON DEV DATA ONLY: candidates are fresh dev-split seeds from SEED_BASE (3,000,000) up; roles from
data.role_of (train 75%, val 10%, holdout 15%); the deadzone is chosen on the val incidents by playing them
through the harness. The held-out gate (t1/rl/remediation/suite/heldout-*.jsonl) is never opened: every entry
point and worker installs t1/rl/remediation/guard.py first.

DEADZONE (declared rule, decided before any result): candidates DEADZONES; on the val incidents, played through
t3/evals/run.py's run_incident, measure per-cell recovered (percell-v1), network success, and false changes on
the same networks without the fault. Start from no deadzone; adopt the candidate with the fewest healthy false
changes among those whose per-cell recovered AND network success are within DZ_TOLERANCE of no-deadzone's; ties
go to the smaller deadzone. If none reduces false changes, no deadzone.
"""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from t1.rl.remediation import bc, guard
from t1.rl.remediation import data as D

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DATA_VERSION = "t1-remediation-data-v2"
SEED_BASE = 3_000_000
N_CANDIDATES = 2400
CANDIDATES_PATH = HERE / "candidates-v2.jsonl"
META_PATH = HERE / "data-v2-meta.json"
CANDIDATES_REL = "t1/rl/remediation/candidates-v2.jsonl"
MODEL_ID = "t1-remediation-bc-delta"
ARTIFACT = "remediation_v2"
OBS_MODE = "masked_delta"
DEADZONES = ((0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (1.5, 1.5))
DZ_TOLERANCE = 0.01
HYPER = {**bc.HYPER, "dagger_rounds": 2, "rollout_deadzone": [0.0, 0.0],
         "net": "ppo_sm_50k's policy kwargs (MultiInputPolicy 64-64 tanh, log_std_init -0.5) on the extended "
                "observation space"}
GUARD_REASON = "remediation-v2 development (t1/rl/remediation/v2.py)"


def protect() -> None:
    """The held-out gate stays unread in this process (guard.py)."""
    guard.install(GUARD_REASON)


# --------------------------------------------------------------------------- candidates (fresh dev seeds)

def candidates(n: int = N_CANDIDATES) -> list[dict[str, Any]]:
    """v1's grid (data.spec_for) on the first n dev-split seeds >= SEED_BASE that nothing else uses."""
    return [D.spec_for(k, s) for k, s in enumerate(D.fresh_seeds(n, start=SEED_BASE))]


def load_candidates(path: Path = CANDIDATES_PATH, check: bool = True) -> list[dict[str, Any]]:
    data = Path(path).read_bytes()
    specs = [json.loads(x) for x in data.decode("utf-8").splitlines() if x.strip()]
    if check and D.candidates_bytes(candidates(len(specs))) != data:
        raise RuntimeError(f"{path} does not match the generator (v2.candidates): regenerate it and commit")
    return specs


def heldout_files() -> list[Path]:
    """The gate's files, by name only (never opened here): excluded from every scan."""
    from t3.evals.lessons import HELDOUT_DIR

    return sorted(p for p in Path(HELDOUT_DIR).glob("heldout-*"))


def exclusion_checks(specs: list[dict[str, Any]]) -> dict[str, Any]:
    """The seeds against every seed set and range the repo uses, plus a literal scan of the repo that skips the
    held-out gate's files (it is never opened during development; the gate is disjoint by seed block)."""
    from t3.evals import confirm as C
    from t3.evals.lessons import HELDOUT_SEED_BLOCK

    seeds = sorted(int(s["seed"]) for s in specs)
    sets = C.explicit_seed_sets()
    sets["confirm"] = {int(i["seed"]) for i in C.load_confirm()[0]}
    sets["remediation_v1_candidates"] = {int(s["seed"]) for s in D.load_candidates(check=False)}
    collisions = {k: sorted(set(seeds) & v) for k, v in sets.items()}
    lo, hi = seeds[0], seeds[-1]
    v1_lo, v1_hi = min(sets["remediation_v1_candidates"]), max(sets["remediation_v1_candidates"])
    ranges = C.KNOWN_SEED_RANGES + [
        {"what": "confirmation suite", "lo": min(sets["confirm"]), "hi": max(sets["confirm"]),
         "source": "t3/evals/suite/incidents-confirm.jsonl"},
        {"what": "remediation-v1 training candidates", "lo": v1_lo, "hi": v1_hi,
         "source": "t1/rl/remediation/candidates.jsonl"},
        {"what": "remediation held-out block (the gate; by reservation, never opened)", "lo": HELDOUT_SEED_BLOCK[0],
         "hi": HELDOUT_SEED_BLOCK[1], "source": "t3/evals/lessons.py HELDOUT_SEED_BLOCK"}]
    in_ranges = [r["what"] for r in ranges if r["lo"] <= hi and lo <= r["hi"]]
    scan = C.literal_scan(seeds, exclude=tuple([CANDIDATES_PATH, META_PATH] + heldout_files()))
    return {"seed_range": [lo, hi], "n_seeds": len(seeds), "known_ranges": ranges,
            "ranges_overlapping": in_ranges,
            "explicit_sets": {k: len(v) for k, v in sets.items()},
            "explicit_collisions": {k: v for k, v in collisions.items() if v},
            "literal_scan": {"files_scanned": scan["files_scanned"], "hits": scan["hits"][:20],
                             "n_hits": len(scan["hits"]),
                             "skipped": [str(p.relative_to(REPO)) for p in [CANDIDATES_PATH, META_PATH]
                                         + heldout_files()]},
            "split_of": {"dev": sum(1 for s in seeds if D.inc.split_of(s) == "dev"),
                         "test": sum(1 for s in seeds if D.inc.split_of(s) == "test")},
            "clean": not in_ranges and not any(collisions.values()) and not scan["hits"]
            and all(D.inc.split_of(s) == "dev" for s in seeds)}


def spec_overlap(specs: list[dict[str, Any]]) -> dict[str, Any]:
    """Spec-hash overlap with every frozen file except the held-out gate (expected 0 each); the gate is disjoint
    by seed block, and a spec hash includes its seed, so no spec can match it."""
    from t3.evals import confirm as C
    from t3.evals.lessons import HELDOUT_SEED_BLOCK

    mine = {D.spec_hash(s) for s in specs}
    files = {"test": D.inc.TEST_PATH, "dev": D.inc.DEV_PATH, "candidates": D.inc.CANDIDATES_PATH,
             "confirm": C.CONFIRM_PATH, "remediation_v1_candidates": D.CANDIDATES_PATH}
    out: dict[str, Any] = {}
    for name, p in files.items():
        rows = [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
        out[name] = len(mine & {D.spec_hash(r) for r in rows})
    seeds = [int(s["seed"]) for s in specs]
    out["heldout_gate"] = {"method": "seed block (file not opened)", "block": list(HELDOUT_SEED_BLOCK),
                           "seeds_inside": sum(1 for s in seeds if HELDOUT_SEED_BLOCK[0] <= s <= HELDOUT_SEED_BLOCK[1])}
    return out


def write_candidates(n: int = N_CANDIDATES) -> dict[str, Any]:
    """Freeze the v2 generator input and its exclusion record (local, under the guard; both are committed)."""
    protect()
    specs = candidates(n)
    data = D.candidates_bytes(specs)
    CANDIDATES_PATH.write_bytes(data)
    roles: dict[str, int] = {}
    for s in specs:
        roles[s["role"]] = roles.get(s["role"], 0) + 1
    meta = {"version": DATA_VERSION, "n_candidates": n, "seed_base": SEED_BASE, "role_salt": D.ROLE_SALT,
            "roles": roles, "candidates_sha256": hashlib.sha256(data).hexdigest(),
            "exclusion": exclusion_checks(specs), "spec_hash_overlap": spec_overlap(specs),
            "grid": "data.spec_for: incidents.candidate's (preset, tier, template) grid with the base cell over all 4 "
                    "cells (period 112); commissioning from incidents.commissioning(seed, tier)",
            "guard": "written with t1/rl/remediation/guard.py installed: the held-out gate was not opened"}
    META_PATH.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return meta


# --------------------------------------------------------------------------- states with per-cell counters

def _share(v: Any) -> float | None:
    """A share as the state carries it: a float, or None where the twin has none (a cell without UEs)."""
    return None if v is None else float(v)


def with_shares(state: dict[str, Any], twin: Any) -> dict[str, Any]:
    """``state`` plus the twin's per-cell coverage and sla shares (twin must be AT this state). A cell that serves no
    UEs has None (``ReplayTwin.kpis``); the no-UE rule in t1/rl/policy.py turns that into a defined observation."""
    k = twin.kpis()
    per = sorted(k["per_cell"], key=lambda c: int(c["cell"]))
    if [round(float(c["power_dbm"]), 6) for c in per] != [round(float(p), 6) for p in state["power_dbm"]]:
        raise RuntimeError("with_shares: the twin is not at this state")
    out = dict(state)
    out["coverage_share"] = [_share(c["coverage_share"]) for c in per]
    out["sla_share"] = [_share(c["sla_share"]) for c in per]
    return out


def jsonable(state: dict[str, Any]) -> dict[str, Any]:
    out = D.jsonable_state(state)
    for k in ("coverage_share", "sla_share"):
        if k in state:
            out[k] = [_share(x) for x in state[k]]
    return out


def baseline_step(targets: dict[str, Any]) -> int:
    """The baseline's step: the step before the last configuration step preceding the incident (fault_step - 1)."""
    return int(targets["episode"]["fault_step"]) - 1


def run_script(spec: dict[str, Any], targets: dict[str, Any],
               with_fault: bool) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """The harness's script; returns (twin, the last state with shares, the baseline state with shares)."""
    from t3.evals import incidents as inc

    tw = D._twin(spec)
    bstep = baseline_step(targets)
    state = baseline = None
    for i, ch in enumerate(inc.script(spec, targets, with_fault=with_fault), start=1):
        state = with_shares(tw.apply(ch), tw)
        if i == bstep:
            baseline = state
    if baseline is None:
        raise RuntimeError(f"no baseline at step {bstep}")
    return tw, state, baseline


def baseline_from_log(incident: dict[str, Any], action_log: list[dict[str, Any]],
                      targets: dict[str, Any]) -> dict[str, Any]:
    """THE INFERENCE CONTRACT: the baseline is the case's own action log replayed up to the baseline step on a fresh
    twin (verified against the logged filtered actions and KPIs), with its per-cell shares."""
    from t3.evals.twin_backend import ReplayTwin

    n = baseline_step(targets)
    tw = ReplayTwin.from_log(incident["scenario"], incident["seed"], incident["preset"], list(action_log[:n]))
    return with_shares(tw.state, tw)


class BaselineAware:
    """A masked_delta policy as a harness operator (t3/evals/run.py run_incident; evaluate.py healthy check).

    begin(): the baseline from the twin's action log (baseline_from_log); act(): the current state with its shares
    from the same twin, then policy.propose(state, baseline=baseline)."""

    name = "ppo_delta"
    synthetic = False

    def __init__(self, policy: Any) -> None:
        self.policy = policy
        mid = getattr(policy, "model_id", MODEL_ID)
        ver = getattr(policy, "version", None)
        self.model_id = f"{mid}@{ver}" if ver else str(mid)
        self._twin = None
        self._baseline: dict[str, Any] | None = None

    def begin(self, incident: dict[str, Any], ctx: dict[str, Any]) -> None:
        self._twin = ctx["twin"]
        self._baseline = baseline_from_log(incident, self._twin.action_log, ctx["targets"])

    def act(self, state: dict[str, Any], history: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
        from t3.evals.changes import nonzero

        if self._twin is None or self._baseline is None:
            raise RuntimeError("BaselineAware.act before begin()")
        return nonzero(self.policy.propose(with_shares(state, self._twin), baseline=self._baseline))


# --------------------------------------------------------------------------- labelled samples

def incident_samples(spec: dict[str, Any], targets: dict[str, Any]) -> dict[str, Any]:
    """v1's samples (data.incident_samples' kinds, labels and keep rule) on states with per-cell shares, plus the
    incident's baseline (the same for the faulted and the healthy run: the fault comes after it)."""
    from t3.evals import incidents as inc
    from t3.evals.graders import grade_kpis
    from t3.evals.twin_backend import graded

    built, why = inc.build(spec, targets)
    out: dict[str, Any] = {"seed": spec["seed"], "index": spec["index"], "role": spec["role"], "kept": False,
                           "reason": why, "samples": []}
    if built is None:
        return out
    gs = targets["episode"]["grade_step"]
    comm = D.commissioned_config(spec)
    samples: list[tuple[str, dict[str, Any], np.ndarray]] = []
    tw, s, base = run_script(spec, targets, with_fault=True)
    inverse = D.label_for(s, comm)
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
        s = with_shares(tw.apply([]), tw)
        samples.append((f"fault@{step}", s, D.label_for(s, comm)))
    tw, _, _ = run_script(spec, targets, with_fault=True)
    s = with_shares(tw.apply(D.as_change(inverse)), tw)
    if tw.action_log[-1]["clamped"]:
        out["reason"] = f"the inverse was clipped by the safety filter: {tw.action_log[-1]['rules']}"
        return out
    samples.append(("fixed@3", s, D.label_for(s, comm)))
    s = with_shares(tw.apply([]), tw)
    samples.append(("fixed@4", s, D.label_for(s, comm)))
    s = tw.apply([])
    assert tw.step == gs, (tw.step, gs)
    g = grade_kpis(graded(s), built["healthy_kpis_by_step"][gs], built["unfixed_kpis_by_step"][gs], targets)
    if not g["success"]:
        out["reason"] = f"the inverse does not recover the incident (recovery {g['recovery']})"
        return out
    half = np.trunc(np.round(inverse / 2 / D.QUANT, 6)) * D.QUANT + 0.0
    tw, _, _ = run_script(spec, targets, with_fault=True)
    s = with_shares(tw.apply(D.as_change(half)), tw)
    samples.append(("half@3", s, D.label_for(s, comm)))
    tw, s, hbase = run_script(spec, targets, with_fault=False)
    if jsonable(hbase) != jsonable(base):
        out["reason"] = "the healthy run's baseline differs from the faulted run's (the fault came before it?)"
        return out
    samples.append(("healthy@2", s, D.label_for(s, comm)))
    for step in (3, 4):
        s = with_shares(tw.apply([]), tw)
        samples.append((f"healthy@{step}", s, D.label_for(s, comm)))
    out.update(kept=True, reason="kept", incident_id=built["incident_id"], fault_cells=built["fault_cells"],
               inverse=inverse.tolist(), recovery_of_inverse=g["recovery"],
               commissioned={"power_dbm": comm[0].tolist(), "tilt_deg": comm[1].tolist()},
               alarm_on_fault=bool(np.any(samples[0][1].get("alarm", [False]))),
               baseline=jsonable(base),
               samples=[{"kind": k, "state": jsonable(st), "label": lab.tolist()} for k, st, lab in samples])
    return out


def _work(args: tuple[list[dict[str, Any]], dict[str, Any]]) -> list[dict[str, Any]]:
    from t3.evals.twin_backend import single_thread

    specs, targets = args
    protect()
    single_thread()
    out = []
    for spec in specs:
        t0 = time.perf_counter()
        r = incident_samples(spec, targets)
        r["latency_ms"] = (time.perf_counter() - t0) * 1000.0
        out.append(r)
    return out


def generate(specs: list[dict[str, Any]], targets: dict[str, Any], workers: int = 1,
             chunk: int = 25) -> list[dict[str, Any]]:
    chunks = [specs[i:i + chunk] for i in range(0, len(specs), chunk)]
    if workers <= 1:
        res = [_work((c, targets)) for c in chunks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            res = list(pool.map(_work, [(c, targets) for c in chunks]))
    return [r for part in res for r in part]


def to_arrays(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """masked_delta observations (t1.rl.policy.observation_for, the inference path) and normalised targets."""
    from t1.rl.policy import observation_for

    obs: dict[str, list[np.ndarray]] = {}
    tgt, info = [], []
    for r in rows:
        for smp in r["samples"]:
            o = observation_for(smp["state"], OBS_MODE, baseline=r["baseline"])
            for k, v in o.items():
                obs.setdefault(k, []).append(np.asarray(v))
            tgt.append(D.unit_target(np.asarray(smp["label"])))
            info.append((int(r["seed"]), r["role"], smp["kind"]))
    return {"obs": {k: np.stack(v) for k, v in obs.items()}, "target": np.asarray(tgt, dtype=np.float32),
            "info": info}


# --------------------------------------------------------------------------- model

def spaces() -> tuple[Any, Any, dict[str, Any]]:
    """ppo_sm_50k's observation space plus kpi_delta, cell_share and serves_ues; its action space and policy
    kwargs."""
    import gymnasium as gym

    from t1.rl.policy import CELL_SHARE_DIM, DELTA_CLIP, KPI_DELTA_DIM, SERVES_UES_DIM

    obs_space, act_space, kwargs = bc.base_spaces()
    sp = dict(obs_space.spaces)
    sp["kpi_delta"] = gym.spaces.Box(-DELTA_CLIP, DELTA_CLIP, (KPI_DELTA_DIM,), np.float32)
    sp["cell_share"] = gym.spaces.Box(-1.0, 1.0, (CELL_SHARE_DIM,), np.float32)
    sp["serves_ues"] = gym.spaces.Box(0.0, 1.0, (SERVES_UES_DIM,), np.float32)
    return gym.spaces.Dict(sp), act_space, kwargs


def new_model(seed: int = bc.SEED):
    import torch
    from stable_baselines3 import PPO

    torch.manual_seed(seed)
    obs_space, act_space, kwargs = spaces()
    return PPO("MultiInputPolicy", bc._gym_env(obs_space, act_space), policy_kwargs=kwargs, seed=seed,
               device="cpu", verbose=0)


def meta_for(version: str, deadzone: tuple[float, float] = (0.0, 0.0),
             extra: dict[str, Any] | None = None) -> dict[str, Any]:
    m = {"model_id": MODEL_ID, "version": version, "obs": OBS_MODE,
         "postprocess": {"deadzone_power_db": float(deadzone[0]), "deadzone_tilt_deg": float(deadzone[1])},
         "baseline": {"step": "fault_step - 1 (the step before the last configuration step preceding the incident)",
                      "contract": "t1/rl/remediation/README.md section 3 (v2): ran___propose replays the case's "
                                  "action log up to that step and passes both states with per-cell shares"},
         "label": "remediation-v2: behaviour cloning (+ residual DAgger) of the twin-verified way back to the "
                  "commissioned configuration on fresh dev-split incidents; inputs: v1's masked observation plus "
                  "KPI changes against the pre-incident baseline (no configuration history); deterministic"}
    if extra:
        m.update(extra)
    return m


def policy_of(model: Any, version: str, deadzone: tuple[float, float] = (0.0, 0.0)) -> Any:
    from t1.rl.policy import Policy

    return Policy(model, meta_for(version, deadzone))


# --------------------------------------------------------------------------- DAgger (closed loop)

def _rollout_one(spec: dict[str, Any], targets: dict[str, Any], policy: Any) -> dict[str, Any]:
    """The states the policy reaches before its 2nd and 3rd decisions, from the faulted and the healthy network,
    each labelled with the residual (data.label_for)."""
    comm = D.commissioned_config(spec)
    samples, base = [], None
    for with_fault, tag in ((True, "fault"), (False, "healthy")):
        tw, s, b = run_script(spec, targets, with_fault=with_fault)
        base = base or b
        for step in (3, 4):
            s = with_shares(tw.apply(policy.propose(s, baseline=b)), tw)
            samples.append({"kind": f"dagger-{tag}@{step + 1}", "state": jsonable(s),
                            "label": D.label_for(s, comm).tolist()})
    return {"baseline": jsonable(base), "samples": samples}


def _rollout_work(args: tuple[list[dict[str, Any]], dict[str, Any], str, dict[str, Any]]) -> list[dict[str, Any]]:
    from stable_baselines3 import PPO

    from t1.rl.policy import Policy
    from t3.evals.twin_backend import single_thread

    specs, targets, model_path, meta = args
    protect()
    single_thread()
    policy = Policy(PPO.load(model_path, device="cpu"), meta)
    out = []
    for s in specs:
        r = _rollout_one(s, targets, policy)
        out.append({"seed": s["seed"], "role": s["role"], "kept": True, **r})
    return out


def rollouts(specs: list[dict[str, Any]], targets: dict[str, Any], model_path: str, meta: dict[str, Any],
             workers: int = 1, chunk: int = 25) -> list[dict[str, Any]]:
    chunks = [specs[i:i + chunk] for i in range(0, len(specs), chunk)]
    if workers <= 1:
        parts = [_rollout_work((c, targets, model_path, meta)) for c in chunks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            parts = list(pool.map(_rollout_work, [(c, targets, model_path, meta) for c in chunks]))
    return [r for p in parts for r in p]


# --------------------------------------------------------------------------- deadzone (dev val only)

def _play_val(args: tuple[list[dict[str, Any]], dict[str, Any], str, str, list[list[float]]]) -> list[dict]:
    """Each val incident, built here, played through run_incident with every candidate deadzone, graded per cell,
    and checked on the same network without the fault."""
    from stable_baselines3 import PPO

    from t1.rl.policy import Policy
    from t3.evals import incidents as inc
    from t3.evals.changes import nonzero
    from t3.evals.percell import grade_incident
    from t3.evals.run import run_incident
    from t3.evals.twin_backend import ReplayTwin, single_thread

    specs, targets, model_path, version, dzs = args
    protect()
    single_thread()
    model = PPO.load(model_path, device="cpu")
    out = []
    for spec in specs:
        built, _ = inc.build(spec, targets)
        if built is None:
            continue
        row: dict[str, Any] = {"seed": spec["seed"], "incident_id": built["incident_id"], "by_dz": {}}
        for dz in dzs:
            op = BaselineAware(Policy(model, meta_for(version, (dz[0], dz[1]))))
            it = run_incident(built, op, targets)
            v = grade_incident(built, it["extra"]["change_texts"], targets)
            tw = ReplayTwin(built["scenario"], built["seed"], built["preset"])
            tw.reset()
            for ch in inc.script(built, targets, with_fault=False):
                tw.apply(ch)
            op.begin({**built, "fault": []}, {"twin": tw, "targets": targets, "run_id": "dz-healthy-check"})
            false_change = bool(nonzero(op.act(tw.state, [])))
            row["by_dz"][f"{dz[0]}/{dz[1]}"] = {"success": bool(it["passed"]), "recovery": it["score"],
                                                "percell_recovered": bool(v["recovered"]),
                                                "healthy_false_change": false_change}
        out.append(row)
    return out


def choose_deadzone(rows: list[dict[str, Any]], dzs: Iterable[tuple[float, float]] = DEADZONES,
                    tol: float = DZ_TOLERANCE) -> dict[str, Any]:
    """The declared rule (module docstring) on the val table."""
    keys = [f"{d[0]}/{d[1]}" for d in dzs]
    n = len(rows)
    table = {}
    for k in keys:
        rs = [r["by_dz"][k] for r in rows]
        table[k] = {"n": n, "percell_recovered": sum(r["percell_recovered"] for r in rs) / n if n else None,
                    "success": sum(r["success"] for r in rs) / n if n else None,
                    "healthy_false_change": sum(r["healthy_false_change"] for r in rs) / n if n else None}
    base = table[keys[0]]
    ok = [k for k in keys if n and table[k]["percell_recovered"] >= base["percell_recovered"] - tol
          and table[k]["success"] >= base["success"] - tol]
    best = min(ok, key=lambda k: (table[k]["healthy_false_change"], keys.index(k))) if ok else keys[0]
    if n and table[best]["healthy_false_change"] >= base["healthy_false_change"]:
        best = keys[0]
    chosen = tuple(float(x) for x in best.split("/"))
    return {"table": table, "chosen": list(chosen), "rule": "fewest healthy false changes among the deadzones whose "
            f"per-cell recovered and network success are within {tol} of no deadzone's; ties to the smaller; "
            "no deadzone unless it reduces false changes", "n_val_incidents": n}


def tune_deadzone(model_path: str, version: str, val_specs: list[dict[str, Any]], targets: dict[str, Any],
                  workers: int = 1, chunk: int = 10) -> dict[str, Any]:
    dzs = [list(d) for d in DEADZONES]
    chunks = [val_specs[i:i + chunk] for i in range(0, len(val_specs), chunk)]
    args = [(c, targets, model_path, version, dzs) for c in chunks]
    if workers <= 1:
        parts = [_play_val(a) for a in args]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            parts = list(pool.map(_play_val, args))
    rows = [r for p in parts for r in p]
    return {**choose_deadzone(rows), "rows": rows}


# --------------------------------------------------------------------------- training

def train(rows: list[dict[str, Any]], specs_by_seed: dict[int, dict[str, Any]], targets: dict[str, Any], *,
          version: str, workdir: Path, workers: int = 1, hyper: dict[str, Any] | None = None,
          log: Callable[[str], None] = print) -> dict[str, Any]:
    """Round 0 on the twin-labelled states, residual DAgger rounds, then the deadzone on the val incidents."""
    protect()
    h = {**HYPER, **(hyper or {})}
    model = new_model(h["seed"])
    tr_rows = [r for r in rows if r["kept"] and r["role"] == "train"]
    va_rows = [r for r in rows if r["kept"] and r["role"] == "val"]
    record: dict[str, Any] = {"hyper": h, "rounds": []}
    roll_dz = (float(h["rollout_deadzone"][0]), float(h["rollout_deadzone"][1]))

    def arrays(rs):
        if not rs:
            return {"obs": {}, "target": np.zeros((0, 8), dtype=np.float32)}
        a = to_arrays(rs)
        return {"obs": a["obs"], "target": a["target"]}

    for rnd in range(h["dagger_rounds"] + 1):
        train_a, val_a = arrays(tr_rows), arrays(va_rows)
        if not len(train_a["target"]):
            raise RuntimeError("no training samples (no kept train incidents)")
        log(f"[v2] round {rnd}: {len(train_a['target'])} train samples, {len(val_a['target'])} val samples")
        fit_rec = bc.fit(model, train_a, val_a, epochs=h["epochs"] if rnd == 0 else h["dagger_epochs"],
                         patience=h["patience"] if rnd == 0 else h["dagger_patience"], lr=h["lr"], batch=h["batch"],
                         seed=h["seed"] + rnd, log=log)
        pol = policy_of(model, version, roll_dz)
        rec = {"round": rnd, "train_samples": int(len(train_a["target"])), "val_samples": int(len(val_a["target"])),
               "fit": {k: v for k, v in fit_rec.items() if k != "history"},
               "history_tail": fit_rec["history"][-3:], "val_proposals": proposal_stats(pol, va_rows)}
        record["rounds"].append(rec)
        if rnd == h["dagger_rounds"]:
            break
        path = workdir / f"round{rnd}.zip"
        model.save(str(path))
        meta = meta_for(version, roll_dz)
        new_tr = rollouts([specs_by_seed[r["seed"]] for r in tr_rows if not r.get("dagger")], targets, str(path),
                          meta, workers)
        new_va = rollouts([specs_by_seed[r["seed"]] for r in va_rows if not r.get("dagger")], targets, str(path),
                          meta, workers)
        for r in new_tr + new_va:
            r["dagger"] = rnd + 1
        tr_rows += new_tr
        va_rows += new_va
        rec["dagger_added"] = {"train_states": sum(len(r["samples"]) for r in new_tr),
                               "val_states": sum(len(r["samples"]) for r in new_va)}
    final = workdir / "final.zip"
    model.save(str(final))
    val_specs = [specs_by_seed[r["seed"]] for r in va_rows if not r.get("dagger")]
    log(f"[v2] deadzone on {len(val_specs)} val incidents (dev data only)")
    dz = tune_deadzone(str(final), version, val_specs, targets, workers)
    record["deadzone"] = {k: v for k, v in dz.items() if k != "rows"}
    record["final_val_proposals"] = proposal_stats(policy_of(model, version, tuple(dz["chosen"])), va_rows)
    return {"model": model, "record": record, "rows": tr_rows + va_rows, "deadzone": tuple(dz["chosen"]),
            "deadzone_rows": dz["rows"]}


def proposal_stats(policy: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """bc.proposal_stats with the baseline passed (training diagnostics on the val split)."""
    zero = zero_ok = nz = cell_ok = exact = 0
    for r in rows:
        for smp in r["samples"]:
            lab = np.asarray(smp["label"])
            n = len(lab) // 2
            prop = policy.propose(smp["state"], baseline=r["baseline"])
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


def save(model: Any, out_dir: Path, version: str, deadzone: tuple[float, float], provenance: dict[str, Any],
         name: str = ARTIFACT) -> dict[str, Any]:
    """<name>.zip + <name>.meta.json (read by t1.rl.policy.load) + <name>.provenance.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    zp = out_dir / f"{name}.zip"
    model.save(str(zp))
    meta = meta_for(version, deadzone)
    (out_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    sha = hashlib.sha256(zp.read_bytes()).hexdigest()
    prov = {**provenance, "artifact": zp.name, "sha256": sha, "meta": meta}
    (out_dir / f"{name}.provenance.json").write_text(json.dumps(prov, indent=1, sort_keys=True) + "\n")
    return {"zip": str(zp), "sha256": sha, "meta": meta}
