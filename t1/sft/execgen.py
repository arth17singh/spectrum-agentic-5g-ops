"""Execution examples: twin states with a known degradation, labelled by the twin itself.

A spec is (scenario, seed, prefix) for W2's ReplayTwin, like the eval's (t1/evals/gen_execution.py),
with the same five fault families and three operating points plus one more (29 dBm), but:
  - seeds are drawn from [0, SEED_LIMIT) with this module's own key; the eval reserves >= 900000;
  - the label comes from the eval's own `reference(spec)` (imported, not copied): the best of the
    same 40 single-cell candidates by W2's counterfactual gain against no change, skipping
    candidates the safety filter would clip, and the same keep rules (best gain >= 0.02; the fixed
    answer cell=2;+8 dB earns at most half of it). A state those rules reject yields no example.

Generation is deterministic whatever the worker count or timing: tasks run in rounds over
increasing j per family, results are sorted by (family, j), and each family keeps the first N
accepted j. A time budget can stop it between rounds; the manifest then says so.
"""

from __future__ import annotations

import importlib
import math
import os
import time
from typing import Any, Callable

from t1.evals import canon as C
from t1.evals.common import h, pick, render_state

SEED_TAG = "sft-execution-v1"
SEED_LIMIT = 900_000  # seeds >= 900000 are reserved for evaluation (t1/evals/README.md rule 1)
FAMILIES = ("outage", "power_drop", "tilt_misconfig", "interference", "congestion")
OP_STEPS = (0, 1, 2, 3)  # 23, 29, 35, 41 dBm on the official twin (the eval uses 0, 2, 3)
OUTAGE_SCENARIOS = {"outage_cell_2": 0, "outage": 1, "outage_cell_4": 2, "outage_cell_5": 3}  # SDK names
DEFAULT_LABELER = "t1.evals.gen_execution:reference"
FACT_KEYS = ("power_dbm", "tilt_deg", "ue_count", "rsrp_p10_dbm", "sinr_p10_db")

NOTES = [
    "Edge users in this cluster are complaining about weak signal.",
    "The overnight KPI report flagged this cluster for low coverage.",
    "SLA compliance dipped over the last hour; one change, please.",
    "Field engineers report patchy service near these sites.",
    "Coverage alarms are trending up in this area.",
    "A planned change earlier today may have degraded the cluster.",
    "Users on the cluster border see poor throughput.",
    "The optimisation dashboard shows this cluster below its targets.",
]


def _all_cells(dp: float) -> str:
    return C.canon([{"cell": i, "d_power_db": dp, "d_tilt_deg": 0} for i in range(4)])


def exec_spec(family: str, j: int) -> dict[str, Any]:
    """Operating point (0-3 steps of +6 dB on every cell), then the fault, then no-op steps."""
    k = (SEED_TAG, family, j)
    seed = h("seed", k) % SEED_LIMIT
    c = h("cell", k) % 4
    noop = h("noop", k) % 3
    op_steps = pick(OP_STEPS, "op", k)
    prefix = [_all_cells(6)] * op_steps
    if family == "outage":
        scenario = pick(sorted(OUTAGE_SCENARIOS), "sc", k)
        prefix += [""] * (max(0, 6 - op_steps) + h("wait", k) % 4)
        fault = {"kind": "outage", "cell": OUTAGE_SCENARIOS[scenario], "from_step": 6}
    elif family == "power_drop":
        scenario = pick(["nominal", "congestion", "low_load"], "sc", k)
        cuts = pick([1, 2, 2, 3], "cuts", k)
        cut = C.canon([{"cell": c, "d_power_db": -6, "d_tilt_deg": 0}])
        prefix += [cut] * cuts + [""] * noop
        fault = {"kind": "power_drop", "cell": c, "d_power_db": -6 * cuts}
    elif family == "tilt_misconfig":
        scenario = pick(["nominal", "congestion"], "sc", k)
        extra = pick([0, 1, 2, 3, 4], "extra", k)
        prefix.append(C.canon([{"cell": c, "d_power_db": 0, "d_tilt_deg": 5}]))
        if extra:
            prefix.append(C.canon([{"cell": c, "d_power_db": 0, "d_tilt_deg": extra}]))
        prefix += [""] * noop
        fault = {"kind": "tilt_misconfig", "cell": c, "d_tilt_deg": 5 + extra}
    elif family == "interference":
        scenario = "interference"
        prefix += [""] * (1 + h("wait", k) % 4)
        fault = {"kind": "interference"}
    elif family == "congestion":
        scenario = "congestion"
        prefix += [""] * (1 + h("wait", k) % 4)
        fault = {"kind": "congestion"}
    else:
        raise ValueError(f"unknown family {family}")
    fault["operating_point_steps"] = op_steps
    return {"scenario": scenario, "seed": seed, "prefix": prefix, "fault": fault}


def _resolve(path: str) -> Callable[[dict[str, Any]], dict[str, Any] | None]:
    mod, _, fn = path.partition(":")
    return getattr(importlib.import_module(mod), fn)


def init_worker() -> None:
    """One thread per process: the pool provides the parallelism."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass


def exec_task(task: tuple[str, str, int]) -> dict[str, Any]:
    """Label one spec; JSON-safe result. Errors are returned, not raised (counted by the caller)."""
    labeler_path, family, j = task
    spec = exec_spec(family, j)
    t0 = time.perf_counter()
    out: dict[str, Any] = {"family": family, "j": j, "spec": spec, "ref": None}
    try:
        ref = _resolve(labeler_path)(spec)
    except Exception as e:  # noqa: BLE001 - one bad state must not stop the job; it is counted
        out["error"] = f"{type(e).__name__}: {e}"
        out["seconds"] = time.perf_counter() - t0
        return out
    out["seconds"] = time.perf_counter() - t0
    if ref is not None:
        st = ref["state"]
        out["ref"] = {
            "state_text": render_state(st),
            "facts": {key: [float(v) for v in st[key]] for key in FACT_KEYS if key in st},
            "alarm": [bool(a) for a in st["alarm"]],
            **{key: ref[key] for key in ("noop_kpi", "oracle_change", "oracle_gain", "rule_change", "rule_gain",
                                         "constant_gain")},
        }
    return out


def generate(per_family: dict[str, int], workers: int, time_budget_s: float, labeler: str = DEFAULT_LABELER,
             max_tries_factor: int = 60, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Run labelling rounds until every family has its count, the budget runs out, or a family
    exceeds max_tries_factor x its count in attempts. Returns accepted results and statistics."""
    fams = [f for f in FAMILIES if per_family.get(f, 0) > 0]
    accepted: dict[str, list[dict[str, Any]]] = {f: [] for f in fams}
    next_j = {f: 0 for f in fams}
    stats = {f: {"attempts": 0, "accepted": 0, "rejected_by_keep_rules": 0, "errors": 0, "seconds": 0.0}
             for f in fams}
    errors: list[str] = []
    exhausted: list[str] = []
    t0 = time.time()
    pool = None
    if workers > 1:
        import multiprocessing as mp

        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ.setdefault(var, "1")
        pool = mp.get_context("spawn").Pool(workers, initializer=init_worker)
    stopped_by, rounds = None, 0
    try:
        while True:
            need = [f for f in fams if len(accepted[f]) < per_family[f] and f not in exhausted]
            if not need:
                stopped_by = "target_count" if not exhausted else "target_count_or_exhausted"
                break
            if time.time() - t0 > time_budget_s:
                stopped_by = "time_budget"
                break
            per = max(1, math.ceil(2 * max(workers, 1) / len(need)))
            tasks = []
            for f in need:
                if next_j[f] >= per_family[f] * max_tries_factor:
                    exhausted.append(f)
                    continue
                tasks += [(labeler, f, j) for j in range(next_j[f], next_j[f] + per)]
                next_j[f] += per
            if not tasks:
                continue
            results = pool.map(exec_task, tasks, chunksize=1) if pool else [exec_task(t) for t in tasks]
            results.sort(key=lambda r: (FAMILIES.index(r["family"]), r["j"]))
            for r in results:
                s = stats[r["family"]]
                s["attempts"] += 1
                s["seconds"] += r.get("seconds", 0.0)
                if "error" in r:
                    s["errors"] += 1
                    if len(errors) < 20:
                        errors.append(f"{r['family']}/{r['j']}: {r['error']}")
                elif r["ref"] is None:
                    s["rejected_by_keep_rules"] += 1
                elif len(accepted[r["family"]]) < per_family[r["family"]]:
                    accepted[r["family"]].append(r)
                    s["accepted"] += 1
            rounds += 1
            if rounds % 10 == 0:
                done = sum(len(v) for v in accepted.values())
                log(f"execution: round {rounds}, {done}/{sum(per_family[f] for f in fams)} accepted, "
                    f"{time.time() - t0:.0f}s")
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    wall = time.time() - t0
    attempts = sum(s["attempts"] for s in stats.values())
    return {"accepted": accepted, "stats": stats, "stopped_by": stopped_by, "exhausted": exhausted,
            "errors": errors, "wall_s": round(wall, 1), "rounds": rounds,
            "specs_per_s": round(attempts / wall, 3) if wall > 0 else None,
            "examples_per_s": round(sum(len(v) for v in accepted.values()) / wall, 3) if wall > 0 else None,
            "workers": workers, "labeler": labeler, "time_budget_s": time_budget_s}


# --------------------------------------------------------------------------- the answer

def _finite(v: float) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(v)


def reason(change: str, facts: dict[str, list[float]], alarm: list[bool]) -> str:
    """One sentence from the visible state only (numbers formatted as the state table shows them);
    the hidden fault family is never named."""
    items = C.parse(change)
    if not items:
        raise ValueError("an execution label needs a change")
    it = items[0]  # the 40 candidates are single-cell changes
    c, dp, dt = it["cell"], it["d_power_db"], it["d_tilt_deg"]
    acts = []
    if dp:
        acts.append(f"{'raise' if dp > 0 else 'lower'} cell {c} power by {abs(dp):.1f} dB")
    if dt:
        who = "its" if dp else f"cell {c}"
        acts.append(f"{'raise' if dt > 0 else 'lower'} {who} tilt by {abs(dt):.1f} degrees")
    live = [i for i in range(len(alarm)) if not alarm[i]]

    def extreme(key: str, lowest: bool) -> bool:
        vals = [facts[key][i] for i in live if key in facts and _finite(facts[key][i])]
        if not vals or c not in live or not _finite(facts[key][c]):
            return False
        return facts[key][c] == (min(vals) if lowest else max(vals))

    notes = []
    down = [i for i in range(len(alarm)) if alarm[i]]
    if down:
        notes.append(f"cell {down[0]} is down (alarm raised)" if len(down) == 1 else
                     "cells " + ", ".join(str(i) for i in down) + " are down (alarms raised)")
    if c in live:
        if extreme("rsrp_p10_dbm", True):
            notes.append(f"cell {c} has the weakest edge RSRP ({facts['rsrp_p10_dbm'][c]:.2f} dBm)")
        elif extreme("ue_count", False):
            notes.append(f"cell {c} carries the most users ({int(facts['ue_count'][c])} UEs)")
        elif extreme("sinr_p10_db", True):
            notes.append(f"cell {c} has the lowest edge SINR ({facts['sinr_p10_db'][c]:.2f} dB)")
        elif dt < 0 and extreme("tilt_deg", False):
            notes.append(f"cell {c} has the steepest tilt ({facts['tilt_deg'][c]:.1f} degrees)")
        elif dp > 0 and extreme("power_dbm", True):
            notes.append(f"cell {c} has the lowest power ({facts['power_dbm'][c]:.1f} dBm)")
        else:
            notes.append(f"cell {c} is at {facts['power_dbm'][c]:.1f} dBm and {facts['tilt_deg'][c]:.1f} degrees")
    action = " and ".join(acts)
    return action[:1].upper() + action[1:] + ": " + " and ".join(notes[:2]) + "."


def to_example(r: dict[str, Any]) -> dict[str, Any]:
    """One accepted labelling result -> an SFT example (template execution)."""
    ref, spec, family, j = r["ref"], r["spec"], r["family"], r["j"]
    note = pick(NOTES, "note", SEED_TAG, family, j)
    answer = {"decision": "apply", "change": ref["oracle_change"],
              "reason": reason(ref["oracle_change"], ref["facts"], ref["alarm"])}
    return {
        "unit": "execution", "family": family, "key": f"ex-{family}-{j}", "template": "execution",
        "input": {"state": ref["state_text"], "note": note},
        "answer": answer,
        "meta": {"gen": SEED_TAG, "gen_index": j, "scenario": spec["scenario"], "seed": spec["seed"],
                 "prefix": spec["prefix"], "step": len(spec["prefix"]), "fault": spec["fault"],
                 "noop_kpi": ref["noop_kpi"], "oracle_gain": ref["oracle_gain"], "rule_change": ref["rule_change"],
                 "rule_gain": ref["rule_gain"], "constant_gain": ref["constant_gain"]},
    }
