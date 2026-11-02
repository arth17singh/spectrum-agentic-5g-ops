"""Execution items: in-process Maveric twin states with a known degradation.

Each item is (scenario, seed, prefix): reset the SDK's in-process Maveric twin (official
defaults) with `scenario` and `seed`, then apply each change in `prefix` in order (the injected
fault and some no-op steps). The model sees the resulting state and proposes one change.

Grading (graders.grade_execution) replays the item from scratch and compares three next steps
from the same state: no-op, the model's change, and the SDK rule baseline. The KPI is
`coverage_ratio + sla_ratio` (both from the twin) and the gain is the KPI change against the
no-op counterfactual at the same step (contract amendment A5). The item also stores a best-of-K
reference (the best of K fixed single-cell candidates, found here at freeze time) used only to
normalise the score: score = clip(gain / oracle_gain, -1, 1).

Each state starts from one of three operating points (23, 35 or 41 dBm on every cell), then
the fault. Fault families (the hidden ground truth, in meta.fault):
  outage           SDK outage scenarios, a cell's radio off from step 6 (alarm raised)
  power_drop       one cell's power cut by 12 dB in two steps (a misconfiguration)
  tilt_misconfig   one cell's tilt pushed up by 5-9 degrees
  interference     SDK interference scenario (+3 dB power, -2 degrees tilt on every cell)
  congestion       SDK congestion scenario (hotspot on cell 0, 1.5x load)
Only states where the best-of-K gain is at least MIN_ORACLE_GAIN are kept, so every item has a
fix worth finding, and where the fixed answer CONSTANT_CHANGE earns at most half of it. Seeds are drawn from SEED_BASE upward; SFT data must not use seeds >= SEED_BASE.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from t1.evals import canon as C
from t1.evals.common import h, item, pick, render_state

UNIT = "execution"
SEED = "execution-v1"
SEED_BASE = 900_000  # reserved for evaluation; training data must use seeds below this
PER_FAMILY = 16
MIN_ORACLE_GAIN = 0.02
# Fixed-answer control: on the official twin "+8 dB on cell 2" is the best move in many states
# (the twin starts at 23 dBm, where extra power on the busiest cell raises coverage). A state is
# kept only if that fixed answer earns at most CONSTANT_MAX_SHARE of the best-of-K gain, so the
# unit rewards reading the state rather than repeating one change.
CONSTANT_CHANGE = "cell=2;d_power_db=+8.0;d_tilt_deg=+0.0"
CONSTANT_MAX_SHARE = 0.5
KPI_KEYS = ("coverage_ratio", "sla_ratio")
OUTAGE_SCENARIOS = {"outage_cell_2": 0, "outage": 1, "outage_cell_4": 2, "outage_cell_5": 3}
NOTES = [
    "Coverage and SLA KPIs dropped after last night's maintenance window.",
    "Customers in the area report dropped calls and slow data.",
    "The NOC dashboard raised a KPI degradation ticket for this cluster.",
    "Performance is below target this hour; please fix it with one change.",
]


def kpi(state: dict[str, Any]) -> float:
    return float(sum(float(state[k]) for k in KPI_KEYS))


def candidates() -> list[str]:
    """The K fixed best-of-K candidates: single-cell power or tilt moves inside the per-step limits."""
    out = []
    for c in range(4):
        for dp, dt in [(-3, 0), (3, 0), (6, 0), (8, 0), (0, -3), (0, -1), (0, 1), (0, 3), (3, -1), (6, -2)]:
            out.append(C.canon([{"cell": c, "d_power_db": dp, "d_tilt_deg": dt}]))
    return out


def _all_cells(dp: float) -> str:
    return C.canon([{"cell": i, "d_power_db": dp, "d_tilt_deg": 0} for i in range(4)])


def _spec(family: str, j: int) -> dict[str, Any]:
    """Operating point first (0, 2 or 3 steps of +6 dB on every cell: 23, 35 or 41 dBm on the
    official twin), then the fault, then a few no-op steps. Without the higher operating points
    "+8 dB on the busiest cell" is the best answer almost everywhere, because the SDK twin starts
    at 23 dBm where any extra power raises coverage."""
    k = (SEED, family, j)
    seed = SEED_BASE + h("seed", k) % 90_000
    c = h("cell", k) % 4
    noop = h("w", k) % 3
    op_steps = pick([0, 2, 3], "op", k)
    prefix = [_all_cells(6)] * op_steps
    if family == "outage":
        scenario = pick(sorted(OUTAGE_SCENARIOS), "sc", k)
        prefix += [""] * (max(0, 6 - op_steps) + h("w", k) % 4)
        fault = {"kind": "outage", "cell": OUTAGE_SCENARIOS[scenario], "from_step": 6}
    elif family == "power_drop":
        scenario = pick(["nominal", "congestion", "low_load"], "sc", k)
        cut = C.canon([{"cell": c, "d_power_db": -6, "d_tilt_deg": 0}])
        prefix += [cut, cut] + [""] * noop
        fault = {"kind": "power_drop", "cell": c, "d_power_db": -12}
    elif family == "tilt_misconfig":
        scenario = pick(["nominal", "congestion"], "sc", k)
        extra = pick([0, 2, 4], "t", k)
        prefix.append(C.canon([{"cell": c, "d_power_db": 0, "d_tilt_deg": 5}]))
        if extra:
            prefix.append(C.canon([{"cell": c, "d_power_db": 0, "d_tilt_deg": extra}]))
        prefix += [""] * noop
        fault = {"kind": "tilt_misconfig", "cell": c, "d_tilt_deg": 5 + extra}
    elif family == "interference":
        scenario = "interference"
        prefix += [""] * (1 + h("w", k) % 4)
        fault = {"kind": "interference", "power_offset_db": 3, "tilt_offset_deg": -2}
    else:
        scenario = "congestion"
        prefix += [""] * (1 + h("w", k) % 4)
        fault = {"kind": "congestion", "hotspot_cell": 0, "load_multiplier": 1.5}
    fault["operating_point_steps"] = op_steps
    return {"scenario": scenario, "seed": seed, "prefix": prefix, "fault": fault}


def _fork(tw: Any) -> Any:
    """An independent copy of a stand-in ReplayTwin mid-episode (generation-time speed-up only;
    tests check that a fork and a from-scratch replay give identical KPIs)."""
    new = copy.copy(tw)
    inner = copy.copy(tw._twin)
    for name, v in vars(tw._twin).items():
        if isinstance(v, (np.ndarray, np.random.Generator, list, dict)) and name not in ("_models", "_cells", "_scenarios"):
            setattr(inner, name, copy.deepcopy(v))
    new._twin = inner
    new._state = copy.deepcopy(tw._state)
    new.action_log = copy.deepcopy(tw.action_log)
    return new


def build_state(spec: dict[str, Any]) -> Any:
    from t1.evals.twin_standin import ReplayTwin

    tw = ReplayTwin(spec["scenario"], spec["seed"], None)
    tw.reset()
    for ch in spec["prefix"]:
        tw.apply(C.parse(ch))
    return tw


def reference(spec: dict[str, Any]) -> dict[str, Any] | None:
    from t1.evals.twin_standin import rule_baseline_change

    base = build_state(spec)
    state = base.state
    noop_k = kpi(_fork(base).apply([]))
    best, best_gain = "", float("-inf")
    for cand in candidates():
        tw = _fork(base)
        g = kpi(tw.apply(C.parse(cand))) - noop_k
        if tw.action_log[-1]["clamped"]:
            continue  # the reference is a change the filter lets through untouched
        if g > best_gain + 1e-12:
            best, best_gain = cand, g
    rule = rule_baseline_change(state)
    rule_gain = kpi(_fork(base).apply(C.parse(rule))) - noop_k
    const_gain = kpi(_fork(base).apply(C.parse(CONSTANT_CHANGE))) - noop_k
    if best_gain < MIN_ORACLE_GAIN or const_gain > CONSTANT_MAX_SHARE * best_gain:
        return None
    return {
        "state": state,
        "noop_kpi": round(noop_k, 6),
        "oracle_change": best,
        "oracle_gain": round(best_gain, 6),
        "rule_change": rule,
        "rule_gain": round(rule_gain, 6),
        "constant_gain": round(const_gain, 6),
    }


def make_item(family: str, n: int, spec: dict[str, Any], ref: dict[str, Any], gen_index: int) -> dict:
    from t1.evals.twin_standin import sdk_version

    k = (SEED, family, n)
    return item(
        UNIT,
        f"ex-{family}-{n:03d}",
        "execution",
        {"state": render_state(ref["state"]), "note": pick(NOTES, "note", k)},
        {
            "kpi": "+".join(KPI_KEYS),
            "noop_kpi": ref["noop_kpi"],
            "oracle_change": ref["oracle_change"],
            "oracle_gain": ref["oracle_gain"],
            "rule_change": ref["rule_change"],
            "rule_gain": ref["rule_gain"],
            "constant_change": CONSTANT_CHANGE,
            "constant_gain": ref["constant_gain"],
            "min_gain_to_pass": 0.01,
        },
        {
            "family": family,
            "gen": SEED,
            "twin": {"kind": "maveric-inprocess", "config": "official", "sdk_version": sdk_version()},
            "scenario": spec["scenario"],
            "seed": spec["seed"],
            "prefix": spec["prefix"],
            "fault": spec["fault"],
            "k_candidates": len(candidates()),
            "gen_index": gen_index,
        },
    )


FAMILIES = ("outage", "power_drop", "tilt_misconfig", "interference", "congestion")


def generate(per_family: int = PER_FAMILY, families: tuple[str, ...] = FAMILIES) -> list[dict]:
    out = []
    for family in families:
        n, j = 0, 0
        while n < per_family:
            spec = _spec(family, j)
            j += 1
            ref = reference(spec)
            if ref is None:
                continue
            out.append(make_item(family, n, spec, ref, j - 1))
            n += 1
            if j > per_family * 60:
                raise RuntimeError(f"too few fixable states for {family}")
    return out
