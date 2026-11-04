"""Live Fix fault catalog: the operator-language fault buttons, mapped onto the incident templates.

Each button is one of ``incidents.TEMPLATES``, placed on the cell a judge picks (the cascade also
hits the next cell, ``(cell + 1) % 4``). ``make_incident`` returns an incident in the same shape
``incidents.build`` produces (scenario, seed, preset, commissioning, fault, fault_cells, KPIs by step),
plus the per-cell KPIs and health colours the map needs, and the start log the swarm is seeded with
(``incidents.script``: commissioning at step 1, the fault at the fault step).

Seeds come from the DEV split only (``incidents.split_of``), never the frozen test split, so the demo
never touches a scored incident. A seed is used only if ``incidents.build`` keeps it: the fault must
be material at the grade step (no_action fails the tolerance) and the SDK safety filter must clip
neither the commissioning nor the fault. Otherwise the next dev seed is tried, deterministically.

The cache table ``suite/livefix-faults.json`` holds the chosen seed per (fault, cell, preset), so the
UI picks a fault instantly; ``make_incident`` without a seed reads it first and recomputes only if
the entry is missing.

    python -m t3.evals.faults --write          # (re)build the cache table (twin replay, a few minutes)
    python -m t3.evals.faults --show           # print the table
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from t3.evals import incidents as inc
from t3.evals.changes import canon, nonzero
from t3.evals.graders import load_targets
from t3.evals.twin_backend import BACKEND, PRESET_SCENARIO, ReplayTwin

CACHE_PATH = inc.SUITE_DIR / "livefix-faults.json"
CATALOG_VERSION = "livefix-faults-v1"
N_CELLS = 4
DEV_SEED_BASE = 7000  # disjoint from the frozen suite's candidates (SEED_BASE 4100 .. 4183)
MAX_TRIES = 40

# key -> (button label, incidents.TEMPLATES name, one-line description for the UI)
FAULTS: dict[str, dict[str, str]] = {
    "tilt_drift": {"label": "Antenna tilt drift", "template": "tilt+5",
                   "about": "The antenna drifts 5 degrees down: the cell's footprint shrinks."},
    "pa_fault": {"label": "Power amplifier fault", "template": "power-6",
                 "about": "The power amplifier loses 6 dB: the cell dims and edge users drop."},
    "bad_push": {"label": "Bad config push", "template": "power-3+tilt+3",
                 "about": "A bad configuration push: -3 dB power and 3 degrees more downtilt."},
    "hw_failure": {"label": "Hardware failure", "template": "power-6+tilt+5",
                   "about": "Power -6 dB and tilt +5 degrees on one cell together."},
    "cascade": {"label": "Cascade, two cells", "template": "two-cell:power-6|tilt+5",
                "about": "Two cells at once: -6 dB on the chosen cell, +5 degrees tilt on its neighbour."},
}
_TEMPLATE_PARTS = dict(inc.TEMPLATES)

# Per-cell health, relative to the same cell in the healthy counterfactual at the same step.
# A drop is healthy minus faulted (positive = worse). Shares are fractions of the cell's served UEs.
HEALTH_THRESHOLDS = {
    "red": {"coverage_share": 0.10, "sla_share": 0.10, "sinr_p10_db": 3.0},
    "amber": {"coverage_share": 0.03, "sla_share": 0.03, "sinr_p10_db": 1.0},
    "note": ("Per cell, against the same cell in the healthy counterfactual at the same step: red if any "
             "drop reaches the red line (coverage or SLA share down 0.10, or p10 SINR down 3 dB), or the "
             "cell lost all its UEs, or it is in alarm/asleep; amber if any drop reaches the amber line "
             "(0.03 / 0.03 / 1 dB); else green. A share drop counts as red only if it is at least 2 of the "
             "cell's UEs (else amber). Without a reference, absolute lines apply: red below "
             "0.85 coverage or SLA share, amber below 0.95."),
}
ABSOLUTE = {"red": 0.85, "amber": 0.95}
# A share drop turns a cell red only if it is at least this many of the cell's UEs (a 10-UE cell
# moves 0.10 per UE, so one UE alone would otherwise make it red); below that it is amber at most.
MIN_UES_FOR_RED = 2


def is_dev_seed(seed: int) -> bool:
    return inc.split_of(int(seed)) == "dev"


def dev_seeds(start: int = DEV_SEED_BASE):
    """Dev-split seeds from ``start`` upwards, in order (never a test-split seed)."""
    s = int(start)
    while True:
        if is_dev_seed(s):
            yield s
        s += 1


def fault_cells_for(fault_key: str, cell: int) -> list[dict[str, Any]]:
    """The fault change for this button on this cell (``incidents`` fault shape)."""
    if fault_key not in FAULTS:
        raise KeyError(f"unknown fault {fault_key!r}; one of {sorted(FAULTS)}")
    if not 0 <= int(cell) < N_CELLS:
        raise ValueError(f"cell must be 0..{N_CELLS - 1}")
    parts = _TEMPLATE_PARTS[FAULTS[fault_key]["template"]]
    return [{"cell": (int(cell) + off) % N_CELLS, "d_power_db": float(dp), "d_tilt_deg": float(dt)}
            for off, dp, dt in parts]


def spec_for(fault_key: str, cell: int, seed: int, preset: str = "official", tier: str = "visible") -> dict:
    """The candidate spec ``incidents.build`` takes, with this button's fault on this cell."""
    if preset not in inc.PRESETS:
        raise ValueError(f"preset must be one of {inc.PRESETS}")
    if tier not in inc.TIERS:
        raise ValueError(f"tier must be one of {inc.TIERS}")
    return {"seed": int(seed), "preset": preset, "scenario": PRESET_SCENARIO[preset], "tier": tier,
            "template": FAULTS[fault_key]["template"], "commissioning": inc.commissioning(int(seed), tier),
            "fault": fault_cells_for(fault_key, cell)}


def start_log(spec: dict[str, Any], targets: dict[str, Any]) -> list[str]:
    """The steps the swarm is seeded with (ran___seed_case's start_action_log), as live_swarm.start_log."""
    return [canon(nonzero(s)) if nonzero(s) else "" for s in inc.script(spec, targets, with_fault=True)]


# ----- per-cell KPIs and health -----

def _cell_rows(twin: Any) -> list[dict[str, Any]]:
    k = twin.kpis()
    keep = ("cell", "power_dbm", "tilt_deg", "ue_count", "sinr_p10_db", "rsrp_p10_dbm", "coverage_share",
            "sla_share", "alarm", "asleep")
    rows = []
    for row in k["per_cell"]:
        out = {}
        for key in keep:
            v = row.get(key)
            if hasattr(v, "item"):
                v = v.item()
            if isinstance(v, float):
                v = round(v, 6)
            out[key] = v
        rows.append(out)
    return rows


def per_cell_kpis(spec: dict[str, Any], targets: dict[str, Any], with_fault: bool, upto: int | None = None) -> list:
    """Per-cell KPIs right after step ``upto`` (default: the fault step) of the healthy or faulted run."""
    ep = targets["episode"]
    upto = ep["fault_step"] if upto is None else upto
    script = inc.script(spec, targets, with_fault)
    t = ReplayTwin(spec["scenario"], spec["seed"], spec["preset"])
    t.reset()
    for step in range(1, upto + 1):
        t.apply(script[step - 1] if step <= len(script) else [])
    return _cell_rows(t)


def _drop(ref: Any, cur: Any) -> float | None:
    if ref is None or cur is None:
        return None
    return float(ref) - float(cur)


def cell_health(kpis: list[dict[str, Any]], reference: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Per-cell ``{"cell", "health": green|amber|red, "why": [...]}`` (see HEALTH_THRESHOLDS)."""
    out = []
    ref_by = {r["cell"]: r for r in (reference or [])}
    for row in kpis:
        c = row["cell"]
        why: list[str] = []
        level = "green"

        def bump(to: str, reason: str) -> None:
            nonlocal level
            order = {"green": 0, "amber": 1, "red": 2}
            why.append(reason)
            if order[to] > order[level]:
                level = to

        if row.get("alarm") or row.get("asleep"):
            bump("red", "alarm" if row.get("alarm") else "asleep")
        ref = ref_by.get(c)
        if ref is not None:
            if (ref.get("ue_count") or 0) > 0 and not (row.get("ue_count") or 0):
                bump("red", "serves no UEs")
            for key in ("coverage_share", "sla_share", "sinr_p10_db"):
                d = _drop(ref.get(key), row.get(key))
                if d is None:
                    continue
                few = key != "sinr_p10_db" and d * float(ref.get("ue_count") or 0) < MIN_UES_FOR_RED
                if d >= HEALTH_THRESHOLDS["red"][key] and not few:
                    bump("red", f"{key} down {d:.3g}")
                elif d >= HEALTH_THRESHOLDS["amber"][key]:
                    bump("amber", f"{key} down {d:.3g}")
        else:
            for key in ("coverage_share", "sla_share"):
                v = row.get(key)
                if v is None:
                    continue
                if v < ABSOLUTE["red"]:
                    bump("red", f"{key} {v:.3g}")
                elif v < ABSOLUTE["amber"]:
                    bump("amber", f"{key} {v:.3g}")
        out.append({"cell": c, "health": level, "why": why})
    return out


def mark_fault_cells(health: list[dict[str, Any]], fault: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Adds ``fault_cell``: True where the fault was injected (a red neighbour is collateral)."""
    hit = {int(f["cell"]) for f in fault}
    return [dict(h, fault_cell=h["cell"] in hit) for h in health]


def fault_cells_red(incident: dict[str, Any]) -> bool:
    """Every injected cell is red right after the fault (what the judge expects to see)."""
    return all(h["health"] == "red" for h in incident["health_after"] if h.get("fault_cell"))


# ----- the incident -----

def _try(fault_key: str, cell: int, seed: int, preset: str, tier: str, targets: dict) -> tuple[dict | None, str]:
    if not is_dev_seed(seed):
        raise ValueError(f"seed {seed} is in the test split; Live Fix uses dev seeds only")
    spec = spec_for(fault_key, cell, seed, preset, tier)
    built, why = inc.build(spec, targets)
    if built is None:
        return None, why
    healthy = per_cell_kpis(spec, targets, with_fault=False)
    faulted = per_cell_kpis(spec, targets, with_fault=True)
    built.update({
        "suite": CATALOG_VERSION,
        "suite_incident_id": built["incident_id"],
        "incident_id": f"lf-{fault_key}-c{int(cell)}-{'off' if preset == 'official' else 'vvl'}-{tier[:3]}-{seed}",
        "fault_key": fault_key,
        "fault_label": FAULTS[fault_key]["label"],
        "fault_about": FAULTS[fault_key]["about"],
        "chosen_cell": int(cell),
        "start_log": start_log(spec, targets),
        "cells_before": healthy,
        "cells_after": faulted,
        "health_before": cell_health(healthy, reference=healthy),
        "health_after": mark_fault_cells(cell_health(faulted, reference=healthy), built["fault"]),
        "twin_backend": BACKEND,
    })
    return built, why


def make_incident(fault_key: str, cell: int, seed: int | None = None, preset: str = "official",
                  tier: str = "visible", targets: dict | None = None, use_cache: bool = True) -> dict[str, Any]:
    """The Live Fix incident for this button on this cell. Raises if no dev seed makes it material."""
    targets = targets or load_targets()
    tried: list[dict[str, Any]] = []
    if seed is not None:
        built, why = _try(fault_key, cell, int(seed), preset, tier, targets)
        if built is None:
            raise RuntimeError(f"{fault_key} on cell {cell} with seed {seed}: {why}")
        return built
    if use_cache:
        hit = cached_seed(fault_key, cell, preset, tier)
        if hit is not None:
            built, why = _try(fault_key, cell, hit, preset, tier, targets)
            if built is not None:
                return built
            tried.append({"seed": hit, "reason": f"cached seed no longer material: {why}"})
    first_material = None
    for n, s in enumerate(dev_seeds()):
        if n >= MAX_TRIES:
            break
        built, why = _try(fault_key, cell, s, preset, tier, targets)
        if built is None:
            tried.append({"seed": s, "reason": why})
            continue
        if fault_cells_red(built):
            built["seeds_tried"] = tried
            return built
        tried.append({"seed": s, "reason": "material, but an injected cell is not red on the map"})
        first_material = first_material or built
    if first_material is not None:
        first_material["seeds_tried"] = tried
        return first_material
    raise RuntimeError(f"no material dev seed for {fault_key} on cell {cell} ({preset}, {tier}) in "
                       f"{MAX_TRIES} tries: {tried[-3:]}")


# ----- cache table -----

def _cache_key(fault_key: str, cell: int, preset: str, tier: str) -> str:
    return f"{fault_key}|c{int(cell)}|{preset}|{tier}"


def load_cache(path: Path = CACHE_PATH) -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return {"entries": {}}


def cached_seed(fault_key: str, cell: int, preset: str = "official", tier: str = "visible",
                path: Path = CACHE_PATH) -> int | None:
    e = load_cache(path).get("entries", {}).get(_cache_key(fault_key, cell, preset, tier))
    return int(e["seed"]) if e and e.get("material") else None


def build_cache(presets=("official",), tiers=("visible",), targets: dict | None = None, log=print) -> dict:
    targets = targets or load_targets()
    entries: dict[str, Any] = {}
    for preset in presets:
        for tier in tiers:
            for fk in FAULTS:
                for cell in range(N_CELLS):
                    key = _cache_key(fk, cell, preset, tier)
                    try:
                        x = make_incident(fk, cell, preset=preset, tier=tier, targets=targets, use_cache=False)
                        entries[key] = {
                            "material": True, "seed": x["seed"], "incident_id": x["incident_id"],
                            "fault": x["fault"], "material_kpis": x["material_kpis"],
                            "red_cells": [h["cell"] for h in x["health_after"] if h["health"] == "red"],
                            "amber_cells": [h["cell"] for h in x["health_after"] if h["health"] == "amber"],
                            "fault_cells_red": fault_cells_red(x),
                            "seeds_tried": len(x.get("seeds_tried", [])),
                        }
                    except RuntimeError as exc:
                        entries[key] = {"material": False, "reason": str(exc)[:300]}
                    log(f"{key}: {entries[key].get('seed', '-')} red={entries[key].get('red_cells', '')} "
                        f"amber={entries[key].get('amber_cells', '')} ok={entries[key].get('fault_cells_red')}"
                        f"{'' if entries[key]['material'] else ' NOT MATERIAL'}")
    return {"catalog": CATALOG_VERSION, "suite": inc.SUITE_VERSION, "twin_backend": BACKEND,
            "dev_seed_base": DEV_SEED_BASE, "split": "dev only (incidents.split_of)",
            "faults": FAULTS, "health_thresholds": HEALTH_THRESHOLDS, "entries": entries}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--presets", default="official")
    a = ap.parse_args(argv)
    if a.write:
        inc.single_thread()
        table = build_cache(presets=tuple(a.presets.split(",")))
        CACHE_PATH.write_text(json.dumps(table, indent=1, sort_keys=True) + "\n")
        print(f"wrote {CACHE_PATH}")
    if a.show or not a.write:
        for k, e in load_cache().get("entries", {}).items():
            print(k, e.get("seed"), e.get("red_cells"), "" if e.get("material") else "NOT MATERIAL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
