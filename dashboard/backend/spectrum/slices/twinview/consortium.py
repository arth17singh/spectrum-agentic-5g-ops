"""Consortium mode: the Track 2 operator silos as networks of 4-cell twins, and what federation produced.

Inputs, all read by the route and passed in here (pure functions, no I/O):
  manifest   silos/manifest.json, written by our generator (silos/generate.py): dataset FACTS, with hashes.
             These are simulated operators on the in-process Maveric twin, not partner data.
  preview    silos/baselines_results.json: a LAPTOP preview (local FedAvg, not the FL service). Shown only
             as a labelled local simulation, never as a result.
  results    `result` records whose unit starts with "t2.fl" (see RESULT_SPEC): the real Track 2 runs,
             keyed by run_id. Until one exists, every metric says "not run yet".

Only aggregates are returned: never a data row, never a seed list.
"""

from __future__ import annotations

from typing import Any

from .logic import _num, prov, provenance

Rec = dict[str, Any]

SILOS = ("A", "B", "C")
KIND_ORDER = ("zero", "single_power", "single_tilt", "single_both", "multi")
FL_UNIT_PREFIX = "t2.fl"
PREVIEW_FED = "local FedAvg, not the FL service (row-weighted)"

# What a Track 2 federated run must write so this mode can show it (the ask to the Track 2 runner).
RESULT_SPEC = {
    "record": "result (BUILD_CONTRACT envelope, synthetic false, aws_time from the FL service job)",
    "unit": "t2.fl (or t2.fl.<variant>, e.g. t2.fl.dp)",
    "metric": "skill on the shared test (mean MAE / no-op MAE, lower is better)",
    "details": {
        "split": "disjoint | overlap",
        "shared_test_sha256": "sha256 of <split>/shared_test/test.csv from silos/manifest.json (must match)",
        "manifest_run_id": "silos manifest run_id, e.g. silos-v1-ecf8b521",
        "model": "{model_id, version} of the federated model served by ran___predict",
        "rounds_completed": "int: aggregation rounds completed; final_round = the round whose weights were kept",
        "final_round": "int",
        "dp": "{enabled: bool, epsilon: number|null, delta: number|null}",
        "per_silo": "{A|B|C: {local_only_skill, federated_skill, n_train_rows}} on that silo's own test segment",
        "shared_skill": "{local_only: {A,B,C}, federated: number} on the whole shared test",
    },
}


def _f(v: Any, nd: int = 4) -> float | None:
    n = _num(v)
    return None if n is None else round(n, nd)


def _mix(op: Rec) -> Rec:
    counts = op.get("change_kind_counts") if isinstance(op.get("change_kind_counts"), dict) else {}
    total = sum(v for v in counts.values() if isinstance(v, (int, float))) or 0
    shares = {k: (round(counts.get(k, 0) / total, 4) if total else None) for k in KIND_ORDER}
    pf, tf = _f(op.get("power_change_fraction")), _f(op.get("tilt_change_fraction"))
    lean = None
    if pf is not None and tf is not None:
        lean = "mostly power changes" if pf > tf + 0.1 else "mostly tilt changes" if tf > pf + 0.1 else "power and tilt"
    return {"kinds": shares, "power_change_fraction": pf, "tilt_change_fraction": tf, "lean": lean}


def silo_facts(manifest: Rec, split: str = "disjoint") -> Rec:
    cfg = manifest.get("config") if isinstance(manifest.get("config"), dict) else {}
    scfg = cfg.get("silos") if isinstance(cfg.get("silos"), dict) else {}
    files = [f for f in manifest.get("files") or [] if isinstance(f, dict) and f.get("split") == split]
    silos = []
    for s in SILOS:
        c = scfg.get(s) if isinstance(scfg.get(s), dict) else {}
        train = next((f for f in files if f.get("silo") == s and f.get("role") == "train"), {})
        test = next((f for f in files if f.get("silo") == s and f.get("role") == "test"), {})
        op = train.get("operating_point") if isinstance(train.get("operating_point"), dict) else {}
        clusters = c.get("clusters") if isinstance(c.get("clusters"), int) else train.get("clusters")
        silos.append({
            "silo": s, "size": c.get("size"), "clusters": clusters,
            "cells": clusters * 4 if isinstance(clusters, int) else None,
            "users_per_cluster": c.get("n_ues"),  # each cluster is its own twin run with this many UEs
            "mobility": c.get("mobility"), "noise_floor_dbm": _f(c.get("noise_floor_dbm"), 1),
            "power_start_dbm": c.get("power_start_dbm") if split == "disjoint" else cfg.get("overlap_power_start_dbm"),
            "tilt_start_deg": cfg.get("tilt_start_deg"),
            "train_rows": train.get("rows"), "test_rows": test.get("rows"),
            "train_sha256": train.get("sha256"), "test_sha256": test.get("sha256"),
            "mix": _mix(op),
        })
    shared = next((f for f in files if f.get("role") == "shared_test"), {})
    return {
        "split": split, "silos": silos,
        "shared_test": {"rows": shared.get("rows"), "sha256": shared.get("sha256"), "segments": shared.get("segments"),
                        "clusters": shared.get("clusters")},
        "manifest": {"run_id": manifest.get("run_id"), "config_sha256": manifest.get("config_sha256"),
                     "generator_commit": manifest.get("generator_commit"), "label": manifest.get("label"),
                     "twin": manifest.get("twin"), "contract_version": manifest.get("contract_version")},
    }


def preview_view(preview: Rec | None, split: str = "disjoint") -> Rec | None:
    """The laptop preview, cut to: per silo, local-only skill on its own segment vs local FedAvg there. Labelled LOCAL."""
    if not isinstance(preview, dict) or not isinstance(preview.get("splits"), dict):
        return None
    sp = preview["splits"].get(split)
    if not isinstance(sp, dict) or not isinstance(sp.get(PREVIEW_FED), dict):
        return None

    def skill(model: str, seg: str) -> float | None:
        m = sp.get(model)
        v = m.get(seg, {}).get("skill") if isinstance(m, dict) and isinstance(m.get(seg), dict) else None
        return _f(v[0] if isinstance(v, list) and v else v)

    rows = [{"silo": s, "segment": f"region_{s}", "local_only_skill": skill(f"local-only {s}", f"region_{s}"),
             "federated_skill": skill(PREVIEW_FED, f"region_{s}")} for s in SILOS]
    cfg = preview.get("config") if isinstance(preview.get("config"), dict) else {}
    return {"label": "LOCAL", "what": str(cfg.get("label") or "local FedAvg, not the FL service"),
            "model": cfg.get("model"), "seeds": cfg.get("seeds"), "split": split, "rows": rows,
            "metric": "skill = mean MAE / no-op MAE on the silo's own test segment (lower is better)",
            "note": "Laptop preview with local FedAvg on our own machine. Not the FL service and not a result."}


def run_view(rec: Rec, shared_sha: str | None) -> Rec:
    d = rec.get("details") if isinstance(rec.get("details"), dict) else {}
    dp = d.get("dp") if isinstance(d.get("dp"), dict) else {}
    per = d.get("per_silo") if isinstance(d.get("per_silo"), dict) else {}
    sha = d.get("shared_test_sha256")
    return {
        "run_id": rec.get("run_id"), "unit": rec.get("unit"), "metric": rec.get("metric"), "value": _num(rec.get("value")),
        "passed": rec.get("passed") if isinstance(rec.get("passed"), bool) else None,
        "time": rec.get("aws_time") or rec.get("checked_at") or rec.get("created_at"),
        "split": d.get("split"), "model": d.get("model") if isinstance(d.get("model"), dict) else None,
        "rounds_completed": d.get("rounds_completed") if isinstance(d.get("rounds_completed"), int) else None,
        "final_round": d.get("final_round") if isinstance(d.get("final_round"), int) else None,
        "dp": {"enabled": dp.get("enabled") is True, "epsilon": _num(dp.get("epsilon")), "delta": _num(dp.get("delta"))},
        "per_silo": {s: {"local_only_skill": _f((per.get(s) or {}).get("local_only_skill")),
                         "federated_skill": _f((per.get(s) or {}).get("federated_skill")),
                         "n_train_rows": (per.get(s) or {}).get("n_train_rows")} for s in SILOS if isinstance(per.get(s), dict)},
        "same_test_set": None if not sha or not shared_sha else sha == shared_sha,
        **prov(rec),
    }


def build(manifest: Rec | None, preview: Rec | None, results: list[Rec], split: str = "disjoint") -> Rec:
    facts = silo_facts(manifest, split) if isinstance(manifest, dict) else None
    shared_sha = facts["shared_test"]["sha256"] if facts else None
    fl = [r for r in results if isinstance(r, dict) and str(r.get("unit") or "").startswith(FL_UNIT_PREFIX)]
    runs = [run_view(r, shared_sha) for r in fl]
    runs.sort(key=lambda r: r["time"] or "", reverse=True)
    return {
        "split": split, "facts": facts,
        "facts_note": None if facts else "silos/manifest.json not found: no dataset facts to show",
        "runs": runs, "runs_state": "run" if runs else "not run yet",
        "runs_provenance": provenance(fl),
        "preview": preview_view(preview, split),
        "result_spec": RESULT_SPEC,
        "hub": {"name": "the FL service", "carries": "model weights only; no data leaves a silo",
                "feeds": ["ran___predict (the response model the agents call)", "trust gate (predicted vs measured)"]},
    }
