"""Judged runs (judging.json), superseded runs (superseded.json), Track 3 live units and the real Track 2
consortium results. Generic by unit and kind: no run id is named in this file.

judging.json / superseded.json are hand-in files kept in git, not in S3. They are looked up in the evidence
root first (the sync copies them there with HANDIN=<git ref>), then at the repo root (judging.json) and under
evidence/ and t3/evals/evidence/ (superseded.json).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .logic import Store, _ci, jobs_for, num, recompute

REPO = Path(__file__).resolve().parents[5]


def _load(p: Path) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _first(paths: list[Path]) -> tuple[Any, str | None]:
    for p in paths:
        d = _load(p)
        if d is not None:
            try:
                where = p.resolve().relative_to(REPO).as_posix()
            except ValueError:
                where = f"evidence root/{p.name}"
            return d, where
    return None, None


def superseded(store: Store) -> dict[str, dict]:
    """run_id -> {reason, by: [run ids], source}. Every superseded.json found is merged."""
    out: dict[str, dict] = {}
    for p in (store.root / "superseded.json", REPO / "evidence" / "superseded.json", REPO / "t3" / "evals" / "evidence" / "superseded.json"):
        d = _load(p)
        if not isinstance(d, dict):
            continue
        for e in d.get("superseded") or []:
            if not isinstance(e, dict):
                continue
            by = (e.get("superseded_by") or {}).get("run_ids") or []
            for rid in e.get("run_ids") or []:
                if isinstance(rid, str):
                    out.setdefault(rid, {"reason": str(e.get("reason") or ""), "by": [b for b in by if isinstance(b, str)], "source": p.name})
    return out


def short_reason(reason: str) -> str:
    m = re.match(r"^([^:.]{3,60})[:.]", reason or "")
    return (m.group(1) if m else (reason or "superseded"))[:60]


# --------------------------------------------------------------------- judged runs

def judged(store: Store) -> dict | None:
    d, where = _first([store.root / "judging.json", REPO / "judging.json"])
    if not isinstance(d, dict):
        return None
    sup = superseded(store)

    def status(rid: str) -> dict:
        decl, res = store.declared(rid), store.result(rid)
        st = "missing" if not decl and not res else "declared" if not res else (recompute(res, decl) or "inconsistent")
        return {"run_id": rid, "state": st, "value": num((res or {}).get("value")), "metric": (res or {}).get("metric"),
                "target": (decl or {}).get("target"), "checked_at": (res or {}).get("checked_at"),
                "superseded": sup.get(rid), "check": (store.check(rid) or {}).get("exit_code")}

    def slot(s: dict) -> dict:
        ids = [x for x in ([s.get("run_id")] + list(s.get("run_ids") or [])) if isinstance(x, str)]
        return {"slot": s.get("slot"), "name": s.get("name"), "declared_status": s.get("status"), "note": s.get("note"),
                "waiting_for": s.get("waiting_for"), "bar": s.get("bar"),
                "related": [x for x in (s.get("bar_run_id"), s.get("base_run_id"), s.get("expected_run_id")) if isinstance(x, str)],
                "runs": [status(r) for r in ids],
                "related_status": [status(r) for r in (s.get("bar_run_id"), s.get("base_run_id")) if isinstance(r, str)]}

    slots = [slot(s) for s in d.get("slots") or [] if isinstance(s, dict)]
    edges = [slot(s) for s in d.get("edge_cases") or [] if isinstance(s, dict)]
    listed = [r for r in d.get("run_ids") or [] if isinstance(r, str)]
    return {"source": where, "primary_track": d.get("primary_track"), "slots": slots, "edge_cases": edges,
            "run_ids": [status(r) for r in listed], "supporting_run_ids": [r for r in d.get("supporting_run_ids") or [] if isinstance(r, str)],
            "filled": sum(1 for s in slots if s["runs"]), "total": len(slots)}


# --------------------------------------------------------------------- Track 3 live units

SCALAR = (str, int, float, bool)


def _facts(det: dict) -> list[str]:
    """Short, generic facts from a result's details: matched/n pairs and small scalars."""
    out = []
    for k, v in det.items():
        if isinstance(v, dict) and v and all(isinstance(x, dict) and "n" in x for x in v.values()):
            out.append(f"{k}: " + ", ".join(f"{kk} {x.get('matched', x.get('passed', '?'))}/{x['n']}" for kk, x in v.items()))
        elif isinstance(v, bool) or (isinstance(v, (int, float)) and not isinstance(v, bool)) or (isinstance(v, str) and len(v) <= 60):
            out.append(f"{k}: {v}")
        elif isinstance(v, list) and len(v) <= 4 and all(isinstance(x, SCALAR) for x in v):
            out.append(f"{k}: {v}")
    return out[:10]


def track3_live(store: Store, runs: list[str]) -> list[dict]:
    """Every t3.* run that is not an offline eval (t3.evals.*) or a trust/ladder run: deny, allow, pause, ..."""
    sup = superseded(store)
    out = []
    for run in runs:
        if not run.startswith("t3.") or run.startswith(("t3.evals.", "t3.trust.")):
            continue
        d, r = store.declared(run) or {}, store.result(run)
        det = (r or {}).get("details") or {}
        out.append({"run_id": run, "unit": d.get("unit") or (r or {}).get("unit"), "model_id": d.get("model_id"),
                    "state": (recompute(r, d) or "inconsistent") if r else "not checked", "value": num((r or {}).get("value")),
                    "metric": (r or {}).get("metric"), "target": d.get("target"), "n_items": (r or {}).get("n_items"),
                    "n_declared": d.get("n_items"), "facts": _facts(det), "ci": _ci(r) if r else None,
                    "checked_at": (r or {}).get("checked_at"), "declared_at": d.get("declared_at"),
                    "synthetic": bool(d.get("synthetic") or (r or {}).get("synthetic")), "superseded": sup.get(run),
                    "jobs": jobs_for(store, run)})
    return sorted(out, key=lambda x: (x["unit"] or "", x["checked_at"] or ""))


# --------------------------------------------------------------------- Track 2 consortium

FACT_KEYS = {
    "rounds": ("rounds", "n_rounds", "num_rounds", "rounds_completed", "final_round"),
    "clients": ("clients", "n_clients", "num_clients", "participants"),
    "secure_aggregation": ("secure_aggregation", "secagg", "aggregation", "protocol"),
    "epsilon": ("epsilon",),
    "session_id": ("session_id",),
    "final_aggregation_file": ("final_aggregation_file",),
}


def _find_fact(objs: list[dict], keys: tuple[str, ...]) -> tuple[Any, str | None]:
    for label, o in objs:
        stack = [o]
        while stack:
            x = stack.pop()
            if isinstance(x, dict):
                for k, v in x.items():
                    if k in keys and v is not None and not isinstance(v, (dict, list)):
                        return v, label
                    if isinstance(v, (dict, list)):
                        stack.append(v)
            elif isinstance(x, list):
                stack.extend(x)
    return None, None


def _segments(res: dict | None) -> dict[str, dict]:
    ps = ((res or {}).get("details") or {}).get("per_segment") or {}
    return {k: {"score": num(v.get("mean_score")), "noop": num(v.get("mean_score_noop")), "n": v.get("n")}
            for k, v in ps.items() if isinstance(v, dict)}


def track2_consortium(store: Store, runs: list[str]) -> dict:
    """Score runs (local_only.<silo>, central, fedavg_local, fl_*) and compare runs (t2.compare.*), per split."""
    sup = superseded(store)
    by_split: dict[str, dict] = {}
    for run in runs:
        if not run.startswith("t2.") or run in sup:
            continue
        d, r = store.declared(run) or {}, store.result(run)
        unit = d.get("unit") or (r or {}).get("unit") or ""
        parts = unit.split(".")
        if len(parts) < 2 or not r:
            continue
        split = parts[-1] if parts[-1] in ("disjoint", "overlap") else (r.get("details") or {}).get("split")
        if not split:
            continue
        S = by_split.setdefault(split, {"split": split, "local": {}, "central": None, "fedavg_local": None, "federated": [], "compare": []})
        det = r.get("details") or {}
        base = {"run_id": run, "unit": unit, "model_id": d.get("model_id"), "score": num(det.get("mean_score")),
                "noop": num(det.get("mean_score_noop")), "segments": _segments(r), "checked_at": r.get("checked_at"),
                "jobs": jobs_for(store, run)}
        kind = parts[1]
        if kind == "local_only" and len(parts) >= 4:
            S["local"][parts[2]] = base
        elif kind == "central":
            S["central"] = base
        elif kind == "fedavg_local":
            S["fedavg_local"] = {**base, "label": det.get("label")}
        elif kind.startswith("fl"):
            facts = {}
            objs = [("result", det), ("declaration", d)]
            for name, keys in FACT_KEYS.items():
                v, src = _find_fact(objs, keys)
                facts[name] = None if v is None else {"value": v, "from": src}
            S["federated"].append({**base, "variant": kind, "facts": facts, "note": ((((d.get("twin") or {}).get("config") or {}).get("details") or {}).get("note")),
                                   "population": ((((d.get("twin") or {}).get("config") or {}).get("details") or {}).get("population"))})
        elif kind == "compare":
            S["compare"].append({"run_id": run, "name": ".".join(parts[2:-1]) or parts[2], "state": recompute(r, d) or "inconsistent",
                                 "target": d.get("target"), "value": num(r.get("value")), "ci": _ci(r), "x_run": det.get("x_run"),
                                 "y_run": det.get("y_run"), "x_model": det.get("x_model"), "y_model": det.get("y_model"),
                                 "mean_diff": num(det.get("mean_diff")), "kind": det.get("kind"), "checked_at": r.get("checked_at")})
    # per silo: that silo's own test segment, local-only vs federated vs central (lower scaled MAE is better)
    for S in by_split.values():
        fed = max(S["federated"], key=lambda x: x["checked_at"] or "", default=None)
        S["current_federated"] = fed["run_id"] if fed else None
        silos = sorted(set(S["local"]) | {k.split("_", 1)[1] for k in (fed or {}).get("segments", {}) if k.startswith("region_")})
        rows = []
        for s in silos:
            seg = f"region_{s}"
            lo = (S["local"].get(s) or {}).get("segments", {}).get(seg, {}).get("score")
            fe = ((fed or {}).get("segments", {}).get(seg) or {}).get("score")
            ce = ((S["central"] or {}).get("segments", {}).get(seg) or {}).get("score")
            rows.append({"silo": s, "segment": seg, "local_only": lo, "federated": fe, "central": ce,
                         "federated_minus_local": None if lo is None or fe is None else fe - lo,
                         "federated_minus_central": None if fe is None or ce is None else fe - ce,
                         "local_run_id": (S["local"].get(s) or {}).get("run_id")})
        S["per_silo"] = rows
        S["distance_to_central"] = None if not fed or fed["score"] is None or not S["central"] or S["central"]["score"] is None else {
            "value": fed["score"] - S["central"]["score"], "federated": fed["score"], "central": S["central"]["score"],
            "how": "difference of the recorded mean scaled MAE on the whole shared test (federated minus central); a paired interval appears when a t2.compare run exists"}
    return {"splits": sorted(by_split.values(), key=lambda x: x["split"]),
            "score_note": "score = mean per-row scaled MAE on the shared test (declared scales); lower is better. No-op = predicting no change."}
