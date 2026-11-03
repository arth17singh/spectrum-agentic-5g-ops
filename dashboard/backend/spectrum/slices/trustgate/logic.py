"""Trust gate and ladder: pure functions over evidence records (lists of dicts).

No I/O and no recomputation of anything the gate or the ladder checker already decided: trust,
error, direction agreement, gains and retention are shown exactly as recorded. What this module
adds is selection (which records belong to which model version), ordering by time, and honesty
flags (probes kept apart, unscored changes shown but excluded from the trust line, a ladder rung
whose label does not match its twin kind flagged).

Record shapes follow BUILD_CONTRACT.md plus amendment 1 (A5, A6, A14, A15): predictions and
measurements are deltas against the no-op counterfactual; only the Track 2 predictor has a trust
line (a change belongs to its `predictor`, never to its `policy`); `probe: true` changes have no
predictor; `scored: false` changes never feed trust.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Iterable

KPI_ORDER = ("coverage_ratio", "sla_ratio", "sinr_p10_db", "sinr_mean_db")
LADDER_UNIT = "t3.ladder"
NO_VIAVI_REASON = "no VIAVI rung (no Viavi access yet): retention = gain(VIAVI) / gain(REHEARSAL) is undefined"
LADDER_LABELS = ("MAVERIC", "REHEARSAL (Viavi-like Maveric)", "VIAVI")
MAVERIC_KINDS = ("maveric-inprocess", "maveric-hosted")
# threshold keys we pass through when a declared thresholds object carries them
THRESHOLD_KEYS = (
    "tau", "scales", "pause_below", "watch_below", "alpha", "warmup", "n_min", "window", "k_of",
    "direction_deadband", "weights",
)


# ---------------------------------------------------------------------------- small helpers

def num(x: Any) -> float | None:
    """A finite number, or None. Booleans are not numbers here."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    return float(x) if math.isfinite(x) else None


def parse_time(s: Any) -> float | None:
    """UTC ISO-8601 string to epoch seconds, or None."""
    if not isinstance(s, str) or not s:
        return None
    try:
        d = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.timestamp()


def _ref(x: Any) -> tuple[str, str] | None:
    if isinstance(x, dict) and isinstance(x.get("model_id"), str) and isinstance(x.get("version"), str):
        return x["model_id"], x["version"]
    return None


def _dicts(xs: Iterable[Any] | None) -> list[dict]:
    return [x for x in (xs or []) if isinstance(x, dict)]


def change_time(rec: dict) -> float | None:
    """When the change happened: the AWS-side time when the writer had one, else our write time."""
    return parse_time(rec.get("aws_time")) or parse_time(rec.get("created_at"))


def change_time_iso(rec: dict) -> str | None:
    t = rec.get("aws_time") if parse_time(rec.get("aws_time")) is not None else rec.get("created_at")
    return t if isinstance(t, str) else None


def is_probe(rec: dict) -> bool:
    return rec.get("probe") is True


def is_scored(rec: dict) -> bool:
    """Probes are never scored. `scored` is explicit in the contract; a missing flag counts as scored."""
    if is_probe(rec):
        return False
    return rec.get("scored", True) is not False


def model_key(rec: dict) -> tuple[str, str] | None:
    """The trust line belongs to the Track 2 predictor only (A6)."""
    if is_probe(rec):
        return None
    return _ref(rec.get("predictor"))


def _kpi_keys(*dicts: Any) -> list[str]:
    seen = {k for d in dicts if isinstance(d, dict) for k in d}
    return [k for k in KPI_ORDER if k in seen] + sorted(k for k in seen if k not in KPI_ORDER)


def kpi_rows(rec: dict) -> list[dict]:
    pd, md, er = rec.get("predicted_delta"), rec.get("measured_delta"), rec.get("error")
    return [
        {
            "key": k,
            "predicted_delta": num((pd or {}).get(k)) if isinstance(pd, dict) else None,
            "measured_delta": num((md or {}).get(k)) if isinstance(md, dict) else None,
            "error": num((er or {}).get(k)) if isinstance(er, dict) else None,
        }
        for k in _kpi_keys(pd, md)
    ]


def _policy(rec: dict) -> dict | None:
    r = _ref(rec.get("policy"))
    return {"model_id": r[0], "version": r[1]} if r else None


def change_row(rec: dict, tau: float | None = None) -> dict | None:
    """One change as the page shows it. Returns None when the record has no usable time."""
    t = change_time(rec)
    if t is None:
        return None
    err = rec.get("error") if isinstance(rec.get("error"), dict) else {}
    e = num(err.get("summary"))
    twin = rec.get("twin") if isinstance(rec.get("twin"), dict) else {}
    cells = [c for c in (rec.get("change") or []) if isinstance(c, dict)]
    agree = rec.get("direction_agree")
    return {
        "change_id": rec.get("change_id"),
        "item_id": rec.get("item_id"),
        "case_id": rec.get("case_id"),
        "run_id": rec.get("run_id"),
        "time": change_time_iso(rec),
        "t": t,
        "change_text": rec.get("change_text") if isinstance(rec.get("change_text"), str) else None,
        "cells": cells,
        "twin": {k: twin.get(k) for k in ("kind", "version", "scenario", "seed", "step")},
        "probe": is_probe(rec),
        "scored": is_scored(rec),
        "unscored_reason": rec.get("unscored_reason") if isinstance(rec.get("unscored_reason"), str) else None,
        "kpis": kpi_rows(rec),
        "error_summary": e,
        "direction_agree": agree if isinstance(agree, bool) else None,
        "over_tau": (e > tau) if (e is not None and tau is not None and is_scored(rec)) else None,
        "trust_before": num(rec.get("trust_before")),
        "trust_after": num(rec.get("trust_after")),
        "policy": _policy(rec),
        "synthetic": rec.get("synthetic") is True,
    }


# ---------------------------------------------------------------------------- thresholds

def extract_thresholds(obj: Any) -> dict | None:
    """The declared thresholds (tau, scales, pause line ...) out of a dict, or None when it has none.

    Accepts either {"thresholds": {...}} or the keys directly. Only known keys are kept, and a key
    is kept only if its value has the right type; nothing is invented.
    """
    if not isinstance(obj, dict):
        return None
    src = obj.get("thresholds") if isinstance(obj.get("thresholds"), dict) else obj
    out: dict[str, Any] = {}
    for k in THRESHOLD_KEYS:
        v = src.get(k)
        if isinstance(v, dict):
            d = {kk: n for kk, vv in v.items() if (n := num(vv)) is not None}
            if d:
                out[k] = d
        elif (n := num(v)) is not None:
            out[k] = n
    return out or None


def find_thresholds(model_id: str, version: str, model_rec: dict | None, results: list[dict], run_ids: set[str]) -> dict | None:
    """Declared thresholds for a model version.

    Order: the newest result record (by checked_at) whose details carry thresholds and that names this
    model version, or, if it names none, belongs to a run that has changes or controls for it; then
    the model record's metrics. `source` says which one was used.
    """
    best: tuple[float, dict, str] | None = None
    for r in _dicts(results):
        d = r.get("details")
        th = extract_thresholds(d)
        if not th or r.get("unit") == LADDER_UNIT:
            continue
        ref = _ref(d) if isinstance(d, dict) else None
        if ref is not None:
            if ref != (model_id, version):
                continue
        elif r.get("run_id") not in run_ids:
            continue
        ts = parse_time(r.get("checked_at")) or 0.0
        if best is None or ts >= best[0]:
            best = (ts, th, f"result:{r.get('run_id')}")
    if best:
        return {**best[1], "source": best[2]}
    th = extract_thresholds((model_rec or {}).get("metrics"))
    if th:
        return {**th, "source": "model.metrics"}
    return None


# ---------------------------------------------------------------------------- pauses and status

def control_event(rec: dict) -> dict | None:
    t = parse_time(rec.get("time"))
    action, by = rec.get("action"), rec.get("by")
    if t is None or action not in ("pause", "resume") or not isinstance(by, str):
        return None
    return {
        "time": rec.get("time"),
        "t": t,
        "action": action,
        "by": by,
        "by_kind": "trust-gate" if by == "trust-gate" else "person",
        "reason": rec.get("reason") if isinstance(rec.get("reason"), str) else "",
        "policy_name": rec.get("policy_name") if isinstance(rec.get("policy_name"), str) else None,
        "run_id": rec.get("run_id"),
        "synthetic": rec.get("synthetic") is True,
    }


def status_of(events: list[dict], model_rec: dict | None) -> dict:
    """Is the model paused, by whom, and does the model record agree with the control records?

    The newest control record wins. Without any, the model record's status is used. A disagreement is
    reported, never smoothed over.
    """
    ms = (model_rec or {}).get("status") if isinstance((model_rec or {}).get("status"), str) else None
    latest = events[-1] if events else None
    if latest is not None:
        paused = latest["action"] == "pause"
        basis = "control"
        conflict = ms in ("paused", "approved") and (ms == "paused") != paused
    else:
        paused, basis, conflict = ms == "paused", ("model" if ms else "none"), False
    return {
        "paused": paused,
        "paused_by": latest["by"] if (latest and paused) else None,
        "paused_by_kind": latest["by_kind"] if (latest and paused) else None,
        "paused_at": latest["time"] if (latest and paused) else None,
        "model_status": ms,
        "status_basis": basis,
        "status_conflict": bool(conflict),
    }


# ---------------------------------------------------------------------------- models and changes

def _sorted_rows(changes: list[dict], tau: float | None) -> list[dict]:
    rows = [r for c in changes if (r := change_row(c, tau)) is not None]
    rows.sort(key=lambda r: r["t"])
    return rows


def _events(controls: list[dict], key: tuple[str, str]) -> list[dict]:
    evs = [e for c in controls if _ref(c) == key and (e := control_event(c)) is not None]
    evs.sort(key=lambda e: e["t"])
    return evs


def model_keys(changes: list[dict], controls: list[dict]) -> list[tuple[str, str]]:
    keys = {k for c in _dicts(changes) if (k := model_key(c))}
    keys |= {k for c in _dicts(controls) if (k := _ref(c))}
    return sorted(keys)


def model_detail(
    key: tuple[str, str],
    changes: list[dict],
    controls: list[dict],
    model_rec: dict | None,
    results: list[dict],
) -> dict:
    """Everything the page needs for one model version."""
    changes, controls = _dicts(changes), _dicts(controls)
    mine = [c for c in changes if model_key(c) == key]
    events = _events(controls, key)
    run_ids = {c.get("run_id") for c in mine} | {c.get("run_id") for c in controls if _ref(c) == key}
    th = find_thresholds(key[0], key[1], model_rec, results, {r for r in run_ids if isinstance(r, str)})
    tau = th.get("tau") if th and isinstance(th.get("tau"), float) else None
    rows = _sorted_rows(mine, tau)
    scored = [r for r in rows if r["scored"]]
    trajectory = [
        {k: r[k] for k in ("time", "t", "change_id", "trust_before", "trust_after", "error_summary", "direction_agree", "over_tau")}
        for r in scored
    ]
    agree = [r["direction_agree"] for r in scored if r["direction_agree"] is not None]
    st = status_of(events, model_rec)
    synthetic = any(r["synthetic"] for r in rows) or any(e["synthetic"] for e in events) or (model_rec or {}).get("synthetic") is True
    return {
        "model_id": key[0],
        "version": key[1],
        **st,
        "model_kind": (model_rec or {}).get("model_kind"),
        "has_model_record": model_rec is not None,
        "thresholds": th,
        "pause_below": th.get("pause_below") if th and isinstance(th.get("pause_below"), float) else None,
        "n_changes": len(rows),
        "n_scored": len(scored),
        "n_unscored": len(rows) - len(scored),
        "n_direction_agree": sum(1 for a in agree if a),
        "n_direction_known": len(agree),
        "trust_latest": next((r["trust_after"] for r in reversed(scored) if r["trust_after"] is not None), None),
        "pauses": events,
        "points": trajectory,
        "rows": rows,
        "synthetic": bool(synthetic),
    }


def models_summary(detail: dict) -> dict:
    """The /models entry: the detail without the heavy per-change rows."""
    return {k: v for k, v in detail.items() if k != "rows"}


def probe_rows(changes: list[dict]) -> list[dict]:
    return _sorted_rows([c for c in _dicts(changes) if is_probe(c)], None)


# ---------------------------------------------------------------------------- ladder

def _kind_ok(label: str, twin_kind: Any) -> bool:
    if label == "VIAVI":
        return twin_kind == "viavi"
    return twin_kind in MAVERIC_KINDS


def _ci(x: Any) -> list[float] | None:
    if isinstance(x, (list, tuple)) and len(x) == 2:
        a, b = num(x[0]), num(x[1])
        if a is not None and b is not None:
            return [a, b]
    return None


def ladder_view(result: dict) -> dict | None:
    """A `t3.ladder` result as the page shows it, or None if it is not a usable ladder.

    Gains, CIs and retention are copied, never recomputed. A rung whose label is not one of the three
    allowed strings makes the ladder unusable; a rung whose label contradicts its twin kind is kept but
    flagged, so the page can refuse to present it under that label.
    """
    d = result.get("details")
    if result.get("unit") != LADDER_UNIT or not isinstance(d, dict):
        return None
    mid, ver = d.get("model_id"), d.get("version")
    if not isinstance(mid, str) or not isinstance(ver, str) or not isinstance(d.get("rungs"), list):
        return None
    rungs: list[dict] = []
    for r in d["rungs"]:
        if not isinstance(r, dict) or r.get("label") not in LADDER_LABELS:
            return None
        ok = _kind_ok(r["label"], r.get("twin_kind"))
        rungs.append({
            "name": r.get("name"),
            "label": r["label"],
            "twin_kind": r.get("twin_kind"),
            "n_cases": r.get("n_cases") if isinstance(r.get("n_cases"), int) and not isinstance(r.get("n_cases"), bool) else None,
            "gain": num(r.get("gain")),
            "gain_ci": _ci(r.get("gain_ci")),
            "label_ok": ok,
            "label_problem": None if ok else f"labelled {r['label']} but the twin kind is {r.get('twin_kind')!r}",
        })
    if not rungs:
        return None
    synthetic = result.get("synthetic") is True
    by_label = {r["label"]: r for r in rungs if r["label_ok"]}
    viavi, reh, mav = by_label.get("VIAVI"), by_label.get(LADDER_LABELS[1]), by_label.get("MAVERIC")
    # Retention is gain(VIAVI) / gain(REHEARSAL) and nothing else (W5's checker). Without a Viavi gain and a
    # rehearsal gain it is null whatever the record says; REHEARSAL / MAVERIC is a settings shift, not retention.
    defined = viavi is not None and viavi["gain"] is not None and reh is not None and reh["gain"] is not None
    retention = num(d.get("retention")) if defined else None
    null_reason = None
    if retention is None:
        recorded = d.get("retention_null_reason")
        null_reason = recorded if isinstance(recorded, str) and recorded else (
            NO_VIAVI_REASON if viavi is None else "no measured gain on the VIAVI or REHEARSAL rung")
    shift = d.get("settings_shift")
    settings_shift = None
    if isinstance(shift, dict) and mav is not None and reh is not None:
        settings_shift = {
            "heading": "Settings shift MAVERIC → REHEARSAL (not retention)",
            "from_label": "MAVERIC", "to_label": LADDER_LABELS[1], "point": num(shift.get("point")), "ci": _ci(shift.get("ci")),
            "what": shift.get("what") if isinstance(shift.get("what"), str) else None,
        }
    is_retention_result = result.get("metric") == "retention" and retention is not None
    return {
        "run_id": result.get("run_id"),
        "checked_at": result.get("checked_at"),
        "model_id": mid,
        "version": ver,
        "rungs": rungs,
        "top_label": rungs[-1]["label"],
        "has_viavi": viavi is not None,
        # real = a measured Viavi gain from a record that is not marked synthetic; only then may the page
        # treat the Viavi rung as a result (colour, check mark, pass state)
        # (a record's own has_real_viavi: false is a veto)
        "real_viavi": viavi is not None and viavi["gain"] is not None and not synthetic and d.get("has_real_viavi") is not False,
        "label_ok": all(r["label_ok"] for r in rungs),
        "retention": retention,
        "retention_ci": _ci(d.get("retention_ci")) if retention is not None else None,
        "retention_from": LADDER_LABELS[1] if retention is not None else None,
        "retention_to": "VIAVI" if retention is not None else None,
        "retention_null_reason": null_reason,
        # the title of the retention figure names the rungs actually divided, never the ladder's top rung
        "retention_heading": (
            "Retention REHEARSAL → VIAVI" if retention is not None
            else "Retention: not measured (no Viavi run)" if viavi is None
            else "Retention: not measured (no measured gain on the Viavi or rehearsal rung)"
        ),
        "settings_shift": settings_shift,
        # pass/fail belongs to the retention target only; a gain-only result has no retention to pass
        "passed": result.get("passed") if is_retention_result and isinstance(result.get("passed"), bool) else None,
        "target": result.get("target") if is_retention_result and isinstance(result.get("target"), dict) else None,
        "synthetic": synthetic,
    }


def latest_ladders(results: list[dict]) -> list[dict]:
    """The newest ladder per (model version, top rung label), so a rehearsal ladder and a Viavi ladder
    for the same model coexist but an older run of either is superseded."""
    best: dict[tuple[str, str, str], tuple[float, dict]] = {}
    for r in _dicts(results):
        v = ladder_view(r)
        if v is None:
            continue
        k = (v["model_id"], v["version"], v["top_label"])
        ts = parse_time(v["checked_at"]) or 0.0
        if k not in best or ts >= best[k][0]:
            best[k] = (ts, v)
    return [v for _, v in sorted(best.values(), key=lambda p: (p[1]["model_id"], p[1]["version"], p[1]["top_label"]))]
