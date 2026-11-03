"""Scorecard logic: pure functions, no I/O.

Joins `declared` and `result` records by run_id into one row per scored unit.

The scorecard never trusts a stored boolean. Pass or fail is recomputed from the measured value and
the DECLARED target (the target copied into the result is only compared against it). A row is
"inconsistent" when anything on the record contradicts something else; it is never silently shown as
a pass. A declared run with no result is "not checked": not a failure and not a pass.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from numbers import Real
from shlex import quote
from typing import Any

OPS = (">=", "<=", "==", ">", "<")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SHOW_RECORD = "python3 tools/show_record.py --run-id"


def show_record_command(run_id: str) -> str:
    return f"{SHOW_RECORD} {quote(run_id)}"


def is_number(v: Any) -> bool:
    return isinstance(v, Real) and not isinstance(v, bool) and math.isfinite(v)


def compare(measured: float, op: str, value: float) -> bool:
    """measured <op> value. `==` allows 1e-9 relative slack so 0.1 + 0.2 == 0.3 is not a failure."""
    if op == ">=":
        return measured >= value
    if op == "<=":
        return measured <= value
    if op == ">":
        return measured > value
    if op == "<":
        return measured < value
    if op == "==":
        return measured == value or math.isclose(measured, value, rel_tol=1e-9, abs_tol=0.0)
    raise ValueError(f"unknown comparison operator {op!r}")


def _time(s: Any) -> datetime | None:
    if not isinstance(s, str):
        return None
    try:
        t = datetime.fromisoformat(s.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def _target_problem(t: Any) -> str | None:
    if not isinstance(t, dict):
        return "declaration has no target object"
    if not isinstance(t.get("metric"), str) or not t["metric"]:
        return "declared target has no metric"
    if t.get("op") not in OPS:
        return f"declared target operator {t.get('op')!r} is not one of {', '.join(OPS)}"
    if not is_number(t.get("value")):
        return f"declared target value {t.get('value')!r} is not a finite number"
    return None


def _same_target(a: Any, b: Any) -> bool:
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    return all(a.get(k) == b.get(k) for k in ("metric", "op", "value", "unit"))


def _gain(result: dict | None, declared: dict) -> tuple[dict | None, list[dict], bool]:
    """-> (gain or None, problems, gain_missing)."""
    baseline = declared.get("baseline_run_id")
    details = result.get("details") if result else None
    g = details.get("gain") if isinstance(details, dict) else None
    if g is None:
        return None, [], bool(baseline and result)
    if not isinstance(g, dict):
        return None, [{"code": "gain_malformed", "detail": "details.gain is not an object"}], False
    out = {k: g.get(k) for k in ("value", "ci_low", "ci_high", "baseline_run_id", "paired_n")}
    problems: list[dict] = []
    if not all(is_number(out[k]) for k in ("value", "ci_low", "ci_high")):
        problems.append({"code": "gain_malformed", "detail": "gain value or interval is not a finite number"})
    elif not out["ci_low"] <= out["value"] <= out["ci_high"]:
        problems.append({"code": "gain_outside_interval",
                         "detail": f"gain {out['value']} lies outside its own interval [{out['ci_low']}, {out['ci_high']}]"})
    if baseline and out["baseline_run_id"] != baseline:
        problems.append({"code": "baseline_mismatch",
                         "detail": f"declared baseline {baseline!r}, gain computed against {out['baseline_run_id']!r}"})
    return out, problems, False


def build_row(run_id: str, declared: dict, result: dict | None) -> dict:
    """One scored unit. `declared` is required; `result` may be None (not checked)."""
    problems: list[dict] = []
    target = declared.get("target")
    tp = _target_problem(target)
    if tp:
        problems.append({"code": "target_invalid", "detail": tp})

    row: dict[str, Any] = {
        "track": declared["track"] if isinstance(declared.get("track"), str) and declared["track"] else "unknown",
        "unit": declared["unit"] if isinstance(declared.get("unit"), str) and declared["unit"] else run_id,
        "run_id": run_id,
        "target": {k: target.get(k) for k in ("metric", "op", "value", "unit")} if isinstance(target, dict) else None,
        "measured": None,
        "state": "not checked",
        "recomputed": None,
        "stored_passed": None,
        "n_items": None,
        "declared_n_items": declared.get("n_items"),
        "checked_at": None,
        "checker": None,
        "declared_at": declared.get("declared_at"),
        "baseline_run_id": declared.get("baseline_run_id"),
        "synthetic": bool(declared.get("synthetic")) or bool((result or {}).get("synthetic")),
    }

    gain, gprob, gain_missing = _gain(result, declared)
    row["gain"] = gain
    row["gain_missing"] = gain_missing
    if gain and not row["baseline_run_id"]:
        row["baseline_run_id"] = gain.get("baseline_run_id")

    if result is not None:
        measured = result.get("value")
        row.update(
            measured=measured if is_number(measured) else None,
            n_items=result.get("n_items"),
            checked_at=result.get("checked_at"),
            checker=result.get("checker"),
            stored_passed=result.get("passed") if isinstance(result.get("passed"), bool) else None,
        )
        problems += gprob
        if not is_number(measured):
            problems.append({"code": "measured_invalid", "detail": f"measured value {measured!r} is not a finite number"})
        if not tp and is_number(measured):
            ok = compare(measured, target["op"], target["value"])
            row["recomputed"] = "pass" if ok else "fail"
            sp = result.get("passed")
            if not isinstance(sp, bool):
                problems.append({"code": "stored_passed_invalid", "detail": f"stored passed is {sp!r}, not true or false"})
            elif sp != ok:
                problems.append({"code": "stored_passed_disagrees",
                                 "detail": f"stored passed={sp}, but {measured} {target['op']} {target['value']} is {ok}"})
        if not tp and result.get("metric") != target["metric"]:
            problems.append({"code": "metric_mismatch",
                             "detail": f"result measured {result.get('metric')!r}, declared target is on {target['metric']!r}"})
        if isinstance(target, dict) and not _same_target(result.get("target"), target):
            problems.append({"code": "target_changed",
                             "detail": "the target copied into the result differs from the declared target"})
        if result.get("unit") is not None and result.get("unit") != declared.get("unit"):
            problems.append({"code": "unit_mismatch",
                             "detail": f"result is for unit {result.get('unit')!r}, declaration is for {declared.get('unit')!r}"})
        if result.get("n_items") != declared.get("n_items"):
            problems.append({"code": "n_items_mismatch",
                             "detail": f"scored {result.get('n_items')!r} items, declared {declared.get('n_items')!r}"})
        td, tc = _time(declared.get("declared_at")), _time(result.get("checked_at"))
        if td and tc and tc < td:
            problems.append({"code": "checked_before_declared",
                             "detail": f"result checked at {result.get('checked_at')} is earlier than the declaration at {declared.get('declared_at')}"})
        if problems:
            row["state"] = "inconsistent"
        elif row["recomputed"]:
            row["state"] = row["recomputed"]
    elif tp:
        row["state"] = "inconsistent"

    row["problems"] = problems
    return row


def _track_key(t: str) -> tuple:
    m = re.fullmatch(r"track-(\d+)", t)
    return (0, int(m.group(1)), t) if m else (1, 0, t)


def _count(rows: list[dict]) -> dict:
    return {
        "declared": len(rows),
        "passed": sum(r["state"] == "pass" for r in rows),
        "failed": sum(r["state"] == "fail" for r in rows),
        "not_checked": sum(r["state"] == "not checked" for r in rows),
        "inconsistent": sum(r["state"] == "inconsistent" for r in rows),
    }


def build_scorecard(pairs: list[tuple[str, Any, Any]]) -> dict:
    """pairs: [(run_id, declared|None, result|None)] -> the whole scorecard.

    A result without a declaration is not a scored unit (nothing was promised before the run); it is
    listed under `orphans` and counted nowhere else."""
    rows: list[dict] = []
    orphans: list[dict] = []
    for run_id, decl, res in pairs:
        decl = decl if isinstance(decl, dict) else None
        res = res if isinstance(res, dict) else None
        if decl is None:
            if res is not None:
                orphans.append({"run_id": run_id, "reason": "has a result but no declaration, so it is not scored",
                                "unit": res.get("unit"), "synthetic": bool(res.get("synthetic"))})
            continue
        rows.append(build_row(run_id, decl, res))
    rows.sort(key=lambda r: (_track_key(r["track"]), r["unit"], r["run_id"]))
    tracks = []
    for t in sorted({r["track"] for r in rows}, key=_track_key):
        trs = [r for r in rows if r["track"] == t]
        tracks.append({"track": t, **_count(trs), "units": trs})
    return {
        "tracks": tracks,
        "totals": _count(rows),
        "synthetic_rows": sum(r["synthetic"] for r in rows),
        "orphans": sorted(orphans, key=lambda o: o["run_id"]),
        "empty": not rows and not orphans,
    }
