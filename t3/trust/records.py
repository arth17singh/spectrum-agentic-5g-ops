"""Record helpers for the trust gate: envelope, canonical change text, the kit's approval key,
and `change` / `control` / `result` records.

Field names follow BUILD_CONTRACT.md as corrected by amendment 1 (A2, A5, A14, A15). W1's
pydantic models and ``team13_contracts.change`` are the authority at merge; the canonical-text
functions here defer to W1's when it is importable and otherwise implement A2 exactly.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from typing import Any

SCORED_KEYS = ("coverage_ratio", "sla_ratio", "sinr_p10_db")
KPI_KEYS = ("coverage_ratio", "sla_ratio", "sinr_p10_db", "sinr_mean_db")
TEAM = "team13"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# ---------------------------------------------------------------- canonical change text (A2)
def _fmt(x: float) -> str:
    v = round(float(x), 1)
    if v == 0:
        v = 0.0  # no "-0.0"
    return f"{v:+.1f}"


def _canon_local(change) -> str:
    if isinstance(change, str):
        change = _parse_local(change)
    parts = sorted(change, key=lambda c: int(c["cell"]))
    return "|".join(f"cell={int(c['cell'])};d_power_db={_fmt(c.get('d_power_db', 0.0))};"
                    f"d_tilt_deg={_fmt(c.get('d_tilt_deg', 0.0))}" for c in parts)


def _parse_local(text: str) -> list[dict]:
    out = []
    if not text:
        return out
    for part in text.split("|"):
        kv = dict(item.split("=", 1) for item in part.split(";"))
        out.append({"cell": int(kv["cell"]), "d_power_db": float(kv["d_power_db"]),
                    "d_tilt_deg": float(kv["d_tilt_deg"])})
    return out


try:  # W1's implementation wins at merge
    from team13_contracts.change import canon as canon_change, parse as parse_change  # type: ignore
except Exception:  # pragma: no cover - exercised before the merge
    canon_change, parse_change = _canon_local, _parse_local


def change_key(case_id: str, item_id: str, change_text: str) -> str:
    """The kit's approval id over [case_id, item_id, change] with change as the canonical string."""
    fields = [case_id, item_id, change_text]
    return "chg-" + hashlib.sha256(json.dumps(fields).encode()).hexdigest()[:12]


def envelope(kind: str, run_id: str, *, synthetic: bool, created_at: str | None = None,
             aws_time: str | None = None, aws_time_source: str | None = None) -> dict[str, Any]:
    if not run_id:
        raise ValueError("every Track 3 record carries a run_id")
    return {"schema_version": "1", "kind": kind, "run_id": run_id, "team": TEAM,
            "created_at": created_at or utc_now(), "aws_time": aws_time,
            "aws_time_source": aws_time_source, "synthetic": bool(synthetic)}


# ---------------------------------------------------------------- scoring helpers (on deltas)
def _sign(x: float, band: float) -> int:
    if x == 0 or abs(x) < band:
        return 0
    return 1 if x > 0 else -1


def direction_agreement(predicted_delta: dict, measured_delta: dict, deadband: dict,
                        keys=SCORED_KEYS) -> bool | None:
    """Did the model call the direction of the change's effect right?

    Per key: skipped when the measured delta is inside the dead band (nothing material moved);
    otherwise it agrees when the predicted delta has the same sign (raw sign; a predicted zero
    disagrees). True when agreeing keys outnumber disagreeing ones, False when disagreeing keys
    outnumber agreeing ones, None on a tie or when every key was skipped (no verdict).
    """
    agree = disagree = 0
    for k in keys:
        band = float(deadband.get(k, 0.0))
        m = _sign(float(measured_delta[k]), band)
        if m == 0:
            continue
        if _sign(float(predicted_delta[k]), 0.0) == m:
            agree += 1
        else:
            disagree += 1
    if agree == disagree:
        return None
    return agree > disagree


def delta_errors(predicted_delta: dict, measured_delta: dict, keys=SCORED_KEYS) -> dict[str, float]:
    """Exact absolute errors on deltas (never rounded, A15)."""
    return {k: abs(float(predicted_delta[k]) - float(measured_delta[k])) for k in keys}


def error_e(predicted_delta: dict, measured_delta: dict, scales: dict[str, float]) -> float:
    """A6: e = mean_k |pred_delta_k - meas_delta_k| / scale_k (unclipped; inf if not finite)."""
    terms = [abs(float(predicted_delta[k]) - float(measured_delta[k])) / float(s) for k, s in scales.items()]
    e = sum(terms) / len(terms)
    return e if math.isfinite(e) else math.inf


def kpi_delta(after: dict, baseline: dict, keys=KPI_KEYS) -> dict[str, float]:
    return {k: float(after[k]) - float(baseline[k]) for k in keys}


# ---------------------------------------------------------------- change record (A15)
def make_change_record(*, run_id: str, case_id: str, item_id: str, change, twin: dict,
                       baseline_kpis: dict, measured_kpis: dict, synthetic: bool,
                       predicted_delta: dict | None = None, predictor: dict | None = None,
                       policy: dict | None = None, scales: dict | None = None,
                       deadband: dict | None = None, scored: bool = True,
                       unscored_reason: str | None = None, probe: bool = False,
                       trust_before: float | None = None, trust_after: float | None = None) -> dict:
    change_text = canon_change(change)
    change_list = parse_change(change_text)
    measured_delta = kpi_delta(measured_kpis, baseline_kpis)
    rec = envelope("change", run_id, synthetic=synthetic)
    rec.update({
        "case_id": case_id,
        "change_id": change_key(case_id, item_id, change_text),
        "item_id": item_id,
        "change": change_list,
        "change_text": change_text,
        "twin": {k: twin.get(k) for k in ("kind", "version", "scenario", "seed", "step")},
        "baseline_kpis": {k: float(baseline_kpis[k]) for k in KPI_KEYS},
        "measured_kpis": {k: float(measured_kpis[k]) for k in KPI_KEYS},
        "measured_delta": measured_delta,
        "scored": bool(scored) and not probe,
        "probe": bool(probe),
        "trust_before": trust_before,
        "trust_after": trust_after,
    })
    if not rec["scored"]:
        rec["unscored_reason"] = unscored_reason or ("probe" if probe else "unspecified")
    if predicted_delta is not None:
        pd = {k: float(predicted_delta[k]) for k in predicted_delta}
        rec["predicted_delta"] = pd
        if scales is not None:
            rec["error"] = {**delta_errors(pd, measured_delta),
                            "summary": error_e(pd, measured_delta, scales)}
        if deadband is not None:
            rec["direction_agree"] = direction_agreement(pd, measured_delta, deadband)
    if predictor is not None:
        rec["predictor"] = {"model_id": predictor["model_id"], "version": predictor["version"]}
    if policy is not None:
        rec["policy"] = {"model_id": policy["model_id"], "version": policy["version"]}
    return rec


# ---------------------------------------------------------------- control record
CONTROL_FIELDS = ("model_id", "version", "action", "by", "reason", "policy_name", "time")


def make_control_record(*, run_id: str, model_id: str, version: str, action: str, by: str,
                        reason: str, policy_name: str, time: str, synthetic: bool) -> dict:
    if action not in ("pause", "resume"):
        raise ValueError(f"control action must be pause or resume, not {action!r}")
    if not (by == "trust-gate" or (by.startswith("person:") and len(by) > len("person:"))):
        raise ValueError(f"control 'by' must be 'trust-gate' or 'person:<pseudonym>', not {by!r}")
    rec = envelope("control", run_id, synthetic=synthetic, created_at=time)
    rec.update({"model_id": model_id, "version": version, "action": action, "by": by,
                "reason": reason, "policy_name": policy_name, "time": time})
    return rec


# ---------------------------------------------------------------- result record
def make_result_record(*, run_id: str, unit: str, metric: str, value, target: dict | None,
                       passed: bool | None, n_items: int, checker: dict, details: dict,
                       synthetic: bool) -> dict:
    rec = envelope("result", run_id, synthetic=synthetic)
    rec.update({"unit": unit, "metric": metric, "value": value, "target": target,
                "passed": passed, "n_items": n_items, "checked_at": utc_now(),
                "checker": checker, "details": details})
    return rec
