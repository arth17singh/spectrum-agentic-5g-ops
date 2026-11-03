"""Pure functions behind the Twin view: change records (plus the declarations, decisions and suite items
that say who approved them) in; per-case windows with the 4-cell state before and after each change out.

Nothing here reads a file or calls AWS, and nothing is filled in. In particular:
  - per-cell power, tilt, users and KPIs come ONLY from the PROPOSED `twin_state` field on a change
    record (see FIELD_SPEC). The W1 contract does not have it yet, so real records show the headline
    KPIs and say that the per-cell state was not recorded;
  - the rung is R3 only for twin kind `viavi`, R2 only when the preset (twin.preset) is `viavi-like`, R1 only
    when it is known to be another Maveric config; an unknown preset is said to be unknown;
  - an approver is shown only when a record names one; a decision ALLOW gives the time, not the person;
  - provenance is REAL only if every record shown is not synthetic and has a time AWS recorded for the event;
  - every window and case also carries `aws_time` + `aws_time_source` (cloudwatch | cloudwatch_attest | s3_stored | policy_createdAt |
    gateway_date | none, see `_shared/awstime.py`) and `runner_time`, kept apart: a runner or Lambda clock is never
    reported as AWS time. `time` stays the sort key it was.
"""

from __future__ import annotations

import re
import shlex
from datetime import datetime, timezone
from typing import Any, Iterable

from spectrum.slices._shared import awstime

Rec = dict[str, Any]

KPI_KEYS = ("coverage_ratio", "sla_ratio", "sinr_p10_db", "sinr_mean_db")
CELL_NAMES = ("S1/B3/C1", "S2/B3/C1", "S3/B3/C1", "S4/B3/C1")
CELL_KEYS = ("power_dbm", "tilt_deg", "ue_count", "coverage_share", "sla_share", "sinr_p10_db", "sinr_mean_db",
             "rsrp_mean_dbm", "rsrp_p10_dbm", "alarm", "asleep")
MAVERIC_KINDS = ("maveric-inprocess", "maveric-hosted")
REHEARSAL_PRESET = "viavi-like"

# What a change record carries for this view to draw the cells (contract decision 2 Nov: W1 adds twin_state
# with per-cell CellState and twin.preset; the approver is the existing approved_by / approved_at).
FIELD_SPEC = {
    "field": "twin_state",
    "on": "change record (written by ran___apply_change)",
    "shape": {
        "before": "TwinSnapshot at twin.step, the state the agents observed",
        "after": "TwinSnapshot after the change and one step (what measured_kpis summarises)",
        "baseline": "optional TwinSnapshot: the no-change counterfactual at the same step",
    },
    "snapshot": "{step, coverage_ratio, sla_ratio, per_cell: [CellState {cell, power_dbm, tilt_deg, ue_count, coverage_share, "
                "sla_share, sinr_p10_db, sinr_mean_db, rsrp_mean_dbm, rsrp_p10_dbm, alarm, asleep}]}",
    "shares": "coverage_share = served UEs with serving RSRP > -97 dBm (RSRP only); sla_share = SINR > 0 dB (strict); "
              "None for a 0-UE cell (W2 2977a7e: twin.kpis(state, shares=...))",
    "also": "twin.preset (e.g. official | viavi-like) on the change record; approver from the top-level approved_by / approved_at",
}


# ---------------------------------------------------------------------------------------------- small helpers

def _num(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    f = float(v)
    return f if f == f and f not in (float("inf"), float("-inf")) else None


def _parse(ts: Any) -> datetime | None:
    if not isinstance(ts, str) or not ts:
        return None
    try:
        d = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def _iso_time(v: Any) -> str | None:
    """ISO-8601 UTC from an ISO string or an epoch number (the approvals table stores epoch seconds)."""
    if isinstance(v, str) and _parse(v):
        return v
    n = _num(v)
    if n is not None and n > 0:
        return datetime.fromtimestamp(n, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return None


def record_time(rec: Rec) -> str | None:
    return rec.get("aws_time") or rec.get("time") or rec.get("created_at")


_ACCOUNT = re.compile(r"\b\d{12}\b")


def pseudonym(who: Any) -> str | None:
    """A display name for an approver. An STS/IAM ARN is cut to its last path segment (the role-session or
    user name) so an account id is never shown; any other 12-digit run is masked too."""
    if not isinstance(who, str) or not who.strip():
        return None
    s = who.strip()
    if s.startswith("arn:"):
        tail = s.split(":", 5)[-1]
        s = tail.rsplit("/", 1)[-1] if "/" in tail else tail
    return _ACCOUNT.sub("••••", s) or None


def case_prov(rec: Rec) -> Rec:
    """Synthetic / real flags plus the record's AWS time, its source and its runner clock (see awstime.resolve).
    `real` needs a time AWS recorded for the event (CloudWatch, a job's attestation line, policy createdAt, Gateway
    Date): the S3 stored time alone does not make a record real, and a Lambda clock is not AWS time at all."""
    synthetic = rec.get("synthetic") is True
    t = awstime.resolve(rec)
    return {"synthetic": synthetic, "real": (not synthetic) and t["aws_time_source"] in awstime.EVENT_SOURCES, **t,
            "no_event_reason": awstime.no_event_reason(rec)}


def _provenance(ps: list[Rec]) -> Rec:
    n = len(ps)
    n_syn = sum(1 for p in ps if p["synthetic"])
    n_unv = sum(1 for p in ps if not p["synthetic"] and not p["real"])
    if n == 0:
        state, label, note = "empty", "", ""
    elif n_syn == n:
        state, label, note = "synthetic", "SYNTHETIC", "Synthetic records (test fixtures): not a real run."
    elif n_syn == 0 and n_unv == 0:
        state, label, note = "real", "REAL", f"Every record shown has an AWS-recorded event time and is not marked synthetic."
    elif n_syn == 0:
        state, label = "unverified", "UNVERIFIED"
        note = f"{n_unv} of {n} records have no AWS-recorded event time, so they are not shown as real."
    else:
        state, label = "mixed", "MIXED"
        note = f"{n_syn} of {n} records synthetic" + (f"; {n_unv} more have no AWS-recorded event time." if n_unv else ".")
    return {"state": state, "label": label, "note": note, "n_records": n, "n_synthetic": n_syn, "n_unverified": n_unv}


def case_provenance(records: Iterable[Rec]) -> Rec:
    """REAL only when every record shown has a time AWS recorded for the event and none is synthetic.
    `aws_time_summary` is "AWS time on k of n (cloudwatch a, S3 stored b, ...)" over the same records."""
    ps = [case_prov(r) for r in records]
    out = _provenance(ps)
    out["aws_time_summary"] = awstime.summary(ps)
    out["unverified_reasons"] = awstime.reasons(ps)
    return out


# ---------------------------------------------------------------------------------------------- rung

def preset_of(change: Rec, declared: Rec | None) -> tuple[str | None, str]:
    """(preset name or None, where it came from). Order: the change record's `twin.preset` (the contract field),
    then a string `twin.config` on the change record (older writers), then the run's declaration
    (`declared.twin.config`: a preset name, a dict with "preset", or None meaning the official preset)."""
    tw = change.get("twin") if isinstance(change.get("twin"), dict) else {}
    if isinstance(tw.get("preset"), str) and tw["preset"]:
        return tw["preset"], "change.twin.preset"
    if isinstance(tw.get("config"), str) and tw["config"]:
        return tw["config"], "change.twin.config"
    dt = (declared or {}).get("twin")
    if isinstance(dt, dict) and "config" in dt:
        cfg = dt["config"]
        if cfg is None:
            return "official", "declared.twin.config"
        if isinstance(cfg, str) and cfg:
            return cfg, "declared.twin.config"
        if isinstance(cfg, dict) and isinstance(cfg.get("preset"), str) and cfg["preset"]:
            return cfg["preset"], "declared.twin.config"
        return None, "declared.twin.config"
    return None, "not recorded"


def rung(twin_kind: Any, preset: str | None, preset_source: str) -> Rec:
    if twin_kind == "viavi":
        return {"code": "R3", "label": "Viavi", "known": True, "twin_kind": twin_kind, "preset": preset,
                "basis": "twin kind viavi"}
    if twin_kind in MAVERIC_KINDS:
        if preset == REHEARSAL_PRESET:
            return {"code": "R2", "label": "Rehearsal", "known": True, "twin_kind": twin_kind, "preset": preset,
                    "basis": f"Maveric with the {REHEARSAL_PRESET} preset ({preset_source})"}
        if preset is not None:
            return {"code": "R1", "label": "Maveric", "known": True, "twin_kind": twin_kind, "preset": preset,
                    "basis": f"Maveric, preset {preset} ({preset_source})"}
        return {"code": None, "label": "Maveric, config unknown", "known": False, "twin_kind": twin_kind,
                "preset": None, "basis": "twin kind is Maveric but no record names its config, so R1 and R2 cannot be told apart"}
    return {"code": None, "label": "unknown twin", "known": False, "twin_kind": twin_kind, "preset": preset,
            "basis": f"twin kind {twin_kind!r} is not a ladder rung"}


# ---------------------------------------------------------------------------------------------- per-cell state

def snapshot(raw: Any) -> Rec | None:
    """A TwinSnapshot as the page draws it, or None when absent or unusable. Unknown per-cell values stay null."""
    if not isinstance(raw, dict) or not isinstance(raw.get("per_cell"), list) or not raw["per_cell"]:
        return None
    cells = []
    for i, c in enumerate(raw["per_cell"]):
        if not isinstance(c, dict):
            return None
        cell = c.get("cell") if isinstance(c.get("cell"), int) and not isinstance(c.get("cell"), bool) else i
        row: Rec = {"cell": cell, "name": CELL_NAMES[cell] if 0 <= cell < len(CELL_NAMES) else f"cell {cell}"}
        for k in CELL_KEYS:
            v = c.get(k)
            row[k] = bool(v) if k in ("alarm", "asleep") and isinstance(v, bool) else (_num(v) if k not in ("alarm", "asleep") else None)
        cells.append(row)
    cells.sort(key=lambda r: r["cell"])
    return {"step": raw.get("step") if isinstance(raw.get("step"), int) else None,
            "coverage_ratio": _num(raw.get("coverage_ratio")), "sla_ratio": _num(raw.get("sla_ratio")),
            "per_cell": cells}


def applied_deltas(change: list[Rec], before: Rec | None, after: Rec | None) -> list[Rec]:
    """Per touched cell: requested Δ (from the record) and, when both snapshots exist, the Δ the twin
    actually shows (power and tilt after - before). A difference means the safety filter clipped it."""
    out = []
    by_b = {c["cell"]: c for c in (before or {}).get("per_cell", [])}
    by_a = {c["cell"]: c for c in (after or {}).get("per_cell", [])}
    for c in change:
        if not isinstance(c, dict) or not isinstance(c.get("cell"), int):
            continue
        cell = c["cell"]
        req_p, req_t = _num(c.get("d_power_db")), _num(c.get("d_tilt_deg"))
        b, a = by_b.get(cell), by_a.get(cell)
        act_p = act_t = None
        if b and a:
            if b["power_dbm"] is not None and a["power_dbm"] is not None:
                act_p = round(a["power_dbm"] - b["power_dbm"], 3)
            if b["tilt_deg"] is not None and a["tilt_deg"] is not None:
                act_t = round(a["tilt_deg"] - b["tilt_deg"], 3)
        differs = any(x is not None and y is not None and abs(x - y) > 0.05 for x, y in ((act_p, req_p), (act_t, req_t)))
        out.append({"cell": cell, "name": CELL_NAMES[cell] if 0 <= cell < len(CELL_NAMES) else f"cell {cell}",
                    "requested": {"d_power_db": req_p, "d_tilt_deg": req_t},
                    "applied": None if act_p is None and act_t is None else {"d_power_db": act_p, "d_tilt_deg": act_t},
                    "differs": differs})
    return out


# ---------------------------------------------------------------------------------------------- approvals

def approval_for(change: Rec, decisions: list[Rec], items: list[Rec]) -> tuple[Rec, list[Rec]]:
    """Who approved a change and when, and the records that say so. Sources, best first:
      1. the change record's top-level `approved_by` / `approved_at` (contract fields, written by the tools Lambda),
      2. a suite `item` with extra.approval_id == change_id and extra.approval {decided_by, decided_at},
      3. an ALLOWed apply_change `decision` naming the approval id: gives the time Cedar let it through, not the person.
    """
    cid = change.get("change_id")
    if change.get("approved_by") or change.get("approved_at"):
        return {"status": "APPROVED", "by": pseudonym(change.get("approved_by")), "at": _iso_time(change.get("approved_at")),
                "reason": None, "source": "change.approved_by", "allowed_at": None}, []
    used: list[Rec] = []
    out: Rec = {"status": None, "by": None, "at": None, "reason": None, "source": None, "allowed_at": None}
    for it in items:
        ex = it.get("extra") if isinstance(it.get("extra"), dict) else {}
        ap = ex.get("approval") if isinstance(ex.get("approval"), dict) else None
        if cid and ex.get("approval_id") == cid and ap:
            out.update(status=ap.get("status"), by=pseudonym(ap.get("decided_by")), at=_iso_time(ap.get("decided_at")),
                       reason=ap.get("reason"), source="item.extra.approval")
            used.append(it)
            break
    allows = [d for d in decisions if cid and d.get("approval_id") == cid and d.get("decision") == "ALLOW"
              and str(d.get("tool") or "").split("___")[-1] == "apply_change"]
    if allows:
        first = min(allows, key=lambda d: _parse(record_time(d)) or datetime.max.replace(tzinfo=timezone.utc))
        out["allowed_at"] = record_time(first)
        out["by"] = out["by"] or pseudonym(first.get("approved_by"))  # the kit's checkpoint line carries it
        if out["source"] is None:
            out.update(status="APPROVED", source="decision")
        used.append(first)
    return out, used


# ---------------------------------------------------------------------------------------------- cases

def _group(records: Iterable[Rec]) -> dict[tuple[str, str], list[Rec]]:
    out: dict[tuple[str, str], list[Rec]] = {}
    for r in records:
        cid = r.get("case_id")
        if isinstance(cid, str) and cid and r.get("kind", "change") == "change" and r.get("record") != "change_applied":
            out.setdefault((str(r.get("run_id") or ""), cid), []).append(r)
    return out


def _step(rec: Rec) -> int:
    s = (rec.get("twin") or {}).get("step") if isinstance(rec.get("twin"), dict) else None
    return s if isinstance(s, int) and not isinstance(s, bool) else 10**9


def _sorted_changes(rows: list[Rec]) -> list[Rec]:
    far = datetime.max.replace(tzinfo=timezone.utc)
    return sorted(rows, key=lambda r: (_step(r), _parse(record_time(r)) or far))


def _kpis(v: Any) -> Rec | None:
    if not isinstance(v, dict):
        return None
    return {k: _num(v.get(k)) for k in KPI_KEYS}


def window(rec: Rec, decisions: list[Rec], items: list[Rec]) -> tuple[Rec, list[Rec]]:
    ts = rec.get("twin_state") if isinstance(rec.get("twin_state"), dict) else {}
    before, after, baseline = snapshot(ts.get("before")), snapshot(ts.get("after")), snapshot(ts.get("baseline"))
    change = [c for c in rec.get("change") or [] if isinstance(c, dict)]
    approval, used = approval_for(rec, decisions, items)
    tw = rec.get("twin") if isinstance(rec.get("twin"), dict) else {}
    w = {
        "change_id": rec.get("change_id"), "item_id": rec.get("item_id"), "step": tw.get("step"),
        "time": record_time(rec), "change_text": rec.get("change_text"),
        "cells": applied_deltas(change, before, after),
        "before": before, "after": after, "baseline": baseline, "has_state": before is not None and after is not None,
        "baseline_kpis": _kpis(rec.get("baseline_kpis")), "measured_kpis": _kpis(rec.get("measured_kpis")),
        "measured_delta": _kpis(rec.get("measured_delta")), "predicted_delta": _kpis(rec.get("predicted_delta")),
        "error": rec.get("error") if isinstance(rec.get("error"), dict) else None,
        "direction_agree": rec.get("direction_agree") if isinstance(rec.get("direction_agree"), bool) else None,
        "scored": rec.get("scored") if isinstance(rec.get("scored"), bool) else None,
        "unscored_reason": rec.get("unscored_reason"), "probe": rec.get("probe") is True,
        "predictor": rec.get("predictor"), "policy": rec.get("policy"),
        "trust_before": _num(rec.get("trust_before")), "trust_after": _num(rec.get("trust_after")),
        "approval": approval, **case_prov(rec),
    }
    return w, used


def summarize(run_id: str, case_id: str, rows: list[Rec], declared: Rec | None) -> Rec:
    rows = _sorted_changes(rows)
    first = rows[0]
    preset, src = preset_of(first, declared)
    kinds = sorted({str((r.get("twin") or {}).get("kind")) for r in rows})
    shown = rows + ([declared] if declared else [])
    times = sorted(t for t in (record_time(r) for r in rows) if t)
    return {
        "case_id": case_id, "run_id": run_id or None, "n_changes": len(rows),
        "n_with_state": sum(1 for r in rows if isinstance(r.get("twin_state"), dict) and snapshot(r["twin_state"].get("before"))),
        "rung": rung((first.get("twin") or {}).get("kind"), preset, src), "twin_kinds": kinds,
        "scenario": (first.get("twin") or {}).get("scenario"), "seed": (first.get("twin") or {}).get("seed"),
        "first_time": times[0] if times else None, "last_time": times[-1] if times else None,
        "provenance": case_provenance(shown),
    }


def build_overview(changes: list[Rec], declared: dict[str, Rec | None]) -> Rec:
    groups = _group(changes)
    rows = [summarize(k[0], k[1], v, declared.get(k[0])) for k, v in groups.items()]
    rows.sort(key=lambda r: r["last_time"] or "", reverse=True)
    shown = [r for v in groups.values() for r in v] + [d for rid, d in declared.items() if d and any(k[0] == rid for k in groups)]
    return {"cases": rows, "provenance": case_provenance(shown)}


def build_case(case_id: str, changes: list[Rec], decisions: list[Rec], items: list[Rec],
               declared: dict[str, Rec | None], run_id: str | None = None) -> Rec | None:
    groups = _group(changes)
    keys = [k for k in groups if k[1] == case_id and (run_id is None or k[0] == run_id)]
    if not keys:
        return None
    key = max(keys, key=lambda k: max((record_time(r) or "") for r in groups[k]))
    rows = _sorted_changes(groups[key])
    dec = [d for d in decisions if d.get("case_id") == case_id and (d.get("run_id") or "") == key[0]]
    its = [i for i in items if (i.get("run_id") or "") == key[0]]
    windows, used = [], []
    prev_after_step = None
    for r in rows:
        w, u = window(r, dec, its)
        b = (w["before"] or {}).get("step")
        w["gap_steps"] = (b - prev_after_step) if isinstance(b, int) and isinstance(prev_after_step, int) and b > prev_after_step else 0
        prev_after_step = (w["after"] or {}).get("step")
        windows.append(w)
        used += u
    decl = declared.get(key[0])
    head = summarize(key[0], case_id, rows, decl)
    seen, shown = set(), []
    for r in rows + used + ([decl] if decl else []):
        if id(r) not in seen:
            seen.add(id(r))
            shown.append(r)
    missing = []
    if any(not w["has_state"] for w in windows):
        missing.append({"field": "twin_state", "n_windows": sum(1 for w in windows if not w["has_state"]),
                        "effect": "per-cell power, tilt, users and KPIs are not drawn for these windows"})
    if any(w["has_state"] and any(c["coverage_share"] is None and (c["ue_count"] or 0) > 0 for c in w["after"]["per_cell"]) for w in windows):
        missing.append({"field": "twin_state.*.per_cell[].coverage_share", "effect": "cells are not coloured by coverage"})
    if any(w["approval"]["by"] is None for w in windows):
        missing.append({"field": "approved_by", "n_windows": sum(1 for w in windows if w["approval"]["by"] is None),
                        "effect": "the approver is shown as not recorded"})
    if head["rung"]["code"] is None:
        missing.append({"field": "twin.preset", "effect": head["rung"]["basis"]})
    return {
        **head, "windows": windows, "provenance": case_provenance(shown), "missing": missing, "field_spec": FIELD_SPEC,
        "cell_names": list(CELL_NAMES), "other_runs": [k[0] for k in keys if k != key],
        "show_record": f"python3 tools/show_record.py --run-id {shlex.quote(key[0])} --item {shlex.quote(case_id)}",
    }
