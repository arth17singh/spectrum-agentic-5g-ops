"""GET /api/scorecard/units and /api/scorecard/run/{run_id}. Read-only; fresh read on every call."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from .logic import RUN_ID_RE, build_row, build_scorecard, show_record_command

router = APIRouter(prefix="/api/scorecard")


def _reader():
    """Lazy so this package imports without the shared reader; tests monkeypatch this function."""
    from spectrum.slices._shared.records import RecordsReader

    return RecordsReader()


def _run_ids(reader) -> list[str]:
    out: list[str] = []
    for r in reader.runs() or []:
        rid = r.get("run_id") if isinstance(r, dict) else r
        if isinstance(rid, str) and RUN_ID_RE.match(rid) and rid not in out:
            out.append(rid)
    return out


def _source(reader) -> Any:
    try:
        return reader.source()
    except Exception:  # the source label is context, never a reason to fail the scorecard
        return None


def _unavailable(e: Exception) -> JSONResponse:
    return JSONResponse({"detail": f"evidence unavailable: {type(e).__name__}"}, status_code=503)


GOVERNED = ("check failed", "unverified", "incomplete")


def _counts(rows: list[dict]) -> dict:
    """Each bucket named by what it is. Superseded runs are counted only as superseded, never as declared."""
    live = [r for r in rows if r["state"] != "superseded"]
    out = {
        "declared": len(live),
        "passed": sum(r["state"] == "pass" for r in live),
        "failed": sum(r["state"] == "fail" for r in live),
        "check_failed": sum(r["state"] == "check failed" for r in live),
        "unverified": sum(r["state"] == "unverified" for r in live),
        "incomplete": sum(r["state"] == "incomplete" for r in live),
        "not_checked": sum(r["state"] == "not checked" for r in live),
        "inconsistent": sum(r["state"] == "inconsistent" for r in live),
        "superseded": len(rows) - len(live),
    }
    # the evidence buckets appear only when used, so the base shape is unchanged
    return {k: v for k, v in out.items() if v or k not in ("check_failed", "unverified", "incomplete", "superseded")}


def _govern(card: dict, pairs: list) -> None:
    """The evidence checks govern the row state (cached check_unit in the mirror's .sync/checks):
      superseded     listed in superseded.json: shown apart, counted nowhere else
      incomplete     the result says complete: false
      check failed   a check_unit part FAILED (the parts are named)
      unverified     check_unit part c did not pass (no AWS window, so record times are not verified)
    "inconsistent" stays reserved for a record that contradicts itself (the scorecard's own checks)."""
    from spectrum.slices.realrun.extra import superseded
    from spectrum.slices.realrun.logic import Store, failed_parts, part_c_not_pass, root_from_env, verdict

    root = root_from_env()
    st = Store(root) if root is not None else None
    if st is None:
        return
    sup = superseded(st)
    by = {rid: (d, r) for rid, d, r in pairs}
    for t in card.get("tracks", []):
        for row in t["units"]:
            rid = row["run_id"]
            if rid in sup:
                row["state"] = "superseded"
                row["superseded"] = sup[rid]
                continue
            d, r = by.get(rid, (None, None))
            if not isinstance(r, dict) or row["state"] not in ("pass", "fail"):
                continue
            v = verdict(st, r, d if isinstance(d, dict) else None)
            if v in GOVERNED:
                row["state"] = v
                if v == "check failed":
                    row["evidence_reason"] = "check_unit " + "; ".join(failed_parts(st, rid))
                elif v == "unverified":
                    row["evidence_reason"] = f"check_unit {part_c_not_pass(st, rid)}"
                else:
                    row["evidence_reason"] = "the result says complete: false"
        t.update(_counts(t["units"]))
    card["totals"] = _counts([r for t in card.get("tracks", []) for r in t["units"]])


@router.get("/units")
def units():
    try:
        reader = _reader()
        pairs = [(rid, reader.declared(rid), reader.result(rid)) for rid in _run_ids(reader)]
        card = build_scorecard(pairs)
        _govern(card, pairs)
        card["source"] = _source(reader)
    except Exception as e:
        return _unavailable(e)
    return card


@router.get("/run/{run_id}")
def run(run_id: str):
    if not RUN_ID_RE.match(run_id):
        return JSONResponse({"detail": "invalid run id"}, status_code=400)
    try:
        reader = _reader()
        if run_id not in _run_ids(reader):
            return JSONResponse({"detail": f"no such run: {run_id}"}, status_code=404)
        declared, result = reader.declared(run_id), reader.result(run_id)
        item_records = None
        if isinstance(result, dict):
            try:
                item_records = len(list(reader.records(run_id, "item")))
            except Exception:
                item_records = None
    except Exception as e:
        return _unavailable(e)
    out: dict[str, Any] = {
        "run_id": run_id,
        "row": build_row(run_id, declared, result if isinstance(result, dict) else None) if isinstance(declared, dict) else None,
        "declared": declared,
        "result": result,
        "item_records": item_records,
        "command": show_record_command(run_id),
        "source": _source(reader),
    }
    if out["row"] is None:
        out["note"] = "no declaration for this run, so it is not a scored unit"
    return out
