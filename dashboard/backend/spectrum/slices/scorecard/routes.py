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


@router.get("/units")
def units():
    try:
        reader = _reader()
        pairs = [(rid, reader.declared(rid), reader.result(rid)) for rid in _run_ids(reader)]
        card = build_scorecard(pairs)
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
