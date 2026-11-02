"""Trace slice: rebuilds each Track 3 case from the persisted `decision`, `control` and `change` records and
checks it against the declared expected sequence. Read-only; no AWS calls of its own.

Mounted by discovery as `/api/trace/*`.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from . import logic

router = APIRouter(prefix="/api/trace")


def _reader():
    # imported lazily: the shared reader belongs to another stream and reads its data root at call time
    from spectrum.slices._shared.records import RecordsReader

    return RecordsReader()


def _load() -> tuple[list[dict], list[dict], list[dict], Any]:
    r = _reader()
    try:
        src = r.source()
    except Exception:  # the source description is a courtesy, never a reason to fail
        src = None
    return list(r.all_records("decision")), list(r.all_records("change")), list(r.all_records("control")), src


def _unavailable(e: Exception) -> JSONResponse:
    return JSONResponse({"detail": f"records unavailable: {type(e).__name__}: {e}"}, status_code=503)


@router.get("/cases")
def cases():
    try:
        decisions, changes, controls, src = _load()
    except Exception as e:  # noqa: BLE001 - reported to the page, which shows it as an error, not as "no records"
        return _unavailable(e)
    rows = logic.build_cases(decisions, changes, controls)
    return {"cases": rows, "n_cases": len(rows), "n_records": len(decisions) + len(changes) + len(controls), "source": src}


@router.get("/case/{case_id}")
def case(case_id: str, run_id: str | None = None):
    try:
        decisions, changes, controls, src = _load()
    except Exception as e:  # noqa: BLE001
        return _unavailable(e)
    out = logic.build_case(case_id, decisions, changes, controls, run_id)
    if out is None:
        return JSONResponse({"detail": f"unknown case: {case_id}"}, status_code=404)
    return {**out, "source": src}
