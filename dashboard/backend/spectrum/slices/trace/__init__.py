"""Trace slice: rebuilds each Track 3 case from the persisted `decision`, `control` and `change` records and
checks it against the declared expected sequence. Read-only; no AWS calls of its own.

Mounted by discovery as `/api/trace/*`.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from spectrum.slices._shared import awstime

from . import logic

router = APIRouter(prefix="/api/trace")


def _reader():
    # imported lazily: the shared reader belongs to another stream and reads its data root at call time
    from spectrum.slices._shared.records import RecordsReader

    return RecordsReader()


def _records(r, kind: str) -> list[dict]:
    """The records of a kind, each carrying the AWS times of its file (see awstime.read_records)."""
    return awstime.read_records(r, kind)


def _pause_items(r) -> dict[str, list[dict]]:
    """The per-call `item` records of the t3.pause runs: that run records no decisions or case ids, so these (with its
    pause/resume controls) are its evidence. Read per run, never the whole item store (it holds 100k+ files)."""
    try:
        runs = [x for x in r.runs() if logic.pause_run(x)]
    except Exception:  # noqa: BLE001 - a reader without runs() (a test double) has none
        return {}
    return {x: list(r.records(x, "item")) for x in runs}


def _load() -> tuple[list[dict], list[dict], list[dict], Any, dict[str, list[dict]]]:
    r = _reader()
    try:
        src = r.source()
    except Exception:  # the source description is a courtesy, never a reason to fail
        src = None
    return _records(r, "decision"), _records(r, "change"), _records(r, "control"), src, _pause_items(r)


def _govern_listing(out: dict) -> dict:
    """check_unit governs "verified": each case's provenance by its run, the listing's by all runs shown."""
    from spectrum.slices._shared.verified import govern

    for c in out.get("cases") or []:
        if isinstance(c, dict):
            c["provenance"] = govern(c.get("provenance"), [c.get("run_id")])
    out["provenance"] = govern(out.get("provenance"), [c.get("run_id") for c in out.get("cases") or [] if isinstance(c, dict)])
    return out


def _unavailable(e: Exception) -> JSONResponse:
    return JSONResponse({"detail": f"records unavailable: {type(e).__name__}: {e}"}, status_code=503)


@router.get("/cases")
def cases(show_superseded: bool = False):
    """Superseded runs (the list the evidence code keeps) are left out unless `show_superseded`; the counts and the
    banner are computed on the runs actually shown, and the response says how many it is hiding."""
    from spectrum.slices._shared.verified import hide_superseded

    try:
        decisions, changes, controls, src, pause = _load()
    except Exception as e:  # noqa: BLE001 - reported to the page, which shows it as an error, not as "no records"
        return _unavailable(e)
    (decisions, changes, controls), sup = hide_superseded([decisions, changes, controls], show_superseded)
    ov = logic.build_overview(decisions, changes, controls, pause)
    rows = ov["cases"]
    return _govern_listing({"cases": rows, "n_cases": len(rows), "n_records": len(decisions) + len(changes) + len(controls) + sum(len(v) for v in pause.values()),
                            "provenance": ov["provenance"], "source": src, "superseded_runs": sup, "n_superseded": len(sup),
                            "showing_superseded": bool(show_superseded)})


@router.get("/case/{case_id}")
def case(case_id: str, run_id: str | None = None):
    try:
        decisions, changes, controls, src, pause = _load()
    except Exception as e:  # noqa: BLE001
        return _unavailable(e)
    out = logic.build_case(case_id, decisions, changes, controls, run_id, pause)
    if out is None:
        return JSONResponse({"detail": f"unknown case: {case_id}"}, status_code=404)
    from spectrum.slices._shared.verified import govern

    out["provenance"] = govern(out.get("provenance"), [out.get("run_id")])
    return {**out, "source": src}
