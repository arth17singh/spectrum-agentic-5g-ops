"""GET /api/twinview/cases, /api/twinview/case/{case_id}, /api/twinview/consortium. Fresh read on every call."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from . import consortium, logic

router = APIRouter(prefix="/api/twinview")

SILOS_DIR = Path(__file__).resolve().parents[5] / "silos"


def _reader():
    """Lazy so this package imports without the shared reader; tests monkeypatch this function."""
    from spectrum.slices._shared.records import RecordsReader

    return RecordsReader()


def _silos_dir() -> Path:
    return Path(os.environ.get("SPECTRUM_SILOS_DIR") or SILOS_DIR)


def _source(r) -> Any:
    try:
        return r.source()
    except Exception:  # the source label is context, never a reason to fail
        return None


def _unavailable(e: Exception) -> JSONResponse:
    return JSONResponse({"detail": f"records unavailable: {type(e).__name__}: {e}"}, status_code=503)


def _declared(r, run_ids) -> dict:
    return {rid: r.declared(rid) for rid in sorted({x for x in run_ids if isinstance(x, str) and x})}


@router.get("/cases")
def cases():
    try:
        r = _reader()
        changes = list(r.all_records("change"))
        decl = _declared(r, [c.get("run_id") for c in changes])
    except Exception as e:  # noqa: BLE001 - shown as an error, never as "no records"
        return _unavailable(e)
    ov = logic.build_overview(changes, decl)
    return {"cases": ov["cases"], "n_cases": len(ov["cases"]), "n_changes": len(changes),
            "provenance": ov["provenance"], "field_spec": logic.FIELD_SPEC, "source": _source(r)}


@router.get("/case/{case_id}")
def case(case_id: str, run_id: str | None = None):
    try:
        r = _reader()
        changes = list(r.all_records("change"))
        decisions = list(r.all_records("decision"))
        items = list(r.all_records("item"))
        decl = _declared(r, [c.get("run_id") for c in changes if c.get("case_id") == case_id])
    except Exception as e:  # noqa: BLE001
        return _unavailable(e)
    out = logic.build_case(case_id, changes, decisions, items, decl, run_id)
    if out is None:
        return JSONResponse({"detail": f"unknown case: {case_id}"}, status_code=404)
    return {**out, "source": _source(r)}


def _read_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


@router.get("/consortium")
def consortium_view(split: str = "disjoint"):
    if split not in ("disjoint", "overlap"):
        return JSONResponse({"detail": "split must be disjoint or overlap"}, status_code=400)
    try:
        d = _silos_dir()
        manifest = _read_json(d / "manifest.json")
        preview = _read_json(d / "baselines_results.json")
        r = _reader()
        results = [x for x in (r.result(rid) for rid in r.runs()) if isinstance(x, dict)]
    except Exception as e:  # noqa: BLE001
        return _unavailable(e)
    return {**consortium.build(manifest, preview, results, split), "source": _source(r)}
