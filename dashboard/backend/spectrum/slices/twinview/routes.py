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
    return _govern_listing({"cases": ov["cases"], "n_cases": len(ov["cases"]), "n_changes": len(changes),
                            "provenance": ov["provenance"], "field_spec": logic.FIELD_SPEC, "source": _source(r)})


@router.get("/case/{case_id}")
def case(case_id: str, run_id: str | None = None):
    try:
        r = _reader()
        changes = list(r.all_records("change"))
        decisions = list(r.all_records("decision"))
        case_runs = sorted({c.get("run_id") for c in changes if c.get("case_id") == case_id and isinstance(c.get("run_id"), str)})
        items = [x for rid in case_runs for x in r.records(rid, "item")]
        decl = _declared(r, case_runs)
    except Exception as e:  # noqa: BLE001
        return _unavailable(e)
    out = logic.build_case(case_id, changes, decisions, items, decl, run_id)
    if out is None:
        return JSONResponse({"detail": f"unknown case: {case_id}"}, status_code=404)
    from spectrum.slices._shared.verified import govern

    out["provenance"] = govern(out.get("provenance"), [out.get("run_id")])
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


# ---------------------------------------------------------------------------- Viavi sites (context only)

VIAVI_GLOB = "outputs/viavi-sites/*/viavi_sites.json"


def _num(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v else None


def viavi_sites_view(doc: Any) -> dict | None:
    """The extract as the panel draws it; only the documented fields are kept. A site with no usable
    coordinate keeps its name and cells but no lat/lon. None if the document is not the expected shape."""
    if not isinstance(doc, dict) or not isinstance(doc.get("scenarios"), list):
        return None
    out = []
    for sc in doc["scenarios"]:
        if not isinstance(sc, dict) or not isinstance(sc.get("name"), str):
            continue
        sites = []
        for s in sc.get("sites") or []:
            if not isinstance(s, dict):
                continue
            lat, lon = _num(s.get("lat")), _num(s.get("lon"))
            ok = lat is not None and lon is not None and -90 <= lat <= 90 and -180 <= lon <= 180
            sites.append({"name": str(s.get("name") or "?"), "cells": _num(s.get("cells")),
                          "lat": lat if ok else None, "lon": lon if ok else None})
        mc = sc.get("map_center") if isinstance(sc.get("map_center"), dict) else {}
        clat, clon = _num(mc.get("lat")), _num(mc.get("lon"))
        centre = {"lat": clat, "lon": clon} if clat is not None and clon is not None and -90 <= clat <= 90 and -180 <= clon <= 180 else None
        out.append({"name": sc["name"], "cells": _num(sc.get("cells")), "ues": _num(sc.get("ues")),
                    "note": sc.get("note") if isinstance(sc.get("note"), str) else None, "sites": sites,
                    "map_center": centre,
                    "coordinate_source": sc.get("coordinate_source") if isinstance(sc.get("coordinate_source"), str) else None})
    return {"source": doc.get("source") if isinstance(doc.get("source"), str) else None,
            "read_at": doc.get("read_at") if isinstance(doc.get("read_at"), str) else None, "scenarios": out}


@router.get("/viavi-sites")
def viavi_sites():
    """The newest Viavi RSG site extract mirrored from S3 (outputs/viavi-sites/<UTC>/viavi_sites.json).
    Organiser data shown for context only; it is never committed. {"loaded": false} until it exists."""
    try:
        r = _reader()
        root = getattr(r, "dir", None)
        if root is None:
            return {"loaded": False, "reason": "the Viavi extract is read from a local evidence mirror"}
        files = sorted(Path(root).glob(VIAVI_GLOB))
        if not files:
            return {"loaded": False}
        newest = files[-1]
        view = viavi_sites_view(json.loads(newest.read_text(encoding="utf-8")))
    except Exception as e:  # noqa: BLE001
        return _unavailable(e)
    if view is None:
        return {"loaded": False, "reason": f"{newest.parent.name}/viavi_sites.json is not the expected shape"}
    return {"loaded": True, "extract": newest.parent.name, **view}
