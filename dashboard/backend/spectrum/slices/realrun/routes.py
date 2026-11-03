"""GET /api/realrun/overview and /api/realrun/run/{run_id}. Fresh reads of the local evidence mirror."""

from __future__ import annotations

import re

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from . import logic

router = APIRouter(prefix="/api/realrun")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


def _store() -> logic.Store | None:
    root = logic.root_from_env()
    return logic.Store(root) if root is not None else None


def _no_store() -> JSONResponse:
    return JSONResponse({"detail": "the real-run views read a local evidence mirror; SPECTRUM_RECORDS_ROOT is an s3:// root"},
                        status_code=503)


@router.get("/overview")
def overview():
    st = _store()
    if st is None:
        return _no_store()
    try:
        runs = st.run_ids()
        return {
            "source": st.source(),
            "runs": [logic.run_summary(st, r) for r in runs],
            "t1": logic.track1(st, runs),
            "rl": logic.rl(st, runs),
            "t3": logic.track3(st, runs),
            "t2": logic.track2(st, runs),
            "empty": not runs,
        }
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"detail": f"evidence unavailable: {type(e).__name__}: {str(e)[:200]}"}, status_code=503)


@router.get("/run/{run_id}")
def run(run_id: str):
    if not RUN_ID_RE.match(run_id) or ".." in run_id:
        return JSONResponse({"detail": "invalid run id"}, status_code=400)
    st = _store()
    if st is None:
        return _no_store()
    if run_id not in st.run_ids():
        return JSONResponse({"detail": f"no such run: {run_id}"}, status_code=404)
    try:
        return logic.run_detail(st, run_id)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"detail": f"evidence unavailable: {type(e).__name__}: {str(e)[:200]}"}, status_code=503)
