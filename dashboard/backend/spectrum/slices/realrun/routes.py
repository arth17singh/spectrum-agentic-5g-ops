"""GET /api/realrun/overview and /api/realrun/run/{run_id}. Fresh reads of the local evidence mirror."""

from __future__ import annotations

import re

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from . import extra, logic

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
        sup = extra.superseded(st)
        summaries = [logic.run_summary(st, r) for r in runs]
        for x in summaries:
            x["superseded"] = sup.get(x["run_id"])
        return {
            "source": st.source(),
            "runs": summaries,
            "t1": logic.track1(st, runs, sup),
            "rl": logic.rl(st, runs),
            "t3": logic.track3(st, runs, sup),
            "t3_live": extra.track3_live(st, runs),
            "t2": logic.track2(st, runs),
            "t2_consortium": extra.track2_consortium(st, runs),
            "judged": extra.judged(st),
            "viavi_operate": extra.viavi_operate(st, runs),
            "curves": {"t2_convergence": extra.latest_unit(st, runs, "t2.fl_convergence"),
                       "t1": extra.latest_unit(st, runs, "t1.curves")},
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
        out = logic.run_detail(st, run_id)
        out["superseded"] = extra.superseded(st).get(run_id)
        return out
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"detail": f"evidence unavailable: {type(e).__name__}: {str(e)[:200]}"}, status_code=503)
