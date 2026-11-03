"""GET /api/trustgate/{models,changes,ladder}. Read-only; everything comes from the evidence records."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from . import logic

router = APIRouter(prefix="/api/trustgate")

SAFE_ID = r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$"


def get_reader() -> Any:
    """The shared records reader. Imported lazily so this slice loads (and is testable) without it."""
    from spectrum.slices._shared.records import RecordsReader

    return RecordsReader()


def _call(fn, *args, default):
    try:
        out = fn(*args)
    except Exception:  # an absent or unreadable record is "not there", never a made-up value
        return default
    return default if out is None else out


def _load(reader: Any) -> tuple[dict, str | None]:
    """All the records this view needs. A failed read yields empty lists plus an error string."""
    data: dict[str, Any] = {"change": [], "control": [], "result": []}
    errors: list[str] = []
    for kind in ("change", "control"):
        try:
            data[kind] = [r for r in (reader.all_records(kind) or []) if isinstance(r, dict)]
        except Exception as e:  # noqa: BLE001
            errors.append(f"{kind}: {type(e).__name__}")
    try:
        for run in reader.runs() or []:
            rid = run.get("run_id") if isinstance(run, dict) else run
            if not isinstance(rid, str):
                continue
            res = _call(reader.result, rid, default=None)
            if isinstance(res, dict):
                data["result"].append(res if res.get("run_id") else {**res, "run_id": rid})
    except Exception as e:  # noqa: BLE001
        errors.append(f"result: {type(e).__name__}")
    return data, ("; ".join(errors) or None)


def _source(reader: Any) -> Any:
    return _call(reader.source, default=None)


def _detail(reader: Any, data: dict, key: tuple[str, str]) -> dict:
    model_rec = _call(reader.model, key[0], key[1], default=None)
    return logic.model_detail(key, data["change"], data["control"], model_rec if isinstance(model_rec, dict) else None, data["result"])


def _verification(data: dict) -> dict:
    """check_unit governs "verified" for the runs behind this screen; synthetic describes the inputs only."""
    from spectrum.slices._shared.verified import govern

    runs = sorted({r.get("run_id") for k in ("change", "control") for r in data[k] if isinstance(r.get("run_id"), str)})
    recs = data["change"] + data["control"]
    n_syn = sum(1 for r in recs if r.get("synthetic") is True)
    base = {"state": "unverified", "label": "NOT YET CHECKED", "note": "no cached check_unit output for these runs",
            "n_records": len(recs), "n_synthetic": n_syn, "n_unverified": 0}
    g = govern(base, runs)
    return {**g, "runs": runs, "n_synthetic_inputs": n_syn}


@router.get("/models")
def models(reader: Any = Depends(get_reader)) -> dict:
    data, error = _load(reader)
    out = [logic.models_summary(_detail(reader, data, k)) for k in logic.model_keys(data["change"], data["control"])]
    d = logic.default_model(out)
    return {"source": _source(reader), "error": error, "models": out, "probes": logic.probe_rows(data["change"]),
            "default_key": [d["model_id"], d["version"]] if d else None,
            "verification": _verification(data)}


@router.get("/changes")
def changes(
    model_id: str = Query(pattern=SAFE_ID),
    version: str = Query(pattern=SAFE_ID),
    reader: Any = Depends(get_reader),
) -> dict:
    data, error = _load(reader)
    key = (model_id, version)
    if key not in logic.model_keys(data["change"], data["control"]):
        raise HTTPException(status_code=404, detail=f"no change or control records for {model_id} {version}")
    return {"source": _source(reader), "error": error, **_detail(reader, data, key)}


@router.get("/ladder")
def ladder(reader: Any = Depends(get_reader)) -> dict:
    data, error = _load(reader)
    return {"source": _source(reader), "error": error, "ladders": logic.latest_ladders(data["result"])}
