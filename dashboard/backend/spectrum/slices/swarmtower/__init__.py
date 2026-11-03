"""Swarm Tower slice: the live Track 3 swarm run and the Track 2 training silos, read from the evidence records.

GET /api/swarmtower. Read-only: no AWS calls, no ledger writes. See logic.py for what each field means.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from spectrum.slices._shared import awstime

from . import logic

router = APIRouter(prefix="/api/swarmtower")

LIVE_PREFIX = "t3.swarm-live-"


def _reader():
    from spectrum.slices._shared.records import RecordsReader

    return RecordsReader()


def _unavailable(e: Exception) -> JSONResponse:
    return JSONResponse({"detail": f"records unavailable: {type(e).__name__}: {e}"}, status_code=503)


def _items_with_times(r: Any, run_id: str) -> list[dict]:
    """The run's item records, each with its AWS time resolved by the shared rule (awstime.resolve)."""
    entries = r._entries(run_id, "item")
    times = r._file_times(f"records/{run_id}/item/", [k for k, _ in entries]) if hasattr(r, "_file_times") else {}
    out = []
    for key, rec in entries:
        if not isinstance(rec, dict):
            continue
        t = awstime.resolve(awstime.with_times(rec, times.get(key)))
        out.append({**rec, "aws_time_resolved": t["aws_time"], "aws_time_source_resolved": t["aws_time_source"],
                    "runner_time": t["runner_time"]})
    return out


def _actions(r: Any, run_id: str) -> list[dict]:
    text = r._read_text(f"outputs/{run_id}/swarm-actions.jsonl")
    out = []
    for line in (text or "").splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if isinstance(d, dict):
            for k in ("start_action_log",):
                if isinstance(d.get(k), str):
                    try:
                        d[k] = json.loads(d[k])
                    except ValueError:
                        pass
            out.append(d)
    return out


def _scoring_run(r: Any, live_run: str) -> tuple[str | None, dict | None]:
    """The newest scoring run of the swarm's proposals (details.operator == swarm), with its result."""
    best = None
    for rid in r.runs():
        if not rid.startswith("t3.evals.swarm"):
            continue
        res = r.result(rid)
        det = (res or {}).get("details") or {}
        if isinstance(res, dict) and det.get("operator") == "swarm":
            if best is None or (res.get("checked_at") or "") > (best[1].get("checked_at") or ""):
                best = (rid, res)
    return best if best else (None, None)


def _silos() -> dict | None:
    from spectrum.slices.twinview import consortium
    from spectrum.slices.twinview.routes import _read_json, _silos_dir

    m = _read_json(_silos_dir() / "manifest.json")
    return consortium.silo_facts(m, "disjoint") if isinstance(m, dict) else None


@router.get("")
def swarmtower():
    try:
        from spectrum.slices._shared.verified import govern, superseded_ids

        r = _reader()
        sup = superseded_ids()
        live = sorted(x for x in r.runs() if x.startswith(LIVE_PREFIX) and x not in sup)
        silos = _silos()
        if not live:
            return {"run": None, "incidents": [], "silos": silos}
        run_id = live[-1]
        items = _items_with_times(r, run_id)
        actions = _actions(r, run_id)
        score_run, score_res = _scoring_run(r, run_id)
        scores = {}
        if score_run:
            for it in r.records(score_run, "item"):
                ex = it.get("extra") if isinstance(it.get("extra"), dict) else {}
                scores[it.get("item_id")] = {"success": ex.get("success"), "recovery": ex.get("recovery"),
                                              "incident": ex.get("incident")}
        incidents = logic.group_incidents(items, actions, scores)
        trace_cases = {d.get("case_id") for d in r.all_records("decision") if isinstance(d, dict)}
        for inc in incidents:
            for s in inc["steps"]:
                s["in_trace"] = s["case_id"] in trace_cases
        rows = [{"aws_time_source": s["aws_time_source"]} for i in incidents for s in i["steps"]]
        summ = awstime.summary(rows)
        n_syn = sum(1 for i in incidents for s in i["steps"] if s["synthetic"])
        prov = {"state": "real" if summ["k"] == summ["n"] and not n_syn else "unverified",
                "label": "REAL" if summ["k"] == summ["n"] and not n_syn else "UNVERIFIED",
                "note": summ["text"], "n_records": summ["n"], "n_unverified": summ["n"] - summ["k"], "n_synthetic": n_syn}
        prov = govern(prov, [run_id])
        decl = r.declared(run_id) or {}
        res = r.result(run_id) or {}
        sdet = (score_res or {}).get("details") or {}
        g = sdet.get("gain") if isinstance(sdet.get("gain"), dict) else {}
        return {
            "run": {"run_id": run_id, "unit": decl.get("unit"), "model_id": decl.get("model_id"),
                    "models": logic.parse_models(decl.get("model_id")),
                    "approval_policy": ((decl.get("twin") or {}).get("config") or {}).get("approval_policy"),
                    "pass_rate": res.get("value"), "n_items": res.get("n_items"), "checked_at": res.get("checked_at")},
            "scoring": None if not score_run else {
                "run_id": score_run, "metric": score_res.get("metric"), "value": score_res.get("value"),
                "passed": score_res.get("passed"), "success_count": sdet.get("success_count"), "n_items": sdet.get("n_items"),
                "mean_recovery": sdet.get("mean_recovery"),
                "gain": {"value": g.get("value"), "ci_low": g.get("ci_low"), "ci_high": g.get("ci_high"),
                         "baseline_model_id": g.get("baseline_model_id"), "baseline_run_id": g.get("baseline_run_id")}},
            "provenance": prov, "aws_time_summary": summ, "incidents": incidents, "silos": silos,
        }
    except Exception as e:  # noqa: BLE001
        return _unavailable(e)
