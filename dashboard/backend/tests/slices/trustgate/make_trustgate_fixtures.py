"""Generate the synthetic trust-gate and ladder fixtures (every record has synthetic: true).

Run:  python make_trustgate_fixtures.py     (rewrites ./fixtures/, deterministic)

Layout is the evidence layout: declared/<run>.json, records/<run>/<kind>.jsonl, results/<run>.json,
models/<model_id>/<version>.json. Fields follow BUILD_CONTRACT.md as amended by amendment 1 (deltas,
scored, probe). Scenarios:
  healthy   t2-response-mlp v1           10 scored changes + 1 clipped (unscored); never paused
  degraded  t2-response-mlp v1-degraded  paused by trust-gate after change 4 (2 of the last 3 over tau,
                                         while the EWMA trust is still above the pause line);
                                         a person resumes it; one more change follows
  probe     two no-model probe changes (no predictor)
  ladders   t1-ppo-maveric v3 -> REHEARSAL (no Viavi yet);  t1-ppo-maveric v4 -> VIAVI
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "fixtures"

SCALES = {"coverage_ratio": 0.08, "sla_ratio": 0.06, "sinr_p10_db": 1.0}
THRESHOLDS = {"tau": 1.0, "scales": SCALES, "pause_below": 0.55, "watch_below": 0.65, "alpha": 0.2, "n_min": 3}
CODE_SHA = "0123456789abcdef0123456789abcdef01234567"
TEST_SHA = "4788224c17388846cdac203276c985bc2baa97f5289c535c191ef9b49aea416f"
ART_SHA = "af3d3a8289f138597ccca5ebe56d13727d793c85cc9a32196636bbbe6d3596d0"
BASE = datetime(2026, 11, 2, 20, 10, 0, tzinfo=timezone.utc)
TWIN = {"kind": "maveric-inprocess", "version": "ran-rl-1.2.9", "scenario": "interference", "seed": 11}


def iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def env(kind: str, run_id: str, when: datetime) -> dict:
    return {"schema_version": "1", "kind": kind, "run_id": run_id, "team": "team13", "created_at": iso(when),
            "aws_time": None, "aws_time_source": None, "synthetic": True}


def canon(cells: list[dict]) -> str:
    return "|".join(
        f"cell={c['cell']};d_power_db={c['d_power_db']:+.1f};d_tilt_deg={c['d_tilt_deg']:+.1f}"
        for c in sorted(cells, key=lambda c: c["cell"]))


def change_id(case_id: str, item_id: str, text: str) -> str:
    return "chg-" + hashlib.sha256(json.dumps([case_id, item_id, text]).encode()).hexdigest()[:12]


MEASURED = {"coverage_ratio": 0.10, "sla_ratio": 0.08, "sinr_p10_db": 1.5, "sinr_mean_db": 1.2}
ALL_KPIS = tuple(MEASURED)


def make_change(run_id: str, n: int, when: datetime, *, predictor: dict | None, e: float | None, trust: tuple[float, float] | None,
                scored: bool = True, reason: str | None = None, probe: bool = False, step: int | None = None) -> dict:
    """One change record. For a model change the per-KPI error is e * scale, so the summary is exactly e."""
    cells = [{"cell": 1 + n % 3, "d_power_db": -1.0 - (n % 2), "d_tilt_deg": 1.0}, {"cell": 5, "d_power_db": -0.5, "d_tilt_deg": 0.5}]
    text = canon(cells)
    case_id, item_id = f"case-{hashlib.sha256(run_id.encode()).hexdigest()[:8]}-{n:02d}", f"step-{n}"
    rec = env("change", run_id, when)
    base = {"coverage_ratio": 0.80, "sla_ratio": 0.75, "sinr_p10_db": 2.0, "sinr_mean_db": 11.5}
    meas_d = {**{k: round(v + 0.002 * (n % 3), 6) for k, v in MEASURED.items()}}
    measured_kpis = {k: round(base[k] + meas_d[k], 6) for k in meas_d}
    rec.update({
        "case_id": case_id, "change_id": change_id(case_id, item_id, text), "item_id": item_id,
        "change": cells, "change_text": text, "twin": {**TWIN, "step": step if step is not None else n},
        "baseline_kpis": base, "measured_kpis": measured_kpis, "measured_delta": meas_d,
        "scored": scored, "probe": probe,
    })
    if probe:
        reason = "probe"
    if reason:
        rec["unscored_reason"] = reason
    if predictor is not None and e is not None:
        # predicted = measured - e * scale  (so each KPI's error is e * scale and the mean over KPIs is e)
        pred = {k: round(meas_d[k] - e * SCALES[k], 6) for k in SCALES}
        pred["sinr_mean_db"] = pred["sinr_p10_db"]  # not scored by the gate; mirrors p10 in the fixture
        err = {k: round(abs(pred[k] - meas_d[k]), 6) for k in ALL_KPIS}
        err["sinr_mean_db"] = round(abs(pred["sinr_mean_db"] - meas_d["sinr_mean_db"]), 6)
        err["summary"] = round(sum(err[k] / SCALES[k] for k in SCALES) / len(SCALES), 6)
        same = sum(1 for k in SCALES if pred[k] * meas_d[k] > 0)
        rec.update({"predictor": predictor, "predicted_delta": pred, "error": err, "direction_agree": same >= 2})
    if trust is not None:
        rec.update({"trust_before": trust[0], "trust_after": trust[1]})
    return rec


def model_rec(run_id: str, model_id: str, version: str, status: str, kind: str, trained_on: str, metrics: dict) -> dict:
    r = env("model", run_id, BASE - timedelta(hours=1))
    r.update({"model_id": model_id, "version": version, "model_kind": kind, "artifact_uri": f"file://t2/weights/{model_id}-{version}.npz",
              "artifact_sha256": ART_SHA, "status": status, "trained_on": trained_on, "metrics": metrics})
    return r


def declared(run_id: str, unit: str, track: str, metric: str, op: str, value: float, tunit: str, model_id: str, n_items: int) -> dict:
    r = env("declared", run_id, BASE - timedelta(minutes=5))
    r.update({"unit": unit, "track": track, "target": {"metric": metric, "op": op, "value": value, "unit": tunit}, "n_items": n_items,
              "test_set_sha256": TEST_SHA, "model_id": model_id, "grader": "t3/suite/grade_trust.py", "code_sha": CODE_SHA,
              "region": "us-east-2", "twin": TWIN, "declared_at": iso(BASE - timedelta(minutes=5))})
    return r


def result(run_id: str, unit: str, metric: str, op: str, value: float, tvalue: float, tunit: str, n: int, when: datetime, passed: bool, details: dict) -> dict:
    r = env("result", run_id, when)
    r.update({"unit": unit, "metric": metric, "value": value, "target": {"metric": metric, "op": op, "value": tvalue, "unit": tunit},
              "passed": passed, "n_items": n, "checked_at": iso(when), "checker": {"script": "tools/check_unit.py", "code_sha": "d65cb86"},
              "details": details})
    return r


def build() -> dict[str, list]:
    files: dict[str, list] = {}

    def put(path: str, rec) -> None:
        files.setdefault(path, []).append(rec)

    # ---- healthy
    h_run, h_pred = "t3-trust-healthy-20261102-a", {"model_id": "t2-response-mlp", "version": "v1"}
    es = [0.30, 0.45, 0.25, 0.55, 0.40, 1.10, 0.35, 0.30, 0.50, 0.28]
    trust = 0.78
    t = BASE
    for i, e in enumerate(es, 1):
        t = BASE + timedelta(seconds=40 * i)
        after = round(trust + 0.2 * ((1.0 - min(e, 1.0)) - trust), 4)  # an EWMA-like drift, fixture only
        put(f"records/{h_run}/change.jsonl", make_change(h_run, i, t, predictor=h_pred, e=e, trust=(round(trust, 4), after)))
        trust = after
    # a change clipped by the safety filter: shown, never scored (no trust fields)
    put(f"records/{h_run}/change.jsonl", make_change(h_run, 11, BASE + timedelta(seconds=40 * 11), predictor=h_pred, e=2.0, trust=None,
                                                     scored=False, reason="clipped"))
    put(f"models/{h_pred['model_id']}/{h_pred['version']}.json", model_rec(h_run, "t2-response-mlp", "v1", "approved", "t2-response", "t2-train-20261102-a", {"holdout_mae": 0.31}))
    put(f"declared/{h_run}.json", declared(h_run, "t3.trust-healthy", "track-3", "pauses", "==", 0, "count", "t2-response-mlp", 11))
    put(f"results/{h_run}.json", result(h_run, "t3.trust-healthy", "pauses", "==", 0, 0, "count", 11, BASE + timedelta(minutes=10), True,
                                        {"model_id": "t2-response-mlp", "version": "v1", "thresholds": THRESHOLDS}))

    # ---- degraded
    d_run, d_pred = "t3-trust-pause-20261102-a", {"model_id": "t2-response-mlp", "version": "v1-degraded"}
    es_d = [0.50, 0.70, 1.50, 1.80]
    trusts = [(0.80, 0.78), (0.78, 0.71), (0.71, 0.64), (0.64, 0.58)]
    for i, (e, tr) in enumerate(zip(es_d, trusts), 1):
        put(f"records/{d_run}/change.jsonl", make_change(d_run, i, BASE + timedelta(seconds=40 * i), predictor=d_pred, e=e, trust=tr))
    pause_t = BASE + timedelta(seconds=40 * 4 + 1.2)
    resume_t = BASE + timedelta(minutes=8)
    put(f"records/{d_run}/control.jsonl", {
        **env("control", d_run, pause_t), "model_id": "t2-response-mlp", "version": "v1-degraded", "action": "pause", "by": "trust-gate",
        "reason": "2 of the last 3 changes have e above tau 1.0 (e 1.5, 1.8); EWMA trust 0.58 is still above the pause line 0.55",
        "policy_name": "team13_ranpause_t2_response_mlp_v1_degraded", "time": iso(pause_t)})
    put(f"records/{d_run}/control.jsonl", {
        **env("control", d_run, resume_t), "model_id": "t2-response-mlp", "version": "v1-degraded", "action": "resume", "by": "person:approver-2",
        "reason": "Retrained response model queued; resuming to collect two more scored changes",
        "policy_name": "team13_ranpause_t2_response_mlp_v1_degraded", "time": iso(resume_t)})
    put(f"records/{d_run}/change.jsonl", make_change(d_run, 5, resume_t + timedelta(seconds=45), predictor=d_pred, e=1.60, trust=(0.58, 0.51)))
    put(f"models/{d_pred['model_id']}/{d_pred['version']}.json", model_rec(d_run, "t2-response-mlp", "v1-degraded", "approved", "t2-response", "t2-train-20261102-b", {"holdout_mae": 0.92}))
    put(f"declared/{d_run}.json", declared(d_run, "t3.trust-pause", "track-3", "changes_to_pause", "<=", 6, "count", "t2-response-mlp", 5))
    put(f"results/{d_run}.json", result(d_run, "t3.trust-pause", "changes_to_pause", "<=", 4, 6, "count", 5, resume_t + timedelta(minutes=2), True,
                                        {"model_id": "t2-response-mlp", "version": "v1-degraded", "thresholds": THRESHOLDS,
                                         "pause_at": iso(pause_t), "paused_by": "trust-gate"}))

    # ---- probes (no model)
    p_run = "t3-probe-20261102-a"
    for i in (1, 2):
        put(f"records/{p_run}/change.jsonl", make_change(p_run, i, BASE - timedelta(minutes=3) + timedelta(seconds=20 * i), predictor=None, e=None,
                                                         trust=None, scored=False, probe=True))
    put(f"declared/{p_run}.json", declared(p_run, "t3.probe", "track-3", "probes_run", ">=", 2, "count", "none", 2))

    # ---- ladders
    def ladder(run_id: str, version: str, rungs: list[dict], retention: float, ci, when: datetime) -> None:
        put(f"declared/{run_id}.json", declared(run_id, "t3.ladder", "track-3", "retention", ">=", 0.5, "ratio", "t1-ppo-maveric", sum(r["n_cases"] for r in rungs)))
        put(f"results/{run_id}.json", result(run_id, "t3.ladder", "retention", ">=", retention, 0.5, "ratio", sum(r["n_cases"] for r in rungs), when, retention >= 0.5,
                                             {"rungs": rungs, "retention": retention, "retention_ci": ci, "model_id": "t1-ppo-maveric", "version": version}))

    maveric = {"name": "rung-1", "label": "MAVERIC", "twin_kind": "maveric-inprocess", "n_cases": 40, "gain": 0.120, "gain_ci": [0.090, 0.150]}
    ladder("t3-ladder-rehearsal-20261102-a", "v3",
           [maveric, {"name": "rung-2", "label": "REHEARSAL (Viavi-like Maveric)", "twin_kind": "maveric-inprocess", "n_cases": 40, "gain": 0.084, "gain_ci": [0.050, 0.118]}],
           0.7, [0.42, 1.05], BASE + timedelta(minutes=30))
    ladder("t3-ladder-viavi-20261102-a", "v4",
           [{**maveric, "gain": 0.140, "gain_ci": [0.110, 0.170]}, {"name": "rung-2", "label": "VIAVI", "twin_kind": "viavi", "n_cases": 12, "gain": 0.063, "gain_ci": [0.010, 0.116]}],
           0.45, None, BASE + timedelta(minutes=45))
    for v, ms in (("v3", "approved"), ("v4", "candidate")):
        put(f"models/t1-ppo-maveric/{v}.json", model_rec("t1-ppo-20261102-a", "t1-ppo-maveric", v, ms, "ppo", "t1-ppo-20261102-a", {"mean_episode_reward": 12.84}))
    return files


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    for path, recs in build().items():
        p = OUT / path
        p.parent.mkdir(parents=True, exist_ok=True)
        if path.endswith(".jsonl"):
            p.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in recs))
        else:
            p.write_text(json.dumps(recs[0], indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
