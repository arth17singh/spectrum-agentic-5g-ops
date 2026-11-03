"""Generate the SYNTHETIC Twin view fixtures (every record has synthetic: true, aws_time null).

Run (needs the ran-rl SDK, e.g. the shared venv-t3):
    PYTHONPATH=<repo root> <venv-t3>/bin/python make_twinview_fixtures.py     (rewrites ./fixtures/)

The per-cell numbers are real outputs of W2's in-process Maveric twin (twin.replay.ReplayTwin) for the
scenario, seed and changes below; the records around them (predictions, approvals, times, trust) are
invented. The tests never import the SDK: they read the committed JSON.

Each change record carries the PROPOSED field `twin_state` (not in the W1 contract yet):

    twin_state: {
      before:   TwinSnapshot   the state the agents observed, at twin.step (before the change)
      after:    TwinSnapshot   after the change and one step (what measured_kpis summarises)
      baseline: TwinSnapshot   optional: the no-change counterfactual at the same step
    }
    TwinSnapshot: {step, coverage_ratio, sla_ratio, per_cell: [{cell, power_dbm, tilt_deg, ue_count,
                   coverage_share, sla_share, sinr_p10_db, sinr_mean_db, rsrp_mean_dbm, rsrp_p10_dbm, alarm, asleep}]}

The preset name is `twin.preset` on the change record (contract decision, 2 Nov).

Per-cell shares, as W2 defines them (commit 2977a7e):
  coverage_share = share of the cell's served UEs with serving RSRP > -97 dBm (RSRP only, no SINR condition,
                   same as the SDK's network coverage_ratio);
  sla_share      = share with SINR > 0 dB (strictly greater);
  both None for a cell with 0 UEs.
This script computes them from the twin's per-UE arrays and checks that the UE-weighted mean equals the
network ratio. When the twin exposes W2's `ReplayTwin.kpis()` (2977a7e), its per-cell shares are used
for the live states and must equal this computation.

Some changes carry the contract's top-level `approved_by` / `approved_at` (written by the tools Lambda);
others get their approver from a suite `item` record (extra.approval) or only a `decision` ALLOW (when,
not who), so the view's three approval sources are all exercised.
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
OUT = HERE / "fixtures"
REPO = HERE.parents[4]
sys.path.insert(0, str(REPO))

from twin.replay import ReplayTwin, _snapshot  # noqa: E402
from twin.presets import VIAVI_LIKE, VIAVI_SCENARIO  # noqa: E402

KPI = ("coverage_ratio", "sla_ratio", "sinr_p10_db", "sinr_mean_db")
BASE = datetime(2026, 11, 2, 21, 0, 0, tzinfo=timezone.utc)
CODE_SHA = "0123456789abcdef0123456789abcdef01234567"
TEST_SHA = "9f2c1d0a6b7e4c3f8a1b2c3d4e5f60718293a4b5c6d7e8f9a0b1c2d3e4f50617"


def iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def canon(cells: list[dict]) -> str:
    def f(v):
        s = f"{float(v):+.1f}"
        return "+0.0" if s == "-0.0" else s
    return "|".join(f"cell={c['cell']};d_power_db={f(c['d_power_db'])};d_tilt_deg={f(c['d_tilt_deg'])}"
                    for c in sorted(cells, key=lambda c: c["cell"]))


def change_key(case_id: str, item_id: str, text: str) -> str:
    return "chg-" + hashlib.sha256(json.dumps([case_id, item_id, text]).encode()).hexdigest()[:12]


def per_cell_shares(mt) -> tuple[list, list, float, float]:
    """Per-cell coverage / SLA share from the twin's per-UE arrays, as MavericTwin.observe() computes
    the network ratios (same prediction, same attachment, same thresholds)."""
    from ran_rl.twin.maveric.model import predict
    from ran_rl.twin.maveric.vendor.attachment import attach

    pred = predict(mt._models, mt._cells, mt._ue, mt._tilt, mt._power)
    rsrp = pred.rsrp_dbm
    rsrp[:, mt._alarm | mt._asleep] = -np.inf
    serving, served, sinr = attach(rsrp, mt._freq, mt._noise_floor_dbm)
    cov_ok = served > mt._cfg.coverage_rsrp_threshold_dbm
    sla_ok = sinr > mt._cfg.sla_sinr_threshold_db
    cov, sla = [], []
    for c in range(mt._n_cells):
        m = serving == c
        cov.append(float(cov_ok[m].mean()) if m.any() else None)
        sla.append(float(sla_ok[m].mean()) if m.any() else None)
    return cov, sla, float(cov_ok.mean()), float(sla_ok.mean())


def snapshot(mt, state: dict, step: int) -> dict:
    cov, sla, cov_all, sla_all = per_cell_shares(mt)
    assert abs(cov_all - state["coverage_ratio"]) < 1e-9, (cov_all, state["coverage_ratio"])
    assert abs(sla_all - state["sla_ratio"]) < 1e-9, (sla_all, state["sla_ratio"])
    cells = []
    for i in range(len(state["power_dbm"])):
        cells.append({
            "cell": i, "power_dbm": round(float(state["power_dbm"][i]), 3), "tilt_deg": round(float(state["tilt_deg"][i]), 3),
            "ue_count": int(state["ue_count"][i]),
            "coverage_share": None if cov[i] is None else round(cov[i], 4),
            "sla_share": None if sla[i] is None else round(sla[i], 4),
            "sinr_p10_db": round(float(state["sinr_p10_db"][i]), 3), "sinr_mean_db": round(float(state["sinr_mean_db"][i]), 3),
            "rsrp_mean_dbm": round(float(state["rsrp_mean_dbm"][i]), 3),
            "rsrp_p10_dbm": round(float(state["rsrp_p10_dbm"][i]), 3),
            "alarm": bool(state["alarm"][i]), "asleep": bool(state["asleep"][i]),
        })
    return {"step": step, "coverage_ratio": round(float(state["coverage_ratio"]), 4),
            "sla_ratio": round(float(state["sla_ratio"]), 4), "per_cell": cells}


def check_against_w2(rt, snap: dict) -> None:
    """If W2's ReplayTwin.kpis() exists, its per-cell shares must equal ours (same definitions)."""
    if not callable(getattr(rt, "kpis", None)):
        return
    theirs = rt.kpis()["per_cell"]
    for mine, w2 in zip(snap["per_cell"], theirs):
        for k in ("coverage_share", "sla_share"):
            a, b = mine[k], w2.get(k)
            assert (a is None and b is None) or (a is not None and b is not None and abs(a - b) < 1e-3), (k, a, b)


def headline(snap: dict) -> dict:
    """The four headline KPIs from a snapshot (UE-weighted SINR, as twin.kpis does)."""
    cells = snap["per_cell"]
    n = sum(c["ue_count"] for c in cells) or 1
    return {
        "coverage_ratio": snap["coverage_ratio"], "sla_ratio": snap["sla_ratio"],
        "sinr_p10_db": round(sum(c["ue_count"] * c["sinr_p10_db"] for c in cells) / n, 4),
        "sinr_mean_db": round(sum(c["ue_count"] * c["sinr_mean_db"] for c in cells) / n, 4),
    }


def envelope(kind: str, run_id: str, t: datetime) -> dict:
    return {"schema_version": "1", "kind": kind, "run_id": run_id, "team": "team13", "created_at": iso(t),
            "aws_time": None, "aws_time_source": None, "synthetic": True}


def run_case(*, run_id, case_id, scenario, seed, config, twin_kind, plan, rng, t0):  # config: the preset name
    """plan: list of dicts {change|None, approval_via ('record'|'item'|'decision'|'none'), probe, predict}."""
    rt = ReplayTwin(scenario, seed, config)
    state = rt.reset()
    changes, decisions, items = [], [], []
    t = t0
    for step, p in enumerate(plan):
        if p["change"] is None:  # a no-op window, nothing recorded
            state = rt.apply([])
            t += timedelta(minutes=2)
            continue
        cells = p["change"]
        text = canon(cells)
        item_id = f"step-{step}"
        key = change_key(case_id, item_id, text)
        before = snapshot(rt._twin, state, step)
        # baseline: no-change counterfactual on a snapshot
        trial = _snapshot(rt._twin)
        trial.apply_action(_noop_action(rt, state))
        base_state = trial.advance(rt.config["step_seconds"])
        baseline = snapshot(trial, base_state, step + 1)
        state = rt.apply(cells)
        after = snapshot(rt._twin, state, step + 1)
        check_against_w2(rt, after)
        entry = rt.action_log[-1]
        hb, hm = headline(baseline), headline(after)
        md = {k: hm[k] - hb[k] for k in KPI}
        t_req = t + timedelta(seconds=10)
        t_dec = t + timedelta(seconds=55 + rng.randint(0, 40))
        t_app = t_dec + timedelta(seconds=4)
        rec = {**envelope("change", run_id, t_app), "case_id": case_id, "change_id": key, "item_id": item_id,
               "change": [{"cell": c["cell"], "d_power_db": float(f"{c['d_power_db']:.1f}"),
                           "d_tilt_deg": float(f"{c['d_tilt_deg']:.1f}")} for c in sorted(cells, key=lambda c: c["cell"])],
               "change_text": text,
               "twin": {"kind": twin_kind, "version": "ran-rl-1.2.9", "scenario": scenario, "seed": seed, "step": step,
                        "preset": config},
               "baseline_kpis": hb, "measured_kpis": hm, "measured_delta": md,
               "predicted_delta": None, "predictor": None, "policy": None, "error": None, "direction_agree": None,
               "scored": False, "unscored_reason": None, "probe": bool(p.get("probe")),
               "trust_before": None, "trust_after": None,
               "twin_state": {"before": before, "after": after, "baseline": baseline}}
        if entry.get("clamped"):
            rec["unscored_reason"] = "clipped"
        elif p.get("probe"):
            rec["unscored_reason"] = "probe"
        else:
            noise = p.get("noise", 0.25)
            pd = {k: md[k] + (rng.uniform(-1, 1) * noise * (0.05 if "ratio" in k else 1.0)) for k in KPI}
            pd = {k: round(v, 4) for k, v in pd.items()}
            err = {k: abs(pd[k] - md[k]) for k in KPI}
            err["summary"] = round(sum(err[k] / s for k, s in (("coverage_ratio", 0.08), ("sla_ratio", 0.06), ("sinr_p10_db", 1.0))) / 3, 4)
            agree = all((pd[k] > 0) == (md[k] > 0) or abs(md[k]) < 1e-3 for k in ("coverage_ratio", "sla_ratio"))
            tb = p.get("trust_before", 0.8)
            rec.update(predicted_delta=pd, predictor={"model_id": "t2-response-mlp", "version": "v1"},
                       policy={"model_id": "t1-ppo-maveric", "version": "v3"}, error=err, direction_agree=agree,
                       scored=True, trust_before=tb, trust_after=round(tb + (0.02 if err["summary"] < 0.5 else -0.08), 3))
        approver = p.get("approver", "person:navigator-1")
        via = p["approval_via"]
        if via == "record":
            rec["approved_by"] = approver
            rec["approved_at"] = iso(t_dec)
        if via == "item":
            items.append({**envelope("item", run_id, t_app), "item_id": f"t3-{case_id[-2:]}-{step}",
                          "input_sha256": "0" * 64, "output_sha256": "1" * 64, "score": 1, "passed": True,
                          "grader": "approval-chain", "backend": "fixture", "latency_ms": None, "cost_usd": None,
                          "extra": {"case_id": case_id, "approval_step": True, "approval_id": key,
                                    "approval": {"status": "APPROVED", "decided_by": approver,
                                                 "decided_at": int(t_dec.timestamp())}}})
        if via in ("record", "item", "decision"):
            for when, verdict, rule in ((t_req, "DENY", "team13_require_human_approval"), (t_dec, "ALLOW", "team13_may_apply_change")):
                decisions.append({**envelope("decision", run_id, when), "case_id": case_id, "trace_id": f"t-{case_id}",
                                  "span_id": f"s-{step}-{verdict.lower()}", "time": iso(when), "agent": "lead",
                                  "tool": "ran___apply_change", "decision": verdict, "deciding_rule": rule,
                                  "reason": "needs a person's approval" if verdict == "DENY" else None,
                                  "guardrail_scores": None, "approval_id": key, "source": "/team13/checkpoint"})
        changes.append(rec)
        t += timedelta(minutes=3)
    return changes, decisions, items


def _noop_action(rt, state):
    from twin._change import filter_change
    action, _ = filter_change([], state, rt._safety, len(rt.action_log))
    return action


def declared(run_id: str, scenario: str, seed: int, config, kind: str, t: datetime) -> dict:
    return {**envelope("declared", run_id, t), "unit": "t3.twin-view", "track": "track-3",
            "target": {"metric": "changes_applied", "op": ">=", "value": 1, "unit": "count"},
            "baseline_run_id": None, "n_items": 3, "test_set_sha256": TEST_SHA, "model_id": "t1-ppo-maveric",
            "grader": "fixture", "code_sha": CODE_SHA, "region": "us-east-2",
            "twin": {"kind": kind, "version": "ran-rl-1.2.9", "scenario": scenario, "seed": seed, "config": config},
            "declared_at": iso(t)}


def write(run_id: str, kind: str, rows: list[dict]) -> None:
    if not rows:
        return
    d = OUT / "records" / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{kind}.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))


def main() -> None:
    rng = random.Random(1102)
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "declared").mkdir(parents=True)

    # R1 Maveric (official preset): three windows, three changes
    r1 = "t3-twin-maveric-20261102-a"
    (OUT / "declared" / f"{r1}.json").write_text(json.dumps(declared(r1, "interference", 11, {"preset": "official"}, "maveric-inprocess", BASE), indent=2) + "\n")
    c, d, i = run_case(run_id=r1, case_id="case-twin-01", scenario="interference", seed=11, config="official",
                       twin_kind="maveric-inprocess", rng=rng, t0=BASE + timedelta(minutes=1), plan=[
                           {"change": [{"cell": 1, "d_power_db": 3.0, "d_tilt_deg": -1.0}], "approval_via": "record", "trust_before": 0.78},
                           {"change": [{"cell": 3, "d_power_db": 2.0, "d_tilt_deg": 0.0}], "approval_via": "item", "trust_before": 0.80},
                           {"change": [{"cell": 0, "d_power_db": -1.0, "d_tilt_deg": 1.0}, {"cell": 2, "d_power_db": 2.0, "d_tilt_deg": 0.0}],
                            "approval_via": "decision", "trust_before": 0.82},
                       ])
    write(r1, "change", c), write(r1, "decision", d), write(r1, "item", i)

    # R2 Rehearsal (viavi-like preset on Maveric): a probe, a no-op window, then a change with no approval record
    r2 = "t3-twin-rehearsal-20261102-a"
    (OUT / "declared" / f"{r2}.json").write_text(json.dumps(declared(r2, VIAVI_SCENARIO, 3, {"preset": VIAVI_LIKE}, "maveric-inprocess", BASE), indent=2) + "\n")
    c, d, i = run_case(run_id=r2, case_id="case-twin-02", scenario=VIAVI_SCENARIO, seed=3, config=VIAVI_LIKE,
                       twin_kind="maveric-inprocess", rng=rng, t0=BASE + timedelta(minutes=20), plan=[
                           {"change": [{"cell": 2, "d_power_db": -2.0, "d_tilt_deg": 0.0}], "approval_via": "record", "probe": True,
                            "approver": "person:navigator-2"},
                           {"change": None},
                           {"change": [{"cell": 1, "d_power_db": 2.0, "d_tilt_deg": -2.0}], "approval_via": "none", "trust_before": 0.7},
                       ])
    write(r2, "change", c), write(r2, "decision", d), write(r2, "item", i)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
