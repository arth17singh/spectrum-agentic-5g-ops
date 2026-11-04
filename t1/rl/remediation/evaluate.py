"""remediation-v1 vs ppo_sm_50k, config_revert and rule_baseline on held-out incidents: one evidence unit.

    python -m t1.rl.remediation.evaluate --print-declaration > /tmp/rem-eval-decl.json
    python -m tools.sm.run --run-id t3.evals.remediation-policy-<UTC> --declaration /tmp/rem-eval-decl.json \\
      --instance-type ml.m5.xlarge --max-run 3600 --requirement "stable-baselines3==2.9.2" \\
      --requirement "gymnasium==1.4.0" --requirement "pydantic>=2.7,<3" \\
      --entry "-m t1.rl.remediation.evaluate --run-id {run_id} --evidence-dir {evidence_dir} \\
               --output-dir {output_dir} --workers 4" [--dry-run]

Every operator plays every incident through t3/evals/run.py's ``run_incident`` (reset, commissioning,
fault, three operator steps, graded at step 5 against the same-step healthy counterfactual): the path the
judged swarm and baselines were scored on. Suites (t1/rl/remediation/targets.json, declared first):

    test-masked, test-visible   the frozen test split (frozen on x86_64: replayed as frozen in the job)
    confirm                     the frozen confirmation suite (frozen on Darwin-arm64: rebuilt from its
                                frozen specs on the job's platform, keep filter not re-applied)
    dev-holdout                 role=holdout candidates (never trained on), first N kept, built in the job

One ``item`` per incident: score = recovery(v1) - recovery(ppo_sm_50k), passed = v1's success, and every
operator's outcome and secondary measures in extra. The declared target is the 95% paired-bootstrap lower
bound of that difference on the test-masked items (details.gain, replayable by tools/check_unit.py).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
TARGETS_PATH = HERE / "targets.json"
V1_PATH = REPO / "t1" / "rl" / "artifacts" / "remediation_v1.zip"
PPO_PATH = REPO / "t1" / "rl" / "artifacts" / "ppo_sm_50k.zip"
PRIMARY = "test-masked"
N_HOLDOUT = 100
SMOKE_MAX = 50
OPERATORS = ("v1", "ppo_sm_50k", "config_revert", "rule_baseline")
MEASURES = ("recovery", "success", "diagnosis_hit", "cells_touched", "total_abs_power_db", "total_abs_tilt_deg",
            "headroom_used", "power_above_healthy_db", "healthy_false_change")


def declared() -> dict[str, Any]:
    return json.loads(TARGETS_PATH.read_text())


def platform_tag() -> str:
    return f"{platform.system()}-{platform.machine()}"


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- suites

def rebuild(frozen: dict[str, Any], targets: dict[str, Any]) -> dict[str, Any]:
    """A frozen incident's trajectories recomputed on THIS platform from its frozen spec (keep filter not
    re-applied: the incident was chosen where it was frozen). Same loop as incidents.build."""
    from t3.evals import incidents as inc
    from t3.evals.graders import grade_kpis
    from t3.evals.twin_backend import ReplayTwin, graded

    spec = {k: frozen[k] for k in ("seed", "preset", "scenario", "tier", "template", "commissioning", "fault")}
    built, why = inc.build(spec, targets)
    out = dict(frozen)
    if built is not None:
        if built["incident_id"] != frozen["incident_id"]:
            raise RuntimeError(f"rebuilt id {built['incident_id']} != frozen {frozen['incident_id']}")
        for k in ("fault_filtered", "pre_fault_kpis", "post_fault_kpis", "healthy_kpis_by_step",
                  "unfixed_kpis_by_step", "material_kpis"):
            out[k] = built[k]
        out["rebuilt"] = {"platform": platform_tag(), "kept_here": True}
        return out
    ep = targets["episode"]
    fs, gs = ep["fault_step"], ep["grade_step"]
    h, u = ReplayTwin(spec["scenario"], spec["seed"], spec["preset"]), ReplayTwin(spec["scenario"], spec["seed"],
                                                                                   spec["preset"])
    s0 = h.reset()
    u.reset()
    hs, us = inc.script(spec, targets, False), inc.script(spec, targets, True)
    ht, ut = [graded(s0)], [graded(s0)]
    for step in range(1, gs + 1):
        ht.append(graded(h.apply(hs[step - 1] if step <= fs else [])))
        ut.append(graded(u.apply(us[step - 1] if step <= fs else [])))
    g = grade_kpis(ut[-1], ht[-1], ut[-1], targets)
    out.update(fault_filtered=u.action_log[fs - 1]["filtered"], pre_fault_kpis=inc._r(ht[fs - 1]),
               post_fault_kpis=inc._r(ut[fs]), healthy_kpis_by_step=[inc._r(k) for k in ht],
               unfixed_kpis_by_step=[inc._r(k) for k in ut],
               material_kpis=[k for k, p in g["per_kpi"].items() if p["material"]],
               rebuilt={"platform": platform_tag(), "kept_here": False, "why": why})
    return out


def holdout(targets: dict[str, Any], n: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The first n role=holdout candidates (index order) that incidents.build keeps here."""
    from t1.rl.remediation import data as D
    from t3.evals import incidents as inc

    specs = [s for s in D.load_candidates() if s["role"] == "holdout"]
    kept, tried = [], 0
    for s in specs:
        tried += 1
        built, _ = inc.build(s, targets)
        if built is not None:
            kept.append({**built, "suite": "remediation-dev-holdout", "split": "dev-holdout"})
            if len(kept) == n:
                break
    if len(kept) < n:
        raise RuntimeError(f"only {len(kept)} of {n} holdout incidents kept from {len(specs)} holdout candidates")
    return kept, {"holdout_candidates": len(specs), "tried": tried, "kept": len(kept),
                  "candidates_sha256": sha256_file(D.CANDIDATES_PATH)}


def load_suites(targets: dict[str, Any], *, n_holdout: int, limit: dict[str, int] | None = None,
                rebuild_mode: str = "auto") -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    from t3.evals import confirm as C
    from t3.evals import incidents as inc

    limit = limit or {}
    here = platform_tag()
    test, test_sha = inc.load_test()
    test_platform = json.loads(inc.META_PATH.read_text()).get("platform")
    conf, conf_sha = C.load_confirm()
    conf_platform = json.loads(C.CONFIRM_META_PATH.read_text()).get("platform")

    def maybe(incs, frozen_on):
        do = rebuild_mode == "always" or (rebuild_mode == "auto" and frozen_on != here)
        return ([rebuild(i, targets) for i in incs] if do else incs), do

    suites: dict[str, list[dict[str, Any]]] = {}
    info: dict[str, Any] = {}
    for name, tier in (("test-masked", "masked"), ("test-visible", "visible")):
        sel = [i for i in test if i.get("tier", "visible") == tier][: limit.get(name, None)]
        suites[name], rb = maybe(sel, test_platform)
        info[name] = {"file": "t3/evals/suite/incidents-test.jsonl", "sha256": test_sha, "tier": tier,
                      "frozen_on": test_platform, "rebuilt_here": rb, "n": len(suites[name])}
    sel = conf[: limit.get("confirm", None)]
    suites["confirm"], rb = maybe(sel, conf_platform)
    info["confirm"] = {"file": "t3/evals/suite/incidents-confirm.jsonl", "sha256": conf_sha, "frozen_on": conf_platform,
                       "rebuilt_here": rb, "n": len(suites["confirm"])}
    suites["dev-holdout"], hinfo = holdout(targets, n_holdout)
    info["dev-holdout"] = {**hinfo, "built_here": here, "n": len(suites["dev-holdout"])}
    return suites, info


# --------------------------------------------------------------------------- operators

def operator_specs(v1_path: Path, ppo_path: Path) -> dict[str, dict[str, Any]]:
    return {"v1": {"kind": "ppo", "path": str(v1_path)}, "ppo_sm_50k": {"kind": "ppo", "path": str(ppo_path)},
            "config_revert": {"kind": "config_revert"}, "rule_baseline": {"kind": "rule_baseline"}}


def make_ops(specs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    from t1.rl.policy import load
    from t3.evals.operators import make

    return {k: (make("ppo", policy=load(s["path"])) if s["kind"] == "ppo" else make(s["kind"]))
            for k, s in specs.items()}


def operator_info(specs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ops = make_ops(specs)
    out = {}
    for k, s in specs.items():
        out[k] = {"model_id": ops[k].model_id}
        if s["kind"] == "ppo":
            p = Path(s["path"])
            out[k].update(artifact=p.name, sha256=sha256_file(p))
            side = p.with_suffix(".meta.json")
            if side.exists():
                out[k]["meta"] = json.loads(side.read_text())
    return out


# --------------------------------------------------------------------------- one incident

def _changes(text: str) -> list[dict[str, Any]]:
    from t1.rl.policy import parse_change

    return parse_change(text) if text else []


def applied_measures(incident: dict[str, Any], change_texts: list[str], targets: dict[str, Any]) -> dict[str, Any]:
    """Replay the operator's requested changes on a fresh twin: what the safety filter let through."""
    from t3.evals import incidents as inc
    from t3.evals.twin_backend import ReplayTwin, graded

    fs = targets["episode"]["fault_step"]
    hc = ReplayTwin(incident["scenario"], incident["seed"], incident["preset"])
    hc.reset()
    comm = np.asarray(hc.apply(list(incident["commissioning"]))["power_dbm"], dtype=float)
    pmax = float(hc.config["safety"]["power_max_dbm"])
    tw = ReplayTwin(incident["scenario"], incident["seed"], incident["preset"])
    tw.reset()
    for ch in inc.script(incident, targets, with_fault=True):
        tw.apply(ch)
    state = None
    for text in change_texts:
        state = tw.apply(_changes(text))
    log = tw.action_log[fs:]
    touched: set[int] = set()
    tot_p = tot_t = 0.0
    first_cell = None
    for entry in log:
        dp = np.asarray(entry["filtered"]["power_delta_db"], dtype=float)
        dt = np.asarray(entry["filtered"]["tilt_delta_deg"], dtype=float)
        moved = np.flatnonzero((dp != 0) | (dt != 0))
        touched |= {int(c) for c in moved}
        tot_p += float(np.abs(dp).sum())
        tot_t += float(np.abs(dt).sum())
        if first_cell is None and len(moved):
            first_cell = int(np.argmax(np.abs(dp) / 6.0 + np.abs(dt) / 5.0))
    final_p = np.asarray(tw.state["power_dbm"], dtype=float)
    above = float(np.maximum(final_p - comm, 0.0).sum())
    room = float(np.maximum(pmax - comm, 0.0).sum())
    return {"cells_touched": len(touched), "total_abs_power_db": round(tot_p, 6),
            "total_abs_tilt_deg": round(tot_t, 6), "power_above_healthy_db": round(above, 6),
            "headroom_used": round(above / room, 6) if room > 0 else 0.0,
            "first_change_cell": first_cell,
            "diagnosis_hit": first_cell is not None and first_cell in set(incident["fault_cells"]),
            "replayed_final_kpis": graded(state) if state is not None else None}


def healthy_check(incident: dict[str, Any], op: Any, targets: dict[str, Any]) -> dict[str, Any]:
    """The operator on the same network without the fault, before its first step."""
    from t3.evals import incidents as inc
    from t3.evals.changes import canon, nonzero
    from t3.evals.twin_backend import ReplayTwin

    tw = ReplayTwin(incident["scenario"], incident["seed"], incident["preset"])
    tw.reset()
    for ch in inc.script(incident, targets, with_fault=False):
        tw.apply(ch)
    if hasattr(op, "begin"):
        op.begin(incident, {"twin": tw, "targets": targets, "run_id": "healthy-check"})
    change = nonzero(op.act(tw.state, []))
    return {"healthy_false_change": bool(change), "healthy_change_text": canon(change) if change else ""}


def play(suite: str, incident: dict[str, Any], ops: dict[str, Any], targets: dict[str, Any]) -> dict[str, Any]:
    from t3.evals.run import run_incident

    res: dict[str, Any] = {}
    for key, op in ops.items():
        it = run_incident(incident, op, targets)
        ex = it["extra"]
        am = applied_measures(incident, ex["change_texts"], targets)
        replayed = am.pop("replayed_final_kpis")
        replay_ok = replayed is not None and all(abs(replayed[k] - v) <= 1e-9 for k, v in ex["final_kpis"].items())
        res[key] = {"model_id": op.model_id, "recovery": it["score"], "success": it["passed"],
                    "change_texts": ex["change_texts"], "changes_used": ex["changes_used"],
                    "wrong_cell_actions": ex["wrong_cell_actions"], "clamped": ex["clamped"], "rules": ex["rules"],
                    "latency_ms": it["latency_ms"], "replay_consistent": bool(replay_ok), **am,
                    **healthy_check(incident, op, targets)}
    return {"suite": suite, "incident": incident, "operators": res}


_OPS: dict[str, dict[str, Any]] = {}


def _work(args: tuple[str, dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]) -> dict[str, Any]:
    from t3.evals.run import single_thread

    suite, incident, specs, targets = args
    single_thread()
    key = json.dumps(specs, sort_keys=True)
    if key not in _OPS:  # one set of operators per process (the policies are deterministic and stateless)
        _OPS[key] = make_ops(specs)
    return play(suite, incident, _OPS[key], targets)


# --------------------------------------------------------------------------- records

def item_for(r: dict[str, Any]) -> dict[str, Any]:
    from t3.evals import incidents as inc
    from t3.evals.graders import GRADER_ID
    from t3.evals.twin_backend import BACKEND

    i, ops = r["incident"], r["operators"]
    v1, pp = ops["v1"], ops["ppo_sm_50k"]
    if v1["recovery"] is None or pp["recovery"] is None:
        raise RuntimeError(f"{i['incident_id']}: no material KPI here, so no recovery (refusing to guess)")
    outputs = {k: o["change_texts"] for k, o in ops.items()}
    return {
        "item_id": f"{r['suite']}/{i['incident_id']}",
        "input_sha256": hashlib.sha256(inc.canonical_line(i).encode()).hexdigest(),
        "output_sha256": hashlib.sha256(inc.canonical_line(outputs).encode()).hexdigest(),
        "score": float(v1["recovery"]) - float(pp["recovery"]),
        "passed": bool(v1["success"]),
        "grader": GRADER_ID + "; item score = recovery(v1) - recovery(ppo_sm_50k)",
        "backend": BACKEND,
        "latency_ms": float(v1["latency_ms"]),
        "extra": {"suite": r["suite"],
                  "incident": {k: i.get(k) for k in ("incident_id", "seed", "preset", "tier", "template", "fault",
                                                     "fault_cells", "split")},
                  "rebuilt": i.get("rebuilt"),
                  "operators": ops},
    }


def value_of(o: dict[str, Any], m: str) -> float:
    v = o[m]
    return float(v) if not isinstance(v, bool) else (1.0 if v else 0.0)


def summarize(rows: list[dict[str, Any]], boot: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    from t3.evals.stats import paired_bootstrap

    suites = sorted({r["suite"] for r in rows})
    summary: dict[str, Any] = {}
    secondary: dict[str, Any] = {}
    for s in suites:
        rs = [r for r in rows if r["suite"] == s]
        summary[s] = {}
        for op in OPERATORS:
            os_ = [r["operators"][op] for r in rs]
            summary[s][op] = {"n": len(os_), **{f"mean_{m}": float(np.mean([value_of(o, m) for o in os_]))
                                                for m in MEASURES},
                              "success_count": sum(1 for o in os_ if o["success"]),
                              "no_change_incidents": sum(1 for o in os_ if o["first_change_cell"] is None),
                              "wrong_cell_actions_total": sum(o["wrong_cell_actions"] for o in os_),
                              "clamped_incidents": sum(1 for o in os_ if o["clamped"]),
                              "model_id": os_[0]["model_id"] if os_ else None}
        secondary[s] = {}
        ids = [r["incident"]["incident_id"] for r in rs]
        for m in MEASURES:
            secondary[s][m] = {}
            cand = {i: value_of(r["operators"]["v1"], m) for i, r in zip(ids, rs)}
            for b in OPERATORS[1:]:
                base = {i: value_of(r["operators"][b], m) for i, r in zip(ids, rs)}
                g = paired_bootstrap(cand, base, boot["resamples"], boot["seed"], boot["ci"])
                secondary[s][m][f"v1_minus_{b}"] = g
    return summary, secondary


def declaration(specs: dict[str, dict[str, Any]], *, n_holdout: int, limit: dict[str, int] | None = None,
                smoke: bool = False, run_id: str | None = None) -> dict[str, Any]:
    from t3.evals import confirm as C
    from t3.evals import incidents as inc
    from t3.evals.graders import GRADER_ID, TARGETS_PATH as T3_TARGETS
    from t3.evals.twin_backend import BACKEND, twin_version

    tg = declared()
    limit = limit or {}
    test, test_sha = inc.load_test()
    conf, conf_sha = C.load_confirm()
    n_masked = len([i for i in test if i.get("tier") == "masked"][: limit.get("test-masked", None)])
    n_visible = len([i for i in test if i.get("tier", "visible") == "visible"][: limit.get("test-visible", None)])
    n_conf = len(conf[: limit.get("confirm", None)])
    ops = operator_info(specs)
    d: dict[str, Any] = {
        "unit": tg["unit"], "track": "track-3", "target": dict(tg["primary"]["target"]),
        "n_items": n_masked + n_visible + n_conf + n_holdout, "test_set_sha256": test_sha,
        "subset": f"one item per incident: {n_masked} test-masked + {n_visible} test-visible + {n_conf} confirm + "
                  f"{n_holdout} dev-holdout; the target is on extra.suite={PRIMARY}",
        "model_id": ops["v1"]["model_id"], "grader": GRADER_ID + "; item score = recovery(v1) - recovery(ppo_sm_50k)",
        "synthetic": bool(smoke),
        "twin": {"kind": "maveric-inprocess", "version": twin_version(), "scenario": "incident-suite",
                 "seed": min(int(i["seed"]) for i in test + conf),
                 "config": {"backend": BACKEND, "targets": "t1/rl/remediation/targets.json",
                            "targets_sha256": sha256_file(TARGETS_PATH),
                            "t3_targets_sha256": sha256_file(T3_TARGETS),
                            "suites": {"test": {"file": "t3/evals/suite/incidents-test.jsonl", "sha256": test_sha},
                                       "confirm": {"file": "t3/evals/suite/incidents-confirm.jsonl",
                                                   "sha256": conf_sha,
                                                   "rule": "rebuilt from its frozen specs where the platform differs "
                                                           "from the one it was frozen on (keep filter not re-applied)"},
                                       "dev-holdout": {"file": "t1/rl/remediation/candidates.jsonl",
                                                       "sha256": sha256_file(HERE / "candidates.jsonl"),
                                                       "rule": f"role=holdout candidates in index order, the first "
                                                               f"{n_holdout} that incidents.build keeps here"}},
                            "limits": limit or None,
                            "operators": ops, "primary": tg["primary"],
                            "secondary_measures": tg["secondary_measures"],
                            "executed": "SageMaker training job started by tools/sm/run.py on ml.m5.xlarge (x86_64, "
                                        "the platform the test split was frozen on); in-process twin (twin/, W2)",
                            "evidence_writer": "tools/evlib.py: a local store in the job, uploaded and attested by "
                                               "tools/sm/entry.py",
                            "smoke": bool(smoke)}},
    }
    if run_id:
        d["run_id"] = run_id
    return d


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id")
    ap.add_argument("--evidence-dir", type=Path)
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--policy", type=Path, default=V1_PATH, help="remediation-v1 artifact (zip + .meta.json)")
    ap.add_argument("--baseline-policy", type=Path, default=PPO_PATH)
    ap.add_argument("--holdout", type=int, default=N_HOLDOUT)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--rebuild", choices=("auto", "always", "never"), default="auto")
    ap.add_argument("--smoke", action="store_true", help="local smoke: allows the limits below; synthetic")
    ap.add_argument("--limit-test", type=int, help="smoke: first K incidents of each test tier")
    ap.add_argument("--limit-confirm", type=int, help="smoke: first K confirmation incidents")
    ap.add_argument("--declare", action="store_true", help="also write the declaration (local runs only)")
    ap.add_argument("--print-declaration", action="store_true")
    a = ap.parse_args(argv)

    from t3.evals.evshim import Evidence, code_sha
    from t3.evals.graders import load_targets
    from t3.evals.stats import ci_spec, paired_bootstrap

    limit: dict[str, int] = {}
    if a.limit_test is not None:
        limit["test-masked"] = limit["test-visible"] = a.limit_test
    if a.limit_confirm is not None:
        limit["confirm"] = a.limit_confirm
    if (limit or a.holdout != N_HOLDOUT) and not a.smoke:
        ap.error("limits and a different --holdout are for --smoke runs only (the job runs the declared suites)")
    if a.smoke:
        total = (2 * limit.get("test-masked", 18)) + limit.get("confirm", 18) + a.holdout
        if total > SMOKE_MAX:
            ap.error(f"a local smoke may touch at most {SMOKE_MAX} incidents (this one: {total})")
    specs = operator_specs(a.policy, a.baseline_policy)
    if a.print_declaration:
        print(json.dumps(declaration(specs, n_holdout=a.holdout, limit=limit, smoke=a.smoke), indent=2, sort_keys=True))
        return 0
    if not (a.run_id and a.evidence_dir and a.output_dir):
        ap.error("--run-id, --evidence-dir and --output-dir are required to run")
    targets = load_targets()
    boot = targets["bootstrap"]
    ev = Evidence(a.evidence_dir)
    decl = declaration(specs, n_holdout=a.holdout, limit=limit, smoke=a.smoke, run_id=a.run_id)
    if a.declare:  # a local run; in a job tools/sm/run.py wrote it (with code_sha and region) before the job
        ev.write_declared({**decl, "code_sha": code_sha(), "region": "us-east-2"})
    t0 = time.time()
    suites, sinfo = load_suites(targets, n_holdout=a.holdout, limit=limit, rebuild_mode=a.rebuild)
    work = [(s, i, specs, targets) for s in ("test-masked", "test-visible", "confirm", "dev-holdout")
            for i in suites[s]]
    if a.workers > 1:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            rows = list(pool.map(_work, work))
    else:
        rows = [_work(w) for w in work]
    items = [item_for(r) for r in rows]
    if len(items) != decl["n_items"]:
        raise RuntimeError(f"{len(items)} items, declared {decl['n_items']}")
    for it in items:
        it["synthetic"] = bool(a.smoke)
    ev.append_records(a.run_id, "item", items)
    sub = {"field": "extra.suite", "equals": PRIMARY}
    prim = {it["item_id"]: it["score"] for it in items if it["extra"]["suite"] == PRIMARY}
    g = paired_bootstrap(prim, {k: 0.0 for k in prim}, boot["resamples"], boot["seed"], boot["ci"])
    ppo_mid = rows[0]["operators"]["ppo_sm_50k"]["model_id"]
    gain = {**ci_spec(boot, None, ppo_mid, "score", sub), **g, "metric": "recovery", "suite": PRIMARY,
            "note": "items hold recovery(v1) - recovery(ppo_sm_50k); baseline_run_id null"}
    summary, secondary = summarize(rows, boot)
    tgt = decl["target"]
    value = float(g["ci_low"])
    passed = {">": value > tgt["value"], ">=": value >= tgt["value"]}[tgt["op"]]
    result = {
        "run_id": a.run_id, "unit": decl["unit"], "metric": tgt["metric"], "value": value, "target": tgt,
        "passed": bool(passed), "n_items": len(items), "synthetic": bool(a.smoke),
        "checker": {"script": "t1/rl/remediation/evaluate.py", "code_sha": code_sha()},
        "details": {"gain": gain, "summary": summary, "secondary": secondary, "suites": sinfo,
                    "operators": decl["twin"]["config"]["operators"], "platform": platform_tag(),
                    "wall_s": round(time.time() - t0, 1)},
    }
    from t1.rl.remediation.job_train import finite

    ev.write_result(finite(result))
    a.output_dir.mkdir(parents=True, exist_ok=True)
    (a.output_dir / "remediation-eval-rows.jsonl").write_text(
        "".join(json.dumps(finite({"suite": r["suite"], "incident_id": r["incident"]["incident_id"],
                                   "operators": r["operators"]}), sort_keys=True) + "\n" for r in rows))
    brief = {s: {op: {k: summary[s][op][k] for k in ("n", "mean_recovery", "success_count", "mean_diagnosis_hit",
                                                     "mean_cells_touched", "mean_total_abs_power_db",
                                                     "mean_headroom_used", "mean_healthy_false_change")}
                     for op in OPERATORS} for s in summary}
    print(json.dumps(finite({"run_id": a.run_id, "metric": tgt["metric"], "value": value, "passed": passed,
                             "gain": {k: gain[k] for k in ("value", "ci_low", "ci_high", "paired_n")},
                             "summary": brief, "suites": sinfo, "wall_s": result["details"]["wall_s"]}), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
