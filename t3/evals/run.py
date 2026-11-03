"""Score one operator on the frozen incident suite, write evidence, and compute paired gains.

    python -m t3.evals.run --operator no_action
    python -m t3.evals.run --operator rule_baseline --baseline-run-id <no_action run>
    python -m t3.evals.run --operator config_revert --baseline-run-id <no_action run>
    python -m t3.evals.run --operator oracle --workers 10 \\
        --baseline-run-id <no_action run> <rule_baseline run> <config_revert run>
    python -m t3.evals.run --operator swarm --mode proposal --subset masked --swarm-client mymod:factory \\
        --baseline-run-id <config_revert run> <rule_baseline run>
    python -m t3.evals.run gains --run <A> --baseline <B>          # paired gain from stored items

Baseline arguments (``--baseline-run-id``, one or more run ids in the same evidence store):
- the FIRST id is ``declared.baseline_run_id`` and ``result.details.gain``'s baseline (except for the
  swarm, whose ``details.gain`` is its declared primary below);
- ``rule_baseline``, ``config_revert`` and ``ppo`` declare "gain over no_action", so their list must
  include a no_action run (found by its declared model_id ``no_action``);
- ``swarm`` declares "gain over config_revert on the masked tier": its list must include a
  config_revert run (model_id ``config-revert-median-v1``, run on the FULL test split so every masked
  incident is paired) and should include a rule_baseline run for the secondary wrong-cell check;
- every id listed gets a paired gain in ``result.details.gains``.

Order per run: the declaration (target from targets.json, suite hash, item count, code sha) is
written first, then one ``item`` record per incident, then the ``result``. Gains are paired by
item_id against the baseline runs' stored items (read back from the evidence store), with a 95%
bootstrap interval: ``result.details.gain = {value, ci_low, ci_high, baseline_run_id, paired_n}``.

Episode: reset (step 0), commissioning (step 1), the incident's fault (step 2), then exactly 3
operator steps (3-5); a step with no change still advances. Graded at step 5 against the same-step
healthy counterfactual.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from t3.evals import incidents as inc_mod
from t3.evals.changes import canon, nonzero
from t3.evals.evshim import EVLIB, Evidence, code_dirty, code_sha, new_run_id, utc_now
from t3.evals.graders import GRADER_ID, TARGETS_PATH, action_stats, grade_kpis, load_targets
from t3.evals.operators import make
from t3.evals.stats import paired_bootstrap
from t3.evals.twin_backend import BACKEND, ReplayTwin, graded, twin_version

DEFAULT_EVIDENCE = Path(__file__).with_name("evidence")
CHECKER = "t3/evals/run.py"
REPLAY_TOL = 1e-6


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _close(a: dict[str, float], b: dict[str, float], keys) -> bool:
    return not replay_diffs(a, b, keys)


def replay_diffs(got: dict[str, float], frozen: dict[str, float], keys) -> list[str]:
    """One line per KPI that misses the replay tolerance: value replayed, value frozen, difference."""
    out = []
    for k in keys:
        if not math.isclose(got[k], frozen[k], rel_tol=REPLAY_TOL, abs_tol=1e-8):
            d = got[k] - frozen[k]
            rel = abs(d) / max(abs(frozen[k]), 1e-12)
            out.append(f"{k}: replayed {got[k]!r} vs frozen {frozen[k]!r} (diff {d:+.3e}, rel {rel:.3e}, "
                       f"tolerance rel {REPLAY_TOL:g})")
    return out


class ReplayMismatch(RuntimeError):
    """The twin on this machine does not reproduce the frozen suite (wrong backend or platform)."""


def run_incident(incident: dict[str, Any], op: Any, targets: dict[str, Any], run_id: str = "adhoc") -> dict[str, Any]:
    """Play one incident with ``op`` and grade it. Returns the item record (without envelope)."""
    ep = targets["episode"]
    keys = targets["graded_kpis"]
    twin = ReplayTwin(incident["scenario"], incident["seed"], incident["preset"])
    twin.reset()
    for change in inc_mod.script(incident, targets, with_fault=True):
        state = twin.apply(change)
    diffs = replay_diffs(graded(state), incident["post_fault_kpis"], keys)
    if diffs:
        import platform

        raise ReplayMismatch(
            f"{incident['incident_id']}: replayed post-fault KPIs differ from the frozen suite "
            f"(twin backend {BACKEND}, platform {platform.system()}-{platform.machine()}; see "
            f"suite/incidents-meta.json for where it was frozen): " + "; ".join(diffs))
    ctx = {"twin": twin, "targets": targets, "run_id": run_id}
    if hasattr(op, "begin"):
        op.begin(incident, ctx)
    history: list[dict[str, Any]] = []
    latency_ms = 0.0
    texts = []
    for _ in range(ep["operator_steps"]):
        t0 = time.perf_counter()
        change = nonzero(op.act(twin.state, list(history)))
        latency_ms += (time.perf_counter() - t0) * 1000.0
        state = twin.apply(change)
        texts.append(canon(change) if change else "")
        history.append({"step": twin.step, "change": change, "kpis": graded(state)})
    if hasattr(op, "end"):
        op.end()
    step = twin.step
    assert step == incident["grade_step"], (step, incident["grade_step"])
    final = graded(state)
    g = grade_kpis(final, incident["healthy_kpis_by_step"][step], incident["unfixed_kpis_by_step"][step], targets)
    st = action_stats(twin.action_log[ep["fault_step"]:], set(incident["fault_cells"]))
    extra = {
        "operator": op.name,
        "incident": {k: incident[k] for k in ("scenario", "seed", "preset", "tier", "template", "fault",
                                              "fault_cells")},
        "recovery": g["recovery"],
        "success": g["success"],
        "per_kpi": g["per_kpi"],
        "final_kpis": final,
        "change_texts": texts,
        **st,
    }
    if hasattr(op, "mode"):
        extra["mode"] = op.mode
    if hasattr(op, "extra"):
        extra["operator_extra"] = op.extra()
    # A live operator (swarm) may report the KPIs its own twin measured; they must match ours.
    mismatch = []
    for i, tr in enumerate(getattr(op, "trace", []) or []):
        got = tr.get("measured_kpis") if isinstance(tr, dict) else None
        if got and i < len(history):
            bad = {k: [got[k], history[i]["kpis"][k]] for k in keys
                   if k in got and not math.isclose(got[k], history[i]["kpis"][k], rel_tol=REPLAY_TOL, abs_tol=1e-8)}
            if bad:
                mismatch.append({"step": history[i]["step"], "theirs_vs_ours": bad})
    if mismatch:
        extra["kpi_mismatch"] = mismatch
    return {
        "item_id": incident["incident_id"],
        "input_sha256": sha256_text(inc_mod.canonical_line(incident)),
        "output_sha256": sha256_text("\n".join(texts)),
        "score": g["recovery"],
        "passed": g["success"],
        "grader": GRADER_ID,
        "backend": BACKEND,
        "latency_ms": latency_ms,
        "extra": extra,
    }


def single_thread() -> None:
    """One torch thread per process: the GP twin is several times faster than with the default pool."""
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass


def _worker(args: tuple[dict[str, Any], str, dict[str, Any], str, dict[str, Any]]) -> dict[str, Any]:
    incident, name, targets, run_id, kw = args
    single_thread()
    return run_incident(incident, make(name, **kw), targets, run_id)


def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(items)
    rec = [it["score"] for it in items]
    ex = [it["extra"] for it in items]
    by: dict[str, Any] = {}
    groups = [("preset", v) for v in sorted({e["incident"]["preset"] for e in ex})]
    groups += [("tier", v) for v in sorted({e["incident"].get("tier", "visible") for e in ex})]
    for field, value in groups:
        sub = [it for it in items if it["extra"]["incident"].get(field, "visible") == value]
        by[value] = {
            "n": len(sub),
            "mean_recovery": sum(i["score"] for i in sub) / len(sub),
            "success_rate": sum(1 for i in sub if i["passed"]) / len(sub),
        }
    return {
        "n_items": n,
        "mean_recovery": sum(rec) / n,
        "success_rate": sum(1 for it in items if it["passed"]) / n,
        "success_count": sum(1 for it in items if it["passed"]),
        "mean_changes_used": sum(e["changes_used"] for e in ex) / n,
        "wrong_cell_actions_total": sum(e["wrong_cell_actions"] for e in ex),
        "items_with_wrong_cell_action": sum(1 for e in ex if e["wrong_cell_actions"]),
        "items_clamped": sum(1 for e in ex if e["clamped"]),
        "by_group": by,
    }


def gains_against(ev: Evidence, items: list[dict[str, Any]], baseline_ids: list[str],
                  boot: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    cand = {it["item_id"]: it["score"] for it in items}
    cand_s = {it["item_id"]: float(it["passed"]) for it in items}
    for bid in baseline_ids:
        base = ev.read_records(bid, "item")
        if not base:
            raise RuntimeError(f"baseline run {bid} has no item records in {ev.root}")
        decl = ev.read_declared(bid) or {}
        g = paired_bootstrap(cand, {b["item_id"]: b["score"] for b in base}, boot["resamples"], boot["seed"], boot["ci"])
        gs = paired_bootstrap(cand_s, {b["item_id"]: float(b["passed"]) for b in base}, boot["resamples"],
                              boot["seed"], boot["ci"])
        out.append({**g, "baseline_run_id": bid, "baseline_model_id": decl.get("model_id"),
                    "metric": "recovery", "success_rate_gain": gs})
    return out


CONFIG_REVERT_MODEL = "config-revert-median-v1"
RULE_MODEL_PREFIX = "ran_rl.evaluation.baseline:RuleBasedController"
SUBSETS = {"masked": "tier=masked"}


def _tier(item: dict[str, Any]) -> str:
    return item["extra"]["incident"].get("tier", "visible")


def baseline_by_model(ev: Evidence, baseline_ids: list[str], pred) -> tuple[str, list[dict[str, Any]]] | None:
    for bid in baseline_ids:
        if pred((ev.read_declared(bid) or {}).get("model_id", "")):
            return bid, ev.read_records(bid, "item")
    return None


def swarm_primary(ev: Evidence, items: list[dict[str, Any]], baseline_ids: list[str],
                  boot: dict[str, Any], tier: str = "masked") -> dict[str, Any]:
    """Paired recovery gain over config_revert on one tier, plus the declared secondary numbers."""
    found = baseline_by_model(ev, baseline_ids, lambda m: m == CONFIG_REVERT_MODEL)
    if found is None:
        raise SystemExit("the swarm target needs a config_revert baseline: pass --baseline-run-id <config_revert run>")
    bid, base = found
    mine = [it for it in items if _tier(it) == tier]
    g = paired_bootstrap({i["item_id"]: i["score"] for i in mine}, {b["item_id"]: b["score"] for b in base},
                         boot["resamples"], boot["seed"], boot["ci"])
    if g is None:
        raise SystemExit(f"no {tier} items shared with the config_revert run {bid}")
    ids = {i["item_id"] for i in mine}
    rule = baseline_by_model(ev, baseline_ids, lambda m: m.startswith(RULE_MODEL_PREFIX))
    secondary: dict[str, Any] = {
        "tier": tier,
        "success_rate": sum(1 for i in mine if i["passed"]) / len(mine),
        "wrong_cell_actions": sum(i["extra"]["wrong_cell_actions"] for i in mine),
    }
    if rule is None:
        secondary["rule_baseline"] = None
        secondary["wrong_cell_not_worse_than_rule"] = None
    else:
        rid, ritems = rule
        rw = sum(b["extra"]["wrong_cell_actions"] for b in ritems if b["item_id"] in ids)
        secondary.update(rule_baseline=rid, rule_baseline_wrong_cell_actions=rw,
                         wrong_cell_not_worse_than_rule=secondary["wrong_cell_actions"] <= rw)
    return {"gain": {**g, "baseline_run_id": bid, "baseline_model_id": CONFIG_REVERT_MODEL, "tier": tier},
            "secondary": secondary}


def compare(value: float, op: str, target: float) -> bool:
    return {">=": value >= target, "<=": value <= target, "==": value == target, ">": value > target,
            "<": value < target}[op]


# Fields the contract's DeclaredRecord accepts (W1 forbids unknown keys); everything else of ours goes
# into twin.config, which is a free-form object.
DECLARED_FIELDS = frozenset({
    "schema_version", "kind", "run_id", "team", "created_at", "aws_time", "aws_time_source", "synthetic",
    "unit", "track", "target", "baseline_run_id", "n_items", "n_cases", "test_set_sha256", "subset",
    "prompt_sha256", "parser", "decoding", "model_id", "grader", "code_sha", "region", "twin", "declared_at",
})
REGION = "us-east-2"  # the team's region (where this evidence is kept); the run itself is local and offline


def build_declared(run_id: str, operator: str, op: Any, unit: dict[str, Any], incidents: list[dict[str, Any]],
                   digest: str, baseline_run_ids: list[str], targets: dict[str, Any],
                   subset: str | None = None, suite_path: str = "t3/evals/suite/incidents-test.jsonl") -> dict[str, Any]:
    d = {
        "run_id": run_id,
        "synthetic": bool(getattr(op, "synthetic", False)),
        "unit": unit["unit"],
        "track": "track-3",
        "target": unit["target"],
        "n_items": len(incidents),
        "test_set_sha256": digest,
        "model_id": op.model_id,
        "grader": GRADER_ID,
        "code_sha": code_sha(),
        "region": REGION,
        "twin": {
            "kind": "maveric-inprocess",
            "version": twin_version(),
            "scenario": "incident-suite",
            "seed": min(i["seed"] for i in incidents),
            "config": {
                "backend": BACKEND,
                "suite": suite_path,
                "suite_version": inc_mod.SUITE_VERSION,
                "presets": sorted({i["preset"] for i in incidents}),
                "tiers": sorted({i.get("tier", "visible") for i in incidents}),
                "seeds": "one per incident (see the suite); twin.seed is the smallest",
                "eval": {
                    "operator": operator,
                    "targets_sha256": hashlib.sha256(TARGETS_PATH.read_bytes()).hexdigest(),
                    "episode": targets["episode"],
                    "executed": "local, offline, in-process twin (no AWS call)",
                    "evidence_writer": EVLIB,
                    "mode": getattr(op, "mode", None),
                    "proposal_steps": getattr(op, "proposal_steps", None) if getattr(op, "mode", None) == "proposal"
                    else None,
                },
            },
        },
    }
    if baseline_run_ids:
        d["baseline_run_id"] = baseline_run_ids[0]
    if subset:
        d["subset"] = SUBSETS[subset]
    unknown = set(d) - DECLARED_FIELDS
    if unknown:
        raise ValueError(f"declared carries keys the contract forbids: {sorted(unknown)}")
    return d


def run(operator: str, evidence_dir: Path, baseline_run_ids: list[str], workers: int = 1,
        allow_dirty: bool = False, op_kwargs: dict[str, Any] | None = None, run_id: str | None = None,
        subset: str | None = None, suite: str | Path | None = None) -> dict[str, Any]:
    single_thread()
    targets = load_targets()
    unit = targets["units"][operator]
    suite_file = Path(suite) if suite else inc_mod.TEST_PATH
    incidents, digest = inc_mod.load_test(suite_file)
    try:
        suite_label = suite_file.resolve().relative_to(inc_mod.SUITE_DIR.parents[2].resolve()).as_posix()
    except ValueError:
        suite_label = f"(outside the repo) {suite_file.name}"
    if subset is not None:
        if subset not in SUBSETS:
            raise SystemExit(f"unknown subset {subset!r}; known: {sorted(SUBSETS)}")
        incidents = [i for i in incidents if i.get("tier", "visible") == "masked"]
    if code_dirty() and not allow_dirty:
        raise SystemExit("t3/evals has uncommitted changes: commit first so code_sha is the code that runs "
                         "(or pass --allow-dirty for a dry run)")
    kw = dict(op_kwargs or {})
    op = make(operator, **kw)
    ev = Evidence(evidence_dir)
    run_id = run_id or new_run_id(f"t3.evals.{operator}")
    synthetic = bool(getattr(op, "synthetic", False))
    declared = build_declared(run_id, operator, op, unit, incidents, digest, baseline_run_ids, targets, subset,
                              suite_label)
    ev.write_declared(declared)
    t0 = time.time()
    if workers > 1 and operator != "swarm":
        with ProcessPoolExecutor(max_workers=workers) as pool:
            items = list(pool.map(_worker, [(i, operator, targets, run_id, kw) for i in incidents]))
    else:
        items = [run_incident(i, op, targets, run_id) for i in incidents]
    for it in items:
        it["synthetic"] = synthetic
    ev.append_records(run_id, "item", items)
    summary = summarize(items)
    gains = gains_against(ev, items, baseline_run_ids, targets["bootstrap"]) if baseline_run_ids else []
    metric = unit["target"]["metric"]
    if metric == "mean_score":
        value = summary["mean_recovery"]
    elif metric == "gain_ci_low_vs_no_action":
        na = [g for g in gains if g.get("baseline_model_id") == "no_action"]
        if not na:
            raise SystemExit(f"{operator}'s target needs a no_action baseline: pass --baseline-run-id <no_action run>")
        value = na[0]["ci_low"]
    elif metric == "gain_ci_low_vs_config_revert_masked":
        sp = swarm_primary(ev, items, baseline_run_ids, targets["bootstrap"])
        value = sp["gain"]["ci_low"]
    else:
        raise SystemExit(f"unknown target metric {metric}")
    primary = gains[0] if gains else None
    extra_details: dict[str, Any] = {"subset": SUBSETS.get(subset) if subset else None,
                                     "mode": getattr(op, "mode", None)}
    if metric == "gain_ci_low_vs_config_revert_masked":
        primary = sp["gain"]
        extra_details.update(primary_gain=sp["gain"], secondary=sp["secondary"])
    result = {
        "run_id": run_id,
        "synthetic": synthetic,
        "unit": unit["unit"],
        "metric": metric,
        "value": value,
        "target": unit["target"],
        "passed": compare(value, unit["target"]["op"], unit["target"]["value"]),
        "n_items": len(items),
        "checked_at": utc_now(),
        "checker": {"script": CHECKER, "code_sha": declared["code_sha"]},
        "details": {
            **summary,
            "operator": operator,
            "model_id": op.model_id,
            "test_set_sha256": digest,
            "wall_s": round(time.time() - t0, 3),
            "gain": None if primary is None else {k: primary[k] for k in
                                                  ("value", "ci_low", "ci_high", "baseline_run_id", "paired_n")},
            "gains": gains,
            **extra_details,
        },
    }
    ev.write_result(result)
    return result


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "gains":
        ap = argparse.ArgumentParser(prog="t3.evals.run gains")
        ap.add_argument("--run", required=True)
        ap.add_argument("--baseline", required=True, nargs="+")
        ap.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE)
        a = ap.parse_args(argv[1:])
        ev = Evidence(a.evidence_dir)
        print(json.dumps(gains_against(ev, ev.read_records(a.run, "item"), a.baseline,
                                       load_targets()["bootstrap"]), indent=2))
        return 0
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--operator", required=True, choices=["no_action", "rule_baseline", "config_revert", "ppo", "oracle", "swarm"])
    ap.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE)
    ap.add_argument("--baseline-run-id", nargs="*", default=[])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--ppo-path", default=None, help="W9 policy artifact for t1.rl.policy.load")
    ap.add_argument("--swarm-client", default=None, help="module:factory returning the live swarm client; "
                                                          "omitted = the dry-run fake (synthetic)")
    ap.add_argument("--mode", choices=["proposal", "full"], default="proposal",
                    help="swarm only. proposal: each step is one front-door case stopped at the approval refusal; "
                         "the proposed change is graded on our twin (no approvals). full: real approvals.")
    ap.add_argument("--proposal-steps", type=int, default=3, help="swarm proposal mode: cases per incident")
    ap.add_argument("--suite", type=Path, default=None,
                    help="incident file to score (default: the frozen t3/evals/suite/incidents-test.jsonl, whose "
                         "sha256 is checked against suite/incidents-test.sha256)")
    ap.add_argument("--subset", choices=sorted(SUBSETS), default=None,
                    help="masked = the masked-tier incidents of the frozen test split")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args(argv)
    res = run(a.operator, a.evidence_dir, a.baseline_run_id, a.workers, a.allow_dirty,
              {"ppo_path": a.ppo_path, "swarm_client": a.swarm_client, "swarm_mode": a.mode,
               "proposal_steps": a.proposal_steps}, a.run_id, a.subset, a.suite)
    d = res["details"]
    print(json.dumps({
        "run_id": res["run_id"], "synthetic": res["synthetic"], "metric": res["metric"], "value": res["value"],
        "target": res["target"], "passed": res["passed"], "n_items": res["n_items"],
        "mean_recovery": d["mean_recovery"], "success_rate": d["success_rate"],
        "mean_changes_used": d["mean_changes_used"], "wrong_cell_actions_total": d["wrong_cell_actions_total"],
        "items_clamped": d["items_clamped"], "by_group": d["by_group"],
        "gains": [{k: g[k] for k in ("baseline_run_id", "baseline_model_id", "value", "ci_low", "ci_high",
                                     "paired_n")} | {"success_rate_gain": g["success_rate_gain"]} for g in d["gains"]],
        "wall_s": d["wall_s"], "subset": d.get("subset"), "mode": d.get("mode"),
        "primary_gain": d.get("primary_gain"), "secondary": d.get("secondary"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
