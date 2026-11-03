"""Platform check and re-freeze for the execution items: CPU, no model, run as a SageMaker job.

The execution items were frozen on a Mac (Darwin arm64) with W2's twin; every real eval runs in an
x86_64 job, and the grader stops with TwinDrift if a replay does not reproduce an item's frozen values
within graders.DRIFT_TOL (1e-9). These two jobs settle that on the job platform.

    check     replay all 80 items (dev and test) and compare the replayed noop_kpi, rule_gain,
              oracle_gain and constant_gain with the frozen ones. One item record per item
              (score = largest |difference|, passed = within DRIFT_TOL) and a result
              (pass_rate == 1.0 means no drift anywhere).
    refreeze  only if `check` drifts: rebuild every item from its own spec (same ids, scenario, seed,
              prefix and candidates) on this platform and write execution.jsonl, a regenerated
              price_latency.jsonl (it copies execution prompts) and their hashes to --output-dir.
              One item record per item (what changed) and a result. Commit those files, then run
              `python -m t1.evals.freeze --units` to rewrite MANIFEST.json.

Under the runner (tools/sm), print the declaration first and launch with it, so the module's own
declaration matches the one stored before the job:

    python -m t1.evals.replay_job check --print-declaration > /tmp/t1-replay-decl.json
    python -m tools.sm.run --run-id <id> --declaration /tmp/t1-replay-decl.json --instance-type ml.m5.xlarge \
        --entry "-m t1.evals.replay_job check --run-id {run_id} --evidence-dir {evidence_dir}" [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

from t1.evals import common, freeze
from t1.evals.common import dumps, lines_sha256
from t1.evals.graders import DRIFT_TOL

UNIT = {"check": "t1.execution.replay-check", "refreeze": "t1.execution.refreeze"}
KEYS = ("noop_kpi", "rule_gain", "oracle_gain", "constant_gain")
GRADER = {
    "check": f"t1/evals/replay_job.py check: |replayed - frozen| <= {DRIFT_TOL:g} on " + ", ".join(KEYS),
    "refreeze": "t1/evals/replay_job.py refreeze: every execution item rebuilt from its own spec on this platform",
}


def platform_info() -> dict[str, Any]:
    def ver(p: str) -> str | None:
        try:
            return md.version(p)
        except md.PackageNotFoundError:
            return None

    return {"system": platform.system(), "machine": platform.machine(), "python": platform.python_version(),
            "versions": {p: ver(p) for p in ("ran-rl", "torch", "numpy", "gpytorch", "linear-operator")}}


def declaration(mode: str) -> dict[str, Any]:
    """The judged fields, identical for --print-declaration (launcher) and the module's own declaration."""
    from t1.evals.gen_execution import SEED_BASE
    from t1.evals.twin_access import TWIN_SOURCE

    items, sha = freeze.select("execution", "all")
    target = ({"metric": "pass_rate", "op": "==", "value": 1.0, "unit": "ratio"} if mode == "check"
              else {"metric": "n_items", "op": "==", "value": float(len(items)), "unit": "count"})
    return {
        "unit": UNIT[mode], "track": "track-1", "target": target, "n_items": len(items), "test_set_sha256": sha,
        "subset": "split=all", "model_id": "none (twin replay only)", "grader": GRADER[mode], "synthetic": False,
        "twin": {"kind": "maveric-inprocess", "version": "ran-rl 1.2.9", "scenario": "per-item", "seed": SEED_BASE,
                 "config": {"preset": "official", "backend": TWIN_SOURCE,
                            "per_item": "t1/evals/items/execution.jsonl meta.scenario/seed/prefix"}},
    }


def _replayed(it: dict) -> dict[str, float]:
    from t1.evals.twin_access import case, outcome

    m, e = it["meta"], it["expected"]
    tw = case(m["scenario"], m["seed"], m["prefix"])
    return {"noop_kpi": outcome(tw, "")["noop"], "rule_gain": outcome(tw, e["rule_change"])["gain"],
            "oracle_gain": outcome(tw, e["oracle_change"])["gain"],
            "constant_gain": outcome(tw, e["constant_change"])["gain"]}


def check(ev, run_id: str) -> dict:
    lines = freeze.read_lines("execution")
    recs, worst = [], {k: 0.0 for k in KEYS}
    for line in lines:
        it = json.loads(line)
        t0 = time.perf_counter()
        got = _replayed(it)
        ms = (time.perf_counter() - t0) * 1000
        diff = {k: abs(got[k] - it["expected"][k]) for k in KEYS}
        for k in KEYS:
            worst[k] = max(worst[k], diff[k])
        out = {"replayed": got}
        recs.append({
            "item_id": it["item_id"], "input_sha256": common.sha256_text(line),
            "output_sha256": common.sha256_text(dumps(out)), "score": max(diff.values()),
            "passed": max(diff.values()) <= DRIFT_TOL, "grader": "t1.evals.replay_job:check",
            "backend": f"twin.replay on {platform.system()}-{platform.machine()}", "latency_ms": round(ms, 3),
            "extra": {"split": it["split"], "family": it["meta"]["family"],
                      "frozen": {k: it["expected"][k] for k in KEYS}, "replayed": got, "abs_diff": diff},
        })
    ev.append_records(run_id, "item", recs)
    drifted = [r["item_id"] for r in recs if not r["passed"]]
    value = 1 - len(drifted) / len(recs)
    return {"value": value, "details": {
        "n_items": len(recs), "n_drifted": len(drifted), "drifted": drifted[:20], "drift_tol": DRIFT_TOL,
        "max_abs_diff": max(worst.values()), "max_abs_diff_by_value": worst,
        "n_drifted_by_split": {s: sum(1 for r in recs if not r["passed"] and r["extra"]["split"] == s)
                               for s in ("dev", "test")},
        "platform": platform_info(), "verdict": "no drift: keep the frozen items" if not drifted
        else "drift: run `python -m t1.evals.replay_job refreeze` on this platform"}}


def refreeze(ev, run_id: str, out_dir: Path) -> dict:
    from t1.evals import gen_execution as E
    from t1.evals import gen_price_latency as P

    old_lines = freeze.read_lines("execution")
    new_items, recs = [], []
    for line in old_lines:
        it = json.loads(line)
        m = it["meta"]
        spec = {"scenario": m["scenario"], "seed": m["seed"], "prefix": m["prefix"], "fault": m["fault"]}
        if E._spec(m["family"], m["gen_index"]) != spec:
            raise SystemExit(f"{it['item_id']}: its spec no longer regenerates from gen_index {m['gen_index']}")
        t0 = time.perf_counter()
        ref = E.reference(spec, apply_filter=False)
        n = int(it["item_id"].rsplit("-", 1)[1])
        new = E.make_item(m["family"], n, spec, ref, m["gen_index"])
        ms = (time.perf_counter() - t0) * 1000
        new_items.append(new)
        diff = {k: abs(new["expected"][k] - it["expected"][k]) for k in KEYS}
        recs.append({
            "item_id": it["item_id"], "input_sha256": common.sha256_text(line),
            "output_sha256": common.sha256_text(dumps(new)), "score": max(diff.values()), "passed": True,
            "grader": "t1.evals.replay_job:refreeze", "backend": f"twin.replay on {platform.system()}-{platform.machine()}",
            "latency_ms": round(ms, 3),
            "extra": {"split": it["split"], "changed": dumps(new) != line, "state_text_changed":
                      new["input"] != it["input"], "changes_changed": any(
                          new["expected"][k] != it["expected"][k] for k in ("oracle_change", "rule_change")),
                      "kept_by_filter_here": ref["kept_by_filter"], "abs_diff": diff},
        })
    ev.append_records(run_id, "item", recs)
    out_dir.mkdir(parents=True, exist_ok=True)
    ex_lines = [dumps(it) for it in new_items]
    (out_dir / "execution.jsonl").write_text("".join(x + "\n" for x in ex_lines), encoding="utf-8")
    src = {u: freeze.load_items(u) for u in P.MIX if u != "execution"}
    src["execution"] = new_items
    pl_lines = [dumps(it) for it in P.generate(src)]
    (out_dir / "price_latency.jsonl").write_text("".join(x + "\n" for x in pl_lines), encoding="utf-8")

    def split_sha(lines: list[str], split: str) -> dict:
        sel = [x for x in lines if json.loads(x)["split"] == split]
        return {"n": len(sel), "sha256": lines_sha256(sel)}

    hashes = {u: {"sha256": lines_sha256(ls), "n_items": len(ls), "dev": split_sha(ls, "dev"),
                  "test": split_sha(ls, "test"), "previous_sha256": lines_sha256(freeze.read_lines(u))}
              for u, ls in (("execution", ex_lines), ("price_latency", pl_lines))}
    summary = {"run_id": run_id, "platform": platform_info(), "hashes": hashes,
               "n_changed": sum(r["extra"]["changed"] for r in recs),
               "n_state_text_changed": sum(r["extra"]["state_text_changed"] for r in recs),
               "n_changes_changed": sum(r["extra"]["changes_changed"] for r in recs),
               "n_not_kept_by_filter_here": sum(not r["extra"]["kept_by_filter_here"] for r in recs),
               "next": "copy execution.jsonl and price_latency.jsonl into t1/evals/items/, run "
                       "`python -m t1.evals.freeze --units` (rewrites MANIFEST.json), add the old hashes to "
                       "freeze.SUPERSEDED, commit; declarations then carry the new test_set_sha256"}
    (out_dir / "refreeze-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return {"value": float(len(recs)), "details": summary}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["check", "refreeze"])
    ap.add_argument("--print-declaration", action="store_true", help="print the judged fields for tools/sm/run.py")
    ap.add_argument("--run-id")
    ap.add_argument("--evidence-dir", help="local evlib store ({evidence_dir} under the runner)")
    ap.add_argument("--output-dir", type=Path, help="refreeze: where the new item files go ({output_dir})")
    ap.add_argument("--region", default="us-east-2")
    ap.add_argument("--code-sha")
    a = ap.parse_args(argv)
    decl = declaration(a.mode)
    if a.print_declaration:
        print(json.dumps(decl, indent=2, sort_keys=True))
        return 0
    if not (a.run_id and a.evidence_dir):
        ap.error("--run-id and --evidence-dir are required")
    if a.mode == "refreeze" and not a.output_dir:
        ap.error("refreeze needs --output-dir")
    from t1.evals.evidence import Evidence, utc_now
    from t1.evals.run import resolve_code_sha

    sha = resolve_code_sha(a.code_sha)
    ev = Evidence(a.evidence_dir)
    ev.write_declared({**decl, "run_id": a.run_id, "code_sha": sha, "region": a.region})
    out = check(ev, a.run_id) if a.mode == "check" else refreeze(ev, a.run_id, a.output_dir)
    t = decl["target"]
    value = out["value"]
    passed = {"==": value == t["value"], ">=": value >= t["value"]}[t["op"]]
    ev.write_result({"run_id": a.run_id, "unit": decl["unit"], "metric": t["metric"], "value": value, "target": t,
                     "passed": passed, "n_items": decl["n_items"], "checked_at": utc_now(),
                     "checker": {"script": "t1/evals/replay_job.py", "code_sha": sha}, "details": out["details"],
                     "synthetic": False})
    d = out["details"]
    print(json.dumps({k: d.get(k) for k in ("n_items", "n_drifted", "max_abs_diff", "max_abs_diff_by_value",
                                             "n_changed", "verdict", "platform")} | {"passed": passed}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
