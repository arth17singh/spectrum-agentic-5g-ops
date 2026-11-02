"""PPO v0 vs the SDK's rule baseline vs a no-op, on the same scenario x seed list, through the SDK's own
`run_episode` and fixed Scoring Objective. v0, local, not federated / not the official job.

    python -m t1.rl.compare --model t1/rl/artifacts/ppo_v0.zip --out t1/rl/models/compare_v0.json

Episodes: the SDK's published eval set (nominal 101-105, congestion 200-204, interference 300-304,
outage 400-404) plus low_load 500-504 and mobility_stress 600-604; 24 steps each, in-process Maveric
twin. The PPO policy runs through `Policy.action`, i.e. the quantised, safety-filtered proposal that
`propose` returns, so this measures what the tool would actually apply.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

LABEL = "v0, local, not federated / not the official job"
EVAL_SET = [("nominal", 101), ("congestion", 200), ("interference", 300), ("outage", 400),
            ("low_load", 500), ("mobility_stress", 600)]
N_SEEDS = 5


def run(policy_path: str) -> dict[str, Any]:
    import ran_rl
    from ran_rl.env.config import EnvConfig
    from ran_rl.evaluation.baseline import RuleBasedController
    from ran_rl.evaluation.evaluate import run_episode

    from t1.rl.policy import load

    env = ran_rl.make_env(twin="maveric", env_config=EnvConfig())
    pol = load(policy_path)
    rule = RuleBasedController(env._cfg.safety)
    n = env._cfg.n_cells
    policies = {
        "ppo_v0": lambda obs, raw: pol.action(raw),
        "rule_baseline": lambda obs, raw: rule.act(raw),
        "no_op": lambda obs, raw: np.zeros(2 * n),
    }
    rows: list[dict[str, Any]] = []
    for scenario, base in EVAL_SET:
        for seed in range(base, base + N_SEEDS):
            for name, fn in policies.items():
                r = run_episode(env, fn, scenario, seed)
                rows.append({"policy": name, "scenario": scenario, "seed": seed,
                             "total_objective": r["total_objective"], "total_reward": r["total_reward"],
                             "sla_ratio_mean": r["sla_ratio_mean"], "coverage_mean": r["coverage_mean"]})
    env.close()
    return {"label": LABEL, "rows": rows}


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pols = sorted({r["policy"] for r in rows})
    keys = ("total_objective", "sla_ratio_mean", "coverage_mean")
    out: dict[str, Any] = {"overall": {}, "by_scenario": {}}
    for p in pols:
        sel = [r for r in rows if r["policy"] == p]
        out["overall"][p] = {k: float(np.mean([r[k] for r in sel])) for k in keys} | {"n_episodes": len(sel)}
    for sc in sorted({r["scenario"] for r in rows}):
        out["by_scenario"][sc] = {p: {k: float(np.mean([r[k] for r in rows if r["policy"] == p and r["scenario"] == sc]))
                                      for k in keys} for p in pols}
    ppo = {(r["scenario"], r["seed"]): r for r in rows if r["policy"] == "ppo_v0"}
    base = {(r["scenario"], r["seed"]): r for r in rows if r["policy"] == "rule_baseline"}
    diff = np.array([ppo[k]["total_objective"] - base[k]["total_objective"] for k in ppo])
    out["ppo_minus_rule"] = {"mean_objective_difference": float(diff.mean()), "ppo_wins": int((diff > 0).sum()),
                             "rule_wins": int((diff < 0).sum()), "n": len(diff),
                             "std_error": float(diff.std(ddof=1) / np.sqrt(len(diff)))}
    ratio = out["overall"]["ppo_v0"]["total_objective"] / out["overall"]["rule_baseline"]["total_objective"]
    out["uplift_ratio_vs_rule"] = float(ratio)
    out["sdk_uplift_gate_1.05"] = bool(ratio >= 1.05)
    out["sdk_service_gate_sla_within_0.02_of_rule"] = bool(
        out["overall"]["ppo_v0"]["sla_ratio_mean"] >= out["overall"]["rule_baseline"]["sla_ratio_mean"] - 0.02)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = run(a.model)
    res["summary"] = summarise(res["rows"])
    Path(a.out).write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res["summary"]["overall"], indent=1))
    print(json.dumps({k: v for k, v in res["summary"].items() if k not in ("overall", "by_scenario")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
