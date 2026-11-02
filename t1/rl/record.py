"""Write the PPO `model` record. v0, local, not federated / not the official job.

    python -m t1.rl.record --model t1/rl/artifacts/ppo_v0.zip --train-log <train_log.json> \
        --compare t1/rl/models/compare_v0.json --out t1/rl/models/t1-ppo-v0.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

LABEL = "v0, local, not federated / not the official job"


def build(model: Path, train_log: dict, compare: dict) -> dict:
    s = compare["summary"]
    return {
        "schema_version": "1", "kind": "model", "run_id": "w9-models-v0-local", "team": "team13",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "aws_time": None, "aws_time_source": None, "synthetic": False,
        "model_id": "t1-ppo-v0", "version": "v0", "model_kind": "ppo",
        "artifact_uri": f"t1/rl/artifacts/{model.name}",
        "artifact_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "status": "candidate",
        "trained_on": f"{LABEL}. PPO (stable-baselines3) on the in-process Maveric twin through ran_rl.make_env "
                      f"(CCO use case, SDK Default Reward), scenarios {', '.join(train_log['scenarios'])}; "
                      f"{train_log['timesteps']} steps in {train_log['wall_seconds']} s ({train_log['stopped_by']})",
        "metrics": {
            "label": LABEL,
            "training": {k: train_log[k] for k in ("timesteps", "wall_seconds", "stopped_by", "n_envs", "seed")},
            "final_ep_rew_mean": train_log["curve"][-1]["ep_rew_mean"] if train_log["curve"] else None,
            "comparison_n_episodes": s["overall"]["ppo_v0"]["n_episodes"],
            "ppo_v0": s["overall"]["ppo_v0"], "rule_baseline": s["overall"]["rule_baseline"], "no_op": s["overall"]["no_op"],
            "ppo_minus_rule": s["ppo_minus_rule"], "uplift_ratio_vs_rule": s["uplift_ratio_vs_rule"],
            "sdk_uplift_gate_1.05": s["sdk_uplift_gate_1.05"],
            "note": "SDK Scoring Objective summed over 24-step episodes; coverage_mean reported because the objective "
                    "does not reward it",
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--train-log", required=True)
    ap.add_argument("--compare", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rec = build(Path(a.model), json.loads(Path(a.train_log).read_text()), json.loads(Path(a.compare).read_text()))
    Path(a.out).write_text(json.dumps(rec, indent=2) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
