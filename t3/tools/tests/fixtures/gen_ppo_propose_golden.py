"""Generate ppo_propose_golden.json: what ran___propose answers with ppo_sm_50k on real twin states.

Run ONCE at the deployed commit 614fc3b (the policy module as served today), BEFORE t1/rl/policy.py is replaced:

    PYTHONPATH=.:contracts <venv>/bin/python t3/tools/tests/fixtures/gen_ppo_propose_golden.py

The test t3/tools/tests/test_propose_baseline.py replays the same calls with the new policy module and asserts the
answers are byte-identical (ppo_sm_50k must serve exactly as before). Dev incidents only (incidents-dev.jsonl):
operator step 0 (start log = commissioning + fault) and operator step 1 (plus the policy's own first change).
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT), str(ROOT / "contracts")]

from t3.evals import incidents as inc  # noqa: E402
from t3.tools import app, store  # noqa: E402
from t3.tools.changes import canon  # noqa: E402

N = 8
OUT = Path(__file__).with_name("ppo_propose_golden.json")


def deps(tmp: str, policy_path: Path) -> app.Deps:
    from t1.rl.policy import load
    from twin.kpis import kpis as twin_kpis
    from twin.replay import ReplayTwin

    pol = load(policy_path)
    return app.Deps(store=store.from_env({"T3_STORE": f"local:{tmp}"}),
                    twin_factory=lambda s, seed, cfg: ReplayTwin(s, seed, cfg), kpis=twin_kpis,
                    policy=lambda: pol, synthetic=True, now=lambda: "2026-11-04T00:00:00.000Z")


def start_log(spec: dict, targets: dict) -> list:
    return [canon(s) if s else [] for s in inc.script(spec, targets, with_fault=True)]


def main() -> None:
    targets = json.loads((ROOT / "t3/evals/targets.json").read_text())
    rows = [json.loads(line) for line in (ROOT / "t3/evals/suite/incidents-dev.jsonl").read_text().splitlines()][:N]
    sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    out = {"generated_at_commit": sha, "policy": "t1/rl/artifacts/ppo_sm_50k.zip",
           "procedure": __doc__.strip().splitlines()[0], "cases": []}
    with tempfile.TemporaryDirectory() as tmp:
        tools = app.Tools(deps(tmp, ROOT / "t1/rl/artifacts/ppo_sm_50k.zip"))
        for i, spec in enumerate(rows):
            log0 = start_log(spec, targets)
            a0 = tools.seed_case({"case_id": f"case-golden-{i:02d}s0", "scenario": spec["scenario"],
                                  "seed": spec["seed"], "preset": spec["preset"],
                                  "start_action_log_json": json.dumps(log0)})
            p0 = tools.propose({"case_id": f"case-golden-{i:02d}s0"})
            log1 = log0 + ([p0["change"]] if p0.get("change") else [[]])
            tools.seed_case({"case_id": f"case-golden-{i:02d}s1", "scenario": spec["scenario"], "seed": spec["seed"],
                             "preset": spec["preset"], "start_action_log_json": json.dumps(log1)})
            p1 = tools.propose({"case_id": f"case-golden-{i:02d}s1"})
            out["cases"].append({"incident_id": spec["incident_id"], "scenario": spec["scenario"], "seed": spec["seed"],
                                 "preset": spec["preset"], "start_log": log0, "seeded_step": a0["step"],
                                 "answers": [p0, p1]})
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"wrote {OUT} ({len(out['cases'])} incidents, commit {sha[:12]})")


if __name__ == "__main__":
    main()
