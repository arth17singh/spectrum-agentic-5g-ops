"""Generate v2_gate_golden.json: what the GATE's v2 operator proposes, step by step, on dev incidents.

The gate evaluation (t3.evals.remediation-policy-v2-20261104T113942Z) played remediation-v2 through
``t1.rl.remediation.v2.BaselineAware`` inside ``t3.evals.run.run_incident``. This script runs THAT code, from a
checkout of the gated branch (feat/remediation-v2 at 3caba77; not on this deploy branch), with the same loop as
run_incident (reset, the incident script with the fault, ``op.begin(incident, ctx)``, then per operator step
``op.act(twin.state, history)`` and ``twin.apply``), minus run_incident's frozen-KPI check, which only holds on the
x86 host the suite was frozen on. Dev incidents only (t3/evals/suite/incidents-dev.jsonl), never the held-out gate.

    git archive 3caba77 | tar -x -C <export>
    PYTHONPATH=<export>:<export>/contracts <venv>/bin/python t3/tools/tests/fixtures/gen_v2_gate_golden.py <export>

t3/tools/tests/test_propose_baseline.py then asks ran___propose (this branch's app, same twin, same artifact) for
the same incidents and steps (each step seeded with the earlier steps in its start log, as the harness's live
runners do) and asserts the same change.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

N = 8
OUT = Path(__file__).with_name("v2_gate_golden.json")


def main(export: Path) -> None:
    from t1.rl.policy import load
    from t1.rl.remediation.v2 import BaselineAware
    from t3.evals import incidents as inc
    from t3.evals.changes import canon, nonzero
    from twin.replay import ReplayTwin

    targets = json.loads((export / "t3/evals/targets.json").read_text())
    rows = [json.loads(line) for line in (export / "t3/evals/suite/incidents-dev.jsonl").read_text().splitlines()]
    zp = export / "t1/rl/artifacts/remediation_v2.zip"
    policy = load(zp)
    sha = subprocess.run(["git", "rev-parse", "3caba77"], capture_output=True, text=True,
                         cwd=Path(__file__).resolve().parent).stdout.strip()
    out = {"gate_code_commit": sha, "artifact": "t1/rl/artifacts/remediation_v2.zip",
           "artifact_sha256": hashlib.sha256(zp.read_bytes()).hexdigest(),
           "model": f"{policy.model_id}@{policy.version}", "operator": "t1.rl.remediation.v2.BaselineAware",
           "procedure": __doc__.strip().splitlines()[0], "cases": []}
    for spec in rows[:N]:
        op = BaselineAware(policy)
        twin = ReplayTwin(spec["scenario"], spec["seed"], spec["preset"])
        twin.reset()
        script = inc.script(spec, targets, with_fault=True)
        for change in script:
            twin.apply(change)
        op.begin(spec, {"twin": twin, "targets": targets, "run_id": "golden"})
        history, texts = [], []
        for _ in range(targets["episode"]["operator_steps"]):
            change = nonzero(op.act(twin.state, list(history)))
            state = twin.apply(change)
            texts.append(canon(change) if change else "")
            history.append({"step": twin.step, "change": change})
        out["cases"].append({"incident_id": spec["incident_id"], "scenario": spec["scenario"], "seed": spec["seed"],
                             "preset": spec["preset"], "fault_cells": spec["fault_cells"],
                             "start_log": [canon(s) if s else [] for s in script], "step_texts": texts})
        del state
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"wrote {OUT} ({len(out['cases'])} incidents)")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
