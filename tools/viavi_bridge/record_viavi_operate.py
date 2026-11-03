#!/usr/bin/env python3
"""Record the Viavi operate run: one item per step of reports/operate.json, then the result. Local only.

    python -I tools/viavi_bridge/record_viavi_operate.py --run-id t1.rl.viavi_operate-... \\
        --evidence-dir <dir> --report <ran-rl-starter>/reports/operate.json [--dry-run]

Reads the declaration written by declare_viavi_operate.py (<evidence-dir>/declared/<run_id>.json), checks the
report is for the declared model (model_sha256), and writes, never overwriting:
  <evidence-dir>/records/<run_id>/item/<key>.json   (the S3 "objects" layout, one per step)
  <evidence-dir>/results/<run_id>.json
validated with contracts/ (team13_contracts) through tools/evlib.py. No AWS calls, no S3 writes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "contracts"), str(ROOT / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import evlib  # noqa: E402

SCRIPT = "tools/viavi_bridge/record_viavi_operate.py"
BACKEND = ("viavi RSG (ai-rsg 1.0) via organisers' adapter ran_rl_host.adapter.serve --backend viavi_rsg on the "
           "team's Mac, via tools/viavi_bridge/relay.py through the team's signed-in browser session")


class Refusal(SystemExit):
    pass


def canon_sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def finite(x):
    """JSON-safe copy: non-finite floats become null (the contracts reject NaN/inf)."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: finite(v) for k, v in x.items()}
    if isinstance(x, list):
        return [finite(v) for v in x]
    return x


def build(decl: dict, report: dict, report_sha: str) -> tuple[list[dict], dict]:
    """(item records, result details). Refuses a report for another model or with no steps."""
    want = decl["twin"]["config"]["model"]["artifact_sha256"]
    if report.get("model_sha256") != want:
        raise Refusal(f"refused: report model_sha256 {report.get('model_sha256')} != declared {want}")
    steps = report.get("steps") or []
    if not steps:
        raise Refusal("refused: the report has no steps (operate failed?): nothing to record")
    items = []
    for s in steps:
        s = finite(s)
        n = s["step"]
        inp = {"run_id": decl["run_id"], "step": n, "model_sha256": want, "scenario": decl["twin"]["scenario"],
               "seed": report.get("seed")}
        objective = s.get("objective")
        items.append({
            "run_id": decl["run_id"],
            "item_id": f"step-{n}",
            "input_sha256": canon_sha(inp),
            "output_sha256": canon_sha(s),
            "score": objective if isinstance(objective, (int, float)) and not isinstance(objective, bool) else None,
            "passed": None,
            "grader": "measure-only: step recorded from reports/operate.json (ran_rl.operate, SDK 1.2.9)",
            "backend": BACKEND,
            "synthetic": bool(decl.get("synthetic", False)),
            "extra": {
                "step": n,
                "requested_action": s.get("raw_action"),
                "filtered_action": s.get("filtered_action"),
                "safety_rules_fired": s.get("rule_ids"),
                "objective": objective,
                "kpis": s.get("kpis"),
                "input_rule": "input_sha256 = sha256(canonical JSON of {run_id, step, model_sha256, scenario, seed})",
                "output_rule": "output_sha256 = sha256(canonical JSON of this step's entry in reports/operate.json)",
            },
        })
    details = {
        "report": "ran-rl-starter/reports/operate.json",
        "report_sha256": report_sha,
        "twin_class": report.get("twin"),
        "use_case": report.get("use_case"),
        "model_sha256": report.get("model_sha256"),
        "episode": finite(report.get("episode") or {}),
        "steps_completed": len(steps),
        "measure_only": "the target counts completed steps; no KPI threshold was declared",
        "aws": "no AWS job: the run went through the team's browser session; check_unit part c has no AWS window",
        "code_note": "steps from the organisers' SDK (ran_rl.operate); this wrapper is tools/viavi_bridge",
    }
    return items, details


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--report", required=True, help="ran-rl-starter/reports/operate.json")
    ap.add_argument("--evidence-dir", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    store = evlib.LocalStore(a.evidence_dir)
    decl = evlib.read_declared(a.run_id, store)
    if decl is None:
        raise Refusal(f"refused: no declaration {a.evidence_dir}/declared/{a.run_id}.json")
    raw = Path(a.report).read_bytes()
    items, details = build(decl, json.loads(raw), hashlib.sha256(raw).hexdigest())
    result = evlib.make_result(decl, items, script=SCRIPT, sha=decl["code_sha"], details=details)
    if a.dry_run:
        for it in items:
            evlib.validate(evlib._envelope(it, "item", a.run_id))
        evlib.validate(evlib._envelope(result, "result", a.run_id))
        print(f"dry run: {len(items)} items valid; result value={result['value']} passed={result['passed']}")
        return 0
    paths = evlib.append_records(a.run_id, "item", items, store=store, layout="objects")
    rpath = evlib.write_result(result, store=store)
    print(f"items -> {len(paths)} files under {a.evidence_dir}/records/{a.run_id}/item/")
    print(f"result -> {rpath}  value={result['value']} passed={result['passed']}")
    print(f"upload: aws s3 cp --profile fyuz --region us-east-2 --recursive {a.evidence_dir}/records/{a.run_id}/ "
          f"s3://<team bucket>/evidence/records/{a.run_id}/ && aws s3api put-object --profile fyuz "
          f"--region us-east-2 --bucket <team bucket> --key evidence/results/{a.run_id}.json --body {rpath} "
          f"--if-none-match '*'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
