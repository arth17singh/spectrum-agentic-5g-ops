#!/usr/bin/env python3
"""Declare the Viavi operate run (unit t1.rl.viavi_operate) BEFORE it runs. Local file only.

    python -I tools/viavi_bridge/declare_viavi_operate.py --staging <ran-rl-staging dir> \\
        --evidence-dir <dir> [--run-id t1.rl.viavi_operate-<UTC>-<hex>] [--model t1/rl/artifacts/ppo_sm_50k.zip]

Writes <evidence-dir>/declared/<run_id>.json (never overwrites), validated with contracts/
(team13_contracts) through tools/evlib.py, and prints the run id and the S3 upload command for the lead.
No AWS calls, no S3 writes.

Target is measure-only: n_items == 5, i.e. all five operate steps completed and recorded (one item per step
of reports/operate.json). The run goes through the team's signed-in browser session (relay.py -> the
organisers' Viavi adapter on the team's Mac); there is no AWS job, so check_unit part c has no AWS window.
No RSG host, URL, IP, cookie or token is recorded.

--staging is the organisers' ran-rl staging folder (twin-source/ and ran-rl-starter/); it is outside git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "contracts"), str(ROOT / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import evlib  # noqa: E402

UNIT = "t1.rl.viavi_operate"
N_STEPS = 5
MODEL = ROOT / "t1" / "rl" / "artifacts" / "ppo_sm_50k.zip"
PROVENANCE = ROOT / "t1" / "rl" / "artifacts" / "ppo_sm_50k.provenance.json"
SCENARIO_REL = Path("twin-source/config/viavi_scenarios/maveric_4cell.json")
RSG_CONFIG_REL = Path("twin-source/config/viavi_rsg.yaml")
STARTER_REL = Path("ran-rl-starter")


class Refusal(SystemExit):
    pass


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return Path(p).name


def build(run_id: str, model: Path, staging: Path, provenance: Path = PROVENANCE,
          code_sha: str | None = None) -> dict:
    """The declaration (before evlib's envelope). Refuses a model whose sha256 is not the provenance's."""
    prov = json.loads(Path(provenance).read_text())
    model_sha = sha(model)
    if model_sha != prov["artifact_sha256"]:
        raise Refusal(f"refused: {model} sha256 {model_sha} != provenance {prov['artifact_sha256']}")
    starter = staging / STARTER_REL
    return {
        "run_id": run_id,
        "unit": UNIT,
        "track": "track-1",
        "target": {"metric": "n_items", "op": "==", "value": N_STEPS, "unit": "count"},
        "n_items": N_STEPS,
        "test_set_sha256": sha(staging / SCENARIO_REL),
        "model_id": prov["model_id"],
        "grader": (
            "measure-only: ran_rl.operate (RAN RL SDK 1.2.9, organisers' code, `make operate`) plays one 5-step "
            "episode of the PPO policy on the team's Viavi RSG twin; tools/viavi_bridge/record_viavi_operate.py "
            "turns reports/operate.json into one item per step (requested action, action after the SDK safety "
            "filter, rules fired, objective, KPIs) and counts the steps; no KPI gate"
        ),
        "code_sha": code_sha or evlib.code_sha(ROOT),
        "region": evlib.DEFAULT_REGION,
        "synthetic": False,
        "twin": {
            "kind": "viavi",
            "version": "Viavi RSG ai-rsg 1.0 via ran_rl_host viavi_rsg adapter (SDK 1.2.9)",
            "scenario": "ran-rl-maveric-4cell",
            "seed": 0,
            "config": {
                "purpose": "Track 1 RL stage 6 (operate): the served PPO policy acts on the Viavi RSG, a stand-in "
                           "for a production network it was never trained on (trained and scored on Maveric)",
                "route": "team's signed-in browser session: tools/viavi_bridge/relay.py (same-origin fetch to the "
                         "RSG's /mcp in the Viavi app tab; CloudFront 502s retried per its README, advance/start "
                         "never applied twice) -> organisers' adapter (python -m ran_rl_host.adapter.serve "
                         "--backend viavi_rsg) on the team's Mac -> make operate (RAN_RL_VIAVI_TWIN_URL). "
                         "RSG host, URL, IPs, cookies and tokens are not recorded",
                "aws": "no AWS job and no AWS-written log: check_unit part c has no AWS window for this run "
                       "(expected CANNOT CHECK, not a pass); region is the team's default, required by the schema",
                "model": {
                    "model_id": prov["model_id"],
                    "version": prov.get("version"),
                    "alias": "same artifact as ppo-cco-reward-v1-ckpt50k in t1.rl.eval-20261103T043115Z-bf52",
                    "artifact": prov.get("artifact", rel(model)),
                    "artifact_sha256": model_sha,
                    "provenance": rel(provenance),
                    "trained_by_run": (prov.get("training") or {}).get("run_id"),
                    "evaluated_by_run": (prov.get("evaluation") or {}).get("run_id"),
                },
                "scenario_file": "twin-source/config/viavi_scenarios/maveric_4cell.json (organisers' twin-host "
                                 "source 1.2.9; saved on the RSG by make create-viavi-scenario); its sha256 is "
                                 "test_set_sha256",
                "rsg_config_sha256": sha(staging / RSG_CONFIG_REL),
                "env_config_sha256": sha(starter / "config" / "env_viavi.yaml"),
                "reward_config_sha256": sha(starter / "config" / "reward.yaml"),
                "sdk_wheel_sha256": sha(starter / "ran_rl-1.2.9-py3-none-any.whl"),
                "episode": {"steps": N_STEPS, "step_seconds": 60.0, "use_case": "cco", "seed": 0,
                            "policy": "deterministic (ran_rl.evaluation.evaluate.load_policy)"},
                "items_rule": "one item per entry of reports/operate.json steps[]; item_id step-<n>; "
                              "value = number of items; passed iff value == 5",
            },
        },
    }


def new_run_id() -> str:
    return f"{UNIT}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{secrets.token_hex(2)}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--staging", default=os.environ.get("RAN_RL_STAGING"),
                    help="organisers' ran-rl staging folder (or RAN_RL_STAGING)")
    ap.add_argument("--evidence-dir", required=True)
    ap.add_argument("--run-id")
    ap.add_argument("--model", default=str(MODEL))
    ap.add_argument("--dry-run", action="store_true", help="validate and print, write nothing")
    a = ap.parse_args(argv)
    if not a.staging:
        raise Refusal("--staging (or RAN_RL_STAGING) is required")
    run_id = a.run_id or new_run_id()
    decl = build(run_id, Path(a.model), Path(a.staging))
    if a.dry_run:
        rec = evlib._envelope(decl, "declared", run_id)
        rec["declared_at"] = evlib.utc_now()
        evlib.validate(rec)
        print(json.dumps(rec, indent=1))
        print("dry run: valid, nothing written")
        return 0
    path = evlib.write_declared(decl, evlib.LocalStore(a.evidence_dir))
    print(f"run_id={run_id}")
    print(f"declared -> {path}")
    print(f"upload (BEFORE create-viavi-scenario/operate): aws s3api put-object --profile fyuz "
          f"--region us-east-2 --bucket <team bucket> --key evidence/declared/{run_id}.json --body {path} "
          f"--if-none-match '*'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
