"""Smoke test of the agents' brains: a 3-turn conversation with noc_lead through the checkpoint's front door.

Turn 1: a real request. The lead should ask its students, call apply_change and stop with an approval id.
Turn 2: after a PERSON approves that id (`checkpoint_kit.py approve chg-... "why"`), "Approved.". The lead
        should apply the same change and have ran_monitor verify it.
Turn 3: a follow-up question that needs a student again.
This is a smoke test, not a unit: it writes no evidence. Dry run by default (prints the calls it would send).

  python -m t3.agents.brain_smoke [--tokens-dir ~/kit/tokens] [--live]
Environment: T3_GATEWAY_URL, a team_app token (T3_TOKEN_TEAM_APP | T3_TOKENS_SECRET | T3_TOKENS_DIR),
T3_APPROVALS_TABLE (read only here), AWS_REGION.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

from t3.suite import common
from t3.suite.common import RUNNING

TURNS = [
    "Cell 2 users report poor coverage at the cell edge. Find the cause and fix it.",
    "Approved.",
    "Ask ran_monitor to read the twin once more and tell me in one line whether the change held.",
]


def poll(gw, case_id, sleep, clock, timeout_s=600.0, poll_s=5.0):
    deadline = clock() + timeout_s
    while True:
        sleep(poll_s)
        status = gw.call_json("team_app", "front___case_status", {"case_id": case_id}, timeout=60)
        if status.get("state") not in RUNNING or clock() >= deadline:
            return status


def wait_for_person(ddb, table, approval_id, sleep, clock, timeout_s=900.0, log=print):
    log(f"WAITING FOR A PERSON: python checkpoint_kit.py approve {approval_id} \"smoke test\"")
    deadline = clock() + timeout_s
    while clock() < deadline:
        row = common.ddb_get(ddb, table, {"pk": approval_id}) or {}
        if row.get("status") in ("APPROVED", "REJECTED", "APPLIED"):
            return row["status"]
        sleep(5)
    return "TIMEOUT"


def converse(gw, ddb, table, *, tag, sleep=time.sleep, clock=time.time, log=print) -> dict:
    checks = {}
    started = gw.call_json("team_app", "front___start_case", {"request": f"run_id: {tag}\n\n{TURNS[0]}"}, timeout=60)
    case_id = started["case_id"]
    s1 = poll(gw, case_id, sleep, clock)
    log(f"turn 1 [{s1.get('state')}]: {str(s1.get('reply'))[:600]}")
    waiting = s1.get("waiting_approvals") or []
    checks["turn1_mentions_case"] = case_id in str(s1.get("reply"))
    checks["turn1_waits_for_approval"] = bool(waiting)
    replies = {"turn1": s1}
    if waiting:
        approval_id = waiting[0]["approval_id"]
        checks["approval_id_in_reply"] = approval_id in str(s1.get("reply"))
        decision = wait_for_person(ddb, table, approval_id, sleep, clock, log=log)
        checks["person_approved"] = decision == "APPROVED"
        if decision == "APPROVED":
            gw.call_json("team_app", "front___continue_case", {"case_id": case_id, "message": TURNS[1]}, timeout=60)
            s2 = poll(gw, case_id, sleep, clock)
            log(f"turn 2 [{s2.get('state')}]: {str(s2.get('reply'))[:600]}")
            checks["turn2_applied"] = bool(re.search(r"\bapplied\b", str(s2.get("reply")), re.I))
            replies["turn2"] = s2
    gw.call_json("team_app", "front___continue_case", {"case_id": case_id, "message": TURNS[2]}, timeout=60)
    s3 = poll(gw, case_id, sleep, clock)
    log(f"turn 3 [{s3.get('state')}]: {str(s3.get('reply'))[:600]}")
    checks["turn3_answered"] = bool(s3.get("reply"))
    replies["turn3"] = s3
    return {"case_id": case_id, "checks": checks, "passed": all(checks.values()),
            "trace": f"python -m t3.suite.trace check --case {case_id}"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tokens-dir")
    ap.add_argument("--live", action="store_true")
    a = ap.parse_args(argv)
    tag = f"smoke-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}"
    if not a.live:
        print(f"DRY RUN ({tag}): as team_app through the front door")
        print(f"  1. front___start_case  request = 'run_id: {tag}\\n\\n{TURNS[0]}'; poll front___case_status")
        print("     expect: the lead asked ran_monitor, ran_optimizer, ran_explainer and stopped with chg-...")
        print("  2. wait (read only) until a PERSON approves chg-... ; front___continue_case 'Approved.'")
        print(f"  3. front___continue_case '{TURNS[2]}'")
        print("Nothing sent. Add --live to run it.")
        return 0
    import boto3
    region = os.environ.get("AWS_REGION", "us-east-2")
    session = boto3.session.Session(region_name=region)
    secrets = session.client("secretsmanager") if os.environ.get("T3_TOKENS_SECRET") else None
    gw = common.Gateway(os.environ.get("T3_GATEWAY_URL", ""), common.Tokens(secrets_client=secrets,
                                                                            tokens_dir=a.tokens_dir))
    out = converse(gw, session.client("dynamodb"), os.environ.get("T3_APPROVALS_TABLE", "team13-approvals"), tag=tag)
    print(json.dumps(out, indent=1))
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
