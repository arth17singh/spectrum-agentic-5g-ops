"""Post-deploy smoke for remediation-v2 behind ran___propose, through the AgentCore Gateway (runbook step 5).

    python -m t3.tools.image.smoke_gateway_v2            # dry run: prints the two calls, sends nothing
    python -m t3.tools.image.smoke_gateway_v2 --live     # sends them (needs the deploy approved and done)

Two calls, nothing applied, nothing to approve:
  1. ran___seed_case as team_app (the only principal Cedar lets seed): a FRESH case at the Live Fix fault
     "power amplifier fault on cell 2" (dev seed 7002, official preset: commissioning none, then cell 2 -6 dB);
  2. ran___propose as ran_optimizer (the only principal Cedar lets propose) on that case.
Pass: the answer names t1-remediation-bc-delta@v2-c282d94, baseline_step 1, and a change on cell 2 only with a
positive power delta (the local image proposed cell=2;d_power_db=+6.2;d_tilt_deg=+0.0 for this exact case).
Environment: T3_GATEWAY_URL and T3_TOKENS_DIR (or T3_TOKEN_TEAM_APP / T3_TOKEN_RAN_OPTIMIZER), as the runners use.
Never prints a token or the Gateway URL.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

START_LOG = [[], "cell=2;d_power_db=-6.0;d_tilt_deg=+0.0"]
EXPECT_MODEL = ("t1-remediation-bc-delta", "v2-c282d94")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--live", action="store_true", help="send the two calls (default: dry run)")
    a = ap.parse_args(argv)
    case_id = f"case-smokev2{time.strftime('%H%M%S', time.gmtime())}"
    seed_args = {"case_id": case_id, "scenario": "nominal", "seed": 7002, "preset": "official",
                 "start_action_log_json": json.dumps(START_LOG)}
    if not a.live:
        print("DRY RUN (add --live to send):")
        print("  team_app      ran___seed_case", json.dumps(seed_args))
        print("  ran_optimizer ran___propose  ", json.dumps({"case_id": case_id}))
        return 0
    from t3.suite import common

    gw = common.Gateway(os.environ.get("T3_GATEWAY_URL", ""), common.Tokens())
    seeded = gw.call("team_app", "ran___seed_case", seed_args, timeout=120)
    if seeded["decision"] != "ALLOW" or seeded.get("tool_error"):
        print("FAIL seed:", seeded["decision"], (seeded.get("tool_error") or seeded.get("text") or "")[:300])
        return 1
    out = gw.call("ran_optimizer", "ran___propose", {"case_id": case_id}, timeout=120)
    if out["decision"] != "ALLOW" or out.get("tool_error"):
        print("FAIL propose:", out["decision"], (out.get("tool_error") or out.get("text") or "")[:300])
        return 1
    ans = json.loads(out["text"])
    pol = ans.get("policy") or {}
    cells = [p for p in (ans.get("change") or "").split("|") if p]
    ok = ((pol.get("model_id"), pol.get("version")) == EXPECT_MODEL and ans.get("baseline_step") == 1
          and len(cells) == 1 and cells[0].startswith("cell=2;d_power_db=+") and "d_power_db=+0.0" not in cells[0])
    print(json.dumps({"case_id": case_id, "policy": pol, "baseline_step": ans.get("baseline_step"),
                      "change": ans.get("change"), "gateway_date": out.get("gateway_date"),
                      "latency_ms": out.get("latency_ms")}, indent=1))
    print("PASS" if ok else "FAIL: not the expected served model or not a targeted cell-2 power fix")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
