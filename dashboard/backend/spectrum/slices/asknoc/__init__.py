"""Ask the NOC slice (stream W7f): an operator talks to the swarm's lead agent noc_lead through the organisers'
checkpoint kit, as the kit's app identity `team_app`, using only the three front-door tools
(front___start_case, front___continue_case, front___case_status). The panel can't approve or apply anything.

Mounted by discovery as `/api/asknoc/*`. Settings (environment):
  ASKNOC_KIT_DIR          the kit folder (kit-state.json, tokens/team_app.txt; only builder_host from kit.json)
  ASKNOC_STATE_DIR        local history (default ~/.spectrum/asknoc; never the evidence root)
  ASKNOC_REGION           AWS region for the read-only CloudWatch and DynamoDB reads (default us-east-2)
  ASKNOC_APPROVALS_TABLE  the kit's table (default team13-approvals)
  ASKNOC_LEAD_TIMEOUT_S, ASKNOC_RELEASE_SLOT_S, ASKNOC_LOST_S, ASKNOC_MAX_INFLIGHT, ASKNOC_ALLOWED_ORIGINS,
  ASKNOC_REDACT_HOSTS, ASKNOC_NONJUDGED_RUNS
  AWS_PROFILE             the laptop's AWS session for the reads (e.g. fyuz)
Under SPECTRUM_OFFLINE=1 (tests, the merge gate) nothing is read from or written to disk and no call is made.
"""

from __future__ import annotations

from .routes import get_service, router

__all__ = ["router", "run"]


async def run(ctx) -> None:
    svc = get_service()
    if svc.offline or getattr(ctx, "offline", False):
        return
    svc.resume()
