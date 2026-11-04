"""Live Fix: a fault injected on the twin, diagnosed live by the swarm through the AgentCore Gateway, held by Cedar
until a person approves, applied, and graded against the healthy twin, with its own evidence run.

Mounted by discovery as /api/livefix/*. Settings (environment), shared with Ask the NOC:
  ASKNOC_KIT_DIR          the kit folder (gateway URL and the app's token; never served or logged)
  ASKNOC_REGION           AWS region of the approvals table and the audit log (default us-east-2)
  ASKNOC_APPROVALS_TABLE  the kit's approvals table (default team13-approvals)
  EVIDENCE_S3 | TEAM13_DATA_BUCKET  where the run's evidence goes (else the team bucket from the account)
  SPECTRUM_EVIDENCE_DIR   this laptop's evidence folder (a byte-identical copy, for the Evidence screen)
  T3_SWARM_MODEL_ID       the agents' models, recorded as model_id
  AWS_PROFILE             the laptop's AWS session
Under SPECTRUM_OFFLINE=1 only recorded mode works and nothing calls AWS.
"""

from __future__ import annotations

from .routes import get_service, router

__all__ = ["router", "run"]


async def run(ctx) -> None:
    get_service()  # build it at startup (reads the kit folder), not on the first click
