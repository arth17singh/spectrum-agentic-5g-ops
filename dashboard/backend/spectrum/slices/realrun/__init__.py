"""Real-run slice: Track 1 base vs tuned vs Bedrock, the interface/execution bar, the RL gate, Track 2 and 3
baselines, and a per-run evidence explorer, all read from the mirrored S3 evidence (make evidence-sync).

Mounted by discovery as `/api/realrun/*`. Read-only; never calls AWS (the sync does, read-only).
"""

from .routes import router

__all__ = ["router"]
