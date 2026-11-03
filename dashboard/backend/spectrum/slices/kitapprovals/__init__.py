"""Kit approvals (W7g): a person approves or rejects the kit's pending changes (chg-...) on this laptop.

The real table (team13-approvals), the kit's exact decision and control record, a PIN on every decision.
Separate from the simulator's /api/approvals/* and /oversight, and from the asknoc slice (no imports either
way). See routes.py.
"""

from .routes import router

__all__ = ["router"]
