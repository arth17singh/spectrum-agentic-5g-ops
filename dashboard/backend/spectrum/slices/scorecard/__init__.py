"""Scorecard slice: every scored unit, its declared target, what was measured, pass or fail, and the
gain over its baseline. Read-only view over the declared/result records."""

from .routes import router

__all__ = ["router"]
