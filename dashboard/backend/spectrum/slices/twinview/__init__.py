"""Twin view slice: the 4-cell twin as the agents see and change it, window by window, plus the consortium
of Track 2 operator silos that trained the shared response model. Read-only; no AWS calls of its own.

Mounted by discovery as `/api/twinview/*`.
"""

from .routes import router

__all__ = ["router"]
