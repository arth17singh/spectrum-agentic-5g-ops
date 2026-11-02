"""Helpers shared by every slice (not a slice itself: names starting with "_" are skipped)."""

from .records import RecordsError, RecordsReader

__all__ = ["RecordsError", "RecordsReader"]
