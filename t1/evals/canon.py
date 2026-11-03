"""The canonical change string (contract amendment A2), from W1's team13_contracts.change only.

    cell=<int>;d_power_db=<+x.x>;d_tilt_deg=<+x.x>   0-based cells, one decimal, explicit sign,
                                                     "+0.0" for zero, "|" between cells sorted by
                                                     cell; zero-change cells are KEPT as given

There is no local fallback: `team13_contracts.change.canon` is the single source of truth. This
module adds only what an evaluation needs around it:
  - the empty string "" means "no change" (a refusal or a no-op step). A2 has no empty change
    (W1's canon raises), so "" is never sent to a tool; here it is mapped to the empty list;
  - cells must exist on the 4-cell twin (0..3);
  - errors are raised as ChangeError (a ValueError).
"""

from __future__ import annotations

from typing import Any

from team13_contracts.change import canon as _w1_canon
from team13_contracts.change import parse as _w1_parse

N_CELLS = 4
SOURCE = "team13_contracts.change"


class ChangeError(ValueError):
    """A change that is not canonical, names a cell outside 0..3, or is malformed."""


def _is_empty(change: Any) -> bool:
    return change is None or change == "" or (isinstance(change, (list, tuple)) and len(change) == 0)


def _check_cells(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for it in items:
        if not 0 <= it["cell"] < N_CELLS:
            raise ChangeError(f"cell {it['cell']} out of range 0..{N_CELLS - 1}")
    return items


def canon(change: Any) -> str:
    """W1's canonical string for a list of {cell, d_power_db, d_tilt_deg} or a canonical string;
    "" for no change."""
    if _is_empty(change):
        return ""
    try:
        text = _w1_canon(change)
        _check_cells(_w1_parse(text))
    except ChangeError:
        raise
    except (ValueError, TypeError) as e:
        raise ChangeError(str(e)) from e
    return text


def parse(text: str) -> list[dict[str, Any]]:
    """W1's parse of a canonical string; [] for ""."""
    if text == "":
        return []
    try:
        return _check_cells(_w1_parse(text))
    except ChangeError:
        raise
    except (ValueError, TypeError) as e:
        raise ChangeError(str(e)) from e


def to_vector(change: Any, n_cells: int = N_CELLS) -> list[float]:
    """Raw action vector (n power deltas then n tilt deltas) from a change string or list."""
    raw = [0.0] * (2 * n_cells)
    for item in parse(canon(change)):
        raw[item["cell"]] += item["d_power_db"]
        raw[n_cells + item["cell"]] += item["d_tilt_deg"]
    return raw
