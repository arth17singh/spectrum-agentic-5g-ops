"""The canonical change string (contract amendment 1, A2).

A change is passed to tools as ONE string, so the kit's approval key
(sha256 over [case_id, item_id, change]) is stable when an agent resends it:

    cell=<int>;d_power_db=<+x.x|-x.x>;d_tilt_deg=<+x.x|-x.x>

- cells are 0-based ints, no leading zeros;
- values have exactly one decimal and an explicit sign; zero (and -0.0) is "+0.0";
- several cells are joined with "|" and sorted by cell; a cell appears once.

`canon(list | str) -> str` rounds a list of {cell, d_power_db, d_tilt_deg} to that
form (round-half-even on the float value, as Python's format does), or checks that
a string already is canonical. `parse(str) -> list[dict]` accepts ONLY canonical
strings and raises ValueError for anything else, so `canon(parse(s)) == s` for
every string parse accepts, and `parse(canon(x))` is a fixed point.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Iterable, Mapping, Union

VALUE_RE = r"[+-](?:0|[1-9][0-9]*)\.[0-9]"
CELL_RE = r"(?:0|[1-9][0-9]*)"
ITEM_RE = rf"cell={CELL_RE};d_power_db={VALUE_RE};d_tilt_deg={VALUE_RE}"
# Shape only (sorting/uniqueness/no "-0.0" are checked in Python). Used in the JSON Schema.
CHANGE_TEXT_PATTERN = rf"^{ITEM_RE}(?:\|{ITEM_RE})*$"

_ITEM = re.compile(rf"^cell=({CELL_RE});d_power_db=({VALUE_RE});d_tilt_deg=({VALUE_RE})$")

ChangeLike = Union[str, Iterable[Any]]


def _fmt(v: Any, name: str) -> str:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"{name} must be a number, got {v!r}")
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {v!r}")
    s = f"{float(v):+.1f}"
    return "+0.0" if s == "-0.0" else s


def _get(item: Any, key: str) -> Any:
    if isinstance(item, Mapping):
        if key not in item:
            raise ValueError(f"change item is missing {key!r}: {item!r}")
        return item[key]
    if hasattr(item, key):  # e.g. a CellChange model
        return getattr(item, key)
    raise ValueError(f"change item is missing {key!r}: {item!r}")


def _canon_items(items: Iterable[Any]) -> str:
    rows: list[tuple[int, str]] = []
    for item in items:
        if isinstance(item, Mapping):
            unknown = set(item) - {"cell", "d_power_db", "d_tilt_deg"}
            if unknown:
                raise ValueError(f"change item has unknown keys {sorted(unknown)}")
        cell = _get(item, "cell")
        if isinstance(cell, bool) or not isinstance(cell, int) or cell < 0:
            raise ValueError(f"cell must be a non-negative int, got {cell!r}")
        p = _fmt(_get(item, "d_power_db"), "d_power_db")
        t = _fmt(_get(item, "d_tilt_deg"), "d_tilt_deg")
        rows.append((cell, f"cell={cell};d_power_db={p};d_tilt_deg={t}"))
    if not rows:
        raise ValueError("a change needs at least one cell")
    rows.sort(key=lambda r: r[0])
    cells = [c for c, _ in rows]
    if len(set(cells)) != len(cells):
        raise ValueError(f"a cell appears more than once: {cells}")
    return "|".join(text for _, text in rows)


def parse(text: str) -> list[dict[str, Any]]:
    """Parse a canonical change string. Raises ValueError unless `text` is exactly canonical."""
    if not isinstance(text, str):
        raise ValueError(f"change text must be a string, got {type(text).__name__}")
    out: list[dict[str, Any]] = []
    for part in text.split("|"):
        m = _ITEM.match(part)
        if not m:
            raise ValueError(f"malformed change item {part!r}; expected cell=<int>;d_power_db=<+x.x>;d_tilt_deg=<+x.x>")
        cell_s, p_s, t_s = m.groups()
        for v in (p_s, t_s):
            if v == "-0.0":
                raise ValueError(f"zero must be written +0.0, got {part!r}")
        out.append({"cell": int(cell_s), "d_power_db": float(p_s), "d_tilt_deg": float(t_s)})
    cells = [d["cell"] for d in out]
    if cells != sorted(cells):
        raise ValueError(f"cells must be sorted ascending, got {cells}")
    if len(set(cells)) != len(cells):
        raise ValueError(f"a cell appears more than once: {cells}")
    return out


def canon(change: ChangeLike) -> str:
    """Canonical change string from a list of cell changes, or validate an existing string."""
    if isinstance(change, str):
        parse(change)  # raises unless already canonical
        return change
    if isinstance(change, Mapping):
        raise ValueError("pass a list of cell changes, not a single dict")
    return _canon_items(change)


def approval_key(case_id: str, item_id: str, change_text: str) -> str:
    """The kit's approval id for a change: "chg-" + sha256(json.dumps([case_id, item_id, change]))[:12]."""
    return "chg-" + hashlib.sha256(json.dumps([case_id, item_id, change_text]).encode()).hexdigest()[:12]
