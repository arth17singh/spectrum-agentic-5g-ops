"""The canonical change string (contract amendment 1, A2).

    cell=<int>;d_power_db=<+x.x|-x.x>;d_tilt_deg=<+x.x|-x.x>

Several cells are joined with `|` and sorted by cell; cells are 0-based; one decimal, explicit sign,
`+0.0` for zero. W1's `team13_contracts.change` is the reference; this module uses it when it can
be imported and otherwise implements the same grammar.

`parse` accepts the canonical text, a slightly loose spelling of it (no sign, no decimal, cells in
any order) and, for local test events, a list of `{cell, d_power_db, d_tilt_deg}`. It refuses
anything that would not survive `canon` unchanged in value (more than one decimal).
"""
from __future__ import annotations

import math
import re
from typing import Any

_ITEM = re.compile(r"cell=(\d+);d_power_db=([+-]?\d+(?:\.\d)?);d_tilt_deg=([+-]?\d+(?:\.\d)?)")


class ChangeError(ValueError):
    pass


def _fmt(x: float) -> str:
    x = round(float(x), 1)
    return "+0.0" if x == 0 else f"{x:+.1f}"


def _one_decimal(v: Any, what: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ChangeError(f"{what} must be a finite number")
    if abs(round(v, 1) - v) > 1e-9:
        raise ChangeError(f"{what}={v} has more than one decimal")
    return float(round(v, 1)) + 0.0


def _from_list(items: Any) -> list[dict]:
    if not isinstance(items, (list, tuple)) or not items:
        raise ChangeError("a change needs at least one cell")
    out = []
    for i, c in enumerate(items):
        if not isinstance(c, dict) or set(c) - {"cell", "d_power_db", "d_tilt_deg"}:
            raise ChangeError(f"change item {i} must be {{cell, d_power_db, d_tilt_deg}}")
        cell = c.get("cell")
        if isinstance(cell, float) and cell.is_integer():
            cell = int(cell)
        if isinstance(cell, bool) or not isinstance(cell, int) or cell < 0:
            raise ChangeError(f"change item {i}: cell must be an integer >= 0")
        out.append({"cell": cell, "d_power_db": _one_decimal(c.get("d_power_db", 0.0), "d_power_db"),
                    "d_tilt_deg": _one_decimal(c.get("d_tilt_deg", 0.0), "d_tilt_deg")})
    if len({c["cell"] for c in out}) != len(out):
        raise ChangeError("a change lists the same cell twice")
    return sorted(out, key=lambda c: c["cell"])


def _local_parse(text: Any) -> list[dict]:
    if isinstance(text, (list, tuple)):
        return _from_list(text)
    if not isinstance(text, str) or not text.strip():
        raise ChangeError("change must be a string like cell=2;d_power_db=+0.0;d_tilt_deg=-2.0")
    items = []
    for part in text.strip().split("|"):
        m = _ITEM.fullmatch(part.strip().replace(" ", ""))
        if not m:
            raise ChangeError(f"bad change item {part!r}: use cell=<int>;d_power_db=<+x.x>;d_tilt_deg=<+x.x>")
        items.append({"cell": int(m.group(1)), "d_power_db": float(m.group(2)), "d_tilt_deg": float(m.group(3))})
    return _from_list(items)


def round_items(change: Any) -> Any:
    """A list from a model may carry more decimals: round it to the grammar's one decimal."""
    if not isinstance(change, (list, tuple)):
        return change
    out = []
    for c in change:
        if isinstance(c, dict):
            c = dict(c)
            for k in ("d_power_db", "d_tilt_deg"):
                v = c.get(k, 0.0)
                if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
                    c[k] = round(float(v), 1)
        out.append(c)
    return out


def _local_canon(change: Any) -> str:
    cells = _local_parse(change) if isinstance(change, str) else _from_list(round_items(change))
    return "|".join(f"cell={c['cell']};d_power_db={_fmt(c['d_power_db'])};d_tilt_deg={_fmt(c['d_tilt_deg'])}"
                    for c in cells)


try:  # W1's reference implementation, when it is installed
    from team13_contracts.change import canon as _w1_canon, parse as _w1_parse  # type: ignore

    def parse(text: Any) -> list[dict]:
        if isinstance(text, (list, tuple)):
            return _from_list(text)
        try:
            return _w1_parse(text)
        except Exception:  # noqa: BLE001 — fall back to the tolerant spelling, same grammar
            return _local_parse(text)

    def canon(change: Any) -> str:
        return _w1_canon(change)
    SOURCE = "team13_contracts.change"
except ImportError:
    parse, canon = _local_parse, _local_canon
    SOURCE = "t3.tools.changes"
