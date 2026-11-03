"""Canonical change strings (contract amendment A2): ``cell=<int>;d_power_db=<+x.x>;d_tilt_deg=<+x.x>``.

Multi-cell changes are joined with ``|`` and sorted by cell; one decimal, explicit sign, ``+0.0`` for
zero. Uses W1's ``team13_contracts.change`` when importable, else this compatible local version.
"""

from __future__ import annotations

import re
from typing import Any

_ITEM = re.compile(r"^cell=(\d+);d_power_db=([+-]\d+\.\d);d_tilt_deg=([+-]\d+\.\d)$")


def _fmt(x: float) -> str:
    x = round(float(x), 1) + 0.0  # + 0.0 turns -0.0 into +0.0
    return f"{x:+.1f}"


def _canon_local(change: list[dict[str, Any]] | str) -> str:
    if isinstance(change, str):
        change = _parse_local(change)
    merged: dict[int, list[float]] = {}
    for item in change:
        cell = int(item["cell"])
        acc = merged.setdefault(cell, [0.0, 0.0])
        acc[0] += float(item.get("d_power_db", 0.0))
        acc[1] += float(item.get("d_tilt_deg", 0.0))
    return "|".join(
        f"cell={c};d_power_db={_fmt(p)};d_tilt_deg={_fmt(t)}" for c, (p, t) in sorted(merged.items())
    )


def _parse_local(text: str) -> list[dict[str, Any]]:
    text = text.strip()
    if not text:
        return []
    out = []
    for part in text.split("|"):
        m = _ITEM.match(part.strip())
        if not m:
            raise ValueError(f"not a canonical change item: {part!r}")
        out.append({"cell": int(m[1]), "d_power_db": float(m[2]), "d_tilt_deg": float(m[3])})
    return out


try:  # stream W1 (merged)
    from team13_contracts.change import canon, parse  # type: ignore[import-not-found]
except ImportError:
    canon, parse = _canon_local, _parse_local


def nonzero(change: list[dict[str, Any]] | str | None) -> list[dict[str, Any]]:
    """The change as a list without all-zero items (None and "" are the empty change)."""
    if change is None:
        return []
    if isinstance(change, str):
        change = parse(change)
    return [
        {"cell": int(i["cell"]), "d_power_db": float(i.get("d_power_db", 0.0)),
         "d_tilt_deg": float(i.get("d_tilt_deg", 0.0))}
        for i in change
        if float(i.get("d_power_db", 0.0)) != 0.0 or float(i.get("d_tilt_deg", 0.0)) != 0.0
    ]
