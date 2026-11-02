"""The canonical change string (build contract amendment 1, A2).

    cell=<int>;d_power_db=<+x.x>;d_tilt_deg=<+x.x>   cells 0-based, one decimal, explicit sign,
                                                     "+0.0" for zero, multi-cell joined with "|"
                                                     and sorted by cell

`canon(list|str) -> str` and `parse(str) -> list[{cell, d_power_db, d_tilt_deg}]`. W1's
`team13_contracts.change` is used when importable; this fallback has the same signatures and
behaviour so the swap at merge is a no-op. The empty change (no-op) is the empty string.
"""

from __future__ import annotations

import math
import re
from typing import Any

N_CELLS = 4

_PART = re.compile(
    r"^cell=(\d+);d_power_db=([+-]\d+(?:\.\d+)?);d_tilt_deg=([+-]\d+(?:\.\d+)?)$"
)


class ChangeError(ValueError):
    """A change that cannot be canonicalised (bad shape, bad cell, non-finite number)."""


def _fmt(x: float) -> str:
    v = round(float(x), 1)
    if v == 0:
        v = 0.0  # no "-0.0"
    return f"{v:+.1f}"


def _num(x: Any, name: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        if isinstance(x, str):
            try:
                x = float(x)
            except ValueError:
                raise ChangeError(f"{name} must be a number, got {x!r}") from None
        else:
            raise ChangeError(f"{name} must be a number, got {x!r}")
    if not math.isfinite(float(x)):
        raise ChangeError(f"{name} must be finite, got {x!r}")
    return float(x)


def _cell(x: Any, n_cells: int) -> int:
    if isinstance(x, bool):
        raise ChangeError(f"cell must be an integer, got {x!r}")
    if isinstance(x, float) and x.is_integer():
        x = int(x)
    if isinstance(x, str) and re.fullmatch(r"\d+", x.strip()):
        x = int(x.strip())
    if not isinstance(x, int):
        raise ChangeError(f"cell must be an integer, got {x!r}")
    if not 0 <= x < n_cells:
        raise ChangeError(f"cell {x} out of range 0..{n_cells - 1}")
    return x


def _fallback_parse(text: str, n_cells: int = N_CELLS) -> list[dict[str, Any]]:
    if not isinstance(text, str):
        raise ChangeError(f"change text must be a string, got {type(text).__name__}")
    text = text.strip()
    if text == "":
        return []
    out = []
    for part in text.split("|"):
        m = _PART.match(part.strip().replace(" ", ""))
        if not m:
            raise ChangeError(f"not a canonical change part: {part!r}")
        out.append(
            {
                "cell": _cell(int(m.group(1)), n_cells),
                "d_power_db": float(m.group(2)),
                "d_tilt_deg": float(m.group(3)),
            }
        )
    return out


def _fallback_canon(change: Any, n_cells: int = N_CELLS) -> str:
    if isinstance(change, str):
        change = _fallback_parse(change, n_cells)
    if isinstance(change, dict):
        change = [change]
    if not isinstance(change, (list, tuple)):
        raise ChangeError("change must be a canonical string or a list of {cell, d_power_db, d_tilt_deg}")
    per_cell: dict[int, list[float]] = {}
    for item in change:
        if not isinstance(item, dict) or "cell" not in item:
            raise ChangeError(f"each change item needs a 'cell': {item!r}")
        extra = set(item) - {"cell", "d_power_db", "d_tilt_deg"}
        if extra:
            raise ChangeError(f"unknown change keys {sorted(extra)}")
        c = _cell(item["cell"], n_cells)
        acc = per_cell.setdefault(c, [0.0, 0.0])
        acc[0] += _num(item.get("d_power_db", 0.0), "d_power_db")
        acc[1] += _num(item.get("d_tilt_deg", 0.0), "d_tilt_deg")
    parts = []
    for c in sorted(per_cell):
        p, t = per_cell[c]
        if round(p, 1) == 0 and round(t, 1) == 0:
            continue  # a zero entry is no change; keep the canonical form minimal
        parts.append(f"cell={c};d_power_db={_fmt(p)};d_tilt_deg={_fmt(t)}")
    return "|".join(parts)


try:  # W1's contracts package, when merged
    from team13_contracts.change import canon as _canon  # type: ignore
    from team13_contracts.change import parse as _parse  # type: ignore

    SOURCE = "team13_contracts.change"
except Exception:  # noqa: BLE001 - absent before merge
    _canon, _parse = _fallback_canon, _fallback_parse
    SOURCE = "t1.evals.canon (fallback)"


def canon(change: Any) -> str:
    """Canonical string for a list of changes or a (possibly non-canonical) string."""
    try:
        return _canon(change)
    except ChangeError:
        raise
    except Exception as e:  # noqa: BLE001 - normalise the contracts package's errors
        raise ChangeError(str(e)) from e


def parse(text: str) -> list[dict[str, Any]]:
    """List of {cell, d_power_db, d_tilt_deg} from a canonical string."""
    try:
        return _parse(text)
    except ChangeError:
        raise
    except Exception as e:  # noqa: BLE001
        raise ChangeError(str(e)) from e


def to_vector(change: Any, n_cells: int = N_CELLS) -> list[float]:
    """Raw action vector (n power deltas then n tilt deltas) from a change string or list."""
    raw = [0.0] * (2 * n_cells)
    for item in parse(canon(change)):
        raw[item["cell"]] += item["d_power_db"]
        raw[n_cells + item["cell"]] += item["d_tilt_deg"]
    return raw
