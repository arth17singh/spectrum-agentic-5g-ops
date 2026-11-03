"""Shared helpers for item generation: hash-based choices, the dev/test split, canonical JSON lines."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

ITEMS_DIR = Path(__file__).with_name("items")
UNITS = ("interface", "execution", "safety", "knowledge", "price_latency")
SPLIT_SALT = "t1-split-v1"
DEV_FRACTION = 0.2  # 20% dev, 80% test


def h(*parts: Any) -> int:
    """A stable 64-bit integer from the parts (sha256, not Python's salted hash)."""
    s = "\x1f".join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest()[:8], "big")


def pick(seq: Sequence[Any], *key: Any) -> Any:
    return seq[h(*key) % len(seq)]


def split_of(unit: str, item_id: str) -> str:
    """dev or test, from sha256(unit, item_id, salt): stable, independent of file order."""
    return "dev" if (h(unit, item_id, SPLIT_SALT) % 10_000) < DEV_FRACTION * 10_000 else "test"


def dumps(obj: Any) -> str:
    """One canonical JSON line (sorted keys, compact, UTF-8, no NaN)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def lines_sha256(lines: list[str]) -> str:
    """sha256 of the lines exactly as they sit in a .jsonl file (each followed by \\n)."""
    return sha256_text("".join(line + "\n" for line in lines))


def item(unit: str, item_id: str, template: str, inp: dict, expected: dict, meta: dict) -> dict:
    return {
        "unit": unit,
        "item_id": item_id,
        "split": split_of(unit, item_id),
        "template": template,
        "input": inp,
        "expected": expected,
        "meta": meta,
    }


NUM_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}


def amount_text(a: float, key: Any) -> str:
    """Digits or a number word (integers 1-8 only), chosen by hash."""
    if float(a).is_integer() and int(a) in NUM_WORDS and h("numword", key) % 2:
        return NUM_WORDS[int(a)]
    return f"{a:g}"


def render_state(state: dict[str, Any], hour: bool = True) -> str:
    """The network state as the model sees it: rounded to 2 decimals so the text is machine-stable."""
    cols = [
        ("power_dbm", "power_dbm", 1),
        ("tilt_deg", "tilt_deg", 1),
        ("ue_count", "ue_count", 0),
        ("rsrp_mean_dbm", "rsrp_mean_dbm", 2),
        ("rsrp_p10_dbm", "rsrp_p10_dbm", 2),
        ("sinr_mean_db", "sinr_mean_db", 2),
        ("sinr_p10_db", "sinr_p10_db", 2),
    ]
    cols = [c for c in cols if c[0] in state]
    head = "cell | " + " | ".join(c[1] for c in cols) + " | alarm"
    rows = [head]
    for i in range(len(state["power_dbm"])):
        vals = []
        for key, _, nd in cols:
            v = state[key][i]
            vals.append(str(int(v)) if nd == 0 else f"{float(v):.{nd}f}")
        rows.append(f"{i} | " + " | ".join(vals) + " | " + ("RAISED" if state["alarm"][i] else "no"))
    net = []
    if "coverage_ratio" in state:
        net.append(f"coverage_ratio={float(state['coverage_ratio']):.3f}")
    if "sla_ratio" in state:
        net.append(f"sla_ratio={float(state['sla_ratio']):.3f}")
    if hour and "hour" in state:
        net.append(f"hour={int(round(float(state['hour'])))}")
    if net:
        rows.append("network: " + " ".join(net))
    return "\n".join(rows)
