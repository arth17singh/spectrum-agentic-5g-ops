"""Pure helpers for the kit approvals slice: the canonical change text, redaction, display names, rows.

Nothing here calls AWS. Copied (never imported) from other code, because a slice must not import another
slice or the t3 packages:
  - `parse` / `canon`: t3/tools/changes.py `_local_parse` / `_local_canon` (pinned by a parity fixture)
  - `display_who`: twinview's `pseudonym()`, fixed so an email is never shown (review F3)
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import unicodedata
from decimal import Decimal
from typing import Any

PSEUDONYM_RE = re.compile(r"^person:[A-Za-z0-9_.-]+$")  # contracts/schema/control.schema.json
APPROVAL_ID_RE = r"^chg-[0-9a-f]{12}$"

# ---------------------------------------------------------------- canonical change text (t3/tools/changes.py)
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


def parse(text: Any) -> list[dict]:
    if isinstance(text, (list, tuple)):
        return _from_list(text)
    if not isinstance(text, str) or not text.strip():
        raise ChangeError("change must be a string like cell=2;d_power_db=+0.0;d_tilt_deg=-2.0")
    items = []
    for part in text.strip().split("|"):
        m = _ITEM.fullmatch(part.strip().replace(" ", ""))
        if not m:
            raise ChangeError(f"bad change item {part!r}")
        items.append({"cell": int(m.group(1)), "d_power_db": float(m.group(2)), "d_tilt_deg": float(m.group(3))})
    return _from_list(items)


def canon(change: Any) -> str:
    cells = parse(change)
    return "|".join(f"cell={c['cell']};d_power_db={_fmt(c['d_power_db'])};d_tilt_deg={_fmt(c['d_tilt_deg'])}"
                    for c in cells)


def try_canon(change: Any) -> tuple[list[dict] | None, str | None]:
    try:
        cells = parse(change)
        return cells, canon(cells)
    except (ChangeError, TypeError, ValueError):
        return None, None


# ---------------------------------------------------------------- redaction and reasons
_REDACT = [
    re.compile(r"https?://\S+", re.I),
    re.compile(r"\barn:aws[a-z-]*:\S+", re.I),
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]*"),
    re.compile(r"\b\d{12}\b"),
    re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
]


def has_identifier(text: str) -> bool:
    return any(p.search(text) for p in _REDACT)


def redact(text: Any, cap: int = 300) -> str | None:
    if not isinstance(text, str):
        return None
    for p in _REDACT:
        text = p.sub("[redacted]", text)
    return text[:cap]


def clean_reason(reason: str) -> str:
    """The person's reason: NFC, stripped, 3-300 characters, no control or format characters, and no
    identifier (URL, ARN, email, JWT, account id, IP), because it is stored in the table and CloudWatch."""
    r = unicodedata.normalize("NFC", reason).strip()
    if not 3 <= len(r) <= 300:
        raise ValueError("reason must be 3 to 300 characters")
    if any(unicodedata.category(ch) in ("Cc", "Cf") for ch in r):
        raise ValueError("reason has control characters")
    if has_identifier(r):
        raise ValueError("reason must not contain a URL, ARN, email, token, account id or IP address")
    return r


# ---------------------------------------------------------------- display names (never an email or ARN)
def session_name(arn: Any) -> str | None:
    """The last path segment of an STS/IAM ARN (the role-session name, which is an email here)."""
    if not isinstance(arn, str) or not arn.startswith("arn:"):
        return None
    tail = arn.split(":", 5)[-1]
    return tail.rsplit("/", 1)[-1] if "/" in tail else tail


def display_who(who: Any, approvers: dict[str, str] | None = None, key: bytes = b"") -> str | None:
    """A display name for `decided_by`. `person:*` passes through; an ARN or any other value (the kit CLI
    writes the caller ARN, whose session name is an email) is mapped through APPROVERS, else shown as
    person:op-<6 hex of HMAC(key, value)>. Never the raw value."""
    if not isinstance(who, str) or not who.strip():
        return None
    s = who.strip()
    if PSEUDONYM_RE.fullmatch(s):
        return s
    name = session_name(s) or s
    mapped = (approvers or {}).get(name)
    if isinstance(mapped, str) and PSEUDONYM_RE.fullmatch(mapped):
        return mapped
    return "person:op-" + hmac.new(key or b"kitapprovals", name.encode(), hashlib.sha256).hexdigest()[:6]


# ---------------------------------------------------------------- DynamoDB values
def plain(v: Any) -> Any:
    """Decimal -> int/float, recursively (FastAPI cannot encode Decimal)."""
    if isinstance(v, Decimal):
        return int(v) if v == v.to_integral_value() else float(v)
    if isinstance(v, dict):
        return {k: plain(x) for k, x in v.items()}
    if isinstance(v, list):
        return [plain(x) for x in v]
    return v


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def parse_change_attr(raw: Any) -> dict:
    """The row's `change` attribute is a JSON string {item_id, change, reason} (KIT L295)."""
    if not isinstance(raw, str):
        return {}
    try:
        out = json.loads(raw)
    except ValueError:
        return {}
    return out if isinstance(out, dict) else {}


def num(v: Any) -> int | None:
    v = plain(v)
    return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def row_view(item: dict, now: float, approvers: dict | None, key: bytes) -> dict:
    """One approval row for the UI. `change_sha256` hashes the stored attribute exactly (what is decided)."""
    raw = item.get("change") if isinstance(item.get("change"), str) else ""
    body = parse_change_attr(raw)
    text = body.get("change")
    cells, canonical = try_canon(text)
    created = num(item.get("created_at"))
    return {
        "approval_id": item.get("pk"),
        "status": item.get("status"),
        "agent": item.get("agent"),
        "case_id": item.get("case_id"),
        "item_id": body.get("item_id"),
        "change_text": text if isinstance(text, str) else (json.dumps(plain(text)) if text is not None else None),
        "canonical_text": canonical,
        "canonical": canonical is not None and canonical == text,
        "cells": cells,
        "change_sha256": sha256_text(raw),
        "created_at": created,
        "age_s": int(now - created) if created is not None else None,
        "decided_by": display_who(item.get("decided_by"), approvers, key),
        "decided_at": num(item.get("decided_at")),
        "decision_reason": redact(item.get("decision_reason")),
        "applied_at": num(item.get("applied_at")),
    }


EVAL_UNITS = {"t3.evals.swarm-proposals"}
# live_swarm.case_id_for: case-<8 hex of the run id>-i<incident>s<step>. Front-door cases are case-<8 hex> only.
EVAL_CASE_RE = re.compile(r"^case-[0-9a-f]{8}-i\d{2}s\d$")


def is_eval_mapping(row: dict) -> bool:
    """A t3-<case> row that maps the case to an evaluation run (t3/evals/live_swarm.py writes run_id
    t3.swarm-live-<UTC>, unit t3.evals.swarm-proposals, source t3/evals/live_swarm.py)."""
    run_id, unit, source = row.get("run_id"), row.get("unit"), row.get("source")
    return ((isinstance(run_id, str) and run_id.startswith("t3.swarm-")) or unit in EVAL_UNITS
            or (isinstance(source, str) and source.endswith("evals/live_swarm.py")))


def order_rows(rows: list[dict]) -> list[dict]:
    pending = sorted((r for r in rows if r["status"] == "PENDING"), key=lambda r: r["created_at"] or 0)
    rest = sorted((r for r in rows if r["status"] != "PENDING"),
                  key=lambda r: -(r["applied_at"] or r["decided_at"] or r["created_at"] or 0))
    return (pending + rest)[:200]


def find_prediction(doc: dict, item_id: Any, canonical: str | None) -> dict | None:
    """Same selection as t3/tools/app.py _find_prediction (lenient): newest `predict` for this canonical
    change and item, else the newest `predict` for this change."""
    if canonical is None:
        return None
    preds = [p for p in doc.get("predictions") or [] if isinstance(p, dict) and p.get("change_text") == canonical]
    same = next((p for p in reversed(preds) if p.get("kind") == "predict" and p.get("item_id") == item_id), None)
    return same or next((p for p in reversed(preds) if p.get("kind") == "predict"), None)
