"""One statement of "verified" for every screen: the run's cached check_unit verdict governs it.

check_unit (tools/check_unit.py, cached by `make evidence-sync CHECK=1` in <evidence root>/.sync/checks/) checks,
in part c, that every record is keyed to the run and that its AWS time agrees with AWS, deriving the time from
S3 LastModified or CloudWatch when a record's own aws_time is null. So "k of n records carry their own AWS time"
is a detail line, not a verdict, and "synthetic" describes the inputs, not whether the run is evidence.

govern(prov, run_ids) rewrites a slice's provenance dict:
  * every run shown has a cached check with part c PASS and no FAILED part -> state "real", label "VERIFIED"
  * any run shown has part c not PASS, or a FAILED part                     -> state "unverified", label "UNVERIFIED"
  * no cached check for some run                                            -> unchanged (the slice's own rule)
The slice's AWS-time and synthetic counts are kept as detail lines.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable


def _root() -> Path | None:
    raw = os.environ.get("SPECTRUM_RECORDS_ROOT") or os.environ.get("SPECTRUM_EVIDENCE_DIR")
    if not raw or raw.startswith("s3://"):
        return None
    return Path(raw[len("file://"):] if raw.startswith("file://") else raw)


def check_of(run_id: str, root: Path | None = None) -> dict | None:
    root = root or _root()
    if root is None or not isinstance(run_id, str) or "/" in run_id or ".." in run_id:
        return None
    try:
        d = json.loads((root / ".sync" / "checks" / f"{run_id}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    parts = ((d.get("summary") or {}).get("parts")) or {}
    return {"exit_code": d.get("exit_code"), "checked_at": d.get("checked_at"),
            "parts": {k: (v or {}).get("state") for k, v in parts.items() if isinstance(v, dict)}}


def superseded_ids(root: Path | None = None) -> set[str]:
    root = root or _root()
    out: set[str] = set()
    for p in ([root / "superseded.json"] if root else []) + [Path(__file__).resolve().parents[5] / "evidence" / "superseded.json"]:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for e in (d.get("superseded") or []) if isinstance(d, dict) else []:
            out.update(r for r in (e.get("run_ids") or []) if isinstance(r, str))
    return out


def govern(prov: Any, run_ids: Iterable[Any], root: Path | None = None) -> Any:
    if not isinstance(prov, dict) or prov.get("state") == "empty":
        return prov
    runs = sorted({r for r in run_ids if isinstance(r, str) and r})
    if not runs:
        return prov
    if len(runs) > 1:  # a listing: runs withdrawn in superseded.json are not counted (their own view says why)
        sup = superseded_ids(root)
        runs = [r for r in runs if r not in sup] or runs
    checks = {r: check_of(r, root) for r in runs}
    if any(c is None for c in checks.values()):
        return prov
    n, n_unv, n_syn = prov.get("n_records") or 0, prov.get("n_unverified") or 0, prov.get("n_synthetic") or 0
    detail = [f"AWS time written on {n - n_unv - n_syn} of {n} records; check_unit derives the rest from S3 / CloudWatch."]
    if n_syn:
        detail.append(f"Inputs: {n_syn} of {n} records are synthetic fixtures (see the run's declaration).")
    bad = {r: c for r, c in checks.items() if c["parts"].get("c") != "PASS" or "FAIL" in c["parts"].values()}
    if bad:
        why = "; ".join(f"{r}: check_unit " + ", ".join(f"[{k}] {v}" for k, v in c["parts"].items() if v != "PASS")
                        for r, c in sorted(bad.items()))
        return {**prov, "state": "unverified", "label": "UNVERIFIED", "note": f"check_unit did not verify: {why}.",
                "claim": "", "governed_by": "check_unit", "detail": detail}
    return {**prov, "state": "real", "label": "VERIFIED",
            "note": f"check_unit verified {'the run' if len(runs) == 1 else f'all {len(runs)} runs'} shown (parts a-e). " + " ".join(detail),
            "claim": "Verified by check_unit against AWS." if not n_syn else "",
            "governed_by": "check_unit", "detail": detail}
