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


def superseded_info(root: Path | None = None) -> dict[str, dict]:
    """run_id -> {"reason", "by": [run ids]} from the superseded.json files the evidence code keeps (the mirror's, the
    repo's evidence/ and the incident-suite one): the one list that says which runs must not be used for a claim."""
    root = root or _root()
    repo = Path(__file__).resolve().parents[5]
    out: dict[str, dict] = {}
    for p in ([root / "superseded.json"] if root else []) + [repo / "evidence" / "superseded.json", repo / "t3" / "evals" / "evidence" / "superseded.json"]:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for e in (d.get("superseded") or []) if isinstance(d, dict) else []:
            if not isinstance(e, dict):
                continue
            by = [b for b in ((e.get("superseded_by") or {}).get("run_ids") or []) if isinstance(b, str)]
            for rid in e.get("run_ids") or []:
                if isinstance(rid, str):
                    out.setdefault(rid, {"reason": str(e.get("reason") or ""), "by": by})
    return out


def hide_superseded(groups: list[list[dict]], show: bool, root: Path | None = None) -> tuple[list[list[dict]], list[dict]]:
    """Drop the records of superseded runs from each group unless `show`. Returns the groups and the superseded runs
    that have records at all ([{run_id, by}]), so a page can offer to show them and say how many it is hiding."""
    info = superseded_info(root)
    present = sorted({r["run_id"] for g in groups for r in g if isinstance(r.get("run_id"), str) and r["run_id"] in info})
    if not show:
        groups = [[r for r in g if r.get("run_id") not in info] for g in groups]
    return groups, [{"run_id": x, "by": info[x]["by"]} for x in present]


def superseded_provenance(run_id: str, info: dict, root: Path | None = None) -> dict:
    """What a screen says about one superseded run: SUPERSEDED, by what, and which check_unit parts it failed (from the
    cached check). Never UNVERIFIED: the attempt was withdrawn, not left unchecked."""
    by = info.get("by") or []
    chk = check_of(run_id, root)
    failed = sorted(k for k, v in ((chk or {}).get("parts") or {}).items() if v == "FAIL")
    note = "replaced by " + (", ".join(by) if by else "a later run")
    if failed:
        note += "; this attempt failed check_unit part" + ("s " if len(failed) > 1 else " ") + ", ".join(failed)
    return {"state": "superseded", "label": "SUPERSEDED", "note": note + ".", "claim": "", "governed_by": "superseded.json",
            "superseded_by": by, "reason": info.get("reason") or ""}


def govern(prov: Any, run_ids: Iterable[Any], root: Path | None = None) -> Any:
    if not isinstance(prov, dict) or prov.get("state") == "empty":
        return prov
    runs = sorted({r for r in run_ids if isinstance(r, str) and r})
    if not runs:
        return prov
    if len(runs) == 1 and runs[0] in (sup_info := superseded_info(root)):
        return superseded_provenance(runs[0], sup_info[runs[0]], root)
    if len(runs) > 1:  # a listing: runs withdrawn in superseded.json are not counted (their own view says why)
        sup = superseded_ids(root)
        runs = [r for r in runs if r not in sup] or runs
    checks = {r: check_of(r, root) for r in runs}
    if any(c is None for c in checks.values()):
        return prov
    n, n_unv, n_syn = prov.get("n_records") or 0, prov.get("n_unverified") or 0, prov.get("n_synthetic") or 0
    agg = prov.get("aws_time_summary") if isinstance(prov.get("aws_time_summary"), dict) else None
    if agg:  # the same count the banner shows, so the note and the banner cannot disagree
        detail = [f"{agg['text']}." + (" check_unit derives the others from S3 / CloudWatch." if agg.get("k", 0) < agg.get("n", 0) else "")]
    else:
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
