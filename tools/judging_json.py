#!/usr/bin/env python3
"""judging_json.py: fill `run_ids` and `reported_results` in judging.json from the result records.

Reads `results/<run_id>.json` (contract kind "result") from the evidence store and writes ONLY those two
fields; every other field, including the comment field that lists the planned slots, is left as it is.
Read-only on AWS: s3:ListBucket and s3:GetObject. Nothing is written to S3 (copy the file to
`judging/judging.json` in the bucket yourself once it is final).

Selection, in this order:
  1. every result record in the store (a file whose contents are not a valid result is SKIPPED and reported);
  2. results marked synthetic are dropped unless --allow-synthetic;
  3. run ids in a superseded list (--superseded FILE, the format of t3/evals/evidence/superseded.json) or
     given with --exclude are dropped;
  4. with --include, only those run ids are kept;
  5. the order is --order-file (one run id per line) first, then the rest by created_at.
More than --max-runs (20) selected is an error: trim with --include or --exclude. Nothing is written when a
file was skipped or the count is too high, unless --skip-bad is given for the former.

Usage:
  judging_json.py --s3 s3://<bucket>/evidence/ [--judging judging.json] [--dry-run]
  judging_json.py --dir evidence [--exclude RUN_ID ...] [--include RUN_ID ...] [--dry-run]
Exit codes: 0 ok (or dry run ok); 1 nothing written because of a problem; 2 could not read the store.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_JUDGING = HERE.parent / "judging.json"
MAX_RUNS = 20


class StoreError(Exception):
    pass


# --- reading the store ------------------------------------------------------------------------------
def read_dir(root: Path) -> dict[str, bytes]:
    d = root / "results"
    if not d.is_dir():
        raise StoreError(f"no results/ folder under {root}")
    return {p.name: p.read_bytes() for p in sorted(d.glob("*.json"))}


def read_s3(uri: str) -> dict[str, bytes]:
    if not uri.startswith("s3://"):
        raise StoreError("--s3 must look like s3://bucket/evidence/")
    bucket, _, prefix = uri[5:].partition("/")
    prefix = prefix.rstrip("/") + "/results/" if prefix else "results/"
    try:
        import boto3  # imported late: the offline path and the tests need no AWS libraries
        s3 = boto3.client("s3")
        out: dict[str, bytes] = {}
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                name = obj["Key"][len(prefix):]
                if name.endswith(".json") and "/" not in name:
                    out[name] = s3.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read()
        return dict(sorted(out.items()))
    except StoreError:
        raise
    except Exception as exc:  # no credentials, no access, no network: say so, never guess
        raise StoreError(f"could not read the S3 store ({type(exc).__name__})") from exc


# --- turning records into rows -----------------------------------------------------------------------
def parse_result(name: str, raw: bytes) -> dict[str, Any]:
    try:
        rec = json.loads(raw)
    except ValueError as exc:
        raise ValueError("not valid JSON") from exc
    if not isinstance(rec, dict) or rec.get("kind") != "result":
        raise ValueError("not a result record (kind != 'result')")
    run_id = rec.get("run_id")
    if not isinstance(run_id, str) or name != f"{run_id}.json":
        raise ValueError("run_id does not match the file name")
    for k in ("unit", "metric", "value", "passed", "n_items", "target"):
        if k not in rec:
            raise ValueError(f"missing field {k}")
    return rec


def row_of(rec: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run_id": rec["run_id"],
        "unit": rec["unit"],
        "metric": rec["metric"],
        "value": rec["value"],
        "target": rec["target"],
        "passed": rec["passed"],
        "n_items": rec["n_items"],
        "checked_at": rec.get("checked_at"),
    }
    gain = (rec.get("details") or {}).get("gain")
    if isinstance(gain, dict):
        row["gain"] = {k: gain[k] for k in ("value", "ci_low", "ci_high", "baseline_run_id") if k in gain}
    return row


def superseded_ids(path: Path) -> set[str]:
    doc = json.loads(path.read_text())
    ids: set[str] = set()
    for entry in doc.get("superseded", []):
        ids.update(entry.get("run_ids", []))
    return ids


def select(raws: dict[str, bytes], *, allow_synthetic: bool, exclude: set[str], include: set[str] | None,
           order: list[str]) -> tuple[list[dict[str, Any]], list[tuple[str, str]], list[tuple[str, str]]]:
    """Return (rows, skipped files with reason, dropped run ids with reason)."""
    recs, skipped, dropped = [], [], []
    for name, raw in raws.items():
        try:
            recs.append(parse_result(name, raw))
        except ValueError as exc:
            skipped.append((name, str(exc)))
    keep = []
    for rec in recs:
        rid = rec["run_id"]
        if rec.get("synthetic") and not allow_synthetic:
            dropped.append((rid, "synthetic"))
        elif rid in exclude:
            dropped.append((rid, "excluded or superseded"))
        elif include is not None and rid not in include:
            dropped.append((rid, "not in --include"))
        else:
            keep.append(rec)
    rank = {rid: i for i, rid in enumerate(order)}
    keep.sort(key=lambda r: (rank.get(r["run_id"], len(rank)), str(r.get("created_at") or ""), r["run_id"]))
    return [row_of(r) for r in keep], skipped, dropped


def _ids(entry: dict) -> list[str]:
    ids = list(entry.get("run_ids") or [])
    if entry.get("run_id"):
        ids.append(entry["run_id"])
    return ids


def slots_report(path: Path) -> int:
    """Check the optional `slots` and `edge_cases` lists against `run_ids` and print them. Exit 1 if they disagree.

    An entry is filled when status is "filled" and it names run ids (`run_id` or `run_ids`), all of which are
    listed in `run_ids`; a pending entry names none. Runs an entry only compares against (`bar_run_id`,
    `base_run_id`) are supporting runs and are not required in `run_ids`.
    """
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        print(f"judging_json: cannot read {path}: {exc}", file=sys.stderr)
        return 2
    slots, edges = doc.get("slots"), doc.get("edge_cases", [])
    if not isinstance(slots, list) or not isinstance(edges, list):
        print("judging_json: no `slots` list in judging.json", file=sys.stderr)
        return 2
    run_ids = set(doc.get("run_ids", []))
    bad, listed = [], set()

    def show(entry: dict) -> bool:
        ids = _ids(entry)
        filled = entry.get("status") == "filled"
        if filled != bool(ids) or (filled and not set(ids) <= run_ids):
            bad.append(entry.get("slot"))
        listed.update(ids)
        detail = ", ".join(ids) if filled else f"pending: {entry.get('waiting_for', '')}"
        print(f"  {str(entry.get('slot')):>3}  {'FILLED ' if filled else 'pending'}  {entry.get('name')}  ->  {detail}")
        return filled

    print("slots:")
    n_filled = sum(show(sl) for sl in slots)
    if edges:
        print("edge cases:")
    e_filled = sum(show(e) for e in edges)
    print(f"{n_filled} of {len(slots)} slots filled; {e_filled} of {len(edges)} edge cases filled; {len(run_ids)} run id(s) listed")
    extra = sorted(run_ids - listed)
    if extra:
        print(f"run ids listed but in no slot or edge case: {', '.join(extra)}")
    if bad or extra:
        print(f"judging_json: slots and run_ids disagree (entries {bad})", file=sys.stderr)
        return 1
    return 0


# --- main ---------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--s3", help="evidence prefix, s3://bucket/evidence/")
    src.add_argument("--dir", type=Path, help="local evidence directory (with results/)")
    ap.add_argument("--judging", type=Path, default=DEFAULT_JUDGING)
    ap.add_argument("--dry-run", action="store_true", help="show what would change; write nothing")
    ap.add_argument("--include", nargs="+", metavar="RUN_ID")
    ap.add_argument("--exclude", nargs="+", default=[], metavar="RUN_ID")
    ap.add_argument("--superseded", type=Path, help="a superseded.json whose run ids are dropped")
    ap.add_argument("--order-file", type=Path, help="run ids, one per line, in the order to list them")
    ap.add_argument("--allow-synthetic", action="store_true")
    ap.add_argument("--skip-bad", action="store_true", help="write even if some result files were unreadable")
    ap.add_argument("--max-runs", type=int, default=MAX_RUNS)
    ap.add_argument("--slots-report", action="store_true",
                    help="print which of the planned slots in judging.json are filled; reads no store, writes nothing")
    args = ap.parse_args(argv)

    if args.slots_report:
        return slots_report(args.judging)
    if not (args.s3 or args.dir):
        ap.error("one of --s3 or --dir is required (or --slots-report)")

    try:
        judging = json.loads(args.judging.read_text())
        if not isinstance(judging, dict):
            raise ValueError("judging.json is not an object")
        raws = read_s3(args.s3) if args.s3 else read_dir(args.dir)
        exclude = set(args.exclude)
        if args.superseded:
            exclude |= superseded_ids(args.superseded)
        order = [ln.strip() for ln in args.order_file.read_text().splitlines() if ln.strip()] if args.order_file else []
    except (StoreError, OSError, ValueError) as exc:
        print(f"judging_json: cannot start: {exc}", file=sys.stderr)
        return 2

    rows, skipped, dropped = select(raws, allow_synthetic=args.allow_synthetic, exclude=exclude,
                                    include=set(args.include) if args.include else None, order=order)
    run_ids = [r["run_id"] for r in rows]

    print(f"{len(raws)} result file(s) read; {len(rows)} selected; {len(dropped)} dropped; {len(skipped)} skipped")
    for rid, why in dropped:
        print(f"  dropped  {rid}  ({why})")
    for name, why in skipped:
        print(f"  SKIPPED  {name}  ({why})")
    for r in rows:
        verdict = "PASS" if r["passed"] else "FAIL"
        print(f"  {verdict}  {r['run_id']}  {r['unit']}  {r['metric']}={r['value']}  n={r['n_items']}")

    problems = []
    if len(rows) > args.max_runs:
        problems.append(f"{len(rows)} runs selected, the limit is {args.max_runs}: trim with --include or --exclude")
    if skipped and not args.skip_bad:
        problems.append("some result files were skipped: fix them or pass --skip-bad")
    for p in problems:
        print(f"judging_json: {p}", file=sys.stderr)

    old_ids = judging.get("run_ids", [])
    print(f"run_ids: {len(old_ids)} -> {len(run_ids)}; reported_results: {len(judging.get('reported_results', []))} -> {len(rows)}")
    if args.dry_run:
        print("dry run: nothing written")
        return 1 if problems else 0
    if problems:
        print("nothing written", file=sys.stderr)
        return 1
    judging["run_ids"] = run_ids
    judging["reported_results"] = rows
    args.judging.write_text(json.dumps(judging, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {args.judging}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
