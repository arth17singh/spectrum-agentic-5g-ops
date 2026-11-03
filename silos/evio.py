"""Evidence plumbing shared by the silo job entry points (service-neutral).

  * `get_evlib()` loads `tools/evlib.py` (the team's evidence writer) from the repo, or from the job package;
  * environment contract of every job: RUN_ID (or EVIDENCE_RUN_ID), CODE_SHA (40 hex, no git history is
    available inside a job), EVIDENCE_S3 (or EVIDENCE_DIR), AWS_REGION;
  * `Sink` collects records, validates them with evlib, and writes each kind once at the end;
  * `declaration()` builds a declared/<run_id>.json document; the job runner writes it BEFORE the run.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

_CACHE: dict[str, Any] = {}
TRACK = "track-2"
SHA40 = re.compile(r"^[0-9a-f]{40}$")


def get_evlib():
    if "evlib" in _CACHE:
        return _CACHE["evlib"]
    try:
        from tools import evlib  # type: ignore

        if hasattr(evlib, "append_records"):
            _CACHE["evlib"] = evlib
            return evlib
    except ImportError:
        pass
    here = Path(__file__).resolve().parent
    for cand in (here.parent / "tools" / "evlib.py", here / "evlib.py"):
        if cand.is_file():
            spec = importlib.util.spec_from_file_location("team13_tools_evlib", cand)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["team13_tools_evlib"] = mod
            spec.loader.exec_module(mod)
            _CACHE["evlib"] = mod
            return mod
    raise ImportError("tools/evlib.py is not available (package it next to silos/, see silos/launch.py)")


def env_run_id() -> str | None:
    return os.environ.get("RUN_ID") or os.environ.get("EVIDENCE_RUN_ID")


def env_code_sha(required: bool = True) -> str:
    sha = os.environ.get("CODE_SHA", "").strip().lower()
    if SHA40.match(sha):
        return sha
    if required:
        raise SystemExit("CODE_SHA must be the 40-hex git commit this code was packaged from (a job has no git history)")
    return "0" * 40


def region() -> str:
    return os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-2"


def declaration(*, run_id: str, unit: str, target: dict, n_items: int, test_set_sha256: str, model_id: str, grader: str,
                code_sha: str, extra: dict | None = None) -> dict:
    d = {"run_id": run_id, "unit": unit, "track": TRACK, "target": target, "n_items": int(n_items),
         "test_set_sha256": test_set_sha256, "model_id": model_id, "grader": grader, "code_sha": code_sha,
         "region": region(), "synthetic": False,
         "population": "simulated silos: three simulated operators on the in-process Maveric twin, run in one account; not partner data"}
    d.update(extra or {})
    return d


def declare_first(declared: dict) -> None:
    """Write the declaration via evlib, then wait so its S3 LastModified is strictly earlier than the records."""
    get_evlib().write_declared(declared)
    time.sleep(2.0)


class Sink:
    """Collect records in memory, validate them with evlib, write each kind once at the end (batch JSONL)."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.records: dict[str, list[dict]] = {}

    def add(self, kind: str, recs: list[dict]) -> None:
        self.records.setdefault(kind, []).extend(recs)

    def validate(self) -> None:
        ev = get_evlib()
        for kind, recs in self.records.items():
            ev._prepare(self.run_id, kind, recs)

    def flush(self, store=None) -> list[str]:
        ev = get_evlib()
        store = store or ev.default_store()
        out = []
        for kind, recs in sorted(self.records.items()):
            if recs:
                out.append(ev.write_records_jsonl(self.run_id, kind, recs, store=store))
        return out


def item(*, item_id: str, input_sha256: str, output_sha256: str, grader: str, backend: str, **fields: Any) -> dict:
    d = {"item_id": item_id, "input_sha256": input_sha256, "output_sha256": output_sha256, "grader": grader, "backend": backend}
    d.update(fields)
    return d


def finish(*, declared: dict, items: list[dict], script: str, code_sha: str, details: dict, dry_run: bool, sink: Sink,
           extra_records: dict[str, list[dict]] | None = None) -> dict:
    """Build the result from the item records with evlib.make_result (so check_unit recomputes it), validate all
    records, and (unless dry_run) write items, extra records and the result. Returns a summary dict."""
    ev = get_evlib()
    sink.add("item", items)
    for kind, recs in (extra_records or {}).items():
        sink.add(kind, recs)
    result = ev.make_result(declared, items, script, code_sha, details)
    sink.validate()
    ev.validate(ev._envelope(result, "result", declared["run_id"]))
    written: list[str] = []
    if not dry_run:
        written = sink.flush()
        written.append(ev.write_result(result))
    return {"result": result, "written": written, "n_items": len(items)}


def print_json(obj: Any) -> None:
    print(json.dumps(obj, indent=1, sort_keys=True, default=str))
