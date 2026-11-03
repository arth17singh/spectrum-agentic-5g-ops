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
        """Write each kind. On S3: one object per record (evlib's "objects" layout, so check_unit can give every record
        an AWS time from its own LastModified), 32 puts in parallel. Locally: one JSONL file per kind."""
        ev = get_evlib()
        store = store or ev.default_store()
        out: list[str] = []
        for kind, recs in sorted(self.records.items()):
            if not recs:
                continue
            if isinstance(store, ev.S3Store):
                out += put_objects(ev, store, self.run_id, kind, recs)
            else:
                out.append(ev.write_records_jsonl(self.run_id, kind, recs, store=store))
        return out


def put_objects(ev, store, run_id: str, kind: str, recs: list[dict], workers: int = 32) -> list[str]:
    """What evlib.append_records(layout="objects") does, concurrently (it puts one record at a time)."""
    from concurrent.futures import ThreadPoolExecutor

    lines = ev._prepare(run_id, kind, recs)

    def one(line: bytes) -> str:
        rel = f"records/{run_id}/{kind}/{ev.record_key(line)}.json"
        try:
            return store.put_new(rel, line)
        except FileExistsError:
            if store.get(rel) != line:  # same key, different bytes: never silently accept
                raise
            return store.uri(rel)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, lines))


def read_records_fast(run_id: str, kind: str, store=None, workers: int = 32) -> list[dict]:
    """evlib.read_records, with the per-record objects fetched concurrently (a score run has one per test row)."""
    from concurrent.futures import ThreadPoolExecutor

    ev = get_evlib()
    store = store or ev.default_store()
    lay = ev._record_layout(run_id, store).get(kind)
    if not lay:
        return []
    out: list[dict] = []
    if lay["jsonl"]:
        out += [json.loads(x) for x in (store.get(lay["jsonl"]) or b"").decode("utf-8").splitlines() if x.strip()]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        out += list(pool.map(lambda rel: json.loads((store.get(rel) or b"{}").decode("utf-8")), sorted(lay["objects"])))
    return out


RUNNER_KEYS = ("unit", "track", "target", "baseline_run_id", "n_items", "n_cases", "test_set_sha256", "subset", "prompt_sha256",
               "parser", "decoding", "model_id", "grader", "twin", "synthetic")
JUDGED = ("unit", "track", "target", "n_items", "test_set_sha256", "model_id", "grader", "code_sha")


def runner_declaration(decl: dict, twin: dict | None = None) -> dict:
    """The declaration in the form the job runner (`python -m tools.sm.run --declaration`) accepts: only the contract's
    keys, no run_id / code_sha / region (the runner sets them). Everything else we say about the run (population label,
    note, scales, comparison details) travels in twin.config.details."""
    out = {k: decl[k] for k in RUNNER_KEYS if k in decl}
    extras = {k: v for k, v in decl.items() if k not in RUNNER_KEYS and k not in ("run_id", "code_sha", "region")}
    t = dict(twin or {"kind": "none", "version": "none", "scenario": "none", "seed": 0})
    cfg = dict(t.get("config") or {})
    cfg["details"] = extras
    t["config"] = cfg
    out["twin"] = t
    out.setdefault("synthetic", False)
    return out


def reconcile(declared: dict) -> dict:
    """Inside a runner job the declaration already stored in S3 is the one check_unit judges. Refuse to run if it
    disagrees with what this code would have declared; return the stored one so the result is built from it."""
    if not os.environ.get("EVIDENCE_S3"):
        return declared
    ev = get_evlib()
    stored = ev.read_declared(declared["run_id"], ev.default_store())
    if stored is None:
        return declared
    bad = [k for k in JUDGED if stored.get(k) != declared.get(k)]
    if bad:
        raise SystemExit("the stored declaration disagrees with this code on " + ", ".join(
            f"{k}: stored {stored.get(k)!r} vs code {declared.get(k)!r}" for k in bad) + "; refusing to run")
    return stored


def item(*, item_id: str, input_sha256: str, output_sha256: str, grader: str, backend: str, **fields: Any) -> dict:
    d = {"item_id": item_id, "input_sha256": input_sha256, "output_sha256": output_sha256, "grader": grader, "backend": backend}
    d.update(fields)
    return d


def finish(*, declared: dict, items: list[dict], script: str, code_sha: str, details: dict, dry_run: bool, sink: Sink,
           extra_records: dict[str, list[dict]] | None = None) -> dict:
    """Build the result from the item records with evlib.make_result (so check_unit recomputes it), validate all
    records, and (unless dry_run) write items, extra records and the result. Returns a summary dict."""
    ev = get_evlib()
    declared = reconcile(declared)
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
