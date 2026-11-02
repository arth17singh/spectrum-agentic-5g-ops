"""Evidence writing: W6's `tools/evlib.py` when merged, else a minimal local shim with the same calls.

    write_declared(dict) -> path      declared/<run_id>.json, never overwritten, stamps declared_at
    append_records(run_id, kind, records) -> path     records/<run_id>/<kind>.jsonl, append only
    write_result(dict) -> path        results/<run_id>.json, never overwritten
    read_declared(run_id), read_records(run_id, kind), append_output(run_id, name, lines)

The shim writes only to a local directory (EVIDENCE_DIR or an explicit root) and checks the
fields evlib checks (envelope, required fields, 64-hex test_set_sha256, 40-hex code_sha, numeric
target), so records that pass here are shaped for evlib too. It never talks to AWS.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:  # W6, after merge
    from tools import evlib as _ev  # type: ignore

    SOURCE = "tools/evlib.py (W6)"
except Exception:  # noqa: BLE001
    _ev = None
    SOURCE = "t1/evals/evidence.py (local shim for W6 tools/evlib.py)"

ENVELOPE = ("schema_version", "kind", "run_id", "team", "created_at", "aws_time", "aws_time_source", "synthetic")
REQUIRED = {
    "declared": ("unit", "track", "target", "n_items", "test_set_sha256", "model_id", "grader", "code_sha",
                 "region", "declared_at"),
    "item": ("item_id", "input_sha256", "output_sha256", "grader", "backend"),
    "result": ("unit", "metric", "value", "target", "passed", "n_items", "checked_at", "checker"),
}
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class EvidenceError(ValueError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def code_sha(repo: Path | None = None) -> str:
    cwd = repo or Path(__file__).resolve().parent
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, check=True, capture_output=True,
                          text=True).stdout.strip()


def code_dirty(repo: Path | None = None) -> bool:
    cwd = repo or Path(__file__).resolve().parent
    out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=cwd, check=True,
                         capture_output=True, text=True).stdout
    return bool(out.strip())


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def check(rec: dict) -> None:
    errs = [f"missing {k}" for k in ENVELOPE if k not in rec]
    kind = rec.get("kind")
    errs += [f"{kind}: missing {k}" for k in REQUIRED.get(kind, ()) if rec.get(k) is None]
    if not RUN_ID_RE.match(str(rec.get("run_id", ""))):
        errs.append("bad run_id")
    if not isinstance(rec.get("synthetic"), bool):
        errs.append("synthetic must be bool")
    if kind in ("declared", "result"):
        t = rec.get("target") or {}
        if not (isinstance(t.get("metric"), str) and t.get("op") in (">=", "<=", "==", ">", "<")
                and _num(t.get("value")) and "unit" in t):
            errs.append("target must be {metric, op, value, unit}")
    if kind == "declared":
        if not re.fullmatch(r"[0-9a-f]{64}", str(rec.get("test_set_sha256"))):
            errs.append("test_set_sha256 must be 64 hex")
        if not re.fullmatch(r"[0-9a-f]{40}", str(rec.get("code_sha"))):
            errs.append("code_sha must be 40 hex")
        if not isinstance(rec.get("n_items"), int):
            errs.append("n_items must be int")
    if kind == "item":
        for k in ("score", "latency_ms", "cost_usd"):
            if rec.get(k) is not None and not _num(rec[k]):
                errs.append(f"item.{k} must be a number or null")
        if rec.get("passed") is not None and not isinstance(rec["passed"], bool):
            errs.append("item.passed must be bool or null")
    if kind == "result":
        if not _num(rec.get("value")):
            errs.append("result.value must be a number")
        if not isinstance(rec.get("passed"), bool):
            errs.append("result.passed must be bool")
    if errs:
        raise EvidenceError(f"{kind} record invalid: " + "; ".join(errs))
    json.dumps(rec, allow_nan=False)


def _envelope(rec: dict, kind: str, run_id: str) -> dict:
    out = dict(rec)
    out.update(kind=kind, run_id=run_id)
    out.setdefault("schema_version", "1")
    out.setdefault("team", "team13")
    out.setdefault("created_at", utc_now())
    out.setdefault("aws_time", None)
    out.setdefault("aws_time_source", None)
    out.setdefault("synthetic", False)
    return out


def _root(root: str | os.PathLike | None) -> Path:
    r = root or os.environ.get("EVIDENCE_DIR", "").strip()
    if not r:
        raise RuntimeError("no evidence target: pass --evidence-dir or set EVIDENCE_DIR (or merge W6 evlib)")
    return Path(r)


def _put_new(path: Path, data: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as f:  # "x": never overwrite evidence
        f.write(data)
    return str(path)


def _dump_line(rec: dict) -> str:
    return json.dumps(rec, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n"


class Evidence:
    """One evidence target. With W6's evlib merged and no explicit root, evlib decides the store
    (EVIDENCE_S3 or EVIDENCE_DIR); otherwise everything goes under `root`."""

    def __init__(self, root: str | os.PathLike | None = None):
        self.root = Path(root) if root else None
        self.use_evlib = _ev is not None
        self._store = None
        if self.use_evlib:
            self._store = _ev.LocalStore(self.root) if self.root else _ev.default_store()
        else:
            self.root = _root(root)

    def write_declared(self, declared: dict) -> str:
        if self.use_evlib:
            return _ev.write_declared(declared, store=self._store)
        rec = _envelope(declared, "declared", declared["run_id"])
        rec["declared_at"] = utc_now()
        check(rec)
        return _put_new(self.root / "declared" / f"{rec['run_id']}.json", json.dumps(rec, indent=2, sort_keys=True) + "\n")

    def append_records(self, run_id: str, kind: str, records: Iterable[dict]) -> str:
        if self.use_evlib:
            return _ev.append_records(run_id, kind, records, store=self._store)
        lines = []
        for r in records:
            rec = _envelope(r, kind, run_id)
            check(rec)
            lines.append(_dump_line(rec))
        p = self.root / "records" / run_id / f"{kind}.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write("".join(lines))
        return str(p)

    def write_result(self, result: dict) -> str:
        if self.use_evlib:
            return _ev.write_result(result, store=self._store)
        rec = _envelope(result, "result", result["run_id"])
        rec.setdefault("checked_at", utc_now())
        check(rec)
        return _put_new(self.root / "results" / f"{rec['run_id']}.json", json.dumps(rec, indent=2, sort_keys=True) + "\n")

    def append_output(self, run_id: str, name: str, rows: list[dict]) -> str:
        """Raw model outputs under outputs/<run_id>/<name> (append only)."""
        data = "".join(_dump_line(r) for r in rows).encode("utf-8")
        rel = f"outputs/{run_id}/{name}"
        if self.use_evlib:
            return self._store.append(rel, data)
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "ab") as f:
            f.write(data)
        return str(p)

    def read_declared(self, run_id: str) -> dict | None:
        if self.use_evlib:
            return _ev.read_declared(run_id, store=self._store)
        p = self.root / "declared" / f"{run_id}.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None

    def read_records(self, run_id: str, kind: str) -> list[dict]:
        if self.use_evlib:
            return _ev.read_records(run_id, kind, store=self._store)
        p = self.root / "records" / run_id / f"{kind}.jsonl"
        if not p.is_file():
            return []
        return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
