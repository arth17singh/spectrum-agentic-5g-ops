"""Evidence writer: W6's ``tools/evlib.py`` when importable, else a minimal local SHIM (until merge).

The shim implements only what the evals use, with the contract's layout under a local directory:
``declared/<run_id>.json``, ``records/<run_id>/<kind>.jsonl`` (append), ``results/<run_id>.json``.
It fills the envelope (schema_version, kind, run_id, team, created_at, aws_time=None,
aws_time_source=None, synthetic) and refuses to overwrite declarations and results.
"""

from __future__ import annotations

import json
import os
import secrets
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:  # stream W6 (merged)
    from tools import evlib as _ev  # type: ignore[import-not-found]

    EVLIB = "tools/evlib.py"
except ImportError:
    _ev = None
    EVLIB = "t3/evals/evshim.py (local shim)"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


_SHA40 = "0123456789abcdef"


def pinned_sha() -> str | None:
    """The commit a runner job was packaged from (``CODE_SHA``, else ``TEAM13_CODE_SHA``), if set.

    A SageMaker job runs a ``git archive`` with no ``.git``: it must never call git, so the pinned sha
    wins whenever it is present (the archive IS that commit, clean by construction)."""
    for name in ("CODE_SHA", "TEAM13_CODE_SHA"):
        v = os.environ.get(name, "").strip()
        if v:
            if len(v) != 40 or any(c not in _SHA40 for c in v):
                raise ValueError(f"{name}={v!r} is not a full 40-character commit sha")
            return v
    return None


def code_sha() -> str:
    pinned = pinned_sha()
    if pinned:
        return pinned
    if _ev is not None:
        return _ev.code_sha()
    here = Path(__file__).resolve().parent  # local shim only (W6 not merged)
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True,
                          check=True).stdout.strip()


def code_dirty() -> bool:
    """Uncommitted changes to tracked files (evlib's rule). Never dirty in a runner job."""
    if pinned_sha():
        return False
    if _ev is not None:
        return _ev.code_dirty()
    here = Path(__file__).resolve().parent  # local shim only (W6 not merged)
    out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=here,
                         capture_output=True, text=True, check=True).stdout
    return bool(out.strip())


def new_run_id(unit: str) -> str:
    if _ev is not None:
        return _ev.new_run_id(unit)
    return f"{unit}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{secrets.token_hex(2)}"


class _Local:
    def __init__(self, root: str | os.PathLike) -> None:
        self.root = Path(root)

    def _env(self, rec: dict[str, Any], kind: str, run_id: str) -> dict[str, Any]:
        out = dict(rec)
        out.update(kind=kind, run_id=run_id)
        out.setdefault("schema_version", "1")
        out.setdefault("team", "team13")
        out.setdefault("created_at", utc_now())
        out.setdefault("aws_time", None)
        out.setdefault("aws_time_source", None)
        out.setdefault("synthetic", False)
        return out

    def _new(self, rel: str, obj: dict[str, Any]) -> str:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "x") as f:  # refuses to overwrite
            f.write(json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + "\n")
        return str(p)

    def write_declared(self, d: dict[str, Any]) -> str:
        rec = self._env(d, "declared", d["run_id"])
        rec["declared_at"] = utc_now()
        return self._new(f"declared/{d['run_id']}.json", rec)

    def append_records(self, run_id: str, kind: str, records: Iterable[dict[str, Any]]) -> str:
        p = self.root / f"records/{run_id}/{kind}.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(self._env(r, kind, run_id), sort_keys=True, separators=(",", ":"), allow_nan=False)
                 for r in records]
        with open(p, "a") as f:
            f.write("".join(x + "\n" for x in lines))
        return str(p)

    def write_result(self, r: dict[str, Any]) -> str:
        rec = self._env(r, "result", r["run_id"])
        rec.setdefault("checked_at", utc_now())
        return self._new(f"results/{r['run_id']}.json", rec)

    def read_records(self, run_id: str, kind: str) -> list[dict[str, Any]]:
        p = self.root / f"records/{run_id}/{kind}.jsonl"
        return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []

    def read_declared(self, run_id: str) -> dict[str, Any] | None:
        p = self.root / f"declared/{run_id}.json"
        return json.loads(p.read_text()) if p.exists() else None


class Evidence:
    """One evidence target (a local directory). Delegates to W6's evlib when it is importable."""

    def __init__(self, root: str | os.PathLike) -> None:
        self.root = Path(root)
        self._local = _Local(root)
        self._store = _ev.LocalStore(root) if _ev is not None else None

    def write_declared(self, d: dict[str, Any]) -> str:
        return _ev.write_declared(d, store=self._store) if _ev else self._local.write_declared(d)

    def append_records(self, run_id: str, kind: str, records: list[dict[str, Any]]) -> str:
        if _ev:
            return _ev.append_records(run_id, kind, records, store=self._store)
        return self._local.append_records(run_id, kind, records)

    def write_result(self, r: dict[str, Any]) -> str:
        return _ev.write_result(r, store=self._store) if _ev else self._local.write_result(r)

    def read_records(self, run_id: str, kind: str) -> list[dict[str, Any]]:
        if _ev:
            return _ev.read_records(run_id, kind, store=self._store)
        return self._local.read_records(run_id, kind)

    def read_declared(self, run_id: str) -> dict[str, Any] | None:
        return _ev.read_declared(run_id, store=self._store) if _ev else self._local.read_declared(run_id)
