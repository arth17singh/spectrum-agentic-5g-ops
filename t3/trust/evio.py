"""Thin adapter over W6's ``tools/evlib.py`` (same function names and paths).

When ``tools.evlib`` is importable (after the merge with int/v4) every call goes to it, so its
validation and stores (EVIDENCE_DIR / EVIDENCE_S3) are the authority. Until then a local fallback
writes the same layout under ``root``: ``records/<run_id>/<kind>.jsonl`` (append) and
``results/<run_id>.json`` (never overwritten), after checking the fields evlib requires for the
kinds the trust gate writes (change, control, result). ``root=None`` keeps everything in memory
(tests).

    w = EvidenceWriter(root)          # or EvidenceWriter() for memory
    w.append_records(run_id, kind, [records])   -> list of written paths
    w.write_result(result)                       -> path
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

try:  # W6's library wins when present
    from tools import evlib as _evlib  # type: ignore
except Exception:  # pragma: no cover - before the merge
    _evlib = None

REQUIRED = {
    "change": ("case_id", "change_id", "item_id", "change", "change_text", "twin", "baseline_kpis",
               "measured_kpis", "measured_delta", "scored", "probe"),
    "control": ("model_id", "version", "action", "by", "reason", "policy_name", "time"),
    "result": ("unit", "metric", "value", "target", "passed", "n_items", "checked_at", "checker"),
}
ENVELOPE = ("schema_version", "kind", "run_id", "team", "created_at", "aws_time", "aws_time_source", "synthetic")


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and abs(v) != float("inf")


def check(rec: dict) -> None:
    kind = rec.get("kind")
    errs = [f"missing {k}" for k in ENVELOPE if k not in rec]
    errs += [f"{kind}: missing {k}" for k in REQUIRED.get(kind, ()) if rec.get(k) is None]
    if kind == "result":
        t = rec.get("target")
        if not (isinstance(t, dict) and t.get("op") in (">=", "<=", "==", ">", "<") and _num(t.get("value"))
                and "unit" in t and t.get("metric")):
            errs.append("result.target must be {metric, op, value, unit}")
        if not _num(rec.get("value")):
            errs.append("result.value must be a JSON number")
        if not isinstance(rec.get("passed"), bool):
            errs.append("result.passed must be true or false")
        c = rec.get("checker") or {}
        if not (c.get("script") and c.get("code_sha")):
            errs.append("result.checker must be {script, code_sha}")
    if kind == "control" and rec.get("action") not in ("pause", "resume"):
        errs.append("control.action must be pause or resume")
    if errs:
        raise ValueError(f"{kind} record invalid: " + "; ".join(errs))
    json.dumps(rec, allow_nan=False)


class EvidenceWriter:
    def __init__(self, root: str | Path | None = None, *, use_evlib: bool = True):
        self.root = Path(root) if root is not None else None
        self.memory: dict[tuple[str, str], list[dict]] = defaultdict(list)
        self._store = None
        if use_evlib and _evlib is not None and self.root is not None:
            self._store = _evlib.LocalStore(self.root)

    @property
    def backend(self) -> str:
        return "tools.evlib" if self._store is not None else ("local-fallback" if self.root else "memory")

    def append_records(self, run_id: str, kind: str, records: list[dict]) -> list[str]:
        if self._store is not None:
            return _evlib.append_records(run_id, kind, records, store=self._store)
        for r in records:
            if r.get("run_id") != run_id or r.get("kind") != kind:
                raise ValueError("record run_id/kind does not match the append call")
            check(r)
        self.memory[(run_id, kind)].extend(records)
        if self.root is None:
            return [f"memory:records/{run_id}/{kind}"]
        p = self.root / "records" / run_id / f"{kind}.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a") as fh:
            for r in records:
                fh.write(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n")
        return [str(p)]

    def write_result(self, result: dict) -> str:
        if self._store is not None:
            return _evlib.write_result(result, store=self._store)
        check(result)
        run_id = result["run_id"]
        if (run_id, "result") in self.memory:
            raise FileExistsError(f"result for {run_id} already written")
        self.memory[(run_id, "result")].append(result)
        if self.root is None:
            return f"memory:results/{run_id}.json"
        p = self.root / "results" / f"{run_id}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "x") as fh:
            fh.write(json.dumps(result, sort_keys=True, indent=1))
        return str(p)
