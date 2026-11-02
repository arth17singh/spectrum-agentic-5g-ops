"""STAND-IN for W6's ``tools/evlib.append_records``. Not the evidence library.

Writes ``<root>/records/<run_id>/<kind>.jsonl`` (the contract's storage layout), one JSON line
per record. Also usable as an in-memory sink (``root=None``) for tests.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path


class LocalRecordWriter:
    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root is not None else None
        self.records: dict[tuple[str, str], list[dict]] = defaultdict(list)

    def append_records(self, run_id: str, kind: str, records: list[dict]) -> None:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", run_id) or not re.fullmatch(r"[a-z_]+", kind):
            raise ValueError(f"bad run_id or kind: {run_id!r} {kind!r}")
        for r in records:
            if r.get("run_id") != run_id or r.get("kind") != kind:
                raise ValueError("record run_id/kind does not match the append call")
        self.records[(run_id, kind)].extend(records)
        if self.root is not None:
            d = self.root / "records" / run_id
            d.mkdir(parents=True, exist_ok=True)
            with open(d / f"{kind}.jsonl", "a") as fh:
                for r in records:
                    fh.write(json.dumps(r, sort_keys=True) + "\n")

    __call__ = append_records
