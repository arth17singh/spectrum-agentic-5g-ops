"""Evidence writing through W6's `tools/evlib.py` (required, no local shim).

evlib validates every record with W1's `team13_contracts.validate_record` and its own checks,
never overwrites declared/result files, and writes records/<run_id>/item.jsonl locally or one
object per record on S3 (A11). This wrapper only picks the store and stores raw model outputs.

    Evidence(root)      root given: a local directory (dry runs, tests)
    Evidence(None)      evlib's default store: EVIDENCE_S3=s3://... or EVIDENCE_DIR=...
"""

from __future__ import annotations

import os
from typing import Iterable

from tools import evlib  # W6; ImportError here is deliberate: no shim

SOURCE = "tools/evlib.py (W6)"
EvidenceError = evlib.EvidenceError
utc_now = evlib.utc_now
code_sha = evlib.code_sha
code_dirty = evlib.code_dirty


class Evidence:
    def __init__(self, root: str | os.PathLike | None = None):
        self.store = evlib.LocalStore(root) if root else evlib.default_store()
        self._out_batches: dict[str, int] = {}

    def write_declared(self, declared: dict) -> str:
        return evlib.write_declared(declared, store=self.store)

    def append_records(self, run_id: str, kind: str, records: Iterable[dict]) -> list[str]:
        return evlib.append_records(run_id, kind, records, store=self.store)

    def write_result(self, result: dict) -> str:
        return evlib.write_result(result, store=self.store)

    def write_outputs(self, run_id: str, rows: list[dict]) -> str:
        """Raw model outputs: outputs/<run_id>/outputs-<nnnn>.jsonl, one new file per batch (S3
        cannot append; each file is written once and never overwritten)."""
        n = self._out_batches.get(run_id, 0)
        self._out_batches[run_id] = n + 1
        data = b"".join(evlib.dumps_line(r) for r in rows)
        return self.store.put_new(f"outputs/{run_id}/outputs-{n:04d}.jsonl", data)

    def read_declared(self, run_id: str) -> dict | None:
        return evlib.read_declared(run_id, store=self.store)

    def read_records(self, run_id: str, kind: str) -> list[dict]:
        return evlib.read_records(run_id, kind, store=self.store)
