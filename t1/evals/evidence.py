"""Evidence writing through W6's `tools/evlib.py` (required, no local shim).

evlib validates every record with W1's `team13_contracts.validate_record` and its own checks,
never overwrites declared/result files, and writes records/<run_id>/item.jsonl locally or one
object per record on S3 (A11). This wrapper picks the stores:

    Evidence(root)   declared, records and result go to a local evlib store at `root`
                     (`--evidence-dir {evidence_dir}` under tools/sm: the job uploads and attests them).
                     When EVIDENCE_S3 is also set (it is in every runner job), raw model outputs go
                     straight to S3 (the runner uploads only records and results), and a baseline's
                     declaration and items are read from S3 when the local store does not have them.
    Evidence(None)   evlib's default store: EVIDENCE_S3=s3://... or EVIDENCE_DIR=... for everything.
                     Not for runner jobs: the launcher has already written declared/<run_id>.json there.
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
        s3 = os.environ.get("EVIDENCE_S3", "").strip()
        self.side = evlib.S3Store(s3) if (root and s3) else None
        self._out_batches: dict[str, int] = {}

    def write_declared(self, declared: dict) -> str:
        return evlib.write_declared(declared, store=self.store)

    def append_records(self, run_id: str, kind: str, records: Iterable[dict]) -> list[str]:
        return evlib.append_records(run_id, kind, records, store=self.store)

    def write_result(self, result: dict) -> str:
        return evlib.write_result(result, store=self.store)

    def put_output(self, rel: str, data: bytes) -> str:
        """A file under outputs/<run_id>/, written once (S3 when running under the runner)."""
        return (self.side or self.store).put_new(rel, data)

    def write_outputs(self, run_id: str, rows: list[dict]) -> str:
        """Raw model outputs: outputs/<run_id>/outputs-<nnnn>.jsonl, one new file per batch (S3
        cannot append; each file is written once and never overwritten)."""
        n = self._out_batches.get(run_id, 0)
        self._out_batches[run_id] = n + 1
        data = b"".join(evlib.dumps_line(r) for r in rows)
        return self.put_output(f"outputs/{run_id}/outputs-{n:04d}.jsonl", data)

    def get_output(self, rel: str) -> bytes | None:
        """A file under outputs/ from the local store, else from S3 when running under the runner."""
        b = self.store.get(rel)
        if b is None and self.side is not None:
            b = self.side.get(rel)
        return b

    def read_declared(self, run_id: str) -> dict | None:
        d = evlib.read_declared(run_id, store=self.store)
        if d is None and self.side is not None:
            d = evlib.read_declared(run_id, store=self.side)
        return d

    def read_records(self, run_id: str, kind: str) -> list[dict]:
        recs = evlib.read_records(run_id, kind, store=self.store)
        if not recs and self.side is not None:
            recs = evlib.read_records(run_id, kind, store=self.side)
        return recs
