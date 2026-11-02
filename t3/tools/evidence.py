"""Evidence writer for the tool Lambda: one object per record (contract amendment 1, A11).

    writer.put_record(run_id, kind, record_key, record) -> location

Layout: `records/<run_id>/<kind>/<record_key>.json` (S3 cannot append; readers merge this layout
with the batch writers' `records/<run_id>/<kind>.jsonl`). A record is never overwritten: one
approval applies one change, so a second write of the same key is an error.

Selection (`from_env`, `T3_EVIDENCE`):
- `local:<dir>` writes under <dir>;
- `s3://<bucket>/<prefix>` writes under s3://<bucket>/<prefix>/ (prefix default `evidence`).
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Protocol

SAFE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")


class Writer(Protocol):
    def put_record(self, run_id: str, kind: str, record_key: str, record: dict) -> str: ...


class RecordExists(RuntimeError):
    pass


def _rel(run_id: str, kind: str, record_key: str) -> str:
    for v in (run_id, kind, record_key):
        if not isinstance(v, str) or not SAFE.fullmatch(v):
            raise ValueError(f"unsafe path part {v!r}")
    return f"records/{run_id}/{kind}/{record_key}.json"


def _body(record: dict) -> bytes:
    return (json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _code(exc: Exception) -> str | None:
    return (getattr(exc, "response", None) or {}).get("Error", {}).get("Code")


class LocalWriter:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def put_record(self, run_id: str, kind: str, record_key: str, record: dict) -> str:
        path = self.root / _rel(run_id, kind, record_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("xb") as f:
                f.write(_body(record))
        except FileExistsError as exc:
            raise RecordExists(str(path)) from exc
        return str(path)


class S3Writer:
    def __init__(self, bucket: str, prefix: str = "evidence", client=None):
        self.bucket, self.prefix = bucket, prefix.strip("/")
        self._client = client

    @property
    def client(self):
        if self._client is None:
            import boto3

            self._client = boto3.client("s3")
        return self._client

    def put_record(self, run_id: str, kind: str, record_key: str, record: dict) -> str:
        key = "/".join(p for p in (self.prefix, _rel(run_id, kind, record_key)) if p)
        try:
            self.client.put_object(Bucket=self.bucket, Key=key, Body=_body(record),
                                   ContentType="application/json", IfNoneMatch="*")
        except Exception as exc:
            if _code(exc) in ("PreconditionFailed", "ConditionalRequestConflict"):
                raise RecordExists(f"s3://{self.bucket}/{key}") from exc
            raise
        return f"s3://{self.bucket}/{key}"


def from_env(env: dict[str, str] | None = None) -> Writer:
    env = os.environ if env is None else env
    spec = env.get("T3_EVIDENCE", "")
    if spec.startswith("local:"):
        return LocalWriter(spec[len("local:"):])
    if spec.startswith("s3://"):
        bucket, _, prefix = spec[len("s3://"):].partition("/")
        return S3Writer(bucket, prefix or "evidence")
    raise ValueError(f"T3_EVIDENCE={spec!r}: use local:<dir> or s3://bucket/prefix")
