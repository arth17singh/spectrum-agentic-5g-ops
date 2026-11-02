"""Evidence writer with the same call shape as W6's `tools/evlib.append_records`.

    writer.append_records(run_id, kind, [record, ...])

Selection (`from_env`):
- `T3_EVIDENCE=local:<dir>` writes `<dir>/records/<run_id>/<kind>.jsonl`;
- `T3_EVIDENCE=s3://<bucket>/<prefix>` writes `<prefix>/records/<run_id>/<kind>.jsonl` (read, append,
  put with If-Match/If-None-Match, retried: S3 has no append);
- unset: `tools.evlib.append_records` if it can be imported (the canonical writer), else an error.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Callable, Protocol

SAFE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")


class Writer(Protocol):
    def append_records(self, run_id: str, kind: str, records: list[dict]) -> str: ...


def _check(run_id: str, kind: str) -> None:
    for v in (run_id, kind):
        if not isinstance(v, str) or not SAFE.fullmatch(v):
            raise ValueError(f"unsafe path part {v!r}")


def _lines(records: list[dict]) -> str:
    return "".join(json.dumps(r, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n" for r in records)


def _code(exc: Exception) -> str | None:
    return (getattr(exc, "response", None) or {}).get("Error", {}).get("Code")


class LocalWriter:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def append_records(self, run_id: str, kind: str, records: list[dict]) -> str:
        _check(run_id, kind)
        path = self.root / "records" / run_id / f"{kind}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(_lines(records))
        return str(path)


class S3Writer:
    def __init__(self, bucket: str, prefix: str, client=None):
        self.bucket, self.prefix = bucket, prefix.strip("/")
        self._client = client

    @property
    def client(self):
        if self._client is None:
            import boto3

            self._client = boto3.client("s3")
        return self._client

    def append_records(self, run_id: str, kind: str, records: list[dict]) -> str:
        _check(run_id, kind)
        key = "/".join(p for p in (self.prefix, "records", run_id, f"{kind}.jsonl") if p)
        for _ in range(5):
            try:
                obj = self.client.get_object(Bucket=self.bucket, Key=key)
                old, cond = obj["Body"].read().decode(), {"IfMatch": obj["ETag"]}
            except Exception as exc:  # noqa: BLE001
                if _code(exc) not in ("NoSuchKey", "404"):
                    raise
                old, cond = "", {"IfNoneMatch": "*"}
            try:
                self.client.put_object(Bucket=self.bucket, Key=key, Body=(old + _lines(records)).encode(),
                                       ContentType="application/x-ndjson", **cond)
                return f"s3://{self.bucket}/{key}"
            except Exception as exc:  # a concurrent writer won: read again
                if _code(exc) not in ("PreconditionFailed", "ConditionalRequestConflict"):
                    raise
        raise RuntimeError(f"could not append to s3://{self.bucket}/{key}")


class FuncWriter:
    def __init__(self, fn: Callable[[str, str, list[dict]], object]):
        self.fn = fn

    def append_records(self, run_id: str, kind: str, records: list[dict]) -> str:
        out = self.fn(run_id, kind, records)
        return str(out) if out is not None else "evlib"


def from_env(env: dict[str, str] | None = None) -> Writer:
    env = os.environ if env is None else env
    spec = env.get("T3_EVIDENCE", "")
    if spec.startswith("local:"):
        return LocalWriter(spec[len("local:"):])
    if spec.startswith("s3://"):
        bucket, _, prefix = spec[len("s3://"):].partition("/")
        return S3Writer(bucket, prefix or "evidence")
    if spec:
        raise ValueError(f"T3_EVIDENCE={spec!r}: use local:<dir> or s3://bucket/prefix")
    try:
        from tools.evlib import append_records  # W6's canonical writer
    except ImportError as exc:
        raise RuntimeError("no evidence writer: set T3_EVIDENCE or ship tools/evlib.py") from exc
    return FuncWriter(append_records)
