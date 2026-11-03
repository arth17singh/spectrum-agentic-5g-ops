"""RecordsReader: one read-only view of the Track 3 evidence store, local or S3, for every slice.

Root: the `root` argument, else env SPECTRUM_RECORDS_ROOT, else the repo's evidence/ directory.
  file:///abs/dir        a local directory (no AWS needed)
  s3://bucket/prefix/    read-only via boto3 (region from SPECTRUM_RECORDS_REGION, AWS_REGION or
                         AWS_DEFAULT_REGION; credentials from the usual chain, e.g. AWS_PROFILE)
  /abs/dir               a bare path is the same as file://

Layout under the root (BUILD_CONTRACT storage section + amendment A11):
  declared/<run_id>.json
  records/<run_id>/<kind>.jsonl                  batch writers, one record per line
  records/<run_id>/<kind>/<record_key>.json      Lambdas, one object per record (S3 cannot append)
  results/<run_id>.json
  models/<model_id>/<version>.json

records() merges both record layouts, ordered by record time (created_at, else time, else aws_time;
records with no parseable time last), then by a stable key (the object path, line number for jsonl).

Failures are loud: a malformed JSON line or object, an S3 permission or network error, and an s3://
root while SPECTRUM_OFFLINE=1 all raise RecordsError. Only a missing file or key reads as "absent"
(None, or an empty list).
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path(__file__).resolve().parents[5] / "evidence"
_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._:@+=-]{0,199}$")
_TIME_FIELDS = ("created_at", "time", "aws_time")


class RecordsError(RuntimeError):
    """The store could not be read (bad data, no access, offline). Never an empty success."""


def _name(value: str, what: str) -> str:
    if not isinstance(value, str) or not _NAME.match(value) or ".." in value:
        raise ValueError(f"invalid {what}: {value!r}")
    return value


def _record_time(rec: Any) -> float | None:
    if not isinstance(rec, dict):
        return None
    for f in _TIME_FIELDS:
        v = rec.get(f)
        if isinstance(v, str) and v:
            try:
                dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
            except ValueError:
                continue
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
    return None


def _order(entries: list[tuple[str, Any]]) -> list[Any]:
    """entries: (stable key, record). Timed records first by time, then key; untimed last by key."""
    def sort_key(e):
        t = _record_time(e[1])
        return (t is None, t or 0.0, e[0])
    return [rec for _, rec in sorted(entries, key=sort_key)]


def _parse_json(text: str, where: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise RecordsError(f"malformed JSON in {where}: {e}") from None


def _parse_jsonl(text: str, where: str) -> list[tuple[str, Any]]:
    out = []
    for n, line in enumerate(text.splitlines(), 1):
        if line.strip():
            out.append((f"{where}#{n:08d}", _parse_json(line, f"{where} line {n}")))
    return out


class RecordsReader:
    def __init__(self, root: str | None = None, client: Any = None):
        raw = root or os.environ.get("SPECTRUM_RECORDS_ROOT") or str(DEFAULT_ROOT)
        if raw.startswith("s3://"):
            bucket, _, prefix = raw[len("s3://"):].partition("/")
            if not bucket:
                raise ValueError(f"no bucket in {raw!r}")
            self.kind = "s3"
            self.bucket = bucket
            self.prefix = prefix.strip("/") + "/" if prefix.strip("/") else ""
            self.root = f"s3://{bucket}/{self.prefix}"
            self._client = client
        else:
            path = raw[len("file://"):] if raw.startswith("file://") else raw
            if not os.path.isabs(path):
                raise ValueError(f"records root must be absolute: {raw!r}")
            self.kind = "file"
            self.dir = Path(path)
            self.root = str(self.dir)

    # ------------------------------------------------------------------ public API

    def source(self) -> dict:
        return {"kind": self.kind, "root": self.root}

    def runs(self) -> list[str]:
        ids = set()
        for name in self._list_names("declared/") + self._list_names("results/"):
            if name.endswith(".json"):
                ids.add(name[:-5])
        for name in self._list_names("records/"):
            ids.add(name.rstrip("/"))
        return sorted(i for i in ids if _NAME.match(i))

    def declared(self, run_id: str) -> dict | None:
        return self._read_json(f"declared/{_name(run_id, 'run_id')}.json")

    def result(self, run_id: str) -> dict | None:
        return self._read_json(f"results/{_name(run_id, 'run_id')}.json")

    def model(self, model_id: str, version: str) -> dict | None:
        return self._read_json(f"models/{_name(model_id, 'model_id')}/{_name(version, 'version')}.json")

    def records(self, run_id: str, kind: str) -> list[dict]:
        return _order(self._entries(_name(run_id, "run_id"), _name(kind, "kind")))

    def all_records(self, kind: str) -> list[dict]:
        kind = _name(kind, "kind")
        entries = []
        for run_id in self.runs():
            entries += self._entries(run_id, kind)
        return _order(entries)

    # ------------------------------------------------------------------ layout

    def _entries(self, run_id: str, kind: str) -> list[tuple[str, Any]]:
        base = f"records/{run_id}/"
        out: list[tuple[str, Any]] = []
        text = self._read_text(f"{base}{kind}.jsonl")
        if text is not None:
            out += _parse_jsonl(text, f"{base}{kind}.jsonl")
        for name in self._list_names(f"{base}{kind}/"):
            if name.endswith(".json"):
                rel = f"{base}{kind}/{name}"
                obj = self._read_text(rel)
                if obj is None:  # deleted between list and read
                    continue
                out.append((rel, _parse_json(obj, rel)))
        return out

    def _read_json(self, rel: str) -> dict | None:
        text = self._read_text(rel)
        return None if text is None else _parse_json(text, rel)

    # ------------------------------------------------------------------ backends

    def _read_text(self, rel: str) -> str | None:
        if self.kind == "file":
            p = self.dir / rel
            try:
                return p.read_text(encoding="utf-8")
            except FileNotFoundError:
                return None
            except (OSError, UnicodeDecodeError) as e:
                raise RecordsError(f"cannot read {p}: {e}") from None
        key = self.prefix + rel
        try:
            obj = self._s3().get_object(Bucket=self.bucket, Key=key)
            return obj["Body"].read().decode("utf-8")
        except Exception as e:  # noqa: BLE001 - every failure except a missing key is an error
            if _s3_code(e) in ("NoSuchKey", "404", "NotFound"):
                return None
            raise RecordsError(f"cannot read s3://{self.bucket}/{key}: {_describe(e)}") from None

    def _list_names(self, rel_dir: str) -> list[str]:
        """Immediate children of rel_dir: file names, and sub-directory names ending in '/'."""
        if self.kind == "file":
            d = self.dir / rel_dir
            try:
                return sorted(c.name + ("/" if c.is_dir() else "") for c in d.iterdir())
            except (FileNotFoundError, NotADirectoryError):
                return []
            except OSError as e:
                raise RecordsError(f"cannot list {d}: {e}") from None
        prefix = self.prefix + rel_dir
        names = []
        try:
            pages = self._s3().get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=prefix, Delimiter="/")
            for page in pages:
                names += [c["Prefix"][len(prefix):] for c in page.get("CommonPrefixes", [])]
                names += [o["Key"][len(prefix):] for o in page.get("Contents", []) if o["Key"] != prefix]
        except Exception as e:  # noqa: BLE001
            raise RecordsError(f"cannot list s3://{self.bucket}/{prefix}: {_describe(e)}") from None
        return sorted(names)

    def _s3(self):
        if self._client is None:
            if os.environ.get("SPECTRUM_OFFLINE") == "1":
                raise RecordsError(f"SPECTRUM_OFFLINE=1: refusing to read {self.root} from S3")
            import boto3  # lazy: the file kind never needs AWS
            region = (os.environ.get("SPECTRUM_RECORDS_REGION") or os.environ.get("AWS_REGION")
                      or os.environ.get("AWS_DEFAULT_REGION"))
            self._client = boto3.client("s3", region_name=region)
        return self._client


def _s3_code(e: Exception) -> str | None:
    resp = getattr(e, "response", None)
    if isinstance(resp, dict):
        return str(resp.get("Error", {}).get("Code"))
    return None


def _describe(e: Exception) -> str:
    code = _s3_code(e)
    return f"{code}: {e}" if code else f"{type(e).__name__}: {e}"
