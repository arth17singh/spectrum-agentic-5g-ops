"""Where evidence artifacts are read from: a local directory (default) or S3 (read-only, opt-in).

Both sources use the same layout, relative to the evidence root (or the S3 prefix):
  declared/<run_id>.json        declaration, written before the run
  outputs/<run_id>/...          per-item outputs
  records/<run_id>/...          per-run records
  logs/<run_id>/...             logs
  judging.json                  the run ids submitted for judging (at most 20)
Anything else is ignored. Both return Artifact objects with the file's SHA-256, size and source time.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .store import RESERVED, utc_iso

RUN_DIRS = {"outputs": "output", "records": "record", "logs": "log"}
JSON_KINDS = {"declaration", "judging"}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
MAX_JUDGED = 20


@dataclass
class Artifact:
    path: str  # posix, relative to the evidence root / S3 prefix
    kind: str
    run_id: str
    sha256: str
    size: int
    source_time: str  # ISO UTC: file mtime (local) or S3 LastModified
    source: str  # "local" | "s3"
    version_id: str | None = None
    content: bytes | None = field(default=None, repr=False)  # kept only for JSON kinds


def classify(rel: str) -> tuple[str, str] | None:
    """(kind, run_id) for a path in the evidence layout, or None to ignore it."""
    parts = rel.split("/")
    if any(p.startswith(".") or p == "" for p in parts):
        return None
    if len(parts) == 1:
        if rel == "judging.json":
            return "judging", ""
        return None  # README.md, ledger.jsonl, anchors.jsonl, public_keys.json, anything else
    if parts[0] == "declared" and len(parts) == 2 and parts[1].endswith(".json"):
        return "declaration", parts[1][: -len(".json")]
    if parts[0] in RUN_DIRS and len(parts) >= 3:
        return RUN_DIRS[parts[0]], parts[1]
    return None


# --------------------------------------------------------------------------- validation

def _is_str(v) -> bool:
    return isinstance(v, str) and v.strip() != ""


def validate_declaration(content: bytes, run_id_from_name: str) -> tuple[dict | None, list[str]]:
    """(declaration, errors). The declaration keeps only the schema fields."""
    try:
        d = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        return None, [f"not valid JSON: {e}"]
    if not isinstance(d, dict):
        return None, ["top level must be a JSON object"]
    errors: list[str] = []
    for k in ("run_id", "unit", "model_id", "grader"):
        if not _is_str(d.get(k)):
            errors.append(f"{k} must be a non-empty string")
    if _is_str(d.get("run_id")) and d["run_id"] != run_id_from_name:
        errors.append(f"run_id {d['run_id']!r} does not match the file name {run_id_from_name!r}")
    t = d.get("target")
    if not isinstance(t, dict):
        errors.append("target must be an object with metric and value")
    else:
        if not _is_str(t.get("metric")):
            errors.append("target.metric must be a non-empty string")
        if isinstance(t.get("value"), bool) or not isinstance(t.get("value"), (int, float)):
            errors.append("target.value must be a number")
    h = d.get("test_set_sha256")
    if not (isinstance(h, str) and HEX64.match(h)):
        errors.append("test_set_sha256 must be 64 lowercase hex characters")
    if errors:
        return None, errors
    return {"run_id": d["run_id"], "unit": d["unit"], "target": {"metric": t["metric"], "value": t["value"]},
            "test_set_sha256": h, "model_id": d["model_id"], "grader": d["grader"]}, []


def validate_judging(content: bytes) -> tuple[list[str] | None, list[str]]:
    try:
        d = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        return None, [f"not valid JSON: {e}"]
    ids = d.get("run_ids") if isinstance(d, dict) else d
    if not isinstance(ids, list) or not all(_is_str(x) for x in ids):
        return None, ["expected a list of run ids, or an object with a run_ids list"]
    if len(ids) > MAX_JUDGED:
        return None, [f"{len(ids)} run ids listed; at most {MAX_JUDGED} are allowed"]
    return ids, []


# --------------------------------------------------------------------------- sources

class LocalSource:
    kind = "local"

    def __init__(self, root: Path):
        self.root = root
        self._cache: dict[str, tuple[int, int, str]] = {}  # rel -> (mtime_ns, size, sha256)

    def describe(self) -> dict:
        return {"kind": self.kind, "uri": str(self.root), "time_basis": "file modification time"}

    def list_artifacts(self) -> list[Artifact]:
        out: list[Artifact] = []
        if not self.root.exists():
            return out
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
            for name in sorted(filenames):
                p = Path(dirpath) / name
                rel = p.relative_to(self.root).as_posix()
                if rel in RESERVED:
                    continue
                c = classify(rel)
                if not c or not p.is_file():
                    continue
                kind, run_id = c
                st = p.stat()
                cached = self._cache.get(rel)
                content = p.read_bytes() if kind in JSON_KINDS else None
                if content is not None:
                    sha = hashlib.sha256(content).hexdigest()
                elif cached and cached[:2] == (st.st_mtime_ns, st.st_size):
                    sha = cached[2]
                else:
                    sha = _file_sha(p)
                self._cache[rel] = (st.st_mtime_ns, st.st_size, sha)
                out.append(Artifact(rel, kind, run_id, sha, st.st_size,
                                    utc_iso(datetime.fromtimestamp(st.st_mtime, timezone.utc)), "local",
                                    content=content))
        return out


def _file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"expected s3://bucket/prefix/, got {uri!r}")
    bucket, _, prefix = uri[len("s3://"):].partition("/")
    if not bucket:
        raise ValueError(f"no bucket in {uri!r}")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return bucket, prefix


class S3Source:
    """Read-only: ListObjectsV2 + GetObject only. Enabled only by SPECTRUM_EVIDENCE_S3=s3://bucket/prefix/.
    boto3 is imported lazily and uses whatever profile/credentials the environment provides."""

    kind = "s3"

    def __init__(self, uri: str, client=None):
        self.uri = uri
        self.bucket, self.prefix = parse_s3_uri(uri)
        self._client = client
        # key -> (etag, last_modified iso, sha256, size, version_id, content)
        self._cache: dict[str, tuple] = {}

    def describe(self) -> dict:
        return {"kind": self.kind, "uri": self.uri, "time_basis": "S3 LastModified"}

    def client(self):
        if self._client is None:
            import boto3  # optional dependency: uv sync --extra aws

            self._client = boto3.client("s3")
        return self._client

    def list_artifacts(self) -> list[Artifact]:
        s3 = self.client()
        out: list[Artifact] = []
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                rel = key[len(self.prefix):]
                c = classify(rel)
                if not c or rel in RESERVED:
                    continue
                kind, run_id = c
                lm = utc_iso(obj["LastModified"])
                etag = obj.get("ETag", "")
                hit = self._cache.get(key)
                if hit and hit[0] == etag and hit[1] == lm:
                    _, _, sha, size, version_id, content = hit
                else:
                    resp = s3.get_object(Bucket=self.bucket, Key=key)
                    body = resp["Body"].read()
                    sha, size, version_id = hashlib.sha256(body).hexdigest(), len(body), resp.get("VersionId")
                    content = body if kind in JSON_KINDS else None
                    self._cache[key] = (etag, lm, sha, size, version_id, content)
                out.append(Artifact(rel, kind, run_id, sha, size, lm, "s3", version_id, content))
        return out


def default_source(root: Path) -> LocalSource | S3Source:
    uri = os.environ.get("SPECTRUM_EVIDENCE_S3", "").strip()
    return S3Source(uri) if uri else LocalSource(root)
