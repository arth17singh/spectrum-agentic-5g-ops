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

import hashlib
import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path(__file__).resolve().parents[5] / "evidence"
_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._:@+=-]{0,199}$")
_TIME_FIELDS = ("created_at", "time", "aws_time")


_MANIFEST_LOCK = threading.Lock()
_MANIFEST_CACHE: dict[str, tuple[tuple[int, int], dict[str, tuple[str, int]]]] = {}


def _s3_iso(t: Any) -> str | None:
    """S3 LastModified (a datetime from boto3) as whole-second UTC ISO, the form the evidence sync keeps."""
    if not isinstance(t, datetime):
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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
    return [rec for _, rec in _order_keyed(entries)]


def _order_keyed(entries: list[tuple[str, Any]]) -> list[tuple[str, Any]]:
    def sort_key(e):
        t = _record_time(e[1])
        return (t is None, t or 0.0, e[0])
    return sorted(entries, key=sort_key)


def _manifest(root: Path) -> dict[str, tuple[str, int]]:
    """<root>/.sync/manifest.json (written by tools/evidence_sync.py from S3 ListObjectsV2) as
    rel -> (LastModified, size), without the item records (they are never shown with a stored time). The parse is
    cached until the file changes: the manifest is tens of MB and the views poll."""
    p = root / ".sync" / "manifest.json"
    try:
        st = p.stat()
    except OSError:
        return {}
    sig = (st.st_mtime_ns, st.st_size)
    with _MANIFEST_LOCK:
        hit = _MANIFEST_CACHE.get(str(p))
        if hit and hit[0] == sig:
            return hit[1]
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        out: dict[str, tuple[str, int]] = {}
        if isinstance(raw, dict):
            for k, v in raw.items():
                if isinstance(v, dict) and isinstance(v.get("last_modified"), str) and isinstance(v.get("size"), int) \
                        and "/item/" not in k:
                    out[k] = (v["last_modified"], v["size"])
        _MANIFEST_CACHE[str(p)] = (sig, out)
        return out


_ATTEST_CACHE: dict[str, tuple[tuple[int, int], dict[str, tuple[str, str]]]] = {}
_SHA_CACHE: dict[str, tuple[tuple[int, int], str]] = {}


def _run_of(rel: str) -> str:
    """The run a file belongs to: records/<run>/..., results/<run>.json or declared/<run>.json."""
    parts = rel.split("/")
    if parts[0] == "records" and len(parts) > 2:
        return parts[1]
    return parts[1][:-5] if len(parts) > 1 and parts[1].endswith(".json") else ""


def _attest(root: Path, run_id: str) -> dict[str, tuple[str, str]]:
    """<root>/.sync/attest/<run>.json (the evidence sync's reading of the job's `team13-evidence` log lines) as
    rel -> (CloudWatch log time, sha256), for entries whose state is "match" only, without the item records (they
    are never shown with a time). Cached until the file changes: a run's file can be several MB."""
    if not run_id or "/" in run_id or ".." in run_id:
        return {}
    p = root / ".sync" / "attest" / f"{run_id}.json"
    try:
        st = p.stat()
    except OSError:
        return {}
    sig = (st.st_mtime_ns, st.st_size)
    with _MANIFEST_LOCK:
        hit = _ATTEST_CACHE.get(str(p))
        if hit and hit[0] == sig:
            return hit[1]
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        out: dict[str, tuple[str, str]] = {}
        files = raw.get("files") if isinstance(raw, dict) else None
        if isinstance(files, dict):
            for k, v in files.items():
                if isinstance(v, dict) and v.get("state") == "match" and isinstance(v.get("log_time"), str) \
                        and isinstance(v.get("sha256"), str) and "/item/" not in k:
                    out[k] = (v["log_time"], v["sha256"])
        _ATTEST_CACHE[str(p)] = (sig, out)
        return out


def _file_sha(p: Path, st: os.stat_result) -> str | None:
    sig = (st.st_mtime_ns, st.st_size)
    with _MANIFEST_LOCK:
        hit = _SHA_CACHE.get(str(p))
        if hit and hit[0] == sig:
            return hit[1]
    try:
        h = hashlib.sha256(p.read_bytes()).hexdigest()
    except OSError:
        return None
    with _MANIFEST_LOCK:
        _SHA_CACHE[str(p)] = (sig, h)
    return h


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

    def all_records_times(self, kind: str) -> list[tuple[dict, dict]]:
        """Like all_records(kind), each record paired with the AWS times that belong to its own file:
        {"stored": S3 LastModified, "attested": CloudWatch time of the job log line that attests the file}. Each is a
        whole-second UTC ISO string or None.
          stored    only a per-record object (records/<run>/<kind>/<key>.json) gets one; a record read from a .jsonl
                    batch file never does, because the file's time bounds many records and is not theirs. S3 itself
                    for an s3:// root; the evidence sync's manifest for a mirrored directory (and only for a file
                    whose size still matches what S3 stored).
          attested  mirrored directory only: <root>/.sync/attest/<run>.json, written by the evidence sync from the
                    SageMaker job's `team13-evidence` log lines. Used only when the entry's state is "match" AND the
                    sha256 it names is the sha256 of the file on disk."""
        kind = _name(kind, "kind")
        entries: list[tuple[str, Any]] = []
        meta: dict[str, dict] = {}
        for run_id in self.runs():
            es = self._entries(run_id, kind)
            meta.update(self._file_times(f"records/{run_id}/{kind}/", [k for k, _ in es]))
            entries += es
        return [(rec, meta.get(key) or {"stored": None, "attested": None}) for key, rec in _order_keyed(entries)]

    def all_records_stored(self, kind: str) -> list[tuple[dict, str | None]]:
        """all_records_times with only the S3 stored time."""
        return [(rec, m.get("stored")) for rec, m in self.all_records_times(kind)]

    def declared_times(self, run_id: str) -> dict:
        rel = f"declared/{_name(run_id, 'run_id')}.json"
        return self._file_times("declared/", [rel]).get(rel) or {"stored": None, "attested": None}

    def declared_stored(self, run_id: str) -> str | None:
        return self.declared_times(run_id).get("stored")

    def result_times(self, run_id: str) -> dict:
        rel = f"results/{_name(run_id, 'run_id')}.json"
        return self._file_times("results/", [rel]).get(rel) or {"stored": None, "attested": None}

    # ------------------------------------------------------------------ AWS times of a file (S3 stored, job attestation)

    def _file_times(self, prefix: str, rels: list[str]) -> dict[str, dict]:
        rels = [r for r in rels if r.startswith(prefix) and r.endswith(".json") and "#" not in r]
        if not rels:
            return {}
        if self.kind == "file":
            man = _manifest(self.dir)
            out: dict[str, dict] = {}
            for r in rels:
                stored = attested = None
                ent = man.get(r)
                try:
                    st = (self.dir / r).stat()
                except OSError:
                    continue
                if ent and st.st_size == ent[1]:  # else the local file is not what S3 stored
                    stored = ent[0]
                att = _attest(self.dir, _run_of(r)).get(r)
                if att and _file_sha(self.dir / r, st) == att[1]:
                    attested = att[0]
                out[r] = {"stored": stored, "attested": attested}
            return out
        want = set(rels)
        full = self.prefix + prefix
        out = {}
        try:
            for page in self._s3().get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=full):
                for o in page.get("Contents", []):
                    rel = o["Key"][len(self.prefix):]
                    if rel in want and (iso := _s3_iso(o.get("LastModified"))):
                        out[rel] = {"stored": iso, "attested": None}  # the attestation cache lives with a mirror only
        except Exception as e:  # noqa: BLE001
            raise RecordsError(f"cannot list s3://{self.bucket}/{full}: {_describe(e)}") from None
        return out

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
