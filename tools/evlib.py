"""Team13 evidence writer and reader: the one library every stream uses to write evidence.

Layout (the same under a local directory and under an S3 prefix):
  declared/<run_id>.json              the declaration, written BEFORE the run (never overwritten)
  records/<run_id>/<kind>.jsonl       per-item records, one JSON object per line (batch / local writers)
  records/<run_id>/<kind>/<key>.json  one record per object (S3 writers, e.g. Lambdas: S3 cannot append)
  results/<run_id>.json               the pass/fail result (never overwritten)
  models/<model_id>/<version>.json    model registry entries (never overwritten)

Where it writes (first match wins):
  an explicit `store=` argument; EVIDENCE_S3=s3://bucket/prefix/ (boto3, region from AWS_REGION or
  AWS_DEFAULT_REGION, default us-east-2); EVIDENCE_DIR=/some/dir.

Import-light on purpose (Lambdas import it): the standard library only at import time. boto3 is
imported when an S3 store is first used, and `team13_contracts` (stream W1) is used for validation
when importable; otherwise a minimal built-in check of the contract's required fields runs.

Records (contract amendment A11): on S3, append_records writes ONE OBJECT PER RECORD at
records/<run_id>/<kind>/<key>.json with If-None-Match: *, where <key> is the first 24 hex characters of
the sha256 of the record's canonical line, so concurrent Lambdas never overwrite each other and a retried
write of the same record is a no-op. Locally it appends whole lines to records/<run_id>/<kind>.jsonl (a
local JSONL file only ever grows, so the evidence ledger can tell an append from an edit).
write_records_jsonl writes a whole <kind>.jsonl once (batch writers). Every reader merges both layouts:
the JSONL lines first, in file order, then the per-record objects sorted by (created_at, key).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = "1"
TEAM = "team13"
DEFAULT_REGION = "us-east-2"

RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
KIND_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@+-]{0,127}$")  # model_id / version path segments
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SHA40 = re.compile(r"^[0-9a-f]{40}$")
OPS = (">=", "<=", "==", ">", "<")

ENVELOPE = ("schema_version", "kind", "run_id", "team", "created_at", "aws_time", "aws_time_source", "synthetic")
# Required (non-optional) fields per kind, from the build contract. Extra fields are always allowed.
REQUIRED: dict[str, tuple[str, ...]] = {
    "declared": ("unit", "track", "target", "n_items", "test_set_sha256", "model_id", "grader", "code_sha",
                 "region", "declared_at"),
    "item": ("item_id", "input_sha256", "output_sha256", "grader", "backend"),
    "decision": ("case_id", "time", "agent", "tool", "decision", "source"),
    # amendment A15: deltas against the no-change counterfactual; probes (A14) omit the prediction fields
    "change": ("case_id", "change_id", "item_id", "change", "change_text", "twin", "baseline_kpis", "measured_kpis",
               "measured_delta", "scored", "probe"),
    "control": ("model_id", "version", "action", "by", "reason", "policy_name", "time"),
    "job": ("job_name", "arn", "job_kind", "region", "status", "aws_start", "aws_end", "instance_type",
            "billable_s", "est_cost_usd"),
    "model": ("model_id", "version", "model_kind", "artifact_uri", "artifact_sha256", "status", "trained_on",
              "metrics"),
    "result": ("unit", "metric", "value", "target", "passed", "n_items", "checked_at", "checker"),
}
ENUMS: dict[tuple[str, str], tuple] = {
    ("decision", "decision"): ("ALLOW", "DENY"),
    ("control", "action"): ("pause", "resume"),
    ("job", "job_kind"): ("training", "processing", "endpoint", "lambda"),
    ("model", "model_kind"): ("ppo", "t2-response", "llm-lora"),
    ("model", "status"): ("candidate", "approved", "paused", "withdrawn"),
}


class EvidenceError(ValueError):
    """A record failed validation (nothing was written)."""


# --------------------------------------------------------------------------- small helpers

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_time(s: str) -> datetime:
    """ISO-8601 with an explicit UTC offset or Z. Naive times are rejected: they are ambiguous."""
    if not isinstance(s, str):
        raise ValueError(f"time must be a string, got {type(s).__name__}")
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError(f"time {s!r} has no UTC offset")
    return dt.astimezone(timezone.utc)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def new_run_id(unit: str = "run") -> str:
    """<unit>-<UTC yyyymmddThhmmssZ>-<4 hex>, e.g. t3.deny-20261102T203000Z-9f2c."""
    stem = re.sub(r"[^A-Za-z0-9._-]", "-", unit)[:60] or "run"
    return f"{stem}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{secrets.token_hex(2)}"


def _pinned_sha() -> str | None:
    """Inside a SageMaker job started by tools/sm/run.py the code is `git archive <sha>` (no .git), and the
    launcher pins that sha in TEAM13_CODE_SHA. Used ONLY when there is no git work tree: in a repo, git wins."""
    v = os.environ.get("TEAM13_CODE_SHA", "").strip()
    return v if SHA40.match(v) else None


def code_sha(repo: str | os.PathLike | None = None) -> str:
    """The full commit sha of HEAD in the repo that holds this file (or `repo`); in a runner job without
    a .git, the commit the launcher packaged (TEAM13_CODE_SHA)."""
    cwd = Path(repo) if repo else Path(__file__).resolve().parent
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, check=True, capture_output=True,
                              text=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pinned = _pinned_sha()
        if pinned:
            return pinned
        raise


def code_dirty(repo: str | os.PathLike | None = None) -> bool:
    """Uncommitted changes to tracked files. A runner job's archive is the commit itself: never dirty."""
    cwd = Path(repo) if repo else Path(__file__).resolve().parent
    try:
        out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=cwd, check=True,
                             capture_output=True, text=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        if _pinned_sha():
            return False
        raise
    return bool(out.strip())


def dumps_line(obj: dict) -> bytes:
    """One JSONL line: sorted keys, compact, UTF-8, no NaN/Infinity (they are not JSON)."""
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
            + "\n").encode("utf-8")


def dumps_doc(obj: dict) -> bytes:
    return (json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


# --------------------------------------------------------------------------- validation

def _is_str(v: Any) -> bool:
    return isinstance(v, str) and v.strip() != ""


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def builtin_errors(rec: dict) -> list[str]:
    """The minimal check: envelope, the kind's required fields, a few types and enums."""
    errs: list[str] = []
    if not isinstance(rec, dict):
        return ["record must be a JSON object"]
    for k in ENVELOPE:
        if k not in rec:
            errs.append(f"missing envelope field {k}")
    kind = rec.get("kind")
    if rec.get("schema_version") != SCHEMA_VERSION:
        errs.append(f"schema_version must be {SCHEMA_VERSION!r}")
    if rec.get("team") != TEAM:
        errs.append(f"team must be {TEAM!r}")
    if not (isinstance(rec.get("run_id"), str) and RUN_ID_RE.match(rec["run_id"])):
        if not (kind == "model" and rec.get("run_id") in ("", None)):
            errs.append(f"run_id must match {RUN_ID_RE.pattern}")
    if not isinstance(rec.get("synthetic"), bool):
        errs.append("synthetic must be true or false")
    for k in ("created_at",):
        try:
            parse_time(rec.get(k))
        except (ValueError, TypeError):
            errs.append(f"{k} must be an ISO-8601 UTC time")
    if rec.get("aws_time") is not None:
        try:
            parse_time(rec["aws_time"])
        except (ValueError, TypeError):
            errs.append("aws_time must be null or an ISO-8601 UTC time")
        if not _is_str(rec.get("aws_time_source")):
            errs.append("aws_time_source must name the AWS source of aws_time")
    if not (isinstance(kind, str) and KIND_RE.match(kind)):
        errs.append(f"kind must match {KIND_RE.pattern}")
        return errs
    for k in REQUIRED.get(kind, ()):
        if k not in rec or rec[k] is None:
            errs.append(f"{kind}: missing required field {k}")
    for (kd, field), allowed in ENUMS.items():
        if kd == kind and field in rec and rec[field] not in allowed:
            errs.append(f"{kind}.{field} must be one of {', '.join(allowed)}")
    if kind in ("declared", "result"):
        t = rec.get("target")
        if not isinstance(t, dict):
            errs.append("target must be an object {metric, op, value, unit}")
        else:
            if not _is_str(t.get("metric")):
                errs.append("target.metric must be a non-empty string")
            if t.get("op") not in OPS:
                errs.append(f"target.op must be one of {' '.join(OPS)}")
            if not _is_num(t.get("value")):
                errs.append("target.value must be a JSON number")
            if "unit" not in t:
                errs.append("target.unit is required (use \"\" for a ratio)")
    if kind == "declared":
        if not (isinstance(rec.get("test_set_sha256"), str) and HEX64.match(rec["test_set_sha256"])):
            errs.append("test_set_sha256 must be 64 lowercase hex characters")
        if not (isinstance(rec.get("code_sha"), str) and SHA40.match(rec["code_sha"])):
            errs.append("code_sha must be a full 40-character git commit sha")
        if not (isinstance(rec.get("n_items"), int) and not isinstance(rec.get("n_items"), bool)
                and rec["n_items"] >= 0):
            errs.append("n_items must be a non-negative integer")
        for k in ("unit", "track", "model_id", "grader", "region"):
            if k in rec and not _is_str(rec[k]):
                errs.append(f"{k} must be a non-empty string")
    if kind == "result":
        if not _is_num(rec.get("value")):
            errs.append("result.value must be a JSON number")
        if not isinstance(rec.get("passed"), bool):
            errs.append("result.passed must be true or false")
        c = rec.get("checker")
        if not (isinstance(c, dict) and _is_str(c.get("script")) and _is_str(c.get("code_sha"))):
            errs.append("result.checker must be {script, code_sha}")
    if kind == "item":
        for k in ("score", "latency_ms", "cost_usd"):
            if rec.get(k) is not None and not _is_num(rec[k]):
                errs.append(f"item.{k} must be a JSON number or null")
        if rec.get("passed") is not None and not isinstance(rec["passed"], bool):
            errs.append("item.passed must be true, false or null")
        if not _is_str(rec.get("item_id")):
            errs.append("item.item_id must be a non-empty string")
    return errs


def _contracts_validator():
    try:
        import team13_contracts as tc  # stream W1; optional
    except Exception:  # noqa: BLE001 - absent or broken: fall back to the built-in check
        return None
    return getattr(tc, "validate_record", None)


def validate(rec: dict) -> None:
    """Raise EvidenceError if the record is invalid. Uses team13_contracts.validate_record(rec) when
    importable (it must raise on an invalid record), then the built-in check in every case."""
    fn = _contracts_validator()
    if fn is not None:
        try:
            fn(rec)
        except Exception as e:  # noqa: BLE001
            raise EvidenceError(f"{rec.get('kind')}: team13_contracts rejected it: {e}") from e
    errs = builtin_errors(rec)
    if errs:
        raise EvidenceError(f"{rec.get('kind')} record invalid: " + "; ".join(errs))
    try:
        dumps_line(rec)
    except (TypeError, ValueError) as e:
        raise EvidenceError(f"{rec.get('kind')} record is not JSON-serialisable: {e}") from e


def _envelope(rec: dict, kind: str, run_id: str | None) -> dict:
    out = dict(rec)
    if out.get("kind") not in (None, kind):
        raise EvidenceError(f"record kind {out.get('kind')!r} does not match {kind!r}")
    if run_id is not None:
        if out.get("run_id") not in (None, run_id):
            raise EvidenceError(f"record run_id {out.get('run_id')!r} does not match {run_id!r}")
        out["run_id"] = run_id
    out["kind"] = kind
    out.setdefault("schema_version", SCHEMA_VERSION)
    out.setdefault("team", TEAM)
    out.setdefault("created_at", utc_now())
    out.setdefault("aws_time", None)
    out.setdefault("aws_time_source", None)
    out.setdefault("synthetic", False)
    return out


def _check_run_id(run_id: str) -> None:
    if not (isinstance(run_id, str) and RUN_ID_RE.match(run_id)):
        raise EvidenceError(f"run_id {run_id!r} must match {RUN_ID_RE.pattern}")


# --------------------------------------------------------------------------- stores

class LocalStore:
    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)

    def uri(self, rel: str) -> str:
        return str(self.root / rel)

    def put_new(self, rel: str, data: bytes) -> str:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            raise FileExistsError(f"{self.uri(rel)} already exists; evidence is never overwritten") from None
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        return self.uri(rel)

    def append(self, rel: str, data: bytes) -> str:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "ab") as f:  # one write of whole lines
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        return self.uri(rel)

    def get(self, rel: str) -> bytes | None:
        p = self.root / rel
        return p.read_bytes() if p.is_file() else None

    def list(self, prefix: str) -> list[str]:
        base = self.root / prefix
        if not base.is_dir():
            return []
        return sorted(p.relative_to(self.root).as_posix() for p in base.rglob("*") if p.is_file())


class S3Store:
    """put_new = PutObject If-None-Match: * (never overwrites). There is no append on S3."""

    def __init__(self, uri: str, client=None, region: str | None = None):
        if not uri.startswith("s3://"):
            raise ValueError(f"expected s3://bucket/prefix/, got {uri!r}")
        self.bucket, _, prefix = uri[len("s3://"):].partition("/")
        if not self.bucket:
            raise ValueError(f"no bucket in {uri!r}")
        self.prefix = prefix if (not prefix or prefix.endswith("/")) else prefix + "/"
        self.region = region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or DEFAULT_REGION
        self._client = client

    def client(self):
        if self._client is None:
            import boto3

            self._client = boto3.client("s3", region_name=self.region)
        return self._client

    def uri(self, rel: str) -> str:
        return f"s3://{self.bucket}/{self.prefix}{rel}"

    @staticmethod
    def _code(e: Exception) -> str:
        r = getattr(e, "response", None) or {}
        return str((r.get("Error") or {}).get("Code", "")) or str(r.get("ResponseMetadata", {}).get("HTTPStatusCode", ""))

    def put_new(self, rel: str, data: bytes) -> str:
        try:
            self.client().put_object(Bucket=self.bucket, Key=self.prefix + rel, Body=data, IfNoneMatch="*",
                                     ContentType="application/json")
        except Exception as e:  # noqa: BLE001
            if self._code(e) in ("PreconditionFailed", "412", "ConditionalRequestConflict", "409"):
                raise FileExistsError(f"{self.uri(rel)} already exists; evidence is never overwritten") from None
            raise
        return self.uri(rel)

    def _get_with_etag(self, rel: str) -> tuple[bytes | None, str | None]:
        try:
            r = self.client().get_object(Bucket=self.bucket, Key=self.prefix + rel)
        except Exception as e:  # noqa: BLE001
            if self._code(e) in ("NoSuchKey", "404", "NotFound"):
                return None, None
            raise
        return r["Body"].read(), r.get("ETag")

    def get(self, rel: str) -> bytes | None:
        return self._get_with_etag(rel)[0]

    def list(self, prefix: str) -> list[str]:
        out: list[str] = []
        for page in self.client().get_paginator("list_objects_v2").paginate(Bucket=self.bucket,
                                                                            Prefix=self.prefix + prefix):
            out += [o["Key"][len(self.prefix):] for o in page.get("Contents", [])]
        return sorted(out)


def default_store():
    s3 = os.environ.get("EVIDENCE_S3", "").strip()
    if s3:
        return S3Store(s3)
    d = os.environ.get("EVIDENCE_DIR", "").strip()
    if d:
        return LocalStore(d)
    raise RuntimeError("no evidence target: set EVIDENCE_S3=s3://bucket/evidence/ or EVIDENCE_DIR=/path")


# --------------------------------------------------------------------------- writers

def write_declared(declared: dict, store=None) -> str:
    """Write declared/<run_id>.json, stamping declared_at (now, UTC). Refuses to overwrite.
    Returns the written path or s3:// URI."""
    store = store or default_store()
    run_id = declared.get("run_id")
    _check_run_id(run_id)
    rec = _envelope(declared, "declared", run_id)
    rec["declared_at"] = utc_now()
    validate(rec)
    return store.put_new(f"declared/{run_id}.json", dumps_doc(rec))


def _prepare(run_id: str, kind: str, records: Iterable[dict]) -> list[bytes]:
    _check_run_id(run_id)
    if not (isinstance(kind, str) and KIND_RE.match(kind)) or kind in ("declared", "result"):
        raise EvidenceError(f"kind {kind!r} is not a record kind")
    lines = []
    for r in records:
        rec = _envelope(r, kind, run_id)
        validate(rec)
        lines.append(dumps_line(rec))
    if not lines:
        raise EvidenceError("no records to append")
    return lines


def record_key(line: bytes) -> str:
    """Deterministic object key for one record: sha256 of its canonical JSONL line, 24 hex characters."""
    return hashlib.sha256(line).hexdigest()[:24]


def append_records(run_id: str, kind: str, records: Iterable[dict], store=None,
                   layout: str | None = None) -> list[str]:
    """Add records for a run. Every record is validated first; if any is invalid nothing is written.
    layout "objects" (the default on S3): one object per record, records/<run_id>/<kind>/<key>.json,
    written with If-None-Match (re-writing an identical record is a no-op). layout "jsonl" (the default
    locally): append whole lines to records/<run_id>/<kind>.jsonl. Returns the written paths/URIs."""
    store = store or default_store()
    lines = _prepare(run_id, kind, records)
    layout = layout or ("objects" if isinstance(store, S3Store) else "jsonl")
    if layout == "jsonl":
        if isinstance(store, S3Store):
            raise EvidenceError("S3 cannot append: use layout='objects' or write_records_jsonl")
        return [store.append(f"records/{run_id}/{kind}.jsonl", b"".join(lines))]
    if layout != "objects":
        raise EvidenceError(f"unknown layout {layout!r}")
    out = []
    for line in lines:
        rel = f"records/{run_id}/{kind}/{record_key(line)}.json"
        try:
            out.append(store.put_new(rel, line))
        except FileExistsError:
            if store.get(rel) != line:  # same key, different bytes: never silently accept
                raise
            out.append(store.uri(rel))
    return out


def write_records_jsonl(run_id: str, kind: str, records: Iterable[dict], store=None) -> str:
    """Batch writers: write records/<run_id>/<kind>.jsonl once, whole. Refuses to overwrite."""
    store = store or default_store()
    lines = _prepare(run_id, kind, records)
    return store.put_new(f"records/{run_id}/{kind}.jsonl", b"".join(lines))


def write_result(result: dict, store=None) -> str:
    """Write results/<run_id>.json. Refuses to overwrite."""
    store = store or default_store()
    run_id = result.get("run_id")
    _check_run_id(run_id)
    rec = _envelope(result, "result", run_id)
    rec.setdefault("checked_at", utc_now())
    validate(rec)
    return store.put_new(f"results/{run_id}.json", dumps_doc(rec))


def write_model(model: dict, store=None) -> str:
    """Write models/<model_id>/<version>.json. Refuses to overwrite. run_id = the training run id."""
    store = store or default_store()
    mid, ver = model.get("model_id"), model.get("version")
    for name, v in (("model_id", mid), ("version", ver)):
        if not (isinstance(v, str) and NAME_RE.match(v)):
            raise EvidenceError(f"{name} {v!r} must match {NAME_RE.pattern}")
    rec = _envelope(model, "model", model.get("run_id") or "")
    validate(rec)
    return store.put_new(f"models/{mid}/{ver}.json", dumps_doc(rec))


# --------------------------------------------------------------------------- readers

def _load_json(b: bytes | None) -> dict | None:
    return None if b is None else json.loads(b.decode("utf-8"))


def read_declared(run_id: str, store=None) -> dict | None:
    _check_run_id(run_id)
    return _load_json((store or default_store()).get(f"declared/{run_id}.json"))


def read_result(run_id: str, store=None) -> dict | None:
    _check_run_id(run_id)
    return _load_json((store or default_store()).get(f"results/{run_id}.json"))


def read_model(model_id: str, version: str, store=None) -> dict | None:
    return _load_json((store or default_store()).get(f"models/{model_id}/{version}.json"))


def _record_layout(run_id: str, store) -> dict[str, dict]:
    """kind -> {"jsonl": rel or None, "objects": [rel, ...]} for both record layouts."""
    out: dict[str, dict] = {}
    for rel in store.list(f"records/{run_id}/"):
        parts = rel.split("/")
        if len(parts) == 3 and parts[2].endswith(".jsonl"):
            out.setdefault(parts[2][: -len(".jsonl")], {"jsonl": None, "objects": []})["jsonl"] = rel
        elif len(parts) == 4 and parts[3].endswith(".json"):
            out.setdefault(parts[2], {"jsonl": None, "objects": []})["objects"].append(rel)
    return out


def record_kinds(run_id: str, store=None) -> list[str]:
    _check_run_id(run_id)
    return sorted(_record_layout(run_id, store or default_store()))


def read_records(run_id: str, kind: str, store=None) -> list[dict]:
    """Both layouts merged: the <kind>.jsonl lines in file order, then the per-record objects sorted by
    (created_at, key)."""
    _check_run_id(run_id)
    store = store or default_store()
    lay = _record_layout(run_id, store).get(kind)
    if not lay:
        return []
    out: list[dict] = []
    if lay["jsonl"]:
        out += [json.loads(x) for x in (store.get(lay["jsonl"]) or b"").decode("utf-8").splitlines() if x.strip()]
    objs = []
    for rel in lay["objects"]:
        rec = json.loads((store.get(rel) or b"{}").decode("utf-8"))
        objs.append((str(rec.get("created_at", "")), rel, rec))
    out += [r for _, _, r in sorted(objs, key=lambda x: (x[0], x[1]))]
    return out


# --------------------------------------------------------------------------- writer-side metrics
# tools/check_unit.py recomputes these with its OWN code on purpose: the check shares nothing with the writer.

METRICS = ("pass_rate", "accuracy", "pass_count", "fail_count", "n_items", "mean_score", "min_score",
           "max_score", "p50_latency_ms", "p95_latency_ms", "max_latency_ms", "total_cost_usd", "cost_per_1k_usd")


def _nearest_rank(xs: list[float], p: float) -> float:
    s = sorted(xs)
    return s[max(0, math.ceil(p * len(s)) - 1)]


def summarize_items(items: list[dict], metric: str) -> float:
    if not items:
        raise EvidenceError("no items")
    if metric in ("pass_rate", "accuracy", "pass_count", "fail_count"):
        flags = [it.get("passed") for it in items]
        if any(not isinstance(f, bool) for f in flags):
            raise EvidenceError(f"{metric} needs passed=true/false on every item")
        n_pass = sum(1 for f in flags if f)
        return {"pass_count": n_pass, "fail_count": len(flags) - n_pass}.get(metric, n_pass / len(flags))
    if metric == "n_items":
        return len(items)
    field = {"mean_score": "score", "min_score": "score", "max_score": "score", "p50_latency_ms": "latency_ms",
             "p95_latency_ms": "latency_ms", "max_latency_ms": "latency_ms", "total_cost_usd": "cost_usd",
             "cost_per_1k_usd": "cost_usd"}.get(metric)
    if field is None:
        raise EvidenceError(f"unknown metric {metric!r}; known: {', '.join(METRICS)}")
    xs = [it.get(field) for it in items]
    if any(not _is_num(x) for x in xs):
        raise EvidenceError(f"{metric} needs a numeric {field} on every item")
    return {"mean_score": lambda: sum(xs) / len(xs), "min_score": lambda: min(xs), "max_score": lambda: max(xs),
            "p50_latency_ms": lambda: _nearest_rank(xs, 0.50), "p95_latency_ms": lambda: _nearest_rank(xs, 0.95),
            "max_latency_ms": lambda: max(xs), "total_cost_usd": lambda: sum(xs),
            "cost_per_1k_usd": lambda: sum(xs) / len(xs) * 1000}[metric]()


def compare(value: float, op: str, target: float) -> bool:
    return {">=": value >= target, "<=": value <= target, "==": value == target, ">": value > target,
            "<": value < target}[op]


def make_result(declared: dict, items: list[dict], script: str, sha: str | None = None,
                details: dict | None = None) -> dict:
    """A result record computed from the item records, ready for write_result."""
    t = declared["target"]
    value = summarize_items(items, t["metric"])
    return {"run_id": declared["run_id"], "unit": declared["unit"], "metric": t["metric"], "value": value,
            "target": dict(t), "passed": compare(value, t["op"], t["value"]), "n_items": len(items),
            "checked_at": utc_now(), "checker": {"script": script, "code_sha": sha or code_sha()},
            "details": details or {}, "synthetic": bool(declared.get("synthetic", False))}
