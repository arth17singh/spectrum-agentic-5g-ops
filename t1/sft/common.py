"""Shared helpers for the t1/sft jobs: S3 or local paths, sha256 manifests, summaries, evidence.

boto3, torch and transformers are imported inside functions, so --dry-run and the tests need
neither. Every artifact a job writes goes to an explicit prefix (S3 or a local directory) and is
never overwritten: an existing key is an error, because evidence is written once.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

DEFAULT_REGION = "us-east-2"

# The event's base model (CHEDDAR base-model/otel_model_revision.txt names this repo and revision).
BASE_HF_REPO = "farbodtavakkoli/OTel-LLM-1.7B-IT"
BASE_REVISION = "2a2d417b4c84d7ea609d6026a864cdf666ab1491"
BASE_MODEL_ID = "otel-llm-1.7b-it"
LORA_MODEL_ID = "t1-otel-1.7b-lora"

RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class JobError(RuntimeError):
    """A job precondition or check failed; the job stops before writing a misleading artifact."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def log(msg: str) -> None:
    """One line to stdout (CloudWatch in a job), prefixed with the UTC time."""
    print(f"[{utc_now()}] {msg}", flush=True)


def check_run_id(run_id: str) -> str:
    if not (isinstance(run_id, str) and RUN_ID_RE.match(run_id)):
        raise JobError(f"run id {run_id!r} must match {RUN_ID_RE.pattern}")
    return run_id


def code_sha_from_env(explicit: str | None = None) -> str | None:
    """The commit the job runs. A job has no .git, so the runner passes it (--code-sha or CODE_SHA)."""
    sha = explicit or os.environ.get("CODE_SHA") or None
    if sha is not None and not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise JobError(f"code sha {sha!r} must be a full 40-character git sha")
    return sha


# --------------------------------------------------------------------------- hashing

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_files(root: str | os.PathLike) -> list[Path]:
    root = Path(root)
    return sorted(p for p in root.rglob("*") if p.is_file())


def sha256_tree(root: str | os.PathLike, exclude: Iterable[str] = ()) -> dict[str, dict[str, Any]]:
    """{relative posix name: {size, sha256}} for every file under root, sorted by name."""
    root = Path(root)
    skip = set(exclude)
    out: dict[str, dict[str, Any]] = {}
    for p in tree_files(root):
        rel = p.relative_to(root).as_posix()
        if rel in skip:
            continue
        out[rel] = {"size": p.stat().st_size, "sha256": sha256_file(p)}
    return out


def sha256sums_text(files: dict[str, dict[str, Any]]) -> str:
    """`sha256sum` format, sorted by name: "<sha256>  <name>\\n" per file."""
    return "".join(f"{files[n]['sha256']}  {n}\n" for n in sorted(files))


def write_sha256sums(root: str | os.PathLike, name: str = "SHA256SUMS") -> tuple[dict[str, dict[str, Any]], str]:
    """Hash every file under root, write root/SHA256SUMS, return (files, sha256 of SHA256SUMS)."""
    root = Path(root)
    files = sha256_tree(root, exclude=[name])
    text = sha256sums_text(files).encode("utf-8")
    (root / name).write_bytes(text)
    return files, sha256_bytes(text)


# --------------------------------------------------------------------------- JSON

def dumps_doc(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def write_json(path: str | os.PathLike, obj: Any) -> str:
    """Write one JSON document; returns its sha256."""
    data = dumps_doc(obj).encode("utf-8")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(data)
    return sha256_bytes(data)


# --------------------------------------------------------------------------- S3 and local locations

def is_s3(uri: str | None) -> bool:
    return isinstance(uri, str) and uri.startswith("s3://")


def split_s3(uri: str) -> tuple[str, str]:
    """("bucket", "key/prefix/") from s3://bucket/key/prefix/ (a trailing slash is added)."""
    if not is_s3(uri):
        raise JobError(f"expected s3://bucket/prefix/, got {uri!r}")
    bucket, _, prefix = uri[len("s3://"):].partition("/")
    if not bucket:
        raise JobError(f"no bucket in {uri!r}")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return bucket, prefix


def s3_client(region: str | None = None):
    import boto3
    from botocore.config import Config

    return boto3.client("s3", region_name=region or os.environ.get("AWS_REGION") or DEFAULT_REGION,
                        config=Config(retries={"max_attempts": 8, "mode": "standard"}))


def list_s3(uri: str, region: str | None = None, client=None) -> list[dict[str, Any]]:
    """Every object under the prefix: [{name (relative), key, size, etag, last_modified}]."""
    bucket, prefix = split_s3(uri)
    c = client or s3_client(region)
    out = []
    for page in c.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for o in page.get("Contents", []):
            if o["Key"].endswith("/"):
                continue
            out.append({"name": o["Key"][len(prefix):], "key": o["Key"], "size": o["Size"],
                        "etag": o["ETag"].strip('"'),
                        "last_modified": o["LastModified"].astimezone(timezone.utc).isoformat().replace("+00:00", "Z")})
    return sorted(out, key=lambda o: o["name"])


def download_s3(uri: str, dest: str | os.PathLike, region: str | None = None, client=None) -> list[dict[str, Any]]:
    """Download every object under the prefix into dest (same relative names); returns the listing."""
    bucket, _ = split_s3(uri)
    c = client or s3_client(region)
    objs = list_s3(uri, client=c)
    if not objs:
        raise JobError(f"nothing under {uri}")
    dest = Path(dest)
    for o in objs:
        target = dest / o["name"]
        target.parent.mkdir(parents=True, exist_ok=True)
        c.download_file(bucket, o["key"], str(target))
    return objs


def _exists_s3(c, bucket: str, key: str) -> bool:
    try:
        c.head_object(Bucket=bucket, Key=key)
        return True
    except Exception as e:  # noqa: BLE001
        code = str(getattr(e, "response", {}).get("Error", {}).get("Code", ""))
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def upload_tree(src: str | os.PathLike, dest: str, region: str | None = None, client=None) -> list[str]:
    """Copy every file under src to dest (s3:// prefix or a local directory). Refuses to overwrite:
    every target is checked first and nothing is written if any exists."""
    src = Path(src)
    files = tree_files(src)
    rels = [p.relative_to(src).as_posix() for p in files]
    if not is_s3(dest):
        d = Path(dest)
        clash = [r for r in rels if (d / r).exists()]
        if clash:
            raise JobError(f"refusing to overwrite {len(clash)} file(s) under {d}: {clash[:3]}")
        for p, r in zip(files, rels):
            (d / r).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, d / r)
        return [str(d / r) for r in rels]
    bucket, prefix = split_s3(dest)
    c = client or s3_client(region)
    clash = [r for r in rels if _exists_s3(c, bucket, prefix + r)]
    if clash:
        raise JobError(f"refusing to overwrite {len(clash)} object(s) under {dest}: {clash[:3]}")
    from boto3.s3.transfer import TransferConfig

    cfg = TransferConfig(multipart_threshold=64 << 20, multipart_chunksize=64 << 20, max_concurrency=8)
    out = []
    for p, r in zip(files, rels):
        c.upload_file(str(p), bucket, prefix + r, Config=cfg)
        out.append(f"s3://{bucket}/{prefix}{r}")
    return out


def s3_last_modified(uri: str, region: str | None = None, client=None) -> str:
    """LastModified of one object (an AWS time for records), ISO-8601 UTC with Z."""
    bucket, _, key = uri[len("s3://"):].partition("/")
    c = client or s3_client(region)
    r = c.head_object(Bucket=bucket, Key=key)
    return r["LastModified"].astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def materialise(uri_or_path: str | None, workdir: str | os.PathLike, name: str, channel_env: str | None = None,
                region: str | None = None) -> tuple[Path, list[dict[str, Any]] | None]:
    """A local directory holding the input: a SageMaker channel (SM_CHANNEL_<X>) if the runner
    mounted one, else an s3:// prefix downloaded to workdir/name, else a local path as given.
    Returns (path, S3 listing or None)."""
    if channel_env and os.environ.get(channel_env) and not uri_or_path:
        return Path(os.environ[channel_env]), None
    if not uri_or_path:
        raise JobError(f"no location for input {name!r} (and {channel_env or 'no channel'} is unset)")
    if is_s3(uri_or_path):
        dest = Path(workdir) / name
        listing = download_s3(uri_or_path, dest, region)
        return dest, listing
    p = Path(uri_or_path)
    if not p.exists():
        raise JobError(f"input {name!r}: {p} does not exist")
    return p, None


# --------------------------------------------------------------------------- environment

def versions(*modules: str) -> dict[str, str | None]:
    import importlib.metadata as md

    out: dict[str, str | None] = {"python": sys.version.split()[0]}
    for m in modules:
        try:
            out[m] = md.version(m)
        except md.PackageNotFoundError:
            out[m] = None
    return out


def evidence_store(evidence: str | None):
    """evlib store for records: --evidence (s3:// prefix or a directory), else evlib's default
    (EVIDENCE_S3 / EVIDENCE_DIR), else None (no records written)."""
    from tools import evlib

    if evidence:
        return evlib.S3Store(evidence) if is_s3(evidence) else evlib.LocalStore(evidence)
    try:
        return evlib.default_store()
    except RuntimeError:
        return None
