#!/usr/bin/env python3
"""check_unit.py <run_id>: does this unit have all four parts?

A unit counts only with: (1) a target declared before the run, (2) output from code in our repo,
(3) per-item records keyed to one run id whose times agree with what AWS recorded, (4) a pass/fail check.
This script verifies them as five checks:

  [a] declared before the run: declared/<run_id>.json exists and is valid, and its storage time (S3
      LastModified, or the git commit time of the file) is strictly before the run's FIRST AWS time.
  [b] code: the declared code_sha is a commit in this git repo.
  [c] records: records/<run_id>/ holds records (both layouts), every record carries this run_id, and every
      record's AWS time lies inside the run's AWS window. A record's aws_time, when present, is verified
      (and, on S3, must not be after its file's LastModified). When aws_time is null (a Lambda cannot know
      it at write time) it is derived from AWS: (1) the S3 LastModified of its per-record object
      records/<run_id>/<kind>/<key>.json, else (2) the timestamp of its log event (its log_pointer, else
      the change_applied line in /aws/lambda/team13-ran-tools naming its run, case and change id). The
      source used is reported per record; the record fails only when neither exists.
  [d] unchanged: each evidence file's sha256 matches the evidence ledger's entries (when a ledger is
      available); a records file may only have grown by whole lines (append), never been edited.
  [e] result: results/<run_id>.json exists and its value and pass/fail recompute from the item records.

The run's AWS window is read from AWS, never from our own records: SageMaker Describe*Job CreationTime
and end time for each job the run names, CloudWatch Logs Insights over the run's log groups (events that
mention the run id, both Regions), and S3 LastModified of the run's record files. A record's aws_time,
written by our code, is the thing being tested against that window.

Read-only: S3 ListObjectsV2/GetObject, Logs StartQuery/GetQueryResults, SageMaker Describe*.
Self-contained on purpose: it imports nothing from `spectrum` (the dashboard) or `evlib` (the writer),
so a bug shared with the view or the writer cannot make both agree.

Exit codes: 0 every part passed; 1 a part failed (printed); 2 nothing failed but a part could not be
checked (no AWS access, no AWS time for the run, an unknown metric, a ledger that has not caught up).

Usage:
  check_unit.py RUN_ID --s3 s3://bucket/evidence/      (or EVIDENCE_S3)    live, read-only AWS
  check_unit.py RUN_ID --dir /path/to/evidence         (or EVIDENCE_DIR)   local files, AWS for times
  check_unit.py RUN_ID --offline FIXTURE_DIR                               no AWS at all (fixtures)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

PASS, FAIL, CANNOT, SKIP = "PASS", "FAIL", "CANNOT CHECK", "SKIP"
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
SHA40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
OPS = (">=", "<=", "==", ">", "<")
REGIONS = ("us-east-2", "us-west-2")
LOG_GROUPS = ("/team13/checkpoint", "aws/spans", "/team13/bedrock-invocations", "/team13/runner",
              "/aws/lambda/team13-ran-tools", "/aws/sagemaker/TrainingJobs", "/aws/sagemaker/ProcessingJobs")
REPO = Path(__file__).resolve().parents[1]
GENESIS = "0" * 64
NOT_FOUND = ("NoSuchKey", "404", "NotFound")


# --------------------------------------------------------------------------- time

def parse_time(s) -> datetime:
    if isinstance(s, datetime):
        return s.astimezone(timezone.utc) if s.tzinfo else s.replace(tzinfo=timezone.utc)
    if not isinstance(s, str):
        raise ValueError(f"not a time: {s!r}")
    dt = datetime.fromisoformat(s.strip().replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError(f"time {s!r} has no UTC offset")
    return dt.astimezone(timezone.utc)


def insights_time(s: str) -> datetime:
    """Logs Insights prints @timestamp as 'YYYY-MM-DD HH:MM:SS.mmm' in UTC."""
    return datetime.fromisoformat(s.strip().replace(" ", "T")).replace(tzinfo=timezone.utc)


def iso(dt: datetime | None) -> str:
    if dt is None:
        return "-"
    return dt.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def err_code(e: Exception) -> str:
    r = getattr(e, "response", None) or {}
    return str((r.get("Error") or {}).get("Code", "")) or e.__class__.__name__


# --------------------------------------------------------------------------- evidence stores (read-only)

class S3Evidence:
    strong_times = True

    def __init__(self, client, uri: str):
        if not uri.startswith("s3://"):
            raise ValueError(f"expected s3://bucket/prefix/, got {uri!r}")
        self.bucket, _, prefix = uri[len("s3://"):].partition("/")
        self.prefix = prefix if (not prefix or prefix.endswith("/")) else prefix + "/"
        self.s3 = client
        self.uri = f"s3://{self.bucket}/{self.prefix}"

    def describe(self) -> str:
        return f"{self.uri} (times: S3 LastModified)"

    def list(self, prefix: str) -> dict[str, datetime]:
        out: dict[str, datetime] = {}
        for page in self.s3.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=self.prefix + prefix):
            for o in page.get("Contents", []):
                out[o["Key"][len(self.prefix):]] = parse_time(o["LastModified"])
        return out

    def get(self, rel: str) -> bytes | None:
        try:
            return self.s3.get_object(Bucket=self.bucket, Key=self.prefix + rel)["Body"].read()
        except Exception as e:  # noqa: BLE001
            if err_code(e) in NOT_FOUND:
                return None
            raise

    def time(self, rel: str) -> tuple[datetime | None, str]:
        lm = self.list(rel).get(rel)
        return lm, "s3:LastModified"


class LocalEvidence:
    """Local files. A file's time is its git commit time when it is committed and unchanged since;
    otherwise only its modification time, which anyone can set, so it does not count unless --allow-mtime."""

    def __init__(self, root: Path, allow_mtime: bool = False):
        self.root = Path(root)
        self.allow_mtime = allow_mtime
        self.uri = str(self.root)

    def describe(self) -> str:
        return f"{self.root} (times: git commit time, else file mtime)"

    def list(self, prefix: str) -> dict[str, datetime]:
        base = self.root / prefix
        if base.is_file():
            files = [base]
        elif base.is_dir():
            files = [p for p in base.rglob("*") if p.is_file()]
        else:
            files = []
        return {p.relative_to(self.root).as_posix(): datetime.fromtimestamp(p.stat().st_mtime, timezone.utc)
                for p in files}

    def get(self, rel: str) -> bytes | None:
        p = self.root / rel
        return p.read_bytes() if p.is_file() else None

    def time(self, rel: str) -> tuple[datetime | None, str]:
        p = (self.root / rel).resolve()
        if not p.is_file():
            return None, "missing"
        try:
            top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=p.parent, capture_output=True,
                                 text=True, check=True).stdout.strip()
            dirty = subprocess.run(["git", "status", "--porcelain", "--", str(p)], cwd=top, capture_output=True,
                                   text=True, check=True).stdout.strip()
            added = subprocess.run(["git", "log", "--diff-filter=A", "--format=%cI", "--", str(p)], cwd=top,
                                   capture_output=True, text=True, check=True).stdout.split()
            if added and not dirty:
                return parse_time(added[-1]), "git:commit-time"
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
            pass
        mt = datetime.fromtimestamp(p.stat().st_mtime, timezone.utc)
        return mt, "file:mtime (weak)"


# --------------------------------------------------------------------------- AWS reads

class RealAws:
    def __init__(self):
        import boto3  # noqa: F401 - fail early if missing

        self._clients: dict = {}

    def client(self, service: str, region: str | None = None):
        import boto3

        key = (service, region)
        if key not in self._clients:
            self._clients[key] = boto3.client(service, region_name=region or REGIONS[0])
        return self._clients[key]


def run_insights(aws, region: str, group: str, query: str, start: datetime, end: datetime,
                 poll_s: float, timeout_s: float = 120.0) -> list[dict] | None:
    """Rows as {field: value}, or None if the log group does not exist in that Region."""
    logs = aws.client("logs", region)
    try:
        qid = logs.start_query(logGroupNames=[group], startTime=int(start.timestamp()),
                               endTime=int(math.ceil(end.timestamp())), queryString=query, limit=10000)["queryId"]
    except Exception as e:  # noqa: BLE001
        if err_code(e) == "ResourceNotFoundException":
            return None
        raise
    t0 = time.monotonic()
    while True:
        r = logs.get_query_results(queryId=qid)
        st = r.get("status")
        if st == "Complete":
            return [{c["field"]: c["value"] for c in row} for row in r.get("results", [])]
        if st in ("Failed", "Cancelled", "Timeout", "Unknown"):
            raise RuntimeError(f"Logs Insights query in {group} ({region}) ended {st}")
        if time.monotonic() - t0 > timeout_s:
            raise RuntimeError(f"Logs Insights query in {group} ({region}) did not finish in {timeout_s:.0f}s")
        time.sleep(poll_s)


# --------------------------------------------------------------------------- ledger (independent re-implementation)

def _normalize(o):
    if isinstance(o, bool) or o is None or isinstance(o, (int, str)):
        return o
    if isinstance(o, float):
        return f"{o:.6g}"
    if isinstance(o, dict):
        return {str(k): _normalize(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_normalize(v) for v in o]
    raise TypeError(type(o).__name__)


def _canonical(o) -> str:
    return json.dumps(_normalize(o), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def verify_ledger(records: list[dict], pubkeys: dict | None) -> tuple[str | None, str]:
    """(problem or None, note on signatures). Walks hash = sha256(canonical(body) + prev) from genesis."""
    prev = GENESIS
    verify_sig = None
    note = "signatures not checked: no public_keys.json next to the ledger"
    if pubkeys is not None:
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

            keys = {k: v["public_key_hex"] for k, v in (pubkeys.get("keys") or {}).items()}

            def verify_sig(kid, h, sig):  # noqa: F811
                try:
                    Ed25519PublicKey.from_public_bytes(bytes.fromhex(keys[kid])).verify(bytes.fromhex(sig), h.encode())
                    return True
                except Exception:  # noqa: BLE001
                    return False

            note = "Ed25519 signatures verified with public_keys.json"
        except ImportError:
            note = "signatures not checked: the cryptography package is not installed (hash chain checked)"
    for i, r in enumerate(records, start=1):
        try:
            if r.get("seq") != i:
                return f"ledger record #{i}: found seq {r.get('seq')} (gap or reorder)", note
            body = {k: r[k] for k in ("seq", "ts", "type", "actor", "plane", "payload", "key_id")}
            h = hashlib.sha256((_canonical(body) + prev).encode("utf-8")).hexdigest()
            if r.get("prev_hash") != prev:
                return f"ledger record #{i}: link to the previous record is broken", note
            if h != r.get("hash"):
                return f"ledger record #{i}: contents changed after signing (hash mismatch)", note
            if verify_sig and not verify_sig(r.get("key_id"), r["hash"], r.get("sig", "")):
                return f"ledger record #{i}: signature does not verify", note
        except (KeyError, TypeError, ValueError) as e:
            return f"ledger record #{i}: malformed ({e.__class__.__name__})", note
        prev = r["hash"]
    return None, note


def read_jsonl(b: bytes) -> tuple[list[dict], str | None]:
    out = []
    for n, line in enumerate(b.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            o = json.loads(line)
        except ValueError:
            return out, f"line {n} is not valid JSON"
        if not isinstance(o, dict):
            return out, f"line {n} is not a JSON object"
        out.append(o)
    return out, None


def line_prefix_shas(b: bytes) -> dict[str, int]:
    """sha256 of every whole-line prefix of b (including empty and the full content) -> its length."""
    out = {hashlib.sha256(b"").hexdigest(): 0}
    h = hashlib.sha256()
    pos = 0
    for line in b.splitlines(keepends=True):
        h.update(line)
        pos += len(line)
        out[h.copy().hexdigest()] = pos
    out[hashlib.sha256(b).hexdigest()] = len(b)
    return out


# --------------------------------------------------------------------------- metrics (own implementation)

def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def recompute(metric: str, items: list[dict]) -> float:
    """Raises KeyError for an unknown metric, ValueError if the items cannot support it."""
    n = len(items)
    if n == 0:
        raise ValueError("there are no item records")
    if metric in ("pass_rate", "accuracy", "pass_count", "fail_count"):
        passed = [it.get("passed") for it in items]
        bad = [it.get("item_id") for it, p in zip(items, passed) if p is not True and p is not False]
        if bad:
            raise ValueError(f"{len(bad)} item(s) have no true/false passed, e.g. {bad[0]!r}")
        k = passed.count(True)
        if metric == "pass_count":
            return k
        if metric == "fail_count":
            return n - k
        return k / n
    if metric == "n_items":
        return n
    fields = {"mean_score": "score", "min_score": "score", "max_score": "score", "p50_latency_ms": "latency_ms",
              "p95_latency_ms": "latency_ms", "max_latency_ms": "latency_ms", "total_cost_usd": "cost_usd",
              "cost_per_1k_usd": "cost_usd"}
    if metric not in fields:
        raise KeyError(metric)
    f = fields[metric]
    xs = [it.get(f) for it in items]
    if not all(_num(x) for x in xs):
        raise ValueError(f"{metric} needs a number in {f} on every item")
    xs_sorted = sorted(xs)
    if metric.startswith("p50"):
        return xs_sorted[max(0, math.ceil(0.50 * n) - 1)]  # nearest rank
    if metric.startswith("p95"):
        return xs_sorted[max(0, math.ceil(0.95 * n) - 1)]
    if metric in ("max_latency_ms", "max_score"):
        return xs_sorted[-1]
    if metric == "min_score":
        return xs_sorted[0]
    total = math.fsum(xs)
    return {"mean_score": total / n, "total_cost_usd": total, "cost_per_1k_usd": total / n * 1000}[metric]


def holds(value: float, op: str, target: float) -> bool:
    return {">=": value >= target, "<=": value <= target, "==": value == target, ">": value > target,
            "<": value < target}[op]


def close(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))


# --------------------------------------------------------------------------- the check

class Part:
    def __init__(self, key: str, title: str):
        self.key, self.title = key, title
        self.state, self.lines = PASS, []

    def fail(self, msg: str) -> None:
        self.state = FAIL
        self.lines.append("FAIL: " + msg)

    def cannot(self, msg: str) -> None:
        if self.state != FAIL:
            self.state = CANNOT
        self.lines.append("cannot check: " + msg)

    def note(self, msg: str) -> None:
        self.lines.append(msg)


class Checker:
    def __init__(self, run_id: str, ev, aws, ledger_dir: Path | None, regions=REGIONS, log_groups=LOG_GROUPS,
                 slack_s: float = 5.0, lookback_h: float = 72.0, repo: Path = REPO, poll_s: float = 1.0,
                 now: datetime | None = None):
        self.run_id, self.ev, self.aws = run_id, ev, aws
        self.ledger_dir = ledger_dir
        self.regions, self.log_groups = tuple(regions), tuple(log_groups)
        self.slack = timedelta(seconds=slack_s)
        self.lookback = timedelta(hours=lookback_h)
        self.repo, self.poll_s = repo, poll_s
        self.now = now or datetime.now(timezone.utc)
        self.parts = {k: Part(k, t) for k, t in (("a", "declared before the run"), ("b", "code in our repo"),
                                                 ("c", "records keyed to the run, times agree with AWS"),
                                                 ("d", "unchanged since the ledger recorded it"),
                                                 ("e", "result recomputes from the items"))}
        self.window: tuple[datetime, datetime] | None = None
        self.window_sources: list[tuple[str, datetime, datetime]] = []
        self.start_sources: list[tuple[str, datetime, datetime]] = []
        self.derived: list[dict] = []  # records written with aws_time null, and the AWS source used instead
        self._events: list[dict] | None = None

    # ----------------------------------------------------------------- loading

    def load(self) -> None:
        r = self.run_id
        self.decl_rel, self.result_rel = f"declared/{r}.json", f"results/{r}.json"
        self.decl_bytes = self.ev.get(self.decl_rel)
        self.result_bytes = self.ev.get(self.result_rel)
        # Both record layouts (contract amendment A11), merged: records/<run>/<kind>.jsonl (lines in file
        # order) and records/<run>/<kind>/<key>.json (one record per object, sorted by created_at, key).
        self.record_files: dict[str, datetime] = {}
        for rel, t in self.ev.list(f"records/{r}/").items():
            parts = rel.split("/")
            if (len(parts) == 3 and parts[2].endswith(".jsonl")) or (len(parts) == 4 and parts[3].endswith(".json")):
                self.record_files[rel] = t
        self.records: dict[str, list[dict]] = {}
        self.record_src: dict[str, list[tuple[str, int]]] = {}  # parallel to records: (file, line)
        self.record_bytes: dict[str, bytes] = {}
        self.record_errors: list[str] = []
        objects: dict[str, list[tuple[str, str, dict]]] = {}
        for rel in sorted(self.record_files):
            b = self.ev.get(rel) or b""
            self.record_bytes[rel] = b
            recs, err = read_jsonl(b)
            if err:
                self.record_errors.append(f"{rel}: {err}")
            parts = rel.split("/")
            if len(parts) == 3:
                kind = parts[2][: -len(".jsonl")]
                self.records.setdefault(kind, []).extend(recs)
                self.record_src.setdefault(kind, []).extend((rel, i) for i in range(1, len(recs) + 1))
            else:
                if len(recs) != 1:
                    self.record_errors.append(f"{rel}: holds {len(recs)} records; a record object holds exactly one")
                for rec in recs[:1]:
                    objects.setdefault(parts[2], []).append((str(rec.get("created_at", "")), rel, rec))
        for kind, objs in objects.items():
            for _, rel, rec in sorted(objs, key=lambda x: (x[0], x[1])):
                self.records.setdefault(kind, []).append(rec)
                self.record_src.setdefault(kind, []).append((rel, 1))
        self.decl = None
        if self.decl_bytes is not None:
            try:
                self.decl = json.loads(self.decl_bytes.decode("utf-8"))
            except ValueError:
                self.decl = None

    # ----------------------------------------------------------------- AWS window

    def aws_window(self) -> None:
        c = self.parts["c"]
        for j in self.records.get("job", []):
            kind, name = j.get("job_kind"), j.get("job_name")
            region = j.get("region") or self.regions[0]
            if kind == "lambda":
                continue  # Lambda runs are timed by their CloudWatch events below
            op = {"training": ("describe_training_job", "TrainingJobName", "TrainingEndTime"),
                  "processing": ("describe_processing_job", "ProcessingJobName", "ProcessingEndTime"),
                  "endpoint": ("describe_endpoint", "EndpointName", None)}.get(kind)
            if op is None or not isinstance(name, str):
                c.fail(f"job record {name!r} has job_kind {kind!r}; expected training, processing, endpoint or lambda")
                continue
            try:
                d = getattr(self.aws.client("sagemaker", region), op[0])(**{op[1]: name})
            except Exception as e:  # noqa: BLE001
                code = err_code(e)
                if code in ("ValidationException", "ResourceNotFound"):
                    c.fail(f"job {name}: SageMaker in {region} has no such {kind} job")
                else:
                    c.cannot(f"job {name}: SageMaker {op[0]} in {region} failed ({code})")
                continue
            start = parse_time(d["CreationTime"])
            end_raw = d.get(op[2]) if op[2] else None
            end = parse_time(end_raw) if end_raw else (parse_time(d["LastModifiedTime"]) if kind != "endpoint"
                                                       and d.get("LastModifiedTime") else self.now)
            self.window_sources.append((f"sagemaker:CreationTime {kind} job {name} ({region})", start, end))
            if j.get("aws_start"):
                try:
                    if abs(parse_time(j["aws_start"]) - start) > self.slack:
                        c.fail(f"job {name}: record says aws_start {j['aws_start']}, SageMaker CreationTime is {iso(start)}")
                except ValueError:
                    c.fail(f"job {name}: aws_start {j['aws_start']!r} is not a time")
        decl_t = self.decl_time[0] if getattr(self, "decl_time", None) else None
        base = decl_t or self.now
        q_start, q_end = base - self.lookback, self.now + timedelta(minutes=1)
        query = (f'filter @message like "{self.run_id}" '
                 "| stats min(@timestamp) as first, max(@timestamp) as last, count(*) as n")
        for region in self.regions:
            for group in self.log_groups:
                try:
                    rows = run_insights(self.aws, region, group, query, q_start, q_end, self.poll_s)
                except Exception as e:  # noqa: BLE001
                    c.cannot(f"Logs Insights on {group} ({region}) failed ({err_code(e)})")
                    continue
                if rows and rows[0].get("first"):
                    first, last = insights_time(rows[0]["first"]), insights_time(rows[0]["last"])
                    self.window_sources.append(
                        (f"cloudwatch:{group} ({region}, {rows[0].get('n', '?')} events)", first, last))
        if self.window_sources:
            self.window = (min(s[1] for s in self.window_sources), max(s[2] for s in self.window_sources))
        # S3 LastModified of a record file proves the run had output by then, so it bounds the run's FIRST
        # AWS time for part (a); it never widens the job window used for record times in part (c).
        self.start_sources = list(self.window_sources)
        if isinstance(self.ev, S3Evidence):
            for rel, lm in sorted(self.record_files.items()):
                self.start_sources.append((f"s3:LastModified {rel}", lm, lm))

    # ----------------------------------------------------------------- parts

    def part_a(self) -> None:
        a = self.parts["a"]
        if self.decl_bytes is None:
            a.fail(f"no declaration at {self.decl_rel}")
            return
        d = self.decl
        if not isinstance(d, dict):
            a.fail(f"{self.decl_rel} is not a JSON object")
            return
        problems = []
        if d.get("run_id") != self.run_id:
            problems.append(f"run_id {d.get('run_id')!r} is not {self.run_id!r}")
        t = d.get("target")
        if not (isinstance(t, dict) and isinstance(t.get("metric"), str) and t.get("op") in OPS and _num(t.get("value"))):
            problems.append("target must be {metric, op, value, unit} with a numeric value")
        if not (isinstance(d.get("n_items"), int) and not isinstance(d.get("n_items"), bool)):
            problems.append("n_items must be an integer")
        if not (isinstance(d.get("test_set_sha256"), str) and HEX64.match(d["test_set_sha256"])):
            problems.append("test_set_sha256 must be 64 hex characters")
        for k in ("unit", "model_id", "grader"):
            if not (isinstance(d.get(k), str) and d[k].strip()):
                problems.append(f"{k} is missing")
        for p in problems:
            a.fail(f"declaration: {p}")
        dt, src = self.decl_time
        if dt is None:
            a.fail(f"no storage time for {self.decl_rel}")
            return
        weak = "weak" in src
        a.note(f"declared at {iso(dt)} ({src})")
        if d.get("declared_at"):
            try:
                if parse_time(d["declared_at"]) > dt + self.slack:
                    a.fail(f"declared_at {d['declared_at']} is after the file was stored ({iso(dt)})")
            except ValueError:
                a.fail(f"declared_at {d['declared_at']!r} is not a time")
        if not self.start_sources:
            a.cannot("no AWS time found for this run (no SageMaker job, no CloudWatch event mentioning the run id, "
                     "no S3 record object), so 'before the run' cannot be shown")
            return
        first = min(self.start_sources, key=lambda s: s[1])
        if weak and not getattr(self.ev, "allow_mtime", False):
            a.cannot(f"the declaration's only time is a local file mtime, which anyone can set; commit it to git "
                     f"or read it from S3 (or pass --allow-mtime). Run's first AWS time {iso(first[1])} ({first[0]})")
            return
        # S3 LastModified is floored to the second: the real time lies in [dt, dt + 1s), so only
        # dt + 1s <= first proves the declaration came first.
        before = (dt + timedelta(seconds=1) <= first[1]) if src == "s3:LastModified" else (dt < first[1])
        if before:
            a.note(f"before the run's first AWS time {iso(first[1])} ({first[0]}) by {(first[1] - dt).total_seconds():.0f}s")
        else:
            a.fail(f"declaration {iso(dt)} is NOT before the run's first AWS time {iso(first[1])} ({first[0]})")

    def part_b(self) -> None:
        b = self.parts["b"]
        sha = (self.decl or {}).get("code_sha") if isinstance(self.decl, dict) else None
        if not isinstance(sha, str) or not SHA40.match(sha):
            b.fail(f"declared code_sha {sha!r} is not a full 40-character commit sha")
            return
        try:
            r = subprocess.run(["git", "cat-file", "-t", sha], cwd=self.repo, capture_output=True, text=True)
        except FileNotFoundError:
            b.cannot("git is not installed")
            return
        if r.returncode == 0 and r.stdout.strip() == "commit":
            b.note(f"code_sha {sha} is a commit in {self.repo.name}")
        else:
            b.fail(f"code_sha {sha} is not a commit in {self.repo.name} (git cat-file -t: {r.stderr.strip() or r.stdout.strip()})")

    def part_c(self) -> None:
        c = self.parts["c"]
        if not self.record_files:
            c.fail(f"no records under records/{self.run_id}/")
            return
        for e in self.record_errors:
            c.fail(e)
        total = sum(len(v) for v in self.records.values())
        if total == 0:
            c.fail("the record files are empty")
            return
        c.note("records: " + ", ".join(f"{k} {len(v)}" for k, v in sorted(self.records.items())))
        if self.window is None:
            c.cannot("no AWS window for this run (no SageMaker job and no CloudWatch event mentions the run id), "
                     "so record times cannot be compared with AWS")
        else:
            lo, hi = self.window[0] - self.slack, self.window[1] + self.slack
            c.note(f"AWS window {iso(self.window[0])} .. {iso(self.window[1])} from: "
                   + "; ".join(s[0] for s in self.window_sources))
        offenders: list[str] = []
        on_s3 = isinstance(self.ev, S3Evidence)
        for kind, recs in sorted(self.records.items()):
            for r, (rel, line) in zip(recs, self.record_src[kind]):
                lm = self.record_files.get(rel) if on_s3 else None
                ident = r.get("item_id") or r.get("case_id") or r.get("job_name")
                where = (f"{rel.split('/', 2)[2]}" + (f" line {line}" if rel.endswith(".jsonl") else "")
                         + (f" ({ident})" if ident else ""))
                if r.get("run_id") != self.run_id:
                    offenders.append(f"{where}: run_id {r.get('run_id')!r}")
                    continue
                if r.get("kind") not in (None, kind):
                    offenders.append(f"{where}: kind {r.get('kind')!r} filed under {kind}")
                if r.get("aws_time") is None:
                    # A Lambda cannot know its own AWS time when it writes: derive it from what AWS recorded.
                    got = self.derive_time(r, rel, lm)
                    if got is None:
                        offenders.append(f"{where}: aws_time is null and no AWS source gives it (no per-record S3 "
                                         "object, no matching log event)")
                        continue
                    t, src = got
                    self.derived.append({"record": where, "aws_time": iso(t), "source": src})
                    if self.window is not None and not (lo <= t <= hi):
                        offenders.append(f"{where}: derived AWS time {iso(t)} ({src}) is outside the AWS window")
                    continue
                if not (isinstance(r.get("aws_time_source"), str) and r["aws_time_source"].strip()):
                    offenders.append(f"{where}: aws_time has no aws_time_source")
                try:
                    t = parse_time(r["aws_time"])
                except ValueError:
                    offenders.append(f"{where}: aws_time {r['aws_time']!r} is not a UTC time")
                    continue
                if self.window is not None and not (lo <= t <= hi):
                    offenders.append(f"{where}: aws_time {iso(t)} is outside the AWS window")
                if lm is not None and t > lm + self.slack:
                    offenders.append(f"{where}: aws_time {iso(t)} is after its file was written to S3 ({iso(lm)})")
        if self.derived:
            by_src: dict[str, int] = {}
            for x in self.derived:
                kind_of = x["source"].split(" ", 1)[0]
                by_src[kind_of] = by_src.get(kind_of, 0) + 1
            c.note(f"{len(self.derived)} record(s) had aws_time null; AWS time derived from: "
                   + ", ".join(f"{k} {n}" for k, n in sorted(by_src.items())))
            for x in self.derived[:20]:
                c.note(f"  {x['record']}: {x['aws_time']} from {x['source']}")
            if len(self.derived) > 20:
                c.note(f"  ... and {len(self.derived) - 20} more (all listed in --json)")
        for o in offenders[:12]:
            c.fail(o)
        if len(offenders) > 12:
            c.fail(f"... and {len(offenders) - 12} more")

    LAMBDA_GROUP = "/aws/lambda/team13-ran-tools"

    def derive_time(self, r: dict, rel: str, lm: datetime | None) -> tuple[datetime, str] | None:
        """AWS time for a record written without one, from AWS sources only, in order:
        (1) the S3 LastModified of its own per-record object records/<run>/<kind>/<key>.json;
        (2) the timestamp of its log event: the event its log_pointer names, else the change_applied line in
            /aws/lambda/team13-ran-tools that names its run, case (and change id, when it has one)."""
        if lm is not None and rel.count("/") == 3 and rel.endswith(".json"):
            return lm, f"s3:LastModified {rel}"
        events = self._run_events()
        ptr = r.get("log_pointer") if isinstance(r.get("log_pointer"), dict) else None
        case = r.get("case_id")
        change = r.get("change_id")
        cands = []
        if ptr and ptr.get("log_group"):
            cands = [e for e in events if e["group"] == ptr["log_group"]
                     and (not ptr.get("log_stream") or e["stream"] == ptr["log_stream"])
                     and (not case or case in e["message"])]
            exact = [e for e in cands if ptr.get("event_id") and e["ptr"] == ptr["event_id"]]
            cands = exact or cands
        if not cands and isinstance(case, str) and case:
            cands = [e for e in events if e["group"] == self.LAMBDA_GROUP and "change_applied" in e["message"]
                     and case in e["message"]]
            if change:
                cands = [e for e in cands if change in e["message"]]
        if not cands:
            return None
        e = min(cands, key=lambda x: x["time"])
        more = f", {len(cands)} matching events, earliest used" if len(cands) > 1 else ""
        return e["time"], f"cloudwatch:{e['group']} ({e['region']}, stream {e['stream']}{more})"

    def _run_events(self) -> list[dict]:
        """Every log event that mentions the run id, once per Region and log group (cached)."""
        if self._events is not None:
            return self._events
        self._events = []
        decl_t = self.decl_time[0] if getattr(self, "decl_time", None) else None
        q_start, q_end = (decl_t or self.now) - self.lookback, self.now + timedelta(minutes=1)
        query = (f'filter @message like "{self.run_id}" '
                 "| fields @timestamp, @log, @logStream, @message, @ptr | sort @timestamp asc | limit 10000")
        groups = list(self.log_groups) + ([self.LAMBDA_GROUP] if self.LAMBDA_GROUP not in self.log_groups else [])
        for region in self.regions:
            for group in groups:
                try:
                    rows = run_insights(self.aws, region, group, query, q_start, q_end, self.poll_s)
                except Exception as e:  # noqa: BLE001
                    self.parts["c"].cannot(f"Logs Insights on {group} ({region}) failed ({err_code(e)})")
                    continue
                for row in rows or []:
                    self._events.append({"region": region, "group": group, "stream": row.get("@logStream", ""),
                                         "time": insights_time(row["@timestamp"]), "message": row.get("@message", ""),
                                         "ptr": row.get("@ptr", "")})
        return self._events

    def part_d(self) -> None:
        d = self.parts["d"]
        ledger, pubkeys = self._load_ledger()
        if ledger is None:
            d.state = SKIP
            d.note("no evidence ledger available (pass --ledger-dir); file hashes not compared")
            return
        problem, sig_note = verify_ledger(ledger, pubkeys)
        d.note(f"ledger: {len(ledger)} records; {sig_note}")
        if problem:
            d.fail(f"the ledger itself does not verify: {problem}")
            return
        by_path: dict[str, list[dict]] = {}
        for r in ledger:
            p = (r.get("payload") or {}).get("path")
            if p:
                by_path.setdefault(p, []).append(r)
        files = {self.decl_rel: self.decl_bytes, self.result_rel: self.result_bytes, **self.record_bytes}
        result_seq = None
        if self.result_rel in by_path:
            result_seq = by_path[self.result_rel][0]["seq"]
        covered = 0
        for rel, content in sorted(files.items()):
            entries = [r for r in by_path.get(rel, []) if (r.get("payload") or {}).get("sha256")]
            missing_ev = [r for r in by_path.get(rel, []) if r.get("type") == "evidence.missing"]
            if not entries:
                if content is not None:
                    d.cannot(f"{rel}: not in the ledger yet (wait for the next ingest, then re-run)")
                continue
            if (result_seq is not None and rel.startswith("records/") and not rel.endswith(".jsonl")
                    and entries[0]["seq"] > result_seq):
                d.fail(f"{rel}: record object first appeared at ledger #{entries[0]['seq']}, after the result "
                       f"was recorded at #{result_seq}")
            covered += 1
            if content is None:
                d.fail(f"{rel}: recorded at ledger #{entries[0]['seq']} but the file is gone")
                continue
            shas = [r["payload"]["sha256"] for r in entries]
            cur = hashlib.sha256(content).hexdigest()
            if not rel.endswith(".jsonl"):
                bad = next((r for r in entries if r["payload"]["sha256"] != cur), None)
                if bad:
                    d.fail(f"{rel}: changed since ledger #{bad['seq']} (sha256 {bad['payload']['sha256'][:12]}..., "
                           f"now {cur[:12]}...)")
                continue
            prefixes = line_prefix_shas(content)
            last_len = -1
            ok = True
            for r, s in zip(entries, shas):
                if s not in prefixes or prefixes[s] < last_len:
                    d.fail(f"{rel}: edited after ledger #{r['seq']} (its recorded content is no longer the start "
                           f"of the file; an append-only file may only grow)")
                    ok = False
                    break
                last_len = prefixes[s]
                if result_seq is not None and r["seq"] > result_seq and r is not entries[0] and r["type"] != "evidence.reappeared":
                    d.fail(f"{rel}: grew at ledger #{r['seq']}, after the result was recorded at #{result_seq}")
                    ok = False
            if ok and shas[-1] != cur:
                extra = len(content) - last_len
                d.cannot(f"{rel}: {extra} byte(s) appended after the last ledger entry #{entries[-1]['seq']} are "
                         "not ledgered yet (wait for the next ingest, then re-run)")
            if missing_ev and ok:
                d.note(f"{rel}: the ledger once recorded it missing (#{missing_ev[0]['seq']}); content checks out")
        d.note(f"{covered} of {sum(1 for v in files.values() if v is not None)} evidence files are in the ledger")

    def _load_ledger(self) -> tuple[list[dict] | None, dict | None]:
        ld = self.ledger_dir
        if ld is None and isinstance(self.ev, LocalEvidence) and (self.ev.root / "ledger.jsonl").is_file():
            ld = self.ev.root
        if ld is None or not (Path(ld) / "ledger.jsonl").is_file():
            return None, None
        recs, err = read_jsonl((Path(ld) / "ledger.jsonl").read_bytes())
        if err:
            self.parts["d"].fail(f"ledger.jsonl {err}")
        pk = Path(ld) / "public_keys.json"
        try:
            pub = json.loads(pk.read_text()) if pk.is_file() else None
        except ValueError:
            pub = None
            self.parts["d"].fail("public_keys.json is not valid JSON")
        return recs, pub

    def part_e(self) -> None:
        e = self.parts["e"]
        if self.result_bytes is None:
            e.fail(f"no result at {self.result_rel}")
            return
        try:
            res = json.loads(self.result_bytes.decode("utf-8"))
            assert isinstance(res, dict)
        except (ValueError, AssertionError):
            e.fail(f"{self.result_rel} is not a JSON object")
            return
        d = self.decl if isinstance(self.decl, dict) else {}
        target = d.get("target") if isinstance(d.get("target"), dict) else None
        if res.get("run_id") != self.run_id:
            e.fail(f"result run_id {res.get('run_id')!r} is not {self.run_id!r}")
        if target is None:
            e.fail("no valid declared target to check the result against")
            return
        if res.get("target") != target:
            e.fail(f"result target {res.get('target')} differs from the declared target {target}")
        if res.get("unit") != d.get("unit"):
            e.fail(f"result unit {res.get('unit')!r} differs from the declared unit {d.get('unit')!r}")
        if res.get("metric") not in (None, target.get("metric")):
            e.fail(f"result metric {res.get('metric')!r} differs from the declared metric {target.get('metric')!r}")
        items = self.records.get("item", [])
        ids = [it.get("item_id") for it in items]
        if len(set(ids)) != len(ids):
            e.fail(f"item ids repeat ({len(ids) - len(set(ids))} duplicate(s))")
        if isinstance(d.get("n_items"), int) and d["n_items"] != len(items):
            e.fail(f"declared n_items {d['n_items']}, found {len(items)} item records")
        if res.get("n_items") != len(items):
            e.fail(f"result n_items {res.get('n_items')!r}, found {len(items)} item records")
        try:
            value = recompute(target["metric"], items)
        except KeyError:
            e.cannot(f"unknown metric {target['metric']!r}: this checker cannot recompute it")
            return
        except ValueError as ex:
            e.fail(f"cannot compute {target['metric']}: {ex}")
            return
        passed = holds(value, target["op"], target["value"])
        rv = res.get("value")
        if not _num(rv) or not close(value, rv):
            e.fail(f"result value {rv!r}, recomputed {value!r} from {len(items)} items")
        if res.get("passed") is not passed:
            e.fail(f"result passed={res.get('passed')!r}, recomputed passed={passed}")
        verdict = "PASSED" if passed else "MISSED"
        e.note(f"{target['metric']} = {value:.6g} (recomputed from {len(items)} items) {target['op']} "
               f"{target['value']} -> target {verdict}; results file agrees" if e.state == PASS else
               f"recomputed {target['metric']} = {value:.6g} {target['op']} {target['value']} -> {verdict}")

    # ----------------------------------------------------------------- run

    def run(self) -> int:
        self.load()
        self.decl_time = self.ev.time(self.decl_rel) if self.decl_bytes is not None else (None, "missing")
        try:
            self.aws_window()
        except ImportError:
            self.parts["c"].cannot("boto3 is not installed, so AWS times cannot be read")
        self.part_a()
        self.part_b()
        self.part_c()
        self.part_d()
        self.part_e()
        return self.exit_code()

    def exit_code(self) -> int:
        states = [p.state for p in self.parts.values()]
        return 1 if FAIL in states else 2 if CANNOT in states else 0

    def report(self, out=sys.stdout) -> None:
        say = lambda s="": print(s, file=out)  # noqa: E731
        say(f"check_unit {self.run_id}")
        say(f"  evidence: {self.ev.describe()}")
        for p in self.parts.values():
            say(f"[{p.key}] {p.state:<12} {p.title}")
            for line in p.lines:
                say(f"      {line}")
        code = self.exit_code()
        bad = [k for k, p in self.parts.items() if p.state == FAIL]
        unk = [k for k, p in self.parts.items() if p.state == CANNOT]
        skipped = [k for k, p in self.parts.items() if p.state == SKIP]
        tail = f" ({','.join(skipped)} skipped: no ledger; pass --ledger-dir <dashboard evidence dir>)" if skipped else ""
        say({0: f"RESULT: all four parts hold (exit 0){tail}",
             1: f"RESULT: FAILED part(s) {','.join(bad)} (exit 1)",
             2: f"RESULT: could not check part(s) {','.join(unk)} (exit 2)"}[code])

    def as_json(self) -> dict:
        return {"run_id": self.run_id, "exit": self.exit_code(), "evidence": self.ev.describe(),
                "window": [iso(self.window[0]), iso(self.window[1])] if self.window else None,
                "window_sources": [[s[0], iso(s[1]), iso(s[2])] for s in self.window_sources],
                "derived_aws_times": self.derived,
                "first_aws_time": iso(min(s[1] for s in self.start_sources)) if self.start_sources else None,
                "parts": {k: {"state": p.state, "title": p.title, "lines": p.lines} for k, p in self.parts.items()}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_id")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--s3", help="evidence prefix s3://bucket/evidence/ (default $EVIDENCE_S3)")
    src.add_argument("--dir", type=Path, help="local evidence directory (default $EVIDENCE_DIR)")
    src.add_argument("--offline", type=Path, metavar="FIXTURE", help="fixture directory: no AWS calls at all")
    ap.add_argument("--ledger-dir", type=Path, help="directory holding the evidence ledger.jsonl (+ public_keys.json)")
    ap.add_argument("--region", action="append", help=f"Region to search (repeatable; default {' '.join(REGIONS)})")
    ap.add_argument("--log-group", action="append",
                    help="log group to search for the run id (repeatable; default $EVIDENCE_LOG_GROUPS or "
                         + ", ".join(LOG_GROUPS) + ")")
    ap.add_argument("--slack-s", type=float, default=5.0, help="clock tolerance between AWS sources (default 5)")
    ap.add_argument("--lookback-h", type=float, default=72.0, help="search logs from this long before the declaration")
    ap.add_argument("--allow-mtime", action="store_true", help="accept a local file mtime as the declaration time (weak)")
    ap.add_argument("--json", action="store_true", help="print a JSON summary instead of text")
    a = ap.parse_args(argv)

    if not RUN_ID_RE.match(a.run_id):
        print(f"run_id {a.run_id!r} must match {RUN_ID_RE.pattern}", file=sys.stderr)
        return 2
    regions = a.region or list(REGIONS)
    env_groups = [g for g in os.environ.get("EVIDENCE_LOG_GROUPS", "").split(",") if g.strip()]
    groups = a.log_group or env_groups or list(LOG_GROUPS)
    ledger_dir = a.ledger_dir
    poll = 1.0
    now = None
    try:
        if a.offline:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from evfake import FakeAws

            aws, uri = FakeAws.from_fixture(a.offline)
            ev = S3Evidence(aws.client("s3"), uri)
            poll, now = 0.0, aws.now()
            if ledger_dir is None and (a.offline / "ledger" / "ledger.jsonl").is_file():
                ledger_dir = a.offline / "ledger"
        else:
            s3 = a.s3 or (None if a.dir else os.environ.get("EVIDENCE_S3", "").strip() or None)
            local = a.dir or (Path(os.environ["EVIDENCE_DIR"]) if not s3 and os.environ.get("EVIDENCE_DIR") else None)
            if not s3 and not local:
                print("no evidence location: pass --s3, --dir or --offline (or set EVIDENCE_S3 / EVIDENCE_DIR)",
                      file=sys.stderr)
                return 2
            try:
                aws = RealAws()
            except ImportError:
                print("boto3 is not installed: cannot read AWS times (use --offline for fixtures)", file=sys.stderr)
                return 2
            ev = S3Evidence(aws.client("s3", regions[0]), s3) if s3 else LocalEvidence(local, a.allow_mtime)
        if isinstance(ev, LocalEvidence):
            ev.allow_mtime = a.allow_mtime
        ck = Checker(a.run_id, ev, aws, ledger_dir, regions, groups, a.slack_s, a.lookback_h, poll_s=poll, now=now)
        ck.run()
    except Exception as e:  # noqa: BLE001 - anything unexpected is "cannot check", never a pass
        print(f"check_unit: cannot check {a.run_id}: {e.__class__.__name__}: {e}", file=sys.stderr)
        return 2
    if a.json:
        print(json.dumps(ck.as_json(), indent=2))
    else:
        ck.report()
    return ck.exit_code()


if __name__ == "__main__":
    sys.exit(main())
