#!/usr/bin/env python3
"""evidence_sync.py: mirror the S3 evidence prefix into a local directory, byte for byte. Read-only on AWS.

    python tools/evidence_sync.py [--s3 s3://bucket/evidence/] [--dest DIR] [--profile P] [--region R]
                                  [--jobs] [--attest] [--check] [--workers N]

What it does, in order:
  1. ListObjectsV2 over the prefix (logs/ is skipped). A key is fetched only when it is new or its ETag or
     size changed since the last sync (state in DEST/.sync/manifest.json), so a re-run is fast.
  2. Each object is downloaded to a temp file, checked, then renamed into place; its mtime is set to the S3
     LastModified, so the dashboard's "source time" is the AWS time, not the download time.
     Checks on every file: the byte count equals S3's size; a single-part ETag equals the body's MD5; and a
     per-record object records/<run>/<kind>/<key>.json has key == sha256(body)[:24] (the writer names each
     record object by the hash of its bytes, tools/evlib.record_key). A failed check is reported and the
     file is NOT written.
  3. --jobs: for each run's outputs/<run_id>/job-map.json, DescribeTrainingJob / DescribeProcessingJob for
     the jobs it names (read-only) -> DEST/.sync/jobs/<job>.json, so the dashboard can show instance, AWS
     times and billable seconds without calling AWS.
  4. --attest: for SageMaker runs, the job's own log (/aws/sagemaker/TrainingJobs, written by SageMaker)
     carries one `team13-evidence {"path", "sha256", ...}` line per records/result file. Each line's sha256
     is compared with the mirrored file -> DEST/.sync/attest/<run_id>.json (per file: match / MISMATCH /
     not attested).
  5. --check: runs tools/check_unit.py RUN_ID --s3 ... --json for each run that has a result
     -> DEST/.sync/checks/<run_id>.json (parts a-e, as check_unit prints them). Slow (Logs Insights).

Nothing is ever written to S3 and nothing local is deleted: a key that disappeared from S3 is listed as
"gone from S3" in the report and kept. Files the dashboard writes into DEST (ledger.jsonl, anchors.jsonl,
public_keys.json, README.md) are never touched. Hidden paths (.sync/) are ignored by the dashboard's reader.

Exit code: 0 ok; 1 a file failed a check (or an attested sha256 mismatched); 2 AWS could not be read.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _default_s3() -> str:
    """$EVIDENCE_S3, else $TEAM13_DATA_BUCKET's evidence/ prefix, else "" (resolved from STS at run time, the
    same rule as tools/sm/spec.py: the bucket name holds the account id, so it is never written into the repo)."""
    if os.environ.get("EVIDENCE_S3"):
        return os.environ["EVIDENCE_S3"]
    if os.environ.get("TEAM13_DATA_BUCKET"):
        return f"s3://{os.environ['TEAM13_DATA_BUCKET']}/evidence/"
    return ""


def _bucket_from_sts(profile, region: str) -> str:
    import boto3

    acct = boto3.Session(profile_name=profile or None, region_name=region).client("sts").get_caller_identity()["Account"]
    return f"s3://bb26-p-team13-data-{acct}-use2/evidence/"


DEFAULT_S3 = _default_s3()
DEFAULT_DEST = os.environ.get("SPECTRUM_EVIDENCE_DIR") or str(REPO / "dashboard" / ".evidence")
SKIP_TOP = {"logs"}
# Only the evidence layout is mirrored. Anything else under the prefix (e.g. endpoint data capture, whose
# SSE-KMS ETags are not MD5s and whose payloads are not evidence) is listed in the report and skipped.
LAYOUT_TOP = {"declared", "records", "results", "outputs", "models"}
RESERVED = {"ledger.jsonl", "anchors.jsonl", "public_keys.json", "README.md"}
RECORD_OBJ = re.compile(r"^records/[^/]+/[^/]+/([0-9a-f]{24})\.json$")
ATTEST_MARK = "team13-evidence"


def say(*a):
    print(*a, flush=True)


def safe_rel(rel: str) -> bool:
    parts = rel.split("/")
    return bool(rel) and not rel.startswith("/") and all(p not in ("", ".", "..") for p in parts)


def lm_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class Sync:
    def __init__(self, a):
        import boto3
        from botocore.config import Config

        self.a = a
        uri = a.s3
        if not uri.startswith("s3://"):
            raise SystemExit("--s3 must be s3://bucket/prefix/ (or set EVIDENCE_S3)")
        self.bucket, _, prefix = uri[5:].partition("/")
        self.prefix = prefix if (not prefix or prefix.endswith("/")) else prefix + "/"
        self.dest = Path(a.dest).resolve()
        self.state_dir = self.dest / ".sync"
        self.manifest_file = self.state_dir / "manifest.json"
        sess = boto3.Session(profile_name=a.profile or None, region_name=a.region)
        cfg = Config(max_pool_connections=max(10, a.workers), retries={"max_attempts": 8, "mode": "adaptive"})
        self.s3 = sess.client("s3", config=cfg)
        self.sess = sess
        self.problems: list[str] = []
        self.skipped_tops: set[str] = set()

    # ------------------------------------------------------------------ 1-2 mirror

    def load_manifest(self) -> dict:
        try:
            return json.loads(self.manifest_file.read_text())
        except (FileNotFoundError, ValueError):
            return {}

    def list_keys(self) -> list[dict]:
        out = []
        for page in self.s3.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=self.prefix):
            for o in page.get("Contents", []):
                rel = o["Key"][len(self.prefix):]
                top = rel.split("/")[0]
                if "/" in rel and top not in LAYOUT_TOP and top not in SKIP_TOP:
                    self.skipped_tops.add(top)
                if not safe_rel(rel) or top in SKIP_TOP or ("/" in rel and top not in LAYOUT_TOP) or rel in RESERVED or rel.endswith("/"):
                    continue
                out.append({"rel": rel, "etag": o["ETag"].strip('"'), "size": o["Size"],
                            "last_modified": lm_iso(o["LastModified"]), "ts": o["LastModified"].timestamp()})
        return out

    def fetch(self, o: dict) -> dict:
        body = self.s3.get_object(Bucket=self.bucket, Key=self.prefix + o["rel"])["Body"].read()
        errs = []
        if len(body) != o["size"]:
            errs.append(f"size {len(body)} != S3 size {o['size']}")
        if "-" not in o["etag"] and hashlib.md5(body).hexdigest() != o["etag"]:  # noqa: S324 - integrity, not security
            errs.append("MD5 does not match the S3 ETag")
        sha = hashlib.sha256(body).hexdigest()
        m = RECORD_OBJ.match(o["rel"])
        if m and sha[:24] != m.group(1):
            errs.append(f"record object key {m.group(1)} != sha256(body)[:24] {sha[:24]}")
        if errs:
            return {**o, "sha256": sha, "errors": errs}
        p = self.dest / o["rel"]
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=".sync-")
        with os.fdopen(fd, "wb") as f:
            f.write(body)
        os.utime(tmp, (o["ts"], o["ts"]))
        os.replace(tmp, p)
        return {**o, "sha256": sha, "errors": [], "key_attested": bool(m)}

    def mirror(self) -> dict:
        t0 = time.time()
        old = self.load_manifest()
        keys = self.list_keys()
        todo = []
        for o in keys:
            prev = old.get(o["rel"])
            p = self.dest / o["rel"]
            if prev and prev["etag"] == o["etag"] and prev["size"] == o["size"] and p.is_file() \
                    and p.stat().st_size == o["size"]:
                continue
            todo.append(o)
        say(f"S3 lists {len(keys)} evidence objects (logs/ skipped); {len(todo)} new or changed"
            + (f"; not evidence layout, skipped: {', '.join(sorted(self.skipped_tops))}/" if self.skipped_tops else ""))
        new = dict(old)
        failed = 0
        with ThreadPoolExecutor(self.a.workers) as ex:
            futs = [ex.submit(self.fetch, o) for o in todo]
            for i, f in enumerate(as_completed(futs), 1):
                r = f.result()
                if r["errors"]:
                    failed += 1
                    self.problems.append(f"{r['rel']}: " + "; ".join(r["errors"]))
                else:
                    new[r["rel"]] = {k: r[k] for k in ("etag", "size", "last_modified", "sha256")}
                if i % 5000 == 0:
                    say(f"  {i}/{len(todo)}")
        listed = {o["rel"] for o in keys}
        gone = sorted(k for k in new if k not in listed)
        for k in gone:
            new[k]["gone_from_s3"] = True
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_file.write_text(json.dumps(new, sort_keys=True))
        n_rec = sum(1 for k in new if RECORD_OBJ.match(k))
        say(f"mirrored to {self.dest} in {time.time() - t0:.1f}s: {len(todo) - failed} fetched, {failed} failed checks, "
            f"{len(gone)} gone from S3 (kept); {n_rec} record objects carry a sha256-derived key (all checked)")
        return new

    # ------------------------------------------------------------------ 3 jobs

    def job_names(self) -> dict[str, list[tuple[str, str]]]:
        """run_id -> [(job kind, job name)] from outputs/<run>/job-map.json and job records."""
        out: dict[str, list[tuple[str, str]]] = {}
        for jm in sorted((self.dest / "outputs").glob("*/job-map.json")):
            run = jm.parent.name
            try:
                d = json.loads(jm.read_text())
            except ValueError:
                continue
            for name, kind in _find_jobs(d):
                out.setdefault(run, [])
                if (kind, name) not in out[run]:
                    out[run].append((kind, name))
        for rec in sorted((self.dest / "records").glob("*/job/*.json")):  # runs with a job record but no job-map
            run = rec.parent.parent.name
            try:
                d = json.loads(rec.read_text())
            except ValueError:
                continue
            name = d.get("job_name") or d.get("TrainingJobName")
            kind = "processing" if d.get("job_kind") == "processing" else "training"
            if isinstance(name, str) and re.match(r"^[A-Za-z0-9][A-Za-z0-9-]{0,62}$", name):
                out.setdefault(run, [])
                if (kind, name) not in out[run]:
                    out[run].append((kind, name))
        return out

    def describe_jobs(self) -> None:
        sm = self.sess.client("sagemaker")
        jobs_dir = self.state_dir / "jobs"
        jobs_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for run, jobs in self.job_names().items():
            for kind, name in jobs:
                f = jobs_dir / f"{name}.json"
                if f.exists():
                    d = json.loads(f.read_text())
                    if d.get("status") in ("Completed", "Failed", "Stopped"):
                        continue
                try:
                    if kind == "processing":
                        r = sm.describe_processing_job(ProcessingJobName=name)
                        d = {"kind": "processing", "name": name, "status": r.get("ProcessingJobStatus"),
                             "instance_type": r["ProcessingResources"]["ClusterConfig"]["InstanceType"],
                             "instance_count": r["ProcessingResources"]["ClusterConfig"]["InstanceCount"],
                             "creation_time": _iso(r.get("CreationTime")), "start_time": _iso(r.get("ProcessingStartTime")),
                             "end_time": _iso(r.get("ProcessingEndTime")), "failure_reason": r.get("FailureReason"),
                             "billable_s": None}
                    else:
                        r = sm.describe_training_job(TrainingJobName=name)
                        d = {"kind": "training", "name": name, "status": r.get("TrainingJobStatus"),
                             "secondary_status": r.get("SecondaryStatus"),
                             "instance_type": r["ResourceConfig"].get("InstanceType"),
                             "instance_count": r["ResourceConfig"].get("InstanceCount"),
                             "creation_time": _iso(r.get("CreationTime")), "start_time": _iso(r.get("TrainingStartTime")),
                             "end_time": _iso(r.get("TrainingEndTime")), "failure_reason": r.get("FailureReason"),
                             "billable_s": r.get("BillableTimeInSeconds"), "training_s": r.get("TrainingTimeInSeconds"),
                             "max_run_s": (r.get("StoppingCondition") or {}).get("MaxRuntimeInSeconds")}
                    d["run_id"] = run
                    d["described_at"] = lm_iso(datetime.now(timezone.utc))
                    f.write_text(json.dumps(d, indent=1, sort_keys=True))
                    n += 1
                except Exception as e:  # noqa: BLE001
                    say(f"  describe {kind} job {name}: {type(e).__name__}: {str(e)[:120]}")
        say(f"described {n} SageMaker jobs (read-only) -> {jobs_dir}")

    # ------------------------------------------------------------------ 4 attest

    def attest(self, manifest: dict) -> int:
        logs = self.sess.client("logs")
        out_dir = self.state_dir / "attest"
        out_dir.mkdir(parents=True, exist_ok=True)
        mismatches = 0
        for run, jobs in self.job_names().items():
            mine = sorted((rel, m["sha256"]) for rel, m in manifest.items()
                          if rel.startswith(f"records/{run}/") or rel == f"results/{run}.json")
            fp = hashlib.sha256(json.dumps([mine, jobs]).encode()).hexdigest()
            prev_file = out_dir / f"{run}.json"
            try:
                if json.loads(prev_file.read_text()).get("fingerprint") == fp:
                    continue  # same files as last time: the stored comparison still holds
            except (FileNotFoundError, ValueError):
                pass
            seen: dict[str, dict] = {}
            for kind, name in jobs:
                group = "/aws/sagemaker/ProcessingJobs" if kind == "processing" else "/aws/sagemaker/TrainingJobs"
                try:
                    for page in logs.get_paginator("filter_log_events").paginate(
                            logGroupName=group, logStreamNamePrefix=f"{name}/", filterPattern=f'"{ATTEST_MARK}"'):
                        for ev in page.get("events", []):
                            msg = ev.get("message", "")
                            i = msg.find(ATTEST_MARK + " {")
                            if i < 0:
                                continue
                            try:
                                body = json.loads(msg[i + len(ATTEST_MARK) + 1:])
                            except ValueError:
                                continue
                            if body.get("run_id") == run and isinstance(body.get("path"), str):
                                seen[body["path"]] = {"sha256": body.get("sha256"), "bytes": body.get("bytes"),
                                                      "log_time": _iso_ms(ev.get("timestamp")), "job": name}
                except Exception as e:  # noqa: BLE001
                    say(f"  attest {run}/{name}: {type(e).__name__}: {str(e)[:120]}")
            if not seen:
                continue
            files = {}
            for rel, m in manifest.items():
                if not (rel.startswith(f"records/{run}/") or rel == f"results/{run}.json"):
                    continue
                a = seen.get(rel)
                if a is None:
                    files[rel] = {"state": "not attested", "sha256": m["sha256"]}
                elif a["sha256"] == m["sha256"]:
                    files[rel] = {"state": "match", "sha256": m["sha256"], "log_time": a["log_time"], "job": a["job"]}
                else:
                    mismatches += 1
                    files[rel] = {"state": "MISMATCH", "sha256": m["sha256"], "attested": a["sha256"], "job": a["job"]}
                    self.problems.append(f"{rel}: sha256 {m['sha256'][:12]} != attested {a['sha256'][:12]} in {a['job']} log")
            counts: dict[str, int] = {}
            for v in files.values():
                counts[v["state"]] = counts.get(v["state"], 0) + 1
            (out_dir / f"{run}.json").write_text(json.dumps(
                {"run_id": run, "source": "SageMaker job log lines 'team13-evidence' (sha256 of each stored file)",
                 "checked_at": lm_iso(datetime.now(timezone.utc)), "counts": counts, "files": files,
                 "fingerprint": fp}, indent=1, sort_keys=True))
            say(f"  attest {run}: {counts}")
        return mismatches

    # ------------------------------------------------------------------ 5 check_unit

    def check(self, manifest: dict) -> None:
        out_dir = self.state_dir / "checks"
        out_dir.mkdir(parents=True, exist_ok=True)
        runs = sorted({k[len("results/"):-len(".json")] for k in manifest if k.startswith("results/") and k.endswith(".json")})
        if self.a.run:
            runs = [r for r in runs if r in self.a.run]
        env = dict(os.environ)
        if self.a.profile:
            env["AWS_PROFILE"] = self.a.profile

        def one(run):
            t0 = time.time()
            p = subprocess.run([sys.executable, str(REPO / "tools" / "check_unit.py"), run, "--s3", self.a.s3, "--json"],
                               capture_output=True, text=True, env=env, cwd=REPO, timeout=900)
            try:
                summary = json.loads(p.stdout)
            except ValueError:
                summary = None
            doc = {"run_id": run, "exit_code": p.returncode, "checked_at": lm_iso(datetime.now(timezone.utc)),
                   "elapsed_s": round(time.time() - t0, 1), "summary": summary,
                   "stdout": None if summary is not None else p.stdout[-20000:], "stderr": p.stderr[-4000:]}
            (out_dir / f"{run}.json").write_text(json.dumps(doc, indent=1, sort_keys=True))
            return run, p.returncode, doc["elapsed_s"]

        with ThreadPoolExecutor(4) as ex:
            for f in as_completed([ex.submit(one, r) for r in runs]):
                try:
                    run, rc, el = f.result()
                    say(f"  check_unit {run}: exit {rc} ({el}s)")
                except Exception as e:  # noqa: BLE001
                    say(f"  check_unit failed: {type(e).__name__}: {e}")


def copy_handin(ref: str, dest: Path) -> None:
    """Copy the hand-in files the dashboard reads (judging.json, evidence/superseded.json) from a git ref into
    DEST, read with `git show` (the working tree is not touched)."""
    for src, name in (("judging.json", "judging.json"), ("evidence/superseded.json", "superseded.json")):
        p = subprocess.run(["git", "show", f"{ref}:{src}"], cwd=REPO, capture_output=True)
        if p.returncode != 0:
            say(f"  hand-in {ref}:{src}: not found")
            continue
        try:
            json.loads(p.stdout)
        except ValueError:
            say(f"  hand-in {ref}:{src}: not valid JSON, skipped")
            continue
        (dest / name).write_bytes(p.stdout)
        say(f"  hand-in {ref}:{src} -> {dest / name}")


def _find_jobs(d) -> list[tuple[str, str]]:
    """Job names in a job-map: any *job_name / TrainingJobName / ProcessingJobName string, or a 'jobs' list."""
    out = []
    stack = [d]
    while stack:
        x = stack.pop()
        if isinstance(x, dict):
            for k, v in x.items():
                if isinstance(v, str) and k in ("job_name", "training_job_name", "TrainingJobName", "job"):
                    out.append((v, "training"))
                elif isinstance(v, str) and k in ("processing_job_name", "ProcessingJobName"):
                    out.append((v, "processing"))
                else:
                    stack.append(v)
        elif isinstance(x, list):
            stack.extend(x)
    return [(n, k) for n, k in out if re.match(r"^[A-Za-z0-9][A-Za-z0-9-]{0,62}$", n)]


def _iso(dt) -> str | None:
    return lm_iso(dt) if dt else None


def _iso_ms(ms) -> str | None:
    return lm_iso(datetime.fromtimestamp(ms / 1000, timezone.utc)) if ms else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--s3", default=DEFAULT_S3, help="evidence prefix (default $EVIDENCE_S3)")
    ap.add_argument("--dest", default=DEFAULT_DEST, help="local evidence root (default $SPECTRUM_EVIDENCE_DIR or dashboard/.evidence)")
    ap.add_argument("--profile", default=os.environ.get("AWS_PROFILE"))
    ap.add_argument("--region", default=os.environ.get("AWS_REGION") or "us-east-2")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--jobs", action="store_true", help="cache DescribeTrainingJob/DescribeProcessingJob per run")
    ap.add_argument("--attest", action="store_true", help="compare sha256 with the job log's team13-evidence lines")
    ap.add_argument("--check", action="store_true", help="cache check_unit --json per run (slow)")
    ap.add_argument("--run", action="append", help="limit --check to these run ids")
    ap.add_argument("--handin", default=os.environ.get("HANDIN"),
                    help="git ref of the hand-in branch: copy its judging.json and evidence/superseded.json into DEST")
    a = ap.parse_args(argv)
    if not a.s3:
        try:
            a.s3 = _bucket_from_sts(a.profile, a.region)
        except Exception as e:  # noqa: BLE001
            say(f"AWS could not be read (STS, to name the team bucket): {type(e).__name__}: {e}")
            return 2
    try:
        s = Sync(a)
        manifest = s.mirror()
        if a.handin:
            copy_handin(a.handin, s.dest)
        if a.jobs or a.attest:
            s.describe_jobs()
        mism = s.attest(manifest) if a.attest else 0
        if a.check:
            s.check(manifest)
    except Exception as e:  # noqa: BLE001
        if type(e).__name__ in ("ClientError", "NoCredentialsError", "ProfileNotFound", "EndpointConnectionError",
                                "SSOTokenLoadError", "UnauthorizedSSOTokenError", "TokenRetrievalError"):
            say(f"AWS could not be read: {type(e).__name__}: {e}")
            return 2
        raise
    for p in s.problems[:50]:
        say(f"PROBLEM {p}")
    return 1 if s.problems or mism else 0


if __name__ == "__main__":
    sys.exit(main())
