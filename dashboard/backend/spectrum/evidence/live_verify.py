"""Verify one run live against AWS from the console: tools/check_unit.py as a read-only subprocess.

POST /api/evidence/verify/{run_id} runs
    check_unit.py <run_id> --s3 <team evidence prefix> --ledger-dir <evidence dir> --json
from the repo root with the server's own AWS environment (S3 GetObject/List, CloudWatch Logs reads,
SageMaker Describe*: check_unit never writes to AWS). One check at a time, 180 s timeout, only for a run id
the evidence mirror knows. The output is written to <evidence dir>/.sync/checks/<run_id>.json in exactly the
format `make evidence-sync CHECK=1` writes, so every screen picks it up; the response carries parts a-e with
their lines, the derived AWS times with their sources, the window, the exit code and checked_at.
GET returns the last result for the run and whether a check is running.

Guards (as kitapprovals): loopback client and a localhost Host only (403 from the LAN even with the access
token), Origin equal to this console's own origin and Sec-Fetch-Site same-origin/none for POST, and the
X-Spectrum-Client: verify-live header. No AWS credentials -> a plain 503 that says so.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from ..access import is_loopback

REPO = Path(__file__).resolve().parents[4]
CHECK_UNIT = REPO / "tools" / "check_unit.py"
CLIENT_HEADER = "verify-live"
TIMEOUT_S = float(os.environ.get("SPECTRUM_VERIFY_TIMEOUT_S") or 180)
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
NO_STORE = {"Cache-Control": "no-store"}
VERDICT = {0: "all parts passed", 1: "a part failed", 2: "nothing failed, a part could not be checked"}

_lock = asyncio.Lock()
_running: dict[str, Any] = {}


def _err(http: int, error: str, message: str, **extra) -> JSONResponse:
    return JSONResponse({"error": error, "message": message, **extra}, status_code=http, headers=NO_STORE)


def _host_ok(request: Request) -> bool:
    host = (request.headers.get("host") or "").lower()
    name = host.rsplit(":", 1)[0] if not host.startswith("[") else host.split("]")[0] + "]"
    return name in ("localhost", "127.0.0.1", "[::1]")


def guard(request: Request, write: bool) -> JSONResponse | None:
    if not is_loopback(tuple(request.client) if request.client else None) or not _host_ok(request):
        return _err(403, "laptop_only", "Live verification runs only on the presenter laptop (http://localhost:8787).")
    h = request.headers
    if h.get("x-spectrum-client") != CLIENT_HEADER:
        return _err(403, "bad_client", "Missing the X-Spectrum-Client header.")
    if write:
        if h.get("origin") != f"http://{h.get('host')}":
            return _err(403, "bad_origin", "The request must come from this console's own page.")
        if h.get("sec-fetch-site") not in (None, "same-origin", "none"):
            return _err(403, "bad_origin", "Cross-site requests are refused.")
    return None


def evidence_root() -> Path | None:
    raw = os.environ.get("SPECTRUM_EVIDENCE_DIR") or os.environ.get("SPECTRUM_RECORDS_ROOT")
    if not raw or raw.startswith("s3://"):
        return None
    return Path(raw[len("file://"):] if raw.startswith("file://") else raw)


def known_runs(root: Path) -> set[str]:
    ids: set[str] = set()
    for d in ("declared", "results"):
        ids.update(p.name[:-5] for p in (root / d).glob("*.json"))
    rec = root / "records"
    if rec.is_dir():
        ids.update(p.name for p in rec.iterdir() if p.is_dir() and not p.name.startswith("."))
    return ids


EXPIRED = ("UnauthorizedSSOTokenError", "TokenRetrievalError", "SSOTokenLoadError", "ExpiredToken", "ExpiredTokenException",
           "InvalidClientTokenId", "RequestExpired")


def _signin_hint() -> str:
    prof = os.environ.get("AWS_PROFILE") or "fyuz"
    return f"AWS sign-in expired: run `aws sso login --profile {prof}`, then press again."


def credentials() -> tuple[str | None, str | None]:
    """(account, None) or (None, plain reason naming the fix). A cheap sts:GetCallerIdentity before every check:
    each check is a fresh process reading the SSO cache, so a re-login takes effect without a server restart."""
    try:
        import boto3

        return boto3.client("sts", region_name=os.environ.get("AWS_REGION") or "us-east-2").get_caller_identity()["Account"], None
    except Exception as e:  # noqa: BLE001
        code = str(((getattr(e, "response", None) or {}).get("Error") or {}).get("Code") or "")
        name = type(e).__name__
        if name in EXPIRED or code in EXPIRED or "expired" in str(e).lower() or "sso" in str(e).lower():
            return None, _signin_hint()
        if name in ("NoCredentialsError", "ProfileNotFound", "NoRegionError"):
            return None, (f"No AWS credentials are available to this server ({name}). Start it with AWS_PROFILE set "
                          "(the team SSO profile, logged in) to verify live.")
        return None, f"AWS could not be reached to verify ({name}: {str(e)[:160]})."


def s3_prefix(account: str | None = None) -> str:
    """EVIDENCE_S3, else TEAM13_DATA_BUCKET, else the team bucket named from the account (never stored in the repo)."""
    if os.environ.get("EVIDENCE_S3"):
        return os.environ["EVIDENCE_S3"]
    if os.environ.get("TEAM13_DATA_BUCKET"):
        return f"s3://{os.environ['TEAM13_DATA_BUCKET']}/evidence/"
    return f"s3://bb26-p-team13-data-{account}-use2/evidence/"


def _clean(s: str) -> str:
    return re.sub(r"(is (?:not )?a commit) in \S+", r"\1 in the team repo", s)


def parse(doc: dict) -> dict:
    """The response view of one cached check (live or from the sync)."""
    s = doc.get("summary") if isinstance(doc.get("summary"), dict) else {}
    parts = [{"id": k, "state": p.get("state"), "title": p.get("title"), "lines": [_clean(str(x)) for x in (p.get("lines") or [])[:20]]}
             for k, p in (s.get("parts") or {}).items() if isinstance(p, dict)]
    times = [{"record": t.get("record"), "aws_time": t.get("aws_time"), "source": t.get("source")}
             for t in (s.get("derived_aws_times") or []) if isinstance(t, dict)]
    return {"run_id": doc.get("run_id"), "exit_code": doc.get("exit_code"), "verdict": VERDICT.get(doc.get("exit_code"), "unknown"),
            "checked_at": doc.get("checked_at"), "elapsed_s": doc.get("elapsed_s"), "live": bool(doc.get("live")),
            "parts": parts, "derived_aws_times": times[:200], "n_derived_aws_times": len(times),
            "window": s.get("window"), "window_sources": s.get("window_sources"), "first_aws_time": s.get("first_aws_time"),
            "error": None if s else ((doc.get("stderr") or doc.get("stdout") or "")[-1500:] or "check_unit printed no JSON summary")}


def cache_file(root: Path, run_id: str) -> Path:
    return root / ".sync" / "checks" / f"{run_id}.json"


def status(root: Path | None, run_id: str) -> dict:
    last = None
    if root is not None:
        try:
            last = parse(json.loads(cache_file(root, run_id).read_text(encoding="utf-8")))
        except (OSError, ValueError):
            last = None
    return {"run_id": run_id, "running": bool(_running), "running_run_id": _running.get("run_id"),
            "started_at": _running.get("started_at"), "last": last}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _write_atomic(f: Path, doc: dict) -> None:
    """A temp file in the same directory, then os.replace: a reader never sees a half-written cache."""
    f.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=f.parent, prefix=f".{f.name}.", suffix=".tmp")
    with os.fdopen(fd, "w") as h:
        h.write(json.dumps(doc, indent=1, sort_keys=True))
    os.replace(tmp, f)


def _cached(root: Path, run_id: str) -> dict | None:
    try:
        d = json.loads(cache_file(root, run_id).read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else None
    except (OSError, ValueError):
        return None


def _expired_in(stderr: str) -> bool:
    return any(k in stderr for k in EXPIRED) or ("expired" in stderr.lower() and "token" in stderr.lower())


async def run_check(run_id: str, root: Path, prefix: str, timeout_s: float | None = None, python: str = sys.executable) -> dict:
    """Run check_unit for one run; returns the sync-format doc (not written). Raises asyncio.TimeoutError after
    killing the whole process group (check_unit may start helpers)."""
    t0 = time.time()
    proc = await asyncio.create_subprocess_exec(
        python, str(CHECK_UNIT), run_id, "--s3", prefix, "--ledger-dir", str(root), "--json",
        cwd=str(REPO), env=dict(os.environ), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        start_new_session=True)
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout_s or TIMEOUT_S)
    except asyncio.TimeoutError:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        await proc.wait()
        raise
    stdout, stderr = out.decode("utf-8", "replace"), err.decode("utf-8", "replace")
    try:
        summary = json.loads(stdout)
    except ValueError:
        summary = None
    return {"run_id": run_id, "exit_code": proc.returncode, "checked_at": _now(), "elapsed_s": round(time.time() - t0, 1),
            "summary": summary, "stdout": None if summary is not None else stdout[-20000:], "stderr": stderr[-4000:], "live": True}


def refresh_attest(run_id: str, root: Path, prefix: str) -> str:
    """Refresh .sync/attest/<run>.json with evidence_sync's own single-run attest code (same evidence, same
    format; it reads S3 and CloudWatch only and never writes the signed ledger). Written to a temp state dir,
    then moved into place only if it produced a comparison. Returns what happened."""
    import argparse

    sys.path.insert(0, str(REPO / "tools"))
    try:
        import evidence_sync as es  # noqa: PLC0415
    finally:
        sys.path.pop(0)
    a = argparse.Namespace(s3=prefix, dest=str(root), profile=os.environ.get("AWS_PROFILE"),
                           region=os.environ.get("AWS_REGION") or "us-east-2", workers=8, run=None)
    sync = es.Sync(a)
    jobs = sync.job_names().get(run_id)
    if not jobs:
        return "no SageMaker job for this run: nothing to attest"
    sync.job_names = lambda: {run_id: jobs}  # this run only
    try:
        manifest = json.loads((root / ".sync" / "manifest.json").read_text())
    except (OSError, ValueError):
        return "no sync manifest: run make evidence-sync first"
    with tempfile.TemporaryDirectory(dir=root / ".sync") as tmpd:
        sync.state_dir = Path(tmpd)
        sync.attest(manifest)
        out = Path(tmpd) / "attest" / f"{run_id}.json"
        if not out.is_file():
            return "the job log carries no team13-evidence lines: attest cache kept"
        dest = root / ".sync" / "attest" / f"{run_id}.json"
        merged, n = merge_attest(_load_json(dest), json.loads(out.read_text()))
        _write_atomic(dest, merged)
    return f"refreshed ({n} entries re-checked; other entries kept)"


DEFINITIVE_ATTEST = ("match", "MISMATCH")


def _load_json(p: Path) -> dict | None:
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else None
    except (OSError, ValueError):
        return None


def merge_attest(old: dict | None, new: dict) -> tuple[dict, int]:
    """Entry by entry: an entry re-checked with a definitive state (match / MISMATCH) replaces the old one; a
    non-definitive one is only added where there was none; every entry not re-checked is kept (other writers,
    e.g. CloudWatch Lambda-log entries, share this file). Counts are recomputed; other top-level keys are kept."""
    files = dict((old or {}).get("files") or {})
    n = 0
    for path, e in (new.get("files") or {}).items():
        if not isinstance(e, dict):
            continue
        if e.get("state") in DEFINITIVE_ATTEST or path not in files:
            files[path] = e
            n += 1
    counts: dict[str, int] = {}
    for e in files.values():
        st = e.get("state") if isinstance(e, dict) else None
        counts[str(st)] = counts.get(str(st), 0) + 1
    out = {**(old or {}), **{k: v for k, v in new.items() if k not in ("files", "counts")}, "files": files, "counts": counts}
    return out, n


async def handle_post(request: Request, run_id: str) -> JSONResponse | dict:
    refused = guard(request, write=True)
    if refused:
        return refused
    root = evidence_root()
    if root is None:
        return _err(503, "no_mirror", "Live verification needs the local evidence mirror (SPECTRUM_EVIDENCE_DIR).")
    if not RUN_ID_RE.match(run_id) or ".." in run_id or run_id not in await asyncio.to_thread(known_runs, root):
        return _err(404, "unknown_run", f"No run {run_id!r} in the evidence mirror.")
    if _lock.locked():
        return _err(409, "busy", "Another live check is running; one at a time.", running_run_id=_running.get("run_id"),
                    started_at=_running.get("started_at"))
    cached = _cached(root, run_id)
    kept = {"cache": "retained", "cached": parse(cached)} if cached else {"cache": "none"}
    async with _lock:  # released on every path, including a timeout: the button works again afterwards
        _running.update(run_id=run_id, started_at=_now())
        try:
            account, why = await asyncio.to_thread(credentials)
            if account is None:
                return _err(503, "no_aws_credentials", why or "No AWS credentials.", **kept)
            prefix = s3_prefix(account)
            try:
                doc = await run_check(run_id, root, prefix)
            except asyncio.TimeoutError:
                return _err(504, "timeout", f"check_unit did not finish within {int(TIMEOUT_S)} s.", **kept)
            definitive = doc["exit_code"] in (0, 1)
            if not definitive and _expired_in(doc.get("stderr") or ""):
                return _err(503, "no_aws_credentials", _signin_hint(), **kept)
            # never downgrade: replace the cache only with a definitive result, or over a cache that was itself
            # not definitive (or absent)
            if definitive or cached is None or cached.get("exit_code") not in (0, 1):
                await asyncio.to_thread(_write_atomic, cache_file(root, run_id), doc)
                out = {**parse(doc), "cache": "replaced"}
                if definitive:
                    try:
                        out["attest"] = await asyncio.to_thread(refresh_attest, run_id, root, prefix)
                    except Exception as e:  # noqa: BLE001 - the check itself is done and cached
                        out["attest"] = f"attest not refreshed ({type(e).__name__}); the earlier attest cache is kept"
            else:
                out = {**parse(doc), **kept}
        finally:
            _running.clear()
    return JSONResponse(out, headers=NO_STORE)


async def handle_get(request: Request, run_id: str) -> JSONResponse | dict:
    refused = guard(request, write=False)
    if refused:
        return refused
    if not RUN_ID_RE.match(run_id) or ".." in run_id:
        return _err(400, "bad_run_id", "Invalid run id.")
    return JSONResponse(status(evidence_root(), run_id), headers=NO_STORE)
