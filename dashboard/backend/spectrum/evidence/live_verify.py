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
import sys
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


def s3_prefix() -> tuple[str | None, str | None]:
    """(evidence prefix, None) or (None, plain reason). EVIDENCE_S3, else TEAM13_DATA_BUCKET, else STS."""
    if os.environ.get("EVIDENCE_S3"):
        return os.environ["EVIDENCE_S3"], None
    if os.environ.get("TEAM13_DATA_BUCKET"):
        return f"s3://{os.environ['TEAM13_DATA_BUCKET']}/evidence/", None
    try:
        import boto3

        acct = boto3.client("sts", region_name=os.environ.get("AWS_REGION") or "us-east-2").get_caller_identity()["Account"]
    except Exception as e:  # noqa: BLE001
        return None, (f"No AWS credentials are available to this server ({type(e).__name__}). Start it with AWS_PROFILE set "
                      "(e.g. the team SSO profile, logged in) to verify live.")
    return f"s3://bb26-p-team13-data-{acct}-use2/evidence/", None


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


async def run_check(run_id: str, root: Path, prefix: str, timeout_s: float | None = None, python: str = sys.executable) -> dict:
    """Run check_unit for one run and write the sync-format cache file. Raises asyncio.TimeoutError."""
    t0 = time.time()
    env = dict(os.environ)
    proc = await asyncio.create_subprocess_exec(
        python, str(CHECK_UNIT), run_id, "--s3", prefix, "--ledger-dir", str(root), "--json",
        cwd=str(REPO), env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout_s or TIMEOUT_S)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise
    stdout, stderr = out.decode("utf-8", "replace"), err.decode("utf-8", "replace")
    try:
        summary = json.loads(stdout)
    except ValueError:
        summary = None
    doc = {"run_id": run_id, "exit_code": proc.returncode,
           "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
           "elapsed_s": round(time.time() - t0, 1), "summary": summary,
           "stdout": None if summary is not None else stdout[-20000:], "stderr": stderr[-4000:], "live": True}
    f = cache_file(root, run_id)
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True))
    os.replace(tmp, f)
    return doc


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
    async with _lock:
        _running.update(run_id=run_id, started_at=datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"))
        try:
            prefix, why = await asyncio.to_thread(s3_prefix)
            if prefix is None:
                return _err(503, "no_aws_credentials", why or "No AWS credentials.")
            try:
                doc = await run_check(run_id, root, prefix)
            except asyncio.TimeoutError:
                return _err(504, "timeout", f"check_unit did not finish within {int(TIMEOUT_S)} s; nothing was cached.")
        finally:
            _running.clear()
    return JSONResponse(parse(doc), headers=NO_STORE)


async def handle_get(request: Request, run_id: str) -> JSONResponse | dict:
    refused = guard(request, write=False)
    if refused:
        return refused
    if not RUN_ID_RE.match(run_id) or ".." in run_id:
        return _err(400, "bad_run_id", "Invalid run id.")
    return JSONResponse(status(evidence_root(), run_id), headers=NO_STORE)
