"""The job side of tools/sm/run.py: SageMaker's training toolkit runs this file (sagemaker_program).

In order:
  1. job_started: a /team13/runner event (best effort) and the same line on stdout;
  2. checks: the SDK wheel's sha256 (when the sdk channel is present), the declaration exists in S3 and
     names the commit this job runs (the launcher wrote it before submitting);
  3. secrets: TEAM13_SECRET_ARNS maps env names to Secrets Manager ARNs; each value is fetched now and
     set in this process only (it is never printed and never in the job's Environment);
  4. the entry (`-m module args` or `script.py args`) runs in this process, with placeholders filled;
  5. evidence: a module that wrote to {evidence_dir} (a local evlib store) is collected: its own
     declaration must agree with the S3 one on every judged field, then each record goes to S3 as one
     object (evlib "objects" layout) and the result to results/<run_id>.json. A module that wrote to
     S3 itself (EVIDENCE_S3) needs nothing collected;
  6. attestation: every evidence file of the run (records/<run_id>/..., results/<run_id>.json) is read
     back from S3 and its sha256 printed as `team13-evidence {...}`. SageMaker timestamps that line in
     /aws/sagemaker/TrainingJobs, and check_unit matches the hashes against the S3 files.
Exit code: the module's (non-zero fails the job), or 3..6 for the runner's own refusals.
"""

from __future__ import annotations

import hashlib
import json
import os
import runpy
import sys
import traceback
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
if sys.path and Path(sys.path[0] or ".").resolve() == Path(__file__).resolve().parent:
    sys.path.pop(0)  # run as a script: do not let tools/sm/*.py shadow top-level module names
for _p in (str(CODE_ROOT / "contracts"), str(CODE_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools import evlib  # noqa: E402
from tools.sm.events import RunnerLog  # noqa: E402
from tools.sm.spec import CHANNEL_ROOT, LOCAL_EVIDENCE, SDK_CHANNEL, SDK_WHEEL, expand  # noqa: E402

ATTEST_MARK = "team13-evidence"
HYPERPARAMETERS = Path("/opt/ml/input/config/hyperparameters.json")
JUDGED = ("unit", "track", "target", "n_items", "test_set_sha256", "model_id", "grader", "code_sha",
          "baseline_run_id", "subset", "synthetic")
JUDGED_TWIN = ("kind", "version", "scenario", "seed")


class Refused(RuntimeError):
    def __init__(self, code: int, msg: str):
        super().__init__(msg)
        self.code = code


def say(msg: str) -> None:
    print(msg, flush=True)


def entry_argv(hp_path: Path = HYPERPARAMETERS) -> list[str]:
    hps = json.loads(hp_path.read_text())
    v = hps["team13_entry"]
    for _ in range(3):  # the value is JSON text of a list, possibly JSON-quoted once more
        if isinstance(v, list):
            return [str(x) for x in v]
        v = json.loads(v)
    raise Refused(3, "hyperparameter team13_entry is not a JSON list")


def judged_diff(local: dict, remote: dict) -> list[str]:
    """Fields where the module's own declaration disagrees with the one the launcher stored in S3."""
    out = [k for k in JUDGED if local.get(k) != remote.get(k)]
    lt, rt = local.get("twin") or {}, remote.get("twin") or {}
    if lt and rt:
        out += [f"twin.{k}" for k in JUDGED_TWIN if lt.get(k) != rt.get(k)]
        lb, rb = (lt.get("config") or {}).get("backend"), (rt.get("config") or {}).get("backend")
        if lb is not None and rb is not None and lb != rb:
            out.append("twin.config.backend")
    return out


def collect(run_id: str, local_dir: Path, s3, remote_decl: dict) -> list[str]:
    """Upload what the module wrote locally. Returns the S3 URIs written."""
    local = evlib.LocalStore(local_dir)
    mine = evlib.read_declared(run_id, local)
    if mine is not None:
        bad = judged_diff(mine, remote_decl)
        if bad:
            raise Refused(6, "the module declared differently from the declaration stored before the job ("
                             + ", ".join(f"{k}: module {mine.get(k)!r} vs stored {remote_decl.get(k)!r}"
                                         if "." not in k else k for k in bad[:6])
                             + "); nothing uploaded")
    written: list[str] = []
    for kind in evlib.record_kinds(run_id, local):
        recs = evlib.read_records(run_id, kind, local)
        if recs:
            written += evlib.append_records(run_id, kind, recs, store=s3, layout="objects")
    res = evlib.read_result(run_id, local)
    if res is not None:
        written.append(evlib.write_result(res, store=s3))
    return written


def attest(run_id: str, s3) -> list[dict]:
    """Print sha256 of every evidence file the run has in S3 (records and result), as stored."""
    out = []
    rels = [r for r in s3.list(f"records/{run_id}/") if r.endswith((".json", ".jsonl"))]
    if s3.get(f"results/{run_id}.json") is not None:
        rels.append(f"results/{run_id}.json")
    for rel in sorted(rels):
        b = s3.get(rel)
        if b is None:
            continue
        body = {"run_id": run_id, "path": rel, "sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b),
                "store": s3.uri("")}
        say(f"{ATTEST_MARK} {json.dumps(body, sort_keys=True)}")
        out.append(body)
    return out


def fetch_secrets(arns: dict[str, str], region: str, client=None) -> list[str]:
    """Set each secret's value in os.environ (this process and its children). Names only are printed."""
    if not arns:
        return []
    import boto3

    sm = client or boto3.client("secretsmanager", region_name=region)
    for name, arn in sorted(arns.items()):
        try:
            v = sm.get_secret_value(SecretId=arn)
        except Exception as e:  # noqa: BLE001
            code = ((getattr(e, "response", None) or {}).get("Error") or {}).get("Code", e.__class__.__name__)
            raise Refused(5, f"secret for {name} could not be read ({code}); the job role "
                             "(bb26-SageMakerExecution) reads only secrets named AmazonSageMaker-* or tagged "
                             "SageMaker=true") from None
        os.environ[name] = v.get("SecretString") if v.get("SecretString") is not None else v["SecretBinary"].decode()
    return sorted(arns)


def run_argv(argv: list[str]) -> int:
    """Run '-m module args' or 'script.py args' in this process, like python would. Returns the exit code."""
    old = sys.argv[:]
    try:
        if argv[0] == "-m":
            sys.argv = [argv[1], *argv[2:]]
            runpy.run_module(argv[1], run_name="__main__", alter_sys=True)
        else:
            sys.argv = list(argv)
            runpy.run_path(str(CODE_ROOT / argv[0]), run_name="__main__")
        return 0
    except SystemExit as e:
        if e.code is None:
            return 0
        if isinstance(e.code, int):
            return e.code
        say(str(e.code))
        return 1
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 1
    finally:
        sys.argv = old


def adapt_git_helpers(sha: str) -> list[str]:
    """A `git archive` has no .git, so helpers that ask git for the commit get the pinned one instead.
    tools/evlib reads TEAM13_CODE_SHA itself; t3/evals/evshim asks git directly and is adapted here (the
    archive is the commit, clean by construction)."""
    patched = []
    try:
        import t3.evals.evshim as shim  # noqa: PLC0415
    except ImportError:
        return patched
    shim.code_sha = lambda: sha
    shim.code_dirty = lambda: False
    patched.append("t3.evals.evshim")
    return patched


def main() -> int:
    run_id = os.environ["RUN_ID"]
    evidence_s3 = os.environ["EVIDENCE_S3"]
    sha = os.environ["TEAM13_CODE_SHA"]
    region = os.environ.get("TEAM13_REGION", "us-east-2")
    os.environ.setdefault("AWS_DEFAULT_REGION", region)
    os.environ.setdefault("AWS_REGION", region)
    os.environ["PYTHONPATH"] = os.pathsep.join([str(CODE_ROOT / "contracts"), str(CODE_ROOT)]
                                               + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p])
    os.chdir(CODE_ROOT)
    import boto3

    logs = None
    try:
        logs = boto3.client("logs", region_name=region)
    except Exception as e:  # noqa: BLE001
        say(f"team13-runner warning: no CloudWatch Logs client ({e.__class__.__name__})")
    log = RunnerLog(logs, run_id, "job", group=os.environ.get("TEAM13_LOG_GROUP", "/team13/runner"), strict=False)
    job = os.environ.get("TRAINING_JOB_NAME", "")
    log.emit("job_started", job_name=job, code_sha=sha, python=sys.version.split()[0])
    try:
        rc = _run(run_id, evidence_s3, sha, region, log)
    except Refused as e:
        say(f"team13-runner refused: {e}")
        log.emit("job_refused", reason=str(e)[:500], code=e.code)
        rc = e.code
    log.emit("job_entry_finished", rc=rc, job_name=job)
    return rc


def _run(run_id: str, evidence_s3: str, sha: str, region: str, log: RunnerLog) -> int:
    want = os.environ.get("TEAM13_SDK_SHA256")
    wheel = Path(CHANNEL_ROOT) / SDK_CHANNEL / SDK_WHEEL
    if want:
        if not wheel.is_file():
            raise Refused(3, f"the SDK wheel is not in the {SDK_CHANNEL} channel")
        got = hashlib.sha256(wheel.read_bytes()).hexdigest()
        if got != want:
            raise Refused(3, f"SDK wheel sha256 {got} is not the expected {want}")
        say(f"team13-runner sdk wheel {SDK_WHEEL} sha256 {got} (expected)")
    s3 = evlib.S3Store(evidence_s3, region=region)
    decl = evlib.read_declared(run_id, s3)
    if decl is None:
        raise Refused(4, f"no declaration at {s3.uri(f'declared/{run_id}.json')}: the launcher writes it first")
    if decl.get("code_sha") != sha:
        raise Refused(4, f"the declaration names commit {decl.get('code_sha')}, this job runs {sha}")
    names = fetch_secrets(json.loads(os.environ.get("TEAM13_SECRET_ARNS") or "{}"), region)
    if names:
        say(f"team13-runner secrets set from Secrets Manager: {', '.join(names)}")
    patched = adapt_git_helpers(sha)
    argv = expand(entry_argv(), run_id, evidence_s3)
    uses_local = any(LOCAL_EVIDENCE in a for a in argv)
    if uses_local:
        Path(LOCAL_EVIDENCE).mkdir(parents=True, exist_ok=True)
    log.emit("module_started", entry=argv, adapted=patched)
    rc = run_argv(argv)
    log.emit("module_finished", rc=rc)
    if rc != 0:
        return rc
    if uses_local:
        written = collect(run_id, Path(LOCAL_EVIDENCE), s3, decl)
        say(f"team13-runner uploaded {len(written)} evidence file(s) from {LOCAL_EVIDENCE}")
    files = attest(run_id, s3)
    res = next((f for f in files if f["path"] == f"results/{run_id}.json"), None)
    if res:
        log.emit("result_written", path=res["path"], sha256=res["sha256"])
    else:
        say("team13-runner warning: the module wrote no result")
    return 0


if __name__ == "__main__":
    sys.exit(main())
