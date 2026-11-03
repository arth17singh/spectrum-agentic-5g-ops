"""Launch one scored run as a SageMaker training job, in the order that makes it count.

    python -m tools.sm.run --run-id R --declaration decl.json --entry "-m pkg.mod --run-id {run_id}" [opts]
    python -m tools.sm.run --finish R          # wait for a --no-wait launch, summarise, check

Order of a launch (nothing is written before step 3; --dry-run stops after step 2):
  1. refuse: a dirty working tree (unless --allow-dirty, labelled in the declaration), secret-looking
     --env, an unknown instance type, an entry outside the archive, a run id that already has evidence,
     a job name in use, a missing input, a busy GPU type;
  2. build: the source archive (`git archive <sha> -- <paths>` + requirements.txt; its sha256 is
     declared), the declaration (validated with team13_contracts), the CreateTrainingJob request;
  3. declare: evidence/declared/<run_id>.json in S3 FIRST (If-None-Match), then a 2 s gap so the job's
     CreationTime is provably later, then the `declared` event in /team13/runner;
  4. submit: upload jobs/<run_id>/source/sourcedir.tar.gz, CreateTrainingJob (name from the run id,
     HyperParameters team13_run_id/code_sha/source_sha256), `job_submitted` event, then
     evidence/outputs/<run_id>/job-map.json (job, source sha256, pathspec, requirements, image);
  5. the job (tools/sm/entry.py) runs the module, writes per-item records and the result through evlib,
     and prints each evidence file's sha256 to its log;
  6. wait (unless --no-wait): a background `aws sagemaker wait training-job-completed-or-stopped`;
  7. summarise: `job_finished` event, evidence/outputs/<run_id>/job-summary.json (allow-listed fields of
     DescribeTrainingJob, never Environment) and a contract `job` record with billable seconds and cost;
  8. check: tools/check_unit.py <run_id> as a separate process; its exit code is ours.

Entry placeholders (filled in the job): {run_id} {evidence_s3} {evidence_dir} {output_dir} {model_dir}
{channel:<name>}. A module that writes a local evlib store passes --evidence-dir {evidence_dir}; the job
uploads and attests it. A module that writes to S3 itself uses EVIDENCE_S3 and RUN_ID from its env.

Secrets never go in --env (DescribeTrainingJob shows Environment): --secret NAME=<Secrets Manager ARN>
puts only the ARN in the job; the job reads the value at start.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO / "contracts"), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools import evlib  # noqa: E402
from tools.sm import package, spec  # noqa: E402
from tools.sm.events import RunnerLog, ensure_group  # noqa: E402

DECLARED_KEYS = ("unit", "track", "target", "baseline_run_id", "n_items", "n_cases", "test_set_sha256", "subset",
                 "prompt_sha256", "parser", "decoding", "model_id", "grader", "twin", "synthetic")
TEMPLATE_ONLY = ("test_set", "prompt")  # paths in the repo, hashed at the pinned commit
NO_TWIN = {"kind": "none", "version": "none", "scenario": "none", "seed": 0}


class Refusal(RuntimeError):
    """The launch must not go ahead (nothing has been written)."""


class Aws:
    def __init__(self, profile: str | None, region: str):
        import boto3

        self.session = boto3.Session(profile_name=profile, region_name=region)
        self.region = region
        self._c: dict = {}

    def client(self, service: str, region: str | None = None):
        key = (service, region or self.region)
        if key not in self._c:
            self._c[key] = self.session.client(service, region_name=key[1])
        return self._c[key]


def code(e: Exception) -> str:
    r = getattr(e, "response", None) or {}
    return str((r.get("Error") or {}).get("Code", "")) or e.__class__.__name__


def cli_waiter(job: str, region: str, profile: str | None) -> int:
    """`aws sagemaker wait training-job-completed-or-stopped` as a background process; we block on its
    exit (no sleep loop of our own). It polls every 120 s, up to 6 h; non-zero when the job failed."""
    cmd = ["aws", "sagemaker", "wait", "training-job-completed-or-stopped", "--training-job-name", job,
           "--region", region] + (["--profile", profile] if profile else [])
    return subprocess.Popen(cmd).wait()


# --------------------------------------------------------------------------- building

def load_template(arg: str | None) -> dict[str, Any]:
    if not arg:
        return {}
    p = Path(arg)
    text = p.read_text() if p.is_file() else arg
    try:
        d = json.loads(text)
    except ValueError as e:
        raise Refusal(f"--declaration is neither a JSON file nor JSON text: {e}") from None
    if not isinstance(d, dict):
        raise Refusal("--declaration must be a JSON object")
    return d


def at_commit(sha: str, path: str) -> bytes:
    b = package.blob(REPO, sha, path)
    if b is None:
        raise Refusal(f"{path} is not in commit {sha[:12]} (the job sees the commit, not your working tree)")
    return b


def build_declaration(a, sha: str, runner: dict[str, Any]) -> dict[str, Any]:
    d = load_template(a.declaration)
    unknown = set(d) - set(DECLARED_KEYS) - set(TEMPLATE_ONLY) - {"run_id", "code_sha", "region"}
    if unknown:
        raise Refusal(f"--declaration has keys the contract does not accept: {sorted(unknown)}")
    for k in ("run_id", "code_sha", "region"):
        if k in d and d[k] != {"run_id": a.run_id, "code_sha": sha, "region": a.region}[k]:
            raise Refusal(f"--declaration {k}={d[k]!r} disagrees with the launch; leave it out")
    for flag in ("unit", "track", "model_id", "grader", "subset", "parser", "baseline_run_id"):
        if getattr(a, flag, None) is not None:
            d[flag] = getattr(a, flag)
    if a.metric is not None or a.op is not None or a.value is not None:
        if None in (a.metric, a.op, a.value):
            raise Refusal("--metric, --op and --value go together")
        d["target"] = {"metric": a.metric, "op": a.op, "value": a.value, "unit": a.target_unit}
    if a.n_items is not None:
        d["n_items"] = a.n_items
    if a.n_cases is not None:
        d["n_cases"] = a.n_cases
    test_set = a.test_set or d.pop("test_set", None)
    if test_set:
        body = at_commit(sha, test_set)
        digest = evlib.sha256_bytes(body)
        if d.get("test_set_sha256") not in (None, digest):
            raise Refusal(f"test_set_sha256 {d['test_set_sha256']} is not the sha256 of {test_set} at "
                          f"{sha[:12]} ({digest})")
        d["test_set_sha256"] = digest
        d.setdefault("n_items", sum(1 for line in body.decode("utf-8").splitlines() if line.strip()))
    prompt = a.prompt or d.pop("prompt", None)
    if prompt:
        d["prompt_sha256"] = evlib.sha256_bytes(at_commit(sha, prompt))
    if a.decoding:
        d["decoding"] = json.loads(a.decoding)
    twin = json.loads(a.twin) if a.twin else d.get("twin")
    twin = dict(twin) if twin else dict(NO_TWIN)
    cfg = dict(twin.get("config") or {})
    if "runner" in cfg:
        raise Refusal("twin.config.runner is written by the runner; leave it out")
    cfg["runner"] = runner
    twin["config"] = cfg
    d["twin"] = twin
    d["synthetic"] = bool(a.synthetic or d.get("synthetic", False))
    d.update(run_id=a.run_id, code_sha=sha, region=a.region)
    check = {**d, "schema_version": evlib.SCHEMA_VERSION, "kind": "declared", "team": evlib.TEAM,
             "created_at": evlib.utc_now(), "declared_at": evlib.utc_now(), "aws_time": None, "aws_time_source": None}
    try:
        evlib.validate(check)
    except evlib.EvidenceError as e:
        raise Refusal(f"declaration invalid: {e}") from None
    return d


def plan(a, aws=None) -> dict[str, Any]:
    """Everything a launch needs, built and checked locally. No AWS writes; no AWS reads unless the
    account has to come from STS."""
    if not evlib.RUN_ID_RE.match(a.run_id or ""):
        raise Refusal(f"--run-id must match {evlib.RUN_ID_RE.pattern}")
    sha = package.resolve_commit(REPO, a.commit)
    dirty = package.tree_dirty(REPO)
    if dirty and not a.allow_dirty:
        raise Refusal(f"the working tree has uncommitted changes to {len(dirty)} tracked file(s) (e.g. {dirty[0]}); "
                      "commit them (the job runs the commit, not your tree) or pass --allow-dirty")
    try:
        device = spec.device_for(a.instance_type, True if a.gpu else None)
        if not a.gpu and device == "gpu":
            raise spec.SpecError(f"{a.instance_type} is a GPU type: pass --gpu")
        price = spec.price_per_hour(a.instance_type, a.price_per_hour)
        spec.check_role(a.role)
        env = spec.parse_kv(a.env, "--env")
        spec.check_env(env)
        secrets = spec.parse_kv(a.secret, "--secret")
        spec.check_secrets(secrets)
        overlap = set(env) & set(secrets)
        if overlap:
            raise spec.SpecError(f"{sorted(overlap)} given as both --env and --secret")
        argv = spec.parse_entry(a.entry)
        channels = []
        sdk = not a.no_sdk
        if sdk:
            channels.append((spec.SDK_CHANNEL, f"s3://{a.bucket}/{spec.SDK_KEY}"))
        if a.model_s3:
            if not a.gpu:
                raise spec.SpecError("--model-s3 is part of the --gpu profile")
            channels.append((spec.MODEL_CHANNEL, a.model_s3))
        channels += spec.parse_inputs(a.inputs)
        names = [n for n, _ in channels]
        if len(set(names)) != len(names):
            raise spec.SpecError(f"input channel names repeat: {names}")
        reqs = spec.requirements_txt(sdk, a.requirement)
    except spec.SpecError as e:
        raise Refusal(str(e)) from None
    paths = tuple(a.paths or spec.DEFAULT_PATHS)
    files = package.tracked_files(REPO, sha, paths)
    if spec.ENTRY_PROGRAM not in files:
        raise Refusal(f"the pathspec {list(paths)} leaves out {spec.ENTRY_PROGRAM}; include tools")
    if not any(p in files for p in spec.entry_source_paths(argv)):
        raise Refusal(f"the entry's code ({' or '.join(spec.entry_source_paths(argv))}) is not in the archive "
                      f"of {sha[:12]} with pathspec {list(paths)}")
    source = package.build_source(REPO, sha, paths, {"requirements.txt": reqs.encode("utf-8")})
    src_sha = package.sha256(source)
    job = spec.job_name(a.run_id)
    source_key = f"{spec.JOBS_PREFIX}{a.run_id}/source/sourcedir.tar.gz"
    source_uri = f"s3://{a.bucket}/{source_key}"
    volume = a.volume_gb or spec.DEFAULT_VOLUME_GB[device]
    image = spec.image_uri(device, a.region)
    runner = {
        "launcher": "tools/sm/run.py", "job_name": job, "role": a.role, "image": image, "instance_type": a.instance_type,
        "max_run_s": a.max_run, "entry": argv, "source_uri": source_uri, "archive_sha256": src_sha,
        "archive_bytes": len(source), "pathspec": list(paths), "files": len(files),
        "requirements_sha256": evlib.sha256_bytes(reqs.encode("utf-8")),
        "shipped": "the committed tree at code_sha only (git archive); uncommitted changes never reach the job",
        "tree_dirty_at_launch": bool(dirty), "allow_dirty": bool(a.allow_dirty),
    }
    if dirty:
        runner["dirty_note"] = (f"{len(dirty)} tracked file(s) had uncommitted changes at launch; they were NOT "
                                f"shipped: the job ran commit {sha}")
    decl = build_declaration(a, sha, runner)
    account = a.account
    if not account and aws is not None:
        account = aws.client("sts").get_caller_identity()["Account"]
    role_arn = spec.role_arn(account or "<account>", a.role)
    try:
        request = spec.build_request(run_id=a.run_id, job=job, role_arn=role_arn, region=a.region, bucket=a.bucket,
                                     device=device, instance_type=a.instance_type, volume_gb=volume,
                                     max_run_s=a.max_run, source_uri=source_uri, source_sha256=src_sha, code_sha=sha,
                                     argv=argv, channels=channels, env=env, secrets=secrets, sdk=sdk)
    except spec.SpecError as e:
        raise Refusal(str(e)) from None
    return {"sha": sha, "dirty": dirty, "device": device, "price": price, "argv": argv, "channels": channels,
            "env": env, "secrets": secrets, "sdk": sdk, "requirements": reqs, "paths": list(paths), "files": files,
            "source": source, "source_sha256": src_sha, "source_key": source_key, "source_uri": source_uri,
            "job": job, "image": image, "volume": volume, "declaration": decl, "request": request,
            "role_arn": role_arn}


def print_plan(p: dict[str, Any], a, out: Callable[[str], None]) -> None:
    out(f"run_id      {a.run_id}")
    out(f"job name    {p['job']}  ({a.instance_type}, {p['device']}, ${p['price']:.3f}/h; at most "
        f"${p['price'] * a.max_run / 3600:.3f} for --max-run {a.max_run}s)")
    out(f"commit      {p['sha']}" + (f"  (working tree dirty: {len(p['dirty'])} file(s), NOT shipped)" if p['dirty'] else ""))
    out(f"source      {len(p['files'])} files from {p['paths']} + requirements.txt -> {p['source_uri']}")
    out(f"            sha256 {p['source_sha256']}, {len(p['source'])} bytes")
    out(f"image       {p['image']}")
    out(f"role        {p['role_arn']}")
    out(f"entry       {' '.join(p['argv'])}")
    out("requirements.txt:")
    for line in p["requirements"].splitlines():
        out(f"  {line}")


# --------------------------------------------------------------------------- AWS steps

def preflight(p: dict[str, Any], a, aws, store, out) -> None:
    rid = a.run_id
    found = [rel for rel in (f"declared/{rid}.json", f"results/{rid}.json") if store.get(rel) is not None]
    found += store.list(f"records/{rid}/")[:3] + store.list(f"outputs/{rid}/")[:3]
    if found:
        raise Refusal(f"run id {rid} already has evidence ({', '.join(found[:3])}); a run id is used once")
    name, path = spec.role_name_and_path(a.role)
    try:  # a missing role would fail CreateTrainingJob AFTER the declaration: check it first
        got = aws.client("iam").get_role(RoleName=name)["Role"]
    except Exception as e:  # noqa: BLE001
        raise Refusal(f"role {a.role}: IAM GetRole failed ({code(e)}); create it in CloudShell first") from None
    if got.get("Path") != path or got.get("Arn") != p["role_arn"]:
        raise Refusal(f"role {name} is {got.get('Arn')}, not {p['role_arn']}")
    sm = aws.client("sagemaker")
    try:
        sm.describe_training_job(TrainingJobName=p["job"])
        raise Refusal(f"a training job named {p['job']} already exists")
    except Refusal:
        raise
    except Exception as e:  # noqa: BLE001
        if code(e) not in ("ValidationException", "ResourceNotFound"):
            raise
    s3 = aws.client("s3")
    for name, uri in p["channels"]:
        bucket, _, key = uri[len("s3://"):].partition("/")
        r = s3.list_objects_v2(Bucket=bucket, Prefix=key, MaxKeys=1)
        if not r.get("KeyCount") and not r.get("Contents"):
            raise Refusal(f"input channel {name}: nothing at {uri}")
    busy = []
    for page in sm.get_paginator("list_training_jobs").paginate(StatusEquals="InProgress"):
        for s in page.get("TrainingJobSummaries", []):
            d = sm.describe_training_job(TrainingJobName=s["TrainingJobName"])
            if (d.get("ResourceConfig") or {}).get("InstanceType") == a.instance_type:
                busy.append(s["TrainingJobName"])
    if busy and a.instance_type in spec.GPU_TYPES:
        raise Refusal(f"{a.instance_type} is busy ({', '.join(busy)}); the quota is one job per GPU type")
    if busy:
        out(f"note: {len(busy)} other {a.instance_type} job(s) in progress ({', '.join(busy)})")
    if ensure_group(aws.client("logs")):
        out(f"created log group {spec.LOG_GROUP} (retention {spec.LOG_RETENTION_DAYS} days)")


def launch(a, aws, out: Callable[[str], None] = print, waiter: Callable[..., int] = cli_waiter,
           sleep: Callable[[float], None] = time.sleep) -> int:
    p = plan(a, aws)
    print_plan(p, a, out)
    store = evlib.S3Store(f"s3://{a.bucket}/{spec.EVIDENCE_PREFIX}", client=aws.client("s3"), region=a.region)
    preflight(p, a, aws, store, out)

    # 3. the declaration, FIRST
    decl_uri = evlib.write_declared(p["declaration"], store)
    lm = store.client().list_objects_v2(Bucket=store.bucket, Prefix=f"{store.prefix}declared/{a.run_id}.json",
                                        MaxKeys=1).get("Contents", [{}])[0].get("LastModified")
    out(f"declared    {decl_uri}  (S3 LastModified {spec.iso(lm) if lm else '?'})")
    sleep(a.gap_s)  # S3 LastModified is floored to the second: keep CreationTime clearly later
    log = RunnerLog(aws.client("logs"), a.run_id, "launcher", strict=False, echo=out, clock=getattr(aws, "clock", None))
    log.emit("declared", declaration=decl_uri, s3_last_modified=spec.iso(lm) if lm else None,
             code_sha=p["sha"], source_sha256=p["source_sha256"])

    # 4. submit
    s3 = aws.client("s3")
    s3.put_object(Bucket=a.bucket, Key=p["source_key"], Body=p["source"], IfNoneMatch="*",
                  ContentType="application/gzip")
    sm = aws.client("sagemaker")
    try:
        sm.create_training_job(**p["request"])
    except Exception as e:  # noqa: BLE001
        log.emit("submit_failed", job_name=p["job"], error=code(e))
        out(f"CreateTrainingJob failed: {code(e)}: {e}")
        out(f"the declaration {decl_uri} stays (a declared run that never started)")
        return 1
    d = sm.describe_training_job(TrainingJobName=p["job"])
    created = spec.iso(d["CreationTime"])
    log.emit("job_submitted", job_name=p["job"], job_arn=d["TrainingJobArn"], region=a.region,
             creation_time=created, source_sha256=p["source_sha256"])
    jm = {
        "run_id": a.run_id, "job_name": p["job"], "job_arn": d["TrainingJobArn"], "job_kind": "training",
        "region": a.region, "creation_time": created, "declaration": decl_uri, "code_sha": p["sha"],
        "source_uri": p["source_uri"], "source_sha256": p["source_sha256"], "source_bytes": len(p["source"]),
        "pathspec": p["paths"], "source_extras": list(spec.SOURCE_EXTRAS), "requirements_txt": p["requirements"],
        "requirements_sha256": evlib.sha256_bytes(p["requirements"].encode("utf-8")), "image": p["image"],
        "role": a.role, "role_arn": p["role_arn"],
        "instance_type": a.instance_type, "volume_gb": p["volume"], "max_run_s": a.max_run, "entry": p["argv"],
        "inputs": dict(p["channels"]), "sdk_wheel_sha256": spec.SDK_WHEEL_SHA256 if p["sdk"] else None,
        "secrets": p["secrets"], "env_keys": sorted(p["request"]["Environment"]), "launcher": "tools/sm/run.py",
        "price_usd_per_hour": p["price"], "price_source": spec.PRICE_SOURCE if a.price_per_hour is None
        else "--price-per-hour",
    }
    jm_uri = store.put_new(f"outputs/{a.run_id}/job-map.json", evlib.dumps_doc(jm))
    out(f"submitted   {p['job']}  (CreationTime {created})")
    out(f"job-map     {jm_uri}")
    if not a.wait:
        out(f"not waiting; later: python -m tools.sm.run --finish {a.run_id}"
            + (f" --profile {a.profile}" if a.profile else ""))
        return 0
    return finish(a, aws, out, waiter)


def finish(a, aws, out: Callable[[str], None] = print, waiter: Callable[..., int] = cli_waiter) -> int:
    rid = a.run_id
    store = evlib.S3Store(f"s3://{a.bucket}/{spec.EVIDENCE_PREFIX}", client=aws.client("s3"), region=a.region)
    raw = store.get(f"outputs/{rid}/job-map.json")
    if raw is None:
        raise Refusal(f"no job-map for {rid} (outputs/{rid}/job-map.json): was it launched with tools/sm/run.py?")
    jm = json.loads(raw)
    job, region = jm["job_name"], jm.get("region", a.region)
    sm = aws.client("sagemaker", region)
    log = RunnerLog(aws.client("logs", region), rid, "launcher", strict=False, echo=out,
                    clock=getattr(aws, "clock", None))
    out(f"waiting     {job} (background: aws sagemaker wait training-job-completed-or-stopped)")
    for attempt in range(5):  # the CLI waiter gives up after 6 h; a job may run 24 h
        wrc = waiter(job, region, a.profile)
        d = sm.describe_training_job(TrainingJobName=job)
        if d["TrainingJobStatus"] not in ("InProgress", "Stopping"):
            break
        out(f"waiter exited {wrc} with the job still {d['TrainingJobStatus']} (attempt {attempt + 1})")
    else:
        raise Refusal(f"{job} is still {d['TrainingJobStatus']}; run --finish {rid} again later")
    status = d["TrainingJobStatus"]
    billable = int(d.get("BillableTimeInSeconds") or 0)
    price = float(jm.get("price_usd_per_hour") or spec.price_per_hour(jm["instance_type"]))
    cost = round(billable / 3600.0 * price, 6)
    log.emit("job_finished", job_name=job, status=status, secondary_status=d.get("SecondaryStatus"),
             aws_start=spec.iso(d.get("CreationTime")), aws_training_start=spec.iso(d.get("TrainingStartTime")),
             aws_end=spec.iso(d.get("TrainingEndTime")), billable_s=billable, failure_reason=d.get("FailureReason"))
    summ = spec.summary(d)
    summ.update(run_id=rid, est_cost_usd=cost, cost_basis=f"BillableTimeInSeconds x ${price}/h ({jm.get('price_source')})")
    summ_uri = store.put_new(f"outputs/{rid}/job-summary.json", evlib.dumps_doc(summ))
    rec_uri = evlib.append_records(rid, "job", [spec.job_record(d, region, price)], store=store)[0]
    log.emit("summary_written", job_summary=summ_uri, job_record=rec_uri)
    out(f"finished    {job}: {status} {d.get('SecondaryStatus', '')}  created {spec.iso(d.get('CreationTime'))}, "
        f"training {spec.iso(d.get('TrainingStartTime'))} .. {spec.iso(d.get('TrainingEndTime'))}")
    out(f"billable    {billable} s x ${price}/h = ${cost:.6f}")
    if d.get("FailureReason"):
        out(f"failure     {d['FailureReason']}")
    res = evlib.read_result(rid, store)
    out(f"result      {res['metric']} = {res['value']} target {res['target']['op']} {res['target']['value']} -> "
        f"{'PASSED' if res['passed'] else 'MISSED'}" if res else "result      none written")
    if a.no_check:
        return 0 if status == "Completed" and res else 1
    cmd = [sys.executable, str(REPO / "tools" / "check_unit.py"), rid, "--s3", store.uri("")]
    if a.ledger_dir:
        cmd += ["--ledger-dir", str(a.ledger_dir)]
    env = {**os.environ, "AWS_DEFAULT_REGION": region}
    if a.profile:
        env["AWS_PROFILE"] = a.profile
    out(f"checking    {' '.join(cmd[1:])}")
    rc = subprocess.run(cmd, env=env).returncode
    log.emit("checked", exit=rc)
    return rc if status == "Completed" else 1


# --------------------------------------------------------------------------- CLI

def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m tools.sm.run", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--run-id")
    ap.add_argument("--finish", metavar="RUN_ID", help="wait for, summarise and check a --no-wait launch")
    g = ap.add_argument_group("declaration (a JSON file/text, and/or fields; flags win)")
    g.add_argument("--declaration", help="JSON object of declared fields; may use test_set/prompt: repo paths")
    g.add_argument("--unit")
    g.add_argument("--track", choices=["track-1", "track-2", "track-3", "bridge"])
    g.add_argument("--metric")
    g.add_argument("--op", choices=list(evlib.OPS))
    g.add_argument("--value", type=float)
    g.add_argument("--target-unit", default="ratio")
    g.add_argument("--test-set", help="repo path of the frozen test set (hashed at the commit)")
    g.add_argument("--n-items", type=int)
    g.add_argument("--n-cases", type=int)
    g.add_argument("--model-id")
    g.add_argument("--grader")
    g.add_argument("--subset")
    g.add_argument("--parser")
    g.add_argument("--prompt", help="repo path of the prompt (hashed at the commit)")
    g.add_argument("--decoding", help="JSON object")
    g.add_argument("--twin", help="JSON object {kind, version, scenario, seed, config?}")
    g.add_argument("--baseline-run-id")
    g.add_argument("--synthetic", action="store_true")
    j = ap.add_argument_group("job")
    j.add_argument("--entry", help="'-m module args' or 'path/script.py args' (placeholders: see above)")
    j.add_argument("--instance-type")
    j.add_argument("--gpu", action="store_true", help=f"GPU profile: GPU DLC, default {spec.DEFAULT_GPU}, 100 GB")
    j.add_argument("--model-s3", help="GPU profile: model weights prefix, arrives as {channel:model}")
    j.add_argument("--max-run", type=int, default=3600, help="seconds (the spend bound; default 3600)")
    j.add_argument("--volume-gb", type=int)
    j.add_argument("--inputs", action="append", default=[], help="[name=]s3://bucket/prefix (repeatable)")
    j.add_argument("--env", action="append", default=[], help="NAME=value, not secrets (repeatable)")
    j.add_argument("--secret", action="append", default=[], help="NAME=<Secrets Manager ARN> (repeatable)")
    j.add_argument("--requirement", action="append", default=[], help="extra requirements.txt line (repeatable)")
    j.add_argument("--no-sdk", action="store_true", help="do not install the ran-rl wheel")
    j.add_argument("--paths", action="append", help=f"pathspec to ship (repeatable; default {' '.join(spec.DEFAULT_PATHS)})")
    j.add_argument("--role", default=spec.ROLE_NAME,
                   help=f"execution role: {spec.ROLE_NAME} (default) or bb26/team13-<name>, e.g. "
                        "bb26/team13-sm-bedrock-eval (Bedrock baselines)")
    j.add_argument("--commit", default="HEAD")
    j.add_argument("--allow-dirty", action="store_true", help="launch from a dirty tree (labelled; the commit ships)")
    j.add_argument("--price-per-hour", type=float, help="for an instance type with no price on file")
    w = ap.add_argument_group("run")
    w.add_argument("--wait", dest="wait", action="store_true", default=True)
    w.add_argument("--no-wait", dest="wait", action="store_false")
    w.add_argument("--dry-run", action="store_true", help="print the plan, declaration and request; write nothing")
    w.add_argument("--no-check", action="store_true")
    w.add_argument("--ledger-dir", type=Path, help="passed to check_unit (part d)")
    w.add_argument("--gap-s", type=float, default=2.0)
    w.add_argument("--profile", default=os.environ.get("AWS_PROFILE"))
    w.add_argument("--region", default=spec.REGION)
    w.add_argument("--bucket", default=spec.BUCKET)
    w.add_argument("--account", help="account id for the role ARN (default: STS)")
    return ap


def main(argv: list[str] | None = None, aws=None, out: Callable[[str], None] = print,
         waiter: Callable[..., int] = cli_waiter, sleep: Callable[[float], None] = time.sleep) -> int:
    a = parser().parse_args(argv)
    os.environ.setdefault("AWS_SDK_UA_APP_ID", "AWSSkill-SageMaker")
    if a.finish:
        a.run_id = a.finish
        return finish(a, aws or Aws(a.profile, a.region), out, waiter)
    if not a.run_id or not a.entry:
        parser().error("--run-id and --entry are required (or --finish RUN_ID)")
    a.instance_type = a.instance_type or (spec.DEFAULT_GPU if a.gpu else spec.DEFAULT_CPU)
    try:
        if a.dry_run:
            if aws is None and not a.account:
                try:
                    aws = Aws(a.profile, a.region)
                except Exception:  # noqa: BLE001 - no boto3 or no profile: print with a placeholder
                    aws = None
            try:
                p = plan(a, aws)
            except Refusal:
                raise
            except Exception as e:  # noqa: BLE001 - STS unavailable in a dry run: placeholder account
                out(f"note: account not resolved ({code(e)}); the role ARN shows <account>")
                p = plan(a, None)
            out("DRY RUN: nothing is written (no S3 object, no log group or event, no job).")
            print_plan(p, a, out)
            out(f"declaration -> s3://{a.bucket}/{spec.EVIDENCE_PREFIX}declared/{a.run_id}.json (written FIRST):")
            out(json.dumps(p["declaration"], indent=2, sort_keys=True))
            out("CreateTrainingJob request (Environment values redacted):")
            out(json.dumps(spec.redacted(p["request"]), indent=2, sort_keys=True))
            return 0
        return launch(a, aws or Aws(a.profile, a.region), out, waiter, sleep)
    except Refusal as e:
        out(f"refused: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
