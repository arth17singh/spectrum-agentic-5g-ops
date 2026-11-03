"""Fixed facts and pure builders for the SageMaker job runner. No AWS calls in this file.

Everything a CreateTrainingJob request needs is built here so `run.py --dry-run` can print the exact
request and tests can check it without AWS.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from datetime import datetime, timezone
from typing import Any

REGION = "us-east-2"
BUCKET = "bb26-p-team13-data-483598626742-use2"
ROLE_NAME = "bb26-SageMakerExecution"  # ready-made by the organisers; never create IAM from here
# --role: the organisers' role, or a team role created in CloudShell under /bb26/ with the event boundary
# (e.g. bb26/team13-sm-bedrock-eval from team13-kit/iam/bedrock_eval_role.sh). Nothing else is passed.
TEAM_ROLE_RE = re.compile(r"^bb26/team13-[A-Za-z0-9+=,.@_-]{1,57}$")
EVIDENCE_PREFIX = "evidence/"
JOBS_PREFIX = "jobs/"  # jobs/<run_id>/source/sourcedir.tar.gz and jobs/<run_id>/output/
LOG_GROUP = "/team13/runner"
LOG_RETENTION_DAYS = 30  # same as /team13/checkpoint and /team13/bedrock-invocations

# The organisers' SDK wheel, copied to our private bucket (not redistributed, never committed).
SDK_VERSION = "1.2.9"
SDK_WHEEL = f"ran_rl-{SDK_VERSION}-py3-none-any.whl"
SDK_WHEEL_SHA256 = "9b2978a25bb950f9d0703457537e04403763043349c0976809ea495ca0099fcf"
SDK_KEY = f"sdk/{SDK_VERSION}/{SDK_WHEEL}"
SDK_CHANNEL = "sdk"
# The twin is a GP model (float32 torch); pin the GP libraries to the team venv's versions so a job
# replays artifacts frozen locally as closely as the platform allows.
SDK_PINS = ("gpytorch==1.15.2", "linear-operator==0.6.1")

# AWS Deep Learning Containers, PyTorch training, SageMaker flavour (training toolkit inside), x86_64.
# py312 matches the team venv (3.12.7). Pinned full tags, never a floating one.
DLC_ACCOUNT = "763104351884"
IMAGE_TAGS = {
    "cpu": "pytorch-training:2.9.0-cpu-py312-ubuntu22.04-sagemaker-v2.2",
    "gpu": "pytorch-training:2.9.0-gpu-py312-cu130-ubuntu22.04-sagemaker-v2.2",
}

# On-demand training prices, us-east-2, USD per hour: AWS Price List API, service AmazonSageMaker,
# usagetype USE2-Train:<instance>, effective 2026-11-01 (read 2026-11-02).
PRICE_SOURCE = ("AWS Price List API: AmazonSageMaker OnDemand, usagetype USE2-Train:<instance>, us-east-2, "
                "effective 2026-11-01 (read 2026-11-02)")
PRICES = {
    "ml.m5.xlarge": 0.230, "ml.m5.2xlarge": 0.461, "ml.m5.4xlarge": 0.922,
    "ml.c5.xlarge": 0.204, "ml.c5.2xlarge": 0.408, "ml.c5.4xlarge": 0.816,
    "ml.g5.2xlarge": 1.515, "ml.g6.2xlarge": 1.222, "ml.g5.4xlarge": 2.030, "ml.g6.xlarge": 1.127,
}
GPU_TYPES = ("ml.g6.2xlarge", "ml.g5.2xlarge", "ml.g5.4xlarge", "ml.g6.xlarge")  # quota: 1 job per type
DEFAULT_CPU = "ml.m5.xlarge"
DEFAULT_GPU = "ml.g6.2xlarge"  # the Participant Guide's first choice
DEFAULT_VOLUME_GB = {"cpu": 30, "gpu": 100}
MODEL_CHANNEL = "model"  # --gpu --model-s3: the weights arrive here, /opt/ml/input/data/model

# The code that runs: these top-level paths of the commit (git pathspec). Docs, dashboard and images
# stay out (they are 30 MB of the 33 MB archive); `--paths .` ships everything.
DEFAULT_PATHS = ("contracts", "silos", "t1", "t2", "t3", "tools", "twin")
ENTRY_PROGRAM = "tools/sm/entry.py"  # a tracked file, so part b compares it with git like the rest
SOURCE_EXTRAS = ("requirements.txt",)  # the only untracked file in the archive; recorded in the job-map
CODE_ROOT = "/opt/ml/code"
CHANNEL_ROOT = "/opt/ml/input/data"
LOCAL_EVIDENCE = "/opt/ml/output/data/evidence-local"  # {evidence_dir}: for modules that write locally

JOB_NAME_RE = re.compile(r"^[a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}$")
SHA40_RE = re.compile(r"^[0-9a-f]{40}$")
CHANNEL_RE = re.compile(r"^[A-Za-z0-9\.\-_]{1,64}$")
ENV_KEY_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]{0,511}$")
# DescribeTrainingJob shows Environment to anyone who can describe the job: secrets never go there.
SECRETISH_KEY = re.compile(r"TOKEN|SECRET|PASSW|API_?KEY|PRIVATE|CREDENTIAL|ACCESS_?KEY|AUTH|COOKIE|SESSION",
                           re.IGNORECASE)
CONFIDENTIAL_KEYS = ("RAN_RL_TWIN_URL",)  # the twin address is confidential too (not a credential by name)
SECRET_ARN_RE = re.compile(r"^arn:aws[a-z-]*:secretsmanager:[a-z0-9-]+:\d{12}:secret:[A-Za-z0-9/_+=.@-]+$")

# Fields of DescribeTrainingJob that may be written into evidence. Never Environment.
SUMMARY_FIELDS = ("TrainingJobName", "TrainingJobArn", "TrainingJobStatus", "SecondaryStatus", "FailureReason",
                  "CreationTime", "TrainingStartTime", "TrainingEndTime", "LastModifiedTime",
                  "BillableTimeInSeconds", "TrainingTimeInSeconds", "ResourceConfig", "StoppingCondition",
                  "OutputDataConfig", "ModelArtifacts", "SecondaryStatusTransitions", "RoleArn",
                  "EnableNetworkIsolation", "EnableManagedSpotTraining")
SUMMARY_HYPERPARAMETERS = ("sagemaker_program", "sagemaker_submit_directory", "team13_run_id", "team13_code_sha",
                           "team13_source_sha256", "team13_entry")


class SpecError(ValueError):
    """A launch request that must not go out."""


def iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def job_name(run_id: str) -> str:
    """SageMaker job name from a run id: [A-Za-z0-9-], at most 63 characters, always ending in 6 hex
    characters of sha256(run_id) so two run ids that sanitise alike never share a name."""
    h = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:6]
    base = re.sub(r"[^A-Za-z0-9]+", "-", run_id).strip("-") or "run"
    name = f"{base[:63 - 7].rstrip('-')}-{h}"
    assert JOB_NAME_RE.match(name), name
    return name


def stream_name(run_id: str, writer: str) -> str:
    """CloudWatch stream names cannot contain ':' or '*'."""
    return f"{run_id.replace(':', '_').replace('*', '_')}/{writer}"


def check_role(role: str) -> str:
    """The job's execution role, as '<path>/<name>' without the leading slash. Only the organisers' role or a
    team role under /bb26/ named team13-* (created in CloudShell with the event boundary) may be passed."""
    if role == ROLE_NAME or TEAM_ROLE_RE.match(role or ""):
        return role
    raise SpecError(f"--role {role!r}: use {ROLE_NAME} (default) or a team role bb26/team13-<name> "
                    "(created in CloudShell under /bb26/ with the event boundary)")


def role_arn(account: str, role: str) -> str:
    return f"arn:aws:iam::{account}:role/{role}"


def role_name_and_path(role: str) -> tuple[str, str]:
    """('team13-x', '/bb26/') for 'bb26/team13-x'; ('bb26-SageMakerExecution', '/') for the default."""
    path, _, name = role.rpartition("/")
    return name, f"/{path}/" if path else "/"


def device_for(instance_type: str, gpu: bool | None) -> str:
    is_gpu = instance_type.startswith(("ml.g", "ml.p"))
    if gpu is True and not is_gpu:
        raise SpecError(f"--gpu needs a GPU instance type ({', '.join(GPU_TYPES)}), not {instance_type}")
    if gpu is False and is_gpu:
        raise SpecError(f"{instance_type} is a GPU type: pass --gpu (GPU image, larger volume)")
    return "gpu" if is_gpu else "cpu"


def image_uri(device: str, region: str = REGION) -> str:
    return f"{DLC_ACCOUNT}.dkr.ecr.{region}.amazonaws.com/{IMAGE_TAGS[device]}"


def price_per_hour(instance_type: str, override: float | None = None) -> float:
    if override is not None:
        if override < 0:
            raise SpecError("--price-per-hour must be >= 0")
        return float(override)
    if instance_type not in PRICES:
        raise SpecError(f"no price on file for {instance_type}; known: {', '.join(PRICES)} "
                        "(pass --price-per-hour from the AWS Price List to use another type)")
    return PRICES[instance_type]


def requirements_txt(sdk: bool, extra: list[str]) -> str:
    """The job's requirements.txt (the training toolkit pip-installs it before the entry runs)."""
    lines = ["# generated by tools/sm/run.py: pip installs this before tools/sm/entry.py runs"]
    if sdk:
        lines.append(f"ran-rl[maveric] @ file://{CHANNEL_ROOT}/{SDK_CHANNEL}/{SDK_WHEEL}")
        lines += list(SDK_PINS)
    for x in extra:
        x = x.strip()
        if not x or "\n" in x:
            raise SpecError(f"bad requirement line {x!r}")
        lines.append(x)
    return "\n".join(lines) + "\n"


def parse_kv(items: list[str], what: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for it in items or []:
        k, sep, v = it.partition("=")
        if not sep or not ENV_KEY_RE.match(k):
            raise SpecError(f"{what} {it!r}: expected NAME=value with NAME like [A-Za-z_][A-Za-z0-9_]*")
        if k in out:
            raise SpecError(f"{what} {k} given twice")
        out[k] = v
    return out


def check_env(env: dict[str, str]) -> None:
    for k in env:
        if SECRETISH_KEY.search(k) or k in CONFIDENTIAL_KEYS or k.startswith("AWS_"):
            raise SpecError(f"--env {k}: DescribeTrainingJob shows a job's Environment to anyone who can describe "
                            f"it, so secrets (and the twin address) never go there. Store it in Secrets Manager "
                            f"and pass --secret {k}=<secret ARN>; the job fetches it at start.")
        if k.startswith(("TEAM13_", "SM_", "SAGEMAKER_")) or k in ("RUN_ID", "EVIDENCE_S3", "CODE_SHA", "PYTHONPATH"):
            raise SpecError(f"--env {k}: reserved by the runner or SageMaker")


def check_secrets(secrets: dict[str, str]) -> None:
    for k, arn in secrets.items():
        if not SECRET_ARN_RE.match(arn):
            raise SpecError(f"--secret {k}: {arn!r} is not a Secrets Manager secret ARN")


def parse_inputs(items: list[str]) -> list[tuple[str, str]]:
    """[name=]s3://bucket/prefix -> (channel, uri). Unnamed inputs become input0, input1, ..."""
    out: list[tuple[str, str]] = []
    for i, it in enumerate(items or []):
        name, sep, uri = it.partition("=")
        if not sep or name.startswith("s3:"):
            name, uri = f"input{i}", it
        if not uri.startswith("s3://") or len(uri) <= len("s3://"):
            raise SpecError(f"--inputs {it!r}: expected [name=]s3://bucket/prefix")
        if not CHANNEL_RE.match(name):
            raise SpecError(f"--inputs {it!r}: channel name must match {CHANNEL_RE.pattern}")
        out.append((name, uri))
    names = [n for n, _ in out]
    if len(set(names)) != len(names):
        raise SpecError("--inputs: channel names repeat")
    return out


def parse_entry(entry: str) -> list[str]:
    """'-m pkg.mod args...' or 'path/to/script.py args...' -> argv (placeholders still unexpanded)."""
    argv = shlex.split(entry)
    if not argv:
        raise SpecError("--entry is empty")
    if argv[0] == "-m":
        if len(argv) < 2 or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$", argv[1]):
            raise SpecError(f"--entry {entry!r}: '-m' needs a module name")
    elif not argv[0].endswith(".py"):
        raise SpecError(f"--entry {entry!r}: start with '-m <module>' or a path to a .py script")
    return argv


def entry_source_paths(argv: list[str]) -> list[str]:
    """Where the entry's code lives in the repo (to check it is inside the archive)."""
    if argv[0] == "-m":
        base = argv[1].replace(".", "/")
        return [f"{base}.py", f"{base}/__main__.py"]
    return [argv[0]]


def expand(argv: list[str], run_id: str, evidence_s3: str) -> list[str]:
    """Placeholders the job fills in: {run_id} {evidence_s3} {evidence_dir} {output_dir} {model_dir}
    {channel:<name>}. Anything else in braces is left alone."""
    subs = {"{run_id}": run_id, "{evidence_s3}": evidence_s3, "{evidence_dir}": LOCAL_EVIDENCE,
            "{output_dir}": "/opt/ml/output/data", "{model_dir}": "/opt/ml/model"}
    out = []
    for a in argv:
        for k, v in subs.items():
            a = a.replace(k, v)
        a = re.sub(r"\{channel:([A-Za-z0-9.\-_]{1,64})\}", lambda m: f"{CHANNEL_ROOT}/{m.group(1)}", a)
        out.append(a)
    return out


def hp(value: Any) -> str:
    """HyperParameters values are strings; the SageMaker SDK JSON-encodes them and the toolkit decodes."""
    return json.dumps(value)


def build_request(*, run_id: str, job: str, role_arn: str, region: str, bucket: str, device: str,
                  instance_type: str, volume_gb: int, max_run_s: int, source_uri: str, source_sha256: str,
                  code_sha: str, argv: list[str], channels: list[tuple[str, str]], env: dict[str, str],
                  secrets: dict[str, str], sdk: bool) -> dict[str, Any]:
    """The CreateTrainingJob request. HyperParameters carry the run id, commit and source hash: AWS records
    them at creation and they cannot be changed afterwards, so they tie the job to the run."""
    if not 60 <= max_run_s <= 86400:
        raise SpecError("--max-run must be 60..86400 seconds")
    if not SHA40_RE.match(code_sha):
        raise SpecError(f"code_sha {code_sha!r} is not a full 40-character commit sha")
    if not 1 <= volume_gb <= 16384:
        raise SpecError("--volume-gb out of range")
    environment = {
        "RUN_ID": run_id,
        "EVIDENCE_S3": f"s3://{bucket}/{EVIDENCE_PREFIX}",
        "CODE_SHA": code_sha,  # the declared commit; job entry points (e.g. t3.trust.job_*) read it first
        "TEAM13_CODE_SHA": code_sha,
        "TEAM13_REGION": region,
        "TEAM13_LOG_GROUP": LOG_GROUP,
    }
    if sdk:
        environment["TEAM13_SDK_SHA256"] = SDK_WHEEL_SHA256
    if secrets:
        environment["TEAM13_SECRET_ARNS"] = json.dumps(secrets, sort_keys=True)  # ARNs, never values
    environment.update(env)
    return {
        "TrainingJobName": job,
        "RoleArn": role_arn,
        "AlgorithmSpecification": {"TrainingImage": image_uri(device, region), "TrainingInputMode": "File"},
        "HyperParameters": {
            "sagemaker_program": hp(ENTRY_PROGRAM),
            "sagemaker_submit_directory": hp(source_uri),
            "sagemaker_region": hp(region),
            "sagemaker_container_log_level": "20",
            "sagemaker_job_name": hp(job),
            "team13_run_id": hp(run_id),
            "team13_code_sha": hp(code_sha),
            "team13_source_sha256": hp(source_sha256),
            "team13_entry": hp(json.dumps(argv)),
        },
        "InputDataConfig": [
            {"ChannelName": name, "InputMode": "File",
             "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": uri,
                                             "S3DataDistributionType": "FullyReplicated"}}}
            for name, uri in channels
        ],
        "OutputDataConfig": {"S3OutputPath": f"s3://{bucket}/{JOBS_PREFIX}{run_id}/output"},
        "ResourceConfig": {"InstanceType": instance_type, "InstanceCount": 1, "VolumeSizeInGB": volume_gb},
        "StoppingCondition": {"MaxRuntimeInSeconds": max_run_s},
        "Environment": environment,
        "EnableNetworkIsolation": False,  # the toolkit pip-installs requirements.txt from PyPI
        "Tags": [{"Key": "team", "Value": "team13"}, {"Key": "team13:run_id", "Value": run_id},
                 {"Key": "team13:code_sha", "Value": code_sha}, {"Key": "team13:launcher", "Value": "tools/sm/run.py"}],
    }


def redacted(request: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(request))
    out["Environment"] = {k: "<redacted>" for k in out.get("Environment", {})}
    return out


def unquote_hp(v: Any) -> Any:
    if isinstance(v, str):
        try:
            return json.loads(v)
        except ValueError:
            return v
    return v


def summary(describe: dict[str, Any]) -> dict[str, Any]:
    """The allow-listed part of DescribeTrainingJob for evidence/outputs/<run_id>/job-summary.json."""
    out: dict[str, Any] = {}
    for k in SUMMARY_FIELDS:
        if k in describe:
            out[k] = describe[k]
    alg = describe.get("AlgorithmSpecification") or {}
    out["AlgorithmSpecification"] = {k: alg[k] for k in ("TrainingImage", "TrainingInputMode") if k in alg}
    hps = describe.get("HyperParameters") or {}
    out["HyperParameters"] = {k: hps[k] for k in SUMMARY_HYPERPARAMETERS if k in hps}
    out["InputDataConfig"] = [{"ChannelName": c.get("ChannelName"),
                               "S3Uri": ((c.get("DataSource") or {}).get("S3DataSource") or {}).get("S3Uri")}
                              for c in describe.get("InputDataConfig") or []]
    return json.loads(json.dumps(out, default=lambda o: iso(o) if isinstance(o, datetime) else str(o)))


def job_record(describe: dict[str, Any], region: str, usd_per_hour: float) -> dict[str, Any]:
    """The contract's `job` record (evidence records/<run_id>/job/<key>.json), straight from DescribeTrainingJob."""
    billable = int(describe.get("BillableTimeInSeconds") or 0)
    end = describe.get("TrainingEndTime")
    return {
        "job_name": describe["TrainingJobName"],
        "arn": describe["TrainingJobArn"],
        "job_kind": "training",
        "region": region,
        "status": describe["TrainingJobStatus"],
        "aws_start": iso(describe["CreationTime"]),
        "aws_end": iso(end),
        "aws_time": iso(end) if end else iso(describe["CreationTime"]),
        "aws_time_source": "sagemaker:TrainingEndTime" if end else "sagemaker:CreationTime",
        "instance_type": (describe.get("ResourceConfig") or {}).get("InstanceType"),
        "billable_s": billable,
        "est_cost_usd": round(billable / 3600.0 * usd_per_hour, 6),
    }
