"""Build the SageMaker CreateTrainingJob request for each silo job, and package the source directory.

    python -m silos.launch package --out /tmp/silos-source.tar.gz           # tarball of silos/ + tools/evlib.py + run_job.py
    python -m silos.launch data      --dry-run                              # print the request (JSON) and a cost estimate
    python -m silos.launch baseline  --method local --silo A --split disjoint --dry-run
    python -m silos.launch baseline  --method central --split disjoint --dry-run
    python -m silos.launch evaluate  --label federated --weights s3://... --dry-run
    python -m silos.launch compare   --kind beats --x RUN --y RUN --name federated_vs_B --dry-run

THIS MODULE NEVER CALLS AWS. It prints a request; the main session launches it:
    aws sagemaker create-training-job --region us-east-2 --cli-input-json file://request.json
(The generic L1 runner can replace the launch step; the entry points are plain `python -m silos.<job>` and need only the
environment this request sets.)

Inputs come from environment variables so that no account id, bucket or role is committed:
  ROLE_ARN, IMAGE_URI (a CPU PyTorch training image in the job's Region), SOURCE_S3_URI (the uploaded tarball),
  OUTPUT_S3 (s3://.../training-output/), EVIDENCE_S3, CODE_SHA, DATA_COPY_S3, SILO_BUCKET_A/B/C, SDK_S3_URI,
  RUN_ID (default: <unit>-<utc stamp>). Unset values are printed as <PLACEHOLDERS> and listed on stderr.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shlex
import sys
import tarfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REGION = os.environ.get("AWS_REGION", "us-east-2")
# Approximate on-demand list prices in USD per instance-hour, written from memory: VERIFY BEFORE LAUNCHING.
PRICE = {"ml.m5.xlarge": 0.23, "ml.m5.2xlarge": 0.461, "ml.m5.4xlarge": 0.922, "ml.c5.4xlarge": 0.816}
JOBS = {
    # name: (module, default instance, needs the SDK, estimated billable minutes, max runtime s)
    "data": ("silos.job_data", "ml.m5.4xlarge", True, 25, 3600),
    "baseline": ("silos.job_baselines", "ml.m5.2xlarge", False, 8, 3600),
    "evaluate": ("silos.job_evaluate", "ml.m5.xlarge", False, 4, 1800),
    "compare": ("silos.job_compare", "ml.m5.xlarge", False, 4, 1800),
}
ENV_PASS = ["EVIDENCE_S3", "CODE_SHA", "DATA_COPY_S3", "SILO_BUCKET_A", "SILO_BUCKET_B", "SILO_BUCKET_C", "SILO_BUCKET_REGION",
            "LAPTOP_COPY_S3", "SDK_S3_URI", "MODEL_OUT_S3"]
USES = {"data": ["EVIDENCE_S3", "CODE_SHA", "DATA_COPY_S3", "SILO_BUCKET_A", "SILO_BUCKET_B", "SILO_BUCKET_C", "SDK_S3_URI", "LAPTOP_COPY_S3", "SILO_BUCKET_REGION"],
        "baseline": ["EVIDENCE_S3", "CODE_SHA", "DATA_COPY_S3", "MODEL_OUT_S3"],
        "evaluate": ["EVIDENCE_S3", "CODE_SHA", "DATA_COPY_S3"], "compare": ["EVIDENCE_S3", "CODE_SHA"]}


def package(out: Path) -> dict:
    """Source directory for script mode: run_job.py, requirements.txt, silos/ (code, config, manifest) and tools/evlib.py."""
    keep = lambda p: not (p.suffix in (".pyc",) or "__pycache__" in p.parts or "tests" in p.parts or p.name.startswith("baselines_results"))
    with tarfile.open(out, "w:gz") as tar:
        for p in sorted((REPO / "silos").rglob("*")):
            if p.is_file() and keep(p.relative_to(REPO)):
                tar.add(p, arcname=str(p.relative_to(REPO)))
        tar.add(REPO / "tools" / "evlib.py", arcname="tools/evlib.py")
        tar.add(REPO / "silos" / "job_shim.py", arcname="run_job.py")
        req = b"pandas\npyyaml\nboto3\n"
        ti = tarfile.TarInfo("requirements.txt"); ti.size = len(req)
        tar.addfile(ti, io.BytesIO(req))
        init = b""
        ti = tarfile.TarInfo("tools/__init__.py"); ti.size = 0
        tar.addfile(ti, io.BytesIO(init))
    import hashlib

    return {"path": str(out), "bytes": out.stat().st_size, "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}


def build_request(job: str, args: str, unit_tag: str, instance: str | None = None) -> tuple[dict, list[str]]:
    module, default_inst, needs_sdk, minutes, max_s = JOBS[job]
    instance = instance or default_inst
    missing: list[str] = []

    def val(name: str, default: str | None = None) -> str:
        v = os.environ.get(name, "").strip() or default
        if not v:
            missing.append(name)
            return f"<{name}>"
        return v

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_id = os.environ.get("RUN_ID", "").strip() or f"{unit_tag}-{stamp}"
    env = {"JOB_MODULE": module, "JOB_ARGS": args, "RUN_ID": run_id, "AWS_REGION": REGION, "JOB_NEEDS_SDK": "1" if needs_sdk else "0"}
    for name in USES[job]:
        optional = name in ("LAPTOP_COPY_S3", "SILO_BUCKET_REGION", "MODEL_OUT_S3", "SDK_S3_URI") and not (name == "SDK_S3_URI" and needs_sdk)
        v = os.environ.get(name, "").strip()
        if v:
            env[name] = v
        elif not optional:
            env[name] = val(name)
    job_name = f"t13-{unit_tag}-{stamp}".replace("_", "-").replace(".", "-")[:63]
    req = {
        "TrainingJobName": job_name,
        "RoleArn": val("ROLE_ARN"),
        "AlgorithmSpecification": {"TrainingImage": val("IMAGE_URI"), "TrainingInputMode": "File"},
        "HyperParameters": {"sagemaker_program": '"run_job.py"', "sagemaker_submit_directory": json.dumps(val("SOURCE_S3_URI")),
                            "sagemaker_region": json.dumps(REGION)},
        "Environment": env,
        "ResourceConfig": {"InstanceType": instance, "InstanceCount": 1, "VolumeSizeInGB": 30},
        "StoppingCondition": {"MaxRuntimeInSeconds": max_s},
        "OutputDataConfig": {"S3OutputPath": val("OUTPUT_S3")},
        "Tags": [{"Key": "team", "Value": "team13"}, {"Key": "track", "Value": "track-2"}, {"Key": "run_id", "Value": run_id[:255]}],
    }
    return req, sorted(set(missing))


def estimate(job: str, instance: str) -> str:
    minutes = JOBS[job][3]
    p = PRICE.get(instance)
    return (f"estimate: {instance} ~${p}/h (approximate list price, verify) x ~{minutes} billable min incl. startup = "
            f"~${p * minutes / 60:.2f}; max runtime {JOBS[job][4] // 60} min" if p else f"estimate: no price for {instance}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pk = sub.add_parser("package"); pk.add_argument("--out", required=True, type=Path)
    for name in ("data", "baseline", "evaluate", "compare"):
        p = sub.add_parser(name)
        p.add_argument("--dry-run", action="store_true", help="accepted for clarity: this module only ever prints")
        p.add_argument("--instance-type", default=None)
        if name == "baseline":
            p.add_argument("--method", required=True, choices=["local", "central", "fedavg"]); p.add_argument("--silo"); p.add_argument("--split", default="disjoint")
            p.add_argument("--epochs", type=int, default=20); p.add_argument("--seed", type=int, default=0)
        if name == "evaluate":
            p.add_argument("--label", required=True, choices=["federated", "dp"]); p.add_argument("--split", default="disjoint")
            p.add_argument("--weights", required=True); p.add_argument("--session-id"); p.add_argument("--iteration-file"); p.add_argument("--epsilon", type=float)
        if name == "compare":
            p.add_argument("--kind", required=True, choices=["beats", "measure"]); p.add_argument("--x", required=True); p.add_argument("--y", required=True)
            p.add_argument("--name", required=True); p.add_argument("--split", default="disjoint")
    a = ap.parse_args(argv)
    if a.cmd == "package":
        print(json.dumps(package(a.out), indent=1))
        return 0
    if a.cmd == "data":
        args, tag = "", "t2-silo-data"
    elif a.cmd == "baseline":
        args = f"--method {a.method}" + (f" --silo {a.silo}" if a.silo else "") + f" --split {a.split} --epochs {a.epochs} --seed {a.seed}"
        tag = f"t2-{a.method}" + (f"-{a.silo.lower()}" if a.silo else "") + f"-{a.split}"
    elif a.cmd == "evaluate":
        args = f"--label {a.label} --split {a.split} --weights {shlex.quote(a.weights)}" + "".join(
            f" --{k.replace('_', '-')} {shlex.quote(str(v))}" for k, v in (("session_id", a.session_id), ("iteration_file", a.iteration_file), ("epsilon", a.epsilon)) if v is not None)
        tag = f"t2-fl-{a.label}-{a.split}"
    else:
        args = f"--kind {a.kind} --x {a.x} --y {a.y} --name {a.name} --split {a.split}"
        tag = f"t2-compare-{a.name}-{a.split}"
    req, missing = build_request(a.cmd, args, tag, a.instance_type)
    print(json.dumps(req, indent=1))
    print(estimate(a.cmd, req["ResourceConfig"]["InstanceType"]), file=sys.stderr)
    if missing:
        print(f"unset (printed as placeholders): {', '.join(missing)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
