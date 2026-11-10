"""Serve team13's tuned NLM (t1.sft.train_lora's merged/ model) on a SageMaker real-time endpoint.

    python -m t1.sft.serve.deploy plan   --train-run-id R [--profile fyuz]   # read-only: prints every request
    python -m t1.sft.serve.deploy apply  --train-run-id R [--profile fyuz]   # S3 copy + CreateModel/Config/Endpoint
    python -m t1.sft.serve.deploy smoke  [--profile fyuz]                    # one InvokeEndpoint, W3's exact body
    python -m t1.sft.serve.deploy delete [--profile fyuz] [--apply]          # endpoint, config and model

Container: the AWS PyTorch inference DLC (2.6.0, GPU, py312, CUDA 12.4) with t1/sft/serve/inference.py; the
container pip-installs code/requirements.txt (transformers 4.57.6) at start. Our handler renders the model's
own chat template with enable_thinking=False (byte-identical to the eval harness's prompt), accepts W3's
nlm___ask body as sent, and answers in the OpenAI shape nlm.parse_generation reads.
Model data: uncompressed S3 prefix serving/t1-otel-1.7b-lora/<train run>/ = a server-side copy of the run's
merged/ (the training output stays untouched) + code/inference.py + code/requirements.txt.
Instance: ml.g6.xlarge (1x L4 24 GB, bf16; quota "ml.g6.xlarge for endpoint usage" = 1, read 2 Nov), one
instance. Endpoint name team13-otel-nlm: the tools Lambda role /bb26/team13-ran-tools already allows
sagemaker:InvokeEndpoint on exactly that name (t3/deploy/iam_cloudshell.sh, TunedNlm).
Recording: every call prints `team13-nlm-call {hashes, latency}` to /aws/sagemaker/Endpoints/team13-otel-nlm,
and data capture stores every request and response body under s3://<bucket>/evidence/nlm-capture/<endpoint>/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REGION = "us-east-2"
ENDPOINT = "team13-otel-nlm"
INSTANCE = "ml.g6.xlarge"
HOSTING_PRICE = {"ml.g6.xlarge": 1.1267, "ml.g5.xlarge": 1.4084}  # USD/h, Price List API USE2-Host, eff. 2026-11-01
IMAGE = "763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-inference:2.6.0-gpu-py312-cu124-ubuntu22.04-sagemaker-v1.84"
ROLE = "bb26-SageMakerExecution"
LORA_PREFIX = "models/t1-otel-1.7b-lora/"
SERVING_PREFIX = "serving/t1-otel-1.7b-lora/"
CODE_FILES = ("inference.py", "requirements.txt")
MERGED_REQUIRED = ("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                   "generation_config.json")
SMOKE_SYSTEM = "You are a radio access network engineer. Answer in at most 120 words."  # = t3/tools/nlm.py SYSTEM
SMOKE_QUESTION = "In two sentences: what does lowering a cell's antenna tilt usually do to its coverage?"


def session(profile: str | None):
    import boto3

    return boto3.Session(profile_name=profile, region_name=REGION)


def names(train_run_id: str, code_sha: str) -> dict[str, str]:
    h = hashlib.sha256(f"{train_run_id}|{code_sha}".encode()).hexdigest()[:8]
    return {"model": f"{ENDPOINT}-{h}", "config": f"{ENDPOINT}-{h}-cfg", "endpoint": ENDPOINT}


def code_sha256() -> str:
    h = hashlib.sha256()
    for f in CODE_FILES:
        h.update(f.encode() + b"\0" + (HERE / f).read_bytes() + b"\0")
    return h.hexdigest()


def requests(*, account: str, bucket: str, train_run_id: str, sums_sha256: str, instance: str) -> dict[str, Any]:
    n = names(train_run_id, code_sha256())
    serving = f"s3://{bucket}/{SERVING_PREFIX}{train_run_id}/"
    tags = [{"Key": "team", "Value": "team13"}, {"Key": "team13:train_run_id", "Value": train_run_id},
            {"Key": "team13:purpose", "Value": "nlm___ask rehearsal and demo window only"}]
    model = {
        "ModelName": n["model"],
        "ExecutionRoleArn": f"arn:aws:iam::{account}:role/{ROLE}",
        "PrimaryContainer": {
            "Image": IMAGE.format(region=REGION), "Mode": "SingleModel",
            "ModelDataSource": {"S3DataSource": {"S3Uri": serving, "S3DataType": "S3Prefix", "CompressionType": "None"}},
            "Environment": {"SAGEMAKER_PROGRAM": "inference.py", "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
                            "SAGEMAKER_REGION": REGION, "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
                            "SAGEMAKER_MODEL_SERVER_WORKERS": "1", "SAGEMAKER_MODEL_SERVER_TIMEOUT": "180",
                            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
                            "NLM_TRAIN_RUN_ID": train_run_id, "NLM_TRAIN_SHA256SUMS_SHA256": sums_sha256,
                            "NLM_MODEL_LABEL": "team13 tuned OTel-LLM-1.7B-IT (LoRA merged)",
                            "NLM_MAX_NEW_TOKENS_CAP": "1024"}},
        "Tags": tags,
    }
    config = {
        "EndpointConfigName": n["config"],
        "ProductionVariants": [{"VariantName": "AllTraffic", "ModelName": n["model"], "InstanceType": instance,
                                "InitialInstanceCount": 1, "ContainerStartupHealthCheckTimeoutInSeconds": 1200,
                                "ModelDataDownloadTimeoutInSeconds": 1200}],
        "DataCaptureConfig": {"EnableCapture": True, "InitialSamplingPercentage": 100,
                              "DestinationS3Uri": f"s3://{bucket}/evidence/nlm-capture/{ENDPOINT}/",
                              "CaptureOptions": [{"CaptureMode": "Input"}, {"CaptureMode": "Output"}],
                              "CaptureContentTypeHeader": {"JsonContentTypes": ["application/json"]}},
        "Tags": tags,
    }
    endpoint = {"EndpointName": n["endpoint"], "EndpointConfigName": n["config"], "Tags": tags}
    return {"names": n, "serving": serving, "model": model, "config": config, "endpoint": endpoint}


def resolve(profile: str | None, bucket: str | None) -> tuple[Any, str, str]:
    s = session(profile)
    account = s.client("sts").get_caller_identity()["Account"]
    bucket = bucket or os.environ.get("TEAM13_DATA_BUCKET") or f"bb26-p-team13-data-{account}-use2"
    return s, account, bucket


def merged_listing(s3, bucket: str, train_run_id: str) -> tuple[dict[str, int], str]:
    """{name: size} under merged/ (read-only), and sha256 of the run's SHA256SUMS; every required file listed."""
    prefix = f"{LORA_PREFIX}{train_run_id}/"
    sums = s3.get_object(Bucket=bucket, Key=prefix + "SHA256SUMS")["Body"].read()
    listed = {line.split("  ", 1)[1] for line in sums.decode().splitlines() if line.strip()}
    files = {}
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix + "merged/"):
        for o in page.get("Contents", []):
            files[o["Key"][len(prefix + "merged/"):]] = o["Size"]
    missing = [f for f in MERGED_REQUIRED if f not in files or f"merged/{f}" not in listed]
    if missing:
        raise SystemExit(f"merged/ of {train_run_id} lacks {missing} (or SHA256SUMS does not list them)")
    return files, hashlib.sha256(sums).hexdigest()


def cmd_plan(a, apply: bool) -> int:
    s, account, bucket = resolve(a.profile, a.bucket)
    s3 = s.client("s3")
    files, sums_sha = merged_listing(s3, bucket, a.train_run_id)
    r = requests(account=account, bucket=bucket, train_run_id=a.train_run_id, sums_sha256=sums_sha,
                 instance=a.instance_type)
    src = f"{LORA_PREFIX}{a.train_run_id}/merged/"
    dst = r["serving"][len(f"s3://{bucket}/"):]
    print(f"merged/ of {a.train_run_id}: {len(files)} files, {sum(files.values())} bytes; SHA256SUMS sha256 {sums_sha}")
    print(f"serving prefix {r['serving']}: copy {src}* server-side, then code/{', code/'.join(CODE_FILES)} "
          f"(code sha256 {code_sha256()})")
    for k in ("model", "config", "endpoint"):
        print(f"--- Create{'Model' if k == 'model' else 'EndpointConfig' if k == 'config' else 'Endpoint'}:")
        print(json.dumps(r[k], indent=2))
    price = HOSTING_PRICE.get(a.instance_type)
    print(f"instance {a.instance_type} x1: ${price}/h while the endpoint exists" if price else
          f"instance {a.instance_type}: no hosting price on file")
    if not apply:
        print("PLAN ONLY: nothing was written. Run `apply` to create it.")
        return 0
    from botocore.exceptions import ClientError

    sm = s.client("sagemaker")
    try:
        sm.describe_endpoint(EndpointName=ENDPOINT)
        raise SystemExit(f"endpoint {ENDPOINT} already exists: delete it first (or update it by hand)")
    except ClientError as e:
        if "Could not find endpoint" not in str(e):
            raise
    existing = {o["Key"][len(dst):] for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=dst)
                for o in page.get("Contents", [])}
    for name in sorted(files):
        if name not in existing:
            s3.copy({"Bucket": bucket, "Key": src + name}, bucket, dst + name)
            print(f"copied {name}")
    for f in CODE_FILES:
        key = f"{dst}code/{f}"
        if f"code/{f}" in existing:
            if s3.get_object(Bucket=bucket, Key=key)["Body"].read() != (HERE / f).read_bytes():
                raise SystemExit(f"{key} exists with other content: refusing to overwrite")
            continue
        s3.put_object(Bucket=bucket, Key=key, Body=(HERE / f).read_bytes(), IfNoneMatch="*")
        print(f"uploaded code/{f}")
    sm.create_model(**r["model"])
    sm.create_endpoint_config(**r["config"])
    sm.create_endpoint(**r["endpoint"])
    print(f"CreateEndpoint {ENDPOINT} submitted; wait with: aws sagemaker wait endpoint-in-service "
          f"--endpoint-name {ENDPOINT} --region {REGION}" + (f" --profile {a.profile}" if a.profile else ""))
    return 0


def smoke_body() -> dict:
    """Exactly what t3/tools/nlm.py sends (payload `messages`, its default)."""
    return {"messages": [{"role": "system", "content": SMOKE_SYSTEM}, {"role": "user", "content": SMOKE_QUESTION}],
            "max_tokens": 200, "temperature": 0.2}


def cmd_smoke(a) -> int:
    s = session(a.profile)
    rt = s.client("sagemaker-runtime")
    t0 = time.perf_counter()
    resp = rt.invoke_endpoint(EndpointName=ENDPOINT, ContentType="application/json", Accept="application/json",
                              Body=json.dumps(smoke_body()).encode())
    raw = resp["Body"].read()
    wall = (time.perf_counter() - t0) * 1000
    data = json.loads(raw)
    try:
        sys.path.insert(0, str(HERE.parents[2]))
        from t3.tools.nlm import parse_generation  # W3's own parser: the wiring check

        answer = parse_generation(data)
    except ImportError:
        answer = data["choices"][0]["message"]["content"]
    print(json.dumps({"wall_ms": round(wall, 1), "server_latency_ms": data.get("team13", {}).get("latency_ms"),
                      "usage": data.get("usage"), "finish_reason": data["choices"][0].get("finish_reason"),
                      "response_sha256": hashlib.sha256(raw).hexdigest(), "answer": answer}, indent=2))
    return 0


def cmd_delete(a) -> int:
    sm = session(a.profile).client("sagemaker")
    d = sm.describe_endpoint(EndpointName=ENDPOINT)
    cfg = sm.describe_endpoint_config(EndpointConfigName=d["EndpointConfigName"])
    models = [v["ModelName"] for v in cfg["ProductionVariants"]]
    print(f"endpoint {ENDPOINT} ({d['EndpointStatus']}), config {d['EndpointConfigName']}, models {models}")
    if not a.apply:
        print("DRY RUN: add --apply to delete them (billing stops when the endpoint is deleted)")
        return 0
    sm.delete_endpoint(EndpointName=ENDPOINT)
    sm.delete_endpoint_config(EndpointConfigName=d["EndpointConfigName"])
    for m in models:
        sm.delete_model(ModelName=m)
    print("deleted endpoint, endpoint config and model(s); the serving/ prefix and data capture stay in S3")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m t1.sft.serve.deploy", description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=["plan", "apply", "smoke", "delete"])
    ap.add_argument("--train-run-id", help="the t1.sft.train_lora run whose merged/ to serve")
    ap.add_argument("--instance-type", default=INSTANCE)
    ap.add_argument("--bucket", help="default $TEAM13_DATA_BUCKET, else from the STS account")
    ap.add_argument("--profile", default=os.environ.get("AWS_PROFILE"))
    ap.add_argument("--apply", action="store_true", help="delete: really delete")
    a = ap.parse_args(argv)
    os.environ.setdefault("AWS_SDK_UA_APP_ID", "AWSSkill-SageMaker")
    if a.what in ("plan", "apply"):
        if not a.train_run_id:
            ap.error("--train-run-id is required")
        return cmd_plan(a, apply=a.what == "apply")
    return cmd_smoke(a) if a.what == "smoke" else cmd_delete(a)


if __name__ == "__main__":
    sys.exit(main())
