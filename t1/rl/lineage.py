"""Lineage records for the Track 1 RL training run: one `job` record and one `model` record.

    python -m t1.rl.lineage --profile fyuz                 # dry run: prints both records, writes nothing
    python -m t1.rl.lineage --profile fyuz --write         # the main session: writes them to the evidence store

Reads (read-only): the training declaration (S3), DescribeTrainingJob (allow-listed fields only; Environment
is never read out), the model artifact and the packed project (GetObject, hashed; metrics.json and
run_metadata.json are read from inside the artifact).
Writes with --write (never overwrites):
  records/<training run>/job/<key>.json   exactly tools/sm/spec.job_record: name, ARN, status, SageMaker
                                          CreationTime/TrainingEndTime, instance, billable seconds, cost
  models/<model_id>/<job name>.json       the model: artifact URI and sha256, the packed project's sha256,
                                          FinalMetricDataList, metrics.json, SageMaker times, instance, billable s
The job record carries nothing beyond the contract's fields (JobRecord is closed); everything else goes in
the model record's metrics. Both carry aws_time = TrainingEndTime (sagemaker:TrainingEndTime).
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import tarfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "contracts"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools import evlib  # noqa: E402
from tools.sm import spec as smspec  # noqa: E402

TRAIN_RUN_ID = "t1.rl.train-20261103T023805Z-a09e"
TRAIN_JOB = "t1-rl-train-20261103T023805Z-a09e-20261103-023807"
RUN_METADATA_KEYS = ("git_commit", "config_hash", "use_case", "objective_version", "seed", "twin", "reward_plugin",
                     "started_at", "ended_at", "resumed_from")
MAX_BYTES = 512 * 1024 * 1024


class Refusal(RuntimeError):
    pass


def finite(o: Any) -> Any:
    """JSON without NaN/Infinity (they are not JSON): such numbers become null."""
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, dict):
        return {k: finite(v) for k, v in o.items()}
    if isinstance(o, list):
        return [finite(v) for v in o]
    return o


def get_object(s3, uri: str) -> bytes:
    b, _, k = uri[len("s3://"):].partition("/")
    r = s3.get_object(Bucket=b, Key=k)
    if int(r.get("ContentLength") or 0) > MAX_BYTES:
        raise Refusal(f"{uri} is larger than {MAX_BYTES} bytes")
    return r["Body"].read()


def tar_json(data: bytes, name: str) -> tuple[dict | None, list[str]]:
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as t:
        names = sorted(m.name.removeprefix("./") for m in t.getmembers() if m.isfile())
        for m in t.getmembers():
            if m.isfile() and m.name.removeprefix("./") == name:
                f = t.extractfile(m)
                try:
                    return (json.loads(f.read()) if f else None), names
                except ValueError:
                    return None, names
    return None, names


def tar_sha(data: bytes, name: str) -> str | None:
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as t:
        for m in t.getmembers():
            if m.isfile() and m.name.removeprefix("./") == name:
                f = t.extractfile(m)
                return evlib.sha256_bytes(f.read()) if f else None
    return None


def build(aws, run_id: str, job: str, region: str, bucket: str) -> tuple[dict, dict]:
    s3, sm = aws.client("s3"), aws.client("sagemaker", region)
    store = evlib.S3Store(f"s3://{bucket}/{smspec.EVIDENCE_PREFIX}", client=s3)
    decl = evlib.read_declared(run_id, store)
    if decl is None:
        raise Refusal(f"no declaration declared/{run_id}.json")
    d = sm.describe_training_job(TrainingJobName=job)
    status = d.get("TrainingJobStatus")
    if status in ("InProgress", "Stopping"):
        raise Refusal(f"training job {job} is {status}: lineage is written once it has ended")
    prefix = (decl.get("twin") or {}).get("config", {}).get("training", {}).get("job_name_prefix")
    if prefix and not job.startswith(prefix):
        raise Refusal(f"job {job} does not start with the declared job_name_prefix {prefix}")
    itype = (d.get("ResourceConfig") or {}).get("InstanceType")
    price = smspec.price_per_hour(itype)
    job_rec = smspec.job_record(d, region, price)
    job_rec.update(run_id=run_id, synthetic=False)
    model_uri = (d.get("ModelArtifacts") or {}).get("S3ModelArtifacts")
    if status != "Completed" or not model_uri:
        raise Refusal(f"training job {job} is {status} with artifact {model_uri!r}: no model to record")
    model = get_object(s3, model_uri)
    metrics_json, members = tar_json(model, "metrics.json")
    run_meta, _ = tar_json(model, "run_metadata.json")
    project_uri = smspec.unquote_hp((d.get("HyperParameters") or {}).get("sagemaker_submit_directory"))
    project = get_object(s3, project_uri) if isinstance(project_uri, str) and project_uri.startswith("s3://") else None
    final = {m["MetricName"]: {"value": m.get("Value"), "timestamp": smspec.iso(m.get("Timestamp"))}
             for m in d.get("FinalMetricDataList") or []}
    billable = int(d.get("BillableTimeInSeconds") or 0)
    end = smspec.iso(d.get("TrainingEndTime"))
    model_rec = {
        "run_id": run_id, "model_id": decl["model_id"], "version": job, "model_kind": "ppo",
        "artifact_uri": model_uri, "artifact_sha256": evlib.sha256_bytes(model), "status": "candidate",
        "trained_on": run_id, "synthetic": False,
        "aws_time": end, "aws_time_source": "sagemaker:TrainingEndTime",
        "metrics": finite({
            "final_metric_data_list": final,
            "final_metrics_source": ("sagemaker:FinalMetricDataList" if final else
                                     "FinalMetricDataList was empty; see metrics_json (the artifact's metrics.json)"),
            "metrics_json": metrics_json,
            "artifact_bytes": len(model), "artifact_members": members,
            "model_zip_sha256": tar_sha(model, "model.zip"),
            "project_uri": project_uri, "project_sha256": evlib.sha256_bytes(project) if project else None,
            "project_bytes": len(project) if project else None,
            "run_metadata": {k: run_meta[k] for k in RUN_METADATA_KEYS if isinstance(run_meta, dict) and k in run_meta},
            "training_job": {
                "name": job, "arn": d.get("TrainingJobArn"), "status": status,
                "secondary_status": d.get("SecondaryStatus"),
                "creation_time": smspec.iso(d.get("CreationTime")),
                "training_start_time": smspec.iso(d.get("TrainingStartTime")), "training_end_time": end,
                "instance_type": itype, "billable_s": billable,
                "training_time_s": d.get("TrainingTimeInSeconds"),
                "spot": bool(d.get("EnableManagedSpotTraining")),
                "est_cost_usd": round(billable / 3600.0 * price, 6),
                "cost_basis": f"BillableTimeInSeconds x ${price}/h ({smspec.PRICE_SOURCE})",
            },
            "declaration": {"run_id": run_id, "declared_at": decl.get("declared_at"), "code_sha": decl.get("code_sha"),
                            "test_set_sha256": decl.get("test_set_sha256")},
        }),
    }
    return job_rec, model_rec


def check_new(store, run_id: str, job_rec: dict, model_rec: dict) -> None:
    for r in evlib.read_records(run_id, "job", store):
        if r.get("job_name") == job_rec["job_name"]:
            raise Refusal(f"records/{run_id}/job/ already has a record for {job_rec['job_name']}")
    if evlib.read_model(model_rec["model_id"], model_rec["version"], store) is not None:
        raise Refusal(f"models/{model_rec['model_id']}/{model_rec['version']}.json already exists")


def preview(rec: dict, kind: str) -> dict:
    """The record as it would be stored (envelope added), validated."""
    out = {"schema_version": evlib.SCHEMA_VERSION, "kind": kind, "team": evlib.TEAM, "created_at": evlib.utc_now(),
           "aws_time": None, "aws_time_source": None, **rec}
    evlib.validate(out)
    return out


def main(argv: list[str] | None = None, aws=None, out=print) -> int:
    ap = argparse.ArgumentParser(prog="python -m t1.rl.lineage", description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-id", default=TRAIN_RUN_ID)
    ap.add_argument("--job", default=TRAIN_JOB)
    ap.add_argument("--write", action="store_true", help="write the two records (default: dry run)")
    ap.add_argument("--profile", default=os.environ.get("AWS_PROFILE"))
    ap.add_argument("--region", default=smspec.REGION)
    ap.add_argument("--bucket", default=smspec.BUCKET)
    a = ap.parse_args(argv)
    if aws is None:
        from tools.sm.run import Aws  # noqa: PLC0415

        aws = Aws(a.profile, a.region)
    store = evlib.S3Store(f"s3://{a.bucket}/{smspec.EVIDENCE_PREFIX}", client=aws.client("s3"), region=a.region)
    try:
        job_rec, model_rec = build(aws, a.run_id, a.job, a.region, a.bucket)
        check_new(store, a.run_id, job_rec, model_rec)
        pj, pm = preview(job_rec, "job"), preview(model_rec, "model")
    except (Refusal, evlib.EvidenceError) as e:
        out(f"refused: {e}")
        return 2
    out(f"job record -> {store.uri(f'records/{a.run_id}/job/')}<key>.json")
    out(json.dumps(pj, indent=2, sort_keys=True))
    model_rel = "models/{}/{}.json".format(model_rec["model_id"], model_rec["version"])
    out(f"model record -> {store.uri(model_rel)}")
    out(json.dumps(pm, indent=2, sort_keys=True))
    if not a.write:
        out("DRY RUN: nothing written (pass --write)")
        return 0
    uri = evlib.append_records(a.run_id, "job", [job_rec], store=store, layout="objects")[0]
    out(f"wrote {uri}")
    out(f"wrote {evlib.write_model(model_rec, store)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
