"""t1/rl/lineage.py and t1/rl/reward_check.py against FakeAws: dry run writes nothing, --write writes once."""

import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "tools" / "tests")]

from evfake import READ_ONLY_OPS, FakeAws  # noqa: E402
from t1.rl import lineage, reward_check  # noqa: E402
from tools import evlib  # noqa: E402
from tools.sm import spec  # noqa: E402

BUCKET = spec.BUCKET
EV = f"s3://{BUCKET}/evidence/"
HOST = "twin-synthetic.example.internal"


@pytest.fixture(autouse=True)
def _no_aws_env(monkeypatch):
    for k in ("EVIDENCE_S3", "EVIDENCE_DIR", "AWS_PROFILE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing-not-a-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing-not-a-secret")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")


def tgz(files: dict) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for n, b in files.items():
            ti = tarfile.TarInfo(n)
            ti.size = len(b)
            t.addfile(ti, io.BytesIO(b))
    return buf.getvalue()


def put_decl(aws, rid, **fields):
    rec = {"schema_version": "1", "kind": "declared", "team": "team13", "run_id": rid,
           "created_at": "2026-11-03T02:00:00.000Z", "declared_at": "2026-11-03T02:00:00.000Z",
           "aws_time": None, "aws_time_source": None, "synthetic": False, "track": "track-1",
           "target": {"metric": "pass_rate", "op": ">=", "value": 1.0, "unit": "ratio"},
           "test_set_sha256": "d" * 64, "code_sha": "c" * 40, "region": "us-east-2", **fields}
    aws.s3.put(BUCKET, f"evidence/declared/{rid}.json", json.dumps(rec).encode(), "2026-11-03T02:00:00Z")


def job(name, status="Completed", **kw):
    j = {"region": "us-east-2", "kind": "training", "name": name,
         "TrainingJobArn": f"arn:aws:sagemaker:us-east-2:000000000000:training-job/{name.lower()}",
         "TrainingJobStatus": status, "SecondaryStatus": status, "CreationTime": "2026-11-03T02:00:07Z",
         "TrainingStartTime": "2026-11-03T02:01:00Z", "TrainingEndTime": "2026-11-03T04:30:00Z",
         "LastModifiedTime": "2026-11-03T04:30:05Z", "BillableTimeInSeconds": 8900,
         "TrainingTimeInSeconds": 8900, "ResourceConfig": {"InstanceType": "ml.g6.xlarge"},
         "Environment": {"RAN_RL_TWIN_URL": f"https://{HOST}"}}
    j.update(kw)
    return j


# ---------------------------------------------------------------------------- lineage

def lineage_world(status="Completed", final=None):
    aws = FakeAws(now="2026-11-03T05:00:00Z")
    rid, name = lineage.TRAIN_RUN_ID, lineage.TRAIN_JOB
    put_decl(aws, rid, unit="t1.rl.train", n_items=6, model_id="ppo-synthetic", grader="g",
             twin={"kind": "maveric-hosted", "version": "1.2.9", "scenario": "s", "seed": 42,
                   "config": {"training": {"job_name_prefix": "t1-rl-train-20261103T023805Z-a09e"}}})
    model = tgz({"model.zip": b"SYNTHETIC", "metrics.json": json.dumps({"coverage_mean": 0.9, "x": float("nan")}).encode(),
                 "run_metadata.json": json.dumps({"config_hash": "h", "twin": {"name": "HttpDigitalTwin"},
                                                  "secret_field": "never copied"}).encode()})
    proj = tgz({"config/env.yaml": b"x"})
    mk, pk = f"teams/team13/models/{name}/output/model.tar.gz", f"teams/team13/code/{name}/source/sourcedir.tar.gz"
    aws.s3.put(BUCKET, mk, model, "2026-11-03T04:29:00Z")
    aws.s3.put(BUCKET, pk, proj, "2026-11-03T02:00:05Z")
    aws.jobs.append(job(name, status, ModelArtifacts={"S3ModelArtifacts": f"s3://{BUCKET}/{mk}"},
                        HyperParameters={"sagemaker_submit_directory": json.dumps(f"s3://{BUCKET}/{pk}")},
                        FinalMetricDataList=final or []))
    return aws, model, proj


def test_lineage_dry_run_prints_valid_records_and_writes_nothing():
    aws, model, proj = lineage_world()
    out: list[str] = []
    assert lineage.main([], aws=aws, out=out.append) == 0, "\n".join(out)
    text = "\n".join(out)
    assert aws.s3.writes == 0 and aws.ops() <= READ_ONLY_OPS
    assert hashlib.sha256(model).hexdigest() in text and hashlib.sha256(proj).hexdigest() in text
    assert HOST not in text and "never copied" not in text and "DRY RUN" in text
    assert '"billable_s": 8900' in text and "ml.g6.xlarge" in text
    assert "FinalMetricDataList was empty" in text


def test_lineage_write_once_and_matches_describe():
    from datetime import datetime, timezone

    aws, model, _ = lineage_world(final=[{"MetricName": "coverage_mean", "Value": 0.9,
                                          "Timestamp": datetime(2026, 11, 3, 4, 29, tzinfo=timezone.utc)}])
    out: list[str] = []
    assert lineage.main(["--write"], aws=aws, out=out.append) == 0, "\n".join(out)
    store = evlib.S3Store(EV, client=aws.s3)
    jobs = evlib.read_records(lineage.TRAIN_RUN_ID, "job", store)
    assert len(jobs) == 1 and jobs[0]["billable_s"] == 8900 and jobs[0]["status"] == "Completed"
    assert jobs[0]["aws_start"] == "2026-11-03T02:00:07.000Z" and jobs[0]["aws_end"] == "2026-11-03T04:30:00.000Z"
    m = evlib.read_model("ppo-synthetic", lineage.TRAIN_JOB, store)
    assert m["artifact_sha256"] == hashlib.sha256(model).hexdigest() and m["trained_on"] == lineage.TRAIN_RUN_ID
    assert m["metrics"]["final_metric_data_list"]["coverage_mean"]["value"] == 0.9
    assert m["metrics"]["metrics_json"]["x"] is None  # NaN is not JSON
    assert m["aws_time_source"] == "sagemaker:TrainingEndTime"
    out.clear()
    assert lineage.main(["--write"], aws=aws, out=out.append) == 2 and "already" in "\n".join(out)


def test_lineage_refuses_a_running_job():
    aws, _, _ = lineage_world(status="InProgress")
    out: list[str] = []
    assert lineage.main(["--write"], aws=aws, out=out.append) == 2 and aws.s3.writes == 0


def test_lineage_failed_job_with_checkpoint():
    import zipfile

    aws, _, _ = lineage_world(status="Failed")
    j = aws.jobs[0]
    ckp = f"s3://{BUCKET}/teams/team13/checkpoints/{lineage.TRAIN_JOB}"
    j.update(FailureReason="AlgorithmError: ExitCode 1", BillableTimeInSeconds=6034,
             CheckpointConfig={"S3Uri": ckp})
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("data", "{}")
    ck = buf.getvalue()
    aws.s3.put(BUCKET, f"teams/team13/checkpoints/{lineage.TRAIN_JOB}/ppo_50000_steps.zip", ck, "2026-11-03T03:52:46Z")
    aws.events.append({"region": "us-east-2", "log_group": "/aws/sagemaker/TrainingJobs",
                       "log_stream": f"{lineage.TRAIN_JOB}/algo-1", "timestamp": "2026-11-03T04:17:32Z",
                       "message": "METRIC timesteps=67584", "event_id": "t1"})
    aws.log_groups.setdefault("us-east-2", []).append("/aws/sagemaker/TrainingJobs")
    out: list[str] = []
    assert lineage.main(["--write"], aws=aws, out=out.append) == 2  # no model without a checkpoint
    assert aws.s3.writes == 0
    uri = f"{ckp}/ppo_50000_steps.zip"
    assert lineage.main(["--write", "--checkpoint-uri", uri, "--model-id", "ppo-ckpt50k"], aws=aws,
                        out=out.append) == 0, "\n".join(out)
    store = evlib.S3Store(EV, client=aws.s3)
    jobs = evlib.read_records(lineage.TRAIN_RUN_ID, "job", store)
    assert jobs[0]["status"] == "Failed" and jobs[0]["billable_s"] == 6034
    m = evlib.read_model("ppo-ckpt50k", f"{lineage.TRAIN_JOB}-ckpt-50000", store)
    assert m["artifact_uri"] == uri and m["artifact_sha256"] == hashlib.sha256(ck).hexdigest()
    assert m["metrics"]["steps"] == 50000 and m["metrics"]["checkpoint"]["source_checkpoint_uri"] == uri
    assert m["metrics"]["training_job"]["failure_reason"] == "AlgorithmError: ExitCode 1"
    assert m["metrics"]["last_metric_timesteps"]["value"] == 67584
    assert m["aws_time"] == "2026-11-03T03:52:46.000Z"


# ---------------------------------------------------------------------------- reward check

def reward_world(values=(0.70, 0.85, 0.891), status="Completed", dup=True):
    aws = FakeAws(now="2026-11-03T03:00:00Z")
    rid, name = reward_check.RUN_ID, reward_check.JOB
    put_decl(aws, rid, unit="t1.rl.reward-check", n_items=2, model_id="ppo-reward-check-18k",
             grader="SageMaker job status + the job's own final METRIC coverage_mean",
             twin={"kind": "maveric-hosted", "version": "1.2.9", "scenario": "s", "seed": 42, "config": {"items": {
                 "sagemaker-job-completed": "passed when DescribeTrainingJob reports Completed",
                 "training-coverage-not-collapsed": "passed when the job's last METRIC coverage_mean >= 0.82"}}})
    aws.jobs.append(job(name, status, ResourceConfig={"InstanceType": "ml.c5.2xlarge"},
                        TrainingEndTime="2026-11-03T02:32:52Z", BillableTimeInSeconds=370))
    stream = f"{name}/algo-1-1793671415"
    for n, v in enumerate(values):
        ts = f"2026-11-03T02:{10 + n:02d}:00Z"
        aws.events.append({"region": "us-east-2", "log_group": reward_check.LOG_GROUP, "log_stream": stream,
                           "timestamp": ts, "ingestion_time": f"2026-11-03T02:{10 + n:02d}:03Z",
                           "message": f"METRIC coverage_mean={v}", "event_id": f"ev{n}"})
    if dup and values:  # the same event returned twice
        aws.events.append(dict(aws.events[-1]))
    aws.events.append({"region": "us-east-2", "log_group": reward_check.LOG_GROUP, "log_stream": stream,
                       "timestamp": "2026-11-03T02:05:00Z", "message": f"HTTP Request: POST https://{HOST}/reset",
                       "event_id": "evh"})
    aws.log_groups.setdefault("us-east-2", []).append(reward_check.LOG_GROUP)
    return aws


def test_reward_check_dry_run_then_write_once():
    aws = reward_world()
    out: list[str] = []
    assert reward_check.main([], aws=aws, out=out.append) == 0, "\n".join(out)
    assert aws.s3.writes == 0 and aws.ops() <= READ_ONLY_OPS
    assert HOST not in "\n".join(out)
    assert reward_check.main(["--write"], aws=aws, out=out.append) == 0
    store = evlib.S3Store(EV, client=aws.s3)
    items = {r["item_id"]: r for r in evlib.read_records(reward_check.RUN_ID, "item", store)}
    cov = items["training-coverage-not-collapsed"]
    assert cov["passed"] is True and cov["extra"]["value"] == 0.891 and cov["extra"]["event_id"] == "ev2"
    assert cov["extra"]["n_metric_events"] == 3  # de-duplicated
    assert cov["aws_time"] == "2026-11-03T02:12:00.000Z" and cov["extra"]["ingestion_time"] == "2026-11-03T02:12:03.000Z"
    assert cov["aws_time_source"] == "cloudwatch:/aws/sagemaker/TrainingJobs"
    assert items["sagemaker-job-completed"]["passed"] is True
    assert items["sagemaker-job-completed"]["aws_time"] == "2026-11-03T02:32:52.000Z"
    res = evlib.read_result(reward_check.RUN_ID, store)
    assert res["passed"] is True and res["value"] == 1.0 and res["n_items"] == 2
    out.clear()
    assert reward_check.main(["--write"], aws=aws, out=out.append) == 2 and "already" in "\n".join(out)


def test_reward_check_low_coverage_misses():
    aws = reward_world(values=(0.70, 0.81))
    out: list[str] = []
    assert reward_check.main(["--write"], aws=aws, out=out.append) == 0
    res = evlib.read_result(reward_check.RUN_ID, evlib.S3Store(EV, client=aws.s3))
    assert res["passed"] is False and res["value"] == 0.5


def test_reward_check_failed_job_and_no_metric_line():
    aws = reward_world(values=(), status="Failed")
    out: list[str] = []
    assert reward_check.main([], aws=aws, out=out.append) == 0
    text = "\n".join(out)
    assert "item sagemaker-job-completed: FAIL" in text and "item training-coverage-not-collapsed: FAIL" in text
    assert "no METRIC coverage_mean line" in text
