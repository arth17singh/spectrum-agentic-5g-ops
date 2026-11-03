"""t1/rl/eval_declaration.py against FakeAws (read-only), then the printed run.py dry run against the same fake."""

import hashlib
import io
import json
import shlex
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "tools" / "tests")]

from evfake import READ_ONLY_OPS, FakeAws  # noqa: E402
from t1.rl import eval_declaration as ed  # noqa: E402
from t1.rl import job_evaluate as je  # noqa: E402
from tools import evlib  # noqa: E402
from tools.sm import package, spec  # noqa: E402
from tools.sm import run as runner  # noqa: E402

BUCKET = "fixture-data-bucket"
TRAIN_RID = "t1.rl.train-20261103T023805Z-a09e"
TRAIN_JOB = "t1-rl-train-20261103T023805Z-a09e-20261103-023807"
URL_ARN = "arn:aws:secretsmanager:us-east-2:000000000000:secret:AmazonSageMaker-team13-twin-url-AbCdEf"
TOK_ARN = "arn:aws:secretsmanager:us-east-2:000000000000:secret:AmazonSageMaker-team13-twin-token-GhIjKl"
PROJECT = {"config/eval_scenarios.yaml": b"scenario_set_version: SYNTHETIC\n", "config/env.yaml": b"env: 1\n",
           "config/reward.yaml": b"reward: 1\n", "team/rewards.py": b"def r(c):\n    return 0\n"}


@pytest.fixture(autouse=True)
def _no_aws_env(monkeypatch):
    for k in ("EVIDENCE_S3", "EVIDENCE_DIR", "AWS_PROFILE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing-not-a-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing-not-a-secret")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setenv(spec.BUCKET_ENV, BUCKET)
    monkeypatch.setattr(package, "tree_dirty", lambda repo: [])


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def tgz(files: dict) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for n, b in files.items():
            ti = tarfile.TarInfo(n)
            ti.size = len(b)
            t.addfile(ti, io.BytesIO(b))
    return buf.getvalue()


def world(status="Completed", project=PROJECT) -> tuple[FakeAws, dict]:
    aws = FakeAws(now="2026-11-03T06:00:00Z")
    model, proj = tgz({"model.zip": b"SYNTHETIC zip", "metrics.json": b"{}"}), tgz(project)
    model_key = f"teams/team13/models/{TRAIN_JOB}/output/model.tar.gz"
    proj_key = f"teams/team13/code/{TRAIN_JOB}/source/sourcedir.tar.gz"
    aws.s3.put(BUCKET, model_key, model, "2026-11-03T05:00:00Z")
    aws.s3.put(BUCKET, proj_key, proj, "2026-11-03T02:38:10Z")
    train = {
        "run_id": TRAIN_RID, "unit": "t1.rl.train", "track": "track-1",
        "target": {"metric": "pass_rate", "op": ">=", "value": 1.0, "unit": "ratio"}, "n_items": 6,
        "test_set_sha256": sha(PROJECT["config/eval_scenarios.yaml"]), "model_id": "ppo-cco-reward-v1-100k",
        "grader": "g", "code_sha": "9b6a102d3bda3327cb3d9095ca403a57f957ae9b", "region": "us-east-2",
        "synthetic": False,
        "twin": {"kind": "maveric-hosted", "version": "1.2.9", "scenario": "nominal,congestion,interference,outage",
                 "seed": 42, "config": {
                     "items": {k: f"req {k}" for k in je.ITEM_IDS},
                     "env_config_sha256": sha(PROJECT["config/env.yaml"]),
                     "reward_config_sha256": sha(PROJECT["config/reward.yaml"]),
                     "team_rewards_sha256": sha(PROJECT["team/rewards.py"]),
                     "evaluation": {"episodes": 20, "where": "same run id"},
                     "training": {"timesteps": 100000}}},
    }
    rec = {**train, "schema_version": "1", "kind": "declared", "team": "team13",
           "created_at": "2026-11-03T02:38:05.318Z", "declared_at": "2026-11-03T02:38:05.318Z",
           "aws_time": None, "aws_time_source": None}
    aws.s3.put(BUCKET, f"evidence/declared/{TRAIN_RID}.json", json.dumps(rec).encode(), "2026-11-03T02:38:05Z")
    aws.jobs.append({"region": "us-east-2", "kind": "training", "name": TRAIN_JOB,
                     "TrainingJobArn": f"arn:aws:sagemaker:us-east-2:000000000000:training-job/{TRAIN_JOB.lower()}",
                     "TrainingJobStatus": status, "CreationTime": "2026-11-03T02:38:07Z",
                     "TrainingStartTime": "2026-11-03T02:39:00Z", "TrainingEndTime": "2026-11-03T05:00:00Z",
                     "ModelArtifacts": {"S3ModelArtifacts": f"s3://{BUCKET}/{model_key}"},
                     "HyperParameters": {"sagemaker_submit_directory": json.dumps(f"s3://{BUCKET}/{proj_key}")},
                     "Environment": {"RAN_RL_TWIN_URL": "https://must-not-leak.example"}})
    aws.s3.put(BUCKET, spec.SDK_KEY, b"not really a wheel", "2026-11-02T23:00:00Z")
    return aws, {"model": model, "project": proj, "model_uri": f"s3://{BUCKET}/{model_key}",
                 "project_uri": f"s3://{BUCKET}/{proj_key}"}


def fill(tmp_path, aws, *extra) -> tuple[int, str, dict | None]:
    out: list[str] = []
    path = tmp_path / "decl.json"
    rc = ed.main(["--out", str(path), "--run-id", "t1.rl.eval-20261103T060000Z-test", *extra], aws=aws,
                 out=out.append)
    return rc, "\n".join(out), (json.loads(path.read_text()) if path.exists() else None)


def test_fills_the_declaration_read_only(tmp_path):
    aws, w = world()
    rc, text, decl = fill(tmp_path, aws, "--url-secret-arn", URL_ARN, "--token-secret-arn", TOK_ARN)
    assert rc == 0, text
    assert aws.ops() <= READ_ONLY_OPS and aws.s3.writes == 0
    ev = decl["twin"]["config"]["eval"]
    assert ev["model_artifact_sha256"] == sha(w["model"]) and ev["project_sha256"] == sha(w["project"])
    assert ev["training_run_id"] == TRAIN_RID and ev["training_job_name"] == TRAIN_JOB
    assert "BEFORE training" in ev["target_first_declared"] and TRAIN_RID in ev["target_first_declared"]
    assert decl["unit"] == "t1.rl.eval" and decl["n_items"] == 6
    assert decl["target"] == {"metric": "pass_rate", "op": ">=", "value": 1.0, "unit": "ratio"}
    assert decl["test_set_sha256"] == sha(PROJECT["config/eval_scenarios.yaml"])
    assert sorted(decl["twin"]["config"]["items"]) == sorted(je.ITEM_IDS)
    assert "training" not in decl["twin"]["config"] and "runner" not in decl["twin"]["config"]
    assert "must-not-leak" not in text + json.dumps(decl)
    assert "ml.c5.2xlarge" in text and "$0.408" in text


def test_refuses_an_unfinished_training_job(tmp_path):
    aws, _ = world(status="InProgress")
    rc, text, decl = fill(tmp_path, aws)
    assert rc == 2 and "not Completed" in text and decl is None


def test_refuses_a_project_whose_test_set_changed(tmp_path):
    aws, _ = world(project={**PROJECT, "config/eval_scenarios.yaml": b"changed\n"})
    rc, text, decl = fill(tmp_path, aws)
    assert rc == 2 and "eval_scenarios.yaml" in text and decl is None


def test_printed_dry_run_is_a_valid_request_with_no_secret_in_environment(tmp_path):
    aws, w = world()
    rc, text, decl = fill(tmp_path, aws, "--url-secret-arn", URL_ARN, "--token-secret-arn", TOK_ARN)
    assert rc == 0
    dry = text.split("dry_run:\n", 1)[1].splitlines()[0].strip()
    argv = shlex.split(dry)
    assert argv[:3] == ["python", "-m", "tools.sm.run"] and "--dry-run" in argv
    out: list[str] = []
    assert runner.main(argv[3:], aws=aws, out=out.append) == 0, "\n".join(out)
    assert aws.s3.writes == 0 and not aws.created
    # the same plan, unredacted, to look inside Environment
    a = runner.parser().parse_args([x for x in argv[3:] if x != "--dry-run"])
    a.instance_type = a.instance_type or spec.DEFAULT_CPU
    p = runner.plan(a, aws)
    env = p["request"]["Environment"]
    assert not any(k.startswith("RAN_RL_") for k in env)
    assert json.loads(env["TEAM13_SECRET_ARNS"]) == {"RAN_RL_TWIN_URL": URL_ARN, "RAN_RL_ADAPTER_TOKEN": TOK_ARN}
    assert p["request"]["ResourceConfig"]["InstanceType"] == "ml.c5.2xlarge"
    assert p["request"]["StoppingCondition"]["MaxRuntimeInSeconds"] == 3600
    chans = {c["ChannelName"]: c["DataSource"]["S3DataSource"]["S3Uri"] for c in p["request"]["InputDataConfig"]}
    assert chans == {"sdk": f"s3://{BUCKET}/{spec.SDK_KEY}", "model": w["model_uri"], "project": w["project_uri"]}
    d = p["declaration"]
    assert d["unit"] == "t1.rl.eval" and d["twin"]["config"]["runner"]["instance_type"] == "ml.c5.2xlarge"
    assert "t1/rl/job_evaluate.py" in p["files"]
    entry = json.loads(json.loads(p["request"]["HyperParameters"]["team13_entry"]))
    assert entry[:2] == ["-m", "t1.rl.job_evaluate"] and "{channel:project}" in entry
    # the declaration passes the contract as run.py stores it
    evlib.validate({**d, "schema_version": "1", "kind": "declared", "team": "team13",
                    "created_at": evlib.utc_now(), "declared_at": evlib.utc_now(), "aws_time": None,
                    "aws_time_source": None})


def failed_world():
    import zipfile

    aws, w = world(status="Failed")
    j = aws.jobs[0]
    ck_prefix = f"s3://{BUCKET}/teams/team13/checkpoints/{TRAIN_JOB}"
    j.update(FailureReason="AlgorithmError: ExecuteUserScriptError: ExitCode 1", BillableTimeInSeconds=6034,
             CheckpointConfig={"S3Uri": ck_prefix, "LocalPath": "/opt/ml/checkpoints"})
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("data", "{}")
    ck = buf.getvalue()
    aws.s3.put(BUCKET, f"teams/team13/checkpoints/{TRAIN_JOB}/ppo_50000_steps.zip", ck, "2026-11-03T03:52:46Z")
    for n, v in enumerate((65536, 67584)):
        aws.events.append({"region": "us-east-2", "log_group": "/aws/sagemaker/TrainingJobs",
                           "log_stream": f"{TRAIN_JOB}/algo-1-1", "timestamp": f"2026-11-03T04:1{5 + n}:00Z",
                           "message": f"METRIC timesteps={v}", "event_id": f"t{n}"})
    aws.log_groups.setdefault("us-east-2", []).append("/aws/sagemaker/TrainingJobs")
    return aws, w, ck, f"{ck_prefix}/ppo_50000_steps.zip"


def test_failed_training_needs_a_checkpoint_and_records_the_failure(tmp_path):
    aws, w, ck, uri = failed_world()
    rc, text, decl = fill(tmp_path, aws)
    assert rc == 2 and "--checkpoint-uri" in text and decl is None
    rc, text, decl = fill(tmp_path, aws, "--checkpoint-uri", uri, "--model-id", "ppo-cco-reward-v1-ckpt50k",
                          "--url-secret-arn", URL_ARN,
                          "--token-secret-arn", TOK_ARN)
    assert rc == 0, text
    assert aws.ops() <= READ_ONLY_OPS and aws.s3.writes == 0
    ev = decl["twin"]["config"]["eval"]
    assert ev["checkpoint_sha256"] == sha(ck) and ev["checkpoint_steps"] == 50000 and ev["checkpoint_uri"] == uri
    assert "model_artifact_sha256" not in ev
    assert decl["model_id"] == "ppo-cco-reward-v1-ckpt50k" and ev["training_model_id"] == "ppo-cco-reward-v1-100k"
    tf = ev["training_failure"]
    assert tf["status"] == "Failed" and tf["billable_s"] == 6034 and "ExitCode 1" in tf["failure_reason"]
    assert tf["last_metric_timesteps"]["value"] == 67584
    assert "NOT met" in tf["statement"] and "67584" in tf["statement"] and "Connection reset" in tf["cause_observed"]
    assert "BEFORE training" in ev["target_first_declared"]
    argv = shlex.split(text.split("dry_run:\n", 1)[1].splitlines()[0].strip())
    assert f"model={uri}" in argv
    a = runner.parser().parse_args([x for x in argv[3:] if x != "--dry-run"])
    p = runner.plan(a, aws)
    chans = {c["ChannelName"]: c["DataSource"]["S3DataSource"]["S3Uri"] for c in p["request"]["InputDataConfig"]}
    assert chans["model"] == uri and chans["project"] == w["project_uri"]


def test_checkpoint_outside_the_jobs_checkpoint_prefix_is_refused(tmp_path):
    aws, _, _, _ = failed_world()
    rc, text, _ = fill(tmp_path, aws, "--checkpoint-uri", f"s3://{BUCKET}/elsewhere/ppo_50000_steps.zip")
    assert rc == 2 and "CheckpointConfig" in text


def test_placeholder_secret_arns_are_refused_by_the_runner(tmp_path):
    aws, _ = world()
    rc, text, _ = fill(tmp_path, aws)
    assert rc == 0 and "<ARN of AmazonSageMaker-team13-twin-url>" in text
    argv = shlex.split(text.split("dry_run:\n", 1)[1].splitlines()[0].strip())
    out: list[str] = []
    assert runner.main(argv[3:], aws=aws, out=out.append) == 2
    assert "not a Secrets Manager secret ARN" in "\n".join(out)
