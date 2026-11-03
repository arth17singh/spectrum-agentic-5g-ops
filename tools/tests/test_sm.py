"""tools/sm (the SageMaker job runner) and the check_unit rules it is built for. Offline: FakeAws only.

The end-to-end scenario launches a real plan (archive of this repo's HEAD, real declaration, real
CreateTrainingJob request) against FakeAws; the fake "job" is the waiter callback, which does what
tools/sm/entry.py does in SageMaker (module writes local evidence -> collect -> attest -> log lines).
"""

import hashlib
import io
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import check_unit
from evfake import READ_ONLY_OPS, FakeAws
from tools import evlib
from tools.sm import entry, package, spec
from tools.sm import run as runner

REPO = Path(__file__).resolve().parents[2]
HEAD = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
FAKE_ACCOUNT = "000000000000"  # the fake STS account (evfake); never a real id in this repo
BUCKET = f"bb26-p-team13-data-{FAKE_ACCOUNT}-use2"
EVIDENCE = f"s3://{BUCKET}/evidence/"
T0 = "2026-11-03T02:00:00.250Z"


@pytest.fixture(autouse=True)
def _no_bucket_override(monkeypatch):
    monkeypatch.delenv(spec.BUCKET_ENV, raising=False)


def base_args(run_id: str, *extra: str) -> list[str]:
    return ["--run-id", run_id, "--account", "000000000000", "--unit", "t.sm", "--track", "track-3",
            "--metric", "pass_rate", "--op", ">=", "--value", "0.5", "--target-unit", "ratio",
            "--test-set", "tools/sm/spec.py", "--n-items", "3", "--model-id", "m-under-test", "--grader", "g-v1",
            "--entry", "-m tools.sm.spec --run-id {run_id} --evidence-dir {evidence_dir}",
            "--paths", "tools/sm", "--max-run", "900", "--no-check", *extra]


def items(run_id: str) -> list[dict]:
    return [{"item_id": f"i{n}", "input_sha256": hashlib.sha256(f"in{n}".encode()).hexdigest(),
             "output_sha256": hashlib.sha256(f"out{n}".encode()).hexdigest(), "passed": n != 2, "score": float(n != 2),
             "grader": "g-v1", "backend": "fake"} for n in range(3)]


class FakeJob:
    """The waiter callback: plays the job (what tools/sm/entry.py does) on the fake clock."""

    def __init__(self, aws: FakeAws, tmp: Path, run_id: str, status: str = "Completed", attest: bool = True,
                 after=None):
        self.aws, self.tmp, self.run_id, self.status, self.attest, self.after = aws, tmp, run_id, status, attest, after
        self.lines: list[str] = []

    def __call__(self, job: str, region: str, profile) -> int:
        aws, rid = self.aws, self.run_id
        store = evlib.S3Store(EVIDENCE, client=aws.s3)
        aws.advance(95)  # Starting + Downloading
        decl = evlib.read_declared(rid, store)
        local = self.tmp / "evidence-local"
        ls = evlib.LocalStore(local)
        its = items(rid)
        evlib.append_records(rid, "item", its, store=ls)
        evlib.write_result(evlib.make_result(decl, its, "tools/sm/spec.py", decl["code_sha"]), ls)
        say, entry.say = entry.say, self.lines.append
        try:
            aws.advance(5)
            entry.collect(rid, local, store, decl)
            aws.advance(1)
            if self.attest:
                entry.attest(rid, store)
        finally:
            entry.say = say
        aws.log_groups.setdefault(region, []).append(check_unit.TRAINING_LOG_GROUP)
        for n, line in enumerate(self.lines):
            aws.events.append({"region": region, "log_group": check_unit.TRAINING_LOG_GROUP,
                               "log_stream": f"{job}/algo-1-1700000000", "timestamp": aws.now(),
                               "ingestion_time": aws.now(), "message": line, "event_id": f"sm-{n}"})
        if self.after:
            self.after(aws, store, job)
        aws.advance(10)
        aws.finish_job(job, status=self.status, billable_s=111)
        aws.advance(60)  # the CLI waiter notices later
        return 0


@pytest.fixture
def clean_tree(monkeypatch):
    monkeypatch.setattr(package, "tree_dirty", lambda repo: [])


def fake_aws() -> FakeAws:
    # start the fake clock at the real time: evlib stamps declared_at with the real clock, and check_unit
    # (rightly) fails a declaration whose declared_at is later than its S3 LastModified
    aws = FakeAws(now=datetime.now(timezone.utc))
    aws.s3.put(BUCKET, spec.SDK_KEY, b"not really a wheel", "2026-11-02T23:00:00Z")  # the sdk channel exists
    return aws


def launch(tmp_path, run_id="t.sm-run-1", *extra, aws=None, job=None):
    aws = aws or fake_aws()
    job = job or FakeJob(aws, tmp_path, run_id)
    out: list[str] = []
    rc = runner.main(base_args(run_id, *extra), aws=aws, out=out.append, waiter=job, sleep=aws.advance)
    return rc, aws, out


def check(aws: FakeAws, run_id: str, **kw) -> check_unit.Checker:
    ev = check_unit.S3Evidence(aws.client("s3"), EVIDENCE)
    ck = check_unit.Checker(run_id, ev, aws, None, poll_s=0.0, now=aws.now(), **kw)
    ck.run()
    return ck


def report(ck) -> str:
    buf = io.StringIO()
    ck.report(buf)
    return buf.getvalue()


def states(ck) -> dict:
    return {k: p.state for k, p in ck.parts.items()}


# ---------------------------------------------------------------------------- spec

def test_job_name_is_valid_unique_and_bounded():
    a, b = spec.job_name("t3.evals.no_action-20261103T020000Z-ab12"), spec.job_name("t3_evals_no_action-20261103T020000Z-ab12")
    assert a != b and a.startswith("t3-evals-no-action-20261103T020000Z-ab12-")
    long = spec.job_name("r" * 128)
    assert len(long) == 63 and spec.JOB_NAME_RE.match(long)
    assert spec.job_name("x:y") == spec.job_name("x:y")  # deterministic
    assert spec.stream_name("a:b*c", "job") == "a_b_c/job"


def test_images_by_instance_type_and_profile():
    assert spec.device_for("ml.m5.xlarge", None) == "cpu"
    assert spec.device_for("ml.g6.2xlarge", True) == "gpu"
    with pytest.raises(spec.SpecError):
        spec.device_for("ml.m5.xlarge", True)
    assert spec.image_uri("cpu").endswith("pytorch-training:2.9.0-cpu-py312-ubuntu22.04-sagemaker-v2.2")
    assert spec.image_uri("gpu").endswith("pytorch-training:2.9.0-gpu-py312-cu130-ubuntu22.04-sagemaker-v2.2")
    with pytest.raises(spec.SpecError):
        spec.price_per_hour("ml.p5.48xlarge")
    assert spec.price_per_hour("ml.m5.xlarge") == 0.23


@pytest.mark.parametrize("key", ["RAN_RL_ADAPTER_TOKEN", "RAN_RL_TWIN_URL", "OPENAI_API_KEY", "DB_PASSWORD",
                                 "AWS_SECRET_ACCESS_KEY", "RUN_ID", "TEAM13_X"])
def test_secret_looking_or_reserved_env_is_refused(key):
    with pytest.raises(spec.SpecError):
        spec.check_env({key: "v"})


def test_data_bucket_comes_from_the_environment_or_the_account(monkeypatch):
    assert spec.data_bucket("111122223333") == "bb26-p-team13-data-111122223333-use2"
    assert spec.data_bucket(None) == "bb26-p-team13-data-<account>-use2"
    monkeypatch.setenv(spec.BUCKET_ENV, "some-team-bucket")
    assert spec.data_bucket("111122223333") == "some-team-bucket"
    assert not hasattr(spec, "BUCKET")  # no bucket literal: its name holds the account id


def test_dry_run_derives_the_bucket_from_the_sts_account(clean_tree, monkeypatch):
    out: list[str] = []
    args = [x for x in base_args("t.sm-bucket", "--dry-run") if x not in ("--account", FAKE_ACCOUNT)]
    assert runner.main(args, aws=FakeAws(now=T0), out=out.append) == 0  # account from the fake STS
    req, _ = dry_run_parts(out)
    assert req["HyperParameters"]["sagemaker_submit_directory"] == json.dumps(
        f"s3://{BUCKET}/jobs/t.sm-bucket/source/sourcedir.tar.gz")
    assert req["OutputDataConfig"]["S3OutputPath"] == f"s3://{BUCKET}/jobs/t.sm-bucket/output"
    out.clear()
    monkeypatch.setenv(spec.BUCKET_ENV, "override-bucket")
    assert runner.main(base_args("t.sm-bucket2", "--dry-run"), aws=FakeAws(now=T0), out=out.append) == 0
    req, _ = dry_run_parts(out)
    assert req["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3Uri"].startswith("s3://override-bucket/")


def test_secrets_must_be_secrets_manager_arns():
    spec.check_secrets({"RAN_RL_ADAPTER_TOKEN": "arn:aws:secretsmanager:us-east-2:111122223333:secret:AmazonSageMaker-twin-AbCdEf"})
    with pytest.raises(spec.SpecError):
        spec.check_secrets({"RAN_RL_ADAPTER_TOKEN": "hunter2"})


def test_summary_never_carries_environment():
    d = {"TrainingJobName": "j", "Environment": {"X": "secret"}, "HyperParameters": {"team13_run_id": '"r"', "x": "1"},
         "CreationTime": datetime(2026, 11, 3, tzinfo=timezone.utc)}
    s = spec.summary(d)
    assert "Environment" not in s and s["HyperParameters"] == {"team13_run_id": '"r"'}
    assert s["CreationTime"] == "2026-11-03T00:00:00.000Z"


def test_requirements_install_the_wheel_from_the_sdk_channel():
    txt = spec.requirements_txt(True, ["transformers==4.57.0"])
    assert "ran-rl[maveric] @ file:///opt/ml/input/data/sdk/ran_rl-1.2.9-py3-none-any.whl" in txt
    assert txt.splitlines()[-1] == "transformers==4.57.0"
    assert "ran-rl" not in spec.requirements_txt(False, [])


def test_placeholders():
    argv = spec.expand(["--run-id", "{run_id}", "--d", "{evidence_dir}", "--m", "{channel:model}/w"], "R", EVIDENCE)
    assert argv == ["--run-id", "R", "--d", spec.LOCAL_EVIDENCE, "--m", "/opt/ml/input/data/model/w"]


# ---------------------------------------------------------------------------- package

def test_archive_is_the_commit_plus_requirements_and_deterministic():
    extra = {"requirements.txt": b"x==1\n"}
    one = package.build_source(REPO, HEAD, ["tools/sm"], extra)
    assert one == package.build_source(REPO, HEAD, ["tools/sm"], extra)
    files = package.members(one)
    assert files.pop("requirements.txt") == b"x==1\n"
    assert set(files) == set(package.tracked_files(REPO, HEAD, ["tools/sm"]))
    for p, b in files.items():
        assert b == package.blob(REPO, HEAD, p), p
    with pytest.raises(package.PackageError):
        package.build_source(REPO, HEAD, ["tools/sm"], {"tools/sm/spec.py": b""})


# ---------------------------------------------------------------------------- dry run and refusals

def test_dry_run_prints_the_request_redacted_and_writes_nothing(clean_tree):
    aws = FakeAws(now=T0)
    out: list[str] = []
    rc = runner.main(base_args("t.sm-dry", "--dry-run", "--env", "FOO=bar"), aws=aws, out=out.append)
    text = "\n".join(out)
    assert rc == 0 and "DRY RUN" in text
    req = json.loads(text[text.index("CreateTrainingJob request"):].split("\n", 1)[1])
    assert set(req["Environment"].values()) == {"<redacted>"} and "FOO" in req["Environment"]
    assert json.loads(req["HyperParameters"]["team13_run_id"]) == "t.sm-dry"
    assert req["AlgorithmSpecification"]["TrainingImage"].endswith("cpu-py312-ubuntu22.04-sagemaker-v2.2")
    assert "bar" not in text  # the env value itself is never printed
    assert not aws.s3.objects and not aws.created
    assert {op for _, op, _ in aws.calls} <= READ_ONLY_OPS


def test_dry_run_request_sets_code_sha_to_the_declared_commit(clean_tree):
    """W5's job entry points (t3.trust.job_gate_eval, job_ladder) read CODE_SHA first: every job gets it."""
    out: list[str] = []
    assert runner.main(base_args("t.sm-codesha", "--dry-run"), aws=FakeAws(now=T0), out=out.append) == 0
    text = "\n".join(out)
    printed = json.loads(text[text.index("CreateTrainingJob request"):].split("\n", 1)[1])
    decl = json.loads(text[text.index("(written FIRST):") + len("(written FIRST):"):text.index("CreateTrainingJob")])
    assert "CODE_SHA" in printed["Environment"] and decl["code_sha"] == HEAD
    a = runner.parser().parse_args(base_args("t.sm-codesha"))
    a.instance_type = spec.DEFAULT_CPU
    p = runner.plan(a)
    assert p["request"]["Environment"]["CODE_SHA"] == p["declaration"]["code_sha"] == HEAD
    assert spec.SHA40_RE.match(p["request"]["Environment"]["CODE_SHA"])
    with pytest.raises(spec.SpecError):
        spec.check_env({"CODE_SHA": "f" * 40})  # the runner sets it; a launch cannot override it


def dry_run_parts(out: list[str]) -> tuple[dict, dict]:
    text = "\n".join(out)
    req = json.loads(text[text.index("CreateTrainingJob request"):].split("\n", 1)[1])
    decl = json.loads(text[text.index("(written FIRST):") + len("(written FIRST):"):text.index("CreateTrainingJob")])
    return req, decl


def botocore_problems(req: dict) -> str:
    """What botocore's own parameter validator says about a CreateTrainingJob request (no AWS call)."""
    from botocore.session import get_session
    from botocore.validate import ParamValidator

    shape = get_session().get_service_model("sagemaker").operation_model("CreateTrainingJob").input_shape
    return ParamValidator().validate(req, shape).generate_report()


@pytest.mark.parametrize("extra", [("--no-sdk",), (), ("--no-sdk", "--inputs", "data=s3://some-bucket/some/prefix/")])
def test_every_request_shape_passes_botocore_validation(clean_tree, extra):
    """A job with no input channel (--no-sdk, no --inputs) used to send InputDataConfig: [] and CreateTrainingJob refused it
    before AWS saw it (min length 1). The field is now omitted when there is no channel."""
    out: list[str] = []
    assert runner.main(base_args("t.sm-valid", "--dry-run", *extra), aws=FakeAws(now=T0), out=out.append) == 0
    req, _ = dry_run_parts(out)
    req["Environment"] = {k: "x" for k in req["Environment"]}   # the dry run prints them redacted
    assert botocore_problems(req) == ""
    assert ("InputDataConfig" in req) == bool(extra and "--inputs" in extra or "--no-sdk" not in extra)
    assert len(req["TrainingJobName"]) <= 63


def test_role_defaults_to_the_organisers_role(clean_tree):
    out: list[str] = []
    assert runner.main(base_args("t.sm-role0", "--dry-run"), aws=FakeAws(now=T0), out=out.append) == 0
    req, decl = dry_run_parts(out)
    assert req["RoleArn"] == "arn:aws:iam::000000000000:role/bb26-SageMakerExecution"
    assert decl["twin"]["config"]["runner"]["role"] == "bb26-SageMakerExecution"
    assert "role        arn:aws:iam::000000000000:role/bb26-SageMakerExecution" in out


def test_team_role_under_bb26_in_the_dry_run(clean_tree):
    out: list[str] = []
    rc = runner.main(base_args("t.sm-role1", "--dry-run", "--role", "bb26/team13-sm-bedrock-eval"),
                     aws=FakeAws(now=T0), out=out.append)
    assert rc == 0
    req, decl = dry_run_parts(out)
    assert req["RoleArn"] == "arn:aws:iam::000000000000:role/bb26/team13-sm-bedrock-eval"
    assert decl["twin"]["config"]["runner"]["role"] == "bb26/team13-sm-bedrock-eval"
    assert "role        arn:aws:iam::000000000000:role/bb26/team13-sm-bedrock-eval" in out


@pytest.mark.parametrize("role", ["OrganizationAccountAccessRole", "bb26/other-role", "bb26-OrgAdmin", "team13-x",
                                  "bb26/team13-", "bb26/x/team13-y", "aws-reserved/team13-x", "bb26/team13-a*b",
                                  "bb26-SageMakerExecution2"])
def test_other_roles_are_refused(clean_tree, role):
    out: list[str] = []
    aws = FakeAws(now=T0)
    assert runner.main(base_args("t.sm-role2", "--dry-run", "--role", role), aws=aws, out=out.append) == 2
    assert "--role" in out[-1] and not aws.s3.objects


def test_missing_team_role_is_refused_before_the_declaration(tmp_path, clean_tree):
    rc, aws, out = launch(tmp_path, "t.sm-role3", "--role", "bb26/team13-sm-bedrock-eval")
    assert rc == 2 and "create it in CloudShell first" in out[-1]
    assert not aws.s3.write_log and not aws.created  # nothing declared, nothing launched


def test_team_role_is_passed_and_recorded_in_the_job_map(tmp_path, clean_tree):
    aws = fake_aws()
    aws.roles["team13-sm-bedrock-eval"] = "/bb26/"
    rc, aws, out = launch(tmp_path, "t.sm-role4", "--role", "bb26/team13-sm-bedrock-eval", aws=aws)
    assert rc == 0, "\n".join(out)
    arn = "arn:aws:iam::000000000000:role/bb26/team13-sm-bedrock-eval"
    assert aws.created[0]["RoleArn"] == arn
    jm = json.loads(aws.s3.objects[(BUCKET, "evidence/outputs/t.sm-role4/job-map.json")]["body"])
    assert jm["role"] == "bb26/team13-sm-bedrock-eval" and jm["role_arn"] == arn
    decl = json.loads(aws.s3.objects[(BUCKET, "evidence/declared/t.sm-role4.json")]["body"])
    assert decl["twin"]["config"]["runner"]["role"] == "bb26/team13-sm-bedrock-eval"


def test_gpu_profile_dry_run(clean_tree):
    out: list[str] = []
    rc = runner.main(base_args("t.sm-gpu", "--dry-run", "--gpu", "--model-s3", f"s3://{BUCKET}/models/otel-1.7b/"),
                     aws=FakeAws(now=T0), out=out.append)
    text = "\n".join(out)
    req = json.loads(text[text.index("CreateTrainingJob request"):].split("\n", 1)[1])
    assert rc == 0
    assert req["ResourceConfig"] == {"InstanceCount": 1, "InstanceType": spec.DEFAULT_GPU, "VolumeSizeInGB": 100}
    assert req["AlgorithmSpecification"]["TrainingImage"].endswith("gpu-py312-cu130-ubuntu22.04-sagemaker-v2.2")
    assert [c["ChannelName"] for c in req["InputDataConfig"]] == ["sdk", "model"]


def test_gpu_type_without_profile_and_secret_env_are_refused(clean_tree):
    out: list[str] = []
    assert runner.main(base_args("t.sm-x", "--dry-run", "--instance-type", "ml.g5.2xlarge"), aws=FakeAws(now=T0),
                       out=out.append) == 2
    assert runner.main(base_args("t.sm-x", "--dry-run", "--env", "RAN_RL_ADAPTER_TOKEN=abc"), aws=FakeAws(now=T0),
                       out=out.append) == 2
    assert "Secrets Manager" in out[-1] and "abc" not in "\n".join(out)


def test_dirty_tree_is_refused_unless_allowed_and_then_labelled(monkeypatch):
    monkeypatch.setattr(package, "tree_dirty", lambda repo: ["t3/evals/run.py"])
    out: list[str] = []
    assert runner.main(base_args("t.sm-dirty", "--dry-run"), aws=FakeAws(now=T0), out=out.append) == 2
    assert "uncommitted changes" in out[-1]
    out.clear()
    assert runner.main(base_args("t.sm-dirty", "--dry-run", "--allow-dirty"), aws=FakeAws(now=T0), out=out.append) == 0
    text = "\n".join(out)
    decl = json.loads(text[text.index("(written FIRST):") + len("(written FIRST):"):text.index("CreateTrainingJob")])
    meta = decl["twin"]["config"]["runner"]
    assert meta["tree_dirty_at_launch"] is True and "NOT shipped" in meta["dirty_note"] and decl["code_sha"] == HEAD


def test_existing_declaration_is_refused_and_nothing_is_written(tmp_path, clean_tree):
    aws = fake_aws()
    store = evlib.S3Store(EVIDENCE, client=aws.s3)
    evlib.write_declared({"run_id": "t.sm-twice", "unit": "u", "track": "track-3",
                          "target": {"metric": "pass_rate", "op": ">=", "value": 0.5, "unit": "ratio"}, "n_items": 3,
                          "test_set_sha256": "0" * 64, "model_id": "m", "grader": "g", "code_sha": HEAD,
                          "region": "us-east-2"}, store)
    before = dict(aws.s3.objects)
    rc, aws, out = launch(tmp_path, "t.sm-twice", aws=aws)
    assert rc == 2 and "already has evidence" in out[-1]
    assert aws.s3.objects == before and not aws.created


# ---------------------------------------------------------------------------- end to end (fake AWS)

def test_launch_order_job_map_summary_and_check_all_pass(tmp_path, clean_tree):
    rc, aws, out = launch(tmp_path, "t.sm-run-1")
    assert rc == 0, "\n".join(out)
    writes = aws.s3.write_log
    assert writes[0] == "evidence/declared/t.sm-run-1.json"  # the declaration is the first write of all
    assert writes[1] == "jobs/t.sm-run-1/source/sourcedir.tar.gz"
    assert writes.index("evidence/outputs/t.sm-run-1/job-map.json") < writes.index("evidence/results/t.sm-run-1.json")
    first_aws_write = next(i for i, c in enumerate(aws.calls) if c[1] not in READ_ONLY_OPS)
    assert aws.calls[first_aws_write][1] in ("CreateLogGroup", "PutObject")  # only the log group may precede it
    created = aws.created[0]
    decl_lm = aws.s3.objects[(BUCKET, "evidence/declared/t.sm-run-1.json")]["last_modified"]
    job = next(j for j in aws.jobs if j["name"] == created["TrainingJobName"])
    assert job["CreationTime"] >= decl_lm + timedelta(seconds=2)
    jm = json.loads(aws.s3.objects[(BUCKET, "evidence/outputs/t.sm-run-1/job-map.json")]["body"])
    src = aws.s3.objects[(BUCKET, "jobs/t.sm-run-1/source/sourcedir.tar.gz")]["body"]
    assert jm["source_sha256"] == hashlib.sha256(src).hexdigest() and jm["pathspec"] == ["tools/sm"]
    summ = json.loads(aws.s3.objects[(BUCKET, "evidence/outputs/t.sm-run-1/job-summary.json")]["body"])
    assert "Environment" not in summ and summ["BillableTimeInSeconds"] == 111
    steps = [json.loads(e["message"])["step"] for e in aws.events if e["log_group"] == spec.LOG_GROUP]
    assert steps[:2] == ["declared", "job_submitted"] and "job_finished" in steps and "summary_written" in steps

    ck = check(aws, "t.sm-run-1")
    text = report(ck)
    assert states(ck) == {"a": "PASS", "b": "PASS", "c": "PASS", "d": "SKIP", "e": "PASS"}, text
    assert "byte-identical to git" in text and "= declaration (before the run), job-map, job's HyperParameters" in text
    assert "attested in /aws/sagemaker/TrainingJobs" in text and "every sha256 equals the stored file" in text
    assert "corroboration only" in text and "match DescribeTrainingJob" in text
    assert "before training job" in text
    assert not any("team13/runner" in s[0] for s in ck.window_sources)  # our events never make the window


def test_check_unit_is_read_only(tmp_path, clean_tree):
    rc, aws, _ = launch(tmp_path, "t.sm-ro")
    n = len(aws.calls)
    check(aws, "t.sm-ro")
    assert {op for _, op, _ in aws.calls[n:]} <= READ_ONLY_OPS


def test_runner_event_with_a_chosen_timestamp_fails_c(tmp_path, clean_tree):
    rc, aws, _ = launch(tmp_path, "t.sm-skew")
    aws.events.append({"region": "us-east-2", "log_group": spec.LOG_GROUP, "log_stream": "t.sm-skew/launcher",
                       "timestamp": aws.now() - timedelta(seconds=45), "ingestion_time": aws.now(),
                       "message": json.dumps({"run_id": "t.sm-skew", "step": "job_finished", "writer": "launcher"}),
                       "event_id": "forged"})
    ck = check(aws, "t.sm-skew")
    assert states(ck)["c"] == "FAIL" and "is 45s before CloudWatch's @ingestionTime" in report(ck)


def test_runner_events_never_widen_the_window(tmp_path, clean_tree):
    rc, aws, _ = launch(tmp_path, "t.sm-wide")
    base = check(aws, "t.sm-wide").window
    aws.advance(3600)
    aws.events.append({"region": "us-east-2", "log_group": spec.LOG_GROUP, "log_stream": "t.sm-wide/launcher",
                       "timestamp": aws.now(), "ingestion_time": aws.now(),
                       "message": json.dumps({"run_id": "t.sm-wide", "step": "late", "writer": "launcher"}),
                       "event_id": "late"})
    ck = check(aws, "t.sm-wide")
    assert ck.window == base and states(ck)["c"] == "PASS"


def test_record_written_after_the_job_fails_c(tmp_path, clean_tree):
    rc, aws, _ = launch(tmp_path, "t.sm-late")
    aws.advance(600)  # 10 minutes after the job ended: outside end + 180 s
    store = evlib.S3Store(EVIDENCE, client=aws.s3)
    evlib.append_records("t.sm-late", "item", [{**items("t.sm-late")[0], "item_id": "late-1"}], store=store)
    text = report(check(aws, "t.sm-late"))
    assert "outside [job CreationTime" in text and "late-1" not in text.split("[c]")[0]
    assert "no team13-evidence line for it in the job's log" in text


def test_tampered_record_fails_attestation(tmp_path, clean_tree):
    def tamper(aws, store, job):  # after the job attested: flip an item's verdict in place
        key = next(k for k in aws.s3.objects if "/records/" in k[1] and "/item/" in k[1])
        rec = json.loads(aws.s3.objects[key]["body"])
        rec["passed"] = not rec["passed"]
        aws.s3.objects[key]["body"] = json.dumps(rec, sort_keys=True).encode() + b"\n"

    aws = fake_aws()
    rc, aws, _ = launch(tmp_path, "t.sm-tamper", aws=aws, job=FakeJob(aws, tmp_path, "t.sm-tamper", after=tamper))
    ck = check(aws, "t.sm-tamper")
    assert states(ck)["c"] == "FAIL" and "is not the" in report(ck)


def test_tampered_source_fails_b(tmp_path, clean_tree):
    rc, aws, _ = launch(tmp_path, "t.sm-src")
    key = (BUCKET, "jobs/t.sm-src/source/sourcedir.tar.gz")
    files = package.members(aws.s3.objects[key]["body"])
    files["tools/sm/spec.py"] += b"\n# changed after the fact\n"
    import gzip
    import tarfile
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as t:
        for name, b in files.items():
            ti = tarfile.TarInfo(name)
            ti.size = len(b)
            t.addfile(ti, io.BytesIO(b))
    aws.s3.objects[key]["body"] = gzip.compress(buf.getvalue())
    text = report(check(aws, "t.sm-src"))
    assert "[b] FAIL" in text and "the declaration (before the run) says" in text
    assert "differ from commit" in text and "tools/sm/spec.py" in text


def ledger_from_s3(aws: FakeAws, run_id: str, out: Path) -> Path:
    """A ledger that ingested the run's evidence in S3 LastModified order, like an S3-polling ingester."""
    import unitfix

    led = unitfix.MiniLedger()
    objs = [(o["last_modified"], key[len("evidence/"):], o["body"]) for (b, key), o in aws.s3.objects.items()
            if key.startswith("evidence/") and run_id in key and not key.startswith("evidence/outputs/")]
    for lm, rel, body in sorted(objs, key=lambda x: (x[0], x[1])):
        kind = {"declared": "declaration", "records": "record", "results": "result"}[rel.split("/")[0]]
        led.add("evidence.artifact", rel, body, run_id, kind, lm.strftime("%Y-%m-%dT%H:%M:%S.000Z"))
    led.write(out)
    return out


def test_part_d_accepts_the_job_record_after_the_result_but_not_an_item(tmp_path, clean_tree):
    rc, aws, _ = launch(tmp_path, "t.sm-ledger")
    ledger = ledger_from_s3(aws, "t.sm-ledger", tmp_path / "ledger")
    ev = check_unit.S3Evidence(aws.client("s3"), EVIDENCE)
    ck = check_unit.Checker("t.sm-ledger", ev, aws, ledger, poll_s=0.0, now=aws.now())
    ck.run()
    text = report(ck)
    assert states(ck)["d"] == "PASS", text
    assert "as expected for a job summary" in text
    aws.advance(30)  # an item that appears after the result is still a failure
    evlib.append_records("t.sm-ledger", "item", [{**items("t.sm-ledger")[0], "item_id": "late"}],
                         store=evlib.S3Store(EVIDENCE, client=aws.s3))
    ledger = ledger_from_s3(aws, "t.sm-ledger", tmp_path / "ledger2")
    ck = check_unit.Checker("t.sm-ledger", check_unit.S3Evidence(aws.client("s3"), EVIDENCE), aws, ledger,
                            poll_s=0.0, now=aws.now())
    ck.run()
    assert states(ck)["d"] == "FAIL" and "after the result was recorded" in report(ck)


def test_job_record_disagreeing_with_aws_fails_c(tmp_path, clean_tree):
    rc, aws, _ = launch(tmp_path, "t.sm-cost")
    j = next(j for j in aws.jobs if j["kind"] == "training")
    j["BillableTimeInSeconds"] = 999
    assert "billable_s 111, AWS says 999" in report(check(aws, "t.sm-cost"))


def test_failed_job_returns_nonzero(tmp_path, clean_tree):
    aws = fake_aws()
    rc, aws, out = launch(tmp_path, "t.sm-fail", aws=aws, job=FakeJob(aws, tmp_path, "t.sm-fail", status="Failed",
                                                                       attest=False))
    assert rc == 1 and any("Failed" in line for line in out)


def test_no_wait_then_finish(tmp_path, clean_tree):
    aws = fake_aws()
    out: list[str] = []
    assert runner.main(base_args("t.sm-later", "--no-wait"), aws=aws, out=out.append, sleep=aws.advance) == 0
    assert "--finish t.sm-later" in out[-1]
    assert check(aws, "t.sm-later").jobs  # the job-map alone lets check_unit find the job
    job = FakeJob(aws, tmp_path, "t.sm-later")
    assert runner.main(["--finish", "t.sm-later", "--no-check"], aws=aws, out=out.append, waiter=job) == 0
    assert states(check(aws, "t.sm-later"))["c"] == "PASS"


# ---------------------------------------------------------------------------- the job side

def test_collect_refuses_a_module_that_declared_differently(tmp_path):
    aws = FakeAws(now=T0)
    store = evlib.S3Store(EVIDENCE, client=aws.s3)
    remote = {"run_id": "r1", "unit": "u", "track": "track-3", "target": {"metric": "pass_rate", "op": ">=",
              "value": 0.5, "unit": "ratio"}, "n_items": 3, "test_set_sha256": "0" * 64, "model_id": "m",
              "grader": "g", "code_sha": HEAD, "region": "us-east-2"}
    ls = evlib.LocalStore(tmp_path)
    evlib.write_declared({**remote, "model_id": "something-else"}, ls)
    evlib.append_records("r1", "item", items("r1"), store=ls)
    with pytest.raises(entry.Refused) as e:
        entry.collect("r1", tmp_path, store, remote)
    assert e.value.code == 6 and "model_id" in str(e.value)
    assert not aws.s3.objects


def test_attest_prints_the_stored_bytes_hash(capsys):
    aws = FakeAws(now=T0)
    store = evlib.S3Store(EVIDENCE, client=aws.s3)
    evlib.append_records("r2", "item", items("r2"), store=store)
    lines = entry.attest("r2", store)
    out = capsys.readouterr().out
    assert len(lines) == 3 and out.count(entry.ATTEST_MARK) == 3
    for body in lines:
        assert body["sha256"] == hashlib.sha256(store.get(body["path"])).hexdigest()


def test_secrets_are_set_in_process_and_never_printed(monkeypatch, capsys):
    class SM:
        def get_secret_value(self, SecretId):
            assert SecretId.startswith("arn:aws:secretsmanager:")
            return {"SecretString": "s3cr3t-value"}

    monkeypatch.delenv("TWIN_TOKEN", raising=False)
    names = entry.fetch_secrets({"TWIN_TOKEN": "arn:aws:secretsmanager:us-east-2:1:secret:AmazonSageMaker-x"},
                                "us-east-2", client=SM())
    assert names == ["TWIN_TOKEN"]
    import os
    assert os.environ["TWIN_TOKEN"] == "s3cr3t-value"
    assert "s3cr3t" not in capsys.readouterr().out


def test_entry_argv_and_exit_codes(tmp_path):
    hp = tmp_path / "hp.json"
    hp.write_text(json.dumps({"team13_entry": json.dumps(json.dumps(["-m", "x", "--a", "{run_id}"]))}))
    assert entry.entry_argv(hp) == ["-m", "x", "--a", "{run_id}"]
    script = tmp_path / "s.py"
    script.write_text("import sys\nsys.exit(7)\n")
    assert entry.run_argv([str(script)]) == 7


def test_selftest_grades_the_declared_test_set_and_refuses_another(tmp_path, monkeypatch):
    from tools.sm import selftest

    aws = FakeAws(now=T0)
    real = evlib.S3Store

    class OnFake(real):  # a subclass, so evlib's isinstance checks still hold
        def __init__(self, uri, **kw):
            super().__init__(uri, client=aws.s3)

    monkeypatch.setattr(evlib, "S3Store", OnFake)
    monkeypatch.setenv("EVIDENCE_S3", EVIDENCE)
    monkeypatch.chdir(REPO)
    suite = "t3/evals/suite/incidents-test.jsonl"
    decl = {"run_id": "r-self", "unit": "runner.selftest", "track": "track-3",
            "target": {"metric": "pass_rate", "op": ">=", "value": 1.0, "unit": "ratio"}, "n_items": 33,
            "test_set_sha256": hashlib.sha256((REPO / suite).read_bytes()).hexdigest(), "model_id": "none",
            "grader": selftest.GRADER, "code_sha": HEAD, "region": "us-east-2"}
    evlib.write_declared(decl, real(EVIDENCE, client=aws.s3))
    assert selftest.main(["--test-set", suite, "--run-id", "r-self", "--evidence-dir", str(tmp_path)]) == 0
    local = evlib.LocalStore(tmp_path)
    its = evlib.read_records("r-self", "item", local)
    assert len(its) == 33 and all(i["passed"] for i in its) and its[0]["item_id"].startswith("inc-")
    assert evlib.read_result("r-self", local)["value"] == 1.0
    assert selftest.main(["--test-set", "tools/sm/spec.py", "--run-id", "r-self", "--evidence-dir",
                          str(tmp_path / "x")]) == 4  # not the declared test set
    assert selftest.grade('{"b":1,"a":2}') == (False, '{"a":2,"b":1}')


def test_evlib_code_sha_uses_the_pin_only_outside_git(tmp_path, monkeypatch):
    monkeypatch.setenv("TEAM13_CODE_SHA", "a" * 40)
    assert evlib.code_sha(REPO) == HEAD  # inside a repo, git wins
    assert evlib.code_sha(tmp_path) == "a" * 40 and evlib.code_dirty(tmp_path) is False
    monkeypatch.delenv("TEAM13_CODE_SHA")
    with pytest.raises(subprocess.CalledProcessError):
        evlib.code_sha(tmp_path)
