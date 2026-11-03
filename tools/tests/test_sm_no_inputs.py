"""A job with no input channels (--no-sdk, no --model-s3, no --inputs) must not send InputDataConfig:
CreateTrainingJob rejects an empty list client-side ("Invalid length for parameter InputDataConfig,
value: 0, valid min length: 1"), which is how four Bedrock CPU runs never started. Offline: botocore's
own ParamValidator against the SageMaker service model, no AWS call."""

import pytest

from tools.sm import spec

botocore_session = pytest.importorskip("botocore.session")
from botocore.validate import ParamValidator  # noqa: E402

SHA = "c" * 40


def _request(channels, sdk):
    return spec.build_request(
        run_id="t1.interface-bedrock-gpt-oss-20b-2", job=spec.job_name("t1.interface-bedrock-gpt-oss-20b-2"),
        role_arn="arn:aws:iam::000000000000:role/bb26/team13-sm-bedrock-eval", region=spec.REGION,
        bucket="bb26-p-team13-data-000000000000-use2", device="cpu", instance_type="ml.c5.xlarge", volume_gb=30,
        max_run_s=3600, source_uri="s3://bb26-p-team13-data-000000000000-use2/jobs/x/source/sourcedir.tar.gz",
        source_sha256="d" * 64, code_sha=SHA, argv=["-m", "t1.evals.run", "--unit", "interface"], channels=channels,
        env={}, secrets={}, sdk=sdk)


def _errors(request):
    shape = botocore_session.get_session().get_service_model("sagemaker").operation_model("CreateTrainingJob").input_shape
    report = ParamValidator().validate(request, shape)
    return report.generate_report() if report.has_errors() else ""


def test_no_channels_omits_input_data_config_and_validates():
    req = _request([], sdk=False)
    assert "InputDataConfig" not in req
    assert _errors(req) == ""


def test_channels_are_still_sent_and_validate():
    req = _request([(spec.SDK_CHANNEL, "s3://b/sdk/w.whl"), ("adapter", "s3://b/a/")], sdk=True)
    assert [c["ChannelName"] for c in req["InputDataConfig"]] == ["sdk", "adapter"]
    assert _errors(req) == ""


def test_the_old_empty_list_is_what_botocore_refuses():
    req = _request([], sdk=False)
    req["InputDataConfig"] = []
    assert "InputDataConfig" in _errors(req) and "min length: 1" in _errors(req)
