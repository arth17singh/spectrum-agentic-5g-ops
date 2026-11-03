"""register.py: a dry run makes no AWS call; --apply makes exactly the planned calls, each valid
against the botocore service model (checked offline with botocore's own parameter validator).
It makes no IAM calls: roles and the Gateway-role policy come from iam_cloudshell.sh."""
from __future__ import annotations

import json

import boto3
import pytest
from botocore.exceptions import ClientError
from botocore.validate import validate_parameters

from t3.deploy import register
from t3.policies import generate

GW = "arn:aws:bedrock-agentcore:us-east-2:000000000000:gateway/team13-checkpoint-abc123"
FN = "arn:aws:lambda:us-east-2:000000000000:function:team13-ran-tools"
SESSION = boto3.Session(region_name="us-east-2", aws_access_key_id="x", aws_secret_access_key="x")
KIT_TARGETS = ("case", "work", "nlm", "swarm", "front")   # what the kit's deploy creates


@pytest.fixture
def kit_dir(tmp_path):
    (tmp_path / "kit.json").write_text(json.dumps({
        "name": "team13", "region": "us-east-2", "lead": "noc_lead",
        "specialists": ["ran_monitor", "ran_optimizer", "ran_explainer"], "may_change": ["noc_lead"]}))
    (tmp_path / "kit-state.json").write_text(json.dumps({
        "gateway_id": "team13-checkpoint-abc123", "gateway_arn": GW, "engine_id": "team13_policies-xyz"}))
    return tmp_path


class ValidatingFake:
    """A client that validates each request against the real service model and answers minimally."""

    def __init__(self, service, log, world):
        self.service, self.log, self.world = service, log, world
        self.model = SESSION.client(service).meta.service_model

    def __getattr__(self, op):
        shape = self.model.operation_model("".join(p.capitalize() for p in op.split("_"))).input_shape

        def call(**params):
            validate_parameters(params, shape)  # raises ParamValidationError on a bad request
            self.log.append((self.service, op, params))
            return self.world(self.service, op, params)
        return call


def _missing(code, op):
    return ClientError({"Error": {"Code": code, "Message": "no"}}, op)


def make_world(existing_targets=KIT_TARGETS, existing_policies=None, existing_functions=()):
    policies = dict(existing_policies or {})
    targets = list(existing_targets)
    functions = set(existing_functions)

    def world(service, op, params):
        if op == "get_caller_identity":
            return {"Account": "000000000000"}
        if op == "describe_repositories":
            raise _missing("RepositoryNotFoundException", "DescribeRepositories")
        if op == "get_function":
            if params["FunctionName"] not in functions:
                raise _missing("ResourceNotFoundException", "GetFunction")
            return {"Configuration": {"State": "Active", "LastUpdateStatus": "Successful"}}
        if op == "create_function":
            functions.add(params["FunctionName"])
        if op == "list_gateway_targets":
            return {"items": [{"targetId": f"t-{n}", "name": n, "status": "READY"} for n in targets]}
        if op in ("create_gateway_target", "update_gateway_target") and params["name"] not in targets:
            targets.append(params["name"])
        if op == "list_policies":
            return {"policies": [{"name": n, "policyId": f"p-{n}"} for n in policies]}
        if op == "create_policy":
            return {"policyId": f"new-{params['name']}"}
        if op == "get_policy":
            pid = params["policyId"]
            if pid.startswith("new-"):
                return {"status": "ACTIVE", "policyId": pid}
            name = pid[2:]
            if name not in policies:
                raise _missing("ResourceNotFoundException", "GetPolicy")
            return {"status": "ACTIVE", "definition": {"policy": {"statement": policies[name]}}}
        if op == "delete_policy":
            policies.pop(params["policyId"][2:], None)
        return {}
    return world


IMAGE = "000000000000.dkr.ecr.us-east-2.amazonaws.com/team13-ran-tools:0123456789ab"


def run_apply(kit_dir, world, *extra):
    log = []
    rc = register.main(["--kit-dir", str(kit_dir), "--image-uri", IMAGE, "--evidence-bucket", "team-bucket",
                        "--apply", *extra],
                       client_factory=lambda svc: ValidatingFake(svc, log, world), out=lambda *_: None,
                       wait=lambda s: None)
    return rc, log


def test_dry_run_makes_no_aws_call(kit_dir, monkeypatch):
    def no_session(*a, **k):
        raise AssertionError("dry run must not create a boto3 session")
    monkeypatch.setattr(boto3, "Session", no_session)
    lines = []
    assert register.main(["--kit-dir", str(kit_dir), "--image-uri", IMAGE, "--evidence-bucket", "team-bucket"],
                         out=lines.append) == 0
    text = "\n".join(lines)
    # B1: the corrected environment for both functions is visible in the plan
    assert text.count('"EVIDENCE_S3": "s3://team-bucket/evidence/"') == 2 and "T3_EVIDENCE" not in text
    assert '"T1_POLICY_PATH": "/var/task/t1/rl/artifacts/ppo_v0.zip"' in text
    assert '"T2_PREDICTOR_PATH": "/var/task/t2/artifacts/t2_response_v0.npz"' in text
    assert '"T3_TRUST_TIMEOUT_S": "15"' in text and '"T3_TRUST_LIVE": "0"' in text
    assert f'"ImageUri": "{IMAGE}"' in text
    assert text.startswith("DRY RUN") and "WARNING" not in text
    assert " iam." not in text   # IAM is iam_cloudshell.sh's job
    assert "WOULD CALL ecr.create_repository" in text
    assert text.count("WOULD CALL lambda.create_function") == 2
    assert text.count("WOULD CALL bedrock-agentcore-control.create_gateway_target") == 1   # ran
    assert "TAKE OVER the kit's target nlm" in text
    assert text.count("WOULD CALL bedrock-agentcore-control.update_gateway_target") == 1   # nlm
    assert text.count("WOULD CALL bedrock-agentcore-control.create_policy") == len(generate.rules())
    assert "require_human_approval" in text and '"MemorySize": 3008' in text


def test_dry_run_without_a_kit_folder_uses_placeholders():
    lines = []
    assert register.main(["--image-uri", IMAGE], out=lines.append) == 0
    assert any("<gateway_id>" in line for line in lines)


@pytest.mark.parametrize("apply", [False, True])
@pytest.mark.parametrize("image", [None, "000000000000.dkr.ecr.us-east-2.amazonaws.com/team13-ran-tools:latest",
                                   "000000000000.dkr.ecr.us-east-2.amazonaws.com/team13-ran-tools",
                                   "000000000000.dkr.ecr.us-west-2.amazonaws.com/team13-ran-tools:0123456789ab",
                                   "team13-ran-tools:dev"])
def test_refuses_without_a_pinned_image(kit_dir, image, apply):
    """B2: no --image-uri, no tag, :latest, the wrong region or a local tag: refused, no AWS call."""
    lines, made = [], []
    args = ["--kit-dir", str(kit_dir), "--evidence-bucket", "team-bucket"] + (["--image-uri", image] if image else [])
    rc = register.main(args + (["--apply"] if apply else []), client_factory=lambda s: made.append(s), out=lines.append)
    assert rc == 2 and made == [] and "--image-uri" in "\n".join(lines)


def test_digest_pinned_image_is_accepted(kit_dir):
    digest = "000000000000.dkr.ecr.us-east-2.amazonaws.com/team13-ran-tools@sha256:" + "a" * 64
    assert register.main(["--kit-dir", str(kit_dir), "--image-uri", digest], out=lambda *_: None) == 0


def test_apply_refuses_an_image_from_another_account(kit_dir):
    other = IMAGE.replace("000000000000", "111111111111")
    log = []
    rc = register.main(["--kit-dir", str(kit_dir), "--image-uri", other, "--evidence-bucket", "b", "--apply"],
                       client_factory=lambda svc: ValidatingFake(svc, log, make_world()), out=lambda *_: None,
                       wait=lambda s: None)
    assert rc == 2 and [op for _, op, _ in log] == ["get_caller_identity"]


def test_trust_live_is_opt_in_per_run(kit_dir):
    rc, log = run_apply(kit_dir, make_world(), "--trust-live")
    assert rc == 0
    trust = next(p for _, op, p in log if op == "create_function" and p["FunctionName"] == "team13-trust-gate")
    assert trust["Environment"]["Variables"]["T3_TRUST_LIVE"] == "1"


def test_apply_makes_valid_calls_in_order(kit_dir):
    rc, log = run_apply(kit_dir, make_world())
    assert rc == 0
    ops = [op for _, op, _ in log]
    assert not [c for c in log if c[0] == "iam"]  # no IAM calls at all
    assert ops.index("create_repository") < ops.index("create_function") < ops.index("create_gateway_target") \
        < ops.index("create_policy")
    assert "create_table" not in ops   # case rows live in the kit's approvals table (A10)

    [(_, _, repo_policy)] = [c for c in log if c[1] == "set_repository_policy"]
    stmt = json.loads(repo_policy["policyText"])["Statement"][0]
    assert stmt["Principal"] == {"Service": "lambda.amazonaws.com"}

    fns = {p["FunctionName"]: p for _, op, p in log if op == "create_function"}
    tools, trust = fns["team13-ran-tools"], fns["team13-trust-gate"]
    assert tools["Role"] == "arn:aws:iam::000000000000:role/bb26/team13-ran-tools"
    assert (tools["MemorySize"], tools["Timeout"]) == (3008, 120)
    assert tools["ImageConfig"]["Command"] == ["t3.tools.app.handler"]
    assert tools["Code"]["ImageUri"] == IMAGE and trust["Code"]["ImageUri"] == IMAGE
    env = tools["Environment"]["Variables"]
    assert env["APPROVALS_TABLE"] == "team13-approvals" and env["EVIDENCE_S3"] == "s3://team-bucket/evidence/"
    assert env["T1_POLICY_PATH"] == "/var/task/t1/rl/artifacts/ppo_v0.zip"
    assert env["T2_PREDICTOR_PATH"] == "/var/task/t2/artifacts/t2_response_v0.npz"
    assert env["T3_TRUST_TIMEOUT_S"] == "15" and env["T3_TRUST_FUNCTION"] == "team13-trust-gate"
    assert "T3_TOOL_NAME_FALLBACK" not in env and "T3_STORE" not in env and "T3_EVIDENCE" not in env
    assert trust["Environment"]["Variables"]["EVIDENCE_S3"] == "s3://team-bucket/evidence/"
    # the tools' own writer resolves that variable to the same place the trust gate (evlib) writes
    from t3.tools import evidence as ev
    w = ev.from_env(env)
    assert (w.bucket, w.prefix) == ("team-bucket", "evidence")
    assert trust["Role"] == "arn:aws:iam::000000000000:role/bb26/team13-trust-gate"
    assert trust["ImageConfig"]["Command"] == ["t3.trust.handler.handler"]
    assert trust["Environment"]["Variables"]["T3_TRUST_LIVE"] == "0"
    assert trust["Environment"]["Variables"]["T3_POLICY_ENGINE_ID"] == "team13_policies-xyz"

    created = {p["name"]: p for _, op, p in log if op == "create_gateway_target"}
    updated = {p["name"]: p for _, op, p in log if op == "update_gateway_target"}
    assert set(created) == {"ran"} and set(updated) == {"nlm"} and updated["nlm"]["targetId"] == "t-nlm"
    ran_tools = {t["name"]: t for t in created["ran"]["targetConfiguration"]["mcp"]["lambda"]["toolSchema"]["inlinePayload"]}
    assert set(ran_tools) == {"observe", "propose", "predict", "preflight", "apply_change"}
    apply_schema = ran_tools["apply_change"]["inputSchema"]
    assert {"case_id", "item_id", "change", "reason", "approval", "special_context"} == set(apply_schema["properties"])
    assert apply_schema["properties"]["change"]["type"] == "string"
    assert "case_id" not in apply_schema["required"]
    assert created["ran"]["targetConfiguration"]["mcp"]["lambda"]["lambdaArn"] == FN
    nlm = updated["nlm"]["targetConfiguration"]["mcp"]["lambda"]
    assert nlm["lambdaArn"] == FN and [t["name"] for t in nlm["toolSchema"]["inlinePayload"]] == ["ask_nlm", "ask"]

    rules = [p for _, op, p in log if op == "create_policy"]
    assert [p["name"] for p in rules] == [r["name"] for r in generate.rules()]
    for p in rules:
        assert "@GW@" not in p["definition"]["policy"]["statement"] and GW in p["definition"]["policy"]["statement"]


def test_apply_skips_unchanged_rules_and_replaces_changed_ones(kit_dir):
    rules = generate.rules()
    same = {rules[0]["name"]: generate.fill(rules[0]["statement"], GW), rules[1]["name"]: "permit (old text);"}
    rc, log = run_apply(kit_dir, make_world(existing_policies=same))
    assert rc == 0
    created = [p["name"] for _, op, p in log if op == "create_policy"]
    deleted = [p["policyId"] for _, op, p in log if op == "delete_policy"]
    assert rules[0]["name"] not in created and rules[1]["name"] in created
    assert deleted == [f"p-{rules[1]['name']}"]


def test_second_run_updates_instead_of_duplicating(kit_dir):
    rc, log = run_apply(kit_dir, make_world(existing_targets=KIT_TARGETS + ("ran",),
                                            existing_functions=("team13-ran-tools", "team13-trust-gate")))
    assert rc == 0
    assert sorted(p["targetId"] for _, op, p in log if op == "update_gateway_target") == ["t-nlm", "t-ran"]
    assert not [c for c in log if c[1] in ("create_gateway_target", "create_function")]
    touched = {p.get("targetId") for _, op, p in log if op in ("update_gateway_target", "delete_gateway_target")}
    assert not touched & {"t-case", "t-work", "t-swarm", "t-front"}   # the kit's other targets are untouched
    assert sorted(p["FunctionName"] for _, op, p in log if op == "update_function_code") == \
        ["team13-ran-tools", "team13-trust-gate"]
    confs = {p["FunctionName"]: p["MemorySize"] for _, op, p in log if op == "update_function_configuration"}
    assert confs == {"team13-ran-tools": 3008, "team13-trust-gate": 1024}


def test_apply_refuses_without_kit_state_or_bucket(tmp_path, kit_dir):
    lines = []
    empty = tmp_path / "empty"
    empty.mkdir()
    assert register.main(["--kit-dir", str(empty), "--image-uri", IMAGE, "--evidence-bucket", "b", "--apply"], out=lines.append) == 2
    assert register.main(["--kit-dir", str(kit_dir), "--image-uri", IMAGE, "--apply"], out=lines.append) == 2


def test_warns_when_noc_lead_cannot_change(kit_dir):
    cfg = json.loads((kit_dir / "kit.json").read_text())
    cfg["may_change"] = ["ran_optimizer"]
    cfg["specialists"] = ["ran_monitor"]
    warnings = register.preflight_warnings(cfg, "team13")
    assert any("noc_lead is not in kit.json may_change" in w for w in warnings)
    assert any("ran_explainer is not in kit.json" in w for w in warnings)


def test_cloudshell_script_is_parameterised_and_secret_free():
    text = (register.Path(register.__file__).parent / "iam_cloudshell.sh").read_text()
    assert "aws sts get-caller-identity" in text and "--path /bb26/" in text
    assert "bb26-boundary/bb26-Boundary" in text and "team13-ran-tools-invoke" in text
    import re
    assert not re.search(r"\b\d{12}\b", text)   # no account id
