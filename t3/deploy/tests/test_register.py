"""register.py: a dry run makes no AWS call; --apply makes exactly the planned calls, each valid
against the botocore service model (checked offline with botocore's own parameter validator)."""
from __future__ import annotations

import json

import boto3
import pytest
from botocore.exceptions import ClientError
from botocore.validate import validate_parameters

from t3.deploy import register
from t3.policies import generate

FN = "arn:aws:lambda:us-east-2:000000000000:function:team13-ran-tools"
GW = "arn:aws:bedrock-agentcore:us-east-2:000000000000:gateway/team13-checkpoint-abc123"
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


def make_world(existing_targets=KIT_TARGETS, existing_policies=None):
    policies = dict(existing_policies or {})
    targets = list(existing_targets)

    def world(service, op, params):
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
                raise ClientError({"Error": {"Code": "ResourceNotFoundException", "Message": "gone"}}, "GetPolicy")
            return {"status": "ACTIVE", "definition": {"policy": {"statement": policies[name]}}}
        if op == "delete_policy":
            policies.pop(params["policyId"][2:], None)
        return {}
    return world


def run_apply(kit_dir, world):
    log = []
    rc = register.main(["--kit-dir", str(kit_dir), "--function-arn", FN, "--apply"],
                       client_factory=lambda svc: ValidatingFake(svc, log, world), out=lambda *_: None,
                       wait=lambda s: None)
    return rc, log


def test_dry_run_makes_no_aws_call(kit_dir, monkeypatch):
    def no_session(*a, **k):
        raise AssertionError("dry run must not create a boto3 session")
    monkeypatch.setattr(boto3, "Session", no_session)
    lines = []
    assert register.main(["--kit-dir", str(kit_dir), "--function-arn", FN], out=lines.append) == 0
    text = "\n".join(lines)
    assert text.startswith("DRY RUN") and "WARNING" not in text
    assert "WOULD CALL iam.put_role_policy" in text and register.INLINE_POLICY in text
    assert text.count("WOULD CALL bedrock-agentcore-control.create_gateway_target") == 1   # ran
    assert "TAKE OVER the kit's target nlm" in text
    assert text.count("WOULD CALL bedrock-agentcore-control.update_gateway_target") == 1   # nlm
    assert text.count("WOULD CALL bedrock-agentcore-control.create_policy") == len(generate.rules())
    assert "require_human_approval" in text and '"memory_mb": 3008' in text


def test_dry_run_without_a_kit_folder_uses_placeholders():
    lines = []
    assert register.main([], out=lines.append) == 0
    assert any("<gateway_id>" in line for line in lines)


def test_apply_makes_valid_calls_in_order(kit_dir):
    rc, log = run_apply(kit_dir, make_world())
    assert rc == 0
    ops = [op for _, op, _ in log]
    assert ops[0] == "put_role_policy"
    assert ops.index("create_gateway_target") < ops.index("create_policy")
    assert "create_table" not in ops   # case rows live in the kit's approvals table (A10)

    [(_, _, role)] = [c for c in log if c[1] == "put_role_policy"]
    assert role["RoleName"] == "team13-gateway" and role["PolicyName"] == "team13-ran-tools-invoke"
    assert role["PolicyName"] != role["RoleName"]  # the kit rewrites only the policy named like the role
    doc = json.loads(role["PolicyDocument"])
    assert doc["Statement"] == [{"Sid": "InvokeTeam13RanTools", "Effect": "Allow", "Action": "lambda:InvokeFunction",
                                 "Resource": [FN, FN + ":*"]}]

    created = {p["name"]: p for _, op, p in log if op == "create_gateway_target"}
    updated = {p["name"]: p for _, op, p in log if op == "update_gateway_target"}
    assert set(created) == {"ran"} and set(updated) == {"nlm"} and updated["nlm"]["targetId"] == "t-nlm"
    ran_tools = {t["name"]: t for t in created["ran"]["targetConfiguration"]["mcp"]["lambda"]["toolSchema"]["inlinePayload"]}
    assert set(ran_tools) == {"observe", "propose", "predict", "preflight", "apply_change"}
    apply_schema = ran_tools["apply_change"]["inputSchema"]
    assert {"case_id", "item_id", "change", "reason", "approval", "special_context"} == set(apply_schema["properties"])
    assert apply_schema["properties"]["change"]["type"] == "string"
    assert "case_id" not in apply_schema["required"]
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


def test_second_run_updates_targets_instead_of_duplicating(kit_dir):
    rc, log = run_apply(kit_dir, make_world(existing_targets=KIT_TARGETS + ("ran",)))
    assert rc == 0
    assert sorted(p["targetId"] for _, op, p in log if op == "update_gateway_target") == ["t-nlm", "t-ran"]
    assert not [c for c in log if c[1] == "create_gateway_target"]
    # the kit's other targets are never touched
    touched = {p.get("targetId") for _, op, p in log if op in ("update_gateway_target", "delete_gateway_target")}
    assert not touched & {"t-case", "t-work", "t-swarm", "t-front"}


def test_apply_refuses_without_kit_state(tmp_path):
    lines = []
    assert register.main(["--kit-dir", str(tmp_path), "--function-arn", FN, "--apply"], out=lines.append) == 2


def test_warns_when_noc_lead_cannot_change(kit_dir):
    cfg = json.loads((kit_dir / "kit.json").read_text())
    cfg["may_change"] = ["ran_optimizer"]
    cfg["specialists"] = ["ran_monitor"]
    warnings = register.preflight_warnings(cfg, "team13")
    assert any("noc_lead is not in kit.json may_change" in w for w in warnings)
    assert any("ran_explainer is not in kit.json" in w for w in warnings)
