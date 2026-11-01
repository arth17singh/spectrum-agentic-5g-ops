"""AWS-backed adapters. Thin stubs with real boto3 call shapes; untested against a live account.

Environment (set by the CDK stack outputs):
  SPECTRUM_LEDGER_TABLE, SPECTRUM_ANCHOR_BUCKET, SPECTRUM_POLICY_STORE_ID, SPECTRUM_GUARDRAIL_ID,
  SPECTRUM_GUARDRAIL_VERSION, SPECTRUM_APPCONFIG_APP, SPECTRUM_APPCONFIG_ENV, SPECTRUM_APPCONFIG_PROFILE,
  SPECTRUM_TELEMETRY_STREAM, AWS_REGION
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

from .base import Adapters

REGION = os.environ.get("AWS_REGION", "us-west-2")


def _client(name: str):
    import boto3  # optional dependency: `uv sync --extra aws`
    return boto3.client(name, region_name=REGION)


class DynamoLedgerStore:
    backing = "Amazon DynamoDB"

    def __init__(self):
        self.table = os.environ.get("SPECTRUM_LEDGER_TABLE", "spectrum-ledger")
        self.ddb = _client("dynamodb")

    def put(self, record: dict) -> None:
        # Conditional write: a sequence number can never be overwritten.
        self.ddb.put_item(
            TableName=self.table,
            Item={"seq": {"N": str(record["seq"])}, "hash": {"S": record["hash"]},
                  "record": {"S": json.dumps(record, sort_keys=True)}},
            ConditionExpression="attribute_not_exists(seq)",
        )

    def get(self, seq: int) -> dict | None:
        out = self.ddb.get_item(TableName=self.table, Key={"seq": {"N": str(seq)}})
        return json.loads(out["Item"]["record"]["S"]) if "Item" in out else None


class S3ObjectLockAnchors:
    backing = "Amazon S3 Object Lock (COMPLIANCE)"

    def __init__(self):
        self.bucket = os.environ.get("SPECTRUM_ANCHOR_BUCKET", "spectrum-ledger-anchors-us-west-2")
        self.s3 = _client("s3")

    def put_anchor(self, anchor: dict) -> dict:
        retain = datetime.now(timezone.utc) + timedelta(days=2555)
        out = self.s3.put_object(
            Bucket=self.bucket, Key=f"anchors/{anchor['n']:06d}.json", Body=json.dumps(anchor).encode(),
            ObjectLockMode="COMPLIANCE", ObjectLockRetainUntilDate=retain, ChecksumAlgorithm="SHA256",
        )
        return {"version_id": out.get("VersionId"), "etag": out.get("ETag")}


class KmsSigner:
    backing = "AWS KMS"

    def __init__(self):
        self.kms = _client("kms")

    def sign(self, key_id: str, message: str) -> str:
        out = self.kms.sign(KeyId=key_id, Message=message.encode(), MessageType="RAW",
                            SigningAlgorithm="ECDSA_SHA_256")
        return out["Signature"].hex()

    def verify(self, key_id: str, message: str, signature: str) -> bool:
        out = self.kms.verify(KeyId=key_id, Message=message.encode(), MessageType="RAW",
                              Signature=bytes.fromhex(signature), SigningAlgorithm="ECDSA_SHA_256")
        return bool(out["SignatureValid"])


class VerifiedPermissions:
    backing = "Amazon Verified Permissions"

    def __init__(self):
        self.store = os.environ.get("SPECTRUM_POLICY_STORE_ID", "")
        self.avp = _client("verifiedpermissions")

    def is_authorized(self, principal: dict, action: str, resource: dict, context: dict) -> dict:
        def attr(v):
            if isinstance(v, bool):
                return {"boolean": v}
            if isinstance(v, (int, float)):
                return {"long": int(v)}
            if isinstance(v, list):
                return {"set": [attr(x) for x in v]}
            return {"string": str(v)}

        out = self.avp.is_authorized(
            policyStoreId=self.store,
            principal={"entityType": "Spectrum::Agent", "entityId": principal["id"]},
            action={"actionType": "Spectrum::Action", "actionId": action},
            resource={"entityType": "Spectrum::NetworkResource", "entityId": str(resource.get("id"))},
            context={"contextMap": {k: attr(v) for k, v in context.items()}},
        )
        return {"decision": out["decision"].lower(),
                "policies": [p["policyId"] for p in out.get("determiningPolicies", [])]}


class BedrockGuardrails:
    backing = "Amazon Bedrock Guardrails"

    def __init__(self):
        self.rt = _client("bedrock-runtime")
        self.gid = os.environ.get("SPECTRUM_GUARDRAIL_ID", "")
        self.ver = os.environ.get("SPECTRUM_GUARDRAIL_VERSION", "DRAFT")

    def apply(self, text: str, source: str) -> dict | None:
        out = self.rt.apply_guardrail(guardrailIdentifier=self.gid, guardrailVersion=self.ver, source="INPUT",
                                      content=[{"text": {"text": text}}])
        return None if out["action"] == "NONE" else {"action": out["action"], "assessments": out["assessments"]}


class StepFunctionsApprovals:
    backing = "AWS Step Functions (task tokens)"

    def __init__(self):
        self.sfn = _client("stepfunctions")

    def complete(self, task_token: str, decision: dict) -> None:
        if decision.get("decision") == "approved":
            self.sfn.send_task_success(taskToken=task_token, output=json.dumps(decision))
        else:
            self.sfn.send_task_failure(taskToken=task_token, error="Rejected", cause=decision.get("reason", ""))


class CloudWatchMetrics:
    backing = "Amazon CloudWatch"

    def __init__(self):
        self.cw = _client("cloudwatch")

    def put(self, namespace: str, metrics: list[dict]) -> None:
        self.cw.put_metric_data(Namespace=namespace, MetricData=[
            {"MetricName": m["name"], "Value": float(m["value"]), "Unit": m.get("unit", "None"),
             "Dimensions": [{"Name": k, "Value": str(v)} for k, v in m.get("dims", {}).items()]} for m in metrics])


class AppConfigFlags:
    backing = "AWS AppConfig"

    def __init__(self):
        self.ac = _client("appconfigdata")
        self.token = None

    def flags(self) -> dict:
        if self.token is None:
            self.token = self.ac.start_configuration_session(
                ApplicationIdentifier=os.environ.get("SPECTRUM_APPCONFIG_APP", "spectrum"),
                EnvironmentIdentifier=os.environ.get("SPECTRUM_APPCONFIG_ENV", "demo"),
                ConfigurationProfileIdentifier=os.environ.get("SPECTRUM_APPCONFIG_PROFILE", "flags"),
            )["InitialConfigurationToken"]
        out = self.ac.get_latest_configuration(ConfigurationToken=self.token)
        self.token = out["NextPollConfigurationToken"]
        body = out["Configuration"].read()
        return json.loads(body) if body else {}


def aws_adapters() -> Adapters:
    return Adapters("aws", {
        "ledger": DynamoLedgerStore(), "anchors": S3ObjectLockAnchors(), "signer": KmsSigner(),
        "policy": VerifiedPermissions(), "guardrails": BedrockGuardrails(), "approvals": StepFunctionsApprovals(),
        "metrics": CloudWatchMetrics(), "flags": AppConfigFlags(),
    })
