from __future__ import annotations

from .base import Adapters


class _InMemory:
    def __init__(self, backing: str):
        self.backing = backing


def mock_adapters() -> Adapters:
    return Adapters("mock", {
        "ledger": _InMemory("in-memory (maps to DynamoDB)"),
        "anchors": _InMemory("in-memory (maps to S3 Object Lock)"),
        "signer": _InMemory("Ed25519 keyring (maps to AWS KMS)"),
        "policy": _InMemory("Cedar mirror (maps to Verified Permissions)"),
        "guardrails": _InMemory("pattern filter (maps to Bedrock Guardrails)"),
        "approvals": _InMemory("in-memory queue (maps to Step Functions + SNS)"),
        "metrics": _InMemory("in-memory frames (maps to CloudWatch + X-Ray)"),
        "flags": _InMemory("flags.json (maps to AWS AppConfig)"),
        "telemetry": _InMemory("twin simulator (maps to Kinesis)"),
    })
