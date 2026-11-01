"""Adapter seams between the SPECTRUM engines and their backing services.

`mock` keeps everything in memory (the default for the demo). `aws` maps each seam onto the AWS
service that will back it in production. The AWS implementations are thin, typed stubs with the
correct boto3 call shapes; they are intentionally not wired into the engines yet and are untested
against a live account. Swap them in one seam at a time.

| Seam            | Mock                    | AWS                                                        |
|-----------------|-------------------------|------------------------------------------------------------|
| LedgerStore     | Python list             | DynamoDB table `spectrum-ledger` (pk=seq)                  |
| AnchorStore     | Python list             | S3 bucket with Object Lock, mode COMPLIANCE                |
| Signer          | Ed25519 KeyRing         | AWS KMS Sign/Verify (per-agent asymmetric keys)            |
| PolicyEngine    | Trust.authorize mirror  | Amazon Verified Permissions IsAuthorized                   |
| Guardrails      | regex patterns          | Bedrock ApplyGuardrail                                     |
| ApprovalBroker  | Oversight queue         | Step Functions task tokens + SNS                           |
| MetricsSink     | Observability frames    | CloudWatch PutMetricData / EMF, X-Ray via ADOT             |
| FlagSource      | flags.json              | AWS AppConfig GetLatestConfiguration                       |
| Telemetry       | Twin.tick               | Kinesis Data Streams PutRecords                            |
"""

from __future__ import annotations

from .base import Adapters


def load_adapters(mode: str) -> Adapters:
    if mode == "aws":
        from .aws import aws_adapters
        return aws_adapters()
    from .mock import mock_adapters
    return mock_adapters()
