"""AWS access for the kit approvals slice: the user's own SSO session (profile fyuz) on this laptop.

Low-level clients only. The browser never sees credentials. Every call here is blocking and is run by the
routes through asyncio.to_thread.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

TABLE = os.environ.get("APPROVALS_TABLE", "team13-approvals")
AUDIT_LOG = os.environ.get("APPROVALS_AUDIT_LOG", "/team13/checkpoint")
STREAM = "control"  # KIT control_record()


@dataclass
class Aws:
    ddb: Any
    logs: Any
    sts: Any
    label: str  # LIVE | FIXTURE


def live() -> Aws:
    import boto3
    from botocore.config import Config

    session = boto3.Session(profile_name=os.environ.get("APPROVALS_AWS_PROFILE", "fyuz"),
                            region_name=os.environ.get("APPROVALS_REGION", "us-east-2"))
    base = dict(connect_timeout=5, read_timeout=10)
    return Aws(
        ddb=session.client("dynamodb", config=Config(**base, retries={"max_attempts": 2, "mode": "standard"})),
        # PutLogEvents is not idempotent: never let botocore re-send it after a timeout (review F14).
        logs=session.client("logs", config=Config(**base, retries={"total_max_attempts": 1, "mode": "standard"})),
        sts=session.client("sts", config=Config(**base, retries={"max_attempts": 2, "mode": "standard"})),
        label="LIVE",
    )


def to_item(av: dict | None) -> dict | None:
    if av is None:
        return None
    from boto3.dynamodb.types import TypeDeserializer

    d = TypeDeserializer()
    return {k: d.deserialize(v) for k, v in av.items()}


def classify(exc: BaseException) -> str:
    """aws_session_expired | no_access | aws_unreachable | aws_error (review F10: sso-session profiles)."""
    from botocore import exceptions as be

    if isinstance(exc, (be.SSOTokenLoadError, be.TokenRetrievalError, be.UnauthorizedSSOTokenError,
                        be.NoCredentialsError, be.SSOError, be.CredentialRetrievalError, be.ProfileNotFound)):
        return "aws_session_expired"
    if isinstance(exc, be.ClientError):
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if code.startswith("ExpiredToken") or code in ("UnrecognizedClientException", "InvalidClientTokenId"):
            return "aws_session_expired"
        if code.startswith("AccessDenied") or code == "UnauthorizedOperation":
            return "no_access"
        return "aws_error"
    if isinstance(exc, (be.EndpointConnectionError, be.ConnectTimeoutError, be.ReadTimeoutError,
                        be.ConnectionClosedError)):
        return "aws_unreachable"
    return "aws_error"


def maybe_landed(exc: BaseException) -> bool:
    """True when a write may have reached AWS although the call failed (a timeout after sending)."""
    from botocore import exceptions as be

    return isinstance(exc, (be.ReadTimeoutError, be.ConnectionClosedError)) or not isinstance(exc, be.BotoCoreError | be.ClientError)


def error_code(exc: BaseException) -> str:
    from botocore.exceptions import ClientError

    return str(exc.response.get("Error", {}).get("Code", "")) if isinstance(exc, ClientError) else type(exc).__name__


MESSAGES = {
    "offline": "Offline: the approvals table isn't read (SPECTRUM_OFFLINE=1).",
    "aws_session_expired": "Your AWS session expired. On this laptop run `aws sso login --profile fyuz`, then press Refresh.",
    "no_access": "AWS refused access to the approvals table (AccessDenied). Nothing was decided.",
    "aws_unreachable": "AWS could not be reached (network). Nothing was decided.",
    "aws_error": "AWS returned an error. Nothing was decided.",
}
