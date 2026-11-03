"""Read-only AWS access for the panel, with the laptop's own AWS session (AWS_PROFILE, e.g. fyuz).

Two reads, nothing else:
- CloudWatch Logs FilterLogEvents on the kit's audit log (/team13/checkpoint): the swarm's governed calls
  for one case, and the panel's `asknoc-<turn>` tag to learn whether a timed-out send arrived;
- DynamoDB GetItem on the kit's approvals table: the `t3-active` run row and `chg-...` approval rows.
The panel never writes to AWS. Every botocore failure (ClientError, and BotoCoreError, which covers an
expired SSO sign-in and an unreachable endpoint) becomes `AwsUnavailable` with an operator message.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

AUDIT_LOG = "/team13/checkpoint"


class AwsUnavailable(RuntimeError):
    pass


def _explain(exc: Exception) -> str:
    name = type(exc).__name__
    if "SSO" in name or "Token" in name or "ExpiredToken" in str(getattr(exc, "response", "")):
        return "AWS sign-in on this laptop has expired (run: aws sso login --profile fyuz). Chat still works."
    if name in ("NoCredentialsError", "ProfileNotFound", "NoRegionError"):
        return "This laptop has no AWS read access configured (AWS_PROFILE). Chat still works."
    if name == "EndpointConnectionError":
        return "AWS could not be reached from this laptop."
    code = (getattr(exc, "response", None) or {}).get("Error", {}).get("Code")
    if code in ("AccessDeniedException", "AccessDenied", "UnrecognizedClientException"):
        return f"AWS refused the read ({code})."
    return f"AWS read failed ({code or name})."


def message_json(message: Any) -> dict | None:
    if isinstance(message, dict):
        return message
    if not isinstance(message, str):
        return None
    s = message.strip()
    i = s.find("{")
    if i < 0:
        return None
    try:
        d = json.loads(s[i:])
    except ValueError:
        return None
    return d if isinstance(d, dict) else None


def iso_ms(ms: int | float) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class AwsReads:
    def __init__(self, env: dict | None = None, logs=None, ddb=None):
        env = os.environ if env is None else env
        self.region = env.get("ASKNOC_REGION") or "us-east-2"
        self.table = env.get("ASKNOC_APPROVALS_TABLE") or "team13-approvals"
        self.log_group = env.get("ASKNOC_AUDIT_LOG") or AUDIT_LOG
        self._logs, self._ddb = logs, ddb

    def _client(self, name: str):
        try:
            import boto3
            from botocore.config import Config

            cfg = Config(connect_timeout=5, read_timeout=10, retries={"max_attempts": 2})
            return boto3.session.Session(region_name=self.region).client(name, config=cfg)
        except Exception as e:  # noqa: BLE001 - profile/credential setup problems are reported, not raised
            raise AwsUnavailable(_explain(e)) from None

    def _guard(self, fn):
        try:
            from botocore.exceptions import BotoCoreError, ClientError
            errors: tuple = (ClientError, BotoCoreError)
        except ImportError:  # pragma: no cover - boto3 is a dependency
            errors = ()
        try:
            return fn()
        except AwsUnavailable:
            raise
        except errors as e:
            raise AwsUnavailable(_explain(e)) from None

    def filter_events(self, pattern_term: str, start_ms: int, limit: int = 1000) -> list[dict]:
        """Audit-log lines containing the quoted term (ids with '-' must be quoted in a filter pattern)."""
        def run():
            client = self._logs or self._client("logs")
            out, token = [], None
            while len(out) < limit:
                kw = {"logGroupName": self.log_group, "filterPattern": f'"{pattern_term}"',
                      "startTime": int(start_ms), "limit": min(1000, limit - len(out))}
                if token:
                    kw["nextToken"] = token
                page = client.filter_log_events(**kw)
                out += page.get("events") or []
                nxt = page.get("nextToken")
                if not nxt or nxt == token:
                    break
                token = nxt
            return out
        return self._guard(run)

    def get_row(self, pk: str) -> dict | None:
        def run():
            client = self._ddb or self._client("dynamodb")
            item = client.get_item(TableName=self.table, Key={"pk": {"S": pk}}, ConsistentRead=True).get("Item")
            if not item:
                return None
            return {k: next(iter(v.values())) for k, v in item.items() if isinstance(v, dict) and v}
        return self._guard(run)

    def probe(self) -> None:
        """One cheap read of the audit log; raises AwsUnavailable when reads don't work."""
        def run():
            client = self._logs or self._client("logs")
            client.filter_log_events(logGroupName=self.log_group, limit=1)
        self._guard(run)
