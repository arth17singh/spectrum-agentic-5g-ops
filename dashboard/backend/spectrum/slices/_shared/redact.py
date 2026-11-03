"""Redact AWS account ids, bucket names, ARNs and local absolute paths from slice API responses.

The real evidence files name the team bucket (it embeds the account id), job and role ARNs and container
paths. The screens may show those files, the UI must not show the account. So every JSON response under a
slice prefix passes through `RedactMiddleware`: inside JSON *strings* only (numbers, keys and hashes are
untouched),
  * any standalone 12-digit number (an account id: in ARNs, bucket names, ECR hosts) becomes "<acct>";
  * s3://<bucket>/ becomes s3://<bucket>/ with the account redacted as above (bucket names keep their shape);
  * an absolute local path that starts with the user's home directory becomes "~/...";
  * an email, or a person's STS / IAM ARN (e.g. an approval's approved_by), becomes "person:team13-op1";
    a service-role ARN becomes "iam-role:<name>".
A 64-hex sha256 cannot contain a standalone 12-digit run (it is bounded by hex characters), so hashes
shown on screen stay byte-exact.
"""

from __future__ import annotations

import json
import os
import re

ACCOUNT = re.compile(r"(?<![0-9A-Za-z])\d{12}(?![0-9A-Za-z])")
PERSON = "person:team13-op1"
# A person's AWS identity (STS assumed-role session, IAM user) or an email: audit evidence in the records,
# never shown in the UI. Service roles keep their name only, without the ARN.
PERSON_ARN = re.compile(r"arn:aws[a-z-]*:(?:sts::[^:\s\"]*:assumed-role|iam::[^:\s\"]*:user|sts::[^:\s\"]*:federated-user)/[^\s\"',;)]*")
ROLE_ARN = re.compile(r"arn:aws[a-z-]*:iam::[^:\s\"]*:role/((?:[\w+=,.@-]+/)*)([\w+=,.@-]+)")
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
HOME = os.path.expanduser("~")
PREFIXES = ("/api/scorecard", "/api/trace", "/api/trustgate", "/api/twinview", "/api/realrun", "/api/evidence")


def redact_str(s: str) -> str:
    if HOME and HOME != "/" and HOME in s:
        s = s.replace(HOME, "~")
    if "arn:" in s:
        s = PERSON_ARN.sub(PERSON, s)
        s = ROLE_ARN.sub(lambda m: f"iam-role:{m.group(2)}", s)
    if "@" in s:
        s = EMAIL.sub(PERSON, s)
    return ACCOUNT.sub("<acct>", s) if any(c.isdigit() for c in s) else s


def redact(obj):
    if isinstance(obj, str):
        return redact_str(obj)
    if isinstance(obj, list):
        return [redact(x) for x in obj]
    if isinstance(obj, dict):
        return {k: redact(v) for k, v in obj.items()}
    return obj


class RedactMiddleware:
    """Pure ASGI: buffers JSON responses on the slice prefixes and rewrites their strings."""

    def __init__(self, app, prefixes: tuple[str, ...] = PREFIXES):
        self.app = app
        self.prefixes = prefixes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not scope.get("path", "").startswith(self.prefixes):
            return await self.app(scope, receive, send)
        start: dict = {}
        chunks: list[bytes] = []

        async def wrapped(message):
            if message["type"] == "http.response.start":
                start.update(message)
                return
            if message["type"] == "http.response.body":
                chunks.append(message.get("body", b""))
                if message.get("more_body"):
                    return
                body = b"".join(chunks)
                headers = [(k, v) for k, v in start.get("headers", []) if k.lower() != b"content-length"]
                ctype = dict(start.get("headers", [])).get(b"content-type", b"")
                if ctype.startswith(b"application/json"):
                    try:
                        body = json.dumps(redact(json.loads(body)), ensure_ascii=False, allow_nan=False,
                                          separators=(",", ":")).encode("utf-8")
                    except (ValueError, RecursionError):
                        pass  # not JSON after all: pass it through unchanged
                headers.append((b"content-length", str(len(body)).encode()))
                await send({**start, "headers": headers})
                await send({"type": "http.response.body", "body": body})
                return
            await send(message)

        await self.app(scope, receive, wrapped)
