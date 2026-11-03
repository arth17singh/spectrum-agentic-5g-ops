"""Redact identities, ARNs, secret names, registry hosts and local paths from every /api/* JSON response.

The real evidence files keep, as audit evidence, the approver's email and STS identity, job and role ARNs,
the Secrets Manager ARNs a job read its twin token and URL from, ECR image hosts, the team bucket (its name
embeds the account id) and container paths. The screens may show those files; the UI must not show any of
that. So every JSON response under /api/ passes through `RedactMiddleware`, which rewrites JSON *strings*
(values and keys; numbers are untouched):
  * an email, or a person's STS / IAM user ARN            -> "person:team13-op1"
  * a Secrets Manager ARN, or a secret name we create      -> "<secret>"
  * a SageMaker job ARN                                    -> "<arn:sagemaker>/<job name>"
  * an IAM role ARN                                        -> "<arn:iam>/role/<name>"
  * any other ARN                                          -> "<arn:SERVICE>"
  * <account>.dkr.ecr.<region>.amazonaws.com               -> "<ecr>"
  * a standalone 12-digit number (an account id)           -> "<acct>"
  * the user's home directory in a path                    -> "~"
Signed ledger records (a dict with both "hash" and "sig") are passed through untouched: the browser
re-verifies their hashes and signatures, and their payloads are our own metadata, not record content.
/api/evidence/chain (the raw signed chain, tens of MB) is not rewritten for the same reason.
A 64-hex sha256 cannot contain a standalone 12-digit run, so hashes shown on screen stay byte-exact.
"""

from __future__ import annotations

import asyncio
import json
import os
import re

PERSON = "person:team13-op1"
ACCOUNT = re.compile(r"(?<![0-9A-Za-z.])\d{12}(?![0-9A-Za-z])")  # not the fraction of a decimal
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
ARN = re.compile(r"arn:aws[a-z-]*:([a-z0-9-]+):[a-z0-9-]*:[^:\s\"']*:([^\s\"',;)]*)")
ECR = re.compile(r"[0-9<>a-z]+\.dkr\.ecr\.[a-z0-9-]+\.amazonaws\.com(?:\.cn)?")
SECRET_NAME = re.compile(r"AmazonSageMaker-[A-Za-z0-9/_+=.@-]+|team13[-/][A-Za-z0-9/_+=.@-]*(?:token|secret|password|url)[A-Za-z0-9/_+=.@-]*",
                         re.IGNORECASE)
HOME = os.path.expanduser("~")
PREFIXES = ("/api/",)
SKIP = ("/api/evidence/chain",)


def _arn(m: re.Match) -> str:
    service, resource = m.group(1), m.group(2)
    if service == "sts" or (service == "iam" and resource.startswith(("user/", "federated-user/"))):
        return PERSON
    if service == "secretsmanager":
        return "<secret>"
    if service == "sagemaker" and "/" in resource:
        return f"<arn:sagemaker>/{resource.split('/', 1)[1]}"
    if service == "iam" and resource.startswith("role/"):
        return f"<arn:iam>/role/{resource.rsplit('/', 1)[-1]}"
    return f"<arn:{service}>"


def redact_str(s: str) -> str:
    if HOME and HOME != "/" and HOME in s:
        s = s.replace(HOME, "~")
    if "arn:" in s:
        s = ARN.sub(_arn, s)
    if "@" in s:
        s = EMAIL.sub(PERSON, s)
    if ".dkr.ecr." in s:
        s = ECR.sub("<ecr>", s)
    if "AmazonSageMaker-" in s or "team13" in s:
        s = SECRET_NAME.sub("<secret>", s)
    return ACCOUNT.sub("<acct>", s) if any(c.isdigit() for c in s) else s


def redact(obj):
    if isinstance(obj, str):
        return redact_str(obj)
    if isinstance(obj, list):
        return [redact(x) for x in obj]
    if isinstance(obj, dict):
        if "hash" in obj and "sig" in obj:
            return obj  # a signed ledger record: verified byte-exact by the browser
        return {redact_str(k) if isinstance(k, str) else k: redact(v) for k, v in obj.items()}
    return obj


def _rewrite(body: bytes) -> bytes:
    return json.dumps(redact(json.loads(body)), ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


class RedactMiddleware:
    """Pure ASGI: buffers JSON responses under /api/ (except the raw signed chain) and rewrites their strings."""

    def __init__(self, app, prefixes: tuple[str, ...] = PREFIXES, skip: tuple[str, ...] = SKIP):
        self.app = app
        self.prefixes = prefixes
        self.skip = skip

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        if scope["type"] != "http" or not path.startswith(self.prefixes) or path.startswith(self.skip):
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
                    try:  # large bodies are rewritten off the event loop
                        body = await asyncio.to_thread(_rewrite, body) if len(body) > 65536 else _rewrite(body)
                    except (ValueError, RecursionError):
                        pass  # not JSON after all: pass it through unchanged
                headers.append((b"content-length", str(len(body)).encode()))
                await send({**start, "headers": headers})
                await send({"type": "http.response.body", "body": body})
                return
            await send(message)

        await self.app(scope, receive, wrapped)
