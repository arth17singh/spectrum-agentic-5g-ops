"""Redact AWS account ids, bucket names, ARNs and local absolute paths from slice API responses.

The real evidence files name the team bucket (it embeds the account id), job and role ARNs and container
paths. The screens may show those files, the UI must not show the account. So every JSON response under a
slice prefix passes through `RedactMiddleware`: inside JSON *strings* only (numbers, keys and hashes are
untouched),
  * any standalone 12-digit number (an account id: in ARNs, bucket names, ECR hosts) becomes "<acct>";
  * s3://<bucket>/ becomes s3://<bucket>/ with the account redacted as above (bucket names keep their shape);
  * an absolute local path that starts with the user's home directory becomes "~/...".
A 64-hex sha256 cannot contain a standalone 12-digit run (it is bounded by hex characters), so hashes
shown on screen stay byte-exact.
"""

from __future__ import annotations

import json
import os
import re

ACCOUNT = re.compile(r"(?<![0-9A-Za-z])\d{12}(?![0-9A-Za-z])")
HOME = os.path.expanduser("~")
PREFIXES = ("/api/scorecard", "/api/trace", "/api/trustgate", "/api/twinview", "/api/realrun")


def redact_str(s: str) -> str:
    if HOME and HOME != "/" and HOME in s:
        s = s.replace(HOME, "~")
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
