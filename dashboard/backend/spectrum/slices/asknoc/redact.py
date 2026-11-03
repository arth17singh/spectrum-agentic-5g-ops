"""Everything from the kit that goes to the browser passes through `Redactor.clean`.

Exact values first (the Gateway URL and host, the issuer, the token, the account id, the Agent Builder host),
then patterns for anything shaped like an address, a key or an account. The Gateway runs with
exceptionLevel DEBUG and the tools Lambda returns str(exc) (checkpoint_kit.py L274-276), so error text can
carry hosts (an SSL error names the host it could not verify). Every replacement is counted, so the page can
say the text was altered.
"""

from __future__ import annotations

import re

MARK = "[redacted]"

_HEX = re.compile(r"[0-9a-f]+")
PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"eyJ[\w-]+\.[\w-]+\.[\w-]*"),                      # JWT
    re.compile(r"https?://[^\s\"'<>]+", re.I),                      # URL
    re.compile(r"arn:aws[\w-]*:[^\s\"']+"),                         # ARN
    re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),                    # email
    re.compile(r"\b[\w.-]+\.(?:nip|sslip)\.io\b", re.I),            # wildcard-DNS host
    re.compile(r"\b[\w.-]+\.(?:amazonaws\.com|on\.aws)\b", re.I),   # AWS host
    re.compile(r"\b[\w-]+(?:\.[\w-]+)+:\d{2,5}\b"),                 # host:port
    re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),                     # IPv4
    re.compile(r"\b\d{1,3}(?:-\d{1,3}){3}\b"),                      # dashed IPv4 (as in some host names)
    re.compile(r"(?<![\w-])\d{12}(?!\w)"),                          # account id (not inside case-/chg- ids)
)
_LONG = re.compile(r"[A-Za-z0-9+/]{40,}={0,2}")                     # key-like run; pure lowercase hex is kept


class Redactor:
    def __init__(self, exact: tuple[str, ...] = ()):
        self.exact = tuple(sorted({v for v in exact if v and len(v) >= 4}, key=len, reverse=True))

    def clean(self, text: str | None) -> tuple[str, int]:
        if not text:
            return "", 0
        n = 0
        out = text
        for v in self.exact:
            if v in out:
                n += out.count(v)
                out = out.replace(v, MARK)
        for p in PATTERNS:
            out, k = p.subn(MARK, out)
            n += k

        def long(m: re.Match) -> str:
            nonlocal n
            if _HEX.fullmatch(m.group(0)):
                return m.group(0)
            n += 1
            return MARK

        out = _LONG.sub(long, out)
        return out, n

    def text(self, value: str | None) -> str:
        return self.clean(value)[0]

    def finds(self, text: str) -> bool:
        return self.clean(text)[1] > 0
