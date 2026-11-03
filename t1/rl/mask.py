"""Mask the twin's address (and any other host) in text a job prints or a record stores.

The twin address and token are confidential: they never go into a log line, a record or an artifact.
`Masker(secrets)` replaces each given value (and, for URLs, its host and host:port) with a fixed label,
then masks anything that still looks like a URL, an IPv4 address or a dotted host name.
`MaskingStream` wraps a text stream (sys.stdout / sys.stderr) so everything written through it is masked
line by line, including what a library's logging handler or a traceback writes.
"""

from __future__ import annotations

import io
import re
from typing import Iterable
from urllib.parse import urlsplit

URL_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9+.-]*://[^\s\"'<>]+")
IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b")
# a dotted name ending in a letters-only label (example.internal, x.y.amazonaws.com); never a plain number
HOST_RE = re.compile(r"\b(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,63}(?::\d+)?\b")
# names that are not hosts but look like one (module paths, file names we print on purpose)
NOT_HOSTS = re.compile(r"^(?:ran_rl|t1|t3|tools|team|config|evaluation|baseline|model|metrics|run_metadata|"
                       r"effective_config|reward_plugin|sourcedir|requirements|team13|stable_baselines3|torch|"
                       r"numpy|logging|httpx|httpcore)\.|\.(?:json|jsonl|md|yaml|yml|zip|gz|tar|py|txt|whl|pt|csv)$",
                       re.IGNORECASE)


class Masker:
    def __init__(self, secrets: Iterable[str | None] = (), label: str = "<twin>"):
        vals: set[str] = set()
        for s in secrets:
            if not s:
                continue
            s = s.strip()
            vals.add(s)
            try:
                u = urlsplit(s)
            except ValueError:
                u = None
            if u is not None and u.netloc:
                vals.add(u.netloc)
                host = u.hostname or ""
                if host:
                    vals.add(host)
        # longest first, so a URL is replaced whole before its host
        self.values = sorted((v for v in vals if len(v) >= 4), key=len, reverse=True)
        self.label = label

    def mask(self, text: str) -> str:
        for v in self.values:
            if v in text:
                text = text.replace(v, self.label)
        text = URL_RE.sub("<url>", text)
        text = IPV4_RE.sub("<ip>", text)
        return HOST_RE.sub(lambda m: m.group(0) if NOT_HOSTS.search(m.group(0)) else "<host>", text)


class MaskingStream(io.TextIOBase):
    """A write-only text stream that masks whole lines before passing them on."""

    def __init__(self, inner, masker: Masker):
        self.inner, self.masker, self._buf = inner, masker, ""

    def writable(self) -> bool:
        return True

    def write(self, s: str) -> int:
        self._buf += s
        if "\n" in self._buf:
            head, _, self._buf = self._buf.rpartition("\n")
            self.inner.write(self.masker.mask(head + "\n"))
        return len(s)

    def flush(self) -> None:
        if self._buf:
            self.inner.write(self.masker.mask(self._buf))
            self._buf = ""
        self.inner.flush()

    def isatty(self) -> bool:
        return False

    @property
    def encoding(self):  # noqa: D401 - mirrors the wrapped stream
        return getattr(self.inner, "encoding", "utf-8")
