"""Kit settings for Ask the NOC, read at call time from the folder named by ASKNOC_KIT_DIR.

Read: kit-state.json (gateway_url; issuer, gateway_arn and builder_host only to build the redaction set),
tokens/team_app.txt (the app's token). From kit.json ONLY the `builder_host` string is kept (redaction);
the rest of that file, which holds the A2A token, is dropped as soon as it is parsed. Nothing else in the
kit folder is opened. Secrets are wrapped in `Secret`, whose repr is `<redacted>`, and never logged.

A file is re-read when its mtime or size changes, so a kit `deploy` that re-issues tokens is picked up
without a restart. kit-state.json is rewritten non-atomically by the kit (checkpoint_kit.py L43-45): a
half-written file keeps the previous good values and marks the view `stale`.
"""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

TOKEN_FILE = "tokens/team_app.txt"
STATE_FILE = "kit-state.json"
KIT_FILE = "kit.json"


class Secret:
    """A value that must never reach a log line, an exception text or a response."""

    __slots__ = ("_v",)

    def __init__(self, value: str):
        self._v = value

    def reveal(self) -> str:
        return self._v

    def __repr__(self) -> str:
        return "<redacted>"

    __str__ = __repr__


class NotConfigured(RuntimeError):
    pass


@dataclass
class KitView:
    configured: bool
    problem: str | None = None
    gateway_url: Secret | None = None
    token: Secret | None = None
    claims: dict = field(default_factory=dict)  # non-secret claims only: role, agent, aud, exp
    redact_values: tuple[str, ...] = field(default=(), repr=False)
    real_gateway: bool = False  # https on an *.amazonaws.com host
    stale: bool = False

    def token_seconds_left(self, now: float | None = None) -> float | None:
        exp = self.claims.get("exp")
        if not isinstance(exp, (int, float)):
            return None
        return float(exp) - (time.time() if now is None else now)


def _claims(token: str) -> dict:
    """Payload claims of a JWT, NOT verified (the Gateway verifies). Only non-secret keys are kept."""
    try:
        part = token.split(".")[1]
        data = json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4)))
    except (IndexError, ValueError):
        return {}
    return {k: data.get(k) for k in ("role", "agent", "aud", "exp") if k in data}


def _host(url: str) -> str:
    try:
        return urlsplit(url).hostname or ""
    except ValueError:
        return ""


class KitConfig:
    def __init__(self, kit_dir: str | None = None, env: dict | None = None):
        env = os.environ if env is None else env
        raw = kit_dir or env.get("ASKNOC_KIT_DIR") or ""
        self.dir = Path(raw).expanduser() if raw else None
        self.extra_hosts = tuple(h.strip() for h in (env.get("ASKNOC_REDACT_HOSTS") or "").split(",") if h.strip())
        self._cache: dict[str, tuple[tuple[int, int], Any]] = {}
        self._good_state: dict | None = None

    def _read(self, name: str, parse):
        p = self.dir / name
        st = p.stat()
        stamp = (st.st_mtime_ns, st.st_size)
        hit = self._cache.get(name)
        if hit and hit[0] == stamp:
            return hit[1]
        value = parse(p.read_text())
        self._cache[name] = (stamp, value)
        return value

    @staticmethod
    def _parse_state(text: str) -> dict:
        d = json.loads(text)
        if not isinstance(d, dict):
            raise ValueError("kit-state.json is not an object")
        return {k: d.get(k) for k in ("gateway_url", "issuer", "gateway_arn") if isinstance(d.get(k), str)}

    @staticmethod
    def _parse_kit(text: str) -> str:
        d = json.loads(text)  # the dict (with the A2A token) is dropped when this returns
        host = d.get("builder_host") if isinstance(d, dict) else None
        return host if isinstance(host, str) else ""

    def load(self) -> KitView:
        if self.dir is None:
            return KitView(False, "ASKNOC_KIT_DIR is not set on this laptop.")
        if not self.dir.is_dir():
            return KitView(False, "The kit folder named by ASKNOC_KIT_DIR is not readable.")
        stale = False
        try:
            state = self._read(STATE_FILE, self._parse_state)
            self._good_state = state
        except FileNotFoundError:
            return KitView(False, "kit-state.json is missing from the kit folder (has the kit been deployed?).")
        except (ValueError, OSError):
            if self._good_state is None:
                return KitView(False, "kit-state.json could not be read.")
            state, stale = self._good_state, True  # mid-write by a kit deploy: keep the last good values
        try:
            token = self._read(TOKEN_FILE, lambda t: t.strip())
        except (FileNotFoundError, OSError):
            return KitView(False, "The app token (tokens/team_app.txt) is missing from the kit folder.")
        try:
            builder = self._read(KIT_FILE, self._parse_kit)
        except (FileNotFoundError, ValueError, OSError):
            builder = ""
        url = state.get("gateway_url") or ""
        if not url:
            return KitView(False, "kit-state.json has no gateway_url.")
        values = {url, _host(url), token, builder, builder.split(":")[0], *self.extra_hosts}
        issuer = state.get("issuer") or ""
        values |= {issuer, _host(issuer)}
        arn = state.get("gateway_arn") or ""
        parts = arn.split(":")
        if len(parts) > 4 and parts[4].isdigit():
            values.add(parts[4])
        host = _host(url)
        return KitView(
            True, None, Secret(url), Secret(token), _claims(token),
            tuple(sorted((v for v in values if v and len(v) >= 4), key=len, reverse=True)),
            real_gateway=url.startswith("https://") and host.endswith(".amazonaws.com"), stale=stale)
