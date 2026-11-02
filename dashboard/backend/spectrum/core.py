"""Core primitives shared by every engine: canonical JSON, keys, sim clock, event bus, scheduler."""

from __future__ import annotations

import asyncio
import hashlib
import heapq
import hmac
import itertools
import json
import os
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

# ---------------------------------------------------------------------------
# Canonical JSON
#
# Ledger payloads must hash identically in Python and in the browser (WebCrypto).
# Rules: sorted keys, no whitespace, UTF-8, and NO floats - floats are converted to
# fixed-precision strings before storage, so both sides only ever see ints/strings.
# The browser twin of this function lives in frontend/src/core/canonical.ts and is
# tested against tests/fixtures/canonical.json.
# ---------------------------------------------------------------------------


def normalize(obj: Any) -> Any:
    if isinstance(obj, bool) or obj is None or isinstance(obj, int) or isinstance(obj, str):
        return obj
    if isinstance(obj, float):
        return f"{obj:.6g}"
    if isinstance(obj, dict):
        return {str(k): normalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [normalize(v) for v in obj]
    if hasattr(obj, "item"):  # numpy scalar
        return normalize(obj.item())
    raise TypeError(f"cannot canonicalize {type(obj).__name__}")


def canonical(obj: Any) -> str:
    return json.dumps(normalize(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Keys (mock KMS). Each Ed25519 private key is HMAC-SHA256(_KEY_SECRET, key_id).
# The simulation seed must not be used for this: it is public (GET /api/config, the
# Settings screen), and keys derived from it would let anyone forge ledger records,
# human approvals, A2A messages and AI-BOM signatures.
# _KEY_SECRET is the bytes of SPECTRUM_KEY_SECRET if set and non-empty, else 32
# random bytes. It is made once at import and is module-level, not per KeyRing, so
# the new KeyRing that every Engine.reset() builds derives the same keys, and resets
# keep producing identical signatures for the whole server run. Set
# SPECTRUM_KEY_SECRET to keep the same keys across restarts.
# AWS mode swaps this for KMS Sign/Verify.
# ---------------------------------------------------------------------------

_KEY_SECRET: bytes = os.environ.get("SPECTRUM_KEY_SECRET", "").encode() or os.urandom(32)


class KeyRing:
    def __init__(self, secret: bytes | None = None) -> None:
        # `secret` lets a separate ledger (the evidence ledger) use its own persistent secret;
        # the demo engine passes nothing and keeps the module-level per-run secret.
        self._secret = secret or _KEY_SECRET
        self._keys: dict[str, Ed25519PrivateKey] = {}
        self.revoked: dict[str, str] = {}  # key_id -> iso time

    def _key(self, key_id: str) -> Ed25519PrivateKey:
        k = self._keys.get(key_id)
        if k is None:
            raw = hmac.digest(self._secret, key_id.encode(), "sha256")
            k = Ed25519PrivateKey.from_private_bytes(raw)
            self._keys[key_id] = k
        return k

    def sign(self, key_id: str, message: str) -> str:
        return self._key(key_id).sign(message.encode()).hex()

    def verify(self, key_id: str, message: str, sig_hex: str) -> bool:
        try:
            self._key(key_id).public_key().verify(bytes.fromhex(sig_hex), message.encode())
            return True
        except Exception:
            return False

    def public_hex(self, key_id: str) -> str:
        return self._key(key_id).public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()

    def fingerprint(self, key_id: str) -> str:
        return "SHA256:" + sha256_hex(bytes.fromhex(self.public_hex(key_id)))[:32]

    def revoke(self, key_id: str, at: str) -> None:
        self.revoked[key_id] = at


# ---------------------------------------------------------------------------
# Simulation clock. The demo story happens on Saturday 31 Oct 2026, 21:30 PDT.
# ---------------------------------------------------------------------------

SIM_EPOCH = datetime(2026, 10, 31, 21, 30, 0, tzinfo=timezone(timedelta(hours=-7)))


class Clock:
    def __init__(self) -> None:
        self.t = 0.0  # sim seconds since SIM_EPOCH
        self.speed = 1.0
        self.paused = False

    def advance(self, real_dt: float) -> float:
        if self.paused:
            return 0.0
        dt = real_dt * self.speed
        self.t += dt
        return dt

    def iso(self, t: float | None = None) -> str:
        t = self.t if t is None else t
        return (SIM_EPOCH + timedelta(seconds=t)).isoformat(timespec="milliseconds")


# ---------------------------------------------------------------------------
# Event bus: every engine emits envelopes here; the WebSocket layer fans out.
# ---------------------------------------------------------------------------

SEVERITIES = ("debug", "info", "notice", "warn", "critical")


class Bus:
    def __init__(self, clock: Clock, history: int = 20000):
        self.clock = clock
        self.seq = 0
        self.history: deque[dict] = deque(maxlen=history)
        self.subscribers: set[asyncio.Queue] = set()
        self.muted = False  # used while warming up (history is still recorded)

    def emit(
        self,
        type: str,
        plane: str,
        summary: str = "",
        data: dict | None = None,
        severity: str = "info",
        actor: str = "system",
        ledger: dict | None = None,
        record: bool = True,
    ) -> dict:
        self.seq += 1
        ev = {
            "seq": self.seq,
            "ts": self.clock.iso(),
            "sim_t": round(self.clock.t, 3),
            "plane": plane,
            "type": type,
            "severity": severity,
            "actor": actor,
            "summary": summary,
            "data": data or {},
        }
        if ledger:
            ev["ledger"] = ledger
        if record:
            self.history.append(ev)
        if not self.muted:
            for q in list(self.subscribers):
                try:
                    q.put_nowait(ev)
                except asyncio.QueueFull:
                    pass
        return ev

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=50000)
        self.subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self.subscribers.discard(q)


# ---------------------------------------------------------------------------
# Scheduler for scripted sequences (sim-time based).
# ---------------------------------------------------------------------------


class Scheduler:
    def __init__(self, clock: Clock):
        self.clock = clock
        self._q: list[tuple[float, int, Callable[[], None], str]] = []
        self._ids = itertools.count()

    def after(self, delay: float, fn: Callable[[], None], tag: str = "") -> None:
        heapq.heappush(self._q, (self.clock.t + delay, next(self._ids), fn, tag))

    def cancel(self, tag: str) -> None:
        self._q = [item for item in self._q if item[3] != tag]
        heapq.heapify(self._q)

    def run_due(self) -> None:
        while self._q and self._q[0][0] <= self.clock.t:
            _, _, fn, _ = heapq.heappop(self._q)
            fn()


def r(x: float, nd: int = 3) -> float:
    return float(round(float(x), nd))
