"""The evidence ledger: real records only, persisted, signed with a key that survives restarts.

It reuses the demo ledger's code unchanged (canonical JSON, SHA-256 chain, Ed25519 signatures,
Merkle anchors every 32 records). What differs:
  * it is owned by EvidenceService in main.py, never by Engine, so reset() and the beats cannot touch it;
  * every record and anchor is appended to JSONL files in the evidence directory;
  * the signing key is HMAC(secret, key_id) like the demo keys, but the secret lives in a 0600 file
    outside evidence/ and the public key is written to evidence/public_keys.json;
  * loaded files are verified against that public key file only, so anyone can repeat the check
    without the secret, and a broken chain is reported, never repaired.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from ..core import sha256_hex
from ..engines.ledger import ANCHOR_EVERY, GENESIS, Ledger, merkle_levels

log = logging.getLogger("spectrum.evidence")

REPO_ROOT = Path(__file__).resolve().parents[4]
BACKEND_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DIR = REPO_ROOT / "evidence"
DEFAULT_KEY_FILE = BACKEND_DIR / ".evidence-key"

KEY_ID = "evidence/ingester"
ACTOR = "evidence-ingester"
LEDGER_FILE = "ledger.jsonl"
ANCHORS_FILE = "anchors.jsonl"
PUBKEYS_FILE = "public_keys.json"
README_FILE = "README.md"
RESERVED = {LEDGER_FILE, ANCHORS_FILE, PUBKEYS_FILE, README_FILE}


def evidence_dir() -> Path:
    return Path(os.environ.get("SPECTRUM_EVIDENCE_DIR") or DEFAULT_DIR)


def key_file() -> Path:
    return Path(os.environ.get("SPECTRUM_EVIDENCE_KEY") or DEFAULT_KEY_FILE)


def utc_iso(dt: datetime | None = None) -> str:
    dt = dt or datetime.now(timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class WallClock:
    """Real UTC time. The evidence ledger never uses the demo's simulated clock."""

    def iso(self) -> str:
        return utc_iso()


class _NoBus:
    """The evidence ledger appends with emit=False; EvidenceService emits evidence.* itself."""

    def emit(self, *a, **k) -> None:
        return None


# --------------------------------------------------------------------------- keys

def load_or_create_secret(path: Path) -> bytes:
    """Read the signing secret, or create it (32 random bytes, hex, mode 0600). Never logged."""
    if path.exists():
        raw = path.read_text().strip()
        try:
            secret = bytes.fromhex(raw)
        except ValueError as e:
            raise RuntimeError(f"evidence key file {path} is not hex") from e
        if len(secret) < 32:
            raise RuntimeError(f"evidence key file {path} is too short")
        if path.stat().st_mode & 0o077:
            log.warning("evidence key file %s is readable by other users; run chmod 600 on it", path)
        return secret
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(secrets.token_hex(32) + "\n")
    log.info("created a new evidence signing key file at %s (mode 0600, gitignored)", path)
    return bytes.fromhex(path.read_text().strip())


def fingerprint(pub_hex: str) -> str:
    return "SHA256:" + sha256_hex(bytes.fromhex(pub_hex))[:32]


class PublicKeyRing:
    """Verify-only stand-in for KeyRing, built from evidence/public_keys.json (no secret needed)."""

    def __init__(self, keys: dict[str, str]):
        self.keys = keys  # key_id -> public key hex
        self.revoked: dict[str, str] = {}

    @classmethod
    def from_file(cls, path: Path) -> "PublicKeyRing":
        data = json.loads(path.read_text())
        return cls({k: v["public_key_hex"] for k, v in data.get("keys", {}).items()})

    def verify(self, key_id: str, message: str, sig_hex: str) -> bool:
        pk = self.keys.get(key_id)
        if not pk:
            return False
        try:
            Ed25519PublicKey.from_public_bytes(bytes.fromhex(pk)).verify(bytes.fromhex(sig_hex), message.encode())
            return True
        except Exception:
            return False

    def public_hex(self, key_id: str) -> str:
        return self.keys.get(key_id, "")

    def fingerprint(self, key_id: str) -> str:
        pk = self.keys.get(key_id)
        return fingerprint(pk) if pk else ""

    def sign(self, key_id: str, message: str) -> str:
        raise RuntimeError("a public key ring cannot sign")


def public_keys_doc(pub_hex: str) -> dict:
    return {
        "about": "Public keys for the SPECTRUM evidence ledger (ledger.jsonl). Verify each record's "
                 "Ed25519 signature over its hash with the key named by its key_id.",
        "alg": "Ed25519",
        "keys": {KEY_ID: {"public_key_hex": pub_hex, "fingerprint": fingerprint(pub_hex), "created_at": utc_iso()}},
    }


# --------------------------------------------------------------------------- ledger

class EvidenceLedger(Ledger):
    """A Ledger whose records and anchors are appended to JSONL files as they are made."""

    def __init__(self, directory: Path, keys) -> None:
        super().__init__(WallClock(), _NoBus(), keys)  # type: ignore[arg-type]
        self.dir = directory
        # Hook for an S3 copy of every line (out of scope for now): called as mirror(filename, line).
        self.mirror = None

    def _anchor_location(self, n: int, root: str, written: str) -> dict:
        return {"store": "local file", "path": ANCHORS_FILE}

    def append_persisted(self, type: str, payload: dict) -> dict:
        n_anchors = len(self.anchors)
        rec = self.append(type, ACTOR, "prove", payload, key_id=KEY_ID, emit=False)
        try:
            self._write(LEDGER_FILE, rec)
            if len(self.anchors) > n_anchors:
                self._write(ANCHORS_FILE, self.anchors[-1])
        except OSError:
            # keep memory equal to disk: the record does not exist if it could not be written
            self.records.pop()
            del self.anchors[n_anchors:]
            raise
        return rec

    def _write(self, name: str, obj: dict) -> None:
        line = json.dumps(obj, sort_keys=True, ensure_ascii=False) + "\n"
        with open(self.dir / name, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        if self.mirror:
            self.mirror(name, line)


def read_jsonl(path: Path) -> tuple[list[dict], int | None]:
    """All parsed lines, and the 1-based line number of the first unparseable one (reading stops there)."""
    if not path.exists():
        return [], None
    out: list[dict] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError("not an object")
        except ValueError:
            return out, i
        out.append(obj)
    return out, None


def _reason(checks: dict) -> str:
    if not checks.get("hash_matches"):
        return "contents changed after signing (hash mismatch)"
    if not checks.get("link_matches"):
        return "link to the previous record is broken"
    return "signature does not verify with the published public key"


def verify_chain(records: list[dict], anchors: list[dict], keys, parse_error_line: int | None = None) -> dict:
    """Walk the chain from genesis with `keys` (normally the PUBLIC key file) and check every anchor."""
    t0 = time.perf_counter()
    checker = Ledger(WallClock(), _NoBus(), keys)  # type: ignore[arg-type]
    prev = GENESIS
    broken = None
    for i, rec in enumerate(records):
        seq = i + 1
        try:
            if rec.get("seq") != seq:
                broken = {"seq": seq, "reason": f"expected record #{seq}, found #{rec.get('seq')} (gap or reorder)"}
                break
            res = checker.verify_record(rec, prev)
        except (KeyError, TypeError, ValueError) as e:
            broken = {"seq": seq, "reason": f"malformed record ({e.__class__.__name__}: {e})"}
            break
        if not res["ok"]:
            broken = {"seq": seq, "reason": _reason(res["checks"]), "checks": res["checks"],
                      "stored_hash": rec.get("hash"), "recomputed_hash": res["recomputed"]}
            break
        prev = rec["hash"]
    if broken is None and parse_error_line is not None:
        broken = {"seq": len(records) + 1, "reason": f"line {parse_error_line} of {LEDGER_FILE} is not valid JSON"}

    anchor_problem = None
    expected = len(records) // ANCHOR_EVERY
    if broken is None:
        if len(anchors) != expected:
            anchor_problem = f"{len(anchors)} anchors on disk, {expected} expected for {len(records)} records"
        for i, a in enumerate(anchors):
            if anchor_problem:
                break
            n = i + 1
            start, end = (n - 1) * ANCHOR_EVERY + 1, n * ANCHOR_EVERY
            try:
                ok = (a.get("n") == n and list(a.get("range", [])) == [start, end] and end <= len(records)
                      and merkle_levels([records[j - 1]["hash"] for j in range(start, end + 1)])[-1][0] == a.get("root"))
            except (KeyError, TypeError, ValueError):
                ok = False
            if not ok:
                anchor_problem = f"anchor #{n} does not match records {start}-{end}"
    return {
        "ok": broken is None and anchor_problem is None,
        "total": len(records),
        "checked": (broken["seq"] - 1) if broken else len(records),
        "broken": broken,
        "anchors_checked": len(anchors),
        "anchors_ok": anchor_problem is None,
        "anchor_problem": anchor_problem,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
        "at": utc_iso(),
    }


def load_disk(directory: Path) -> tuple[list[dict], list[dict], int | None]:
    records, bad_line = read_jsonl(directory / LEDGER_FILE)
    anchors, _ = read_jsonl(directory / ANCHORS_FILE)
    return records, anchors, bad_line
