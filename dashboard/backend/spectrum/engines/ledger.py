"""Tamper-evident audit ledger.

Real cryptography, even in mock mode:
  hash_n  = SHA-256( canonical(body_n) || hash_{n-1} )
  sig_n   = Ed25519_sign(key_id, hash_n)
  anchor  = Merkle root over every 32 record hashes, written to "S3 Object Lock".

AWS mode: records -> DynamoDB, anchors -> S3 PutObject with ObjectLockMode=COMPLIANCE,
signatures -> KMS Sign (ECC_NIST_P256 / or Ed25519 where available).
"""

from __future__ import annotations

import copy
import hashlib
import time
from datetime import datetime, timedelta

from ..core import Bus, Clock, KeyRing, canonical, normalize, sha256_hex

GENESIS = "0" * 64
ANCHOR_EVERY = 32
ANCHOR_BUCKET = "spectrum-ledger-anchors-us-west-2"


def merkle_levels(leaves: list[str]) -> list[list[str]]:
    levels = [leaves]
    cur = leaves
    while len(cur) > 1:
        nxt = []
        for i in range(0, len(cur), 2):
            left = cur[i]
            right = cur[i + 1] if i + 1 < len(cur) else cur[i]
            nxt.append(hashlib.sha256(bytes.fromhex(left) + bytes.fromhex(right)).hexdigest())
        levels.append(nxt)
        cur = nxt
    return levels


class Ledger:
    def __init__(self, clock: Clock, bus: Bus, keys: KeyRing):
        self.clock = clock
        self.bus = bus
        self.keys = keys
        self.records: list[dict] = []
        self.anchors: list[dict] = []
        self.originals: dict[int, dict] = {}  # seq -> original record before tamper
        self.last_verify: dict | None = None

    # -- writing ---------------------------------------------------------------

    @staticmethod
    def body_of(rec: dict) -> dict:
        return {k: rec[k] for k in ("seq", "ts", "type", "actor", "plane", "payload", "key_id")}

    def append(self, type: str, actor: str, plane: str, payload: dict, key_id: str | None = None,
               ts: str | None = None, emit: bool = True) -> dict:
        seq = len(self.records) + 1
        prev = self.records[-1]["hash"] if self.records else GENESIS
        key_id = key_id or f"kms:spectrum/{actor}"
        rec = {
            "seq": seq,
            "ts": ts or self.clock.iso(),
            "type": type,
            "actor": actor,
            "plane": plane,
            "payload": normalize(payload),
            "key_id": key_id,
        }
        h = sha256_hex(canonical(rec) + prev)
        rec["prev_hash"] = prev
        rec["hash"] = h
        rec["alg"] = "Ed25519"
        rec["sig"] = self.keys.sign(key_id, h)
        self.records.append(rec)
        if emit:
            self.bus.emit("ledger.append", "prove", f"#{seq} {type}", {"record": rec}, "debug", actor)
        if seq % ANCHOR_EVERY == 0:
            self._anchor(emit)
        return rec

    def _anchor(self, emit: bool) -> None:
        n = len(self.anchors) + 1
        start = (n - 1) * ANCHOR_EVERY + 1
        end = n * ANCHOR_EVERY
        leaves = [self.records[i - 1]["hash"] for i in range(start, end + 1)]
        root = merkle_levels(leaves)[-1][0]
        written = self.records[end - 1]["ts"]
        anchor = {"n": n, "range": [start, end], "root": root, **self._anchor_location(n, root, written),
                  "written_at": written}
        self.anchors.append(anchor)
        if emit:
            self.bus.emit("ledger.anchor", "prove", f"anchor #{n} records {start}-{end}", {"anchor": anchor},
                          "info", "ledger")

    def _anchor_location(self, n: int, root: str, written: str) -> dict:
        """Where an anchor is said to be stored. The demo ledger's S3 Object Lock fields are
        simulated; a subclass that persists anchors elsewhere overrides this."""
        retain = (datetime.fromisoformat(written) + timedelta(days=2555)).date().isoformat()
        return {
            "s3_uri": f"s3://{ANCHOR_BUCKET}/anchors/{n:06d}.json",
            "version_id": sha256_hex(f"v{n}{root}")[:32],
            "etag": sha256_hex(root)[:32],
            "object_lock": "COMPLIANCE",
            "retain_until": retain,
        }

    # -- verification ----------------------------------------------------------

    def verify_record(self, rec: dict, prev_hash: str) -> dict:
        recomputed = sha256_hex(canonical(self.body_of(rec)) + prev_hash)
        checks = {
            "hash_matches": recomputed == rec["hash"],
            "link_matches": rec["prev_hash"] == prev_hash,
            "signature_valid": self.keys.verify(rec["key_id"], rec["hash"], rec["sig"]),
        }
        return {"ok": all(checks.values()), "checks": checks, "recomputed": recomputed}

    def verify(self, emit: bool = True) -> dict:
        t0 = time.perf_counter()
        prev = GENESIS
        broken = None
        for rec in self.records:
            res = self.verify_record(rec, prev)
            if not res["ok"]:
                orig = self.originals.get(rec["seq"])
                broken = {
                    "seq": rec["seq"],
                    "checks": res["checks"],
                    "stored_hash": rec["hash"],
                    "recomputed_hash": res["recomputed"],
                    "diff": _diff(orig["payload"], rec["payload"]) if orig else [],
                    "anchor": self.anchor_for(rec["seq"]),
                }
                break
            prev = rec["hash"]
        anchors_ok = all(self._anchor_ok(a) for a in self.anchors)
        result = {
            "ok": broken is None,
            "checked": broken["seq"] if broken else len(self.records),
            "total": len(self.records),
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
            "broken": broken,
            "anchors_checked": len(self.anchors),
            "anchors_ok": anchors_ok,
            "at": self.clock.iso(),
        }
        self.last_verify = result
        if emit:
            if result["ok"]:
                self.bus.emit("ledger.verify", "prove", f"chain verified {result['total']:,} records", result,
                              "notice", "auditor")
            else:
                self.bus.emit("ledger.verify", "prove",
                              f"chain broken at #{broken['seq']}: payload modified after signing", result,
                              "critical", "auditor")
        return result

    def _anchor_ok(self, a: dict) -> bool:
        s, e = a["range"]
        if e > len(self.records):
            return False
        return merkle_levels([self.records[i - 1]["hash"] for i in range(s, e + 1)])[-1][0] == a["root"]

    def anchor_for(self, seq: int) -> dict | None:
        n = (seq - 1) // ANCHOR_EVERY
        return self.anchors[n] if n < len(self.anchors) else None

    def proof(self, seq: int) -> dict:
        anchor = self.anchor_for(seq)
        rec = self.records[seq - 1]
        if not anchor:
            return {"seq": seq, "anchored": False, "pending_until": ((seq - 1) // ANCHOR_EVERY + 1) * ANCHOR_EVERY}
        s, e = anchor["range"]
        # The proof is computed from the hash the record *should* have (recomputed from its
        # current body), which is exactly what an auditor does. A tampered record therefore
        # yields a root that no longer matches the immutable anchor.
        prev = self.records[seq - 2]["hash"] if seq > 1 else GENESIS
        leaf = sha256_hex(canonical(self.body_of(rec)) + prev)
        leaves = [self.records[i - 1]["hash"] for i in range(s, e + 1)]
        idx = seq - s
        leaves[idx] = leaf
        levels = merkle_levels(leaves)
        path = []
        i = idx
        for level in levels[:-1]:
            sib = i ^ 1
            sib_hash = level[sib] if sib < len(level) else level[i]
            path.append({"hash": sib_hash, "side": "left" if sib < i else "right"})
            i //= 2
        root = levels[-1][0]
        return {"seq": seq, "anchored": True, "leaf": leaf, "index": idx, "path": path,
                "computed_root": root, "anchor": anchor, "matches": root == anchor["root"]}

    # -- demo: tamper ----------------------------------------------------------

    def tamper(self, seq: int | None = None) -> dict:
        if seq is None:
            # pick the most recent anchored human approval, else an anchored tool call
            cands = [r for r in self.records if r["type"] == "approval.decision" and self.anchor_for(r["seq"])]
            cands = cands or [r for r in self.records if r["type"] == "tool.call" and self.anchor_for(r["seq"])]
            seq = cands[-1]["seq"]
        rec = self.records[seq - 1]
        self.originals.setdefault(seq, copy.deepcopy(rec))
        self.last_verify = None
        p = rec["payload"]
        # change one meaningful field, as an attacker with database access would
        if "decision" in p:
            p["decision"] = "rejected" if p["decision"] == "approved" else "approved"
            field = "decision"
        elif "params" in p and isinstance(p["params"], dict):
            p["params"]["blast_radius_pct"] = "2"
            field = "params.blast_radius_pct"
        else:
            p["tampered"] = True
            field = "tampered"
        self.bus.emit("ledger.tamper", "prove", f"record #{seq} modified directly in the datastore",
                      {"seq": seq, "field": field}, "warn", "attacker")
        return {"seq": seq, "field": field}

    def restore(self) -> None:
        self.last_verify = None
        for seq, rec in self.originals.items():
            self.records[seq - 1] = rec
        self.originals.clear()

    # -- views -----------------------------------------------------------------

    def snapshot(self, tail: int = 150) -> dict:
        return {
            "height": len(self.records),
            "head": self.records[-1]["hash"] if self.records else GENESIS,
            "records": self.records[-tail:],
            "anchors": self.anchors[-60:],
            "anchors_total": len(self.anchors),
            "last_verify": self.last_verify,
            "tampered": sorted(self.originals.keys()),
            "type_counts": _counts(self.records),
        }

    def public_key(self, key_id: str) -> dict:
        return {"key_id": key_id, "public_key_hex": self.keys.public_hex(key_id),
                "fingerprint": self.keys.fingerprint(key_id), "revoked_at": self.keys.revoked.get(key_id)}


def _counts(records: list[dict]) -> dict:
    out: dict[str, int] = {}
    for r in records:
        out[r["type"]] = out.get(r["type"], 0) + 1
    return out


def _diff(a, b, path: str = "") -> list[dict]:
    out: list[dict] = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            out += _diff(a.get(k), b.get(k), f"{path}.{k}" if path else k)
    elif a != b:
        out.append({"path": path, "before": a, "after": b})
    return out
