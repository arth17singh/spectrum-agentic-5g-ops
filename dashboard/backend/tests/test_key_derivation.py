"""Signing keys must not be derivable from the simulation seed.

The seed is public (GET /api/config, the Settings screen, default 7). Keys used to be
SHA-256("{seed}:{key_id}"), so anyone could forge ledger records, human approvals, A2A messages and
AI-BOM signatures. They are now HMAC-SHA256 of the key id under a per-process secret."""

import hashlib
import hmac
import os
import subprocess
import sys
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from spectrum.core import canonical, sha256_hex
from spectrum.engine import Engine
from spectrum.engines.oversight import HUMANS

BACKEND = Path(__file__).resolve().parents[1]


def seed_key(seed: int, key_id: str) -> Ed25519PrivateKey:
    """The old derivation: what anyone who has read the seed can compute."""
    return Ed25519PrivateKey.from_private_bytes(hashlib.sha256(f"{seed}:{key_id}".encode()).digest())


def pub_hex(key: Ed25519PrivateKey) -> str:
    return key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()


def all_key_ids(e: Engine) -> set[str]:
    """Every key the server has signed with (the KeyRing cache), plus every approver and agent identity."""
    return (set(e.keys._keys) | {r["key_id"] for r in e.ledger.records}
            | {f"kms:spectrum/human/{h}" for h in HUMANS} | {a["key_id"] for a in e.swarm.agents.values()})


def signed_chain(e: Engine) -> list[tuple[str, str]]:
    return [(r["hash"], r["sig"]) for r in e.ledger.records]


@pytest.fixture(scope="module")
def engine():
    return Engine(seed=7)


def test_seed_derived_keys_match_no_server_public_key(engine):
    key_ids = all_key_ids(engine)
    for family in ("human/", "agent/", "operator/", "aggregator", "aibom", "release"):
        assert any(k.startswith(f"kms:spectrum/{family}") for k in key_ids), family
    server = {engine.keys.public_hex(k) for k in key_ids}
    assert len(server) == len(key_ids)  # one distinct key per identity
    assert server.isdisjoint(pub_hex(seed_key(engine.seed, k)) for k in key_ids)


@pytest.mark.parametrize("key_id", [
    "kms:spectrum/aggregator",         # ledger records
    "kms:spectrum/human/m.chen",       # human approvals
    "kms:spectrum/agent/coordinator",  # agent-to-agent messages
    "kms:spectrum/aibom",              # AI-BOM
])
def test_signature_from_seed_derived_key_does_not_verify(engine, key_id):
    message = sha256_hex("forged by someone who read the seed")
    forged = seed_key(engine.seed, key_id).sign(message.encode()).hex()
    assert not engine.keys.verify(key_id, message, forged)
    assert engine.keys.verify(key_id, message, engine.keys.sign(key_id, message))


def test_forged_approval_record_fails_only_on_its_signature(engine, monkeypatch):
    """An attacker with datastore access appends a well-formed approval and signs it with the seed key."""
    last = engine.ledger.records[-1]
    key_id = "kms:spectrum/human/m.chen"
    body = {"seq": last["seq"] + 1, "ts": last["ts"], "type": "approval.decision", "actor": "human:m.chen",
            "plane": "operate", "key_id": key_id,
            "payload": {"approval": "APR-999", "decision": "approved", "by": "m.chen", "device": "console",
                        "changes": {}, "reason": ""}}
    h = sha256_hex(canonical(body) + last["hash"])
    old = seed_key(engine.seed, key_id)
    sig = old.sign(h.encode())
    old.public_key().verify(sig, h.encode())  # valid under the old keys, so this record used to pass
    forged = {**body, "prev_hash": last["hash"], "hash": h, "alg": "Ed25519", "sig": sig.hex()}
    monkeypatch.setattr(engine.ledger, "records", [*engine.ledger.records, forged])
    res = engine.ledger.verify(emit=False)
    assert not res["ok"] and res["broken"]["seq"] == forged["seq"]
    assert res["broken"]["checks"] == {"hash_matches": True, "link_matches": True, "signature_valid": False}


def test_aibom_signed_with_seed_key_is_rejected(engine, monkeypatch):
    bom = engine.proving.aiboms["nlm-v6"]
    forged = seed_key(engine.seed, "kms:spectrum/aibom").sign(bom["digest"].encode()).hex()
    monkeypatch.setitem(engine.proving.aiboms, "nlm-v6", {**bom, "signature": forged})
    res = engine.proving.verify_aibom("nlm-v6")
    assert res["digest_matches"] and not res["signature_valid"]


def test_rebuilds_and_resets_keep_keys_signatures_and_ledger_head(engine):
    """Record hashes do not depend on keys, so equal heads alone would not catch a per-KeyRing secret:
    compare public keys and every signature too."""
    a = Engine(seed=7)
    assert signed_chain(a) == signed_chain(engine)  # two builds in one process
    key_ids = sorted(all_key_ids(a))
    pubs = {k: a.keys.public_hex(k) for k in key_ids}
    ring = a.keys
    a.reset()  # rebuilds everything, including a new KeyRing
    assert a.keys is not ring
    b = Engine(seed=7, warm=False)
    b.reset()
    for e in (a, b):
        assert {k: e.keys.public_hex(k) for k in key_ids} == pubs
    assert a.ledger.records[-1]["hash"] == b.ledger.records[-1]["hash"]
    assert signed_chain(a) == signed_chain(b)
    assert a.ledger.verify(emit=False)["ok"]


def test_existing_signatures_verify_against_published_keys(engine):
    """Each signed artefact verifies with an independent Ed25519 check against the public key the API
    publishes (GET /api/keys/{key_id})."""
    published: dict[str, Ed25519PublicKey] = {}

    def check(key_id: str, message: str, sig_hex: str) -> None:
        if key_id not in published:
            raw = bytes.fromhex(engine.ledger.public_key(key_id)["public_key_hex"])
            published[key_id] = Ed25519PublicKey.from_public_bytes(raw)
        published[key_id].verify(bytes.fromhex(sig_hex), message.encode())  # raises InvalidSignature

    assert engine.ledger.verify(emit=False)["ok"]
    for rec in engine.ledger.records:
        check(rec["key_id"], rec["hash"], rec["sig"])
    approvals = [ap for item in engine.oversight.items.values() for ap in item["approvals"]]
    assert approvals
    for ap in approvals:
        payload = engine.ledger.records[ap["ledger_seq"] - 1]["payload"]
        check(f"kms:spectrum/human/{ap['by']}", sha256_hex(canonical(payload)), ap["sig"])
    assert engine.swarm.messages
    for m in engine.swarm.messages:
        check(m["key_id"], m["digest"], m["sig"])
    for model in engine.proving.aiboms:
        assert engine.proving.verify_aibom(model)["signature_valid"]
    for rows in engine.proving.fed_evals.values():
        for row in rows:
            body = {k: row[k] for k in ("model", "operator", "score", "n", "data_left_bytes")}
            check(row["key_id"], sha256_hex(str(sorted(body.items()))), row["sig"])
    for target in engine.release.targets:
        manifest = {k: v for k, v in target["manifest"].items() if k != "signature"}
        check("kms:spectrum/release", sha256_hex(str(sorted(manifest.items()))), target["manifest"]["signature"])


def public_key_in_new_process(secret: str | None) -> str:
    env = {k: v for k, v in os.environ.items() if k != "SPECTRUM_KEY_SECRET"}
    if secret is not None:
        env["SPECTRUM_KEY_SECRET"] = secret
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(BACKEND), env.get("PYTHONPATH")) if p)
    code = "from spectrum.core import KeyRing; print(KeyRing().public_hex('kms:spectrum/aibom'))"
    out = subprocess.run([sys.executable, "-c", code], env=env, cwd=BACKEND, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def test_key_secret_env_pins_keys_across_processes():
    def hmac_pub(secret: bytes) -> str:
        return pub_hex(Ed25519PrivateKey.from_private_bytes(hmac.digest(secret, b"kms:spectrum/aibom", "sha256")))

    pinned = hmac_pub(b"pinned-for-test")
    assert public_key_in_new_process("pinned-for-test") == pinned
    assert public_key_in_new_process("pinned-for-test") == pinned
    # unset or empty: a fresh random secret in each process (an empty secret is never used as the key)
    unpinned = [public_key_in_new_process(None), public_key_in_new_process(None), public_key_in_new_process("")]
    assert len(set(unpinned)) == 3
    assert pinned not in unpinned and hmac_pub(b"") not in unpinned
