"""EvidenceService: owns the evidence ledger, ingests artifacts, summarises runs, emits evidence.* events.

Created once in main.py next to the demo Engine, never inside it, so Engine.reset() and the demo
beats cannot reach it. Events go to whatever bus the engine has now (reset replaces the bus).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Callable

from ..core import KeyRing
from ..engines.ledger import ANCHOR_EVERY, GENESIS
from .sources import Artifact, default_source, validate_declaration, validate_judging
from .store import (ANCHORS_FILE, KEY_ID, LEDGER_FILE, REPO_ROOT, PUBKEYS_FILE, EvidenceLedger, PublicKeyRing, evidence_dir, fingerprint,
                    key_file, load_disk, load_or_create_secret, public_keys_doc, utc_iso, verify_chain)

log = logging.getLogger("spectrum.evidence")

NOT_CHECKED = {"state": "not_checked", "detail": "not yet checked (later work)"}
UNVERIFIED = {"state": "unverified", "detail": "unverified (chain broken): the ledger these checks come from does not verify"}
CHECK_IDS = ("declared", "declared_before_outputs", "test_set_hash", "grader_recorded", "unchanged", "times_agree_aws",
             "independent_grader")


def clean(s: str) -> str:
    """A string that is always valid UTF-8 (lone surrogates and undecodable bytes become escapes)."""
    return s.encode("utf-8", "backslashreplace").decode("utf-8")


def build_index(records: list[dict]) -> dict[str, dict]:
    """path -> {first, last_sha, missing, events}: the ingest state, rebuilt from the chain itself."""
    idx: dict[str, dict] = {}
    for r in records:
        p = r.get("payload") or {}
        path = p.get("path")
        if not path:
            continue
        t = r.get("type")
        if t == "evidence.artifact" and path not in idx:
            idx[path] = {"first": r, "last_sha": p.get("sha256"), "missing": False, "events": []}
        elif path in idx:
            e = idx[path]
            e["events"].append(r)
            if t == "evidence.changed":
                e["last_sha"], e["missing"] = p.get("sha256"), False
            elif t == "evidence.missing":
                e["missing"] = True
            elif t == "evidence.reappeared":
                e["missing"] = False
    return idx


def summarise_runs(records: list[dict], chain_ok: bool = True) -> tuple[list[dict], list[str]]:
    """One summary per run id seen in the chain (or listed in judging.json), and the judged run ids.
    With chain_ok=False (the ledger does not verify) every check is "unverified": nothing derived
    from unverified records may be shown as passed."""
    runs: dict[str, dict] = {}
    judged: list[str] = []

    def run(rid: str) -> dict:
        return runs.setdefault(rid, {"run_id": rid, "declaration": None, "declaration_status": "missing",
                                     "declaration_errors": [], "declared_at": None, "declared_seq": None,
                                     "paths": set(), "kinds": {}, "first_output_at": None, "changed": [],
                                     "missing": [], "records": 0, "first_seq": None, "last_seq": None,
                                     "last_ts": None, "judged": False})

    for r in records:
        p = r.get("payload") or {}
        t = r.get("type")
        if p.get("kind") == "judging":
            if t == "evidence.artifact" and p.get("status") == "valid":
                judged = list(p.get("run_ids") or [])
            continue
        rid = p.get("run_id")
        if not rid:
            continue
        s = run(rid)
        s["records"] += 1
        s["first_seq"] = s["first_seq"] or r["seq"]
        s["last_seq"], s["last_ts"] = r["seq"], r["ts"]
        if t == "evidence.artifact":
            s["paths"].add(p["path"])
            s["kinds"][p["kind"]] = s["kinds"].get(p["kind"], 0) + 1
            if p["kind"] == "declaration" and s["declaration_status"] == "missing":
                s["declared_at"], s["declared_seq"] = p.get("source_time"), r["seq"]
                if p.get("status") == "valid":
                    s["declaration"], s["declaration_status"] = p.get("declaration"), "valid"
                else:
                    s["declaration_status"], s["declaration_errors"] = "invalid", p.get("errors") or []
            elif p["kind"] == "output":
                st = p.get("source_time")
                if st and (s["first_output_at"] is None or st < s["first_output_at"]):
                    s["first_output_at"] = st
        elif t == "evidence.changed":
            s["changed"].append({"path": p["path"], "seq": r["seq"]})
        elif t == "evidence.missing":
            s["missing"].append({"path": p["path"], "seq": r["seq"]})
    for rid in judged:
        run(rid)["judged"] = True

    out = []
    for s in runs.values():
        s["artifacts"] = len(s.pop("paths"))
        s["checks"] = run_checks(s) if chain_ok else {k: dict(UNVERIFIED) for k in CHECK_IDS}
        out.append(s)
    out.sort(key=lambda s: (s["first_seq"] is None, s["first_seq"] or 0, s["run_id"]))
    return out, judged


def run_checks(s: dict) -> dict:
    d = s["declaration"]
    ok = lambda detail: {"state": "pass", "detail": detail}  # noqa: E731
    bad = lambda detail: {"state": "fail", "detail": detail}  # noqa: E731
    wait = lambda detail: {"state": "pending", "detail": detail}  # noqa: E731
    checks: dict[str, dict] = {}
    if s["declaration_status"] == "valid":
        checks["declared"] = ok(f"declared/{s['run_id']}.json, ledger #{s['declared_seq']}")
    elif s["declaration_status"] == "invalid":
        checks["declared"] = bad("declaration is invalid: " + "; ".join(s["declaration_errors"]))
    else:
        checks["declared"] = bad("no declaration in declared/")
    if s["declaration_status"] == "missing":
        checks["declared_before_outputs"] = bad("no declaration")
    elif s["first_output_at"] is None:
        checks["declared_before_outputs"] = wait("no outputs yet")
    elif s["declared_at"] < s["first_output_at"]:
        checks["declared_before_outputs"] = ok(f"declared {s['declared_at']}, first output {s['first_output_at']}")
    else:
        checks["declared_before_outputs"] = bad(f"declared {s['declared_at']}, but an output is from {s['first_output_at']}")
    checks["test_set_hash"] = ok(d["test_set_sha256"]) if d else bad("no valid declaration")
    checks["grader_recorded"] = ok(d["grader"]) if d else bad("no valid declaration")
    if s["changed"] or s["missing"]:
        checks["unchanged"] = bad(f"{len(s['changed'])} changed, {len(s['missing'])} missing since ingest")
    else:
        checks["unchanged"] = ok("every artifact still has its ingested hash")
    checks["times_agree_aws"] = dict(NOT_CHECKED)
    checks["independent_grader"] = dict(NOT_CHECKED)
    return checks


class EvidenceService:
    def __init__(self, directory: Path | None = None, key_path: Path | None = None, source=None,
                 emit: Callable | None = None, poll_s: float | None = None):
        self.dir = Path(directory or evidence_dir())
        self.dir.mkdir(parents=True, exist_ok=True)
        self.key_path = Path(key_path or key_file())
        self.signer = KeyRing(load_or_create_secret(self.key_path))
        self.pub_hex = self.signer.public_hex(KEY_ID)
        self.poll_s = float(poll_s or os.environ.get("SPECTRUM_EVIDENCE_POLL_S") or 30)
        self.source = source or default_source(self.dir, max_cache_age_s=self.poll_s)
        self.emit = emit or (lambda *a, **k: None)
        self.last_scan: dict | None = None
        self.source_error: str | None = None
        self._lock: asyncio.Lock | None = None
        self.key_state, self.key_detail = self._check_public_key()
        self.reload()

    # ------------------------------------------------------------------ keys / load

    def _check_public_key(self) -> tuple[str, str | None]:
        f = self.dir / PUBKEYS_FILE
        if not f.exists():
            f.write_text(json.dumps(public_keys_doc(self.pub_hex), indent=2) + "\n")
            return "ok", None
        try:
            published = PublicKeyRing.from_file(f).public_hex(KEY_ID)
        except (ValueError, KeyError, TypeError) as e:
            return "mismatch", f"{PUBKEYS_FILE} is unreadable: {e}"
        if published != self.pub_hex:
            return "mismatch", (f"the signing key file ({self.key_path.name}) does not match the public key in "
                                f"{PUBKEYS_FILE} ({fingerprint(published) if published else 'none'}); "
                                "new records are disabled until this is resolved by hand")
        return "ok", None

    def reload(self) -> dict:
        """Read the ledger from disk and verify it with the PUBLISHED public key. Never repairs."""
        records, anchors, bad_line = load_disk(self.dir)
        try:
            verifier = PublicKeyRing.from_file(self.dir / PUBKEYS_FILE)
        except (OSError, ValueError, KeyError, TypeError):
            verifier = PublicKeyRing({})
        self.verify_result = verify_chain(records, anchors, verifier, bad_line)
        self.ledger = EvidenceLedger(self.dir, self.signer)
        self.ledger.records, self.ledger.anchors = records, anchors
        self.index = build_index(records)
        self._disk_stat = self._stat()
        if not self.verify_result["ok"]:
            log.error("evidence ledger verification FAILED: %s", self.problem())
        return self.verify_result

    def _stat(self):
        out = []
        for name in (LEDGER_FILE, ANCHORS_FILE):
            try:
                st = (self.dir / name).stat()
                out.append((st.st_size, st.st_mtime_ns))
            except FileNotFoundError:
                out.append(None)
        return tuple(out)

    @property
    def state(self) -> str:
        if not self.verify_result["ok"]:
            return "broken"
        if self.key_state != "ok":
            return "key_mismatch"
        return "ok"

    def problem(self) -> str | None:
        v = self.verify_result
        if v["broken"]:
            return f"chain broken at #{v['broken']['seq']}: {v['broken']['reason']}"
        if not v["anchors_ok"]:
            return v["anchor_problem"]
        if self.key_state != "ok":
            return self.key_detail
        return None

    # ------------------------------------------------------------------ ingest

    def check_disk(self, force: bool = False) -> bool:
        """Re-read and re-verify the files if they changed on disk since we last wrote them (always, with
        force=True: a same-size edit with the mtime put back would fool the stat check). True if still OK."""
        if force or self._stat() != self._disk_stat:
            was_ok = self.state == "ok"
            self.reload()
            if was_ok and self.state != "ok":
                self._emit_status(critical=True)
        return self.state == "ok"

    def ingest(self, artifacts: list[Artifact]) -> dict:
        counts = {"seen": len(artifacts), "new": 0, "changed": 0, "missing": 0, "reappeared": 0, "invalid": 0}
        if not self.check_disk():
            counts["skipped"] = self.problem()
            return counts
        appended: list[dict] = []
        seen = set()
        for a in sorted(artifacts, key=lambda a: (a.source_time, clean(a.path))):
            a.path, a.run_id = clean(a.path), clean(a.run_id)  # a non-UTF-8 file name must not break hashing
            seen.add(a.path)
            known = self.index.get(a.path)
            if known is None:
                # one bad file must never block the others: anything that goes wrong becomes an invalid record
                try:
                    payload = self._artifact_payload(a)
                    rec = self._append("evidence.artifact", payload)
                except OSError:
                    raise  # the ledger file itself could not be written: stop, do not guess
                except Exception as e:  # noqa: BLE001
                    log.warning("evidence: %s could not be processed (%s)", a.path, e.__class__.__name__)
                    payload = {**self._base(a), "status": "invalid",
                               "errors": [f"could not be processed: {e.__class__.__name__}"]}
                    rec = self._append("evidence.artifact", payload)
                if payload.get("status") == "invalid":
                    counts["invalid"] += 1
                appended.append(rec)
                counts["new"] += 1
            elif a.sha256 != known["last_sha"]:
                appended.append(self._append("evidence.changed", {
                    **self._base(a), "previous_sha256": known["last_sha"],
                    "original_sha256": known["first"]["payload"]["sha256"], "original_seq": known["first"]["seq"]}))
                counts["changed"] += 1
            elif known["missing"]:
                appended.append(self._append("evidence.reappeared", {**self._base(a),
                                                                     "original_seq": known["first"]["seq"]}))
                counts["reappeared"] += 1
        for path, known in list(self.index.items()):
            if path not in seen and not known["missing"]:
                f = known["first"]["payload"]
                appended.append(self._append("evidence.missing", {
                    "run_id": f.get("run_id", ""), "kind": f["kind"], "path": path, "source": f.get("source"),
                    "last_sha256": known["last_sha"], "original_seq": known["first"]["seq"]}))
                counts["missing"] += 1
        self._disk_stat = self._stat()
        self.last_scan = {"at": utc_iso(), **counts}
        for rec in appended:
            sev = {"evidence.changed": "warn", "evidence.missing": "warn"}.get(rec["type"], "info")
            if rec["payload"].get("status") == "invalid":
                sev = "warn"
            self.emit("evidence.record", "prove", _summary(rec), {"record": rec}, sev, "evidence-ingester",
                      {"seq": rec["seq"], "hash": rec["hash"]})
        if appended:
            self.emit("evidence.update", "prove", f"evidence ledger: {len(appended)} new record(s)",
                      self.snapshot(), "debug", "evidence-ingester")
        return counts

    def _append(self, type: str, payload: dict) -> dict:
        rec = self.ledger.append_persisted(type, payload)
        path = payload["path"]
        if type == "evidence.artifact":
            self.index[path] = {"first": rec, "last_sha": payload["sha256"], "missing": False, "events": []}
            return rec
        e = self.index[path]
        e["events"].append(rec)
        if type == "evidence.changed":
            e["last_sha"], e["missing"] = payload["sha256"], False
        else:
            e["missing"] = type == "evidence.missing"
        return rec

    @staticmethod
    def _base(a: Artifact) -> dict:
        return {"run_id": a.run_id, "kind": a.kind, "path": a.path, "sha256": a.sha256, "size": a.size,
                "source_time": a.source_time, "source": a.source, "version_id": a.version_id}

    def _artifact_payload(self, a: Artifact) -> dict:
        p = self._base(a)
        p["status"], p["errors"] = "valid", []
        if a.kind == "declaration":
            decl, errs = validate_declaration(a.content or b"", a.run_id)
            if decl:
                p["declaration"] = decl
            else:
                p["status"], p["errors"] = "invalid", errs
        elif a.kind == "judging":
            ids, errs = validate_judging(a.content or b"")
            if ids is not None:
                p["run_ids"] = ids
            else:
                p["status"], p["errors"] = "invalid", errs
        return p

    def scan_sync(self) -> dict:
        """Collect and ingest in one call (tests and the CLI; the server uses scan())."""
        try:
            arts = self.source.list_artifacts()
            self.source_error = None
        except Exception as e:  # noqa: BLE001 - a source failure must never stop the server
            self.source_error = f"{e.__class__.__name__}: {e}"
            return {"error": self.source_error}
        return self.ingest(arts)

    async def scan(self) -> dict:
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            try:
                arts = await asyncio.to_thread(self.source.list_artifacts)  # file/S3 reads off the loop
                self.source_error = None
            except Exception as e:  # noqa: BLE001
                self.source_error = f"{e.__class__.__name__}: {e}"
                self.emit("evidence.scan", "prove", f"evidence scan failed: {self.source_error}",
                          {"error": self.source_error}, "warn", "evidence-ingester")
                return {"error": self.source_error}
            return self.ingest(arts)  # ledger appends on the event-loop thread

    async def run(self) -> None:
        if self.state != "ok":
            self._emit_status(critical=True)
        while True:
            try:
                await self.scan()
            except Exception as e:  # noqa: BLE001
                log.exception("evidence scan crashed")
                self.source_error = f"{e.__class__.__name__}: {e}"
            await asyncio.sleep(self.poll_s)

    def verify_now(self) -> dict:
        """Re-read the files from disk and verify them, as an outside auditor would."""
        was = self.state
        self.reload()
        if self.state != "ok":
            self._emit_status(critical=True)
        elif was != "ok":
            self._emit_status(critical=False)
        else:
            self.emit("evidence.update", "prove", "evidence ledger re-verified from disk", self.snapshot(),
                      "debug", "auditor")
        return self.verify_result

    def _emit_status(self, critical: bool) -> None:
        summary = (f"Evidence ledger BROKEN: {self.problem()}" if self.state == "broken" else
                   f"Evidence ledger key mismatch: {self.problem()}" if self.state == "key_mismatch" else
                   "Evidence ledger verified")
        self.emit("evidence.status", "prove", summary, self.snapshot(), "critical" if critical else "notice",
                  "auditor")

    # ------------------------------------------------------------------ views

    def status(self) -> dict:
        recs = self.ledger.records
        anchors = self.ledger.anchors
        return {
            "state": self.state,
            "problem": self.problem(),
            "ingest_enabled": self.state == "ok",
            "records": len(recs),
            "head": recs[-1].get("hash") if recs else GENESIS,
            "anchors": len(anchors),
            "anchor_every": ANCHOR_EVERY,
            "last_anchor": anchors[-1] if anchors else None,
            "verify": self.verify_result,
            "key": {"key_id": KEY_ID, "alg": "Ed25519", "public_key_hex": self.pub_hex,
                    "fingerprint": fingerprint(self.pub_hex), "state": self.key_state, "file": PUBKEYS_FILE},
            "dir": _display_dir(self.dir),
            "source": {**self.source.describe(), "error": self.source_error},
            "poll_s": self.poll_s,
            "last_scan": self.last_scan,
        }

    def snapshot(self) -> dict:
        runs, judged = summarise_runs(self.ledger.records, self.verify_result["ok"])
        return {"real": True, "status": self.status(), "runs": runs, "judged": judged,
                "recent": self.ledger.records[-30:]}

    def run_detail(self, run_id: str) -> dict | None:
        runs, _ = summarise_runs(self.ledger.records, self.verify_result["ok"])
        s = next((r for r in runs if r["run_id"] == run_id), None)
        if s is None:
            return None
        recs = [r for r in self.ledger.records if (r.get("payload") or {}).get("run_id") == run_id]
        return {"run": s, "records": recs, "public_key": self.status()["key"]}

    def chain(self) -> dict:
        try:
            published = json.loads((self.dir / PUBKEYS_FILE).read_text())
        except (OSError, ValueError):
            published = {}
        return {"records": self.ledger.records, "anchors": self.ledger.anchors, "public_keys": published,
                "anchor_every": ANCHOR_EVERY}


def _display_dir(d: Path) -> str:
    """The evidence folder relative to the repo (never an absolute path: it would leak the home directory)."""
    try:
        return d.resolve().relative_to(REPO_ROOT).as_posix() or "."
    except ValueError:
        return d.name


def _summary(rec: dict) -> str:
    p = rec["payload"]
    t = rec["type"]
    if t == "evidence.artifact":
        bad = " (INVALID)" if p.get("status") == "invalid" else ""
        return f"#{rec['seq']} ingested {p['kind']} {p['path']}{bad}"
    if t == "evidence.changed":
        return f"#{rec['seq']} CHANGED after ingest: {p['path']} (first recorded at #{p['original_seq']})"
    if t == "evidence.missing":
        return f"#{rec['seq']} MISSING: {p['path']} (recorded at #{p['original_seq']})"
    return f"#{rec['seq']} {t} {p.get('path', '')}"
