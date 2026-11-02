"""The REAL evidence ledger: persisted, verified on load, untouched by the demo, idempotent ingest,
changed/missing/invalid artifacts recorded (never overwritten), S3 source (mocked), offline rebuild."""

import io
import json
import os
import shutil
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from spectrum.evidence.rebuild import rebuild
from spectrum.evidence.service import EvidenceService
from spectrum.evidence.sources import S3Source, classify
from spectrum.evidence.store import ANCHORS_FILE, LEDGER_FILE, PUBKEYS_FILE

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "evidence"
BACKEND = Path(__file__).resolve().parents[1]
R1, R2, R3 = "2026-11-02T00-00-fixture-01", "2026-11-02T00-00-fixture-02", "2026-11-02T00-00-fixture-03"
T0 = datetime(2026, 11, 2, 0, 0, tzinfo=timezone.utc).timestamp()


def make_dir(tmp_path: Path) -> Path:
    """A copy of the synthetic fixtures with deterministic times: declarations first, then outputs."""
    d = tmp_path / "evidence"
    shutil.copytree(FIXTURES, d)
    for p in sorted(d.rglob("*")):
        if p.is_file():
            rel = p.relative_to(d).as_posix()
            t = T0 if rel.startswith("declared/") else T0 + 600 if rel.startswith("outputs/") else T0 + 900
            os.utime(p, (t, t))
    return d


def service(d: Path, tmp_path: Path, **kw) -> EvidenceService:
    return EvidenceService(directory=d, key_path=tmp_path / "evidence-key", **kw)


def lines(d: Path, name: str = LEDGER_FILE) -> list[dict]:
    return [json.loads(x) for x in (d / name).read_text().splitlines() if x.strip()]


# ---------------------------------------------------------------------------- persistence

def test_ingest_persists_and_survives_a_restart(tmp_path):
    d = make_dir(tmp_path)
    s = service(d, tmp_path)
    counts = s.scan_sync()
    assert counts["new"] == 10 and counts["invalid"] == 1  # 3 declarations, 5 outputs, 1 log, judging.json
    assert len(lines(d)) == 10
    before = [r["hash"] for r in s.ledger.records]

    s2 = service(d, tmp_path)  # a restart: new process state, same files and key
    assert s2.state == "ok" and s2.verify_result["ok"]
    assert [r["hash"] for r in s2.ledger.records] == before
    assert s2.scan_sync()["new"] == 0  # the index is rebuilt from the chain, so nothing is re-ingested
    assert len(lines(d)) == 10


def test_signing_key_is_0600_and_public_key_is_published(tmp_path):
    d = make_dir(tmp_path)
    s = service(d, tmp_path)
    key = tmp_path / "evidence-key"
    assert stat.S_IMODE(key.stat().st_mode) == 0o600
    pub = json.loads((d / PUBKEYS_FILE).read_text())
    assert pub["keys"]["evidence/ingester"]["public_key_hex"] == s.pub_hex
    assert key.read_text().strip() not in (d / PUBKEYS_FILE).read_text()
    assert service(d, tmp_path).pub_hex == s.pub_hex  # same key after a restart


def test_anchor_every_32_records_is_persisted_and_verified(tmp_path):
    d = make_dir(tmp_path)
    for i in range(30):
        (d / "outputs" / R1 / f"extra-{i:03d}.json").write_text(json.dumps({"i": i}))
    s = service(d, tmp_path)
    s.scan_sync()
    assert len(s.ledger.records) == 40
    anchors = lines(d, ANCHORS_FILE)
    assert len(anchors) == 1 and anchors[0]["range"] == [1, 32] and anchors[0]["store"] == "local file"
    assert "s3_uri" not in anchors[0]  # no simulated Object Lock claims on real evidence
    assert service(d, tmp_path).state == "ok"
    anchors[0]["root"] = "0" * 64
    (d / ANCHORS_FILE).write_text(json.dumps(anchors[0]) + "\n")
    s3 = service(d, tmp_path)
    assert s3.state == "broken" and "anchor #1" in s3.problem()


def test_an_anchor_edit_while_running_is_caught(tmp_path):
    d = make_dir(tmp_path)
    for i in range(30):
        (d / "outputs" / R1 / f"extra-{i:03d}.json").write_text(json.dumps({"i": i}))
    s = service(d, tmp_path)
    s.scan_sync()
    a = lines(d, ANCHORS_FILE)[0]
    a["root"] = "ab" * 32
    (d / ANCHORS_FILE).write_text(json.dumps(a) + "\n")
    assert s.check_disk() is False and "anchor #1" in s.problem()


# ---------------------------------------------------------------------------- tamper / key

def test_tampered_ledger_is_reported_at_the_exact_seq_and_never_repaired(tmp_path):
    d = make_dir(tmp_path)
    service(d, tmp_path).scan_sync()
    recs = lines(d)
    recs[3]["payload"]["sha256"] = "f" * 64  # record #4, hash and signature left as they were
    (d / LEDGER_FILE).write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in recs))
    raw = (d / LEDGER_FILE).read_bytes()

    s = service(d, tmp_path)
    assert s.state == "broken"
    assert s.verify_result["broken"]["seq"] == 4
    assert "hash mismatch" in s.problem()
    (d / "outputs" / R1 / "late.json").write_text("{}")
    assert "skipped" in s.scan_sync()  # no appends onto a broken chain
    assert (d / LEDGER_FILE).read_bytes() == raw  # nothing repaired or rewritten
    assert s.snapshot()["status"]["ingest_enabled"] is False


def test_edit_while_running_is_caught_on_the_next_scan(tmp_path):
    d = make_dir(tmp_path)
    events = []
    s = service(d, tmp_path, emit=lambda *a, **k: events.append(a))
    s.scan_sync()
    recs = lines(d)
    recs[1]["ts"] = "2020-01-01T00:00:00.000Z"
    (d / LEDGER_FILE).write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in recs))
    s.scan_sync()
    assert s.state == "broken" and s.verify_result["broken"]["seq"] == 2
    assert any(e[0] == "evidence.status" and e[4] == "critical" for e in events)


def test_a_different_signing_key_is_a_loud_mismatch(tmp_path):
    d = make_dir(tmp_path)
    service(d, tmp_path).scan_sync()
    s = EvidenceService(directory=d, key_path=tmp_path / "other-key")
    assert s.state == "key_mismatch" and s.verify_result["ok"]  # old records still verify with the published key
    assert "skipped" in s.scan_sync()


# ---------------------------------------------------------------------------- ingest semantics

def test_rescanning_is_idempotent(tmp_path):
    d = make_dir(tmp_path)
    s = service(d, tmp_path)
    s.scan_sync()
    n = len(s.ledger.records)
    for _ in range(3):
        assert s.scan_sync()["new"] == 0
    assert len(s.ledger.records) == n == len(lines(d))


def test_a_changed_artifact_is_flagged_and_the_original_record_stays(tmp_path):
    d = make_dir(tmp_path)
    s = service(d, tmp_path)
    s.scan_sync()
    target = f"outputs/{R1}/item-0002.json"
    orig = next(r for r in s.ledger.records if r["payload"]["path"] == target)
    (d / target).write_text('{"item": 2, "output": "edited after the run"}\n')
    assert s.scan_sync()["changed"] == 1
    assert s.scan_sync()["changed"] == 0  # flagged once, not on every scan
    ch = s.ledger.records[-1]
    assert ch["type"] == "evidence.changed" and ch["payload"]["original_seq"] == orig["seq"]
    assert ch["payload"]["original_sha256"] == orig["payload"]["sha256"] != ch["payload"]["sha256"]
    assert lines(d)[orig["seq"] - 1] == orig  # original record untouched on disk
    run = next(r for r in s.snapshot()["runs"] if r["run_id"] == R1)
    assert run["checks"]["unchanged"]["state"] == "fail"


def test_a_deleted_artifact_is_recorded_missing(tmp_path):
    d = make_dir(tmp_path)
    s = service(d, tmp_path)
    s.scan_sync()
    (d / f"logs/{R1}/run.log").unlink()
    assert s.scan_sync()["missing"] == 1
    assert s.ledger.records[-1]["type"] == "evidence.missing"
    assert s.scan_sync()["missing"] == 0


def test_malformed_declarations_become_visible_invalid_records(tmp_path):
    d = make_dir(tmp_path)
    (d / "declared" / "2026-11-02T00-00-fixture-04.json").write_text("{not json")
    (d / "declared" / "2026-11-02T00-00-fixture-05.json").write_text(json.dumps({"run_id": "someone-else"}))
    s = service(d, tmp_path)
    s.scan_sync()
    bad = {r["payload"]["run_id"]: r["payload"] for r in s.ledger.records if r["payload"].get("status") == "invalid"}
    assert set(bad) == {R3, "2026-11-02T00-00-fixture-04", "2026-11-02T00-00-fixture-05"}
    assert any("target.value" in e for e in bad[R3]["errors"])
    assert any("64 lowercase hex" in e for e in bad[R3]["errors"])
    assert any("not valid JSON" in e for e in bad["2026-11-02T00-00-fixture-04"]["errors"])
    assert any("does not match the file name" in e for e in bad["2026-11-02T00-00-fixture-05"]["errors"])
    runs = {r["run_id"]: r for r in s.snapshot()["runs"]}
    assert runs[R3]["declaration_status"] == "invalid" and runs[R3]["checks"]["declared"]["state"] == "fail"


def test_run_checks_are_real_or_explicitly_not_checked(tmp_path):
    d = make_dir(tmp_path)
    late = d / "declared" / f"{R2}.json"
    os.utime(late, (T0 + 3600, T0 + 3600))  # R2 declared AFTER its outputs
    s = service(d, tmp_path)
    s.scan_sync()
    runs = {r["run_id"]: r for r in s.snapshot()["runs"]}
    c1, c2 = runs[R1]["checks"], runs[R2]["checks"]
    assert {k: c1[k]["state"] for k in ("declared", "declared_before_outputs", "test_set_hash", "grader_recorded",
                                         "unchanged")} == dict.fromkeys(
        ("declared", "declared_before_outputs", "test_set_hash", "grader_recorded", "unchanged"), "pass")
    assert c2["declared_before_outputs"]["state"] == "fail"
    for c in (c1, c2):
        assert c["times_agree_aws"]["state"] == "not_checked"
        assert c["independent_grader"]["state"] == "not_checked"
    assert runs[R1]["judged"] and runs[R1]["artifacts"] == 5
    assert runs[R1]["declaration"]["target"] == {"metric": "exact_match", "value": "0.7"}


def test_layout_classification():
    assert classify("declared/x.json") == ("declaration", "x")
    assert classify("outputs/x/a/b.json") == ("output", "x")
    assert classify("logs/x/run.log") == ("log", "x")
    assert classify("judging.json") == ("judging", "")
    for ignored in ("ledger.jsonl", "README.md", "outputs/x", "declared/.x.json", "other/x/y"):
        assert classify(ignored) is None


# ---------------------------------------------------------------------------- S3 (mocked; no AWS)

class FakeS3:
    """ListObjectsV2 paginator + GetObject only, like the read-only source uses."""

    def __init__(self, objects: dict[str, bytes]):
        self.objects = objects
        self.calls: list[tuple] = []

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        fake = self

        class P:
            def paginate(self, Bucket, Prefix):
                fake.calls.append(("list", Bucket, Prefix))
                keys = sorted(k for k in fake.objects if k.startswith(Prefix))
                yield {"Contents": [{"Key": k, "ETag": f'"{hash(fake.objects[k])}"',
                                     "LastModified": datetime(2026, 11, 2, 0, i, tzinfo=timezone.utc)}
                                    for i, k in enumerate(keys)]}
        return P()

    def get_object(self, Bucket, Key):
        self.calls.append(("get", Bucket, Key))
        return {"Body": io.BytesIO(self.objects[Key]), "VersionId": f"v-{len(self.calls)}"}

    def __getattr__(self, name):  # anything else (puts, deletes) would be a write: fail loudly
        raise AssertionError(f"S3 source must be read-only; called {name}")


def test_s3_source_reads_with_list_and_get_only(tmp_path):
    objs = {f"evidence/{p.relative_to(FIXTURES).as_posix()}": p.read_bytes()
            for p in FIXTURES.rglob("*") if p.is_file()}
    fake = FakeS3(objs)
    src = S3Source("s3://example-bucket/evidence/", client=fake)
    arts = {a.path: a for a in src.list_artifacts()}
    assert f"declared/{R1}.json" in arts and "README.md" not in arts
    a = arts[f"declared/{R1}.json"]
    assert a.source == "s3" and a.version_id and a.source_time.startswith("2026-11-02T00:")
    gets = sum(1 for c in fake.calls if c[0] == "get")
    src.list_artifacts()
    assert sum(1 for c in fake.calls if c[0] == "get") == gets  # unchanged objects are not downloaded again

    s = EvidenceService(directory=tmp_path / "ledger", key_path=tmp_path / "k", source=src)
    assert s.scan_sync()["new"] == 10 and s.scan_sync()["new"] == 0
    assert s.status()["source"]["kind"] == "s3"


def test_s3_is_off_unless_the_env_var_is_set(tmp_path, monkeypatch):
    monkeypatch.delenv("SPECTRUM_EVIDENCE_S3", raising=False)
    assert service(make_dir(tmp_path), tmp_path).source.kind == "local"


def test_a_failing_source_does_not_crash(tmp_path):
    class Broken:
        kind = "s3"

        def describe(self):
            return {"kind": "s3", "uri": "s3://example-bucket/evidence/"}

        def list_artifacts(self):
            raise RuntimeError("no credentials")
    s = EvidenceService(directory=tmp_path / "e", key_path=tmp_path / "k", source=Broken())
    assert "no credentials" in s.scan_sync()["error"]
    assert "no credentials" in s.status()["source"]["error"]


# ---------------------------------------------------------------------------- rebuild

def test_rebuild_exit_codes(tmp_path):
    d = make_dir(tmp_path)
    service(d, tmp_path).scan_sync()
    out = io.StringIO()
    assert rebuild(d, out) == 0
    text = out.getvalue()
    assert "chain OK: 10 records" in text and R1 in text and "exact_match 0.7" in text

    clean = tmp_path / "copy"
    for mutate, code, needle in [
        (lambda e: (e / f"outputs/{R1}/item-0001.json").write_text("changed"), 1, "CHANGED since ingest"),
        (lambda e: (e / f"logs/{R1}/run.log").unlink(), 1, "MISSING"),
        (lambda e: (e / LEDGER_FILE).write_text((e / LEDGER_FILE).read_text().replace(R2, R2 + "x", 1)), 2,
         "chain broken at #"),
    ]:
        shutil.rmtree(clean, ignore_errors=True)
        shutil.copytree(d, clean)
        mutate(clean)
        out = io.StringIO()
        assert rebuild(clean, out) == code, out.getvalue()
        assert needle in out.getvalue()


def test_rebuild_fails_if_the_ledger_already_recorded_a_change(tmp_path):
    d = make_dir(tmp_path)
    s = service(d, tmp_path)
    s.scan_sync()
    p = d / f"outputs/{R1}/item-0001.json"
    good = p.read_bytes()
    p.write_text("changed")
    s.scan_sync()
    p.write_bytes(good)  # put it back: the recorded change still counts
    out = io.StringIO()
    assert rebuild(d, out) == 1 and "recorded it changed" in out.getvalue()


def test_rebuild_cli_runs_offline(tmp_path):
    d = make_dir(tmp_path)
    service(d, tmp_path).scan_sync()
    env = {k: v for k, v in os.environ.items() if not k.startswith("AWS_")}
    ok = subprocess.run([sys.executable, "-m", "spectrum.evidence", "rebuild", "--dir", str(d)], cwd=BACKEND,
                        env=env, capture_output=True, text=True)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    (d / f"outputs/{R2}/item-0001.json").write_text("x")
    bad = subprocess.run([sys.executable, "-m", "spectrum.evidence", "rebuild", "--dir", str(d)], cwd=BACKEND,
                         env=env, capture_output=True, text=True)
    assert bad.returncode == 1 and "CHANGED since ingest" in bad.stdout


def test_rebuild_of_an_empty_evidence_dir_is_clean(tmp_path):
    assert rebuild(tmp_path, io.StringIO()) == 0


# ---------------------------------------------------------------------------- API + demo isolation

@pytest.fixture
def app_with_fixture(tmp_path, monkeypatch):
    from spectrum import main
    d = make_dir(tmp_path)
    s = EvidenceService(directory=d, key_path=tmp_path / "evidence-key",
                        emit=lambda *a, **k: main.engine.bus.emit(*a, **k))
    monkeypatch.setattr(main, "evidence", s)
    return main, s, d


def test_reset_and_the_demo_beats_leave_the_evidence_ledger_untouched(app_with_fixture):
    main, s, d = app_with_fixture
    c = TestClient(main.app, client=("127.0.0.1", 50000))
    assert c.post("/api/evidence/scan").json()["scan"]["new"] == 10
    disk = (d / LEDGER_FILE).read_bytes()
    hashes = [r["hash"] for r in s.ledger.records]
    for n in range(1, 15):
        assert c.post(f"/api/director/beat/{n}").status_code == 200
    assert c.post("/api/director/reset").status_code == 200
    assert (d / LEDGER_FILE).read_bytes() == disk
    assert [r["hash"] for r in s.ledger.records] == hashes
    snap = c.get("/api/snapshot").json()
    assert snap["evidence"]["status"]["records"] == 10 and snap["evidence"]["real"] is True
    assert all(r["type"].startswith("evidence.") for r in snap["evidence"]["recent"])
    # events still reach the engine's NEW bus after the reset
    (d / f"outputs/{R2}/item-0003.json").write_text("{}")
    c.post("/api/evidence/scan")
    assert any(e["type"] == "evidence.record" for e in main.engine.bus.history)


def test_api_reports_a_tampered_ledger(app_with_fixture, tmp_path):
    main, s, d = app_with_fixture
    c = TestClient(main.app, client=("127.0.0.1", 50000))
    c.post("/api/evidence/scan")
    recs = lines(d)
    recs[6]["payload"]["size"] = 1
    (d / LEDGER_FILE).write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in recs))
    v = c.post("/api/evidence/verify").json()
    assert v["verify"]["ok"] is False and v["verify"]["broken"]["seq"] == 7
    st = c.get("/api/evidence").json()["status"]
    assert st["state"] == "broken" and "#7" in st["problem"]
    chain = c.get("/api/evidence/chain").json()
    assert chain["records"][6]["payload"]["size"] == 1  # served as found on disk, for the browser to verify
    assert c.get(f"/api/evidence/runs/{R1}").json()["run"]["run_id"] == R1
    assert c.get("/api/evidence/runs/nope").status_code == 404
