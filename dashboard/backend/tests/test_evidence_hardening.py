"""Follow-ups from the independent review: nothing unverified shows as passed, hostile files never
stop ingestion, reads re-verify from disk, no absolute paths, wall-clock event times."""

import hashlib
import io
import json
import os

import pytest
from fastapi.testclient import TestClient

from spectrum.evidence.rebuild import rebuild
from spectrum.evidence.sources import LocalSource
from spectrum.evidence.store import LEDGER_FILE
from test_evidence import R1, app_with_fixture, lines, make_dir, service  # noqa: F401 (fixture re-export)


def test_checks_are_unverified_when_the_chain_is_broken(tmp_path):
    d = make_dir(tmp_path)
    service(d, tmp_path).scan_sync()
    recs = lines(d)
    recs[4]["payload"]["size"] = 999
    (d / LEDGER_FILE).write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in recs))
    s = service(d, tmp_path)
    assert s.state == "broken"
    runs = s.snapshot()["runs"]
    assert runs
    for r in runs:
        assert {c["state"] for c in r["checks"].values()} == {"unverified"}, r["run_id"]
    assert all(c["state"] == "unverified" for c in s.run_detail(R1)["run"]["checks"].values())
    assert rebuild(d, io.StringIO()) == 2


HOSTILE = [
    b"[" * 100000 + b"]" * 100000,  # RecursionError inside json
    json.dumps({"run_id": "RID", "unit": "\ud800", "target": {"metric": "exact_match", "value": 0.7},
                "test_set_sha256": "a" * 64, "model_id": "m", "grader": "g"}).encode(),  # lone surrogate escape
    b"\xff\xfe not utf-8",
]


@pytest.mark.parametrize("content", HOSTILE, ids=["deep-nesting", "lone-surrogate", "not-utf8"])
def test_hostile_declarations_are_invalid_records_and_do_not_block_others(tmp_path, content):
    d = make_dir(tmp_path)
    (d / "declared" / "RID.json").write_bytes(content)
    s = service(d, tmp_path)
    counts = s.scan_sync()
    assert counts["new"] == 11 and counts["invalid"] == 2  # fixture-03 and this one; every other file still ingested
    bad = next(r for r in s.ledger.records if r["payload"]["path"] == "declared/RID.json")
    assert bad["payload"]["status"] == "invalid" and bad["payload"]["errors"]
    assert bad["payload"]["sha256"] == hashlib.sha256(content).hexdigest()  # the hash of the raw bytes
    assert service(d, tmp_path).state == "ok"  # and the chain written around it still verifies


def test_hostile_declaration_via_the_api_is_not_a_500(app_with_fixture):  # noqa: F811
    main, s, d = app_with_fixture
    (d / "declared" / "RID.json").write_bytes(HOSTILE[0])
    (d / "declared" / "RID2.json").write_bytes(HOSTILE[1])
    c = TestClient(main.app, client=("127.0.0.1", 50000))
    r = c.post("/api/evidence/scan")
    assert r.status_code == 200 and r.json()["scan"]["new"] == 12


def test_a_non_utf8_file_name_is_recorded_not_fatal(tmp_path):
    d = make_dir(tmp_path)
    try:
        (d / "outputs" / R1 / os.fsdecode(b"bad-\xff.json")).write_text("{}")
    except (OSError, UnicodeEncodeError):
        pytest.skip("this filesystem refuses non-UTF-8 names")
    s = service(d, tmp_path)
    assert s.scan_sync()["new"] == 11
    assert service(d, tmp_path).state == "ok"
    assert s.scan_sync()["new"] == 0


def test_reads_reverify_from_disk_even_if_size_and_mtime_are_restored(app_with_fixture):  # noqa: F811
    main, s, d = app_with_fixture
    c = TestClient(main.app, client=("127.0.0.1", 50000))
    c.post("/api/evidence/scan")
    f = d / LEDGER_FILE
    st = f.stat()
    text = f.read_text()
    i = text.index('"size": ') + len('"size": ')
    f.write_text(text[:i] + ("1" if text[i] != "1" else "2") + text[i + 1:])  # same length
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns))  # mtime put back
    assert f.stat().st_size == st.st_size
    assert c.get("/api/evidence").json()["status"]["state"] == "broken"


def test_status_never_shows_an_absolute_path(app_with_fixture):  # noqa: F811
    main, s, d = app_with_fixture
    c = TestClient(main.app, client=("127.0.0.1", 50000))
    assert not c.get("/api/evidence").json()["status"]["dir"].startswith("/")


def test_local_source_rehashes_old_cache_entries(tmp_path):
    d = make_dir(tmp_path)
    f = d / "outputs" / R1 / "item-0001.json"
    src = LocalSource(d, max_cache_age_s=0)
    before = {a.path: a.sha256 for a in src.list_artifacts()}
    st = f.stat()
    f.write_text(f.read_text().replace("1", "9", 1))  # same size
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns))  # mtime put back
    after = {a.path: a.sha256 for a in src.list_artifacts()}
    assert after[f"outputs/{R1}/item-0001.json"] != before[f"outputs/{R1}/item-0001.json"]


def test_evidence_events_carry_wall_clock_time(app_with_fixture):  # noqa: F811
    main, s, d = app_with_fixture
    s.emit = main._emit_evidence
    c = TestClient(main.app, client=("127.0.0.1", 50000))
    c.post("/api/evidence/scan")
    ev = [e for e in main.engine.bus.history if e["type"] == "evidence.record"][-1]
    assert ev["ts"].endswith("Z") and ev["clock"] == "wall"
