"""The evidence ledger and the team's evidence tools (tools/evlib.py, tools/check_unit.py) agree:
classify() knows results/ and models/ and both record layouts; a records .jsonl that grows by whole lines
is an append, not a change; and check_unit, which shares no code with this package, verifies a ledger
written by the real EvidenceService (hash chain AND Ed25519 signatures)."""

import io
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from spectrum.evidence.rebuild import rebuild
from spectrum.evidence.service import EvidenceService
from spectrum.evidence.sources import LocalSource, classify

HERE = Path(__file__).resolve().parent
LAYOUT = HERE / "fixtures" / "records" / "layout"
REPO = HERE.parents[2]
TOOLS = REPO / "tools"
R = "R-layout-1"
T0 = datetime(2026, 11, 2, 18, 0, tzinfo=timezone.utc).timestamp()


def layout_dir(tmp_path: Path) -> Path:
    d = tmp_path / "evidence"
    shutil.copytree(LAYOUT, d)
    for p in d.rglob("*"):  # declaration first, then records, then the result
        rel = p.relative_to(d).as_posix()
        t = T0 if rel.startswith(("declared/", "models/")) else T0 + 1800 if rel.startswith("results/") else T0 + 900
        os.utime(p, (t, t))
    return d


def service(d: Path, tmp_path: Path) -> EvidenceService:
    return EvidenceService(directory=d, key_path=tmp_path / "evidence-key")


# ---------------------------------------------------------------------------- classify

@pytest.mark.parametrize("path,want", [
    (f"results/{R}.json", ("result", R)),
    ("models/t2-response/v0.json", ("model", "")),
    (f"records/{R}/item.jsonl", ("record", R)),
    (f"records/{R}/change/8cfc0123935380633813c44d.json", ("record", R)),  # amendment A11: one object per record
    (f"declared/{R}.json", ("declaration", R)),
])
def test_classify_new_homes(path, want):
    assert classify(path) == want


@pytest.mark.parametrize("path", ["results/x/y.json", "results/.json", "results/R.txt", "models/m.json",
                                  "models/m/v/extra.json", "models/m/.json", "results/.hidden.json"])
def test_classify_ignores_malformed_new_paths(path):
    assert classify(path) is None


# ---------------------------------------------------------------------------- ingest

def test_layout_fixture_ingests_and_the_run_checks_pass(tmp_path):
    d = layout_dir(tmp_path)
    s = service(d, tmp_path)
    counts = s.scan_sync()
    assert counts["new"] == 6 and counts["invalid"] == 0  # README.md ignored
    run = s.run_detail(R)["run"]
    assert run["kinds"] == {"declaration": 1, "record": 3, "result": 1}
    assert {k: c["state"] for k, c in run["checks"].items() if c["state"] != "not_checked"} == dict.fromkeys(
        ("declared", "declared_before_outputs", "test_set_hash", "grader_recorded", "unchanged"), "pass")
    runs = {r["run_id"] for r in s.snapshot()["runs"]}
    assert runs == {R}  # the model entry is ledgered but is not a run (no "missing declaration" row)
    model = next(r for r in s.ledger.records if r["payload"]["path"] == "models/t2-response/v0.json")
    assert model["payload"]["kind"] == "model" and model["payload"]["run_id"] == ""


def test_records_count_as_outputs_for_declared_before(tmp_path):
    d = layout_dir(tmp_path)
    os.utime(d / "declared" / f"{R}.json", (T0 + 1000, T0 + 1000))  # declared after the records were written
    s = service(d, tmp_path)
    s.scan_sync()
    assert s.run_detail(R)["run"]["checks"]["declared_before_outputs"]["state"] == "fail"


# ---------------------------------------------------------------------------- append vs change

def _jsonl(d: Path) -> Path:
    return d / "records" / R / "item.jsonl"


def test_whole_line_append_is_recorded_as_appended_not_changed(tmp_path):
    d = layout_dir(tmp_path)
    s = service(d, tmp_path)
    s.scan_sync()
    with open(_jsonl(d), "ab") as f:
        f.write(b'{"item_id":"case-9","run_id":"R-layout-1"}\n')
    counts = s.scan_sync()
    assert counts["appended"] == 1 and counts["changed"] == 0
    rec = s.ledger.records[-1]
    assert rec["type"] == "evidence.appended" and rec["payload"]["previous_size"] < rec["payload"]["size"]
    run = s.run_detail(R)["run"]
    assert run["checks"]["unchanged"]["state"] == "pass" and len(run["appended"]) == 1 and not run["changed"]
    assert rebuild(d, io.StringIO()) == 0
    assert service(d, tmp_path).state == "ok"  # and the chain written around it verifies after a restart


@pytest.mark.parametrize("edit", [
    lambda b: b.replace(b'"passed":false', b'"passed":true') + b'{"item_id":"x"}\n',  # edit + append
    lambda b: b + b'{"item_id":"x"',  # a partial line is not an append
    lambda b: b[: len(b) // 2],  # shrink
])
def test_anything_but_a_whole_line_append_is_a_change(tmp_path, edit):
    d = layout_dir(tmp_path)
    s = service(d, tmp_path)
    s.scan_sync()
    _jsonl(d).write_bytes(edit(_jsonl(d).read_bytes()))
    counts = s.scan_sync()
    assert counts["changed"] == 1 and counts["appended"] == 0
    assert s.run_detail(R)["run"]["checks"]["unchanged"]["state"] == "fail"
    assert rebuild(d, io.StringIO()) == 1


def test_rebuild_catches_an_edit_after_an_append(tmp_path):
    d = layout_dir(tmp_path)
    s = service(d, tmp_path)
    s.scan_sync()
    with open(_jsonl(d), "ab") as f:
        f.write(b'{"item_id":"case-9"}\n')
    s.scan_sync()
    _jsonl(d).write_bytes(_jsonl(d).read_bytes().replace(b"case-1", b"case-X"))  # not scanned again
    out = io.StringIO()
    assert rebuild(d, out) == 1 and "is no longer the start of the file" in out.getvalue()


def test_non_jsonl_records_never_count_as_appends(tmp_path):
    d = layout_dir(tmp_path)
    s = service(d, tmp_path)
    s.scan_sync()
    obj = next((d / "records" / R / "item").iterdir())
    obj.write_bytes(obj.read_bytes() + b'{"item_id":"x"}\n')
    assert s.scan_sync()["changed"] == 1


def test_local_prefix_sha_requires_a_line_boundary(tmp_path):
    d = layout_dir(tmp_path)
    src = LocalSource(d)
    a = next(x for x in src.list_artifacts() if x.path.endswith("item.jsonl"))
    first_line = _jsonl(d).read_bytes().split(b"\n")[0] + b"\n"
    import hashlib

    assert src.prefix_sha256(a, len(first_line)) == hashlib.sha256(first_line).hexdigest()
    assert src.prefix_sha256(a, len(first_line) - 1) is None
    assert src.prefix_sha256(a, a.size + 10) is None


# ---------------------------------------------------------------------------- the independent check agrees

def _fixture_for_check_unit(tmp_path: Path) -> tuple[Path, EvidenceService]:
    """tools/tests/fixtures/units/R-objects (evidence + fake AWS), ingested by the REAL EvidenceService."""
    src = TOOLS / "tests" / "fixtures" / "units" / "R-objects"
    fx = tmp_path / "fx"
    shutil.copytree(src / "evidence", fx / "evidence")
    shutil.copy(src / "aws.json", fx / "aws.json")
    s = EvidenceService(directory=fx / "ledger", key_path=tmp_path / "k", source=LocalSource(fx / "evidence"))
    assert s.scan_sync()["new"] == 9
    return fx, s


def _check_unit(fx: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(TOOLS / "check_unit.py"), "t3.deny-fixture-R", "--offline", str(fx)],
                          capture_output=True, text=True, cwd=REPO)


def test_check_unit_verifies_a_ledger_written_by_the_real_service(tmp_path):
    fx, _ = _fixture_for_check_unit(tmp_path)
    r = _check_unit(fx)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "Ed25519 signatures verified with public_keys.json" in r.stdout
    assert "9 of 9 evidence files are in the ledger" in r.stdout


def test_check_unit_and_the_service_both_see_a_tampered_record(tmp_path):
    fx, s = _fixture_for_check_unit(tmp_path)
    p = next((fx / "evidence" / "records" / "t3.deny-fixture-R" / "item").iterdir())
    p.write_bytes(p.read_bytes().replace(b'"passed":true', b'"passed":false'))
    r = _check_unit(fx)
    assert r.returncode == 1 and "[d] FAIL" in r.stdout, r.stdout
    assert s.scan_sync()["changed"] == 1  # the dashboard flags the same file


def test_check_unit_rejects_a_ledger_with_a_forged_signature(tmp_path):
    fx, _ = _fixture_for_check_unit(tmp_path)
    led = fx / "ledger" / "ledger.jsonl"
    lines = led.read_text().splitlines()
    rec = json.loads(lines[0])
    rec["sig"] = "00" * 64
    lines[0] = json.dumps(rec, sort_keys=True, ensure_ascii=False)
    led.write_text("\n".join(lines) + "\n")
    r = _check_unit(fx)
    assert r.returncode == 1 and "signature does not verify" in r.stdout, r.stdout
