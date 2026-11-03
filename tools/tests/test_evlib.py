"""evlib: local and S3 (fake) round trips, refusal to overwrite, validation, append-only, metrics."""

import json
import math
import sys
import types

import pytest

import evlib
import unitfix
from evfake import FakeAws

HEAD = evlib.code_sha()


def decl(run_id="t3.deny-test-1", **over):
    d = {"run_id": run_id, "unit": "t3.deny", "track": "track-3",
         "target": {"metric": "pass_rate", "op": ">=", "value": 0.8, "unit": ""}, "n_items": 3,
         "test_set_sha256": "a" * 64, "model_id": "m", "grader": "g", "code_sha": HEAD, "region": "us-east-2"}
    d.update(over)
    return d


def item(i, passed=True, **over):
    r = {"item_id": f"it-{i}", "input_sha256": "b" * 64, "output_sha256": "c" * 64, "grader": "g", "backend": "x",
         "passed": passed, "score": 1.0 if passed else 0.0, "latency_ms": 100 * i,
         "aws_time": "2026-11-02T18:10:00Z", "aws_time_source": "cloudwatch:/team13/checkpoint"}
    r.update(over)
    return r


@pytest.fixture
def local(tmp_path):
    return evlib.LocalStore(tmp_path / "ev")


@pytest.fixture
def s3():
    aws = FakeAws(now="2026-11-02T18:00:00Z")
    aws.s3.page_size = 2  # exercise pagination
    return evlib.S3Store("s3://bkt/evidence/", client=aws.client("s3")), aws


# ---------------------------------------------------------------------------- local

def test_local_round_trip(local):
    p = evlib.write_declared(decl(), local)
    assert p.endswith("declared/t3.deny-test-1.json")
    d = evlib.read_declared("t3.deny-test-1", local)
    assert d["kind"] == "declared" and d["team"] == "team13" and d["schema_version"] == "1"
    assert d["synthetic"] is False and d["aws_time"] is None
    evlib.parse_time(d["declared_at"])  # stamped, UTC
    evlib.append_records("t3.deny-test-1", "item", [item(1), item(2)], local)
    evlib.append_records("t3.deny-test-1", "item", [item(3, False)], local)
    its = evlib.read_records("t3.deny-test-1", "item", local)
    assert [i["item_id"] for i in its] == ["it-1", "it-2", "it-3"]
    assert all(i["run_id"] == "t3.deny-test-1" and i["kind"] == "item" for i in its)
    assert evlib.record_kinds("t3.deny-test-1", local) == ["item"]
    res = evlib.make_result(d, its, "tools/test", HEAD)
    assert math.isclose(res["value"], 2 / 3) and res["passed"] is False and res["n_items"] == 3
    evlib.write_result(res, local)
    assert evlib.read_result("t3.deny-test-1", local)["value"] == res["value"]


def test_declared_and_result_are_never_overwritten(local):
    evlib.write_declared(decl(), local)
    with pytest.raises(FileExistsError):
        evlib.write_declared(decl(), local)
    r = {"run_id": "t3.deny-test-1", "unit": "t3.deny", "metric": "pass_rate", "value": 1.0,
         "target": {"metric": "pass_rate", "op": ">=", "value": 0.8, "unit": ""}, "passed": True, "n_items": 1,
         "checker": {"script": "x", "code_sha": HEAD}}
    evlib.write_result(r, local)
    with pytest.raises(FileExistsError):
        evlib.write_result(r, local)


def test_appends_only_grow_by_whole_lines(local):
    evlib.append_records("r1", "item", [item(1)], local)
    first = (local.root / "records/r1/item.jsonl").read_bytes()
    evlib.append_records("r1", "item", [item(2)], local)
    after = (local.root / "records/r1/item.jsonl").read_bytes()
    assert after.startswith(first) and first.endswith(b"\n") and after.count(b"\n") == 2


@pytest.mark.parametrize("bad,match", [
    (decl(test_set_sha256="xyz"), "test_set_sha256"),
    (decl(code_sha="abc"), "code_sha"),
    (decl(target={"metric": "pass_rate", "op": "~", "value": 0.8, "unit": ""}), "target.op"),
    (decl(target={"metric": "pass_rate", "op": ">=", "value": "0.8", "unit": ""}), "target.value"),
    (decl(run_id="bad id with spaces"), "run_id"),
    ({k: v for k, v in decl().items() if k != "grader"}, "grader"),
])
def test_invalid_declarations_are_refused_and_nothing_is_written(local, bad, match):
    with pytest.raises(evlib.EvidenceError, match=match):
        evlib.write_declared(bad, local)
    assert not (local.root / "declared").exists() or not any((local.root / "declared").iterdir())


def test_one_bad_record_writes_nothing(local):
    with pytest.raises(evlib.EvidenceError):
        evlib.append_records("r1", "item", [item(1), item(2, latency_ms="slow")], local)
    with pytest.raises(evlib.EvidenceError, match="NaN|JSON|number"):
        evlib.append_records("r1", "item", [item(1, score=float("nan"))], local)
    with pytest.raises(evlib.EvidenceError, match="does not match"):
        evlib.append_records("r1", "item", [item(1, run_id="other")], local)
    with pytest.raises(evlib.EvidenceError, match="aws_time_source"):
        evlib.append_records("r1", "item", [item(1, aws_time_source=None)], local)
    with pytest.raises(evlib.EvidenceError, match="decision"):
        evlib.append_records("r1", "decision", [{"case_id": "c", "time": "2026-11-02T18:00:00Z", "agent": "a",
                                                 "tool": "t", "decision": "MAYBE", "source": "aws/spans"}], local)
    assert not (local.root / "records").exists()


def test_model_registry_entry(local):
    m = {"model_id": "t2-response", "version": "v0", "model_kind": "t2-response", "artifact_uri": "s3://b/k",
         "artifact_sha256": "d" * 64, "status": "candidate", "trained_on": "maveric-official", "metrics": {},
         "run_id": "t2.train-1"}
    assert evlib.write_model(m, local).endswith("models/t2-response/v0.json")
    assert evlib.read_model("t2-response", "v0", local)["kind"] == "model"
    with pytest.raises(FileExistsError):
        evlib.write_model(m, local)


def test_default_store_from_env(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError):
        evlib.default_store()
    monkeypatch.setenv("EVIDENCE_DIR", str(tmp_path))
    assert isinstance(evlib.default_store(), evlib.LocalStore)
    monkeypatch.setenv("EVIDENCE_S3", "s3://b/evidence")
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    s = evlib.default_store()
    assert isinstance(s, evlib.S3Store) and s.prefix == "evidence/" and s.region == "us-west-2"


def test_team13_contracts_is_used_when_importable(monkeypatch, local):
    seen = []

    def validate_record(rec):
        seen.append(rec["kind"])
        if rec.get("model_id") == "rejected":
            raise ValueError("nope")

    monkeypatch.setitem(sys.modules, "team13_contracts", types.SimpleNamespace(validate_record=validate_record))
    evlib.write_declared(decl(), local)
    assert seen == ["declared"]
    with pytest.raises(evlib.EvidenceError, match="team13_contracts rejected"):
        evlib.write_declared(decl(run_id="r2", model_id="rejected"), local)


def test_import_is_light():
    import subprocess

    code = ("import sys; sys.path.insert(0, %r); import evlib; "
            "bad = [m for m in ('boto3', 'botocore', 'spectrum', 'numpy') if m in sys.modules]; print(bad)"
            % str(evlib.__file__.rsplit('/', 1)[0]))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"


# ---------------------------------------------------------------------------- S3 (fake)

def test_s3_round_trip_and_refusals(s3):
    store, aws = s3
    uri = evlib.write_declared(decl(), store)
    assert uri == "s3://bkt/evidence/declared/t3.deny-test-1.json"
    with pytest.raises(FileExistsError):
        evlib.write_declared(decl(), store)
    for i in range(5):
        uris = evlib.append_records("t3.deny-test-1", "item", [item(i, created_at=f"2026-11-02T18:1{i}:00.000Z")], store)
        assert len(uris) == 1 and uris[0].startswith("s3://bkt/evidence/records/t3.deny-test-1/item/")
        assert uris[0].endswith(".json")  # A11: one object per record, never a JSONL append on S3
    evlib.append_records("t3.deny-test-1", "decision", [{"case_id": "c1", "time": "2026-11-02T18:00:00Z",
                                                         "agent": "lead", "tool": "ran___apply_change",
                                                         "decision": "DENY", "source": "aws/spans"}], store)
    assert [i["item_id"] for i in evlib.read_records("t3.deny-test-1", "item", store)] == [f"it-{i}" for i in range(5)]
    assert evlib.record_kinds("t3.deny-test-1", store) == ["decision", "item"]  # paginated listing (page size 2)
    assert evlib.read_declared("nope", store) is None and evlib.read_records("nope", "item", store) == []
    puts = [c for c in aws.calls if c[1] == "PutObject"]
    assert len(puts) == 1 + 1 + 5 + 1  # including the refused overwrite
    assert not any(c[1] == "GetObject" for c in aws.calls[:3])  # writing never read-modify-writes
    with pytest.raises(evlib.EvidenceError, match="cannot append"):
        evlib.append_records("t3.deny-test-1", "item", [item(7)], store, layout="jsonl")


def test_s3_record_keys_are_deterministic_and_never_clobber(s3):
    store, aws = s3
    r = item(1, created_at="2026-11-02T18:10:00.000Z")
    a = evlib.append_records("r1", "item", [r], store)
    b = evlib.append_records("r1", "item", [r], store)  # a retried Lambda: same bytes, same key, no-op
    assert a == b and len(evlib.read_records("r1", "item", store)) == 1
    key = a[0][len("s3://bkt/evidence/"):]
    body = aws.s3.objects[("bkt", "evidence/" + key)]["body"]
    assert key == f"records/r1/item/{evlib.record_key(body)}.json" and body.endswith(b"\n")
    # someone puts different bytes under that key: the next identical write must not accept it silently
    aws.s3.put("bkt", "evidence/" + key, b'{"forged": true}\n', "2026-11-02T18:00:01Z")
    with pytest.raises(FileExistsError):
        evlib.append_records("r1", "item", [r], store)


def test_both_layouts_are_merged_by_readers(s3, local):
    for store in (s3[0], local):
        evlib.write_records_jsonl("r1", "item", [item(1), item(2)], store)
        with pytest.raises(FileExistsError):
            evlib.write_records_jsonl("r1", "item", [item(1)], store)
        evlib.append_records("r1", "item", [item(4, created_at="2026-11-02T18:20:00.000Z"),
                                            item(3, created_at="2026-11-02T18:10:00.000Z")], store, layout="objects")
        assert [r["item_id"] for r in evlib.read_records("r1", "item", store)] == ["it-1", "it-2", "it-3", "it-4"]
        assert evlib.record_kinds("r1", store) == ["item"]


# ---------------------------------------------------------------------------- metrics (writer side)

def test_writer_metrics_golden_values():
    its = [item(i, passed=i != 4, latency_ms=lat, cost_usd=0.001)
           for i, lat in enumerate([100, 200, 300, 400, 1000], start=1)]
    assert evlib.summarize_items(its, "pass_rate") == 0.8
    assert evlib.summarize_items(its, "fail_count") == 1
    assert evlib.summarize_items(its, "p95_latency_ms") == 1000  # nearest rank: ceil(0.95*5)=5th
    assert evlib.summarize_items(its, "p50_latency_ms") == 300
    assert math.isclose(evlib.summarize_items(its, "cost_per_1k_usd"), 1.0)
    with pytest.raises(evlib.EvidenceError, match="unknown metric"):
        evlib.summarize_items(its, "vibes")


def test_fixture_records_are_contract_valid(tmp_path):
    fx = unitfix.build(tmp_path, "R")
    ev = fx / "evidence"
    for p in ev.rglob("*"):
        if p.suffix == ".jsonl":
            for line in p.read_text().splitlines():
                assert evlib.builtin_errors(json.loads(line)) == [], p
        elif p.suffix == ".json":
            assert evlib.builtin_errors(json.loads(p.read_text())) == [], p
