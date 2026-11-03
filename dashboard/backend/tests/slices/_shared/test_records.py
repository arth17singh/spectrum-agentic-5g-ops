"""RecordsReader: both record layouts merged in order, absent vs broken, file and (stubbed) S3."""

import io
import json
from pathlib import Path

import pytest

from spectrum.slices._shared.records import DEFAULT_ROOT, RecordsError, RecordsReader

R = "run-a"


def write(root: Path, rel: str, obj) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(obj if isinstance(obj, str) else json.dumps(obj))


def rec(i, t, **kw):
    return {"kind": "change", "run_id": R, "i": i, "created_at": t, **kw}


@pytest.fixture
def store(tmp_path):
    write(tmp_path, f"declared/{R}.json", {"run_id": R, "n_items": 3})
    write(tmp_path, f"results/{R}.json", {"unit": "t3.deny", "passed": True})
    write(tmp_path, "results/run-only-result.json", {"unit": "x"})
    write(tmp_path, "models/t2-resp/v1.json", {"model_id": "t2-resp", "version": "v1"})
    # batch layout (jsonl) and Lambda layout (one object per record), interleaved in time
    write(tmp_path, f"records/{R}/change.jsonl", "\n".join(json.dumps(r) for r in [
        rec(1, "2026-11-02T20:00:01Z"), rec(4, "2026-11-02T20:00:04Z"), rec(9, None)]) + "\n\n")
    write(tmp_path, f"records/{R}/change/k-b.json", rec(3, "2026-11-02T20:00:03+00:00"))
    write(tmp_path, f"records/{R}/change/k-a.json", rec(2, "2026-11-02T20:00:02.500Z"))
    write(tmp_path, f"records/{R}/change/k-c.json", rec(5, "2026-11-02T20:00:04Z"))  # time tie with i=4
    write(tmp_path, f"records/{R}/decision.jsonl", json.dumps({"decision": "DENY", "time": "2026-11-02T20:00:00Z"}))
    write(tmp_path, "records/run-b/change/x.json", rec(0, "2026-11-02T19:00:00Z", run_id="run-b"))
    return tmp_path


def test_file_root_reads_every_kind(store):
    r = RecordsReader(f"file://{store}")
    assert r.source() == {"kind": "file", "root": str(store)}
    assert r.runs() == [R, "run-b", "run-only-result"]
    assert r.declared(R)["n_items"] == 3 and r.result(R)["passed"] is True
    assert r.model("t2-resp", "v1")["version"] == "v1"
    assert r.declared("missing") is None and r.result("run-b") is None and r.model("t2-resp", "v9") is None
    assert r.records(R, "decision")[0]["decision"] == "DENY"
    assert r.records(R, "control") == [] and r.records("missing", "change") == []


def test_both_layouts_merge_in_time_then_key_order(store):
    r = RecordsReader(str(store))  # a bare absolute path works too
    # tie at :04 -> jsonl key "records/run-a/change.jsonl#..." sorts before "records/run-a/change/k-c.json";
    # the record with no time goes last
    assert [x["i"] for x in r.records(R, "change")] == [1, 2, 3, 4, 5, 9]
    assert [x["i"] for x in r.all_records("change")] == [0, 1, 2, 3, 4, 5, 9]
    assert r.records(R, "change") == r.records(R, "change")  # deterministic


def test_env_root_and_default_root(store, monkeypatch):
    monkeypatch.setenv("SPECTRUM_RECORDS_ROOT", f"file://{store}")
    assert RecordsReader().runs()[0] == R
    monkeypatch.delenv("SPECTRUM_RECORDS_ROOT")
    assert RecordsReader().source() == {"kind": "file", "root": str(DEFAULT_ROOT)}
    assert (DEFAULT_ROOT / "public_keys.json").is_file()  # the repo's evidence/ directory


def test_malformed_data_is_an_error_not_empty(store):
    write(store, f"records/{R}/change.jsonl", '{"i": 1}\n{not json\n')
    with pytest.raises(RecordsError, match="line 2"):
        RecordsReader(str(store)).records(R, "change")
    write(store, f"declared/{R}.json", "{")
    with pytest.raises(RecordsError, match=f"declared/{R}.json"):
        RecordsReader(str(store)).declared(R)


@pytest.mark.parametrize("bad", ["../x", "a/b", "", ".hidden", "x" * 300, "a..b"])
def test_names_cannot_escape_the_root(store, bad):
    with pytest.raises(ValueError):
        RecordsReader(str(store)).records(bad, "change")
    with pytest.raises(ValueError):
        RecordsReader(str(store)).records(R, bad)


def test_relative_root_is_refused():
    with pytest.raises(ValueError):
        RecordsReader("evidence")


# --------------------------------------------------------------------------- S3 (stub client, no AWS)

class ClientError(Exception):
    def __init__(self, code):
        super().__init__(f"An error occurred ({code})")
        self.response = {"Error": {"Code": code}}


class FakeS3:
    """list_objects_v2 paginator (Delimiter="/", 2 keys per page) + get_object over a dict of keys."""

    def __init__(self, objects, fail=None):
        self.objects, self.fail, self.calls = objects, fail, []

    def get_object(self, Bucket, Key):
        self.calls.append(("get", Bucket, Key))
        if self.fail:
            raise ClientError(self.fail)
        if Key not in self.objects:
            raise ClientError("NoSuchKey")
        return {"Body": io.BytesIO(self.objects[Key].encode())}

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return self

    def paginate(self, Bucket, Prefix, Delimiter):
        self.calls.append(("list", Bucket, Prefix))
        if self.fail:
            raise ClientError(self.fail)
        files, dirs = [], set()
        for k in sorted(self.objects):
            if k.startswith(Prefix):
                rest = k[len(Prefix):]
                if Delimiter in rest:
                    dirs.add(Prefix + rest.split(Delimiter)[0] + Delimiter)
                else:
                    files.append({"Key": k})
        items = [("f", f) for f in files] + [("d", {"Prefix": d}) for d in sorted(dirs)]
        for i in range(0, max(len(items), 1), 2):
            chunk = items[i:i + 2]
            yield {"Contents": [x for t, x in chunk if t == "f"], "CommonPrefixes": [x for t, x in chunk if t == "d"]}


def s3_objects(store: Path, prefix: str) -> dict:
    return {prefix + str(p.relative_to(store)): p.read_text() for p in store.rglob("*") if p.is_file()}


def test_s3_root_matches_the_file_root(store):
    fake = FakeS3(s3_objects(store, "team13/evidence/"))
    s3 = RecordsReader("s3://bkt/team13/evidence/", client=fake)
    f = RecordsReader(str(store))
    assert s3.source() == {"kind": "s3", "root": "s3://bkt/team13/evidence/"}
    assert s3.runs() == f.runs()
    assert s3.records(R, "change") == f.records(R, "change")
    assert s3.all_records("change") == f.all_records("change")
    assert s3.declared(R) == f.declared(R) and s3.model("t2-resp", "v1") == f.model("t2-resp", "v1")
    assert s3.result("run-b") is None  # NoSuchKey is "absent"
    assert all(c[1] == "bkt" for c in fake.calls)


@pytest.mark.parametrize("code", ["AccessDenied", "ExpiredToken", "SlowDown"])
def test_s3_errors_are_loud(store, code):
    s3 = RecordsReader("s3://bkt/p", client=FakeS3({}, fail=code))
    for call in (s3.runs, lambda: s3.records(R, "change"), lambda: s3.declared(R)):
        with pytest.raises(RecordsError, match=code):
            call()


def test_s3_is_refused_offline_without_a_client(monkeypatch):
    monkeypatch.setenv("SPECTRUM_OFFLINE", "1")
    with pytest.raises(RecordsError, match="OFFLINE"):
        RecordsReader("s3://bkt/p/").runs()
