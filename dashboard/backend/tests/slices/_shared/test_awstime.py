"""AWS time per record: one source per record, a runner or Lambda clock is never AWS time, S3 stored time is the
fallback for a record read from its own object (never from a .jsonl batch file). Fixture records cover each source,
through the pure helper, the reader, and the Trace and Twin View routes."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from spectrum.slices import trace, twinview
from spectrum.slices._shared import awstime
from spectrum.slices._shared.records import RecordsReader
from spectrum.slices.twinview import routes as tv_routes

RUN = "t3.fixture-20261103T000000Z"


def rec(**kw):
    base = {"schema_version": "1", "run_id": RUN, "team": "team13", "synthetic": False, "case_id": "case-1",
            "created_at": "2026-11-03T06:00:00.000Z"}
    base.update(kw)
    return base


# ---------------------------------------------------------------------------------------------- the pure helper

@pytest.mark.parametrize("label,expect", [
    ("cloudwatch:/team13/checkpoint", "cloudwatch"),
    ("agentcore:policy.createdAt", "policy_createdAt"),
    ("gateway:http-date", "gateway_date"),
    ("s3:LastModified evidence/outputs/x", "s3_stored"),
    ("lambda:clock", None),          # the Lambda's own clock is not AWS time
    ("dynamodb:applied_at", None),   # nor is an attribute our code wrote
    ("sagemaker:TrainingEndTime", None),
    ("unknown", None), (None, None), ("", None), (5, None),
])
def test_classify(label, expect):
    assert awstime.classify(label) == expect


@pytest.mark.parametrize("src,expect", [
    ("cloudwatch:/team13/checkpoint", "cloudwatch"),
    ("agentcore:policy.createdAt", "policy_createdAt"),
    ("gateway:http-date", "gateway_date"),
])
def test_each_event_source_is_kept_as_the_aws_time(src, expect):
    t = awstime.resolve(rec(aws_time="2026-11-03T05:50:00.100Z", aws_time_source=src))
    assert t == {"aws_time": "2026-11-03T05:50:00.100Z", "aws_time_source": expect, "runner_time": "2026-11-03T06:00:00.000Z"}


def test_a_lambda_clock_is_the_runner_clock_not_aws_time():
    r = rec(aws_time="2026-11-03T05:31:52.546Z", aws_time_source="lambda:clock", created_at=None)
    t = awstime.resolve(r)
    assert t["aws_time"] is None and t["aws_time_source"] == "none"
    assert t["runner_time"] == "2026-11-03T05:31:52.546Z"  # shown as the runner clock
    # with an S3 stored time the record falls back to that, still labelled as stored
    t = awstime.resolve({**r, awstime.STORED_KEY: "2026-11-03T05:31:53Z"})
    assert (t["aws_time"], t["aws_time_source"]) == ("2026-11-03T05:31:53Z", "s3_stored")


def test_no_source_or_unparseable_time_is_not_trusted():
    assert awstime.resolve(rec(aws_time="2026-11-03T05:50:00Z"))["aws_time_source"] == "none"  # no source: unattributed
    assert awstime.resolve(rec(aws_time="not a time", aws_time_source="cloudwatch:/x"))["aws_time_source"] == "none"
    assert awstime.resolve(rec(**{awstime.STORED_KEY: "garbage"}))["aws_time_source"] == "none"


def test_s3_stored_is_the_fallback_and_runner_time_stays_separate():
    t = awstime.resolve(rec(**{awstime.STORED_KEY: "2026-11-03T06:00:03Z"}))
    assert t == {"aws_time": "2026-11-03T06:00:03Z", "aws_time_source": "s3_stored", "runner_time": "2026-11-03T06:00:00.000Z"}


def test_an_aws_time_equal_to_the_record_time_is_not_also_the_runner_clock():
    t = awstime.resolve(rec(time="2026-11-03T05:50:00.100Z", created_at=None, aws_time="2026-11-03T05:50:00.100Z",
                            aws_time_source="cloudwatch:/team13/checkpoint"))
    assert t["runner_time"] is None


def test_summary_counts_by_source():
    rows = [{"aws_time_source": s} for s in ("cloudwatch", "cloudwatch", "s3_stored", "policy_createdAt", "gateway_date", "none")]
    s = awstime.summary(rows)
    assert (s["n"], s["k"]) == (6, 5)
    assert s["by_source"] == {"cloudwatch": 2, "s3_stored": 1, "policy_createdAt": 1, "gateway_date": 1, "none": 1}
    assert s["text"] == "AWS time on 5 of 6 (CloudWatch 2, S3 stored 1, policy createdAt 1, gateway Date 1)"
    assert awstime.summary([])["text"] == "AWS time on 0 of 0"
    assert awstime.summary([{"aws_time_source": "none"}])["text"] == "AWS time on 0 of 1"


# ---------------------------------------------------------------------------------------------- a mirrored directory

def _put(root: Path, rel: str, obj) -> int:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj) if not isinstance(obj, str) else obj
    p.write_text(text)
    return len(text.encode())


def _evidence(tmp_path: Path) -> Path:
    """One run with one record of each kind of time, plus the sync manifest S3 LastModified is read from."""
    man = {}

    def put(rel, obj, lm="2026-11-03T06:00:05Z", manifest=True, size=None):
        n = _put(tmp_path, rel, obj)
        if manifest:
            man[rel] = {"etag": "x", "last_modified": lm, "sha256": "0" * 64, "size": n if size is None else size}

    d = f"records/{RUN}/decision/"
    put(d + "cw.json", rec(kind="decision", tool="ran___observe", decision="ALLOW", time="2026-11-03T05:50:00.100Z",
                           aws_time="2026-11-03T05:50:00.100Z", aws_time_source="cloudwatch:/team13/checkpoint"))
    put(d + "gw.json", rec(kind="decision", tool="ran___case_status", decision="ALLOW", time="2026-11-03T05:50:01Z",
                           aws_time="2026-11-03T05:50:01Z", aws_time_source="gateway:http-date"))
    put(d + "stored.json", rec(kind="decision", tool="front___start_case", decision="ALLOW", time="2026-11-03T05:49:59Z",
                               aws_time=None, aws_time_source=None), lm="2026-11-03T06:00:06Z")
    put(d + "nomanifest.json", rec(kind="decision", tool="ran___observe", decision="DENY", time="2026-11-03T05:50:02Z"), manifest=False)
    put(d + "badsize.json", rec(kind="decision", tool="ran___observe", decision="DENY", time="2026-11-03T05:50:03Z"), size=1)
    c = f"records/{RUN}/change/"
    put(c + "lambda.json", rec(kind="change", change_id="chg-1", item_id="step-0", created_at="2026-11-03T05:50:30.545Z",
                               aws_time="2026-11-03T05:50:30.546Z", aws_time_source="lambda:clock",
                               change=[{"cell": 1, "d_power_db": 1.0, "d_tilt_deg": 0.0}], change_text="cell=1",
                               twin={"kind": "maveric-inprocess", "step": 0}), lm="2026-11-03T05:50:31Z")
    # a record in a .jsonl batch file: the file's time bounds many records, so none of them gets it
    _put(tmp_path, f"records/{RUN}/change.jsonl",
         json.dumps(rec(kind="change", change_id="chg-2", item_id="step-1", created_at="2026-11-03T05:50:40Z",
                        change=[{"cell": 2, "d_power_db": 1.0, "d_tilt_deg": 0.0}], change_text="cell=2",
                        twin={"kind": "maveric-inprocess", "step": 1})) + "\n")
    man[f"records/{RUN}/change.jsonl"] = {"etag": "x", "last_modified": "2026-11-03T05:59:00Z", "sha256": "0" * 64, "size": 1}
    k = f"records/{RUN}/control/"
    put(k + "pause.json", rec(kind="control", case_id=None, action="pause", model_id="m", version="v1", time="2026-11-03T05:50:20.591Z",
                              created_at="2026-11-03T05:50:31.773Z", aws_time="2026-11-03T05:50:20.591Z",
                              aws_time_source="agentcore:policy.createdAt"))
    put("declared/" + RUN + ".json", {"run_id": RUN, "twin": {"config": None}})
    (tmp_path / ".sync").mkdir()
    (tmp_path / ".sync" / "manifest.json").write_text(json.dumps(man))
    return tmp_path


def test_reader_gives_the_manifest_s3_time_only_to_per_record_objects(tmp_path):
    r = RecordsReader(str(_evidence(tmp_path)))
    by = {(x["tool"], x["time"]): at for x, at in r.all_records_stored("decision")}
    assert by[("front___start_case", "2026-11-03T05:49:59Z")] == "2026-11-03T06:00:06Z"
    assert by[("ran___observe", "2026-11-03T05:50:00.100Z")] == "2026-11-03T06:00:05Z"
    assert by[("ran___observe", "2026-11-03T05:50:02Z")] is None   # not in the manifest
    assert by[("ran___observe", "2026-11-03T05:50:03Z")] is None   # the file is not what S3 stored (size differs)
    changes = {x["change_id"]: at for x, at in r.all_records_stored("change")}
    assert changes == {"chg-1": "2026-11-03T05:50:31Z", "chg-2": None}  # the .jsonl record gets none
    assert r.declared_stored(RUN) == "2026-11-03T06:00:05Z"
    # same order and same records as all_records
    assert [x for x, _ in r.all_records_stored("decision")] == r.all_records("decision")


def test_reader_without_a_manifest_has_no_stored_times(tmp_path):
    root = _evidence(tmp_path)
    (root / ".sync" / "manifest.json").unlink()
    r = RecordsReader(str(root))
    assert all(at is None for _, at in r.all_records_stored("decision"))
    assert r.declared_stored(RUN) is None


def test_the_manifest_is_parsed_once_until_it_changes(tmp_path, monkeypatch):
    from spectrum.slices._shared import records as rmod

    root = _evidence(tmp_path)
    calls = []
    real = json.loads
    monkeypatch.setattr(rmod.json, "loads", lambda t, *a, **k: (calls.append(len(t)), real(t, *a, **k))[1])
    r = RecordsReader(str(root))
    r.all_records_stored("decision")
    n1 = len([c for c in calls if c > 1000])
    r.all_records_stored("change")
    assert len([c for c in calls if c > 1000]) == n1  # the cached parse was reused


class FakeS3:
    def __init__(self, objects):
        self.objects, self.calls = objects, []

    def get_object(self, Bucket, Key):
        import io
        self.calls.append(("get", Key))
        if Key not in self.objects:
            e = Exception("NoSuchKey")
            e.response = {"Error": {"Code": "NoSuchKey"}}
            raise e
        return {"Body": io.BytesIO(self.objects[Key][0].encode())}

    def get_paginator(self, name):
        return self

    def paginate(self, Bucket, Prefix, Delimiter=None):
        self.calls.append(("list", Prefix, Delimiter))
        if Delimiter:
            dirs = sorted({Prefix + k[len(Prefix):].split("/")[0] + "/" for k in self.objects if k.startswith(Prefix) and "/" in k[len(Prefix):]})
            files = [{"Key": k} for k in sorted(self.objects) if k.startswith(Prefix) and "/" not in k[len(Prefix):]]
            yield {"CommonPrefixes": [{"Prefix": d} for d in dirs], "Contents": files}
        else:
            yield {"Contents": [{"Key": k, "LastModified": v[1]} for k, v in sorted(self.objects.items()) if k.startswith(Prefix)]}


def test_s3_root_uses_the_object_last_modified_from_the_listing():
    t = datetime(2026, 11, 3, 6, 0, 7, 400000, tzinfo=timezone.utc)
    o = rec(kind="decision", tool="ran___observe", decision="ALLOW", time="2026-11-03T05:50:00Z")
    s3 = FakeS3({f"ev/records/{RUN}/decision/a.json": (json.dumps(o), t),
                 f"ev/declared/{RUN}.json": (json.dumps({"run_id": RUN}), t)})
    r = RecordsReader("s3://bucket/ev/", client=s3)
    assert [(x["tool"], at) for x, at in r.all_records_stored("decision")] == [("ran___observe", "2026-11-03T06:00:07Z")]
    assert r.declared_stored(RUN) == "2026-11-03T06:00:07Z"


# ---------------------------------------------------------------------------------------------- the routes

def _trace_client(monkeypatch, root):
    monkeypatch.setattr(trace, "_reader", lambda: RecordsReader(str(root)))
    app = FastAPI()
    app.include_router(trace.router)
    return TestClient(app)


def test_trace_events_carry_each_source_and_the_runner_clock_apart(monkeypatch, tmp_path):
    c = _trace_client(monkeypatch, _evidence(tmp_path))
    j = c.get("/api/trace/case/case-1").json()
    ev = {(e["tool"], e["time"]): e for e in j["timeline"] if e["kind"] != "control"}
    cw = ev[("ran___observe", "2026-11-03T05:50:00.100Z")]
    assert (cw["aws_time"], cw["aws_time_source"]) == ("2026-11-03T05:50:00.100Z", "cloudwatch") and cw["real"] is True
    assert cw["runner_time"] == "2026-11-03T06:00:00.000Z"
    gw = ev[("ran___case_status", "2026-11-03T05:50:01Z")]
    assert gw["aws_time_source"] == "gateway_date" and gw["real"] is True
    st = ev[("front___start_case", "2026-11-03T05:49:59Z")]
    assert (st["aws_time"], st["aws_time_source"]) == ("2026-11-03T06:00:06Z", "s3_stored")
    assert st["real"] is False  # stored time alone does not make the event real
    no = ev[("ran___observe", "2026-11-03T05:50:02Z")]
    assert (no["aws_time"], no["aws_time_source"]) == (None, "none")
    lam = next(e for e in j["timeline"] if e["kind"] == "change" and e["detail"]["change_id"] == "chg-1")
    assert (lam["aws_time"], lam["aws_time_source"]) == ("2026-11-03T05:50:31Z", "s3_stored")  # not the Lambda clock
    assert lam["runner_time"] == "2026-11-03T05:50:30.545Z" and lam["real"] is False
    jl = next(e for e in j["timeline"] if e["kind"] == "change" and e["detail"]["change_id"] == "chg-2")
    assert jl["aws_time_source"] == "none" and jl["aws_time"] is None  # .jsonl: no per-record S3 time
    ctl = next(e for e in j["timeline"] if e["kind"] == "control")
    assert ctl["aws_time_source"] == "policy_createdAt"
    # the private carrier field and the raw label never reach a response
    text = json.dumps(j)
    assert awstime.STORED_KEY not in text and "lambda:clock" not in text and "/team13/checkpoint" not in text.replace("log_group", "")


def test_trace_summary_is_k_of_n_by_source(monkeypatch, tmp_path):
    c = _trace_client(monkeypatch, _evidence(tmp_path))
    s = c.get("/api/trace/cases").json()["provenance"]["aws_time_summary"]
    assert s["by_source"] == {"cloudwatch": 1, "s3_stored": 2, "policy_createdAt": 1, "gateway_date": 1, "none": 3}
    assert (s["k"], s["n"]) == (5, 8)
    assert s["text"] == "AWS time on 5 of 8 (CloudWatch 1, S3 stored 2, policy createdAt 1, gateway Date 1)"
    case = c.get("/api/trace/case/case-1").json()
    assert case["provenance"]["aws_time_summary"]["n"] == 8
    # the case row in the listing carries its own summary
    row = c.get("/api/trace/cases").json()["cases"][0]
    assert row["provenance"]["aws_time_summary"]["k"] == 5


def test_trace_check_does_not_change_with_the_new_fields(monkeypatch, tmp_path):
    """The sequence is judged on `time` as before, whatever the AWS time says."""
    c = _trace_client(monkeypatch, _evidence(tmp_path))
    j = c.get("/api/trace/case/case-1").json()
    assert [e["time"] for e in j["timeline"] if e["kind"] == "change"] == ["2026-11-03T05:50:30.546Z", "2026-11-03T05:50:40Z"]


def test_twinview_windows_carry_aws_time_and_source(monkeypatch, tmp_path):
    root = _evidence(tmp_path)
    monkeypatch.setattr(tv_routes, "_reader", lambda: RecordsReader(str(root)))
    app = FastAPI()
    app.include_router(twinview.router)
    c = TestClient(app)
    j = c.get("/api/twinview/case/case-1").json()
    w = {x["change_id"]: x for x in j["windows"]}
    assert (w["chg-1"]["aws_time"], w["chg-1"]["aws_time_source"]) == ("2026-11-03T05:50:31Z", "s3_stored")
    assert w["chg-1"]["runner_time"] == "2026-11-03T05:50:30.545Z" and w["chg-1"]["real"] is False
    assert (w["chg-2"]["aws_time"], w["chg-2"]["aws_time_source"]) == (None, "none")
    s = j["provenance"]["aws_time_summary"]
    # two changes + the declaration (S3 stored): the declaration is a shown record too
    assert (s["k"], s["n"]) == (2, 3) and s["by_source"]["s3_stored"] == 2
    listing = c.get("/api/twinview/cases").json()
    assert listing["provenance"]["aws_time_summary"]["n"] == 3
    assert awstime.STORED_KEY not in json.dumps(j)
