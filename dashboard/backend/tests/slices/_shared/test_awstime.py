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
    rows = [{"aws_time_source": s} for s in ("cloudwatch", "cloudwatch", "s3_stored", "policy_createdAt", "gateway_date", "cloudwatch_attest", "none")]
    s = awstime.summary(rows)
    assert (s["n"], s["k"]) == (7, 6)
    assert s["by_source"] == {"cloudwatch": 2, "cloudwatch_attest": 1, "policy_createdAt": 1, "gateway_date": 1, "s3_stored": 1, "none": 1}
    assert s["text"] == "AWS time on 6 of 7 (CloudWatch 2, job attestation 1, policy createdAt 1, gateway Date 1, S3 stored 1)"
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
    assert s["by_source"] == {"cloudwatch": 1, "cloudwatch_attest": 0, "policy_createdAt": 1, "gateway_date": 1, "s3_stored": 2, "none": 3}
    assert (s["k"], s["n"]) == (5, 8)
    assert s["text"] == "AWS time on 5 of 8 (CloudWatch 1, policy createdAt 1, gateway Date 1, S3 stored 2)"
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


# ---------------------------------------------------------------------------------------------- job attestation

def test_priority_cloudwatch_then_attestation_then_policy_gateway_then_s3():
    cw = rec(aws_time="2026-11-03T05:50:00Z", aws_time_source="cloudwatch:/team13/checkpoint")
    full = {awstime.ATTEST_KEY: "2026-11-03T05:51:00Z", awstime.STORED_KEY: "2026-11-03T05:52:00Z"}
    assert awstime.resolve({**cw, **full})["aws_time_source"] == "cloudwatch"
    pol = rec(aws_time="2026-11-03T05:50:20Z", aws_time_source="agentcore:policy.createdAt")
    assert awstime.resolve({**pol, **full}) ["aws_time_source"] == "cloudwatch_attest"
    assert awstime.resolve({**pol, awstime.STORED_KEY: "2026-11-03T05:52:00Z"})["aws_time_source"] == "policy_createdAt"
    assert awstime.resolve(rec(**full))["aws_time_source"] == "cloudwatch_attest"
    assert awstime.resolve(rec(**full))["aws_time"] == "2026-11-03T05:51:00Z"
    assert awstime.resolve(rec(**{awstime.STORED_KEY: "2026-11-03T05:52:00Z"}))["aws_time_source"] == "s3_stored"
    assert "cloudwatch_attest" in awstime.EVENT_SOURCES and "s3_stored" not in awstime.EVENT_SOURCES


def _attest(root: Path, run: str, files: dict):
    d = root / ".sync" / "attest"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{run}.json").write_text(json.dumps({"run_id": run, "files": files}))


def _sha(root: Path, rel: str) -> str:
    import hashlib
    return hashlib.sha256((root / rel).read_bytes()).hexdigest()


def test_attestation_is_used_only_for_a_matching_file(tmp_path):
    root = _evidence(tmp_path)
    ok = f"records/{RUN}/decision/stored.json"
    other = f"records/{RUN}/decision/nomanifest.json"
    bad_state = f"records/{RUN}/decision/badsize.json"
    chg = f"records/{RUN}/change/lambda.json"
    _attest(root, RUN, {
        ok: {"job": "j", "log_time": "2026-11-03T06:00:09Z", "sha256": _sha(root, ok), "state": "match"},
        other: {"job": "j", "log_time": "2026-11-03T06:00:10Z", "sha256": "0" * 64, "state": "match"},       # sha differs
        bad_state: {"job": "j", "log_time": "2026-11-03T06:00:11Z", "sha256": _sha(root, bad_state), "state": "not attested"},
        chg: {"job": "j", "log_time": "2026-11-03T06:00:12Z", "sha256": _sha(root, chg), "state": "match"},
    })
    r = RecordsReader(str(root))
    by = {(x["tool"], x["time"]): m for x, m in r.all_records_times("decision")}
    assert by[("front___start_case", "2026-11-03T05:49:59Z")]["attested"] == "2026-11-03T06:00:09Z"
    assert by[("ran___observe", "2026-11-03T05:50:02Z")]["attested"] is None  # sha256 is not the file's
    assert by[("ran___observe", "2026-11-03T05:50:03Z")]["attested"] is None  # state is not "match"
    ch = {x["change_id"]: m for x, m in r.all_records_times("change")}
    assert ch["chg-1"]["attested"] == "2026-11-03T06:00:12Z" and ch["chg-2"]["attested"] is None  # .jsonl: never
    # an edited file no longer matches what the job attested
    (root / chg).write_text((root / chg).read_text() + " ")
    assert {x["change_id"]: m for x, m in RecordsReader(str(root)).all_records_times("change")}["chg-1"]["attested"] is None


def test_trace_shows_attestation_as_the_aws_time_and_counts_it_as_an_event_time(monkeypatch, tmp_path):
    root = _evidence(tmp_path)
    chg = f"records/{RUN}/change/lambda.json"
    _attest(root, RUN, {chg: {"job": "j", "log_time": "2026-11-03T05:50:30Z", "sha256": _sha(root, chg), "state": "match"}})
    j = _trace_client(monkeypatch, root).get("/api/trace/case/case-1").json()
    e = next(e for e in j["timeline"] if e["kind"] == "change" and e["detail"]["change_id"] == "chg-1")
    assert (e["aws_time"], e["aws_time_source"], e["real"]) == ("2026-11-03T05:50:30Z", "cloudwatch_attest", True)
    assert e["runner_time"] == "2026-11-03T05:50:30.545Z"
    assert j["provenance"]["aws_time_summary"]["by_source"]["cloudwatch_attest"] == 1
    assert "_attested_at" not in json.dumps(j)


def test_a_record_without_an_event_time_says_why():
    lam = rec(aws_time="2026-11-03T05:31:52.546Z", aws_time_source="lambda:clock", **{awstime.STORED_KEY: "2026-11-03T05:31:53Z"})
    assert "Lambda's own clock" in awstime.no_event_reason(lam) and "S3 stored" in awstime.no_event_reason(lam)
    assert awstime.no_event_reason(rec(run_id="t1.rl.viavi_operate-20261103T075351Z-25c0")) == awstime.VIAVI_REASON
    assert awstime.no_event_reason(rec(twin={"kind": "viavi"})) == awstime.VIAVI_REASON
    assert awstime.VIAVI_REASON == "ran outside AWS (through the team's browser session): no AWS-recorded time exists"
    assert awstime.no_event_reason(rec(aws_time="2026-11-03T05:50:00Z", aws_time_source="cloudwatch:/x")) is None
    assert awstime.no_event_reason(rec(**{awstime.ATTEST_KEY: "2026-11-03T05:50:00Z"})) is None
    assert "no AWS time found" in awstime.no_event_reason(rec())


def test_trace_lists_the_reasons_a_record_is_unverified(monkeypatch, tmp_path):
    c = _trace_client(monkeypatch, _evidence(tmp_path))
    why = {x["reason"]: x["n"] for x in c.get("/api/trace/cases").json()["provenance"]["unverified_reasons"]}
    assert why  # the lambda-clock change, the .jsonl change and the two unmirrored decisions are named, not hidden
    assert any("Lambda's own clock" in k for k in why) and sum(why.values()) == 5


# ---------------------------------------------------------------------------------------------- per-unit case status

def _case_status(monkeypatch, tmp_path, run, rows):
    """rows: [(kind, record)] for one run id; returns the /cases listing."""
    root = tmp_path
    for i, (kind, r) in enumerate(rows):
        _put(root, f"records/{run}/{kind}/{i:02d}.json", {**r, "run_id": run, "kind": kind})
    return _trace_client(monkeypatch, root).get("/api/trace/cases").json()


def _dec(case, tool, decision, t, **kw):
    return ("decision", rec(case_id=case, tool=tool, decision=decision, time=t, created_at=t, **kw))


def test_a_deny_case_is_complete_when_its_refusal_is_recorded(monkeypatch, tmp_path):
    run = "t3.deny-20261103T000000Z"
    j = _case_status(monkeypatch, tmp_path, run, [
        _dec("case-1", "ran___observe", "DENY", "2026-11-03T05:50:00Z"),
        _dec("(missing)", "front___start_case", "DENY", "2026-11-03T05:50:01Z"),
        _dec("no-case", "ran___observe", "DENY", "2026-11-03T05:50:02Z"),
        _dec("case-3", "ran___observe", "ALLOW", "2026-11-03T05:50:03Z")])
    by = {c["case_id"]: c for c in j["cases"]}
    assert by["case-1"]["check_passed"] and by["(missing)"]["check_passed"] and by["no-case"]["check_passed"]
    assert not by["case-3"]["check_passed"]  # an ALLOWed call is not a refusal
    assert by["(missing)"]["case_label"] == "calls without a case id (deny items that test the case-id rule)"
    assert by["case-1"]["case_label"] is None and by["case-1"]["unit"] == "t3.deny"
    d = _trace_client(monkeypatch, tmp_path).get(f"/api/trace/case/case-1?run_id={run}").json()
    assert d["check"]["by_unit"] is True and d["check"]["expected"] == ["refusal"] and d["check"]["missing"] == []


def test_a_deny_case_that_applied_a_change_is_not_complete(monkeypatch, tmp_path):
    run = "t3.deny-20261103T000001Z"
    j = _case_status(monkeypatch, tmp_path, run, [
        _dec("case-1", "ran___observe", "DENY", "2026-11-03T05:50:00Z"),
        ("change", rec(case_id="case-1", change_id="chg-9", created_at="2026-11-03T05:50:05Z"))])
    assert j["cases"][0]["check_passed"] is False


def test_allow_pause_and_trust_units_have_their_own_expectation(monkeypatch, tmp_path):
    j = _case_status(monkeypatch, tmp_path, "t3.allow-20261103T000000Z", [_dec("c", "ran___observe", "ALLOW", "2026-11-03T05:50:00Z")])
    assert j["cases"][0]["check_passed"] is True
    j = _case_status(monkeypatch, tmp_path / "p", "t3.pause-20261103T000000Z",
                     [_dec("c", "ran___predict", "DENY", "2026-11-03T05:50:00Z", deciding_rule="default_deny", reason="tool is paused")])
    assert j["cases"][0]["check_passed"] is True and j["cases"][0]["pauses"] == 1
    j = _case_status(monkeypatch, tmp_path / "t", "t3.trust.gate_v1_healthy-20261103T000000Z",
                     [("change", rec(case_id="c", change_id="chg-1", created_at="2026-11-03T05:50:00Z"))])
    assert j["cases"][0]["check_passed"] is True
    # an unknown unit keeps the full declared sequence
    j = _case_status(monkeypatch, tmp_path / "u", "other-20261103T000000Z", [_dec("c", "ran___observe", "ALLOW", "2026-11-03T05:50:00Z")])
    assert j["cases"][0]["check_passed"] is False


def test_a_declaration_counts_when_s3_stored_it(monkeypatch, tmp_path):
    from spectrum.slices.twinview import logic

    decl = {"run_id": RUN, "declared_at": "2026-11-03T05:00:00Z", "unit": "t3.allow"}
    assert logic.case_prov({**decl, awstime.STORED_KEY: "2026-11-03T05:00:01Z"})["real"] is True
    assert logic.case_prov(decl)["real"] is False
    assert logic.case_prov({"run_id": RUN, "kind": "change", awstime.STORED_KEY: "2026-11-03T05:00:01Z"})["real"] is False
