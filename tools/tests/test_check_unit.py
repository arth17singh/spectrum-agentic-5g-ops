"""check_unit: the four required cases (R, R-late, R-tampered, R-missing-result) and the edges around them.
Every case runs offline against FakeAws; nothing here can reach AWS."""

import ast
import io
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

import check_unit
import unitfix
from evfake import READ_ONLY_OPS, FakeAws

TOOLS = Path(__file__).resolve().parents[1]
COMMITTED = TOOLS / "tests" / "fixtures" / "units"
HEAD = subprocess.run(["git", "rev-parse", "HEAD"], cwd=TOOLS, capture_output=True, text=True, check=True).stdout.strip()
RID = unitfix.RUN_ID


def check(fx: Path, run_id: str = RID, **kw) -> tuple[int, check_unit.Checker, FakeAws]:
    aws, uri = FakeAws.from_fixture(fx)
    ev = check_unit.S3Evidence(aws.client("s3"), uri)
    ledger = fx / "ledger" if (fx / "ledger" / "ledger.jsonl").is_file() else None
    ck = check_unit.Checker(run_id, ev, aws, ledger, poll_s=0.0, now=aws.now(), **kw)
    code = ck.run()
    return code, ck, aws


def states(ck) -> dict:
    return {k: p.state for k, p in ck.parts.items()}


def text(ck) -> str:
    buf = io.StringIO()
    ck.report(buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------- the four required cases

def test_R_good_run_exits_0(tmp_path):
    code, ck, aws = check(unitfix.build(tmp_path, "R", HEAD))
    assert code == 0, text(ck)
    assert set(states(ck).values()) == {"PASS"}
    # the window came from AWS reads, not from our records
    srcs = [s[0] for s in ck.window_sources]
    assert any(s.startswith("sagemaker:CreationTime") for s in srcs)
    assert any("cloudwatch:/team13/checkpoint (us-east-2" in s for s in srcs)
    assert any("cloudwatch:aws/spans (us-west-2" in s for s in srcs)  # both Regions searched
    assert ck.window[0] == datetime(2026, 11, 2, 18, 5, tzinfo=timezone.utc)


def test_R_late_exits_1_on_part_a(tmp_path):
    code, ck, _ = check(unitfix.build(tmp_path, "R-late", HEAD))
    assert code == 1
    assert states(ck)["a"] == "FAIL" and states(ck)["e"] == "PASS"
    assert "NOT before the run's first AWS time 2026-11-02T18:05:00.000Z" in text(ck)


def test_R_tampered_exits_1_on_part_d(tmp_path):
    code, ck, _ = check(unitfix.build(tmp_path, "R-tampered", HEAD))
    assert code == 1
    assert states(ck)["d"] == "FAIL"
    assert states(ck)["e"] == "PASS"  # the forger made the result agree: only the ledger catches it
    out = text(ck)
    assert "item.jsonl: edited after ledger #" in out and "results/" in out


def test_R_missing_result_exits_1_on_part_e(tmp_path):
    code, ck, _ = check(unitfix.build(tmp_path, "R-missing-result", HEAD))
    assert code == 1 and states(ck)["e"] == "FAIL"
    assert "no result at results/" in text(ck)


@pytest.mark.parametrize("case,want", [("R", 0), ("R-late", 1), ("R-tampered", 1), ("R-missing-result", 1),
                                       ("R-objects", 0)])
def test_committed_fixtures_through_the_cli(case, want):
    r = subprocess.run([sys.executable, str(TOOLS / "check_unit.py"), RID, "--offline", str(COMMITTED / case)],
                       capture_output=True, text=True)
    assert r.returncode == want, r.stdout + r.stderr
    assert f"(exit {want})" in r.stdout


def test_committed_fixtures_match_the_builder(tmp_path):
    """The committed fixtures are exactly what unitfix.py produces (regenerate with python tools/tests/unitfix.py)."""
    unitfix.build_all(tmp_path)
    unitfix.build(tmp_path, "R-objects", objects=True)
    for p in COMMITTED.rglob("*"):
        if p.is_file():
            assert p.read_bytes() == (tmp_path / p.relative_to(COMMITTED)).read_bytes(), p


# ---------------------------------------------------------------------------- read-only and independence

def test_check_is_read_only(tmp_path):
    _, _, aws = check(unitfix.build(tmp_path, "R", HEAD))
    assert aws.ops() <= READ_ONLY_OPS and "PutObject" not in aws.ops()
    assert {"GetObject", "ListObjectsV2", "StartQuery", "GetQueryResults", "DescribeProcessingJob"} <= aws.ops()


@pytest.mark.parametrize("script", ["check_unit.py", "show_record.py"])
def test_check_scripts_import_nothing_from_spectrum_or_the_writer(script):
    tree = ast.parse((TOOLS / script).read_text())
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            mods.add((n.module or "").split(".")[0])
    assert not mods & {"spectrum", "evlib", "check_unit", "run_unit"}, mods
    # numpy: imported lazily by check_unit's paired-bootstrap replay only (the producers' intervals are numpy
    # PCG64 streams); a third-party library, not code shared with the writer or the dashboard
    assert mods - {"evfake"} <= set(sys.stdlib_module_names) | {"boto3", "botocore", "cryptography", "numpy", ""}, mods


def test_window_never_comes_from_our_own_records(tmp_path):
    """Make our job record lie about its start: the window must still be SageMaker's CreationTime, and the
    lie is reported."""
    code, ck, _ = check(unitfix.build(tmp_path, "R", HEAD, job_aws_start="2026-11-02T17:00:00.000Z"))
    assert ck.window[0] == datetime(2026, 11, 2, 18, 5, tzinfo=timezone.utc)
    assert code == 1 and "SageMaker CreationTime is 2026-11-02T18:05:00.000Z" in text(ck)


# ---------------------------------------------------------------------------- part (c)

def test_record_time_outside_the_aws_window_fails(tmp_path):
    fx = unitfix.build(tmp_path, "R", HEAD, item_overrides={2: {"aws_time": "2026-11-02T19:30:00.000Z"}})
    code, ck, _ = check(fx)
    assert code == 1 and states(ck)["c"] == "FAIL" and "outside the AWS window" in text(ck)


NULL_TIME = {"aws_time": None, "aws_time_source": None}


def test_null_aws_time_with_no_aws_source_fails(tmp_path):
    """An item in a batch .jsonl (no own S3 object) naming no case or log pointer: nothing in AWS dates it."""
    code, ck, _ = check(unitfix.build(tmp_path, "R", HEAD, item_overrides={0: dict(NULL_TIME)}))
    assert code == 1 and "aws_time is null and no AWS source gives it" in text(ck)


def test_null_aws_time_derived_from_the_records_own_s3_object(tmp_path):
    fx = unitfix.build(tmp_path, "R-objects", HEAD, objects=True, change_overrides=NULL_TIME,
                       item_overrides={i: dict(NULL_TIME) for i in range(5)})
    code, ck, _ = check(fx)
    assert code == 0, text(ck)
    assert len(ck.derived) == 6 and all(x["source"].startswith("s3:LastModified records/") for x in ck.derived)
    chg = next(x for x in ck.derived if "/change/" in x["source"])
    assert chg["aws_time"] == "2026-11-02T18:12:31.000Z"  # AWS's time for the object, not anything we wrote
    assert "AWS time derived from: s3:LastModified 6" in text(ck)


def test_null_aws_time_derived_from_the_change_applied_log_line(tmp_path):
    fx = unitfix.build(tmp_path, "R", HEAD, change_jsonl=True, change_overrides=NULL_TIME)
    code, ck, _ = check(fx)
    assert code == 0, text(ck)
    (x,) = ck.derived
    assert x["aws_time"] == "2026-11-02T18:12:30.000Z"
    assert x["source"].startswith("cloudwatch:/aws/lambda/team13-ran-tools (us-east-2")


def test_null_aws_time_derived_from_a_log_pointer(tmp_path):
    ptr = {"log_group": "/team13/checkpoint", "log_stream": "gateway/2026-11-02", "event_id": "x"}
    fx = unitfix.build(tmp_path, "R", HEAD,
                       item_overrides={2: {**NULL_TIME, "log_pointer": ptr, "case_id": "case-0003"}})
    code, ck, _ = check(fx)
    assert code == 0, text(ck)
    (x,) = ck.derived
    assert x["aws_time"] == "2026-11-02T18:18:00.000Z" and x["source"].startswith("cloudwatch:/team13/checkpoint")


def test_change_applied_line_must_name_the_same_change_id(tmp_path):
    fx = unitfix.build(tmp_path, "R", HEAD, change_jsonl=True,
                       change_overrides={**NULL_TIME, "change_id": "chg-ffffffffffff"})
    code, ck, _ = check(fx)
    assert code == 1 and "no AWS source gives it" in text(ck)


def test_derived_time_outside_the_window_fails(tmp_path):
    fx = unitfix.build(tmp_path, "R-objects", HEAD, objects=True, change_overrides=NULL_TIME,
                       change_lm="2026-11-02T19:45:00Z")
    code, ck, _ = check(fx)
    assert code == 1 and "derived AWS time 2026-11-02T19:45:00.000Z" in text(ck)


def test_present_aws_time_is_still_verified_against_the_window(tmp_path):
    fx = unitfix.build(tmp_path, "R-objects", HEAD, objects=True,
                       change_overrides={"aws_time": "2026-11-02T17:00:00.000Z"})
    code, ck, _ = check(fx)
    assert code == 1 and "is outside the AWS window" in text(ck) and not ck.derived


def test_derived_times_are_listed_in_json(tmp_path, capsys):
    fx = unitfix.build(tmp_path, "R", HEAD, change_jsonl=True, change_overrides=NULL_TIME)
    assert check_unit.main([RID, "--offline", str(fx), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["derived_aws_times"][0]["source"].startswith("cloudwatch:/aws/lambda/team13-ran-tools")


def test_record_from_another_run_fails(tmp_path):
    code, ck, _ = check(unitfix.build(tmp_path, "R", HEAD, item_overrides={1: {"run_id": "someone-else"}}))
    assert code == 1 and "run_id 'someone-else'" in text(ck)


def test_job_that_sagemaker_never_ran_fails(tmp_path):
    code, ck, _ = check(unitfix.build(tmp_path, "R", HEAD, job_missing_in_aws=True))
    assert code == 1 and "SageMaker in us-east-2 has no such processing job" in text(ck)


def test_lambda_only_run_timed_by_cloudwatch_alone(tmp_path):
    """The Track 3 path: no SageMaker job at all; the window comes from CloudWatch events in both Regions."""
    code, ck, aws = check(unitfix.build(tmp_path, "R", HEAD, no_job=True))
    assert code == 0, text(ck)
    assert all(s[0].startswith("cloudwatch:") for s in ck.window_sources)
    assert ck.window[0] == datetime(2026, 11, 2, 18, 6, tzinfo=timezone.utc)  # the first span (us-west-2)
    assert not any(op.startswith("Describe") for op in aws.ops())


def test_lambda_job_record_is_not_looked_up_in_sagemaker(tmp_path):
    code, ck, aws = check(unitfix.build(tmp_path, "R", HEAD, job_kind="lambda", job_missing_in_aws=True,
                                             job_aws_start="2026-11-02T18:06:00.000Z"))
    assert code == 0, text(ck)
    assert "DescribeProcessingJob" not in aws.ops()


def test_declaration_in_the_same_second_as_the_run_start_fails(tmp_path):
    """S3 LastModified is floored to the second, so 18:05:00 may really be 18:05:00.999: not proven before."""
    code, ck, _ = check(unitfix.build(tmp_path, "R", HEAD, decl_lm="2026-11-02T18:04:59Z",
                                      job_aws_start="2026-11-02T18:05:00.000Z"))
    assert code == 0, text(ck)
    fx = unitfix.build(tmp_path / "b", "R", HEAD)
    spec = json.loads((fx / "aws.json").read_text())
    spec["sagemaker"][0]["CreationTime"] = "2026-11-02T18:00:00.500Z"
    (fx / "aws.json").write_text(json.dumps(spec))
    code, ck, _ = check(fx)
    assert states(ck)["a"] == "FAIL"


def test_skipped_ledger_is_said_on_the_result_line(tmp_path):
    _, ck, _ = check(unitfix.build(tmp_path, "R", HEAD, no_ledger=True))
    assert "(exit 0) (d skipped: no ledger" in text(ck)


def test_no_aws_time_at_all_cannot_check(tmp_path):
    """No job, no log event: S3 LastModified still proves the declaration came first (a), but there is no
    job window for record times (c)."""
    code, ck, _ = check(unitfix.build(tmp_path, "R", HEAD, no_job=True, no_logs=True))
    assert code == 2, text(ck)
    assert states(ck)["a"] == "PASS" and states(ck)["c"] == "CANNOT CHECK"


def test_per_record_objects_are_merged_and_lambda_log_group_times_the_change(tmp_path):
    code, ck, _ = check(unitfix.build(tmp_path, "R-objects", HEAD, objects=True))
    assert code == 0, text(ck)
    assert {k: len(v) for k, v in ck.records.items()} == {"change": 1, "item": 5, "job": 1}
    assert [r["item_id"] for r in ck.records["item"]] == [f"case-000{i}" for i in range(1, 6)]
    assert any("/aws/lambda/team13-ran-tools" in s[0] for s in ck.window_sources)


def test_an_object_holding_two_records_fails(tmp_path):
    fx = unitfix.build(tmp_path, "R-objects", HEAD, objects=True, no_ledger=True)
    p = next((fx / "evidence" / "records" / RID / "item").iterdir())
    p.write_bytes(p.read_bytes() * 2)
    code, ck, _ = check(fx)
    assert code == 1 and "a record object holds exactly one" in text(ck)


# ---------------------------------------------------------------------------- part (b)

def test_code_sha_not_in_git_fails(tmp_path):
    code, ck, _ = check(unitfix.build(tmp_path, "R", "f" * 40))
    assert code == 1 and states(ck)["b"] == "FAIL"


# ---------------------------------------------------------------------------- part (d)

def test_mid_run_append_seen_by_the_ledger_passes(tmp_path):
    """A JSONL file that grew by whole lines before the result was recorded is an append, not an edit."""
    fx = unitfix.build(tmp_path, "R", HEAD, append_after_ingest=True)
    code, ck, _ = check(fx)
    assert states(ck)["d"] == "PASS", text(ck)
    assert states(ck)["e"] == "FAIL"  # 6 items, declared 5: the extra item is still caught by (e)


def test_append_after_the_result_fails(tmp_path):
    code, ck, _ = check(unitfix.build(tmp_path, "R", HEAD, append_after_result=True))
    assert code == 1 and "after the result was recorded" in text(ck)


def test_lines_not_yet_ledgered_cannot_be_checked(tmp_path):
    code, ck, _ = check(unitfix.build(tmp_path, "R", HEAD, pending_tail=True))
    assert states(ck)["d"] == "CANNOT CHECK" and "not ledgered yet" in text(ck)


def test_no_ledger_is_skipped_not_failed(tmp_path):
    code, ck, _ = check(unitfix.build(tmp_path, "R", HEAD, no_ledger=True))
    assert code == 0 and states(ck)["d"] == "SKIP"


def test_a_forged_ledger_fails(tmp_path):
    fx = unitfix.build(tmp_path, "R", HEAD)
    lines = (fx / "ledger" / "ledger.jsonl").read_text().splitlines()
    rec = json.loads(lines[1])
    rec["payload"]["sha256"] = "0" * 64
    lines[1] = json.dumps(rec, sort_keys=True)
    (fx / "ledger" / "ledger.jsonl").write_text("\n".join(lines) + "\n")
    code, ck, _ = check(fx)
    assert code == 1 and "the ledger itself does not verify" in text(ck)


def test_a_record_object_added_after_the_result_fails(tmp_path):
    fx = unitfix.build(tmp_path, "R-objects", HEAD, objects=True)
    led = unitfix.MiniLedger()
    led.records = [json.loads(x) for x in (fx / "ledger" / "ledger.jsonl").read_text().splitlines()]
    extra = {**unitfix.items(RID)[0], "item_id": "case-0099"}
    line = unitfix.jsonl([extra])
    rel = f"records/{RID}/item/forged.json"
    (fx / "evidence" / rel).write_bytes(line)
    spec = json.loads((fx / "aws.json").read_text())
    spec["s3"]["last_modified"][rel] = "2026-11-02T18:40:00Z"
    (fx / "aws.json").write_text(json.dumps(spec))
    led.add("evidence.artifact", rel, line, RID, "record", "2026-11-02T18:40:00.000Z")
    led.write(fx / "ledger")
    code, ck, _ = check(fx)
    assert code == 1 and "after the result was recorded" in text(ck)


def _reledger_result_first(fx):
    """Rewrite the fixture's ledger so the result is ingested BEFORE the record objects (a sync order only)."""
    old = [json.loads(x) for x in (fx / "ledger" / "ledger.jsonl").read_text().splitlines()]
    led = unitfix.MiniLedger()
    order = sorted(old, key=lambda r: (not r["payload"]["path"].startswith("results/"), r["seq"]))
    for r in order:
        pl = r["payload"]
        led.add(r["type"], pl["path"], (fx / "evidence" / pl["path"]).read_bytes(), pl["run_id"], pl["kind"], r["ts"])
    led.write(fx / "ledger")


def test_ledger_order_inverted_but_s3_order_fine_passes_with_a_note(tmp_path):
    """The ledger syncs objects in its own order; S3 LastModified says the records were stored before the result."""
    fx = unitfix.build(tmp_path, "R-objects", HEAD, objects=True)
    _reledger_result_first(fx)
    code, ck, _ = check(fx)
    out = text(ck)
    assert states(ck)["d"] == "PASS", out
    assert "ledger order differs from S3 order" in out and "record before result" in out
    assert "after the result was recorded" not in out


def test_record_stored_after_the_result_still_fails_when_the_ledger_agrees_with_s3(tmp_path):
    fx = unitfix.build(tmp_path, "R-objects", HEAD, objects=True)
    spec = json.loads((fx / "aws.json").read_text())
    rel = next(k for k in spec["s3"]["last_modified"] if "/item/" in k)
    spec["s3"]["last_modified"][rel] = "2026-11-02T18:40:00Z"  # result is 18:32:00
    (fx / "aws.json").write_text(json.dumps(spec))
    _reledger_result_first(fx)
    code, ck, _ = check(fx)
    assert states(ck)["d"] == "FAIL" and "S3 LastModified agrees" in text(ck)


def test_local_evidence_keeps_the_ledger_order_rule_and_says_why(tmp_path):
    fx = unitfix.build(tmp_path, "R-objects", HEAD, objects=True)
    _reledger_result_first(fx)
    ev = check_unit.LocalEvidence(fx / "evidence")
    ck = check_unit.Checker(RID, ev, None, fx / "ledger", poll_s=0.0)
    ck.load()
    ck.decl_time = (None, "")
    ck.part_d()
    assert ck.parts["d"].state == "FAIL" and "local evidence has no S3 times" in "\n".join(ck.parts["d"].lines)


# ---------------------------------------------------------------------------- part (e)

def test_unknown_metric_cannot_be_checked(tmp_path):
    code, ck, _ = check(unitfix.build(tmp_path, "R", HEAD, metric="vibes", op=">=", value=1, result_value=1))
    assert code == 2 and states(ck)["e"] == "CANNOT CHECK"


def test_result_that_disagrees_with_the_items_fails(tmp_path):
    fx = unitfix.build(tmp_path, "R", HEAD, no_ledger=True)
    p = fx / "evidence" / "results" / f"{RID}.json"
    r = json.loads(p.read_text())
    r["value"] = 1.0
    p.write_text(json.dumps(r))
    code, ck, _ = check(fx)
    assert code == 1 and "result value 1.0, recomputed 0.8" in text(ck)


def test_target_missed_is_still_a_valid_unit(tmp_path):
    """Four parts present, target missed: the unit counts (exit 0) and the report says MISSED."""
    fx = unitfix.build(tmp_path, "R", HEAD, value=0.9)
    code, ck, _ = check(fx)
    assert code == 0 and "-> target MISSED" in text(ck)


@pytest.mark.parametrize("metric,items,want", [
    ("pass_rate", [True, True, False, True], 0.75),
    ("fail_count", [True, False, False], 2),
    ("p95_latency_ms", [100, 200, 300, 400, 1000], 1000),
    ("p50_latency_ms", [100, 200, 300, 400, 1000], 300),
    ("mean_score", [1.0, 0.5, 0.0, 0.5], 0.5),
    ("cost_per_1k_usd", [0.001, 0.003], 2.0),
])
def test_recompute_golden_values(metric, items, want):
    field = {"pass_rate": "passed", "fail_count": "passed", "p95_latency_ms": "latency_ms",
             "p50_latency_ms": "latency_ms", "mean_score": "score", "cost_per_1k_usd": "cost_usd"}[metric]
    its = [{"item_id": str(i), field: v} for i, v in enumerate(items)]
    assert check_unit.recompute(metric, its) == pytest.approx(want)


# ---------------------------------------------------------------------------- local mode

def test_local_mode_with_only_mtimes_cannot_check_part_a(tmp_path):
    fx = unitfix.build(tmp_path, "R", HEAD)
    aws, _ = FakeAws.from_fixture(fx)
    t = datetime(2026, 11, 2, 18, 0, tzinfo=timezone.utc).timestamp()
    for p in (fx / "evidence").rglob("*"):
        os.utime(p, (t, t))
    ev = check_unit.LocalEvidence(fx / "evidence")  # not in git: only an mtime
    ck = check_unit.Checker(RID, ev, aws, fx / "ledger", poll_s=0.0, now=aws.now())
    assert ck.run() == 2 and states(ck)["a"] == "CANNOT CHECK"
    ev.allow_mtime = True
    ck = check_unit.Checker(RID, ev, aws, fx / "ledger", poll_s=0.0, now=aws.now())
    assert ck.run() == 0, text(ck)


def test_bad_run_id_and_no_location_exit_2(tmp_path, capsys):
    assert check_unit.main(["bad id"]) == 2
    assert check_unit.main([RID]) == 2  # no --s3/--dir/--offline and no env


def test_json_output(tmp_path, capsys):
    fx = unitfix.build(tmp_path, "R", HEAD)
    assert check_unit.main([RID, "--offline", str(fx), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["exit"] == 0 and out["first_aws_time"] == "2026-11-02T18:05:00.000Z"
    assert out["parts"]["a"]["state"] == "PASS"
