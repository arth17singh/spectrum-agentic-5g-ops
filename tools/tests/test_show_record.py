"""show_record (the judges' live test), offline: what it prints, that it only reads, and when it says no."""

import io
import json
import subprocess
import sys
from pathlib import Path

import show_record
import unitfix
from evfake import FakeAws

TOOLS = Path(__file__).resolve().parents[1]
RID = unitfix.RUN_ID
ALLOWED = {"GetObject", "ListObjectsV2", "StartQuery", "GetQueryResults", "GetItem"}


def run(fx: Path, item: str, **kw) -> tuple[int, str, FakeAws]:
    aws, uri = FakeAws.from_fixture(fx)
    out = io.StringIO()
    code = show_record.show(RID, item, show_record.S3Reader(aws.client("s3"), uri), aws, poll_s=0.0, out=out, **kw)
    return code, out.getvalue(), aws


def test_shows_declaration_record_time_and_matching_event(tmp_path):
    code, out, aws = run(unitfix.build(tmp_path, "R"), "case-0003", cases_table="team13-cases")
    assert code == 0, out
    assert "declared/t3.deny-fixture-R.json" in out and "S3 LastModified: 2026-11-02T18:00:00.000Z" in out
    assert '"item_id": "case-0003"' in out and "item.jsonl line 3" in out
    assert "aws_time: '2026-11-02T18:18:00.000Z'   source: 'cloudwatch:/team13/checkpoint'" in out
    assert "2026-11-02 18:18:00.000 UTC  /team13/checkpoint (us-east-2)" in out and "MATCHES" in out
    assert '"run_id": "t3.deny-fixture-R"' in out.split("== 5.")[1]  # DynamoDB case -> run
    assert "pass_rate = 0.8" in out


def test_only_read_only_apis_in_both_regions(tmp_path):
    _, _, aws = run(unitfix.build(tmp_path, "R"), "case-0001", cases_table="team13-cases")
    assert aws.ops() <= ALLOWED, aws.ops()
    assert {r for s, _, r in aws.calls if s == "logs"} == {"us-east-2", "us-west-2"}
    # Logs Insights is asynchronous: the poll loop saw Running before Complete
    started = len(aws.query_log)  # queries on log groups that exist (the others are ResourceNotFound)
    assert started >= 2 and sum(1 for c in aws.calls if c[1] == "GetQueryResults") == 2 * started


def test_per_record_object_layout(tmp_path):
    fx = unitfix.build(tmp_path, "R-objects", objects=True)
    code, out, _ = run(fx, "case-0002", kind="change", id_field="case_id")
    assert code == 0, out
    assert "records/t3.deny-fixture-R/change/" in out and "/aws/lambda/team13-ran-tools (us-east-2)" in out
    code, out, _ = run(fx, "case-0004")
    assert code == 0 and "records/t3.deny-fixture-R/item/" in out


def test_missing_item_exits_1(tmp_path):
    code, out, _ = run(unitfix.build(tmp_path, "R"), "case-9999")
    assert code == 1 and "NOT FOUND" in out


def test_no_matching_log_event_exits_1(tmp_path):
    code, out, _ = run(unitfix.build(tmp_path, "R", no_logs=True), "case-0003")
    assert code == 1 and "no log event mentions both" in out


def test_aws_time_that_matches_no_event_exits_1(tmp_path):
    fx = unitfix.build(tmp_path, "R", item_overrides={2: {"aws_time": "2026-11-02T18:19:00.000Z"}})
    code, out, _ = run(fx, "case-0003")
    assert code == 1 and "differs by -60.0s" in out and "no matching event has the record's aws_time" in out


def test_cli_offline_and_bad_args():
    fx = TOOLS / "tests" / "fixtures" / "units" / "R"
    r = subprocess.run([sys.executable, str(TOOLS / "show_record.py"), "--run-id", RID, "--item", "case-0005",
                        "--offline", str(fx)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert '"passed": false' in r.stdout
    assert show_record.main(["--run-id", "bad id", "--item", "x", "--offline", str(fx)]) == 2
    assert show_record.main(["--run-id", RID, "--item", "x"]) == 2  # no location


def test_unsafe_item_id_is_not_put_into_the_query(tmp_path):
    fx = unitfix.build(tmp_path, "R")
    aws, uri = FakeAws.from_fixture(fx)
    show_record.show(RID, 'x" | delete', show_record.S3Reader(aws.client("s3"), uri), aws, poll_s=0.0,
                     out=io.StringIO())
    assert all('delete' not in q["q"] for q in aws.query_log)
    json.dumps(aws.query_log)  # nothing odd recorded
