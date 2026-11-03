"""evidence_sync.attest_lambda: a Lambda-stamped change record is attested by the Lambda's own CloudWatch line, with the
Gateway request window from aws/spans; read-only on a fake logs client; additive; never downgrades a match."""

import hashlib
import json
import types
from datetime import datetime, timezone
from pathlib import Path

import evidence_sync as es

RUN = "t3.allow-20261103T052448Z"
CID = "chg-e30320ae9bbc"


def ms(iso: str) -> int:
    return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp() * 1000)


class FakeLogs:
    def __init__(self, groups):
        self.groups, self.calls = groups, []

    def get_paginator(self, name):
        assert name == "filter_log_events"
        return self

    def paginate(self, logGroupName, filterPattern, startTime, endTime, **_):
        self.calls.append((logGroupName, filterPattern, startTime, endTime))
        term = filterPattern.strip('"')
        yield {"events": [e for e in self.groups.get(logGroupName, [])
                          if startTime <= e["timestamp"] <= endTime and term in e["message"]]}


def make(tmp_path, groups, rec_extra=None, run_filter=None):
    dest = tmp_path / "ev"
    rel = f"records/{RUN}/change/{CID}.json"
    rec = {"kind": "change", "run_id": RUN, "case_id": "case-1", "change_id": CID, "created_at": "2026-11-03T05:31:52.545Z",
           "aws_time": "2026-11-03T05:31:52.546Z", "aws_time_source": "lambda:clock", **(rec_extra or {})}
    (dest / rel).parent.mkdir(parents=True)
    (dest / rel).write_text(json.dumps(rec))
    dec = {"kind": "decision", "run_id": RUN, "decision": "ALLOW", "tool": "ran___apply_change", "approval_id": CID,
           "span_id": "6097e6ac2a647dc0", "trace_id": "T1", "aws_time": "2026-11-03T05:31:52.967Z"}
    (dest / f"records/{RUN}/decision").mkdir(parents=True)
    (dest / f"records/{RUN}/decision/d1.json").write_text(json.dumps(dec))
    sha = hashlib.sha256((dest / rel).read_bytes()).hexdigest()
    s = object.__new__(es.Sync)
    s.a = types.SimpleNamespace(run=run_filter)
    s.dest, s.state_dir = dest, dest / ".sync"
    s.problems = []
    s.sess = types.SimpleNamespace(client=lambda name: FakeLogs(groups))
    return s, rel, sha, {rel: {"sha256": sha}}


def lambda_line(**kw):
    body = {"ts": 1.0, "record": "change_applied", "run_id": RUN, "case_id": "case-1", "approval_id": CID}
    body.update(kw)
    return {"timestamp": ms("2026-11-03T05:31:52.884Z"), "message": json.dumps(body)}


SPAN = {"timestamp": ms("2026-11-03T05:31:52.970Z"), "message": json.dumps({
    "traceId": "T1", "spanId": "6097e6ac2a647dc0", "name": "AgentCore.Gateway.InvokeTool",
    "startTimeUnixNano": ms("2026-11-03T05:31:51.874Z") * 1_000_000, "endTimeUnixNano": ms("2026-11-03T05:31:52.970Z") * 1_000_000,
    "durationNano": 1096000000})}


def load(s):
    return json.loads((s.state_dir / "attest" / f"{RUN}.json").read_text())


def test_attests_the_change_and_records_the_gateway_window(tmp_path):
    s, rel, sha, man = make(tmp_path, {es.LAMBDA_LOG_GROUP: [lambda_line()], es.SPANS_LOG_GROUP: [SPAN]})
    assert s.attest_lambda(man) == 1
    e = load(s)["files"][rel]
    assert e["state"] == "match" and e["sha256"] == sha and e["via"] == "lambda_log"
    assert e["log_time"] == "2026-11-03T05:31:52.884Z"
    assert e["gateway_window"]["start"] == "2026-11-03T05:31:51.874Z" and e["gateway_window"]["end"] == "2026-11-03T05:31:52.970Z"
    # the dashboard's reader takes it, as a Lambda log time with the window
    from importlib import import_module
    import sys
    sys.path.insert(0, str(Path(es.__file__).resolve().parents[1] / "dashboard" / "backend"))
    rd = import_module("spectrum.slices._shared.records").RecordsReader(str(s.dest))
    rec, meta = rd.all_records_times("change")[0]
    assert meta["attested"] == "2026-11-03T05:31:52.884Z" and meta["attested_via"] == "lambda_log" and meta["window"]["end"].endswith("52.970Z")


def test_reads_only_a_window_of_ten_minutes_around_the_record_and_the_change_id(tmp_path):
    groups = {es.LAMBDA_LOG_GROUP: [lambda_line()], es.SPANS_LOG_GROUP: [SPAN]}
    s, rel, sha, man = make(tmp_path, groups)
    logs = FakeLogs(groups)
    s.sess = types.SimpleNamespace(client=lambda name: logs)
    s.attest_lambda(man)
    g, pat, lo, hi = logs.calls[0]
    assert g == es.LAMBDA_LOG_GROUP and pat == f'"{CID}"'
    t0 = ms("2026-11-03T05:31:52.545Z")
    assert (lo, hi) == (t0 - 600_000, t0 + 600_000)


def test_a_line_for_another_run_or_case_does_not_attest(tmp_path):
    for bad in ({"run_id": "other"}, {"case_id": "case-2"}, {"record": "something_else"}):
        s, rel, sha, man = make(tmp_path / bad.get("run_id", bad.get("case_id", bad.get("record"))), {es.LAMBDA_LOG_GROUP: [lambda_line(**bad)]})
        assert s.attest_lambda(man) == 0
        assert not (s.state_dir / "attest" / f"{RUN}.json").exists()


def test_a_record_not_stamped_by_the_lambda_is_not_looked_up(tmp_path):
    s, rel, sha, man = make(tmp_path, {es.LAMBDA_LOG_GROUP: [lambda_line()]}, rec_extra={"aws_time_source": "cloudwatch:/x"})
    logs = FakeLogs({})
    s.sess = types.SimpleNamespace(client=lambda name: logs)
    s.attest_lambda(man)
    assert logs.calls == []


def test_never_replaces_a_match_from_a_job_log_and_never_downgrades(tmp_path):
    s, rel, sha, man = make(tmp_path, {es.LAMBDA_LOG_GROUP: [lambda_line()]})
    out = s.state_dir / "attest"
    out.mkdir(parents=True)
    (out / f"{RUN}.json").write_text(json.dumps({"run_id": RUN, "source": "job", "fingerprint": "f", "files": {
        rel: {"state": "match", "sha256": sha, "log_time": "2026-11-03T05:00:00Z", "job": "j"}}}))
    s.attest_lambda(man)
    d = load(s)
    assert d["files"][rel]["job"] == "j" and "via" not in d["files"][rel] and d["fingerprint"] == "f"
    # a later lookup that finds nothing leaves an earlier lambda match alone
    (out / f"{RUN}.json").write_text(json.dumps({"run_id": RUN, "files": {rel: {"state": "match", "sha256": sha, "via": "lambda_log", "log_time": "2026-11-03T05:31:52.884Z"}}}))
    s.sess = types.SimpleNamespace(client=lambda name: FakeLogs({}))
    s.attest_lambda(man)
    assert load(s)["files"][rel]["state"] == "match"


def test_run_filter_limits_the_step(tmp_path):
    s, rel, sha, man = make(tmp_path, {es.LAMBDA_LOG_GROUP: [lambda_line()]}, run_filter=["some-other-run"])
    assert s.attest_lambda(man) == 0 and not (s.state_dir / "attest" / f"{RUN}.json").exists()
