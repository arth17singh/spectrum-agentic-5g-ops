"""Builds offline unit fixtures (evidence tree + aws.json + ledger) for check_unit / show_record.

Cases:
  R                 good: declared 18:00, SageMaker job created 18:05, items 18:10-18:28, result 18:32
  R-late            the declaration is stored at 18:06, after the job's CreationTime (18:05)
  R-tampered        an item flipped from fail to pass after ingest, and the result rewritten to match
  R-missing-result  no results/<run_id>.json
Extra variants (for tests): see build(**opts).

Times are hand-set and the expected values are hand-computed: 5 items, 4 passed -> pass_rate 0.8 >= 0.8.
The ledger here is written by this file's own few lines (hash chain, no signatures), not by the dashboard;
tests/test_evidence_tools.py in the dashboard covers a ledger written by the real EvidenceService.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

BASE_SHA = "d65cb863db084d9cd20c80e48d47e49d2e9fd1b4"  # the repo's base commit: always present in history
RUN_ID = "t3.deny-fixture-R"
BUCKET, PREFIX = "fixture-bucket", "evidence/"
JOB = "team13-suite-fixture"
TEST_SET_SHA = hashlib.sha256(b"fixture test set v1\n").hexdigest()

ITEMS = [  # item_id, passed, latency_ms, aws_time
    ("case-0001", True, 410, "2026-11-02T18:10:00.000Z"),
    ("case-0002", True, 380, "2026-11-02T18:14:00.000Z"),
    ("case-0003", True, 520, "2026-11-02T18:18:00.000Z"),
    ("case-0004", True, 450, "2026-11-02T18:23:00.000Z"),
    ("case-0005", False, 610, "2026-11-02T18:28:00.000Z"),
]


def _env(kind: str, run_id: str, created: str, aws_time, src) -> dict:
    return {"schema_version": "1", "kind": kind, "run_id": run_id, "team": "team13", "created_at": created,
            "aws_time": aws_time, "aws_time_source": src, "synthetic": True}


def declaration(run_id: str, code_sha: str, metric="pass_rate", op=">=", value=0.8) -> dict:
    return {**_env("declared", run_id, "2026-11-02T17:59:59.500Z", None, None),
            "unit": "t3.deny", "track": "track-3", "target": {"metric": metric, "op": op, "value": value, "unit": ""},
            "n_items": len(ITEMS), "test_set_sha256": TEST_SET_SHA, "model_id": "openai.gpt-oss-20b-1:0",
            "grader": "cedar-decision-from-aws-spans", "code_sha": code_sha, "region": "us-east-2",
            "twin": {"kind": "none", "version": "", "scenario": "", "seed": 0},
            "declared_at": "2026-11-02T17:59:59.500Z"}


def items(run_id: str) -> list[dict]:
    out = []
    for iid, passed, lat, t in ITEMS:
        out.append({**_env("item", run_id, t.replace(".000Z", ".900Z"), t, "cloudwatch:/team13/checkpoint"),
                    "item_id": iid, "input_sha256": hashlib.sha256(f"in-{iid}".encode()).hexdigest(),
                    "output_sha256": hashlib.sha256(f"out-{iid}".encode()).hexdigest(), "score": 1.0 if passed else 0.0,
                    "passed": passed, "grader": "cedar-decision-from-aws-spans", "backend": "agentcore-gateway",
                    "latency_ms": lat, "cost_usd": 0.0004, "extra": {"expected": "DENY", "case_id": iid}})
    return out


def job(run_id: str, aws_start="2026-11-02T18:05:00.000Z") -> dict:
    return {**_env("job", run_id, "2026-11-02T18:31:00.000Z", aws_start, "sagemaker:CreationTime"),
            "job_name": JOB, "arn": f"arn:aws:sagemaker:us-east-2:000000000000:processing-job/{JOB}",
            "job_kind": "processing", "region": "us-east-2", "status": "Completed", "aws_start": aws_start,
            "aws_end": "2026-11-02T18:30:00.000Z", "instance_type": "ml.t3.medium", "billable_s": 1500,
            "est_cost_usd": 0.02}


def change(run_id: str) -> dict:
    """A change record (amendment A15 fields), written by the apply tool Lambda."""
    kp = {"coverage_ratio": 0.91, "sla_ratio": 0.87, "sinr_p10_db": 1.5, "sinr_mean_db": 9.0}
    meas = {"coverage_ratio": 0.92, "sla_ratio": 0.88, "sinr_p10_db": 1.75, "sinr_mean_db": 9.25}
    return {**_env("change", run_id, "2026-11-02T18:12:30.500Z", "2026-11-02T18:12:30.000Z",
                   "cloudwatch:/aws/lambda/team13-ran-tools"),
            "case_id": "case-0002", "change_id": "chg-0123456789ab", "item_id": "step-0",
            "change": [{"cell": 1, "d_power_db": 1.0, "d_tilt_deg": 0.0}],
            "change_text": "cell=1;d_power_db=+1.0;d_tilt_deg=+0.0",
            "twin": {"kind": "maveric-inprocess", "version": "1.2.9", "scenario": "official", "seed": 7, "step": 0},
            "baseline_kpis": kp, "measured_kpis": meas,
            "measured_delta": {k: meas[k] - kp[k] for k in kp},
            "scored": False, "unscored_reason": "probe", "probe": True}


def result(run_id: str, code_sha: str, value=0.8, passed=True, n=5, metric="pass_rate", op=">=", target=0.8) -> dict:
    return {**_env("result", run_id, "2026-11-02T18:31:30.000Z", None, None),
            "unit": "t3.deny", "metric": metric, "value": value,
            "target": {"metric": metric, "op": op, "value": target, "unit": ""}, "passed": passed, "n_items": n,
            "checked_at": "2026-11-02T18:31:30.000Z", "checker": {"script": "tools/run_unit.py", "code_sha": code_sha},
            "details": {}}


def jsonl(recs: list[dict]) -> bytes:
    return b"".join((json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n").encode() for r in recs)


def doc(d: dict) -> bytes:
    return (json.dumps(d, sort_keys=True, indent=2) + "\n").encode()


# --------------------------------------------------------------------------- a tiny ledger writer

def _canonical(o) -> str:
    return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class MiniLedger:
    def __init__(self):
        self.records: list[dict] = []

    def add(self, type_: str, path: str, content: bytes, run_id: str, kind: str, ts: str) -> None:
        prev = self.records[-1]["hash"] if self.records else "0" * 64
        rec = {"seq": len(self.records) + 1, "ts": ts, "type": type_, "actor": "evidence-ingester", "plane": "prove",
               "payload": {"run_id": run_id, "kind": kind, "path": path, "sha256": hashlib.sha256(content).hexdigest(),
                           "size": len(content), "source": "s3", "source_time": ts},
               "key_id": "evidence/ingester"}
        rec["prev_hash"] = prev
        rec["hash"] = hashlib.sha256((_canonical(rec_body(rec)) + prev).encode()).hexdigest()
        rec["alg"], rec["sig"] = "Ed25519", ""
        self.records.append(rec)

    def write(self, d: Path) -> None:
        d.mkdir(parents=True, exist_ok=True)
        (d / "ledger.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in self.records))


def rec_body(r: dict) -> dict:
    return {k: r[k] for k in ("seq", "ts", "type", "actor", "plane", "payload", "key_id")}


# --------------------------------------------------------------------------- build

def build(root: Path, case: str = "R", code_sha: str = BASE_SHA, run_id: str = RUN_ID, **opts) -> Path:
    """Write one fixture under root/<case>/ and return it. opts (for tests):
      decl_lm, metric/op/value, item_overrides {idx: {...}}, append_after_ingest (mid-run, before the
      result), append_after_result, pending_tail,
      no_ledger, no_job, no_logs, job_missing_in_aws, job_aws_start, extra_items,
      objects (Lambda layout, amendment A11: one S3 object per record, plus a change record timed by
      /aws/lambda/team13-ran-tools)"""
    fx = root / case
    ev = fx / "evidence"
    lm: dict[str, str] = {}
    metric, op, target = opts.get("metric", "pass_rate"), opts.get("op", ">="), opts.get("value", 0.8)

    decl = declaration(run_id, code_sha, metric, op, target)
    its = items(run_id)
    for idx, over in (opts.get("item_overrides") or {}).items():
        its[idx].update(over)
    extra = {**its[0], "item_id": "case-0006", "aws_time": "2026-11-02T18:28:30.000Z"}
    early = its[:]  # what the ingester saw first when the run appends mid-run
    if opts.get("append_after_ingest"):
        its = its + [extra]
    jb = job(run_id, opts.get("job_aws_start", "2026-11-02T18:05:00.000Z"))
    n_pass = sum(1 for i in its if i["passed"])
    res = result(run_id, code_sha, value=n_pass / len(its) if metric == "pass_rate" else opts.get("result_value", 0),
                 passed=(n_pass / len(its)) >= target if metric == "pass_rate" else True, n=len(its),
                 metric=metric, op=op, target=target)

    files = {f"declared/{run_id}.json": doc(decl), f"records/{run_id}/item.jsonl": jsonl(its)}
    late = "2026-11-02T18:06:00Z"  # after the job's CreationTime (18:05)
    lm[f"declared/{run_id}.json"] = opts.get("decl_lm", late if case == "R-late" else "2026-11-02T18:00:00Z")
    lm[f"records/{run_id}/item.jsonl"] = "2026-11-02T18:29:00Z"
    if not opts.get("no_job"):
        files[f"records/{run_id}/job.jsonl"] = jsonl([jb])
        lm[f"records/{run_id}/job.jsonl"] = "2026-11-02T18:31:00Z"
    if opts.get("objects"):
        del files[f"records/{run_id}/item.jsonl"], lm[f"records/{run_id}/item.jsonl"]
        for it in its:
            line = jsonl([it])
            rel = f"records/{run_id}/item/{hashlib.sha256(line).hexdigest()[:24]}.json"
            files[rel], lm[rel] = line, it["aws_time"].replace(":00.000Z", ":01Z")
        ch = change(run_id)
        line = jsonl([ch])
        rel = f"records/{run_id}/change/{hashlib.sha256(line).hexdigest()[:24]}.json"
        files[rel], lm[rel] = line, "2026-11-02T18:12:31Z"
    if case != "R-missing-result":
        files[f"results/{run_id}.json"] = doc(res)
        lm[f"results/{run_id}.json"] = "2026-11-02T18:32:00Z"

    led = MiniLedger()
    for rel in sorted(files, key=lambda r: lm[r]):
        kind = {"declared": "declaration", "records": "record", "results": "result"}[rel.split("/")[0]]
        if opts.get("append_after_ingest") and rel.endswith("item.jsonl"):
            # mid-run: the ingester saw 5 lines, then the 6th (an evidence.appended record), before the result
            led.add("evidence.artifact", rel, jsonl(early), run_id, kind, "2026-11-02T18:28:00.000Z")
            led.add("evidence.appended", rel, files[rel], run_id, kind, lm[rel].replace("Z", ".000Z"))
            continue
        led.add("evidence.artifact", rel, files[rel], run_id, kind, lm[rel].replace("Z", ".000Z"))

    # changes AFTER the ledger recorded the files
    if case == "R-tampered":
        its2 = copy.deepcopy(its)
        its2[4]["passed"], its2[4]["score"] = True, 1.0  # the failed item now "passes"
        files[f"records/{run_id}/item.jsonl"] = jsonl(its2)
        files[f"results/{run_id}.json"] = doc(result(run_id, code_sha, value=1.0, passed=True))
    if opts.get("append_after_result") or opts.get("pending_tail"):
        rel = f"records/{run_id}/item.jsonl"
        files[rel] = files[rel] + jsonl([extra])
        if opts.get("append_after_result"):  # the ingester saw an append AFTER the result was recorded
            led.add("evidence.appended", rel, files[rel], run_id, "record", "2026-11-02T18:40:00.000Z")
    if opts.get("extra_items"):
        rel = f"records/{run_id}/item.jsonl"
        files[rel] = files[rel] + jsonl(opts["extra_items"])

    for rel, b in files.items():
        p = ev / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b)
    if not opts.get("no_ledger"):
        led.write(fx / "ledger")

    events = []
    if not opts.get("no_logs"):
        for iid, _, _, t in ITEMS:
            events.append({"region": "us-east-2", "log_group": "/team13/checkpoint", "log_stream": "gateway/2026-11-02",
                           "timestamp": t, "message": json.dumps({"run_id": run_id, "item_id": iid, "case_id": iid,
                                                                  "decision": "DENY", "tool": "ran___apply_change"}),
                           "event_id": f"ev-{iid}"})
        events.append({"region": "us-west-2", "log_group": "aws/spans", "log_stream": "spans",
                       "timestamp": "2026-11-02T18:06:00.000Z",
                       "message": json.dumps({"attributes": {"run_id": run_id}, "name": "open_case"}),
                       "event_id": "ev-span-0"})
        if opts.get("objects"):
            events.append({"region": "us-east-2", "log_group": "/aws/lambda/team13-ran-tools",
                           "log_stream": "2026/11/02/[$LATEST]fixture", "timestamp": "2026-11-02T18:12:30.000Z",
                           "message": json.dumps({"record": "change_applied", "case_id": "case-0002", "run_id": run_id,
                                                  "approval_id": "chg-0123456789ab",
                                                  "change_text": "cell=1;d_power_db=+1.0;d_tilt_deg=+0.0"}),
                           "event_id": "ev-lambda-0"})
    jobs = []
    if not opts.get("job_missing_in_aws"):
        jobs.append({"region": "us-east-2", "kind": "processing", "name": JOB, "CreationTime": "2026-11-02T18:05:00Z",
                     "ProcessingStartTime": "2026-11-02T18:07:00Z", "ProcessingEndTime": "2026-11-02T18:30:00Z",
                     "LastModifiedTime": "2026-11-02T18:30:05Z", "ProcessingJobStatus": "Completed"})
    aws = {"now": "2026-11-02T20:00:00Z", "s3": {"bucket": BUCKET, "prefix": PREFIX, "last_modified": lm},
           "sagemaker": jobs, "logs": events,
           "log_groups": {"us-east-2": ["/team13/checkpoint", "/aws/lambda/team13-ran-tools"], "us-west-2": ["aws/spans"]},
           "dynamodb": [{"region": "us-east-2", "table": "team13-cases",
                         "item": {"case_id": iid, "run_id": run_id}} for iid, *_ in ITEMS]}
    (fx / "aws.json").write_text(json.dumps(aws, indent=2, sort_keys=True) + "\n")
    return fx


def build_all(root: Path, code_sha: str = BASE_SHA) -> dict[str, Path]:
    return {c: build(root, c, code_sha) for c in ("R", "R-late", "R-tampered", "R-missing-result")}


if __name__ == "__main__":  # regenerate the committed fixtures: python tools/tests/unitfix.py
    import shutil

    out = Path(__file__).resolve().parent / "fixtures" / "units"
    shutil.rmtree(out, ignore_errors=True)
    build_all(out)
    build(out, "R-objects", objects=True)
    print(f"wrote {sorted(p.name for p in out.iterdir())} under {out}")
