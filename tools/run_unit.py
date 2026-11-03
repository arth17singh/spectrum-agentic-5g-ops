#!/usr/bin/env python3
"""run_unit.py: run one evidence unit end to end, in the order that makes it count.

  1. baseline (optional): run --baseline-cmd as its own declared run (<run_id>-base) and read its value;
  2. declare: write declared/<run_id>.json BEFORE anything runs (target, test-set hash, code sha, grader);
  3. wait --gap-s (S3 LastModified has one-second resolution; the declaration must be strictly earlier);
  4. run the command, with EVIDENCE_RUN_ID=<run_id> in its environment (it writes records with evlib);
  5. collect: count records/<run_id>/*.jsonl; if the command wrote no result, compute one from the item
     records (evlib.make_result) and write results/<run_id>.json;
  6. check: run tools/check_unit.py <run_id> as a separate process (it shares no code with this writer)
     and exit with its code.

Where evidence goes: EVIDENCE_S3=s3://bucket/evidence/ or EVIDENCE_DIR=/path (see evlib).

Example:
  EVIDENCE_S3=s3://team13-evidence/evidence/ tools/run_unit.py --unit t3.deny --track track-3 \\
      --metric pass_rate --op '>=' --value 0.95 --test-set t3/suite/deny.jsonl \\
      --model-id openai.gpt-oss-20b-1:0 --grader cedar-decision-from-aws-spans -- python -m t3.suite.run deny
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evlib  # noqa: E402


def count_lines(path: Path) -> int:
    with open(path, "rb") as f:
        return sum(1 for line in f if line.strip())


def build_declaration(a, run_id: str, sha: str, dirty: bool, target_value: float, baseline_run_id: str | None,
                      model_id: str) -> dict:
    d = {"run_id": run_id, "unit": a.unit, "track": a.track,
         "target": {"metric": a.metric, "op": a.op, "value": target_value, "unit": a.target_unit},
         "n_items": a.n_items if a.n_items is not None else count_lines(a.test_set),
         "test_set_sha256": evlib.sha256_file(a.test_set), "model_id": model_id, "grader": a.grader,
         "code_sha": sha, "region": a.region, "synthetic": a.synthetic}
    if baseline_run_id:
        d["baseline_run_id"] = baseline_run_id
    if a.prompt:
        d["prompt_sha256"] = evlib.sha256_file(a.prompt)
    if a.parser:
        d["parser"] = a.parser
    if a.decoding:
        d["decoding"] = json.loads(a.decoding)
    if a.twin:
        d["twin"] = json.loads(a.twin)
    if dirty:
        d["code_dirty"] = True
    return d


def check_command(run_id: str, extra: list[str]) -> list[str]:
    return [sys.executable, str(HERE / "check_unit.py"), run_id, *extra]


def run_phase(a, run_id: str, cmd: list[str], model_id: str, target_value: float, baseline_run_id: str | None,
              store, sha: str, dirty: bool, timing: dict, tag: str = "") -> dict:
    """Declare, run, collect and (if needed) write the result. Returns a summary dict."""
    t = time.monotonic()
    decl = build_declaration(a, run_id, sha, dirty, target_value, baseline_run_id, model_id)
    where = evlib.write_declared(decl, store)
    decl = evlib.read_declared(run_id, store)
    timing[f"{tag}declare_s"] = round(time.monotonic() - t, 3)
    print(f"[run_unit] declared {run_id}: {a.metric} {a.op} {target_value} -> {where}", flush=True)
    time.sleep(a.gap_s)

    env = {**os.environ, "EVIDENCE_RUN_ID": run_id, "RUN_ID": run_id}
    t = time.monotonic()
    started = evlib.utc_now()
    rc = subprocess.run(cmd, env=env).returncode
    ended = evlib.utc_now()
    timing[f"{tag}command_s"] = round(time.monotonic() - t, 3)
    print(f"[run_unit] command exited {rc} after {timing[f'{tag}command_s']}s", flush=True)

    t = time.monotonic()
    kinds = {k: len(evlib.read_records(run_id, k, store)) for k in evlib.record_kinds(run_id, store)}
    result = evlib.read_result(run_id, store)
    wrote = False
    items = evlib.read_records(run_id, "item", store) if "item" in kinds else []
    if result is None and items:
        try:
            result = evlib.make_result(decl, items, "tools/run_unit.py", sha,
                                       details={"command_rc": rc, "command_started": started, "command_ended": ended,
                                                "timing_s": dict(timing)})
            result["synthetic"] = a.synthetic
            evlib.write_result(result, store)
            wrote = True
        except evlib.EvidenceError as e:
            print(f"[run_unit] could not compute a result: {e}", flush=True)
            result = None
    timing[f"{tag}collect_s"] = round(time.monotonic() - t, 3)
    print(f"[run_unit] records: {kinds or 'none'}; result: "
          + (f"{result['metric']}={result['value']} passed={result['passed']}"
             + (" (written by run_unit)" if wrote else " (written by the command)") if result else "none"), flush=True)
    return {"run_id": run_id, "rc": rc, "records": kinds, "result": result}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--unit", required=True)
    ap.add_argument("--track", required=True)
    ap.add_argument("--metric", required=True, help="one of: " + ", ".join(evlib.METRICS))
    ap.add_argument("--op", required=True, choices=evlib.OPS)
    tv = ap.add_mutually_exclusive_group(required=True)
    tv.add_argument("--value", type=float, help="the target value")
    tv.add_argument("--delta-over-baseline", type=float, help="target = baseline value + this (needs --baseline-cmd)")
    ap.add_argument("--target-unit", help="unit of the target value, e.g. ratio, count, calls, ms "
                    "(default: evlib.METRIC_UNITS for the metric)")
    ap.add_argument("--test-set", type=Path, required=True, help="the frozen test set file (hashed into the declaration)")
    ap.add_argument("--n-items", type=int, help="default: non-empty lines in --test-set")
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--grader", required=True)
    ap.add_argument("--region", default=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
                    or evlib.DEFAULT_REGION)
    ap.add_argument("--prompt", type=Path)
    ap.add_argument("--parser")
    ap.add_argument("--decoding", help="JSON object of decoding parameters")
    ap.add_argument("--twin", help='JSON object {kind, version, scenario, seed, config?}')
    ap.add_argument("--run-id")
    ap.add_argument("--baseline-run-id", help="an existing run to name as the baseline")
    ap.add_argument("--baseline-cmd", help="command (shell-split) to run first as <run_id>-base")
    ap.add_argument("--baseline-model-id")
    ap.add_argument("--synthetic", action="store_true", help="mark every record synthetic (fixtures, dry runs)")
    ap.add_argument("--allow-dirty", action="store_true", help="allow uncommitted changes (recorded as code_dirty)")
    ap.add_argument("--gap-s", type=float, default=2.0, help="pause between declaring and running (default 2)")
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--check-arg", action="append", default=[], help="extra argument for check_unit.py (repeatable)")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- command to run")
    a = ap.parse_args(argv)
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        ap.error("give the command to run after --")
    a.target_unit = a.target_unit or evlib.METRIC_UNITS.get(a.metric)
    if not a.target_unit:
        ap.error(f"--target-unit is required for metric {a.metric!r}")
    if a.delta_over_baseline is not None and not a.baseline_cmd:
        ap.error("--delta-over-baseline needs --baseline-cmd")

    store = evlib.default_store()
    sha = evlib.code_sha()
    dirty = evlib.code_dirty()
    if dirty and not a.allow_dirty:
        print("[run_unit] the working tree has uncommitted changes, so code_sha would not describe the code that "
              "runs; commit first or pass --allow-dirty", file=sys.stderr)
        return 2
    run_id = a.run_id or evlib.new_run_id(a.unit)
    timing: dict = {}
    t0 = time.monotonic()

    baseline_run_id, target_value = a.baseline_run_id, a.value
    if a.baseline_cmd:
        import shlex

        base = run_phase(a, f"{run_id}-base"[:128], shlex.split(a.baseline_cmd), a.baseline_model_id or a.model_id,
                         a.value if a.value is not None else 0.0, None, store, sha, dirty, timing, "baseline_")
        baseline_run_id = base["run_id"]
        if a.delta_over_baseline is not None:
            if not base["result"]:
                print("[run_unit] the baseline produced no result, so the target cannot be set", file=sys.stderr)
                return 1
            target_value = base["result"]["value"] + a.delta_over_baseline

    main_run = run_phase(a, run_id, cmd, a.model_id, target_value, baseline_run_id, store, sha, dirty, timing)
    code = 0 if main_run["rc"] == 0 and main_run["result"] else 1
    if not a.no_check:
        t = time.monotonic()
        ccmd = check_command(run_id, a.check_arg)
        print(f"[run_unit] checking: {' '.join(ccmd[1:])}", flush=True)
        code = subprocess.run(ccmd).returncode
        timing["check_s"] = round(time.monotonic() - t, 3)
    timing["total_s"] = round(time.monotonic() - t0, 3)
    print(json.dumps({"run_id": run_id, "baseline_run_id": baseline_run_id, "exit": code, "timing_s": timing,
                      "records": main_run["records"]}, sort_keys=True), flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
