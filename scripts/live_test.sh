#!/usr/bin/env bash
# Live test for one run id, the way a judge would do it: READ-ONLY (S3 list/get, Logs Insights, DynamoDB get,
# SageMaker Describe). It never calls a model, an agent, a Lambda or Athena, and writes nothing to AWS.
#
#   scripts/live_test.sh <run_id> [--item <item_id>] [--budget <seconds>]
#
# It (1) prints where each of the run's records lives (S3 key, Logs Insights query, DynamoDB key),
# (2) prints the declaration and the result, (3) runs tools/check_unit.py against the S3 evidence prefix,
# and, with --item, (4) runs tools/show_record.py for that item. It stops on its own at LIVE_TEST_BUDGET_S
# seconds (default 150, i.e. 2:30), so a slow log query cannot run past the time a live test is allowed. A run
# with very many records (the 10,000-item T2 runs read ~10k S3 objects, about 1-2 minutes on a good link) may need
# more: pass --budget <seconds> or set LIVE_TEST_BUDGET_S for that run. When the limit is hit the script says
# "took Xs (over the Ys budget)" and stops; it never hangs.
#
# Environment (all optional):
#   EVIDENCE_S3           evidence prefix, s3://bucket/evidence/  (default: the team data bucket)
#   AWS_PROFILE           default "fyuz" when no AWS credentials are set in the environment
#   AWS_DEFAULT_REGION    default us-east-2
#   PYTHON                interpreter with boto3 (default python3)
#   EVIDENCE_CASES_TABLE  DynamoDB table of case rows (default team13-approvals)
#   LIVE_TEST_BUDGET_S    total seconds allowed (default 150; --budget overrides it for one run)
# Exit code: that of check_unit.py (0 all four parts hold, 1 a part failed, 2 could not check); 3 bad usage.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
RUN_ID="${1:-}"
ITEM=""
BUDGET_ARG=""
usage() {
  echo "usage: scripts/live_test.sh <run_id> [--item <item_id>] [--budget <seconds>]   (run ids look like t1.knowledge-otel-base-1)" >&2
  exit 3
}
if [ -z "$RUN_ID" ] || ! [[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$ ]]; then usage; fi
shift
while [ $# -gt 0 ]; do
  case "$1" in
    --item) ITEM="${2:-}"; [ -n "$ITEM" ] || usage; shift 2 ;;
    --budget) BUDGET_ARG="${2:-}"; shift 2 ;;
    *) usage ;;
  esac
done
if [ -n "$BUDGET_ARG" ] && ! [[ "$BUDGET_ARG" =~ ^[1-9][0-9]{0,5}$ ]]; then
  echo "live_test: --budget wants whole seconds, e.g. --budget 300" >&2
  exit 3
fi

export EVIDENCE_S3="${EVIDENCE_S3:-s3://bb26-p-team13-data-483598626742-use2/evidence/}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-2}"
if [ -z "${AWS_ACCESS_KEY_ID:-}" ] && [ -z "${AWS_PROFILE:-}" ]; then export AWS_PROFILE=fyuz; fi
PY="${PYTHON:-python3}"
BUDGET="${BUDGET_ARG:-${LIVE_TEST_BUDGET_S:-150}}"
if ! [[ "$BUDGET" =~ ^[1-9][0-9]{0,5}$ ]]; then echo "live_test: LIVE_TEST_BUDGET_S must be whole seconds, got '$BUDGET'" >&2; exit 3; fi
CASES_TABLE="${EVIDENCE_CASES_TABLE:-team13-approvals}"
START=$(date +%s)
STOPPED=0
LOG="$(mktemp -t live_test.XXXXXX)"

if ! "$PY" -c "import boto3" 2> /dev/null; then
  echo "live_test: $PY has no boto3. Set PYTHON to an interpreter that has it (pip install boto3)." >&2
  exit 3
fi
elapsed() { echo $(( $(date +%s) - START )); }
rule() { printf '\n== %s\n' "$*"; }

# Run a command with a hard time limit (macOS has no `timeout`). Output goes to $LOG; returns 124 on timeout.
run_limited() {
  local limit="$1"; shift
  "$@" > "$LOG" 2>&1 &
  local pid=$! waited=0
  while kill -0 "$pid" 2> /dev/null; do
    if [ "$waited" -ge "$limit" ]; then kill "$pid" 2> /dev/null; sleep 1; kill -9 "$pid" 2> /dev/null; wait "$pid" 2> /dev/null; return 124; fi
    sleep 1; waited=$((waited + 1))
  done
  wait "$pid"
}

echo "live test for run id: $RUN_ID"
echo "evidence prefix:      $EVIDENCE_S3"
echo "mode:                 read-only (S3 list/get, Logs Insights, DynamoDB get, SageMaker Describe)"
echo "budget:               ${BUDGET}s (default 150; --budget <seconds> or LIVE_TEST_BUDGET_S changes it for this run)"

rule "1. Where this run's records live"
RUN_ID="$RUN_ID" CASES_TABLE="$CASES_TABLE" "$PY" - << 'PYEOF'
import json, os, sys
import boto3

run_id = os.environ["RUN_ID"]
bucket, _, prefix = os.environ["EVIDENCE_S3"][5:].partition("/")
prefix = prefix.rstrip("/") + "/"
s3 = boto3.client("s3")


def head(key):
    try:
        h = s3.head_object(Bucket=bucket, Key=key)
        return h["LastModified"].strftime("%Y-%m-%d %H:%M:%S UTC"), h["ContentLength"]
    except Exception:
        return None


def get(key):
    try:
        return json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
    except Exception:
        return None


print("S3 (bucket " + bucket + "):")
for label, key in (("declaration", f"{prefix}declared/{run_id}.json"), ("result", f"{prefix}results/{run_id}.json")):
    h = head(key)
    print(f"  {label:12} s3://{bucket}/{key}   " + (f"stored {h[0]}, {h[1]} bytes" if h else "NOT FOUND"))
for label, sub in (("records", "records"), ("outputs", "outputs")):
    pfx = f"{prefix}{sub}/{run_id}/"
    n, first = 0, None
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=pfx):
        for o in page.get("Contents", []):
            n += 1
            first = first or o["Key"]
    print(f"  {label:12} s3://{bucket}/{pfx}   " + (f"{n} object(s), first: {first[len(pfx):]}" if n else "none"))

decl, res = get(f"{prefix}declared/{run_id}.json"), get(f"{prefix}results/{run_id}.json")
if decl:
    t = decl.get("target", {})
    print("\nDeclaration (written before the run):")
    print(f"  declared_at {decl.get('declared_at')}   unit {decl.get('unit')}   model {decl.get('model_id')}")
    print(f"  target      {t.get('metric')} {t.get('op')} {t.get('value')}   items {decl.get('n_items')}   test_set_sha256 {str(decl.get('test_set_sha256'))[:16]}...   code {str(decl.get('code_sha'))[:9]}")
if res:
    print("Result:")
    print(f"  {res.get('metric')} = {res.get('value')}   passed = {res.get('passed')}   items {res.get('n_items')}   checked_at {res.get('checked_at')}")

# a baseline run the result's interval is replayed against: check_unit reads that run's items too
for key in ("gain", "bootstrap"):
    spec = ((res or {}).get("details") or {}).get(key)
    if isinstance(spec, dict) and spec.get("baseline_run_id"):
        print(f"\nAlso read by check_unit (the interval's baseline run, details.{key}.baseline_run_id): {spec['baseline_run_id']}")
        break

# a case id, when the records carry one, names the DynamoDB row (only t3 case runs print that row, so only they
# need the lookup; scanning every record of a 10,000-item run for it cost one S3 GET per record)
case_id = None
if run_id.startswith("t3.") and not run_id.startswith("t3.evals."):
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=f"{prefix}records/{run_id}/"):
        for o in page.get("Contents", []):
            rec = get(o["Key"])
            if isinstance(rec, dict) and rec.get("case_id"):
                case_id = rec["case_id"]
                break
        if case_id:
            break

start = (decl or {}).get("declared_at", "<declared_at>")
end = (res or {}).get("checked_at", "<checked_at>")
print("\nLogs Insights (regions us-east-2 and us-west-2; start 5 minutes before declared_at, end 30 minutes after checked_at):")
print(f"  log groups: /team13/checkpoint, aws/spans, /team13/bedrock-invocations, /team13/runner, /aws/lambda/team13-ran-tools,")
print(f"              /aws/sagemaker/TrainingJobs, /aws/sagemaker/ProcessingJobs")
print(f"  window:     {start}  to  {end}")
print(f"  query:      fields @timestamp, @log, @message | filter @message like \"{run_id}\" | sort @timestamp asc | limit 50")
if run_id.startswith("t3.") and not run_id.startswith("t3.evals."):
    print(f"\nDynamoDB (table {os.environ['CASES_TABLE']}, GetItem only):")
    print('  run pointer   pk = "t3-active"   (holds the current run_id)')
    print(f'  case row      pk = "t3-{case_id or "<case_id from a record>"}"   (doc.run_id maps the case to this run)')
else:
    print("\nDynamoDB: this run keeps no DynamoDB rows (it is a SageMaker job; the job record names the job, which check_unit reads with DescribeTrainingJob).")
PYEOF

rule "2. check_unit.py: do the declaration, code, records, ledger hashes and result hold?"
echo "testing run $RUN_ID (records are read from ${EVIDENCE_S3}records/${RUN_ID}/)"
LEFT=$(( BUDGET - $(elapsed) - 5 ))
[ "$ITEM" != "" ] && LEFT=$(( LEFT * 2 / 3 ))
[ "$LEFT" -lt 1 ] && LEFT=1
T2=$(date +%s)
run_limited "$LEFT" "$PY" "$REPO/tools/check_unit.py" "$RUN_ID" --s3 "$EVIDENCE_S3"
RC=$?
TOOK=$(( $(date +%s) - T2 ))
if [ "$RC" -eq 124 ]; then
  echo "check_unit took ${TOOK}s (over the ${LEFT}s it was given out of the ${BUDGET}s budget) and was stopped. Partial output: $LOG"
  echo "This run has many records: re-run with a larger budget, e.g.  scripts/live_test.sh $RUN_ID --budget 600"
  RC=2; STOPPED=1
else
  grep -E '^\[[a-e]\]|^RESULT' "$LOG"
  echo "(full check output, with per-record times: $LOG)"
  echo "check_unit took ${TOOK}s"
fi

if [ -n "$ITEM" ]; then
  rule "3. show_record.py: the item's record and the matching AWS log events"
  LEFT=$(( BUDGET - $(elapsed) - 3 ))
  run_limited "$LEFT" "$PY" "$REPO/tools/show_record.py" --run-id "$RUN_ID" --item "$ITEM" --s3 "$EVIDENCE_S3"
  SR=$?
  if [ "$SR" -eq 124 ]; then echo "show_record took over ${LEFT}s (over the ${BUDGET}s budget) and was stopped"; else cut -c1-220 "$LOG" | head -60; fi
fi

rule "done"
SECS=$(elapsed)
if [ "$STOPPED" -eq 1 ]; then OVER="check_unit was stopped at its time limit (budget ${BUDGET}s)"
elif [ "$SECS" -gt "$BUDGET" ]; then OVER="took ${SECS}s (over the ${BUDGET}s budget)"; else OVER="within the ${BUDGET}s budget"; fi
printf 'elapsed %d:%02d (limit %d:%02d): %s   check_unit exit %s\n' $((SECS / 60)) $((SECS % 60)) $((BUDGET / 60)) $((BUDGET % 60)) "$OVER" "$RC"
exit "$RC"
