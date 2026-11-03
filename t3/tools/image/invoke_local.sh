#!/usr/bin/env bash
# Run the image locally under the Lambda runtime interface emulator (RIE, built into the base image)
# and POST sample events to it. No AWS calls, no DynamoDB (T3_STORE=local:/tmp/t3store).
#
#   t3/tools/image/invoke_local.sh [--smoke] [--memory 3g] [--cpus 2] [--keep] [event.json ...]
#
#   --smoke   override CMD with smoke.smoke_handler.handler (proves torch/gpytorch/GP/SB3 work)
#   --memory  docker --memory limit, default 3g (Lambda gives ~1.7 vCPU per 3008 MB)
#   --cpus    docker --cpus limit, default 2
#   --keep    leave the container running after the events (stop it with: docker rm -f team13-ran-tools-local)
#
# Default events when none given: events/smoke_5.json (with --smoke) else events/tool_observe.json.
# Host port 9000, bound to 127.0.0.1 only. The Gateway puts the tool name in
# context.client_context.custom["bedrockAgentCoreToolName"]; the emulator cannot set client context,
# so with T3_TOOL_NAME_FALLBACK=1 the handler (t3.tools.app) must accept event["_tool"] instead.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# build.sh tags images with the commit's 12-hex sha; default to the newest such build.
IMAGE="${IMAGE:-$(docker image ls team13-ran-tools --format '{{.Repository}}:{{.Tag}}' | head -1)}"
[ -n "$IMAGE" ] || { echo "no team13-ran-tools image: run build.sh <sha> first, or set IMAGE" >&2; exit 1; }
NAME=team13-ran-tools-local
PORT=9000
MEMORY=3g
CPUS=2
SMOKE=0
KEEP=0
EVENTS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --smoke) SMOKE=1 ;;
    --memory) MEMORY="$2"; shift ;;
    --cpus) CPUS="$2"; shift ;;
    --keep) KEEP=1 ;;
    *) EVENTS+=("$1") ;;
  esac
  shift
done
if [ ${#EVENTS[@]} -eq 0 ]; then
  if [ $SMOKE -eq 1 ]; then EVENTS=("$HERE/events/smoke_5.json"); else EVENTS=("$HERE/events/tool_observe.json"); fi
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true
cleanup() { [ $KEEP -eq 1 ] || docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

CMD_ARGS=()
if [ $SMOKE -eq 1 ]; then CMD_ARGS=("smoke.smoke_handler.handler"); fi

START=$(python3 -c 'import time; print(time.time())')
docker run -d --name "$NAME" --platform linux/arm64 \
  --memory "$MEMORY" --cpus "$CPUS" \
  -p 127.0.0.1:${PORT}:8080 \
  -e T3_STORE=local:/tmp/t3store \
  -e T3_TOOL_NAME_FALLBACK=1 \
  -e AWS_REGION=us-east-2 -e AWS_DEFAULT_REGION=us-east-2 \
  "$IMAGE" ${CMD_ARGS[@]+"${CMD_ARGS[@]}"} >/dev/null

# Wait until the RIE answers (it listens immediately; the function module loads on first invoke).
for _ in $(seq 1 50); do
  if curl -s -o /dev/null "http://127.0.0.1:${PORT}/" 2>/dev/null; then break; fi
  sleep 0.2
done

for ev in "${EVENTS[@]}"; do
  # Allow paths relative to this directory (events/smoke_5.json) as well as to the cwd.
  if [ ! -f "$ev" ] && [ -f "$HERE/$ev" ]; then ev="$HERE/$ev"; fi
  [ -f "$ev" ] || { echo "event file not found: $ev" >&2; exit 1; }
  echo "--- POST $ev"
  curl -s -o /tmp/t3_invoke_out.json -w "http=%{http_code} total=%{time_total}s\n" \
    -XPOST "http://127.0.0.1:${PORT}/2015-03-31/functions/function/invocations" \
    -d @"$ev"
  head -c 3000 /tmp/t3_invoke_out.json; echo
done

if [ $KEEP -eq 1 ]; then echo "container $NAME left running on 127.0.0.1:${PORT}"; fi
elapsed=$(python3 -c "import time; print(round(time.time()-$START,1))")
echo "total wall time incl. container start: ${elapsed}s"
