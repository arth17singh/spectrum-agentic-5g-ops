#!/usr/bin/env bash
# Merge gate for the SPECTRUM dashboard (run from any directory; `make gate` in dashboard/ calls it).
# Steps, all must pass:
#   1. frontend: tsc (app + node configs), oxlint --max-warnings 111, registry check, vite build
#   2. canonical-hash parity (Python vs JS)
#   3. backend: uv run pytest -q (after the build, so the built-UI tests run too)
#   4. backend on 127.0.0.1:$GATE_PORT (default 8790) with a throwaway evidence dir and key, then the
#      route smoke in headless Chrome over $SMOKE_ROUTES (+ $EXTRA_ROUTES); only the gate's own server
#      (its process group) is stopped after. A port already in use is refused, never freed: pick another.
# Never run it in the main checkout: it rebuilds frontend/dist, which the live :8787 serves.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DASH="$REPO/dashboard"
FE="$DASH/frontend"
BE="$DASH/backend"
BIN="$FE/node_modules/.bin"
PORT="${GATE_PORT:-8790}"
ROUTES="${SMOKE_ROUTES:-/,/federation,/proving,/release,/swarm,/oversight,/trust,/audit,/evidence,/observability,/settings,/m/approve,/presenter,/approvals,/ask-noc}"
[ -n "${EXTRA_ROUTES:-}" ] && ROUTES="$ROUTES,$EXTRA_ROUTES"

if [ "$PORT" = "8787" ]; then echo "gate: refusing port 8787 (the live console)"; exit 2; fi
port_busy() { lsof -nP -iTCP:"$PORT" -sTCP:LISTEN > /dev/null 2>&1; }
refuse_busy() {
  echo "gate: port $PORT is in use by another process; refusing to run. Pick a free GATE_PORT (check with"
  echo "      lsof -nP -iTCP:<port> -sTCP:LISTEN). The gate never stops a server it did not start."
  lsof -nP -iTCP:"$PORT" -sTCP:LISTEN 2> /dev/null | tail -n +2 | awk '{print "      held by pid " $2 " (" $1 ")"}'
  exit 2
}
port_busy && refuse_busy
if [ ! -x "$BIN/tsc" ]; then
  echo "gate: $FE/node_modules missing; in a worktree: ln -s <main checkout>/dashboard/frontend/node_modules $FE/node_modules"
  exit 2
fi

step() { printf '\n== %s\n' "$*"; }

step "tsc (app, node)"
(cd "$FE" && "$BIN/tsc" -p tsconfig.app.json --noEmit && "$BIN/tsc" -p tsconfig.node.json --noEmit)

step "oxlint --max-warnings 111"
(cd "$FE" && "$BIN/oxlint" --max-warnings 111 > /dev/null) || { (cd "$FE" && "$BIN/oxlint" --max-warnings 111 | tail -20); exit 1; }
echo "oxlint ok"

step "feature registry"
node "$FE/scripts/check-features.mjs"

step "vite build"
(cd "$FE" && "$BIN/vite" build --logLevel warn)
echo "built $FE/dist"

step "canonical parity"
node "$FE/scripts/check-canonical.mjs"

step "backend pytest"
(cd "$BE" && uv run pytest -q)

step "route smoke on 127.0.0.1:$PORT"
port_busy && refuse_busy  # again: something may have bound it while the build and tests ran
TMP="$(mktemp -d)"
LOG="$TMP/server.log"
# The server runs in its own process group (job control on), so cleanup stops `uv run` and the uvicorn
# child it spawns, and nothing else: never a pkill by command line, which also hits other people's servers.
cleanup() {
  [ -n "${SERVER_PID:-}" ] || return 0
  kill -TERM -- "-$SERVER_PID" 2> /dev/null || true
  for _ in 1 2 3 4 5 6 7 8 9 10; do kill -0 -- "-$SERVER_PID" 2> /dev/null || return 0; sleep 0.3; done
  kill -KILL -- "-$SERVER_PID" 2> /dev/null || true
}
trap cleanup EXIT
set -m
(cd "$BE" && SPECTRUM_EVIDENCE_DIR="$TMP/ev" SPECTRUM_EVIDENCE_KEY="$TMP/k" SPECTRUM_RECORDS_ROOT="file://$TMP/ev" \
  SPECTRUM_OFFLINE=1 PYTHONPATH=. exec uv run uvicorn spectrum.main:app --host 127.0.0.1 --port "$PORT" > "$LOG" 2>&1) &
SERVER_PID=$!
set +m
for _ in $(seq 60); do
  curl -sf "http://127.0.0.1:$PORT/api/config" > /dev/null 2>&1 && break
  sleep 0.5
done
curl -sf "http://127.0.0.1:$PORT/api/config" > /dev/null || { echo "gate: backend did not start"; tail -30 "$LOG"; exit 1; }
# the listener must be ours (our process group), not a server that took the port in the meantime
owner_pgids=$(for p in $(lsof -nP -t -iTCP:"$PORT" -sTCP:LISTEN 2> /dev/null); do ps -o pgid= -p "$p" | tr -d ' '; done)
echo "$owner_pgids" | grep -qx "$SERVER_PID" || { echo "gate: port $PORT is served by another process, not the gate's server"; tail -30 "$LOG"; exit 2; }
# the catch-all must not swallow unknown API paths
code=$(curl -s -o /dev/null -w '%{http_code} %{content_type}' "http://127.0.0.1:$PORT/api/__gate_nope__")
[ "$code" = "404 application/json" ] || { echo "gate: /api/__gate_nope__ answered '$code', expected a JSON 404"; exit 1; }
echo "unknown /api path -> $code"
node "$FE/scripts/smoke.mjs" --base "http://127.0.0.1:$PORT" --routes "$ROUTES" ${SMOKE_THEMES:+--themes "$SMOKE_THEMES"}

printf '\ngate: PASS\n'
