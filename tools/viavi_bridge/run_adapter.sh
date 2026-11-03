#!/usr/bin/env bash
# Run the organisers' Viavi adapter (ran_rl_host, backend viavi_rsg) on this Mac, pointed at relay.py.
#
#   tools/viavi_bridge/run_adapter.sh <token-file> <ran-rl-staging dir> [relay-port=8790] [adapter-port=8791]
#
# <token-file>: a mode-600 file holding the adapter's bearer token (make one with
#   umask 077; python3 -c "import secrets; print(secrets.token_urlsafe(32))" > <token-file>).
#   `make create-viavi-scenario` / `make operate` send the same value as RAN_RL_VIAVI_ADAPTER_TOKEN.
# PYTHON: a venv with ran-rl 1.2.9 and its `host` and `viavi` extras (mcp 2.x), e.g.
#   uv venv -p 3.11 venv && uv pip install "ran-rl[host,viavi] @ file://<staging>/ran-rl-starter/ran_rl-1.2.9-py3-none-any.whl"
# Configured like the compose service adapter-viavi-team1, but bound to 127.0.0.1 only.
set -euo pipefail
TOKEN_FILE="${1:?usage: run_adapter.sh <token-file> <staging-dir> [relay-port] [adapter-port]}"
STAGING="${2:?usage: run_adapter.sh <token-file> <staging-dir> [relay-port] [adapter-port]}"
RELAY_PORT="${3:-8790}"
ADAPTER_PORT="${4:-8791}"
PYTHON="${PYTHON:-python}"

[ -f "$TOKEN_FILE" ] || { echo "no token file: $TOKEN_FILE" >&2; exit 2; }
case "$(stat -f '%Lp' "$TOKEN_FILE" 2>/dev/null || stat -c '%a' "$TOKEN_FILE")" in
  600|400) ;;
  *) echo "token file must be mode 600 (chmod 600 $TOKEN_FILE)" >&2; exit 2 ;;
esac

export RAN_RL_BACKEND=viavi_rsg
export RAN_RL_VIAVI_RSG_URL="http://127.0.0.1:${RELAY_PORT}/mcp"
export RAN_RL_VIAVI_RSG_INSECURE=1
export RAN_RL_VIAVI_RSG_CONFIG="$STAGING/twin-source/config/viavi_rsg.yaml"
export RAN_RL_ADAPTER_TRACE=0
RAN_RL_ADAPTER_TOKEN="$(cat "$TOKEN_FILE")"
export RAN_RL_ADAPTER_TOKEN
export PYTHONPATH="$STAGING/twin-source/src"
exec "$PYTHON" -m ran_rl_host.adapter.serve --backend viavi_rsg --host 127.0.0.1 --port "$ADAPTER_PORT"
