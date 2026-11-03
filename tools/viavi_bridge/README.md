# Viavi bridge: operate on the Viavi RSG through the team's browser session

Stage 6 of the RAN RL SDK (`make create-viavi-scenario`, then `make operate`) plays our served PPO
policy on the team's Viavi RSG. The SDK reaches the RSG through the organisers' Viavi adapter, and the
adapter speaks MCP (streamable HTTP) to the RSG's `/mcp`.

**Why a bridge.** The RSG's `/mcp` only answers requests that carry a signed-in browser session. The
twin host's adapter has no such session, so every call it makes is redirected to a sign-in page.
Inside the signed-in Viavi app tab, a same-origin `fetch("/mcp")` works (server `ai-rsg` 1.0, 21
tools). So we run the organisers' adapter, unmodified, on the team's Mac. A small local relay carries
its MCP traffic into that tab.

```
make operate ─http─▶ adapter (ran_rl_host, viavi_rsg, 127.0.0.1:8791)
                       ─MCP─▶ relay.py (127.0.0.1:8790/mcp)
                                ─chrome-devtools-axi eval─▶ fetch("/mcp") in the signed-in Viavi tab ─▶ RSG
```

## Files

| File | What it does |
|---|---|
| `relay.py` | Local `/mcp`. Each POST or DELETE becomes a same-origin fetch in the tab, one at a time. The relay returns the status, `Content-Type`, `Mcp-Session-Id` and the full body; large bodies are read back in chunks. GET returns 405. The log never contains the host, cookies, arguments or bodies. |
| `run_adapter.sh` | Starts the organisers' adapter, configured like the compose service `adapter-viavi-team1` but bound to 127.0.0.1. Its token is read from a mode-600 file whose path you pass in. |
| `declare_viavi_operate.py` | Writes the declaration (unit `t1.rl.viavi_operate`) before the run. Local file only. |
| `record_viavi_operate.py` | Turns `reports/operate.json` into one item per step, then writes the result. Local files only. |

## Retry policy (relay.py)

CloudFront in front of the RSG returns `502 Bad Gateway` ("The request could not be satisfied") on
about 3% of calls. One operate episode makes a few hundred calls, so without retries a run almost
always fails. The relay handles a 502, 503 or 504 as follows:

- **Calls that change nothing** are retried up to 3 times with backoff. These are `initialize`,
  `notifications/initialized`, `tools/list`, `ping` and the read-only tools (`simulation_status`,
  `simulation_query`, `scenario_list`, …).
- **Idempotent calls** are retried the same way:
  - `simulation_command` sets an absolute power or tilt, so sending it twice changes nothing;
  - `simulation_stop`.
- **`simulation_advance` and `simulation_start` are never resent blindly.** Before the call, the relay
  reads `simulation_status` (advance) or `simulation_list` (start). After a 502 it reads them again:
  - if the call took effect, the relay answers with a synthesized success marked `relay_note`, which
    includes the new `test_id` for a start;
  - if it did not, the relay sends the call again;
  - if the relay cannot tell, it passes the 502 through.
- **Any other call that changes state** (for example `scenario_save`) passes the 502 through unchanged.

`tests/test_relay.py` covers every rule offline against a fake RSG.

## Run order

```bash
ST=<organisers' ran-rl staging folder>   # holds twin-source/ and ran-rl-starter/; outside git
export CHROME_DEVTOOLS_AXI_SESSION=viavi CHROME_DEVTOOLS_AXI_HEADED=1 CHROME_DEVTOOLS_AXI_USER_DATA_DIR=<profile>
# 0. a person signs in to the Viavi app in that window; select its /sba/ui/ tab (chrome-devtools-axi pages / selectpage)
python -I tools/viavi_bridge/relay.py --port 8790 --profile-dir <profile> &
PYTHON=<venv>/bin/python tools/viavi_bridge/run_adapter.sh <token-file> "$ST" 8790 8791 &
# 1. declare, then upload declared/<run_id>.json to S3 BEFORE step 2
python -I tools/viavi_bridge/declare_viavi_operate.py --staging "$ST" --evidence-dir <dir>
# 2-3. save the scenario, then operate. MODEL_ZIP is required: without it the Makefile picks
#      reports/train-sagemaker/model.zip
cd "$ST/ran-rl-starter"
export RAN_RL_VIAVI_TWIN_URL=http://127.0.0.1:8791 RAN_RL_VIAVI_ADAPTER_TOKEN="$(cat <token-file>)"
make create-viavi-scenario PYTHON="$ST/.venv/bin/python"
make operate PYTHON="$ST/.venv/bin/python" MODEL_ZIP=<repo>/t1/rl/artifacts/ppo_sm_50k.zip
# 4. record, then upload records/<run_id>/ and results/<run_id>.json
python -I tools/viavi_bridge/record_viavi_operate.py --run-id <run_id> --evidence-dir <dir> \
    --report "$ST/ran-rl-starter/reports/operate.json"
```

`.env` in the starter does not need editing. It holds no Viavi keys, and the SDK reads only
`RAN_RL_TWIN_URL` and `RAN_RL_ADAPTER_TOKEN` from it, so the shell values above are the ones used.

After a failed operate, check `simulation_list` for a ran-rl run that is not STOPPED. Stop it, by its
`test_id` only, before you rerun. The adapter's `/reset` starts a new run without stopping the old
one, and the RSG refuses to start a run while another is active.

## Honest limits

- **No AWS window.** The run uses no AWS job and writes no AWS log, so `check_unit` part c reports
  CANNOT CHECK for it. Parts a, b and e can pass.
- **It only works while someone is signed in.** If the session expires mid-run, the relay returns a
  502 `redirect (sign-in expired?)` and operate fails. Nobody types or stores credentials for it.
- **The step data comes from the organisers' code** (`ran_rl.operate`, SDK 1.2.9). Our code is the
  relay and the declare/record wrappers.
- **The target is measure-only:** `n_items == 5`, one item per completed step, with no KPI threshold.
- **No secrets or organiser data are kept.** No RSG host, URL, IP, cookie or token is recorded, and
  no organiser scenario data is committed.

The run `t1.rl.viavi_operate-20261103T075351Z-25c0` used earlier scratch copies of these scripts.
The only differences are the import paths and the path arguments. Its declaration and the one built
by `declare_viavi_operate.py` have the same hashes, target, items rule and model fields. They differ
only in `code_sha`, the times, the `grader` and `route` text, and how the artifact path is written.
