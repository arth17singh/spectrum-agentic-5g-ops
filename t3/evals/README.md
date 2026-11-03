# t3/evals: does the swarm fix the network, and by how much more than simpler options

A frozen suite of misconfiguration incidents on the in-process Maveric twin. Every grade comes from a replay of the twin, never from a model. `PY` below is the shared env: `/private/tmp/claude-501/-Users-satyabratsrikumar-firstmate-homes-AWS-Hackathon/b281b863-2729-4833-baf6-93b1e8b525db/scratchpad/venv-t3/bin/python`. Run every command from the repo root.

- **The suite:** `suite/incidents-test.jsonl` holds the test incidents of record (v4, frozen on x86_64 in SageMaker by run t3.evals.suite-freeze-20261103T031439Z-32b3: 33, of which 15 visible and 18 masked). Its sha256 is in `suite/incidents-test.sha256`. The dev split and the generator metadata sit next to it. Generate it with `incidents.py`. Its input, the 84 candidate specs, is `suite/candidates.jsonl`.
- **Targets and tolerances:** `targets.json`. Every run records this file's sha256 in `declared.twin.config.eval.targets_sha256`.
- **Graders:** `graders.py`. Operators: `operators.py` and `swarm.py`. Runner and paired bootstrap gains: `run.py` and `stats.py`.
- **Evidence:** `evidence/` uses the contract layout: `declared/`, `records/<run_id>/item.jsonl` and `results/`.

## Platform of record: the suite is frozen and scored in SageMaker (x86_64)

The twin is a float32 Gaussian-process model, so frozen KPIs replay to the 1e-6 tolerance only on the platform that froze them. A suite frozen on this Mac (arm64) misses that tolerance in an x86_64 SageMaker job. That happened to suite v3 (`485b3bde…`): `inc-off-mas-4102-tilt+5` failed to replay there. So:

- **Where the suite of record is frozen:** inside a SageMaker job started by `tools/sm/run.py`, on ml.m5.xlarge with the CPU image, using the real twin (`twin/` must be in the shipped pathspec).
- **Where it is scored:** every scored run that replays it (all baselines and the swarm) runs as a runner job on the same instance family.
- **What the freeze job declares:** `test_set_sha256` = sha256 of `suite/candidates.jsonl`, the generator's input (84 candidate specs, one canonical line each, committed).
- **What the freeze job writes:**
  - one `item` per candidate: input = canonical spec, output = canonical incident or drop record;
  - a `result` with `n_items == 84`, written through evlib so that `tools/check_unit.py` applies;
  - the four suite files into `{output_dir}/suite`, never into the code tree.
- **Commit the suite only from the job's output.** Copy the four files into `t3/evals/suite/` and set `FROZEN_SHA` in `tests/test_incidents.py`.
- **Re-declare the baselines.** Re-run them as runner jobs and mark every earlier recorded run as superseded in `evidence/superseded.json`.
- **What still runs locally:** tests and local runs use incidents built on the local machine (`tests/_local.py`, `run.py --suite`). `test_regenerating_the_suite_reproduces_the_frozen_bytes` runs only where `incidents-meta.json`'s platform, backend and suite version match the local ones, and skips elsewhere.
- **Replay errors name the KPI:** a replay mismatch reports each KPI that missed the tolerance, with the replayed value, the frozen value and the difference.
- **No git in a job:** `evshim` takes the commit from `CODE_SHA` / `TEAM13_CODE_SHA` in a job and never calls git there; locally it uses `tools/evlib`.

Freeze job (CPU, ml.m5.xlarge):
```
python -m tools.sm.run --run-id t3.evals.suite-freeze-<UTC>-x86 \
  --declaration tools/sm/examples/t3-evals-suite-freeze.json \
  --entry "-m t3.evals.incidents --out {output_dir}/suite --evidence-dir {evidence_dir} --run-id {run_id}" \
  --instance-type ml.m5.xlarge --max-run 1800 \
  --paths contracts --paths t3 --paths tools --paths twin [--dry-run]
```

Local checks (any platform):
```
PY -m pytest -q t3/evals/tests                      # also with PYTHONPATH=contracts
PY -m t3.evals.incidents --out /tmp/x/suite         # a local freeze, for comparison only
```

## Scoring the swarm (declared target in `targets.json` → `units.swarm`)

- **Primary (the pass line):** the paired recovery gain of `swarm` over `config_revert` on the masked tier. The lower 95% bootstrap bound must be above 0.
- **Secondary (reported only):** the success rate, and wrong-cell actions no higher than `rule_baseline`'s on the same incidents.

Scored run (proposal mode on the masked subset, no human approvals):
```
PY -m t3.evals.run --operator swarm --mode proposal --subset masked \
    --swarm-client <module:factory> --baseline-run-id <config_revert> <rule_baseline>
```
- **What proposal mode does:** each operator step opens one front-door case at the harness's current state. Its `start_action_log` holds commissioning, the fault and any earlier proposals.
- **What the swarm does:** it works the case up to the checkpoint's approval refusal on `ran___apply_change`, then stops.
- **How it's graded:** the client returns the proposed canonical string, and the harness grades it on its own twin replay. It never applies it through the governed tool.
- **What gets recorded:** `mode=proposal` in `declared.twin.config.eval.mode` and on every item (`extra.mode`).
- **Cost:** `--proposal-steps N` (default 3) caps the cases per incident. The masked subset has 18 incidents, so up to 54 cases.

Demo run with real approvals (2-3 cases): use `--mode full`. Each step returns only a change that was actually applied after approval; a refused step counts as no change.

**How it runs tonight (two phases).** The job role reaches neither DynamoDB nor AgentCore, so the swarm acts live from a terminal and the score is computed on x86 from what it did:
1. `python -m t3.evals.live_swarm --run-id t3.swarm-live-<UTC> --live` (team profile): declares first, then per (masked incident, step) maps and seeds a case with `ran___seed_case` (app token only), sends the ticket to `noc_lead` through `front___continue_case`, and reads the proposal from the approval refusal. It writes `outputs/<run>/swarm-actions.jsonl`.
2. The scored run in a SageMaker CPU job, `--swarm-client t3.evals.recorded_client:factory` with that file as the `recording` input: the recorded client returns exactly the live proposals and refuses a harness whose earlier steps differ. Declaration: `tools/sm/examples/t3-evals-swarm-masked.json`.

The client protocol is in the `swarm.py` docstring. `factory()` reads its endpoints and tokens from environment variables. Without `--swarm-client`, the run uses `DryRunSwarmClient`, a fake that is marked synthetic. Don't keep dry-run results in `evidence/`.
