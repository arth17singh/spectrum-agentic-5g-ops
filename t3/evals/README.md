# t3/evals: does the swarm fix the network, and by how much more than simpler options

A frozen suite of misconfiguration incidents on the in-process Maveric twin. Every grade comes from a replay of the twin, never from a model. `PY` below is the shared env: `/private/tmp/claude-501/-Users-satyabratsrikumar-firstmate-homes-AWS-Hackathon/b281b863-2729-4833-baf6-93b1e8b525db/scratchpad/venv-t3/bin/python`. Run every command from the repo root.

- **The suite:** `suite/incidents-test.jsonl` holds 33 test incidents: 15 on the visible tier and 18 on the masked tier, 15 on the official preset and 18 on the viavi-like one. Its sha256 is in `suite/incidents-test.sha256`. The dev split and the generator metadata sit next to it. Generate it with `incidents.py`.
- **Targets and tolerances:** `targets.json`. Every run records this file's sha256 in `declared.twin.config.eval.targets_sha256`.
- **Graders:** `graders.py`. Operators: `operators.py` and `swarm.py`. Runner and paired bootstrap gains: `run.py` and `stats.py`.
- **Evidence:** `evidence/` uses the contract layout: `declared/`, `records/<run_id>/item.jsonl` and `results/`.

## After merging W2 (`twin/`) and W6 (`tools/evlib.py`): the integrator's procedure

`twin_backend.py` picks W2's `twin.replay` and `twin.kpis` automatically once they import. `evshim.py` does the same for W6's `tools.evlib`. Until then, the labelled stand-ins `twin_standin.py` and the local shim are used. The frozen suite and the four offline runs were produced with those stand-ins.

1. Check the suite against W2's real twin:
   ```
   PY -m t3.evals.incidents --check
   PY -m pytest -q t3/evals/tests
   ```
   You expect `"matches_frozen": true` and every test passing. Confirm that W2's twin is the one in use with `PY -c "import t3.evals.twin_backend as b; print(b.BACKEND)"`, which should print `maveric-inprocess`.
2. **If the bytes match:** you're done. The offline runs in `evidence/` stand. Go to "Scoring the swarm".
3. **If the bytes differ: re-freeze once, before any swarm run.** Never do this after a swarm run has used the suite.
   ```
   PY -m t3.evals.incidents --write
   ```
   - Put the new sha256 into `FROZEN_SHA` in `tests/test_incidents.py`.
   - Move the old `evidence/` directory aside; it was graded against the old suite.
   - Commit: tests green, message "re-freeze suite against W2 twin".
   - Redo the four offline runs from the committed tree (the runner refuses a dirty tree). Each run prints its `run_id`; use those ids in the later commands.
     ```
     PY -m t3.evals.run --operator no_action
     PY -m t3.evals.run --operator rule_baseline --baseline-run-id <no_action>
     PY -m t3.evals.run --operator config_revert --baseline-run-id <no_action>
     PY -m t3.evals.run --operator oracle --workers 10 --baseline-run-id <no_action> <rule> <config_revert>
     ```
     The first three take a few seconds each; the oracle takes about 7 minutes with 10 workers.
   - Commit `t3/evals/evidence/`.

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

The client protocol is in the `swarm.py` docstring. `factory()` reads its endpoints and tokens from environment variables. Without `--swarm-client`, the run uses `DryRunSwarmClient`, a fake that is marked synthetic. Don't keep dry-run results in `evidence/`.
