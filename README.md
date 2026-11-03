# Track 3: Multi-Agent Swarms for Application Enablement through NLMs (team13)

Team code `team13`. Hand-in: 12:00 Seattle time, 3 Nov 2026. Primary track: Track 3. Track 1 is the second entry. Track 2 and the Bridge depend on organiser provisioning and may be absent; what is blocked or not attempted is in [`GAPS.md`](GAPS.md). The machine-readable summary for judges is [`judging.json`](judging.json); every dataset and model output is listed in [`DATA.md`](DATA.md).

## The problem

1. Operators will not let an AI model change a live radio network because a demo looked good. They need evidence a risk owner can check: a target written down before each run, a frozen test, and records that agree with what AWS logged.
2. So we put a small team of agents in front of a network twin. Every tool call crosses one policy checkpoint (Cedar decision, guardrails, an audit record with one trace id), and a person approves every state change.
3. A model is pulled out of service automatically when its predictions stop matching what the network does. Every number we report opens to a record you can read with read-only AWS calls, or rebuild offline from `evidence/`.

## Architecture

Short version; the longer plan and the decision records belong in `planning/` (TODO: the lead adds them before hand-in; until then this section is the architecture).

```
person approves a change (phone / dashboard)
        |
 noc_lead (lead agent) --A2A--> ran_monitor . ran_optimizer . ran_explainer     (Tupl Agent Builder, Bedrock models)
        |  every tool call, with the agent's own token
        v
 AgentCore Gateway -> approval interceptor -> Cedar ENFORCE (organiser rules + ours in t3/policies) -> tools Lambda (t3/tools)
        |                                                                     |
        |                                          4-cell network twin: Maveric in-process (R1), Viavi (R3, only with a machine credential)
        v
 trust gate (t3/trust): predicted vs measured KPI change -> trust score -> Cedar pause of a model's tool; only a person resumes

 Evidence: declaration -> SageMaker job / Lambda (run id = job name) -> per-item records -> result -> tools/check_unit.py PASS/FAIL
           -> judging.json -> read-only live test (S3, CloudWatch Logs Insights, DynamoDB) and an offline rebuild
 Dashboard (SPECTRUM, dashboard/): Twin, Ladder, Swarm trace, Scorecard, Evidence explorer
```

Where things live:

| Path | What |
|---|---|
| `contracts/` | the record schemas (declared, item, decision, change, control, job, model, result), one source for Python and TypeScript |
| `twin/` | in-process replay of the Maveric twin and its KPI definitions, with parity checks |
| `t1/evals/` | Track 1 frozen item sets, one prompt, one parser, deterministic graders |
| `t1/rl/` | Track 1 PPO policy (v0, local) |
| `t2/`, `silos/` | Track 2 response-model dataset and the three simulated operator silos |
| `t3/suite/` | the frozen call suite (deny, allow, pause, trace) and its checker |
| `t3/tools/`, `t3/policies/`, `t3/trust/`, `t3/agents/`, `t3/deploy/` | governed RAN tools, Cedar rules, trust gate, agent prompts, registration on the organisers' checkpoint |
| `t3/evals/` | frozen incident suite and offline baselines for the swarm |
| `tools/` | `run_unit.py`, `check_unit.py`, `show_record.py`, `judging_json.py` and the evidence writer |
| `dashboard/` | the SPECTRUM console (see "Before Nov 1") |
| `evidence/` | the evidence mirror and the signed ledger's public key |

## One command per stage

Run from the repository root with Python 3.12. The organisers' `ran-rl` SDK is not redistributed (its terms forbid it), so it is not in this repo; install it from the organisers' package. `<run_id>` is a run id from `judging.json`. TODO marks a step that has not run or whose command is not built yet.

| Stage | Command | Status |
|---|---|---|
| Launch any scored run as a SageMaker job | `python -m tools.sm.run --run-id <run_id> --declaration decl.json --entry "-m <module> --run-id {run_id}" [--dry-run]`; `python -m tools.sm.run --finish <run_id>`. It declares in S3 first, submits the job, summarises and runs the check. Refuses a dirty tree. | built, used for the runs below |
| Link test (SDK to hosted twin) | from the organisers' starter kit, outside this repo: `make check-aws`, then its SageMaker link job (`GAPS.md`, G-001) | done: `t1.rl.link-test-20261102T234623Z-7368` PASS |
| Train: Track 1 RL | `t1/rl/official/prepare.sh` builds the SageMaker project from the starter, outside the repo. Reward check `t1.rl.reward-check-20261103T020244Z-4f92` PASS. The full 100k-step run crashed at about 67.6k steps (G-008) | scored on the 50k checkpoint: `t1.rl.eval-20261103T043115Z-bf52` **FAIL**, 5 of 6 gates (UPLIFT fails). The job entry points for it are on a lane not merged at `175860e` |
| Train: Track 1 policy v0 (local, labelled) | `python -m t1.rl.train --out <dir> --minutes 30` | done, not a scored run |
| Train: Track 1 LoRA fine-tune | `python -m t1.sft.stage_base`, `python -m t1.sft.build`, `python -m t1.sft.train_lora --run-id <id> --base ... --data ...` (request files in `t1/sft/jobs/`) | stage `t1.sft.stage-20261103T034500Z-5b1e` PASS, data `t1.sft.data-20261103T034500Z-7c2a` PASS (1,220 examples); training `t1.sft.train-20261103T034500Z-9d4f` declared, no result yet |
| Train: Track 2 | `python -m silos.job_data`, `python -m silos.job_baselines`, `python -m silos.job_evaluate`, `python -m silos.job_compare` (`python -m silos.launch ... --dry-run` prints the request). Local generators: `python -m silos.generate`, `python -m t2.gen`, `python -m t2.train` | simulated silos: data, local-only A/B/C, central and local FedAvg runs are on AWS and PASS their declared checks; the FL service is not provisioned for the real session (G-005) |
| Evaluate: Track 1 | `python -m t1.evals.freeze --verify`; scored runs go through `tools.sm.run` with entry `-m t1.evals.run --unit <interface\|execution\|safety\|knowledge\|price_latency> --client local-hf --model-dir {channel:model} ...` (tuned run adds `--baseline-run-id <base run id>`) | base runs done for all five units: price-latency PASS; knowledge, execution, safety and interface **FAIL** their declared targets. Tuned runs wait for the LoRA job. Each execution evaluation is pinned to the instance type that froze its items (G-009) |
| Evaluate: replay check and re-freeze | `python -m t1.evals.replay_job check\|refreeze` as SageMaker jobs | runs `t1.execution.replay-check-*` and `t1.execution.refreeze-x86-1` are on AWS; the re-frozen items are not yet in the repo (`DATA.md` section 2) |
| Evaluate: trust gate and ladder (CPU jobs) | `python -m t3.trust.job_gate_eval --version v1 --model healthy` (and `degraded`; `--version v2`), `python -m t3.trust.job_ladder --model healthy` (and `degraded`) | entry points built; AWS runs TODO |
| Evaluate: Track 3 swarm incidents | `python -m t3.evals.incidents --check`; baselines `python -m t3.evals.run --operator no_action` (then `rule_baseline`, `config_revert`, `oracle`, each with `--baseline-run-id`); swarm: `python -m t3.evals.run --operator swarm --mode proposal --subset masked --swarm-client <module:factory> --baseline-run-id <config_revert run>` | baselines `no_action`, `rule_baseline`, `config_revert` (all and masked) are on AWS and PASS; `oracle` and the swarm run TODO |
| Track 3 units (deny, allow, pause, trace) | `python -m t3.suite.runner --run-id <run_id> --unit all --live` (waits for a person to approve; never approves itself) | TODO: needs the deployed checkpoint and the real approvals page (G-011) |
| One unit end to end, locally | `tools/run_unit.py --unit <unit> --track track-3 --metric ... --op ... --value ... --test-set <file> --model-id ... --grader ... -- <command>` | built |
| Check | `python tools/check_unit.py <run_id> --s3 s3://<team data bucket>/evidence/` (exit 0 all parts hold, 1 a part failed, 2 could not check) | built. Part e cannot recompute the gain metric of the Track 3 baselines (G-012) |
| Live test (judges, read-only) | `scripts/live_test.sh <run_id> [--item <item_id>]`: prints where each record lives (S3 key, Logs Insights query, DynamoDB key), then runs `check_unit.py`; stops at 2:30. Lower level: `python tools/show_record.py --run-id <run_id> --item <item_id> --s3 s3://<team data bucket>/evidence/` | built; rehearsed on two real runs in 0:22 to 0:28 (no read-only role to rehearse under, G-004) |
| Fill `judging.json` | `python tools/judging_json.py --s3 s3://<team data bucket>/evidence/ --include <run ids> --order-file <file> --dry-run`, then without `--dry-run`; `python tools/judging_json.py --slots-report` shows which of the 20 planned slots are filled | built; 6 of 20 slots filled |
| Tests | `python -m pytest -q scripts/hooks/tests tools/tests t1/evals/tests t3/suite/tests t3/evals/tests` (more under each package); dashboard gate: `scripts/gate.sh` in a worktree | |

## Rebuild (no AWS access)

```
make -C dashboard evidence-rebuild
```

This re-reads `evidence/` with the committed public key only, re-verifies the signed hash chain, every Merkle anchor and every file hash, and reprints the per-run summary. It exits 0 when clean, 1 when an artifact changed or is missing, 2 when the chain is broken. To check a single unit against local files: `python tools/check_unit.py <run_id> --dir evidence` (times are still read from AWS unless run with `--offline`).

## Before Nov 1

What existed before the event started on 1 Nov (git: `5128f97`, 31 Oct):

- the SPECTRUM dashboard (`dashboard/`: a React console and a FastAPI simulation server) and the static `docs/` pages. It walks one network model through learn, verify, release, operate and prove. **It is mostly simulated:** no model is trained or called; the federation curves, the model scorecards, the release traffic and the swarm's reasoning are scripted or invented. Only the signed hash-chained ledger, the approval state machine and the policy default-deny were real.
- four security fixes to that server (31 Oct, before the event).

What was built at the event (1 to 3 Nov), none of it simulated unless a record says so:

- the evidence chain: record contracts, declarations before runs, per-item records, `tools/` checker and live-test script, and the dashboard's persisted, signed evidence ledger and Evidence screen;
- `twin/`: an in-process replay of the Maveric twin with parity checks; frozen Track 1 item sets and graders; the Track 3 call suite, incident suite and offline baselines; governed RAN tools, Cedar rules and the trust gate;
- the Track 2 operator-silo datasets and response-model baseline, and a Track 1 PPO policy (v0, local);
- the hand-in files, `GAPS.md`, and the pre-push guard.

Anything that runs on a local simulator is labelled as such and listed in `GAPS.md`, because its records cannot agree with AWS times.

## Pre-push guard

A tracked hook scans every commit being pushed, plus `evidence/`, for secrets and organiser files, and blocks the push with the rule and file name only. Enable it once per clone (it is not enabled globally):

```
git config core.hooksPath scripts/hooks
python3 -m pytest -q scripts/hooks/tests        # its tests (all values are fake)
python3 -I scripts/hooks/prepush_scan.py --refs HEAD --no-evidence   # audit a whole history by hand
```

Details are in [`scripts/hooks/README.md`](scripts/hooks/README.md).
