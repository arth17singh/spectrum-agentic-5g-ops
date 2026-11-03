# Trusted Data and AI Evidence Chain (cross-cutting track, every track), team13

Team code `team13`. Hand-in: 12:00 Seattle time, 3 Nov 2026. Primary track: the evidence chain (cross-cutting). It runs through every entry: Track 3 (agent checkpoint and swarm), Track 1 (NLM), Track 2 (one FL-service session). Track 2 ran one session on the FL service (`t2.fl_federated.disjoint-20261103T051311Z`; judged slot 9: silo C PASS, silo B MISSED). The Bridge was not attempted (G-018). What is blocked or not attempted is in [`GAPS.md`](GAPS.md). The machine-readable summary for judges is [`judging.json`](judging.json); every dataset and model output is listed in [`DATA.md`](DATA.md).

## The problem

1. Operators will not let an AI model change a live radio network because a demo looked good. They need evidence a risk owner can check: a target written down before each run, a frozen test, and records that agree with what AWS logged.
2. So we put a small team of agents in front of a network twin. Every tool call crosses one policy checkpoint (Cedar decision, guardrails, an audit record with one trace id), and a person approves every state change.
3. A model is pulled out of service automatically when its predictions stop matching what the network does. Every number we report opens to a record you can read with read-only AWS calls, and the console's signed evidence ledger re-verifies from the team's evidence mirror (see [Rebuild](#rebuild-no-aws-access)).

## Architecture

Short version: this section is the architecture.

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

Run from the repository root with Python 3.12. The organisers' `ran-rl` SDK is not redistributed (its terms forbid it), so it is not in this repo; install it from the organisers' package. `<run_id>` is a run id from `judging.json`. The Status column names the runs each step produced; a step that did not run says so.

| Stage | Command | Status |
|---|---|---|
| Launch any scored run as a SageMaker job | `python -m tools.sm.run --run-id <run_id> --declaration decl.json --entry "-m <module> --run-id {run_id}" [--dry-run]`; `python -m tools.sm.run --finish <run_id>`. It declares in S3 first, submits the job, summarises and runs the check. Refuses a dirty tree. | built, used for the runs below |
| Link test (SDK to hosted twin) | from the organisers' starter kit, outside this repo: `make check-aws`, then its SageMaker link job (`GAPS.md`, G-001) | done once: `t1.rl.link-test-20261102T234623Z-7368`, an infrastructure link test from before the runner pinned code (superseded, not a result; `evidence/superseded.json`) |
| Train: Track 1 RL | `t1/rl/official/prepare.sh` builds the SageMaker project from the starter, outside the repo. Reward check `t1.rl.reward-check-20261103T020244Z-4f92` PASS. The full 100k-step run crashed at about 67.6k steps (G-008) | scored on the 50k checkpoint: `t1.rl.eval-20261103T043115Z-bf52` **FAIL**, 5 of 6 gates (UPLIFT fails). |
| Train: Track 1 policy v0 (local, labelled) | `python -m t1.rl.train --out <dir> --minutes 30` | done, not a scored run |
| Train: Track 1 LoRA fine-tune | `python -m t1.sft.stage_base`, `python -m t1.sft.build`, `python -m t1.sft.train_lora --run-id <id> --base ... --data ...` (request files in `t1/sft/jobs/`) | stage `t1.sft.stage-20261103T034500Z-5b1e` PASS, data `t1.sft.data-20261103T034500Z-7c2a` PASS (1,220 examples); training `t1.sft.train-20261103T034500Z-9d4f` PASS its declared target (mean_score 0.005 against <= 0.5) |
| Train: Track 2 | `python -m silos.job_data`, `python -m silos.job_baselines`, `python -m silos.job_evaluate`, `python -m silos.job_compare` (`python -m silos.launch ... --dry-run` prints the request). Local generators: `python -m silos.generate`, `python -m t2.gen`, `python -m t2.train` | simulated silos: data, local-only A/B/C, central and local FedAvg runs are on AWS and PASS their declared checks (simulated silos, G-005). One FL-service session ran: `t2.fl_federated.disjoint-20261103T051311Z`, compared with the local-only models in `t2.compare.federated_vs_C.disjoint-20261103T061346Z` (PASS) and `t2.compare.federated_vs_B.disjoint-20261103T063916Z` (MISSED), judged slot 9 |
| Evaluate: Track 1 | `python -m t1.evals.freeze --verify`; scored runs go through `tools.sm.run` with entry `-m t1.evals.run --unit <interface\|execution\|safety\|knowledge\|price_latency> --client local-hf --model-dir {channel:model} ...` (tuned run adds `--baseline-run-id <base run id>`) | base runs done for all five units: price-latency PASS; knowledge, execution, safety and interface **FAIL** their declared targets. Tuned runs: execution (`t1.execution-otel-tuned-1`, slot 5) and interface (`t1.interface-otel-tuned-1`, slot 6) PASS their gain targets against base; safety and price-latency tuned runs PASS (supporting runs in `judging.json`). Each execution evaluation is pinned to the instance type that froze its items (G-009) |
| Evaluate: replay check and re-freeze | `python -m t1.evals.replay_job check\|refreeze` as SageMaker jobs | runs `t1.execution.replay-check-*` and `t1.execution.refreeze-x86-1` are on AWS; the x86 re-frozen execution items are the ones in the repo (adopted in `482fe5f`; `DATA.md` section 2) |
| Evaluate: trust gate and ladder (CPU jobs) | `python -m t3.trust.job_gate_eval --version v1 --model healthy` (and `degraded`; `--version v2`), `python -m t3.trust.job_ladder --model healthy` (and `degraded`) | on AWS: `t3.trust.gate_v1_degraded-20261103T045624Z` (slot 10: 20 of 20 paused, PASS), `t3.trust.gate_v1_healthy-20261103T045624Z` (edge case E1: 7 of 20 healthy runs falsely paused, MISSED), ladder `t3.trust.ladder_healthy-20261103T045624Z` (PASS). The degraded ladder and gate v2 were not run |
| Evaluate: Track 3 swarm incidents | `python -m t3.evals.incidents --check`; baselines `python -m t3.evals.run --operator no_action` (then `rule_baseline`, `config_revert`, `oracle`, each with `--baseline-run-id`); swarm: `python -m t3.evals.run --operator swarm --mode proposal --subset masked --swarm-client <module:factory> --baseline-run-id <config_revert run>` | baselines `no_action`, `rule_baseline`, `config_revert` (all and masked) are on AWS and PASS. Swarm: the live proposal-only run `t3.swarm-live-20261103T060140Z`, scored by `t3.evals.swarm-masked-20261103T064410Z-x86` (slot 4: masked-tier gain CI low +0.034, PASS). `oracle` was not run on AWS |
| Track 3 units (deny, allow, pause, trace) | `python -m t3.suite.runner --run-id <run_id> --unit all --live` (waits for a person to approve; never approves itself) | done: `t3.deny-20261103T054914Z` (27 of 27 refused, slot 1), `t3.allow-20261103T052448Z` (14 of 14 applied after a person approved, slot 2), `t3.pause-20261103T055232Z` (9 of 9, slot 3); each PASS. The live demo uses the Track 3 kit's own check, which passed 20 of 20 on 3 Nov at 10:50 PST |
| One unit end to end, locally | `tools/run_unit.py --unit <unit> --track track-3 --metric ... --op ... --value ... --test-set <file> --model-id ... --grader ... -- <command>` | built |
| Check | `python tools/check_unit.py <run_id> --s3 s3://<team data bucket>/evidence/` (exit 0 all parts hold, 1 a part failed, 2 could not check) | built. Part e replays the paired-bootstrap gain intervals (G-012, fixed at `6f616c8`) |
| Live test (judges, read-only) | `scripts/live_test.sh <run_id> [--item <item_id>]`: prints where each record lives (S3 key, Logs Insights query, DynamoDB key), then runs `check_unit.py`; stops at 2:30. Lower level: `python tools/show_record.py --run-id <run_id> --item <item_id> --s3 s3://<team data bucket>/evidence/` | built; rehearsed on two real runs, both passing, in 0:29 and 0:30 (no read-only role to rehearse under, G-004) |
| Fill `judging.json` | `python tools/judging_json.py --s3 s3://<team data bucket>/evidence/ --include <run ids> --order-file <file> --dry-run`, then without `--dry-run`; `python tools/judging_json.py --slots-report` shows which of the ten slots and two edge cases are filled | built; 10 of 10 slots and both edge cases filled |
| Tests | `python -m pytest -q scripts/hooks/tests tools/tests t1/evals/tests t3/suite/tests t3/evals/tests` (more under each package); dashboard gate: `scripts/gate.sh` in a worktree | |

## Evidence chain: the four units

How our records map to the judging deck's four units of the special track. Every claim names a run in `judging.json`.

- **Base security monitoring:** every agent tool call crosses the AgentCore Gateway, gets a Cedar decision and leaves a CloudWatch audit record (decision log and spans; the deny run joins all 27 decisions to their spans). The trust gate pauses a degraded model by itself: `t3.trust.gate_v1_degraded-20261103T045624Z`, 20 of 20 paused (slot 10). It also falsely paused 7 of 20 healthy runs (E1).
- **Adversarial testing:** the deny suite `t3.deny-20261103T054914Z` refused 27 of 27, including 4 prompt-injection items (slot 1). The Track 1 safety suite: the tuned model scores 0.851 (`t1.safety-otel-tuned-1`), below the 0.95 bar the base run declared; its own declared target (gain over base) passes.
- **Trusted data and AI evidence chain:** a target declared in S3 before every run, frozen test-set hashes (`judging.json` `data.test_sets`), rule-based graders (no model grades a model), per-item records with AWS times, `check_unit` parts a to e, the signed ledger, `judging.json` and `DATA.md`.
- **SDLC:** the RL kit's gated pipeline, `t1.rl.eval-20261103T043115Z-bf52`: 5 of 6 gates, UPLIFT FAILS (slot 8). The NLM path: data `t1.sft.data-20261103T034500Z-7c2a` (checks PASS) → train `t1.sft.train-20261103T034500Z-9d4f` (PASS) → evaluate `t1.execution-otel-tuned-1` and `t1.interface-otel-tuned-1` (slots 5 and 6, PASS).
- **Not covered:** the deck's "20 simulations, 10+ seeds" bar is the Bridge's, which we did not attempt (G-018). What we have: the RL evaluation runs 20 episodes on 20 seeds (4 scenarios x 5), and the trust gate 20 twin seeds (3000 to 3019).

## Judged runs

`judging.json` lists a curated set, not twenty runs: **ten slots** and **two edge cases**. Slots 1 to 3 are the Track 3 checkpoint units (deny, allow, pause and live grounding), slot 4 the swarm incident evaluation against the `config_revert` bar, slots 5 to 7 Track 1 (execution and interface, tuned against base, and the Bedrock general model), slot 8 the RL scored evaluation, slot 9 the Track 2 federated comparison and slot 10 the trust gate catching a degraded model. The edge cases are shown on purpose: E1 is the trust gate falsely pausing a healthy model (7 of 20, a known limitation) and E2 is why each evaluation is pinned to one instance type (75 of 80 items replay exactly on ml.g5.4xlarge, 80 of 80 on ml.g5.2xlarge). A slot is either filled with run ids that are in `run_ids`, or pending with what it waits for; everything else is under `supporting_run_ids`, grouped by purpose, and `evidence/superseded.json` lists runs that must not be used. `python tools/judging_json.py --slots-report` prints the current state.

## Rebuild (no AWS access)

```
make -C dashboard evidence-rebuild
```

This re-verifies the dashboard's signed evidence ledger (the hash chain, every signature with the committed public key, every Merkle anchor and every file hash) from an evidence mirror, and reprints the per-run summary. It exits 0 when clean, 1 when an artifact changed or is missing, 2 when the chain is broken. Point it at a mirror with `DIR=<mirror>` (`make -C dashboard evidence-sync` makes one, read-only on AWS).

**The signed ledger is kept on the team laptop. It is not in the repo or in S3.** So on a fresh clone the command prints "nothing to rebuild": `evidence/` in the repo holds the declarations, the public key and `superseded.json`, not the ledger. See [G-023](GAPS.md#g-023).

Recomputing each reported result from its items is `check_unit` part e. The live test (`scripts/live_test.sh <run_id>`) runs it with read-only AWS access; to run it alone: `python tools/check_unit.py <run_id> --s3 s3://<team bucket>/evidence/`.

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

## Clean-up

One line per item: what, who, when. "Lead" is the Team13 lead (main session). Times are 3 Nov, PST.

- **Keep until judging ends:** the team data bucket (`evidence/`, `judging/`), CloudWatch logs, the DynamoDB table `team13-approvals`, the AgentCore Gateway, its Cedar policies, and the Lambdas `team13-ran-tools` and `team13-trust-gate`. The judges' live test reads S3, Logs Insights and DynamoDB, read-only. Who: nobody touches them. When: the organisers make the account read-only at 18:00.
- **SageMaker endpoint `team13-otel-nlm`** (ml.g6.xlarge, about $1.13/h), with its endpoint config and model. Who: lead. When: delete right after the demo; the organisers delete endpoints at 18:05 in any case.
- **EC2 `team13-agentbuilder`** (the Agent Builder host where `noc_lead` runs; Ask the NOC depends on it). Who: lead. When: terminate (not stop) after the demo.
- **EC2 `ran-rl-staging-twin`** (the twin host, c5.4xlarge; the staging guide gives its security group the same name; the Viavi adapter container runs on it). Who: lead. When: after the demo, shut it down per Part F of the ran-rl staging guide (close the ports, stop; full teardown terminates it and deletes the security group and key pair).
- **SageMaker notebook `team13-gpu-g5-4xl`.** Held on the organisers' instruction. Who: lead. When: stop only when the organisers say. The three stopped notebooks (`team13-gpu-g6-2xl`, `team13-gpu-g5-2xl`, `team13-gpu-g6-xl`) keep their disks; delete only on the user's say.
- **Laptop processes:** the Viavi relay and adapter (`tools/viavi_bridge`, ports 8790 and 8791) and the dashboard servers. Who: whoever started them. When: stop after the demo.
- **Twin API token.** Who: lead. When: rotate after the demo. Its value is never written anywhere.
