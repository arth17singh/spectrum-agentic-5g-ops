# Trusted Data and AI Evidence Chain (cross-cutting track, every track), team13

Team code `team13`. Hand-in: 12:00 Seattle time, 3 Nov 2026. Primary track: the evidence chain (cross-cutting). It runs through every entry: Track 3 (agent checkpoint and swarm), Track 1 (NLM), Track 2 (one FL-service session). Track 2 ran one session on the FL service (`t2.fl_federated.disjoint-20261103T051311Z`; judged slot 9: silo C PASS, silo B MISSED). The Bridge was not attempted (G-018). What is blocked or not attempted is in [`GAPS.md`](GAPS.md). The machine-readable summary for judges is [`judging.json`](judging.json); every dataset and model output is listed in [`DATA.md`](DATA.md).

**Since the event (4 Nov 2026).** Team13 were finalists. `main` at `1a23aee` is the state at the hand-in (12:00, 3 Nov); the commits after it are this README and three GAPS disclosures (G-025 to G-027). What we built after the hand-in, a live fault-and-fix loop and a remediation policy that passed its gate, is under [After the hand-in](#after-the-hand-in-4-nov); its code is on branches, not on `main`. Since 4 Nov at 18:34 PST the organisers' service control policy refuses every AWS call from the team's role (explicit deny), so we can no longer run the live test or `check_unit` against S3. Byte-exact copies of 13 of the 14 judged results are in [`evidence/results/`](evidence/results/).

## The problem

1. Operators will not let an AI model change a live radio network because a demo looked good. They need evidence a risk owner can check: a target written down before each run, a frozen test, and records that agree with what AWS logged.
2. So we put a small team of agents in front of a network twin. Every tool call crosses one policy checkpoint (Cedar decision, guardrails, an audit record with one trace id), and a person approves every state change.
3. A model is flagged automatically when its predictions stop matching what the network does, and a person decides whether to pull it (automatic pauses are recorded, not enforced: [G-027](GAPS.md#g-027)). Every number we report opened to a record you could read with read-only AWS calls until the account was locked on 4 Nov, and the console's signed evidence ledger re-verifies from the team's evidence mirror (see [Rebuild](#rebuild-no-aws-access)).

## Architecture

Track 3, where every agent action is governed, the trust gate and the evidence chain:

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
                        (deployed with T3_TRUST_LIVE=0: automatic pauses are recorded, not enforced; G-027)

 Evidence: declaration -> SageMaker job / Lambda (run id = job name) -> per-item records -> result -> tools/check_unit.py PASS/FAIL
           -> judging.json -> read-only live test (S3, CloudWatch Logs Insights, DynamoDB) and an offline rebuild
 Dashboard (SPECTRUM, dashboard/): Twin, Ladder, Swarm trace, Scorecard, Evidence explorer
```

Track 1 and Track 2. Every step is a SageMaker job with a run id and goes through the same evidence chain:

```
 Track 1, language model:  OTel-LLM-1.7B-IT -> LoRA fine-tune (t1/sft) -> five frozen evaluation units (t1/evals: interface,
                           execution, safety, knowledge, price-latency; rule-based graders, tuned vs base) -> SageMaker endpoint
 Track 1, RL policy:       organisers' ran-rl SDK -> PPO on the hosted Maveric twin -> ppo_sm_50k -> the kit's gated evaluation;
                           the tools Lambda serves it to the agents as ran___propose
 Track 2, KPI predictor:   three simulated operator silos (silos/) -> local-only, central and FedAvg response models;
                           one FL-service session -> federated model; the tools Lambda serves it as ran___predict,
                           and the trust gate scores its forecasts against what the twin measures
```

Where things live:

| Path | What |
|---|---|
| `contracts/` | the record schemas (declared, item, decision, change, control, job, model, result), one source for Python and TypeScript |
| `twin/` | in-process replay of the Maveric twin and its KPI definitions, with parity checks |
| `t1/evals/` | Track 1 frozen item sets, one prompt, one parser, deterministic graders |
| `t1/sft/` | Track 1 LoRA fine-tune: stage the base model, build the data, train (SageMaker jobs) |
| `t1/rl/` | Track 1 PPO policy (v0, local) and the script that builds the official SageMaker project |
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
| Evaluate: Track 3 swarm incidents | `python -m t3.evals.incidents --check`; baselines `python -m t3.evals.run --operator no_action` (then `rule_baseline`, `config_revert`, `oracle`, each with `--baseline-run-id`); swarm: `python -m t3.evals.run --operator swarm --mode proposal --subset masked --swarm-client <module:factory> --baseline-run-id <config_revert run>` | baselines `no_action`, `rule_baseline`, `config_revert` (all and masked) are on AWS and PASS. Swarm: the live proposal-only run `t3.swarm-live-20261103T060140Z`, scored by `t3.evals.swarm-masked-20261103T064410Z-x86` (slot 4: masked-tier gain CI low +0.034, PASS). `oracle` was not run on AWS. Found after the hand-in: the ticket named the fault type ([G-025](GAPS.md#g-025)), and the change came from the RL policy behind `ran___propose`, so this gain measures the policy's boost under governance, not the agents' diagnosis ([G-026](GAPS.md#g-026)) |
| Track 3 units (deny, allow, pause, trace) | `python -m t3.suite.runner --run-id <run_id> --unit all --live` (waits for a person to approve; never approves itself) | done: `t3.deny-20261103T054914Z` (27 of 27 refused, slot 1), `t3.allow-20261103T052448Z` (14 of 14 applied after a person approved, slot 2), `t3.pause-20261103T055232Z` (9 of 9, slot 3); each PASS. The live demo uses the Track 3 kit's own check, which passed 20 of 20 on 3 Nov at 10:50 PST |
| One unit end to end, locally | `tools/run_unit.py --unit <unit> --track track-3 --metric ... --op ... --value ... --test-set <file> --model-id ... --grader ... -- <command>` | built |
| Check | `python tools/check_unit.py <run_id> --s3 s3://<team data bucket>/evidence/` (exit 0 all parts hold, 1 a part failed, 2 could not check) | built. Part e replays the paired-bootstrap gain intervals (G-012, fixed at `6f616c8`) |
| Live test (judges, read-only) | `scripts/live_test.sh <run_id> [--item <item_id>]`: prints where each record lives (S3 key, Logs Insights query, DynamoDB key), then runs `check_unit.py`; stops at 2:30. Lower level: `python tools/show_record.py --run-id <run_id> --item <item_id> --s3 s3://<team data bucket>/evidence/` | built; rehearsed on two real runs, both passing, in 0:29 and 0:30 (no read-only role to rehearse under, G-004) |
| Fill `judging.json` | `python tools/judging_json.py --s3 s3://<team data bucket>/evidence/ --include <run ids> --order-file <file> --dry-run`, then without `--dry-run`; `python tools/judging_json.py --slots-report` shows which of the ten slots and two edge cases are filled | built; 10 of 10 slots and both edge cases filled |
| Tests | `python -m pytest -q scripts/hooks/tests tools/tests t1/evals/tests t3/suite/tests t3/evals/tests` (more under each package); dashboard gate: `scripts/gate.sh` in a worktree | |

## Evidence chain: the four units

How our records map to the judging deck's four units of the special track. Every claim names a run in `judging.json`.

- **Base security monitoring:** every agent tool call crosses the AgentCore Gateway, gets a Cedar decision and leaves a CloudWatch audit record (decision log and spans; the deny run joins all 27 decisions to their spans). The trust gate decides by itself to pause a degraded model: `t3.trust.gate_v1_degraded-20261103T045624Z`, 20 of 20 paused (slot 10). It also falsely paused 7 of 20 healthy runs (E1). In the deployed system those automatic pauses are recorded, not enforced, and a person decides ([G-027](GAPS.md#g-027)).
- **Adversarial testing:** the deny suite `t3.deny-20261103T054914Z` refused 27 of 27, including 4 prompt-injection items (slot 1). The Track 1 safety suite: the tuned model scores 0.851 (`t1.safety-otel-tuned-1`), below the 0.95 bar the base run declared; its own declared target (gain over base) passes.
- **Trusted data and AI evidence chain:** a target declared in S3 before every run, frozen test-set hashes (`judging.json` `data.test_sets`), rule-based graders (no model grades a model), per-item records with AWS times, `check_unit` parts a to e, the signed ledger, `judging.json` and `DATA.md`.
- **SDLC:** the RL kit's gated pipeline, `t1.rl.eval-20261103T043115Z-bf52`: 5 of 6 gates, UPLIFT FAILS (slot 8). The NLM path: data `t1.sft.data-20261103T034500Z-7c2a` (checks PASS) → train `t1.sft.train-20261103T034500Z-9d4f` (PASS) → evaluate `t1.execution-otel-tuned-1` and `t1.interface-otel-tuned-1` (slots 5 and 6, PASS).
- **Not covered:** the deck's "20 simulations, 10+ seeds" bar is the Bridge's, which we did not attempt (G-018). What we have: the RL evaluation runs 20 episodes on 20 seeds (4 scenarios x 5), and the trust gate 20 twin seeds (3000 to 3019).

## Judged runs

`judging.json` lists a curated set, not twenty runs: **ten slots** and **two edge cases**. Slots 1 to 3 are the Track 3 checkpoint units (deny, allow, pause and live grounding), slot 4 the swarm incident evaluation against the `config_revert` bar (read it with G-025 and G-026), slots 5 to 7 Track 1 (execution and interface, tuned against base, and the Bedrock general model), slot 8 the RL scored evaluation, slot 9 the Track 2 federated comparison and slot 10 the trust gate catching a degraded model. The edge cases are shown on purpose: E1 is the trust gate falsely pausing a healthy model (7 of 20, a known limitation) and E2 is why each evaluation is pinned to one instance type (75 of 80 items replay exactly on ml.g5.4xlarge, 80 of 80 on ml.g5.2xlarge). A slot is either filled with run ids that are in `run_ids`, or pending with what it waits for; everything else is under `supporting_run_ids`, grouped by purpose, and `evidence/superseded.json` lists runs that must not be used. `python tools/judging_json.py --slots-report` prints the current state.

## After the hand-in (4 Nov)

Built after judging, to make the demo's story happen live: inject a fault, let the agents propose a fix, approve it, and watch every cell recover. The code is on branches in this repository, not merged into `main`. The numbers below come from the team's copies of the result records, pulled before the account was locked; they can no longer be re-checked against S3.

| Branch | What |
|---|---|
| `feat/live-fix` | the Live Fix screen (`/live-fix`) and its backend, the per-cell rule `t3/evals/percell.py`, the fault and attack buttons, lessons, the confidence score, and G-025 to G-027 |
| `feat/remediation-policy`, then `feat/remediation-v2` | the remediation policy (`t1/rl/remediation/`), the frozen held-out set `heldout-v2`, the v1 and v2 artifacts and their evaluations |
| `deploy/remediation-v2` | what the tools Lambda ran after the switch: `ran___propose` with the pre-incident baseline, and the runbook with rollback, `t3/tools/image/DEPLOY-remediation-v2.md` |
| `feat/livefix-*`, `feat/lesson-prompts` | the parts merged into `feat/live-fix`, plus two that are not: the confirmation incident set (`feat/livefix-confirm`) and v2 agent prompts (`t3/agents/v2`, not deployed) |

**Live Fix.** Pick a fault in operator words and a cell, inject it on the twin, and the cells go red. The agents diagnose through the AgentCore Gateway as in the judged swarm, Cedar holds the change, and a person approves it on the same screen (PIN) or on a phone (QR code). Before approval, the card shows the side effects predicted on a preflight copy of the twin: a predicted red cell holds the change, a predicted amber cell marks it for a second reviewer (the co-signature itself is not built). After the change, the twin grades every cell against its own healthy state (rule percell-v1, which tells "not restored" apart from "newly degraded by the change"), and "ask the agents to correct" starts another round.

**What it showed us,** recorded in [`GAPS.md`](GAPS.md):

- [G-025](GAPS.md#g-025): the judged swarm ticket named the fault type through the incident id. Re-run with opaque tickets, 16 of 18 incidents recovered, against 13 of 18 for the judged proposals graded the same way; the hint did not help.
- [G-026](GAPS.md#g-026): the swarm's change comes from the PPO policy `ppo_sm_50k` through `ran___propose`, not from the agents' diagnosis. Its biggest change lands on a faulted cell in 5 of 16 incidents.
- [G-027](GAPS.md#g-027): automatic trust-gate pauses are recorded, not enforced (`T3_TRUST_LIVE=0`); a person decides.

**Remediation policy.** Because of G-026 we trained a policy that targets the fault: behaviour cloning, in `ppo_sm_50k`'s network shape, of the twin-verified way back to each cell's commissioned configuration, with DAgger rounds. Each version was scored against `ppo_sm_50k` on a frozen held-out set, `heldout-v2`: 1,008 masked faults (56 strata of 18), sha256 `4f3a79b6…`, frozen and replayed on ml.m5.xlarge. The declared target is joint: three paired-bootstrap lower bounds (95%) must clear their thresholds, for every cell restored (above 0), network recovery (above −0.05, so no worse than PPO) and diagnosis (above 0).

- **v1**, `t3.evals.remediation-policy-20261104T085949Z`: **MISSED** (joint margin −0.179). It restored every cell more often (+0.177 [0.151, 0.202]) but lost network recovery (−0.200 [−0.229, −0.169]). Not deployed. The learning curve `t1.rl.remediation-learning-curve-20261104T090005Z` was flat between 50% and 100% of the training data (+0.003 [−0.024, +0.030]), so v2 changed the inputs, not the amount of data.
- **v2**, `t3.evals.remediation-policy-v2-20261104T113942Z`, model `t1-remediation-bc-delta@v2-c282d94`: **PASS** (joint margin +0.063). Its inputs add each cell's KPI change against the pre-incident baseline; it reads no configuration history. This was the second use of `heldout-v2` (v1 was the first).

| `heldout-v2`, 1,008 faults | v2 | v1 | `ppo_sm_50k` | exact inverse (oracle) |
|---|---|---|---|---|
| Every cell back to healthy (percell-v1) | 45.4% | 19.4% | 1.8% | 100% |
| Network KPIs back within tolerance | 90.4% | 60.1% | 78.8% | 100% |
| Mean network recovery | 0.895 | 0.660 | 0.859 | 1.000 |
| Biggest first change on a faulted cell | 99.9% | 84.0% | 29.0% | 100% |
| Total power change, mean | 3.7 dB | 3.0 dB | 29.0 dB | 3.6 dB |
| Proposes a change on a healthy network | 5.3% | 26.4% | 100% | 0% |
| Faults with a change clamped at a limit | 0 | 0 | 505 | 0 |

v2 against `ppo_sm_50k`, paired over the 1,008 faults (95% intervals): every cell back to healthy +0.437 [0.405, 0.468]; network recovery +0.036 [0.013, 0.059]; diagnosis +0.709 [0.682, 0.737].

**Deployed.** On 4 Nov at about 07:25 PST the tools Lambda `team13-ran-tools` switched `ran___propose` from `ppo_sm_50k` to remediation-v2, and the Gateway smoke test passed. Of the three Live Fix runs after the switch, `t3.livefix-20261104T153437Z-16bb` recovered with one approved change (every cell green, network recovery 0.995, no side effects predicted). `t3.livefix-20261104T153027Z-33d8` and `t3.livefix-20261104T154003Z-1537` ended with no proposal; we could not read why before the account was locked.

**Not proven yet.** The confidence score on the approval card is not calibrated: on 23 recorded proposals, graded offline on the team laptop (no AWS run record), its expected calibration error is 0.167 and its Brier score 0.179, worse than always predicting the base rate (0.113). Self-improvement has no result: lessons reach the lead agent's ticket but not `ran___propose`, so the live unit was stopped after its before phase (G-026).

## Rebuild (no AWS access)

```
make -C dashboard evidence-rebuild
```

This re-verifies the dashboard's signed evidence ledger (the hash chain, every signature with the committed public key, every Merkle anchor and every file hash) from an evidence mirror, and reprints the per-run summary. It exits 0 when clean, 1 when an artifact changed or is missing, 2 when the chain is broken. Point it at a mirror with `DIR=<mirror>` (`make -C dashboard evidence-sync` makes one, read-only on AWS).

**The signed ledger is kept on the team laptop. It is not in the repo or in S3.** So on a fresh clone the command prints "nothing to rebuild": `evidence/` in the repo holds the declarations, the public key and `superseded.json`, not the ledger. See [G-023](GAPS.md#g-023).

Recomputing each reported result from its items is `check_unit` part e. The live test (`scripts/live_test.sh <run_id>`) runs it with read-only AWS access; to run it alone: `python tools/check_unit.py <run_id> --s3 s3://<team bucket>/evidence/`. Since 4 Nov the team's AWS access is refused (see [Clean-up](#clean-up)), so neither `evidence-sync` nor `check_unit --s3` can run against the bucket now.

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

Since 4 Nov at 18:34 PST (5 Nov 02:34 UTC) the organisers' service control policy refuses every AWS call from the team's role (explicit deny on S3, DynamoDB, Lambda, SageMaker and EC2). We can no longer read, stop or delete anything in the account. The last state we saw, before the lock:

- **Kept on purpose:** the team data bucket (`evidence/`, `judging/`), CloudWatch logs, the DynamoDB table `team13-approvals`, the AgentCore Gateway and its Cedar policies, and the Lambdas `team13-ran-tools` (serving remediation-v2 since 4 Nov) and `team13-trust-gate`.
- **SageMaker endpoint `team13-otel-nlm`** (ml.g6.xlarge, about $1.13/h): kept on the team's decision (3 Nov, about 23:00 PST), not deleted.
- **EC2 `team13-agentbuilder`** (the Agent Builder host where `noc_lead` runs) and **`ran-rl-staging-twin`** (the twin host): not shut down by us.
- **SageMaker notebook `team13-gpu-g5-4xl`:** held on the organisers' instruction. The three stopped notebooks (`team13-gpu-g6-2xl`, `team13-gpu-g5-2xl`, `team13-gpu-g6-xl`) keep their disks.
- **Twin API token:** rotate if access returns. Its value is never written anywhere.
