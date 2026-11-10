# Self-improving agent swarm for radio network operations

Invences (team13) at the AWS and Telecom Infra Project Breaking Barriers hackathon, FYUZ 2026, Seattle, where we were finalists.

A small team of AI agents diagnoses faults in a radio network and proposes fixes. Every action passes a policy check and a person's approval, every result is measured on a network twin, and the system learns from the fixes that worked. One system covers all three hackathon tracks: a network language model and reinforcement learning (Track 1), federated learning (Track 2) and governed agents (Track 3).

**All results come from simulated 4-cell radio networks** (the Maveric and Viavi twins). Nothing has run on a live network yet.

## Goals

1. **Safe by construction.** Agents never touch the network directly. Every tool call gets a Cedar policy decision and an audit record, and a person approves every change.
2. **Better with every verified fix.** Fixes the twin confirms become lessons and training data; failed fixes are kept as "do not repeat". A new model is released only if it beats the current one on a frozen test.
3. **Learn across operators without sharing data.** Shared models are trained with federated learning; raw network data stays with each operator.
4. **Every claim checkable.** Each run declares its target before it starts, uses a frozen test set, and leaves records that an independent checker recomputes.

## How it works

```
fault on the network
  -> lead agent, with monitor, optimizer and explainer agents (Amazon Bedrock models)
  -> governed tools, each call through AgentCore Gateway and a Cedar policy check:
       observe | propose a fix (fix policy) | predict its effect (forecast model)
       | preflight on a twin copy | ask the language model
  -> a person approves, on screen or by phone
  -> the change is applied, and the twin grades every cell against its healthy state
  -> the trust gate compares the forecast with what happened and pauses a model that drifts
  -> verified fixes feed the learning loops below
```

## How the swarm improves itself

We split the work: the agents coordinate, explain and ask for approval, while trained models do the numeric work. Each part learns on its own loop and is tested on its own frozen set before release. This is the design:

| Loop | How often | What learns | From what | Release check |
| --- | --- | --- | --- | --- |
| Lessons | every fix | a lesson memory the agents read | fixes the twin verified; failed fixes stored as "do not repeat" | learns from practice incidents only, never the test set |
| Fix policy | each retrain | the model behind `ran___propose` | imitation of verified fixes (behaviour cloning with DAgger); reinforcement learning explores new fixes inside the twin | must beat the current policy on 1,008 frozen held-out faults |
| Forecast | each federated round | the model behind `ran___predict` | each operator's data, trained with federated learning; no raw data shared | no worse than each operator's own model |
| Language model | each fine-tune | the network language model behind `nlm___ask` | telecom tasks (LoRA fine-tune) | five frozen test units |

What has run so far: the fix-policy loop once, end to end and by hand (v1 missed its gate and was not deployed; v2 passed and was deployed); one 20-round federated session on simulated operators; one language-model fine-tune. The lesson loop is built but does not yet reach the fix (see below).

## What we built

| Part | What it is | Result (simulated networks) |
| --- | --- | --- |
| Network language model (Track 1) | OTel-LLM-1.7B-IT, LoRA fine-tuned on SageMaker and served on an endpoint | 0.62 on 63 network-fixing tasks, against 0.23 (gpt-oss-20b) and 0.05 (Qwen3-32B) on Bedrock. Safety rose from 0.10 to 0.85; our bar is 0.95 |
| Reinforcement learning (Track 1) | PPO trained with the organisers' RAN RL kit | passed 5 of the kit's 6 release gates (uplift failed) |
| Fix policy v2 (after the event) | imitation of twin-verified fixes | on 1,008 held-out faults: every cell restored 45% (PPO 2%), network KPIs back 90% (PPO 79%), right cell first 99.9% (PPO 29%). Beat PPO on every measure; deployed |
| Federated forecast (Track 2) | three simulated operators; a forecast model trained per operator, pooled and federated | federated error 0.65, against 0.78 and 0.67 for two operators' own models; a pooled model was 0.08 better |
| Governed agents (Track 3) | four agents, six tools, Cedar policies, human approval | refused 27 of 27 forbidden calls (4 were prompt injections); applied 14 of 14 changes, each after a person approved; recovered 16 of 18 held-out faults |
| Trust gate (Track 3) | compares forecast with measurement; pauses a drifting model | paused a degraded model 20 of 20 times; paused a healthy one 7 of 20 times |
| Live Fix (after the event) | inject a fault; agents propose; Cedar holds; approve by PIN or phone; every cell graded | recovered faults end to end, live |
| Evidence chain (all tracks) | declare, run as a SageMaker job, record, recompute | backs the 14 results we submitted for judging |

## What we learned

- **The fix came from the policy, not the agents.** In the judged run the agents passed through the RL policy's change, and the ticket hinted at the fault type. With the hint hidden, 16 of 18 faults still recovered. So we trained the fix policy to target the faulted cell (v2).
- **Self-improvement is not shown yet.** Lessons reached the lead agent but not the policy that picks the fix, so our learning test stopped after its baseline. Lesson-aware prompts (`t3/agents/v2`) are written but not deployed.
- **The trust gate is too jumpy.** It paused a healthy model in 7 of 20 runs, and in the deployed system its pauses were recorded, not enforced.
- **Federation needs real data.** It clearly beat one simulated operator's own model but not the other's, and a pooled model still did better.

Every known gap is in [GAPS.md](GAPS.md).

## Next: a 90-day operator pilot

We are looking for two tier-one mobile operators to run the swarm on one cluster each for 90 days: shadow mode first, then changes an engineer approves. The pilot measures whether a swarm learning across both operators beats stand-alone agents trained on one operator's data.

## Repository map

| Path | What |
| --- | --- |
| `t3/agents/` | agent prompts and output schemas (`v2/`: lesson-aware, not deployed) |
| `t3/tools/` | the governed tools Lambda behind AgentCore Gateway |
| `t3/policies/` | Cedar rules |
| `t3/trust/` | trust gate |
| `t3/evals/` | incident suites, graders, and the Live Fix units: faults, lessons, self-improvement, confidence |
| `t3/suite/` | the frozen call suite: deny, allow, pause, trace |
| `t1/sft/`, `t1/evals/` | language model fine-tune and serving, and its five test units |
| `t1/rl/` | PPO policy and the fix policy (`remediation/`) |
| `t2/`, `silos/` | forecast model, simulated operators and federated runs |
| `twin/` | in-process Maveric twin with parity checks |
| `contracts/`, `tools/` | evidence record schemas, the SageMaker runner and the checker |
| `dashboard/` | operator console, including Live Fix (`/live-fix`) |

## Run it

```
python -m pytest -q t3/evals/tests    # each package has its own tests/ folder; run them one at a time
cd dashboard && make demo             # console on port 8787
make -C dashboard evidence-rebuild    # re-verify the signed evidence ledger
```

The twin needs the organisers' `ran-rl` SDK, which we may not redistribute; install it from their package.

## More

- [HANDIN.md](HANDIN.md): the full hackathon hand-in: commands per stage, judged runs, evidence chain, clean-up.
- [GAPS.md](GAPS.md): known gaps and limitations.
- [DATA.md](DATA.md): datasets and model outputs.
- [judging.json](judging.json): judged results, machine-readable.

Since 4 Nov 2026 the organisers have locked the team's AWS account, so results can no longer be re-checked against AWS from our side.
