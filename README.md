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

Run from the repository root with Python 3.12. The organisers' `ran-rl` SDK is not redistributed (its terms forbid it), so it is not in this repo; install it from the organisers' package. `<team data bucket>` stands for the team's evidence bucket; `<run_id>` for a run id from `judging.json`. TODO marks a step whose command is not built yet.

| Stage | Command | Status |
|---|---|---|
| Link test (SDK to hosted twin) | from the organisers' starter kit, outside this repo: `make check-aws`, then its SageMaker link job. Recorded in `GAPS.md` (G-001) | done once; no launcher in this repo |
| Train: Track 1 policy (v0, local) | `python -m t1.rl.train --out <dir> --minutes 30` | done (labelled local, v0) |
| Train: Track 1 RL on SageMaker | `t1/rl/official/prepare.sh` builds the SageMaker project from the organisers' starter, outside git. The scored job against the hosted twin, then evaluate, is TODO | prepared, not run |
| Train: Track 1 LoRA fine-tune | TODO (SageMaker training job) | not built |
| Train: Track 2 data and model | `python -m silos.generate --out <dir>`; `python -m t2.gen --out <dir>`; `python -m t2.train --data <dataset.npz> --out <ckpt.pt>` | done locally; federated run blocked (G-005) |
| Evaluate: Track 1 | `python -m t1.evals.freeze --verify`, then `python -m t1.evals.run --unit <interface\|execution\|safety\|knowledge\|price_latency> --client sagemaker --endpoint <name> --model-id <id> --region us-east-2 ...` (tuned run adds `--baseline-run-id <base run id>`) | harness built; real runs go inside a SageMaker GPU job with `--client local-hf` (see `t1/evals/README.md`); the job launcher `tools/sm` that README names is not in the repo yet (TODO); model runs TODO |
| Evaluate: trust gate and ladder (CPU jobs) | `python -m t3.trust.job_gate_eval --version v1 --model healthy` (and `degraded`; `--version v2`), `python -m t3.trust.job_ladder --model healthy` (and `degraded`), run as SageMaker CPU jobs (the job launcher is TODO, see above) | entry points built; AWS runs TODO |
| Evaluate: Track 3 swarm incidents | `python -m t3.evals.incidents --check`; `python -m t3.evals.run --operator no_action` (then `rule_baseline`, `config_revert`, `oracle` with `--baseline-run-id`), swarm: `python -m t3.evals.run --operator swarm --mode proposal --subset masked --swarm-client <module:factory> --baseline-run-id <config_revert run>` | offline baselines done; swarm run TODO |
| Track 3 units (deny, allow, pause, trace) | `python -m t3.suite.runner --run-id <run_id> --unit all --live` (the runner waits for a person to approve; it never approves itself) | TODO live run |
| One unit end to end (declare, run, collect, check) | `tools/run_unit.py --unit <unit> --track track-3 --metric ... --op ... --value ... --test-set <file> --model-id ... --grader ... -- <command>` | built |
| Check | `python tools/check_unit.py <run_id> --s3 s3://<team data bucket>/evidence/` (exit 0 pass, 1 fail, 2 could not check) | built |
| Live test (judges, read-only) | `python tools/show_record.py --run-id <run_id> --item <item_id> --s3 s3://<team data bucket>/evidence/` (S3 Get/List, Logs Insights, DynamoDB Get only). A `--window` option is TODO | built, not yet rehearsed under a read-only role (G-004) |
| Fill `judging.json` | `python tools/judging_json.py --s3 s3://<team data bucket>/evidence/ --dry-run`, then without `--dry-run` | built |
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
