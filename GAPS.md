# GAPS

What did not work or is blocked, which unit it limits, and what we do instead. Platform blocks also give the time, the service, the exact error, the request id and who we told (Participant Guide p.34). A gap we declare is better than one the judges find (p.19).

Times are Seattle time (PST, UTC−8) with UTC in brackets. Each entry keeps a dated log, newest at the bottom.

## Summary

| ID | Status | Gap | Limits | Opened |
|---|---|---|---|---|
| [G-001](#g-001) | Resolved · team-hosted twin | Hosted twin access file not received; per the organisers, each team hosts its own twin host. Ours runs, and the 2,048-step link job completed against it | Viavi (Part E) still needs our RSG's MCP address and an allow for our host | 2 Nov 11:01 PST |
| [G-002](#g-002) | Open · deviation | SDK bundle location not available; starter kit used as a stand-in | No unit directly; provenance of our SDK copy | 1 Nov, about 23:00 PST |
| [G-003](#g-003) | Declared limit | Agent Builder agents call their Bedrock model directly, not through the Gateway | Track 3: model calls are recorded, not Cedar-decided | 2 Nov 12:31 PST |
| [G-004](#g-004) | Open · platform | No read-only role to rehearse the judges' live test; permission simulation is denied | Live-test rehearsal (evidence chain) | 2 Nov 12:40 PST |
| [G-005](#g-005) | Resolved for the first session · platform | Track 2 FL service was not provisioned at the 16:00 enablement; a federated run (`t2.fl_federated.disjoint-20261103T051311Z`) has since run, comparison runs pending | Track 2 units until the comparisons exist | 2 Nov 16:40 PST |
| [G-006](#g-006) | Declared risk · organiser-side IAM | The Agent Builder host role can create and modify any role under /bb26/; participant SSO role can invoke our tools Lambda directly | Track 3 claim "only the Gateway invokes our tools" holds for non-human principals only | 2 Nov 18:20 PST |
| [G-007](#g-007) | Declared limit · SDK logging | The RAN RL SDK logs every twin HTTP call at INFO with the twin host (never the token) | Evidence hygiene: SDK job logs name the twin host | 3 Nov (time TODO) |
| [G-008](#g-008) | Open · failed run | Full RL training run crashed at about 67.6k of 100k steps (connection reset by the twin host); the 50k checkpoint was scored instead and failed UPLIFT | Track 1 RL unit: scored on the 50k checkpoint, 5 of 6 gates | 2 Nov 20:19 PST |
| [G-009](#g-009) | Open · measurement | Frozen eval references drift across CPU architecture, torch build and instance size | Track 1 execution and Track 3 incident units: frozen values are valid only on the instance type that froze them | 2 Nov 19:14 PST |
| [G-010](#g-010) | Declared risk · organiser-side IAM | The organisers' SageMaker role has no Bedrock access; Bedrock baselines run under a team role. SageMaker roles can read any secret named AmazonSageMaker-* | Track 1 Bedrock baselines; secret exposure | 3 Nov (time TODO) |
| [G-011](#g-011) | Open · feature gap | The dashboard's phone approve page drives the simulator, not the real approvals table | Track 3 human-approval demo until the real page (W7g) is built | 3 Nov (time TODO) |
| [G-012](#g-012) | Resolved · found and fixed 2 Nov | `tools/check_unit.py` could not recompute the paired-bootstrap gain metric, so part [e] said "cannot check" (exit 2) for the Track 3 gain runs | Was: Track 3 baselines and the swarm incident eval (slot 5) | 3 Nov (found during the live-test rehearsal) |
| [G-013](#g-013) | Resolved · harness defect | The Track 1 knowledge parser rejected bare-letter answers (90% parse failures); the first base knowledge run is superseded | Slot 6 (knowledge, base); the other four base runs stand | 2 Nov 20:43 PST |
| [G-014](#g-014) | Resolved · process incident | A stream-branch push contained a real host address in a test literal; the scan reported it but the pipe hid the exit code. Branch deleted within minutes, history rewritten (`008e4bc`) | Confidentiality of the twin host; pushes now scan-gated | 2 Nov (time TODO) |
| [G-015](#g-015) | Resolved · crashed run | The first t3 deny run `t3.deny-20261103T051458Z` crashed at item 23 (the Cedar validator rejected the pause-forbid guard form); fixed in `51010f8` | Slot 1 (t3 deny): the crashed run is superseded by `t3.deny-20261103T054914Z` | 2 Nov about 21:15 PST |
| [G-016](#g-016) | Declared design | Change records carry the approver's AWS identity (`approved_by`) as audit evidence; the dashboard shows a pseudonym | Track 3 audit: identity is in the evidence store, not on screen | 3 Nov (time TODO) |
| [G-017](#g-017) | Resolved · deploy script defect | `t3/deploy/register.py` read only the first page of `list_policies`; with 24 rules it tried to create a rule that already existed, exited 1 without a message, and never created the three seed rules | Track 3 checkpoint: the seed rules were missing after the `614fc3b` deploy | 3 Nov (time TODO) |
| [G-018](#g-018) | Deferred · team decision | Bridge challenge deferred until after the end-to-end pass; no loop code or setup steps, and two different bars | The optional Bridge unit; no Bridge number exists or is claimed yet | 2 Nov 18:00 PST |
| [G-019](#g-019) | Resolved · runner defect | The runner sent an empty `InputDataConfig` for jobs with no input channels; AWS never saw the request, so 8 declared runs (4 T2 compares, 4 Bedrock CPU runs) never started | The T2 comparisons and the Bedrock baselines until relaunched | 3 Nov (time TODO) |
| [G-020](#g-020) | Resolved · harness defects | Two harness defects found by the first comparison and Bedrock runs: a missed target made the T2 compare job exit 1 (result never attested), and a non-finite model answer was stored as `inf` and crashed a Bedrock knowledge run | T2 compare federated_vs_B and Bedrock knowledge (gpt-oss-20b): both relaunched | 3 Nov (time TODO) |
| [G-021](#g-021) | Resolved · checker false positive | `check_unit` part d (ledger comparison) reported a false positive that depended on the order in which the ledger ingested files; fixed in `06ddb9f` | Part d on runs checked against a ledger | 3 Nov (time TODO) |

### Watching

These become entries if still missing at the time shown (2 Nov, PST).

| Due | Waiting for | Last check | Planned fallback |
|---|---|---|---|
| 14:00 | Viavi adapter address and token for team13 | **Partly resolved (~16:40 PST).** Our RSG's MCP server is at `/mcp` on the Viavi UI address (port 443). It sits behind the UI's Cognito sign-in and answers only with a signed-in web session. From the user's own signed-in session we ran read-only calls (handshake, tool list, docs index, scenario and run lists): 21 tools; `ran-rl-maveric-4cell` is not saved yet; no run is active. Still missing: a non-interactive credential for jobs, agents and the twin host's adapter (asked the organisers). | Until then, Viavi runs need the user's signed-in session; unattended jobs can't reach it |
| 16:00 | Track 2 federated-learning service enabled for team13 | 1 Nov 21:00: no Track 2 silo buckets and no stacks in us-west-2 | local federated averaging over the same silos, labelled "not the service" |
| 18:00 | Bridge challenge loop code, and which bar counts (0.73 in the guide, 0.97 in the judging deck) | **Closed.** Neither the loop nor the bar was confirmed by 18:00. The Bridge is deferred, not dropped: see [G-018](#g-018). | Bridge not attempted for now; to be reconsidered after the end-to-end pass |
| first agent run | Claude and OpenAI partner models on Bedrock: Marketplace agreement for the account | 11:55 read-only check: agreement "not available", authorization "authorized"; open-weight models available | agent brains on open-weight models (gpt-oss-120b, available now); entry opened only if the first call is refused for permissions |

---

<a id="g-001"></a>
## G-001 · Hosted twin access file not received

- **Status:** resolved 2 Nov ~15:56 PST (23:56 UTC) for Maveric; Viavi (Part E) still open. The organisers confirmed (to the user, 2 Nov ~15:20 PST) that the twin hosts run in each team's own account, so we stand up our own twin host on EC2 by following the RAN RL Staging Guide 1.2.9 (Parts A to E), and generate our own adapter tokens and `.env`.
- **Service:** RAN RL hosted digital twin for team13 (Maveric adapter).
- **What is missing:** team13's twin access file, which the kit expects at `ran-rl-starter/.env`. Two of its seven keys can only come from the organisers: `RAN_RL_TWIN_URL` and `RAN_RL_ADAPTER_TOKEN`. Viavi access adds `RAN_RL_VIAVI_TWIN_URL` and `RAN_RL_VIAVI_ADAPTER_TOKEN` (tracked under Watching).
- **Time:** our own deadline for the file was 10:00 PST (18:00 UTC). It was still missing at 11:01 PST (19:01 UTC).
- **Exact error:** none. No call to the twin can be made without its address and token.
- **Request id:** none, for the same reason.
- **Who we told:** not recorded yet.
- **What we checked:**
  - our machine, by file name and contents, at 11:01 PST: only the kit's blank `.env.example`;
  - both team S3 buckets at 11:06 PST: us-east-2 holds only the starter kit we uploaded, and us-east-1 is empty.
- **Limits:**
  - Track 1 done-means for RL: a SageMaker training job Completed against the hosted twin, then evaluation (p.24). The hosted twin grades every Track 1 RL result (p.23).
  - The parity check between the hosted twin and the SDK's in-process copy, so data we generate in-process is not yet parity-checked.
  - Every run on the hosted twin, including Track 3 runs there.
- **Meanwhile:** we build against the in-process Maveric copy that ships in the SDK package, which needs no token: the data generator, the evidence records and the agent tools. Every record names its backend, and nothing from the in-process copy is reported as a hosted-twin result.
- **Resolved when:** the file is saved with mode 600 and `make check-aws` ends with "all checks passed". Then come the parity check and the 2,048-step link job.

**Log**
- 2 Nov 10:00 PST (18:00 UTC): our deadline for the file passed.
- 2 Nov 11:01 PST (19:01 UTC): confirmed missing on our machine.
- 2 Nov 11:06 PST (19:06 UTC): confirmed absent from both team buckets.
- 2 Nov ~15:20 PST (23:20 UTC): organisers confirmed the twin host is team-hosted in our own account. Deploying it per the Staging Guide; Part E (Viavi) waits for our RSG's MCP address and for the organisers to allow our host's address.
- 2 Nov 15:38 PST (23:38 UTC): our twin host started (EC2 c5.4xlarge, default VPC). SSH is open from one address; adapter port 8001 is open to everyone and protected by a bearer token, as the guide requires for SageMaker.
- 2 Nov ~15:45 PST: `make check-aws` printed "all checks passed". The team code is set to team13.
- 2 Nov 15:46–15:55 PST (23:46–23:55 UTC): the link job `ran-rl-team13-20261102-234628` Completed against our hosted twin (2,048 `/action` calls served). The declaration was written to S3 7 s before the job. `make evaluate` wrote the report. Run id `t1.rl.link-test-20261102T234623Z-7368`.
- Note: the starter's `make evaluate` uses the SDK's in-process local twin (`backend: local` in `config/env.yaml`), not the hosted Maveric twin. Only training used the hosted twin. The scored run will point evaluation at the hosted twin and run it as a SageMaker job.

<a id="g-002"></a>
## G-002 · SDK bundle location not available; starter kit used as a stand-in

- **Status:** open, a deviation from Foundation step 5.
- **Service:** RAN RL SDK distribution.
- **What happened:** Foundation step 5 takes `RAN_RL_BUNDLE_URI` from the event details page, and we have not been given that value. On the hackathon CDN, the staging bundle, its `SHA256SUMS` file and the wheel paths returned HTTP 403. The starter kit zip (SDK 1.2.7) published on the hackathon hub is byte-identical to the copy we were given: 1,501,816 bytes, SHA-256 `fd3fa97d629fc3b18d3142028f122398681bf75dcbd734bd98d35afa95aa40fc`. We uploaded that copy to our us-east-2 bucket and pointed `RAN_RL_BUNDLE_URI` at it, marked as a stand-in.
- **Exact error:** HTTP 403 on the bundle, `SHA256SUMS` and wheel paths.
- **Request id:** not recorded.
- **Who we told:** not recorded yet.
- **Limits:** no unit directly. We could check only that the hub's copy and ours match, not verify either against an organiser checksum file, so the guide's checksum step (p.21, Part A) was skipped.
- **Resolved when:** the organisers send the official URI, or confirm the starter zip is the intended path; we switch to it and record its checksum.

**Log**
- 1 Nov, about 23:00 PST (07:00 UTC 2 Nov): 403s seen; the hub's copy matched ours.
- 2 Nov 07:26 UTC (1 Nov 23:26 PST): stand-in uploaded to the us-east-2 bucket.
- 2 Nov 09:57 PST (17:57 UTC): the hub published SDK 1.2.9 (SHA-256 `8925cb95994e473e738e9ac72acd7fc9e9e9a9b93ffa0ca54189158297f7d988`) and a Staging Guide that downloads the starter zip directly, with no staging bundle and no `SHA256SUMS`. The Participant Guide (unchanged) still describes the bundle and its checksum (p.21, Part A), so the two guides now differ. The `ran_rl` package code is identical in 1.2.7 and 1.2.9; 1.2.9 adds the twin-host source. Our stand-in is still 1.2.7 and is to move to 1.2.9.

<a id="g-003"></a>
## G-003 · Agent Builder model calls do not cross the Gateway (declared limit)

- **Status:** declared limit, by design of the organisers' governed-team kit, which we adopted for Track 3.
- **What:** the Tupl Agent Builder calls each agent's Bedrock model directly, not through the AgentCore Gateway, so no Cedar rule decides an agent's own model calls. The kit's guide tells teams to declare this limit.
- **What still holds:** every tool call, every call to our NLM and every agent-to-agent call crosses the Gateway, where Cedar decides it, guardrail scores sit inside the rules, and an audit record is written. Every Bedrock call is recorded in the account's invocation log (`/team13/bedrock-invocations`) and can be found by case id when the trace is rebuilt.
- **Limits:** the Track 3 claim "every model call crosses the checkpoint" holds for our NLM tool, not for the agents' own reasoning calls.

**Log**
- 2 Nov 12:31 PST (20:31 UTC): checkpoint deployed; the limit applies from here on.

<a id="g-004"></a>
## G-004 · No read-only role for rehearsing the live test; permission simulation denied

- **Status:** open.
- **Service:** IAM in the team account.
- **What did not work:** `iam:SimulatePrincipalPolicy` is explicitly denied for the participant role (checked read-only during the feasibility review), so permissions cannot be verified in advance. No read-only role exists in the account for rehearsing the judges' read-only live test.
- **Exact error:** AccessDenied on `iam:SimulatePrincipalPolicy` (explicit deny).
- **Request id:** not recorded.
- **Who we told:** not recorded yet.
- **Limits:** the evidence-chain rehearsal. We can show that `tools/show_record.py` uses only read calls (S3 Get/List, Logs Insights, DynamoDB Get), but not prove it under a read-only identity.
- **Meanwhile:** the script's code is limited to read APIs, and a test asserts it never calls a write API.

**Log**
- 2 Nov ~12:40 PST: explicit deny seen.

<a id="g-005"></a>
## G-005 · Track 2 FL service not provisioned at the scheduled enablement

- **Status:** the account was allow-listed and the FL service has since run once (see the log); the comparison runs against local-only and central are pending. Allow-listing check:  a read-only `list-sessions` in us-west-2 at about 18:10 PST returned an empty session list with no access error. The CDK bootstrap in us-west-2 is pending.
- **Service:** the event's federated-learning service (us-west-2).
- **What did not work:** at the 16:00 PST enablement there were no `-usw2` silo buckets (checked 16:40 and again at about 18:30), no install command and no allow-list confirmation.
- **Meanwhile:**
  - the utility package named in the service's guide is readable from our account. It was downloaded, verified against its published SHA-512 sums, and installed outside the repo;
  - our synthetic silo datasets (generated by `silos/`, hashed in `silos/manifest.json`) stand in for organiser silos.
- **Risk still open:** the CDK bootstrap and the utility's own stacks create IAM roles. Our account requires the `/bb26/` path and the permissions boundary, so both may be refused.
- **Who we told:** TODO (user).

**Log**
- 2 Nov about 18:10 PST: read-only `list-sessions` in us-west-2 returned an empty list with no access error.
- 2 Nov about 21:13 PST (3 Nov 05:13 UTC): the run `t2.fl_federated.disjoint-20261103T051311Z` was declared and its result is on S3 (10,000 evaluation items). The comparison against the local-only and central runs is pending (judging slot 9).

<a id="g-006"></a>
## G-006 · Organiser-side IAM breadth (declared risk)

- **Status:** declared; not ours to fix.
- **What we found** (read-only IAM review, 2 Nov about 18:20 PST):
  1. The Agent Builder host's instance role can create roles, put or attach policies and pass roles anywhere under `/bb26/`. Our tools role, trust-gate role and the checkpoint's Gateway role all live under `/bb26/`, so a compromised builder host could rewrite them.
  2. The participant SSO role (PowerUserAccess) can invoke our tools Lambda directly with a forged Gateway context, and can write approval rows. That is also how real human approvals are made.
- **What still holds:**
  - no agent-side role can invoke our Lambda or write the approvals table;
  - agents hold only Gateway tokens;
  - the tools Lambda checks the Gateway id as defence in depth;
  - the trace rebuild fails any tool call without a Gateway decision.
- **Limits:** the claim "only the Gateway invokes our tools" holds for non-human principals only.
- **Who we told:** TODO (user).


<a id="g-007"></a>
## G-007 · SDK logs name the twin host (declared limit)

- **Status:** declared limit; mitigated in our own jobs.
- **Service:** the RAN RL SDK 1.2.9 (HTTP client to the twin host).
- **What happens:** the SDK logs every twin HTTP call at INFO level, with the twin host in the line. It never logs the token. SageMaker job logs and any copy of them therefore contain the host address.
- **Our mitigation:** our own evaluation job lowers those loggers before it makes any call, so its logs and the records it writes do not carry the host. Declarations say "address and token not recorded". Raw job logs are never mirrored into the repo.
- **Limits:** other jobs that run the SDK's own code may still write the host into their CloudWatch logs in the team account (TODO: confirm per job). Those logs are not part of the hand-in.
- **Who we told:** TODO (user).

**Log**
- 3 Nov (time TODO): logged. Our evaluation job lowers the SDK's HTTP loggers.

<a id="g-008"></a>
## G-008 · Full RL training run crashed; the 50k checkpoint was scored instead

- **Status:** open; a documented failure with a run id.
- **Service:** SageMaker training job against the team-hosted twin (HTTP backend).
- **Run:** `t1.rl.train-20261103T023805Z-a09e` (declared 3 Nov 02:38 UTC = 2 Nov 18:38 PST, target: all six gates pass after 100k steps).
- **What happened:** at 20:19 PST (04:19 UTC) the job crashed at about 67.6k of 100k steps.
- **Exact error:** `httpx.ReadError: Connection reset by peer`, raised from a call to the twin host. The SDK's HTTP client does not retry it. The twin's `/health` was OK afterwards, so the host itself was up.
- **Request id:** not recorded.
- **Who we told:** not applicable (our own host); TODO (user) if the organisers should know about the SDK's missing retry.
- **What we did instead:** the 50k checkpoint was scored. The run `t1.rl.eval-20261103T043115Z-bf52` was declared before evaluation (3 Nov 04:31:38 UTC = 20:31 PST; model id `ppo-cco-reward-v1-ckpt50k`). **Result: 5 of 6 gates pass; UPLIFT fails** (the unit's target needs all six). It is reported as a failure.
- **Limits:** Track 1 RL done-means. We have a Completed link job (`t1.rl.link-test-20261102T234623Z-7368`) and a Completed reward check (`t1.rl.reward-check-20261103T020244Z-4f92`), but no Completed full training job that passes all gates.
- **Resolved when:** a rerun completes, or we decide to keep the failure as final.

**Log**
- 2 Nov 18:38 PST (3 Nov 02:38 UTC): the full run was declared and started.
- 2 Nov 20:19 PST (3 Nov 04:19 UTC): crash.
- 2 Nov 20:31 PST (3 Nov 04:31 UTC): the 50k checkpoint's evaluation was declared; result written about 20:36 PST.

<a id="g-009"></a>
## G-009 · Frozen references drift across platforms and instance sizes

- **Status:** open; measured, with a fix in progress.
- **What we found:** the references frozen in our eval sets (the no-change KPI and the rule, best-of-K and fixed-answer gains) are floating-point results of the twin replay, and the replay is not bit-identical everywhere:
  - an arm64 Mac against x86 Linux;
  - the CPU build of torch against the CUDA build;
  - ml.g5.2xlarge (8 threads) against ml.g5.4xlarge (16 threads).
  Users sitting exactly on a coverage threshold flip; about one user's worth of coverage (0.005 to 0.017) on up to 5 or 6 items out of 80.
- **Replay-check runs** (each checks every item against the frozen values to 1e-9; `pass_rate` is the share of items that match):
  - `t1.execution.replay-check-x86-1`, ml.m5.xlarge (x86, CPU torch): 0.9375, 5 items drifted, largest difference 0.0167;
  - `t1.execution.refreeze-x86-1`, ml.g5.2xlarge: rebuilt every item on AWS and wrote new hashes (execution and price-latency items changed; see `DATA.md`);
  - `t1.execution.replay-check-g5-1`, ml.g5.4xlarge (16 threads, CUDA torch): 0.9375, 5 items drifted, largest difference 0.005;
  - `t1.execution.replay-check-g5-2xl-1`, ml.g5.2xlarge (8 threads, CUDA torch): 1.0, no drift.
  The Track 3 incident suite had the same cause and was re-frozen on AWS x86 as suite v4 (`t3.evals.suite-freeze-20261103T031439Z-32b3`; `DATA.md`).
- **Fix:** re-freeze on AWS, and pin each evaluation to the instance type that froze it: execution evals run on ml.g5.2xlarge (the base execution run is declared on it), Track 3 evals on ml.m5.xlarge. A run on another instance type is not comparable and is not reported as such.
- **Limits:** base and tuned runs of a unit must use the same instance type. Results from runs pinned to different instance types are not compared.
- **Who we told:** TODO (user).

**Log**
- 2 Nov 19:14 PST (3 Nov 03:14 UTC): first replay check on x86 (m5.xlarge) shows drift.
- 2 Nov 19:17 PST: suite v4 frozen on AWS x86 (3 Nov 03:17:39 UTC).
- 2 Nov 19:26 PST (03:26 UTC): execution items rebuilt on AWS (refreeze run).
- 2 Nov 19:49 PST (03:49 UTC): g5.4xlarge replay check drifts; 19:57 PST (03:57 UTC): g5.2xlarge replay check passes.

<a id="g-010"></a>
## G-010 · The organisers' SageMaker role has no Bedrock access (declared risk)

- **Status:** declared; workaround in place.
- **Service:** IAM and Bedrock, team account.
- **What did not work:** the organisers' SageMaker execution role (`bb26-SageMakerExecution`) cannot call Bedrock, so Bedrock baselines for Track 1 cannot run under it.
- **What we did:** a team role `/bb26/team13-sm-bedrock-eval`, created in CloudShell under the `/bb26/` path with the permissions boundary, runs the Bedrock baseline jobs.
- **Related risk:** the SageMaker roles can read any Secrets Manager secret whose name starts with `AmazonSageMaker-`. Any secret we name that way is readable by every job role of that kind. We pass a job a Secrets Manager ARN, not the secret, in its `Environment`. TODO (user): confirm which roles this covers.
- **Who we told:** TODO (user).

**Log**
- 3 Nov (time TODO): role created in CloudShell.

<a id="g-011"></a>
## G-011 · The dashboard's phone approve page drives the simulator

- **Status:** open; a real approvals page (W7g) is being built.
- **What:** the dashboard's phone page (`/m/approve`) approves items in the dashboard's own simulator. It does not write to the real approvals table the checkpoint reads, so approving there does not approve a real change.
- **Limits:** Track 3 "human approval before any state change". Until the real page exists, real approvals are made with the kit's approve command, and the demo must not use the simulator page for them.
- **Who we told:** not applicable (our own code).

**Log**
- 3 Nov (time TODO): gap logged; W7g started.

<a id="g-012"></a>
## G-012 · The independent checker could not recompute the gain metric (found and fixed)

- **Status:** resolved. Found and fixed on the same evening.
- **What was found:** `tools/check_unit.py` is deliberately separate from the code that writes results and recomputes a result's value from the item records. At int/v4 `175860e` it had no rule for the paired-bootstrap gain metric (`gain_ci_low_vs_no_action`): part [e] printed `cannot check: unknown metric` and the run exited 2. This was seen on `t3.evals.config_revert-masked-20261103T035108Z-x86` while rehearsing `scripts/live_test.sh`. Parts a (declared before), b (code in the repo) and c (records agree with AWS times) passed.
- **Fix:** commit `6f616c8` (the runner and `check_unit.py`, merged in integration batch 7) makes part [e] replay the `paired-bootstrap-v1` interval exactly (same seed, resamples and pairing).
- **Verified:** after that merge, the live-test rehearsal on `t3.evals.config_revert-masked-20261103T035108Z-x86` ends with all of parts a, b, c and e passing (d skipped, no ledger on this machine). The other two re-run gain baselines (`...T035108Z-x86`) pass part e too, per the lead.
- **Who we told:** not applicable (our own code).

**Log**
- 3 Nov (time TODO): found at `175860e` during the rehearsal.
- 3 Nov (time TODO): fixed in `6f616c8`; rehearsal repeated, exit 0.

<a id="g-013"></a>
## G-013 · The knowledge parser rejected bare-letter answers (harness defect, fixed)

- **Status:** resolved; the affected run is superseded.
- **What happened:** the first base knowledge run `t1.knowledge-otel-base-1` (2 Nov 20:43 PST) scored 0.025 against a declared target of 0.70. The harness's JSON-only parser rejected answers given as a bare letter, so about 90% of the answers counted as parse failures (parse_fail 0.90). That is a defect of our harness, not model behaviour.
- **Fix:** commit `c4b4b5e` (W12) changes the knowledge parser narrowly and versions it for the knowledge unit only. The re-run is `t1.knowledge-otel-base-2`, which is declared as its own run and is running from `c4b4b5e` (result TODO).
- **What stands:** the other four base runs (safety, interface, execution, price-latency) were not changed. W12's diagnosis is that their low scores are model behaviour, not parsing. Their parser and prompt are unchanged.
- **Why we keep the first run:** it stays in S3 and in `evidence/superseded.json`, listed with this reason, so the change of parser is visible. It is not listed in `judging.json`.
- **Who we told:** not applicable (our own code).

**Log**
- 2 Nov 20:43 PST (3 Nov 04:43 UTC): `t1.knowledge-otel-base-1` result written.
- 2 Nov about 20:57 PST: `t1.knowledge-otel-base-2` declared.

<a id="g-014"></a>
## G-014 · A real host address was pushed in a test literal (resolved)

- **Status:** resolved; process fixed.
- **What happened:** on 2 Nov a push of a stream branch contained a real host address inside a literal in a test file. The pre-push scan had reported it, but the command was run through a pipe, which hid the scan's exit code, so the push went ahead.
- **Response:** the remote branch was deleted within minutes and the history was rewritten (`008e4bc`) so no commit we push contains the value. Every push now goes through a script that runs the scan first and stops on its exit code.
- **Limits:** the address was on the remote for a few minutes, until the branch was deleted. TODO (user): decide whether the address should be treated as exposed and changed.
- **Who we told:** TODO (user).

**Log**
- 2 Nov (time TODO): push, scan report, remote branch deleted, history rewritten in `008e4bc`.

<a id="g-015"></a>
## G-015 · The first t3 deny run crashed (resolved)

- **Status:** resolved; the crashed run is superseded.
- **Run:** `t3.deny-20261103T051458Z`, declared 3 Nov 05:14 UTC (2 Nov about 21:15 PST).
- **What happened:** the run crashed at item 23. The Cedar validator rejected the pause-forbid guard form used for the pause-window items.
- **Fix:** commit `51010f8` changes the guard form. The fresh deny run is `t3.deny-20261103T054914Z`: 27 of 27 items refused, each Cedar decision joined to its span (27 spans), `check_unit` parts a, b, c and e pass (pass_rate 1.0 recomputed from 27 items). It fills judging slot 1.
- **Note:** a result record exists on S3 for the crashed run, covering the 22 items that completed; it shows 22 of 22 and must not be cited as the 27-item deny unit. It is listed in `evidence/superseded.json`.
- **Who we told:** not applicable (our own code).

**Log**
- 2 Nov about 21:15 PST (3 Nov 05:14 UTC): run declared (from the run id); crashed at item 23 (time TODO).
- 51010f8: validator-safe guard form.
- 3 Nov about 22:00 PST (3 Nov 06:01 UTC): the fresh deny run `t3.deny-20261103T054914Z` was checked.

<a id="g-016"></a>
## G-016 · Change records carry the approver's AWS identity (declared design)

- **Status:** declared design, kept on purpose.
- **What:** each change record's `approved_by` is the `decided_by` field of the checkpoint's approval row, which is the approver's AWS identity. We keep it in the evidence store as audit evidence, so a judge can match a change to the person who approved it in AWS.
- **On screen:** the dashboard maps it to the pseudonym `person:team13-op1`.
- **Limits:** the evidence files hold an identity. They are in the private team bucket and the private repository. The pre-push scan does not look for identities, so check any file that is going to be published.
- **Who we told:** not applicable.

<a id="g-017"></a>
## G-017 · Seed rules missing from the policy engine after the deploy (diagnosed and fixed)

- **Status:** resolved.
- **What was missing:** the three seed rules were absent from the policy engine after the `614fc3b` deploy.
- **Diagnosis:** `t3/deploy/register.py` read only the first page of the engine's `list_policies` answer. With 24 rules the list spans more than one page, so the script did not see a rule that already existed, tried to create it, failed, and exited 1 without printing an error. It never reached the seed rules.
- **What was done by hand:** the lead created the three seed rules from `generate.rules()` as of `614fc3b`, with the exact parameters `register.py` uses and the forbid rules first. All three are ACTIVE. The other 12 rules were compared and have the same text.
- **Fix:** commit `83f99f8` reads every page, prints failures, and has a test with a two-page list. With cedarpy installed, the 6 seed tests pass.
- **Effect on runs:** the trial `t3.swarm-trial-20261103T052430Z` ran before the seed rules existed and is not a scored run (see the superseded table).
- **Who we told:** not applicable (our own script and account).

**Log**
- 3 Nov (time TODO): missing seed rules noticed after the `614fc3b` deploy; created by hand.
- `83f99f8`: paginated reads, printed failures, two-page test.

<a id="g-018"></a>
## G-018 · Bridge challenge deferred until after the end-to-end pass

- **Status:** deferred (team decision, 2 Nov about 22:30 PST); **no Bridge number exists or is claimed yet.** We finish the end-to-end pass first and reconsider the Bridge at the end. The guide says the Bridge can be added to any track, and plan v3 kept it for last.
- **What the guide describes** (pp. 31 to 32): a loop in which a language model proposes radio-unit configurations, an eight-KPI mock reward scores them, and GRPO trains the model on that reward. The reference baselines are 0.19 for the untrained model and 0.73 for CEM (a classical cross-entropy-method search).
- **What is missing:** the guide gives no code location and no setup steps for the loop, and the starter kit does not contain it. We had no harness to run, reward to call or dataset checks to pass.
- **Two different bars:** the guide's bar is 0.73 (beat CEM). The judging deck sets a different one: a constant policy already scores 0.97, so the unit is to beat that constant perturbed by random search, on the same 20 simulations and seeds. We could not confirm which bar counts.
- **Time:** neither the loop nor the bar was confirmed by the 18:00 PST watch time (2 Nov), so we did not start. The fallback "Bridge not attempted" applies until the deferral is reconsidered.
- **Not a Bridge result:** Track 1's RAN RL work (PPO policy, judging slot 8) is a different loop on the Maveric twin. It is not offered as a Bridge result.
- **Limits:** the optional Bridge unit, and `evidence/reward.csv`, which the guide asks for. We have none.
- **Resolved when:** the organisers give the loop code and confirm the bar, and we choose to run it. If neither happens before the hand-in, this entry stands as "not attempted".
- **To reconsider after the end-to-end pass.**

**Log**
- 2 Nov 18:00 PST: watch time passed with no loop code and no confirmed bar.
- 2 Nov about 22:30 PST: team decision to defer the Bridge, not drop it, until after the end-to-end pass.

<a id="g-019"></a>
## G-019 · The runner sent an empty InputDataConfig, so eight declared runs never started (fixed)

- **Status:** resolved; the affected runs are listed as not run and were relaunched under new ids.
- **What happened:** for jobs with no input channels, the runner put an empty `InputDataConfig` into the `CreateTrainingJob` request. The client library (botocore) refused the request before it reached AWS (a `ParamValidationError`), so no job was created. The declaration had already been written to S3 for each of these runs, so eight declarations exist with no job behind them:
  - four T2 comparisons: `t2.compare.federated_vs_B.disjoint-20261103T054303Z`, `t2.compare.federated_vs_C.disjoint-20261103T054442Z`, `t2.compare.distance_to_central.disjoint-20261103T054624Z`, `t2.compare.federated_vs_localfedavg.disjoint-20261103T054804Z`;
  - four Bedrock CPU runs: `t1.price_latency-bedrock-gpt-oss-20b-1`, `t1.interface-bedrock-gpt-oss-20b-1`, `t1.price_latency-bedrock-qwen3-32b-1`, `t1.interface-bedrock-qwen3-32b-1` (relaunched as `-2`).
- **Root cause and fix:** an empty list is not valid for that parameter. The fixes leave the key out when there are no channels: `33c47be` (W12) and `2e8f846` (W11). Offline tests validate the request with botocore's own parameter validation, and every request is now validated before the launch.
- **Why it is safe to relaunch:** each relaunch takes a new id (`-2` for the Bedrock runs; new ids for the T2 comparisons, to be recorded here when they exist), and the old declarations stay in S3 as "not run" entries in `evidence/superseded.json`.
- **Limits:** the T2 comparisons (judging slot 9) and the Bedrock baselines (slot 7) waited for the relaunches.
- **Who we told:** not applicable (our own code).

**Log**
- 3 Nov (time TODO): launches refused; declarations left without jobs.
- `33c47be`, `2e8f846`: empty `InputDataConfig` left out; request validation added.

<a id="g-020"></a>
## G-020 · Two harness defects found by the first comparison and Bedrock runs (fixed)

- **Status:** resolved; both affected runs are superseded and relaunched under new ids.
- **A missed target made a job exit 1.** The T2 compare job `t2.compare.federated_vs_B.disjoint-20261103T061155Z` finished its measurement, wrote its result, and then exited 1 because the declared target was missed (the interval's lower bound was not above zero). The platform counts that as a failed job, so the result was never attested by the job. Commit `538dc66` makes a missed target a normal result with exit 0; only a real error fails a job. The run was relaunched as `t2.compare.federated_vs_B.disjoint-20261103T063916Z`. The missed target itself is a finding and stays reported (judging slot 9).
- **A non-finite answer crashed a run.** `t1.knowledge-bedrock-gpt-oss-20b-2`: a model's telemath answer was a non-finite string, which was stored as `inf` in an item's details. The record contract allows finite numbers only, so it rejected the item batch and `run.py` exited before writing a result (`ExecuteUserScriptError`, exit 1). Commit `8ca61a4` fixes the handling; the run was relaunched as `t1.knowledge-bedrock-gpt-oss-20b-3`.
- **Limits:** none left once the relaunches have results. No score was changed by either fix; the first run had no attested result and the second had no result at all.
- **Who we told:** not applicable (our own code).

**Log**
- 3 Nov (time TODO): both defects found while the runs were being checked.
- `538dc66`, `8ca61a4`: fixes.
- 2 Nov about 22:50 PST (3 Nov 06:52 UTC): relaunch `t1.knowledge-bedrock-gpt-oss-20b-3` landed: 0.568 against 0.70, MISSED, check_unit parts a, b, c and e pass. The `t2.compare.federated_vs_B` relaunch (`...063916Z`) also landed: lower bound -0.041, a missed target reported as a result.

<a id="g-021"></a>
## G-021 · A false positive in check_unit part d (fixed)

- **Status:** resolved.
- **What:** part d of `tools/check_unit.py` compares each evidence file with what the evidence ledger recorded for it. The result depended on the order in which the ledger had ingested the files, so it could report a mismatch for a file that had not changed.
- **Fix:** commit `06ddb9f`.
- **Limits:** part d needs a ledger, so the judges' live test on a machine without one skips it (the rehearsals above show "SKIP: no ledger"). The false positive could only make a check look worse, never better.
- **Who we told:** not applicable (our own code).

---

## Superseded runs (kept in S3 and in history; not used for any claim)

| Run | Why it is superseded | Replaced by |
|---|---|---|
| `t3.deny-20261103T051458Z` | Crashed at item 23: the Cedar validator rejected the pause-forbid guard form (fixed in `51010f8`, G-015) | `t3.deny-20261103T054914Z` |
| `t2.compare.federated_vs_B.disjoint-20261103T054303Z` | Not run: CreateTrainingJob refused before AWS (empty `InputDataConfig`, G-019); the declaration exists, no job | `t2.compare.federated_vs_B.disjoint-20261103T061155Z` |
| `t2.compare.federated_vs_C.disjoint-20261103T054442Z` | Not run: CreateTrainingJob refused before AWS (empty `InputDataConfig`, G-019); the declaration exists, no job | `t2.compare.federated_vs_C.disjoint-20261103T061346Z` |
| `t2.compare.distance_to_central.disjoint-20261103T054624Z` | Not run: CreateTrainingJob refused before AWS (empty `InputDataConfig`, G-019); the declaration exists, no job | `t2.compare.distance_to_central.disjoint-20261103T061541Z` |
| `t2.compare.federated_vs_localfedavg.disjoint-20261103T054804Z` | Not run: CreateTrainingJob refused before AWS (empty `InputDataConfig`, G-019); the declaration exists, no job | `t2.compare.federated_vs_localfedavg.disjoint-20261103T061725Z` |
| `t2.compare.federated_vs_B.disjoint-20261103T061155Z` | Failed: the job exited 1 on a missed target, so the result was written but never attested; fixed in `538dc66` (G-020) | `t2.compare.federated_vs_B.disjoint-20261103T063916Z` |
| `t1.knowledge-bedrock-gpt-oss-20b-2` | Failed, no result: a non-finite telemath answer was stored as `inf` and the contract rejected the item batch; fixed in `8ca61a4` (G-020) | `t1.knowledge-bedrock-gpt-oss-20b-3` |
| `t3.swarm-trial-20261103T052430Z` | A trial run before the seed rules existed (G-017); not a scored run | the scored swarm run (TODO) |
| `t1.price_latency-bedrock-gpt-oss-20b-1` | Not run: CreateTrainingJob refused before AWS (empty `InputDataConfig`, G-019); the declaration exists, no job | `t1.price_latency-bedrock-gpt-oss-20b-2` |
| `t1.interface-bedrock-gpt-oss-20b-1` | Not run: CreateTrainingJob refused before AWS (empty `InputDataConfig`, G-019); the declaration exists, no job | `t1.interface-bedrock-gpt-oss-20b-2` |
| `t1.price_latency-bedrock-qwen3-32b-1` | Not run: CreateTrainingJob refused before AWS (empty `InputDataConfig`, G-019); the declaration exists, no job | `t1.price_latency-bedrock-qwen3-32b-2` |
| `t1.interface-bedrock-qwen3-32b-1` | Not run: CreateTrainingJob refused before AWS (empty `InputDataConfig`, G-019); the declaration exists, no job | `t1.interface-bedrock-qwen3-32b-2` |
| `t1.rl.link-test-20261102T234623Z-7368` | Infrastructure link test from before the runner pinned code, not a result: check_unit part [b] fails because 18 files the job ran are not in commit `425521f` (G-001) | none (the scored RL run is `t1.rl.eval-20261103T043115Z-bf52`) |
| `t1.rl.reward-check-20261103T020244Z-4f92` | Supporting check, not a judged result: part [c] cannot establish the AWS window, because the spot job's SDK name does not carry the run id; the dev reward check reached coverage 0.891 against a dev bar of 0.82 | none |
| `t1.knowledge-otel-base-1` | Harness defect: the JSON-only knowledge parser rejected bare-letter answers (parse_fail 0.90); fixed narrowly in `c4b4b5e` (G-013) | `t1.knowledge-otel-base-2` |
| `t3.evals.no_action-20261103T023304Z-0cd3` | Failed smoke run on the stand-in twin backend, against the old v2 suite; no result record | `t3.evals.no_action-20261103T032552Z-x86` |
| `t3.evals.no_action-20261103T024101Z-22ae` | Failed smoke run, same cause; no result record | `t3.evals.no_action-20261103T032552Z-x86` |
| `t3.evals.rule_baseline-20261103T033026Z-x86`, `t3.evals.config_revert-20261103T033026Z-x86`, `t3.evals.config_revert_masked-20261103T033026Z-x86` | First gain baselines on suite v4; replaced by the replayable runs below | the `...T035108Z-x86` runs |
| `t3.evals.rule_baseline-20261103T035108Z-x86`, `t3.evals.config_revert-20261103T035108Z-x86`, `t3.evals.config_revert-masked-20261103T035108Z-x86` | current gain baselines | (these are the ones to cite) |
