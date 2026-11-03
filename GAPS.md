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
| [G-005](#g-005) | Resolving · platform | Track 2 FL service not provisioned at the 16:00 enablement | All Track 2 units | 2 Nov 16:40 PST |
| [G-006](#g-006) | Declared risk · organiser-side IAM | The Agent Builder host role can create and modify any role under /bb26/; participant SSO role can invoke our tools Lambda directly | Track 3 claim "only the Gateway invokes our tools" holds for non-human principals only | 2 Nov 18:20 PST |

### Watching

These become entries if still missing at the time shown (2 Nov, PST).

| Due | Waiting for | Last check | Planned fallback |
|---|---|---|---|
| 14:00 | Viavi adapter address and token for team13 | **Partly resolved (~16:40 PST).** Our RSG's MCP server is at `/mcp` on the Viavi UI address (port 443). It sits behind the UI's Cognito sign-in and answers only with a signed-in web session. From the user's own signed-in session we ran read-only calls (handshake, tool list, docs index, scenario and run lists): 21 tools; `ran-rl-maveric-4cell` is not saved yet; no run is active. Still missing: a non-interactive credential for jobs, agents and the twin host's adapter (asked the organisers). | Until then, Viavi runs need the user's signed-in session; unattended jobs can't reach it |
| 16:00 | Track 2 federated-learning service enabled for team13 | 1 Nov 21:00: no Track 2 silo buckets and no stacks in us-west-2 | local federated averaging over the same silos, labelled "not the service" |
| 18:00 | Bridge challenge loop code, and which bar counts (0.73 in the guide, 0.97 in the judging deck) | not in the starter kit | Bridge not attempted |
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

- **Status:** resolving. The account is allow-listed: a read-only `list-sessions` in us-west-2 at about 18:10 PST returned an empty session list with no access error. The CDK bootstrap in us-west-2 is pending.
- **Service:** the event's federated-learning service (us-west-2).
- **What did not work:** at the 16:00 PST enablement there were no `-usw2` silo buckets (checked 16:40 and again at about 18:30), no install command and no allow-list confirmation.
- **Meanwhile:**
  - the utility package named in the service's guide is readable from our account. It was downloaded, verified against its published SHA-512 sums, and installed outside the repo;
  - our synthetic silo datasets (generated by `silos/`, hashed in `silos/manifest.json`) stand in for organiser silos.
- **Risk still open:** the CDK bootstrap and the utility's own stacks create IAM roles. Our account requires the `/bb26/` path and the permissions boundary, so both may be refused.
- **Who we told:** TODO (user).

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

