# Track 3: governed RAN tools

Team13's tools for the agent swarm. Every call crosses the organisers' checkpoint (AgentCore Gateway + Cedar + the kit's interceptor). This folder holds the tool Lambda (`tools/`), our Cedar rules (`policies/`) and the script that registers both on the kit's Gateway (`deploy/`). The Lambda image belongs to `tools/image/` (W3b).

Contract: `BUILD_CONTRACT.md` plus amendment 1 (A1 to A16). The kit's own code is not in this repo, and must not be: its terms forbid redistribution. Line numbers below refer to the organisers' `checkpoint_kit.py`.

## Tools

One Lambda (`t3.tools.app.handler`, image, 3008 MB, 120 s) serves every tool. The Gateway sends the tool name as `<target>___<tool>`. Every tool needs `case_id` (`case-...`). Every answer is JSON and includes `case_id` and `run_id`. A refusal is `{"error": "..."}`.

| Tool | Who may call | Input | Output |
|---|---|---|---|
| `ran___observe` | noc_lead, ran_monitor | `case_id` | `twin {kind, version, scenario, seed, step}`, `kpis` (headline and per cell) |
| `ran___propose` | ran_optimizer | `case_id`, `model_version?` | `item_id` (`step-<n>`), `change` (canonical string), `policy {model_id, version}`; or `{no_change: true, item_id: null, change: null, reason}` when the policy proposes nothing (nothing is recorded) |
| `ran___predict` | ran_optimizer | `case_id`, `item_id`, `change`, `model_version?` | `predicted_delta` (KPI change compared with doing nothing), `predictor`. Stored in the case row |
| `ran___preflight` | ran_optimizer | `case_id`, `change` | `baseline_kpis`, `measured_kpis`, `measured_delta` and `clipped`, all on a copy. Changes nothing, and the trust gate does not score it |
| `ran___apply_change` | noc_lead, after a person's approval | `case_id`, `item_id`, `change`, `reason?`, `approval?` (the checkpoint sets it), `special_context?` | `applied`, `approval_id`, `measured_delta`, `predicted_delta`, `prediction_error` (the record's `error`), `direction_agree`, `scored`, `unscored_reason`, `trust`, `evidence` |
| `nlm___ask` | ran_explainer, if the question passes the prompt-attack guardrail | `case_id`, `question`, `model_version?` | `answer`, `answered_by {kind, id, version, model_label, stand_in}` |

**The change string (A2).** A change is written `cell=<int>;d_power_db=<+x.x>;d_tilt_deg=<+x.x>`. For several cells, join them with `|` and sort them by cell. Each value has one decimal and an explicit sign, and zero is written `+0.0`. Agents copy the string verbatim from `ran___propose`. The approval id hashes the exact string the agent sends.

**Apply.** The approval id is computed as `"chg-" + sha256(json.dumps([case_id, item_id, change]))[:12]`, the same key as the kit's `change_key` (128-131). After the person approves, apply runs these steps:

1. It measures the change against the no-change counterfactual at the same step (A5), giving `measured_delta = measured − baseline`. It uses W2's `ReplayTwin.counterfactual` when the twin has it; otherwise it replays twice, once with no change (the baseline) and once with the change.
2. It runs one DynamoDB `TransactWriteItems` that does two things together:
   - moves the approval from APPROVED to APPLIED, using the kit's own conditional update (161-167);
   - writes the case row.
   One approval therefore applies exactly one change. A second identical call is refused as "already applied".
3. It reads the prediction that `ran___predict` stored for the same `change`, preferring the one with the same `item_id`. It never runs a model itself (A4).
4. For a scored change, it calls the `team13-trust-gate` Lambda **synchronously**:
   - The call is `RequestResponse` with a 5 s read timeout (`T3_TRUST_TIMEOUT_S`) and no retries.
   - The payload is `{record, model_version}`, where `model_version` is the stored predictor's version.
   - The gate answers `{trust_before, trust_after, e, score, action, control_record_key?}`.
   - If the gate errors or times out, the trust fields stay null and the result's `trust.note` says why. The apply never fails because of the gate.
   - The gate keeps the policy-write rights; the tools only invoke it (A7).
5. It writes the `change` record (A15) once, with `trust_before` and `trust_after` from the gate, to `records/<run_id>/change/<approval_id>.json` (one object per record, A11).
   - A clipped change is recorded with `scored: false, unscored_reason: "clipped"`.
   - A change with no prediction is recorded with `unscored_reason: "no_prediction"`.
   - Neither unscored case calls the gate.
6. It prints a `{"record": "change_applied", case_id, run_id, approval_id, change_text}` line.

**`error`** is computed exactly on the deltas, with no rounding:
- for each KPI, `|pred_Δ − meas_Δ|`;
- `summary` is `mean(|pred_Δk − meas_Δk| / scale_k)` over coverage_ratio, sla_ratio and sinr_p10_db.

The scales come from `T3_ERROR_SCALES` (JSON) or from `t3/trust/thresholds.json` → `scales`. If neither exists, every scale is 1.

**`direction_agree`** means the predicted and measured deltas have the same sign for coverage_ratio and sla_ratio. Each comparison has a dead band of 0.002.

**Case state** lives in the kit's approvals table (A10). The case row has `pk = "t3-" + case_id`, a `doc` (JSON) and a `version`. It has no `status` attribute, so the kit's `approve` command never lists it. The `doc` holds:
- `case_id`, `run_id`, `scenario`, `seed`, `config`;
- `action_log` (W2's entries);
- `twin_kind`, `predictions`, `created_at`.

The runner writes `pk = "t3-active"` with the current `run_id`. A case without a run gets `adhoc-<case_id>`. The twin is W2's `ReplayTwin`. It is replayed from the case row on every call and is cached per (scenario, seed, config) (A16).

## Cedar rules (`policies/`)

`python -m t3.policies.generate` writes `policies/generated/*.cedar` and `manifest.json`. `--check` verifies that the committed files are current and passes the offline checks:
- legal names of 48 characters or fewer;
- balanced syntax with exactly one statement;
- the kit's validation modes;
- every action is one of our tools;
- every `context.input.X` is declared in that tool's schema;
- every tool has a permit, and every tool is covered by the case-id forbid.

The files keep `@GW@` in place of the Gateway ARN.

| Rule | Mode | Says |
|---|---|---|
| `team13_ran_allow_<tool>` (five rules) | strict | Only the agents listed in `AGENT_TOOLS` may call the tool (the team's agent tokens) |
| `team13_ran_nlm_safe_questions` | strict | ran_explainer may call `nlm___ask` if the guardrail scores the question ≤ 0.4 for jailbreak and for prompt injection (the same as the kit's `allow_nlm_safe_questions`) |
| `team13_ran_screen_nlm_answers`, `team13_ran_screen_nlm_answers_kit_name` | loose | `suppressOutput` when the answer scores > 0.2 for violence, hate or misconduct (the same as the kit's `screen_nlm_answers`). There is one rule per name because, after the takeover of target `nlm`, both `nlm___ask` and `nlm___ask_nlm` reach our code |
| `team13_ran_no_kit_apply_change` | loose | Forbid the kit's `work___apply_change` for our four agents. Its approval id is the same hash as `ran___apply_change`'s, so a call to it would consume a person's approval without changing the twin. The kit's `check` only expects that call to be REFUSED, which still holds |
| `team13_ran_require_case_id` | loose | Forbid all six tools unless `case_id like "case-*"` |
| `team13_ran_require_human_approval` | loose | Forbid `ran___apply_change` for noc_lead unless `approval == "APPROVED"`. The kit's interceptor rewrites the refusal only when the Cedar error contains `require_human_approval` (314) |
| `team13_ranpause_<tool>_<model>_<version>` | loose | Pause template (A8): `forbid … when { !(context.input has model_version) \|\| context.input.model_version == "<v>" }`. The tools refuse a `model_version` they do not serve, so naming another version cannot reach a paused model. The NLM pause covers both `nlm___ask` and `nlm___ask_nlm`. Never prefix a rule `team13_pause_nlm`: the kit's `resume nlm` deletes everything with that prefix |

## Deploy order

1. **Kit `deploy`.** It creates the Gateway, the policy engine, `team13-approvals`, the interceptor and the targets `case`, `work`, `nlm`, `swarm` and `front`. Set these in `kit.json`:
   - `lead: noc_lead`;
   - `specialists: [ran_monitor, ran_optimizer, ran_explainer]`;
   - `may_change: [noc_lead]`.

   Without `may_change: [noc_lead]`, the interceptor marks every apply NOT_ALLOWED and nobody is asked to approve (290-297).
2. **IAM, in CloudShell (us-east-2):** `EVIDENCE_BUCKET=<team bucket> bash t3/deploy/iam_cloudshell.sh`. This session cannot create roles, so this step runs there. The script is idempotent, reads the account id at run time and contains no secrets. It does three things:
   - creates `/bb26/team13-ran-tools`, a Lambda role with the event boundary. Its inline policy grants the approvals table (keys `chg-*` and `t3-*`), S3 get and put on `evidence/*`, its own logs, a synchronous invoke of `team13-trust-gate`, the Bedrock stand-in model, and our SageMaker endpoint;
   - creates `/bb26/team13-trust-gate`. Its inline policy grants create, delete, get and list of policies on our policy engine only, trust rows (`t3-trust#*`), control records and logs;
   - puts the inline policy `team13-ran-tools-invoke` on the kit's role `team13-gateway`.
3. **Build and push the image from a pinned commit:**
   - Run `t3/tools/image/build.sh <commit sha>`. It builds from an archive of that commit, never the working tree, and tags `team13-ran-tools:<12-hex sha>` with the label `org.opencontainers.image.revision`.
   - Push that tag to ECR `team13-ran-tools` (arm64); see `t3/tools/image/README.md`.
4. **Dry run:** `python -m t3.deploy.register --kit-dir <kit folder> --image-uri <account>.dkr.ecr.us-east-2.amazonaws.com/team13-ran-tools:<sha tag> --evidence-bucket <bucket>`.
   - It makes no IAM calls. Read the plan.
   - `--image-uri` is required, also for the dry run. It is refused if it is missing, is `:latest`, has no tag, or is in another region or account.
5. **Apply:** add `--apply --profile fyuz`. `T3_TRUST_LIVE` stays 0 unless you add `--trust-live` for this run (the trust gate then creates real pause rules). Apply also sets these environment variables:
   - on both functions: `EVIDENCE_S3=s3://<bucket>/evidence/`;
   - on the tools function: `T1_POLICY_PATH=/var/task/t1/rl/artifacts/ppo_v0.zip`, `T2_PREDICTOR_PATH=/var/task/t2/artifacts/t2_response_v0.npz` and `T3_TRUST_TIMEOUT_S=15`.

   It then does four things:
   - creates the ECR repository and its pull policy for Lambda;
   - creates or updates `team13-ran-tools` (3008 MB, 120 s) and `team13-trust-gate` (same image, command `t3.trust.handler.handler`);
   - creates target `ran` and takes over target `nlm`;
   - waits for READY, then creates our rules.
6. **Later kit deploys keep our wiring:**
   - the kit rewrites only its own inline policy (519), never `team13-ran-tools-invoke`;
   - it skips targets that already exist (673-674), so it leaves our `nlm` target alone;
   - it replaces only rules from its own list (723-737).

   Never run the kit's `destroy`: it deletes the table and every target and rule.

## Environment variables

| Variable | Meaning |
|---|---|
| `APPROVALS_TABLE` | The kit's table (`team13-approvals`): approvals, case rows, the run row |
| `T3_STORE=local:<dir>` | Local and Docker runs. JSON files replace DynamoDB, and the tool creates PENDING approvals itself, as the interceptor would. Approve with `python -m t3.tools.store <dir> approve chg-…`, and set the run with `python -m t3.tools.store <dir> run <run_id>` |
| `T3_TOOL_NAME_FALLBACK=1` | Local only: the tool name comes from `event["_tool"]`. Never set it in AWS |
| `EVIDENCE_S3`, `EVIDENCE_DIR` | Where records go: `s3://<bucket>/evidence/` or a folder. These are `tools.evlib`'s variables, shared with the trust gate. `T3_EVIDENCE` (`local:<dir>` or `s3://…`) overrides them for local runs |
| `T3_TRUST_FUNCTION`, `T3_TRUST_TIMEOUT_S`, `T3_TRUST=off` | The trust-gate Lambda (default `team13-trust-gate`). The synchronous call times out after `T3_TRUST_TIMEOUT_S` seconds: 5 in code, 15 as deployed. `T3_TRUST=off` turns the call off |
| `T3_ERROR_SCALES` | JSON `{kpi: scale}` for `error.summary` (default: `t3/trust/thresholds.json`) |
| `T2_PREDICTOR_PATH`, `T1_POLICY_PATH` | The model files (W9) |
| `NLM_ENDPOINT`, `NLM_MODEL_VERSION`, `NLM_PAYLOAD` (`messages` or `tgi`) | Our tuned model on SageMaker, called with SigV4 through boto3 |
| `NLM_STANDIN_MODEL` | A Bedrock model id used until the endpoint exists. Its answers are labelled `stand_in: true` |
| `T3_SCENARIO`, `T3_SEED`, `T3_TWIN_CONFIG`, `T3_TWIN_KIND`, `T3_TWIN_VERSION`, `T3_SYNTHETIC` | Twin defaults for new cases, and the record flags |

## Tests

```
PY=<scratchpad>/venv-t3/bin/python
$PY -m pytest -q t3/tools/tests t3/policies/tests t3/deploy/tests
$PY -m t3.policies.generate --check
```

The tests use fakes for DynamoDB (low-level calls, validated against botocore's model), S3, the twin, the predictor, the policy, the trust gate, SageMaker and Bedrock. They make no network calls.
