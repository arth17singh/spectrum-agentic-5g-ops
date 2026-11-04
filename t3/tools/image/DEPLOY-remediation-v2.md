# Deploy remediation-v2 behind `ran___propose` (runbook)

**Needs the user's go. Every command below that writes to AWS is marked WRITE.** Nothing in this branch has been
pushed or deployed. Run from the repo root of the `deploy/remediation-v2` worktree with
`export AWS_PROFILE=fyuz AWS_REGION=us-east-2`. Never print tokens, the Gateway URL or the account id; the
commands read the account id at run time and never write it to a file.

## What is deployed, and why it is safe to switch in two steps

- **Image:** `team13-ran-tools:<TAG>`, built by `t3/tools/image/build.sh <commit>` from this branch (arm64,
  about 2.07 GB on disk, 470 MB of content). `<TAG>` is the first 12 hex digits of the commit that was built.
  The image carries **both** policies: `t1/rl/artifacts/ppo_sm_50k.zip` (today's) and
  `t1/rl/artifacts/remediation_v2.zip` (sha256 `cacb1def…`, the model that passed
  `t3.evals.remediation-policy-v2-20261104T113942Z`).
- **Code change** (on top of the deployed commit `614fc3b373de`): `ran___seed_case` records `baseline_step`
  (1, the state after commissioning); `ran___propose` gives a policy with `needs_baseline` the pre-incident baseline
  and both per-cell share lists. A policy without it (ppo_sm_50k) takes the old path: its answers are
  byte-identical to the deployed commit (test `t3/tools/tests/test_propose_baseline.py`, fixture recorded at
  614fc3b). The served policy is chosen only by the Lambda variable `T1_POLICY_PATH`.
- So the deploy is two separate, separately reversible steps: (A) new code, still serving ppo_sm_50k;
  (B) switch `T1_POLICY_PATH` to remediation_v2.

## 0. Record the rollback target (read-only)

```bash
aws lambda get-function --function-name team13-ran-tools \
  --query '{image:Code.ImageUri,digest:Code.ResolvedImageUri,updated:Configuration.LastModified}' --output json \
  | sed -E 's/[0-9]{12}\.dkr/<acct>.dkr/g'
aws lambda get-function-configuration --function-name team13-ran-tools \
  --query 'Environment.Variables.T1_POLICY_PATH' --output text
```

Expected today: image tag `team13-ran-tools:614fc3b373de`, digest
`sha256:fbf6a1bb68d74cf0579d6ef3a161d43c872105128728ba08d188962b9b16d992`, and
`T1_POLICY_PATH=/var/task/t1/rl/artifacts/ppo_sm_50k.zip`. If either differs, stop: the Lambda changed since this
runbook was written.

## 1. Push the image (WRITE: ECR)

```bash
TAG=<12-hex tag printed by build.sh>
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REPO="$ACCOUNT.dkr.ecr.us-east-2.amazonaws.com/team13-ran-tools"
aws ecr get-login-password | docker login --username AWS --password-stdin "${REPO%%/*}"
docker tag "team13-ran-tools:$TAG" "$REPO:$TAG"
docker push "$REPO:$TAG"
DIGEST=$(aws ecr describe-images --repository-name team13-ran-tools --image-ids imageTag="$TAG" \
  --query 'imageDetails[0].imageDigest' --output text)
echo "pushed $TAG as $DIGEST"
```

## 2. Step A: new code, same policy (WRITE: Lambda code)

```bash
aws lambda update-function-code --function-name team13-ran-tools --image-uri "$REPO@$DIGEST" \
  --query 'LastUpdateStatus' --output text
aws lambda wait function-updated-v2 --function-name team13-ran-tools
```

The function now runs the new code and still serves ppo_sm_50k, byte for byte as before. A Live Fix run here should
behave exactly as run 2 did (`cell=2;d_power_db=+8.0;…` style changes).

## 3. Step B: switch the served policy (WRITE: Lambda configuration)

`update-function-configuration --environment` REPLACES the whole variable map, so read it, change one key, write it
back. The temporary file is mode 0600 and deleted; no value is printed.

```bash
python3 - <<'EOF'
import json, os, subprocess, tempfile
FN, KEY = "team13-ran-tools", "T1_POLICY_PATH"
OLD, NEW = "/var/task/t1/rl/artifacts/ppo_sm_50k.zip", "/var/task/t1/rl/artifacts/remediation_v2.zip"
cfg = json.loads(subprocess.check_output(["aws", "lambda", "get-function-configuration", "--function-name", FN,
                                          "--output", "json"]))
env = cfg["Environment"]["Variables"]
assert env.get(KEY) == OLD, f"{KEY} is not the expected value; stop"
env[KEY] = NEW
fd, path = tempfile.mkstemp(suffix=".json"); os.fchmod(fd, 0o600)
try:
    with os.fdopen(fd, "w") as f:
        json.dump({"Variables": env}, f)
    subprocess.check_call(["aws", "lambda", "update-function-configuration", "--function-name", FN,
                           "--environment", f"file://{path}", "--query", "LastUpdateStatus", "--output", "text"])
finally:
    os.remove(path)
print(f"{KEY} -> {NEW}")
EOF
aws lambda wait function-updated-v2 --function-name team13-ran-tools
```

## 4. Evidence names on the laptop side

Runs declared by the laptop (Live Fix, live swarm, self-improvement) name the served policy in their declarations.
`t3/evals/live_swarm.py` reads `T3_POLICY_ID`, `T3_POLICY_VERSION` and `T3_POLICY_ARTIFACT` only on
`feat/remediation-policy` (commit 92784a3); the Live Fix server runs from `feat/live-fix`, which does not have it yet,
so until that commit is merged there its declarations would still say ppo_sm_50k. After merging, start the server
with:

```bash
T3_POLICY_ID=t1-remediation-bc-delta T3_POLICY_VERSION=v2-c282d94 \
T3_POLICY_ARTIFACT=t1/rl/artifacts/remediation_v2.zip  <the usual make demo command>
```

The tools' own records (`ran___propose` answers, change records) name the model from the policy's meta in any case:
`{"model_id": "t1-remediation-bc-delta", "version": "v2-c282d94"}`.

## 5. Smoke through the Gateway (sends 2 calls, applies nothing)

```bash
export T3_GATEWAY_URL="$(python3 -c "import json;print(json.load(open('$HOME/firstmate-homes/team13-kit/kit-state.json'))['gateway_url'])")"
export T3_TOKENS_DIR=$HOME/firstmate-homes/team13-kit/tokens
PYTHONPATH=.:contracts python -m t3.tools.image.smoke_gateway_v2 --live
```

It seeds a fresh case at the Live Fix fault "power amplifier fault on cell 2" (as team_app) and asks
`ran___propose` (as ran_optimizer). PASS = `t1-remediation-bc-delta@v2-c282d94`, `baseline_step` 1, a change on cell 2
only with positive power. The local image answered `cell=2;d_power_db=+6.2;d_tilt_deg=+0.0` for this case. Then run
one Live Fix fault from the screen and approve it as usual.

## Rollback

- **Policy only (fast, keeps the new code):** run step 3 with `OLD` and `NEW` swapped. ppo_sm_50k is in the new image
  and serves byte for byte as before.
- **Full (back to the deployed image):** FIRST set `T1_POLICY_PATH` back to ppo_sm_50k (as above: the old image has
  no remediation_v2.zip, so the old code would fail to load the policy), THEN:

```bash
aws lambda update-function-code --function-name team13-ran-tools \
  --image-uri "$REPO@sha256:fbf6a1bb68d74cf0579d6ef3a161d43c872105128728ba08d188962b9b16d992" \
  --query 'LastUpdateStatus' --output text
aws lambda wait function-updated-v2 --function-name team13-ran-tools
```

## Why not `register.py --apply`

It rewrites the Lambda's whole environment from its own flags (anything not passed is dropped: the evidence bucket,
the federated predictor path, the NLM endpoint, the trust settings), and it re-applies the Gateway targets and the
Cedar rules. This change needs one image and one variable, so the narrow commands above are safer and are each
reversible.

## Known effects and risks

- **Chat cases cannot get a proposal under v2.** A case opened without `ran___seed_case` (Ask the NOC) has no
  pre-incident step, so `ran___propose` returns a readable refusal instead of PPO's change. Cases seeded before the
  deploy have no `baseline_step` either and are refused the same way.
- **The served version changes** from `50k-a09e` to `v2-c282d94`. A client that pins `model_version` is refused by
  `check_model_version`; the Cedar pause template for propose is keyed to a version string.
- **Latency and memory:** one replay as before, plus two `twin.kpis()` calls for the shares. Locally (2 vCPU, 3 GB)
  a cold container answered seed in about 1.1 s and the first propose in about 0.4 s; warm proposes took 0.1 s.
  Memory use is unchanged in kind (one more SB3 model of 108 KB on disk).
- **Numerics:** the gate ran on x86 (torch 2.9.0, numpy 2.5.3); the Lambda runs arm64 (torch 2.15.0, numpy 2.5.4).
  Parity with the gate's operator was checked on arm64 with the image's library versions (8 dev incidents x
  3 steps, identical); a proposal near the deadzone or a 0.1 dB quantisation boundary could still differ by one step
  between platforms. Cross-check: with ppo_sm_50k the local image reproduced Live Fix run 2's live answer exactly.
- **Permissions after the event:** ECR push, Lambda code and configuration updates have not been tried with the
  participant role since the freeze.

## Executed (4 Nov 2026, with the user's go)

- **0. Rollback target confirmed:** image `team13-ran-tools:614fc3b373de`, digest
  `sha256:fbf6a1bb68d74cf0579d6ef3a161d43c872105128728ba08d188962b9b16d992`, `T1_POLICY_PATH` = ppo_sm_50k.
  The local image `team13-ran-tools:56d55b4eeff6` carried `remediation_v2.zip` sha256 `cacb1def…` (the gated model)
  and `ppo_sm_50k.zip` sha256 `e04d8c90…`.
- **1. Pushed** `56d55b4eeff6` to ECR as `sha256:b231a5ec76b903b386dd8344ae743f8b87512f318c4a25ea60403bac6089fc3c`.
- **2. Step A** (new code, ppo_sm_50k still served): Lambda `team13-ran-tools` on that digest, Active / Successful.
  Gateway smoke as a check: `ran___propose` answered `t1-ppo-cco-sm@50k-a09e` with
  `cell=0;d_power_db=-1.1;d_tilt_deg=-0.5|cell=1;d_power_db=+3.9;d_tilt_deg=+0.0|cell=2;d_power_db=+8.0;d_tilt_deg=-2.0|cell=3;d_power_db=+2.3;d_tilt_deg=+2.3`,
  byte-identical to Live Fix run 2's live answer (gateway date 2026-11-04T15:23:37Z).
- **3. Step B:** `T1_POLICY_PATH` → `/var/task/t1/rl/artifacts/remediation_v2.zip`; the other 12 variables kept;
  Active / Successful.
- **5. Gateway smoke: PASS.** Case `case-smokev2152409`: `t1-remediation-bc-delta@v2-c282d94`, `baseline_step` 1,
  `cell=2;d_power_db=+6.2;d_tilt_deg=+0.0` (gateway date 2026-11-04T15:24:26Z), the same as the local image.
- **4. Laptop side:** `feat/live-fix` got `92784a3` (as `8ff204f`) and Live Fix declarations now carry
  `served_policy` (`9538ed5`); the demo server was restarted with
  `T3_POLICY_ID=t1-remediation-bc-delta T3_POLICY_VERSION=v2-c282d94 T3_POLICY_ARTIFACT=t1/rl/artifacts/remediation_v2.zip`.
- **Rollback** is unchanged from the section above: policy only (swap `T1_POLICY_PATH` back), or full (policy first,
  then the `fbf6a1bb…` digest).
