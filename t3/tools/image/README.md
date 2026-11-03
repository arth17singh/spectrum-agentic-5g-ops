# Heavy-tools Lambda image (arm64)

Container image for the Gateway tools that replay the organisers' in-process Maveric twin
(torch, gpytorch, the `ran_rl` SDK) and run PPO policies (stable-baselines3). A zip cannot hold
torch (250 MB unzipped limit), so these tools are one Lambda **container image**.

- Base: `public.ecr.aws/lambda/python:3.12`, built for **linux/arm64** (this laptop is Apple Silicon;
  create the Lambda with architecture `arm64`).
- CPU torch 2.15.0 from the PyTorch CPU index; gpytorch 1.15.2, stable-baselines3 2.9.2, numpy 2.5.4,
  pandas, gymnasium, pydantic 2.14.2, boto3 (versions match the shared dev env).
- The organisers' `ran_rl` wheel, which already carries the Maveric GP models
  (`sim_tilt_sweep_4cell`), topology and `scenarios.yaml`. Nothing else from the organisers is needed.
- Our packages (`twin`, `t2`, `t1`, `t3`, `contracts`, `tools`) are copied to `/var/task` in the last layer.
- `CMD ["t3.tools.app.handler"]`. Smoke entry point: `smoke.smoke_handler.handler`.

## Licence rule

The organisers' terms forbid redistribution. The wheel is **never committed**: `build.sh` copies it
into a temporary build context outside the repo and deletes that directory afterwards. The built image
contains the wheel's code, so keep the ECR repository private and never make the image public or
share it outside the team account.

## Build (from a pinned commit)

```bash
t3/tools/image/build.sh <commit sha or ref>        # add --no-cache to force a rebuild
# override: RAN_RL_WHEEL=/path/ran_rl-1.2.9-py3-none-any.whl IMAGE_REPO=team13-ran-tools
```

The code comes from an archive of that commit, never from the working tree, so uncommitted edits cannot
ride along. It stages `wheel/`, `smoke/` (from the commit) and the packages at that commit (`twin t2 t1
t3 contracts tools`; `t3/tools/image`, `tests` and caches are removed; `team13_contracts` is also copied
to the top level so it imports in Lambda), runs `docker build --platform linux/arm64`, and prints the
size. The image is tagged `team13-ran-tools:<first 12 hex of the sha>` and labelled
`org.opencontainers.image.revision=<full sha>`; `/var/task/IMAGE_REVISION` holds the sha too. It never
pushes. Rebuilds after a code change only redo the last layer (a few seconds).

## Run locally (Lambda runtime interface emulator, built into the base image)

```bash
t3/tools/image/invoke_local.sh --smoke                              # smoke handler, events/smoke_5.json
t3/tools/image/invoke_local.sh --smoke events/smoke_20.json         # more replayed steps
t3/tools/image/invoke_local.sh events/tool_observe.json             # real handler (needs t3.tools.app)
t3/tools/image/invoke_local.sh --memory 2g --cpus 1.2 --smoke       # emulate a smaller Lambda
```

The script runs `docker run -p 127.0.0.1:9000:8080 -e T3_STORE=local:/tmp/t3store -e T3_TOOL_NAME_FALLBACK=1 ...`
and POSTs each event to `http://127.0.0.1:9000/2015-03-31/functions/function/invocations`, then removes
the container (`--keep` leaves it up; stop with `docker rm -f team13-ran-tools-local`).

Tool name: the Gateway passes it in `context.client_context.custom["bedrockAgentCoreToolName"]` (for
example `ran___apply_change`). The emulator cannot set client context, so `t3.tools.app` must, only when
`T3_TOOL_NAME_FALLBACK=1`, read the tool name from `event["_tool"]`. Never enable that variable in AWS.
`events/tool_*.json` use that field.

## Push to ECR (documented, not run by anyone yet; the lead does this)

Use profile `fyuz`, region us-east-2, and the account's private repository. Do this from the laptop
where the image was built so the architecture matches (arm64).

```bash
AWS_PROFILE=fyuz; REGION=us-east-2
ACCOUNT=$(aws sts get-caller-identity --profile $AWS_PROFILE --query Account --output text)
REPO=team13-ran-tools
aws ecr create-repository --repository-name $REPO --region $REGION --profile $AWS_PROFILE \
  --image-scanning-configuration scanOnPush=true         # once; private by default
aws ecr get-login-password --region $REGION --profile $AWS_PROFILE \
  | docker login --username AWS --password-stdin $ACCOUNT.dkr.ecr.$REGION.amazonaws.com
TAG=<the 12-hex tag build.sh printed>
docker tag team13-ran-tools:$TAG $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/$REPO:$TAG
docker push $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/$REPO:$TAG
# deploy exactly that tag (register.py refuses :latest and a missing --image-uri):
#   python -m t3.deploy.register ... --image-uri $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/$REPO:$TAG
```

`build.sh` passes `--provenance=false` so the pushed manifest is a plain single-arch image Lambda accepts.
Create the function with `PackageType=Image`, `Architectures=[arm64]`, the ECR image URI, and the
memory/timeout below. If service roles under `/bb26/` need an ECR pull permission, the repository
policy must allow `lambda.amazonaws.com` (`ecr:BatchGetImage`, `ecr:GetDownloadUrlForLayer`).

## Measured on this laptop (Docker Desktop, arm64, smoke handler)

Image: 2.06 GB on disk, 467 MB compressed (what ECR stores). Smoke handler, median of 6 warm calls
(client-side, includes a tiny PPO predict). Numbers vary by about 30% with laptop load.

| container limit | cold first call | 5 steps | 10 steps | 20 steps | peak (docker stats) |
|---|---|---|---|---|---|
| 3 GB, 2 CPU | 1.5-2.0 s | 0.23-0.29 s | 0.30-0.39 s | 0.45-0.62 s | 760-830 MiB |
| 2 GB, 1.2 CPU | 1.4-1.9 s | 0.31-0.41 s | 0.46-0.58 s | 0.71-0.91 s | 760-830 MiB |
| 1 GB, 0.6 CPU | 2.9 s | 0.83 s | 1.19 s | 1.69 s | 848 MiB (near the limit) |

**Recommendation:** Lambda memory 3008 MB (at least 2048 MB), timeout 120 s. A twin rebuilt every call
climbs to about 950 MB RSS over 30 calls, and the real handler adds boto3 and a real PPO. 120 s is below
the kit's A2A (240 s), HTTP (270 s) and Lambda (300 s) waits, so a stuck replay surfaces as a Lambda
error instead of the caller giving up first. Local cold start excludes the image pull: expect 5-15 s on
real Lambda, so send one warm-up invocation before the demo.

Re-run `invoke_local.sh --smoke` to reproduce. Key points:
- a warm container that rebuilds the twin every call grows ~15 MB per call up to ~950 MB RSS;
  one that keeps the twin and only calls `reset` stays flat at ~470 MB and is about 40% faster
  (`"reuse_twin": true` in the smoke event). Real tools should cache the twin object at module level.
- Local numbers are a lower bound: a real cold start also pulls the image layers to the Lambda host.
