# syntax=docker/dockerfile:1
# Team13 heavy-tools Lambda container image (arm64).
# Build with t3/tools/image/build.sh: it stages the organisers' wheel and our packages into a
# temporary build context OUTSIDE the repo. Never COPY organiser files from the repo.
#
# Always built with `docker build --platform linux/arm64` (build.sh does).
# Layers, slowest-changing first:
#   1. CPU torch                 (rarely changes)
#   2. gpytorch / SB3 / pandas / boto3 / pydantic
#   3. the organisers' ran_rl wheel (carries the Maveric GP models, topology, scenarios)
#   4. our packages (twin, t2, t1, t3, contracts, tools): the only layer that changes per edit
FROM public.ecr.aws/lambda/python:3.12

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    MPLCONFIGDIR=/tmp/mpl \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2

# 1. CPU-only torch from the PyTorch CPU index (version matches the shared dev env); drop headers/tests.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0 \
 && rm -rf /var/lang/lib/python3.12/site-packages/torch/include \
           /var/lang/lib/python3.12/site-packages/torch/test

# 2. The rest from PyPI. Key libraries are pinned to the dev env versions so the parity test
#    (in-process replay vs adapter server) runs the same numerics everywhere.
RUN pip install \
      numpy==2.5.4 pandas==2.3.3 scipy==1.18.1 scikit-learn==1.9.1 \
      gymnasium==1.4.0 gpytorch==1.15.2 stable-baselines3==2.9.2 \
      pydantic==2.14.2 pyyaml python-dotenv httpx boto3==1.43.123 \
 && python -c "import pathlib, shutil; [shutil.rmtree(d, ignore_errors=True) for d in list(pathlib.Path('/var/lang/lib/python3.12/site-packages').rglob('tests')) if d.is_dir()]"

# 3. The organisers' SDK wheel, from the staging context. --no-deps: everything it needs is above.
COPY wheel/ /tmp/wheel/
RUN pip install --no-deps /tmp/wheel/*.whl && rm -rf /tmp/wheel

# Smoke test assets: a PPO trained for about 1 s on the local twin at build time, so SB3 is proven.
COPY smoke/ ${LAMBDA_TASK_ROOT}/smoke/
RUN python ${LAMBDA_TASK_ROOT}/smoke/train_tiny_ppo.py ${LAMBDA_TASK_ROOT}/smoke/tiny_ppo.zip

# 4. Our code, last. The staging step copies only the packages that exist in the worktree.
COPY src/ ${LAMBDA_TASK_ROOT}/

# Real handler. For the smoke test, override the CMD (invoke_local.sh --smoke does).
CMD ["t3.tools.app.handler"]
