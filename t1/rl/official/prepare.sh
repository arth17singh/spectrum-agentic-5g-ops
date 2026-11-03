#!/bin/sh
# Build the SageMaker project for the Track 1 official RL job from the organiser starter (kept OUTSIDE git).
#   t1/rl/official/prepare.sh <starter-dir> <out-dir> [n_steps] [log_std_init]
# Uses the starter's .env (twin URL/token, role, bucket); never prints it. Then, from <out-dir>:
#   make PYTHON=<venv>/bin/python train-dry-run
#   make PYTHON=<venv>/bin/python TIMESTEPS=<n> train-sagemaker     # paid spot job: needs the user's go
# Changes vs the starter: reward v1 (t1/rl/reward_v1.py) as team/rewards.py, training scenarios include
# interference, env backend http (never the synthetic local twin), n_steps and log_std_init as given.
set -eu
S="$1"; O="$2"; NS="${3:-1024}"; LS="${4:--0.5}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$O"
cp -R "$S/config" "$S/team" "$S/Makefile" "$S"/ran_rl-*.whl "$S/.env" "$O/"
chmod 600 "$O/.env"
cp "$HERE/reward_v1.py" "$O/team/rewards.py"
printf 'reward:\n  function: team.rewards:coverage_aware_reward\n  params: {coverage: 2.0, floor: 0.9, floor_weight: 8.0}\n' > "$O/config/reward.yaml"
sed -i.bak 's/^scenarios: \[nominal, congestion, outage\]/scenarios: [nominal, congestion, interference, outage]/; s/^  backend: local/  backend: http/' "$O/config/env.yaml"
sed -i.bak "s/^log_std_init: 0.0 /log_std_init: $LS /; s/^n_steps: 2048/n_steps: $NS/" "$O/config/train.yaml"
rm -f "$O"/config/*.bak
