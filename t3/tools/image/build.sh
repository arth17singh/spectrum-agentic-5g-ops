#!/usr/bin/env bash
# Build the Team13 heavy-tools Lambda image for linux/arm64. Never pushes.
#
#   t3/tools/image/build.sh [--no-cache]
#
# Stages a temporary build context OUTSIDE the repo containing:
#   wheel/   the organisers' ran_rl wheel (terms forbid redistribution, so it never enters git)
#   smoke/   the smoke handler and its build-time PPO trainer (from this directory)
#   src/     our packages that exist in the repo: twin t2 t1 t3 contracts tools
# then runs `docker build`, prints the image size and removes the staging directory.
#
# Env overrides: RAN_RL_WHEEL (path to the wheel), IMAGE (default team13-ran-tools:dev).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
WHEEL="${RAN_RL_WHEEL:-/Users/satyabratsrikumar/firstmate-homes/AWS_Hackathon/hackathon_docs/ran-rl-starter/ran_rl-1.2.9-py3-none-any.whl}"
IMAGE="${IMAGE:-team13-ran-tools:dev}"

[ -f "$WHEEL" ] || { echo "ran_rl wheel not found: $WHEEL (set RAN_RL_WHEEL)" >&2; exit 1; }

STAGE="$(mktemp -d "${TMPDIR:-/tmp}/team13-ran-tools-ctx.XXXXXX")"
case "$STAGE" in "$REPO"/*) echo "staging dir must be outside the repo" >&2; exit 1;; esac
trap 'rm -rf "$STAGE"' EXIT

mkdir -p "$STAGE/wheel" "$STAGE/smoke" "$STAGE/src"
cp "$WHEEL" "$STAGE/wheel/"
cp "$HERE/smoke_handler.py" "$HERE/train_tiny_ppo.py" "$STAGE/smoke/"
touch "$STAGE/smoke/__init__.py"
cp "$HERE/Dockerfile" "$STAGE/Dockerfile"

for pkg in twin t2 t1 t3 contracts tools; do
  if [ -d "$REPO/$pkg" ]; then
    rsync -a \
      --exclude '__pycache__' --exclude '*.pyc' --exclude 'tests' --exclude '.pytest_cache' \
      --exclude '/tools/image' \
      "$REPO/$pkg/" "$STAGE/src/$pkg/"
    echo "staged package: $pkg"
  else
    echo "skip (not in this worktree yet): $pkg"
  fi
done
# Never let an organiser file ride along with our code.
if find "$STAGE/src" -name 'checkpoint_kit*' -o -name '*.whl' | grep -q .; then
  echo "refusing: organiser artifact found in src staging" >&2; exit 1
fi

echo "building $IMAGE for linux/arm64 ..."
START=$(date +%s)
docker build --platform linux/arm64 --provenance=false "$@" -t "$IMAGE" "$STAGE"
END=$(date +%s)
echo "build time: $((END - START)) s"
echo -n "image size (docker image ls): "
docker image ls --format '{{.Repository}}:{{.Tag}}  disk={{.Size}}' "$IMAGE"
echo -n "image size (content bytes): "
docker image inspect "$IMAGE" --format '{{.Size}}'
