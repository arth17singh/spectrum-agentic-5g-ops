#!/usr/bin/env bash
# Build the Team13 heavy-tools Lambda image for linux/arm64 FROM A PINNED COMMIT. Never pushes.
#
#   t3/tools/image/build.sh <git sha or ref> [--no-cache]
#
# The code comes from `git archive <sha>`, never from the working tree, so the image is exactly that
# commit. The image is tagged <IMAGE_REPO>:<first 12 hex of the sha> and labelled
# org.opencontainers.image.revision=<full sha>; deploy that tag (register.py refuses :latest).
#
# Stages a temporary build context OUTSIDE the repo containing:
#   wheel/   the organisers' ran_rl wheel (terms forbid redistribution, so it never enters git)
#   smoke/   the smoke handler and its build-time PPO trainer (from the commit)
#   src/     our packages at that commit: twin t2 t1 t3 contracts tools, plus team13_contracts at the
#            top level so `import team13_contracts` works in Lambda
# then runs `docker build`, prints the image size and removes the staging directory.
#
# Env overrides: RAN_RL_WHEEL (path to the wheel), IMAGE_REPO (default team13-ran-tools).
set -euo pipefail

[ $# -ge 1 ] || { echo "usage: $0 <git sha or ref> [docker build args]  (the image is built from that commit)" >&2; exit 2; }
REF="$1"; shift

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
WHEEL="${RAN_RL_WHEEL:-/Users/satyabratsrikumar/firstmate-homes/AWS_Hackathon/hackathon_docs/ran-rl-starter/ran_rl-1.2.9-py3-none-any.whl}"
IMAGE_REPO="${IMAGE_REPO:-team13-ran-tools}"

SHA="$(git -C "$REPO" rev-parse --verify --quiet "${REF}^{commit}")" \
  || { echo "not a commit in this repo: $REF" >&2; exit 2; }
TAG="${SHA:0:12}"
IMAGE="${IMAGE_REPO}:${TAG}"
[ -f "$WHEEL" ] || { echo "ran_rl wheel not found: $WHEEL (set RAN_RL_WHEEL)" >&2; exit 1; }

STAGE="$(mktemp -d "${TMPDIR:-/tmp}/team13-ran-tools-ctx.XXXXXX")"
case "$STAGE" in "$REPO"/*) echo "staging dir must be outside the repo" >&2; exit 1;; esac
trap 'rm -rf "$STAGE"' EXIT

echo "building $IMAGE from commit $SHA (git archive; the working tree is not used)"
mkdir -p "$STAGE/wheel" "$STAGE/smoke" "$STAGE/src"
cp "$WHEEL" "$STAGE/wheel/"
git -C "$REPO" show "$SHA:t3/tools/image/Dockerfile" > "$STAGE/Dockerfile"
git -C "$REPO" show "$SHA:t3/tools/image/smoke_handler.py" > "$STAGE/smoke/smoke_handler.py"
git -C "$REPO" show "$SHA:t3/tools/image/train_tiny_ppo.py" > "$STAGE/smoke/train_tiny_ppo.py"
touch "$STAGE/smoke/__init__.py"

for pkg in twin t2 t1 t3 contracts tools; do
  if git -C "$REPO" cat-file -e "$SHA:$pkg" 2>/dev/null; then
    git -C "$REPO" archive --format=tar "$SHA" "$pkg" | tar -x -C "$STAGE/src"
    echo "staged package: $pkg"
  else
    echo "skip (not in commit $TAG): $pkg"
  fi
done
# Tests, caches and the image tooling itself stay out of the image.
find "$STAGE/src" -type d \( -name tests -o -name __pycache__ -o -name .pytest_cache \) -prune -exec rm -rf {} +
rm -rf "$STAGE/src/t3/tools/image"
# `import team13_contracts` must work in Lambda (the package lives under contracts/ in the repo).
if [ -d "$STAGE/src/contracts/team13_contracts" ]; then
  cp -R "$STAGE/src/contracts/team13_contracts" "$STAGE/src/team13_contracts"
fi
# Never let an organiser file ride along with our code.
if find "$STAGE/src" -name 'checkpoint_kit*' -o -name '*.whl' | grep -q .; then
  echo "refusing: organiser artifact found in src staging" >&2; exit 1
fi
echo "$SHA" > "$STAGE/src/IMAGE_REVISION"

echo "building $IMAGE for linux/arm64 ..."
START=$(date +%s)
docker build --platform linux/arm64 --provenance=false "$@" \
  --label "org.opencontainers.image.revision=$SHA" \
  --label "org.opencontainers.image.title=$IMAGE_REPO" \
  -t "$IMAGE" "$STAGE"
END=$(date +%s)
echo "build time: $((END - START)) s"
echo -n "image size (docker image ls): "
docker image ls --format '{{.Repository}}:{{.Tag}}  disk={{.Size}}' "$IMAGE"
echo -n "image size (content bytes): "
docker image inspect "$IMAGE" --format '{{.Size}}'
echo -n "revision label: "
docker image inspect "$IMAGE" --format '{{ index .Config.Labels "org.opencontainers.image.revision" }}'
echo "built $IMAGE   (push this tag; deploy with register.py --image-uri <ecr>/$IMAGE)"
