#!/usr/bin/env bash
# Fail if a change touches paths the given stream does not own (scripts/owners.txt).
# usage: scripts/check-owners.sh <stream> [base]
#   paths come from stdin when it is piped:  git diff --name-only m1...HEAD | scripts/check-owners.sh W7b
#   else from:                               git diff --name-only <base>...HEAD   (base defaults to main)
# Stream names: W1 W2 W3 W4 W5 W6 W7a W7b W7c W7d W8 W9 lead; aliases: contracts twin tools suite trust
# evidence c1 trace trustgate scorecard agents models. `nobody` paths fail for every stream but lead.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
owners="$here/owners.txt"
stream="${1:?usage: check-owners.sh <stream> [base]}"
case "$stream" in
  contracts) stream=W1 ;; twin) stream=W2 ;; tools) stream=W3 ;; suite) stream=W4 ;; trust) stream=W5 ;;
  evidence) stream=W6 ;; c1|C1) stream=W7a ;; trace) stream=W7b ;; trustgate) stream=W7c ;;
  scorecard) stream=W7d ;; agents) stream=W8 ;; models) stream=W9 ;;
  v1|V1) stream=W7d ;; v2|V2) stream=W7b ;; v3|V3) stream=W7c ;;  # the review's V-slice names
esac
grep -q "^$stream " "$owners" || { echo "check-owners: unknown stream '$stream' (see $owners)"; exit 2; }

if [ -t 0 ]; then
  paths="$(git diff --name-only "${2:-main}...HEAD")"
else
  paths="$(cat)"
fi

mine=() ; never=()
while read -r s pat; do
  [ -z "${s:-}" ] || [ "${s:0:1}" = "#" ] && continue
  pat="${pat//\*\*/*}"
  [ "$s" = "$stream" ] && mine+=("$pat")
  [ "$s" = nobody ] && never+=("$pat")
done < "$owners"

bad=0 n=0
while IFS= read -r p; do
  [ -z "$p" ] && continue
  n=$((n + 1))
  owned=0
  for pat in "${mine[@]}"; do
    # shellcheck disable=SC2254  # the pattern is meant to glob
    case "$p" in $pat) owned=1; break ;; esac
  done
  if [ "$stream" != lead ]; then
    for pat in "${never[@]}"; do
      # shellcheck disable=SC2254
      case "$p" in $pat) owned=0; break ;; esac
    done
  fi
  if [ "$owned" = 0 ]; then echo "NOT OWNED by $stream: $p"; bad=$((bad + 1)); fi
done <<< "$paths"

# an empty list is never a pass: stdin that is neither a pipe nor a tty (make, CI) reads as empty
if [ "$n" -eq 0 ]; then echo "check-owners: no paths to check (pipe git diff --name-only, or run from a tty)"; exit 2; fi

if [ "$bad" -gt 0 ]; then echo "check-owners: $bad of $n path(s) outside $stream"; exit 1; fi
echo "check-owners: $n path(s), all owned by $stream"
