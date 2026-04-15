#!/usr/bin/env bash
# SessionStart hook. If the current `gh` user is a pydantic-ai maintainer,
# exit silently. Otherwise inject `.claude/hooks/contributor-context.md` as
# additionalContext so the session starts with contributor-facing scrutiny
# rules loaded. Fails open: if `gh` is not authenticated, skip injection.
set -euo pipefail

MAINTAINERS=(DouweM samuelcolvin Kludex dmontag dsfaccini alexmojaki adtyavrdhn)

user="$(gh api user --jq .login 2>/dev/null || true)"
for m in "${MAINTAINERS[@]}"; do
  [[ "$user" == "$m" ]] && exit 0
done

ctx_file="$(dirname "$0")/contributor-context.md"
[[ -f "$ctx_file" ]] || exit 0

jq -cn --rawfile ctx "$ctx_file" '{
  hookSpecificOutput: {
    hookEventName: "SessionStart",
    additionalContext: $ctx
  }
}'
