#!/usr/bin/env bash
# SessionStart hook for pydantic-ai.
#
# Behavior:
# - `gh` CLI not installed         -> inject setup-required.md
# - `gh` installed but not authed  -> inject setup-required.md
# - `gh` authed, maintainer login  -> silent exit (lean AGENTS.md only)
# - `gh` authed, other login       -> inject contributor-context.md
set -euo pipefail

MAINTAINERS=(DouweM samuelcolvin Kludex dmontag dsfaccini alexmojaki adtyavrdhn)

inject() {
  local file="$1"
  [[ -f "$file" ]] || exit 0
  jq -cn --rawfile ctx "$file" '{
    hookSpecificOutput: {
      hookEventName: "SessionStart",
      additionalContext: $ctx
    }
  }'
  exit 0
}

hook_dir="$(dirname "$0")"
setup_msg="$hook_dir/setup-required.md"
contributor_msg="$hook_dir/contributor-context.md"

command -v gh >/dev/null 2>&1 || inject "$setup_msg"

user="$(gh api user --jq .login 2>/dev/null || true)"
[[ -z "$user" ]] && inject "$setup_msg"

for m in "${MAINTAINERS[@]}"; do
  [[ "$user" == "$m" ]] && exit 0
done

inject "$contributor_msg"
