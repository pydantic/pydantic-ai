#!/bin/bash
input=$(cat)
command=$(echo "$input" | jq -r '.tool_input.command // ""')

if echo "$command" | grep -qE 'git[[:space:]]+commit' && echo "$command" | grep -qE '(--no-verify|[[:space:]]-[^-[:space:]]*n)'; then
  echo "BLOCKED: Do not skip pre-commit hooks with --no-verify." >&2
  echo "Fix the hook failure instead. If pre-commit fails, investigate and fix the issue." >&2
  exit 1
fi

echo "$input"
exit 0
