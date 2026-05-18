#!/bin/bash
# Append one decision entry to .claude/skills/branch-context/pr-decisions.md
#
# Usage:
#   append-pr-decision.sh <title> <decision> <why> <source-url> [iteration] [supersedes]
#
# Example:
#   append-pr-decision.sh "thread #123: use kwargs not positional" \
#     "Reviewer suggestion adopted: kwargs-only for new param" \
#     "Reduces breakage risk at call sites" \
#     "https://github.com/pydantic/pydantic-ai/pull/4567#discussion_r987654" \
#     3

set -e

if [ $# -lt 4 ]; then
    echo "Usage: $0 <title> <decision> <why> <source-url> [iteration] [supersedes]" >&2
    exit 1
fi

TITLE="$1"
DECISION="$2"
WHY="$3"
SOURCE="$4"
ITER="${5:--}"
SUPERSEDES="$6"

# Must run from a worktree root — locate the file relative to cwd.
FILE=".claude/skills/branch-context/pr-decisions.md"
if [ ! -f "$FILE" ]; then
    echo "error: $FILE not found. Are you at the worktree root? Run /initialize-worktree first." >&2
    exit 1
fi

DATE="$(date -u +%Y-%m-%d)"

{
    echo ""
    echo "## $DATE · $TITLE · iter $ITER"
    echo "- Decision: $DECISION"
    echo "- Why: $WHY"
    echo "- Source: $SOURCE"
    [ -n "$SUPERSEDES" ] && echo "- Supersedes: $SUPERSEDES"
} >> "$FILE"

echo "Appended decision to $FILE"
