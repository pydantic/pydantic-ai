#!/bin/bash
# Append one decision entry to local-notes/branch-context/pr-decisions.md
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

DIR="${BRANCH_CONTEXT_DIR:-local-notes/branch-context}"
FILE="$DIR/pr-decisions.md"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEMPLATE="$SCRIPT_DIR/pr-decisions.template.md"

mkdir -p "$DIR"
if [ ! -f "$FILE" ]; then
    cp "$TEMPLATE" "$FILE"
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
