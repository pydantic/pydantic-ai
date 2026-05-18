#!/bin/bash
# Emit JSON describing the branch-context state of a worktree.
#
# Usage:
#   status.sh                      # use $PWD as the worktree
#   status.sh <worktree-path>      # explicit worktree path
#
# Output (JSON, single line):
#   {"initialized": <bool>,
#    "brief_size": <int>,                # bytes; 0 if missing
#    "decisions_count": <int>,           # number of decision entries (## headers w/ ' · ')
#    "last_brief_update": "<ISO ts|>",   # mtime of issue-brief.md
#    "last_decision_at": "<ISO ts|>",    # mtime of pr-decisions.md
#    "worktree": "<path>"}
#
# `initialized` is true iff issue-brief.md exists, is >200 bytes, AND its first
# line isn't the "Issue Brief Template" placeholder.

set -e

WT="${1:-$PWD}"
WT="$(cd "$WT" && pwd)"
DIR="$WT/.claude/skills/branch-context"
BRIEF="$DIR/issue-brief.md"
DEC="$DIR/pr-decisions.md"

initialized=false
brief_size=0
last_brief_update=""
if [ -f "$BRIEF" ]; then
    brief_size=$(wc -c < "$BRIEF" | tr -d ' ')
    last_brief_update=$(date -u -r "$BRIEF" +"%Y-%m-%dT%H:%M:%SZ" 2>/dev/null || echo "")
    first_line=$(head -1 "$BRIEF" 2>/dev/null || echo "")
    if [ "$brief_size" -gt 200 ] && [[ "$first_line" != *"Issue Brief Template"* ]]; then
        initialized=true
    fi
fi

decisions_count=0
last_decision_at=""
if [ -f "$DEC" ]; then
    decisions_count=$(grep -cE "^## .* · " "$DEC" 2>/dev/null || echo 0)
    last_decision_at=$(date -u -r "$DEC" +"%Y-%m-%dT%H:%M:%SZ" 2>/dev/null || echo "")
fi

# Build JSON without jq dependency
printf '{"initialized":%s,"brief_size":%d,"decisions_count":%d,"last_brief_update":"%s","last_decision_at":"%s","worktree":"%s"}\n' \
    "$initialized" "$brief_size" "$decisions_count" "$last_brief_update" "$last_decision_at" "$WT"
