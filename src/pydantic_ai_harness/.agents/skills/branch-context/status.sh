#!/bin/bash
# Emit JSON describing the branch-context state of a worktree.
#
# Usage:
#   status.sh                      # use $PWD as the worktree
#   status.sh <worktree-path>      # explicit worktree path
#
# Output (JSON, single line):
#   {"initialized": <bool>,
#    "brief_size": <int>,
#    "decisions_count": <int>,
#    "handoffs_count": <int>,
#    "latest_handoff": "<relative path or empty>",
#    "last_brief_update": "<ISO ts|>",
#    "last_decision_at": "<ISO ts|>",
#    "last_handoff_at": "<ISO ts|>",
#    "worktree": "<path>"}

set -e

WT="${1:-$PWD}"
WT="$(cd "$WT" && pwd)"
DIR="$WT/.claude/skills/branch-context"
BRIEF="$DIR/issue-brief.md"
DEC="$DIR/pr-decisions.md"
INDEX="$DIR/handoffs-index.md"
HAND_DIR="$DIR/handoffs"

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

# Count only entries after the live marker so template examples in fenced
# blocks above "<!-- entries below" don't inflate the totals.
count_after_marker() {
    local file="$1" pattern="$2"
    [ -f "$file" ] || { echo 0; return; }
    # `grep -c` already prints 0 on no match, but exits 1; `|| echo 0` would
    # append a second 0 and the caller's printf then sees "0\n0".
    if grep -q '<!-- entries below' "$file" 2>/dev/null; then
        sed -n '/<!-- entries below/,$p' "$file" | grep -cE "$pattern" 2>/dev/null || true
    else
        grep -cE "$pattern" "$file" 2>/dev/null || true
    fi
}

decisions_count=0
last_decision_at=""
if [ -f "$DEC" ]; then
    decisions_count=$(count_after_marker "$DEC" '^## .* · ')
    last_decision_at=$(date -u -r "$DEC" +"%Y-%m-%dT%H:%M:%SZ" 2>/dev/null || echo "")
fi

handoffs_count=0
latest_handoff=""
last_handoff_at=""
if [ -f "$INDEX" ]; then
    handoffs_count=$(count_after_marker "$INDEX" '^## .* · handoffs/')
    # last matching header line after the marker → extract path
    if grep -q '<!-- entries below' "$INDEX" 2>/dev/null; then
        last_line=$(sed -n '/<!-- entries below/,$p' "$INDEX" | grep -E "^## .* · handoffs/" 2>/dev/null | tail -1 || true)
    else
        last_line=$(grep -E "^## .* · handoffs/" "$INDEX" 2>/dev/null | tail -1 || true)
    fi
    if [ -n "$last_line" ]; then
        # ## TS · handoffs/foo.md · summary
        latest_handoff=$(printf '%s' "$last_line" | sed -E 's/^## [^·]+ · (handoffs\/[^ ]+) · .*/\1/')
        if [ -f "$DIR/$latest_handoff" ]; then
            last_handoff_at=$(date -u -r "$DIR/$latest_handoff" +"%Y-%m-%dT%H:%M:%SZ" 2>/dev/null || echo "")
        fi
    fi
elif [ -d "$HAND_DIR" ]; then
    # fallback: newest file in handoffs/
    newest=$(ls -t "$HAND_DIR"/*.md 2>/dev/null | head -1 || true)
    if [ -n "$newest" ]; then
        handoffs_count=$(ls "$HAND_DIR"/*.md 2>/dev/null | wc -l | tr -d ' ')
        latest_handoff="handoffs/$(basename "$newest")"
        last_handoff_at=$(date -u -r "$newest" +"%Y-%m-%dT%H:%M:%SZ" 2>/dev/null || echo "")
    fi
fi

# JSON-escape latest_handoff (minimal: quotes/backslashes)
latest_handoff_esc=$(printf '%s' "$latest_handoff" | sed 's/\\/\\\\/g; s/"/\\"/g')

printf '{"initialized":%s,"brief_size":%d,"decisions_count":%d,"handoffs_count":%d,"latest_handoff":"%s","last_brief_update":"%s","last_decision_at":"%s","last_handoff_at":"%s","worktree":"%s"}\n' \
    "$initialized" "$brief_size" "$decisions_count" "$handoffs_count" "$latest_handoff_esc" \
    "$last_brief_update" "$last_decision_at" "$last_handoff_at" "$WT"
