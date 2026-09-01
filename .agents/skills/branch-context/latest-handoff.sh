#!/bin/bash
# Print the path of THIS LANE's most recent handoff, or nothing at all.
#
# Usage:
#   latest-handoff.sh          # path on stdout, or empty + a reason on stderr
#   latest-handoff.sh --lane   # print the resolved lane id/label and exit
#
# Several managers share this worktree and one handoffs-index.md. Each handoff
# belongs to the lane that wrote it (the host conversation/thread when the
# harness exposes one). Reading another lane's handoff makes a manager adopt PRs it
# does not drive, so this script matches ONLY the caller's lane and prints
# nothing when the lane has no history. Nothing is the correct answer — do not
# fall back to "the newest handoff overall."

set -e

DIR=".claude/skills/branch-context"
INDEX="$DIR/handoffs-index.md"
LANES="$DIR/handoffs/.lanes"

LANE_ID="${HANDOFF_LANE:-${CLAUDE_CODE_HOST_SESSION_ID:-${CODEX_THREAD_ID:-}}}"
if [ -z "$LANE_ID" ] && [ -n "${TMUX:-}" ]; then
    LANE_ID="tmux-$(tmux display-message -p '#S' 2>/dev/null || echo unknown)"
fi
if [ -z "$LANE_ID" ]; then
    if [ "${1:-}" = "--lane" ]; then
        echo 'lane_id=unavailable label='
    else
        echo 'no lane identity is available — nothing to read' >&2
        echo 'Set HANDOFF_LANE explicitly before reading a handoff in this environment.' >&2
    fi
    exit 0
fi

LANE_LABEL=""
if [ -f "$LANES" ]; then
    LANE_LABEL="$(awk -v id="$LANE_ID" '$1 == id { $1 = ""; sub(/^ /, ""); print; exit }' "$LANES")"
fi
[ -z "$LANE_LABEL" ] && LANE_LABEL="$(printf '%s' "${LANE_ID#local_}" | cut -c1-8)"

if [ "${1:-}" = "--lane" ]; then
    echo "lane_id=$LANE_ID label=$LANE_LABEL"
    exit 0
fi

if [ ! -f "$INDEX" ]; then
    echo "no handoffs-index.md yet — nothing to read" >&2
    exit 0
fi

LINE="$(grep -F "lane:$LANE_LABEL]" "$INDEX" | tail -1 || true)"

if [ -z "$LINE" ]; then
    echo "no handoff for lane '$LANE_LABEL' ($LANE_ID)." >&2
    echo "Other lanes' handoffs exist but are NOT yours — do not read them. Start from the live board instead." >&2
    exit 0
fi

FNAME="$(printf '%s' "$LINE" | sed -E 's|.*(handoffs/[^ ]+\.md).*|\1|')"
if [ ! -f "$DIR/$FNAME" ]; then
    echo "index points at a missing file: $FNAME" >&2
    exit 1
fi

echo "$DIR/$FNAME"
