#!/bin/bash
# Print the path of THIS LANE's most recent handoff, or nothing at all.
#
# Usage:
#   latest-handoff.sh          # path on stdout, or empty + a reason on stderr
#   latest-handoff.sh --lane   # print the resolved lane id/label and exit
#
# Several agents can share one worktree and one handoffs-index.md. Each handoff
# belongs to the lane that wrote it (the host conversation/thread when the
# harness exposes one). Reading another lane's handoff makes an agent adopt work it
# does not drive, so this script matches ONLY the caller's lane and prints
# nothing when the lane has no history. Nothing is the correct answer — do not
# fall back to "the newest handoff overall."

set -e

WORKTREE="$(git rev-parse --show-toplevel 2>/dev/null)" || {
    echo 'error: run this helper from inside a git worktree' >&2
    exit 1
}
DIR="$WORKTREE/.claude/skills/branch-context"
INDEX="$DIR/handoffs-index.md"
HAND_DIR="$DIR/handoffs"
LANES="$HAND_DIR/.lanes"

LANE_ID="${HANDOFF_LANE:-${CLAUDE_CODE_HOST_SESSION_ID:-${CODEX_THREAD_ID:-}}}"
if [ -z "$LANE_ID" ] && [ -n "${TMUX:-}" ]; then
    tmux_lane="$(tmux display-message -p '#S' 2>/dev/null || true)"
    [ -n "$tmux_lane" ] && LANE_ID="tmux-$tmux_lane"
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

if [[ "$LANE_ID" == *$'\n'* || "$LANE_ID" == *$'\r'* || "$LANE_ID" == *'\'* ]] || \
    [[ "$LANE_ID" =~ [[:space:]] ]] || \
    LC_ALL=C grep -q '[[:cntrl:]]' <<< "$LANE_ID"; then
    echo 'error: lane id must be one line without whitespace, backslashes, or control characters' >&2
    exit 1
fi
if [[ "$LANE_ID" == *']'* || "$LANE_ID" == *' · '* ]]; then
    echo 'error: lane id must not contain index delimiters' >&2
    exit 1
fi

# Readers participate in the writer's lock, so the index, lane map, ledger, and
# handoff file are observed before or after a complete transition, never midway.
# The lock block below is identical to append-handoff.sh's; keep them in sync.
mkdir -p "$HAND_DIR"

# The lock is flock(2) on fd 9, which this shell holds open until it exits. macOS
# ships no flock(1), so perl takes the lock on the inherited descriptor. A flock
# lock belongs to the open file description, so it outlives perl, and the kernel
# drops it once every process holding fd 9 has exited, SIGKILL included. Nothing
# reclaims it, so no contender can remove a live holder's lock. Do not switch to
# fcntl(2) locks: closing any descriptor for the file drops those.
LOCK_FILE="$HAND_DIR/.transition.lock"
if ! command -v perl >/dev/null 2>&1; then
    echo "error: perl is required to take the branch-context handoff lock" >&2
    exit 1
fi
exec 9>>"$LOCK_FILE" || {
    echo "error: cannot open branch-context handoff lock: $LOCK_FILE" >&2
    exit 1
}
lock_rc=0
perl -MFcntl=:flock -e '
    open(my $fh, ">&=", 9) or exit 2;
    for (1 .. 400) {
        exit 0 if flock($fh, LOCK_EX | LOCK_NB);
        $!{EWOULDBLOCK} or exit 2;
        select(undef, undef, undef, 0.05);
    }
    exit 1;
' || lock_rc=$?
if [ "$lock_rc" -eq 1 ]; then
    echo "error: timed out waiting for branch-context handoff lock: $LOCK_FILE" >&2
    exit 1
elif [ "$lock_rc" -ne 0 ]; then
    echo "error: cannot lock branch-context handoff lock: $LOCK_FILE" >&2
    exit 1
fi

LANE_LABEL=""
if [ -f "$LANES" ]; then
    LANE_LABEL="$(awk -v id="$LANE_ID" '$1 == id { $1 = ""; sub(/^ /, ""); print; exit }' "$LANES")"
fi
if [ -z "$LANE_LABEL" ]; then
    if [ "${1:-}" = "--lane" ]; then
        echo "lane_id=$LANE_ID label="
    else
        echo "lane '$LANE_ID' has no registered handoff identity — nothing to read" >&2
    fi
    exit 0
fi
if [[ "$LANE_LABEL" == *$'\n'* || "$LANE_LABEL" == *$'\r'* || "$LANE_LABEL" == *'\'* ]] || \
    LC_ALL=C grep -q '[[:cntrl:]]' <<< "$LANE_LABEL"; then
    echo 'error: lane label must not contain backslashes or control characters' >&2
    exit 1
fi
if [[ "$LANE_LABEL" == *']'* || "$LANE_LABEL" == *' · '* ]]; then
    echo 'error: lane label must not contain index delimiters' >&2
    exit 1
fi

if [ "${1:-}" = "--lane" ]; then
    echo "lane_id=$LANE_ID label=$LANE_LABEL"
    exit 0
fi

if [ ! -f "$INDEX" ]; then
    echo "no handoffs-index.md yet — nothing to read" >&2
    exit 0
fi

legacy_label=''
label_count="$(awk -v label="$LANE_LABEL" '$1 != "" { $1 = ""; sub(/^ /, ""); if ($0 == label) count++ } END { print count+0 }' "$LANES")"
if [ "$label_count" -eq 1 ]; then
    legacy_label="lane:$LANE_LABEL]"
fi

LINE="$(awk -F ' · ' -v wanted="lane-id:$LANE_ID" -v legacy="$legacy_label" '
    /^## / && $2 ~ /^handoffs\/[^ ]+\.md$/ && $3 ~ /^\[/ {
        if ($4 == wanted || (legacy != "" && substr($4, 1, length(legacy)) == legacy)) line = $0
    }
    END { print line }
' "$INDEX")"

if [ -z "$LINE" ]; then
    echo "no handoff for lane '$LANE_LABEL' ($LANE_ID)." >&2
    echo "Other lanes' handoffs exist but are NOT yours — do not read them. Start from the live board instead." >&2
    exit 0
fi

FNAME="$(printf '%s' "$LINE" | awk -F ' · ' '{ print $2 }')"
if [[ ! "$FNAME" =~ ^handoffs/[A-Za-z0-9][A-Za-z0-9_-]*\.md$ ]]; then
    echo "index contains an unsafe handoff path: $FNAME" >&2
    exit 1
fi
if [ -L "$DIR/$FNAME" ]; then
    echo "index points at a symlink, not a branch-local handoff file: $FNAME" >&2
    exit 1
fi
if [ ! -f "$DIR/$FNAME" ]; then
    echo "index points at a missing file: $FNAME" >&2
    exit 1
fi

echo "$DIR/$FNAME"
