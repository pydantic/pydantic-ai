#!/bin/bash
# Register one append-only handoff under branch-context.
#
# Usage:
#   append-handoff.sh [--writer <name>] <one-line-summary> [body-file]
#
# - Creates handoffs/YYYY-MM-DDTHHMMZ-<slug>.md (from body-file, or a stub to fill)
# - Appends one line to handoffs-index.md, tagged with the writing skill
# - Never overwrites another session's handoff
#
# LANES. Several agents can share one worktree. A handoff is written FOR ONE LANE
# and must only ever be read back by that same lane — reading a neighbour's board
# is how an agent adopts work it does not drive. The lane is the host conversation
# (`$CLAUDE_CODE_HOST_SESSION_ID` or `$CODEX_THREAD_ID`) when available. Override:
# `$HANDOFF_LANE`. Otherwise it falls back to the tmux session. With no lane at all
# it refuses to write, as latest-handoff.sh refuses to read.
#
# One handoff per session: if this session already registered a handoff, the
# second call AMENDS it (same file, same index line) instead of appending a
# competing entry. Two handoffs from one session make the lane's last line
# ambiguous, and the next agent reads the wrong one.
# Session identity comes from the harness session id (override: $HANDOFF_SESSION_ID).
#
# Prints the handoff path on stdout (last line) so agents can Write/Edit the body.

set -e

WRITER="handoff"
if [ "${1:-}" = "--writer" ]; then
    if [ -z "${2:-}" ]; then
        echo "error: --writer needs a name" >&2
        exit 1
    fi
    WRITER="$2"
    shift 2
fi

USAGE="Usage: $0 [--writer <name>] <one-line-summary> [body-file]

The lane is \$HANDOFF_LANE, else \$CLAUDE_CODE_HOST_SESSION_ID or \$CODEX_THREAD_ID, else the
tmux session. Set HANDOFF_LANE where none of those is set; without a lane this refuses to write."
case "${1:-}" in
    -h|--help)
        echo "$USAGE"
        exit 0
        ;;
esac

if [ $# -lt 1 ]; then
    echo "$USAGE" >&2
    exit 1
fi

SUMMARY="$1"
BODY_SRC="${2:-}"
SESSION_ID="${HANDOFF_SESSION_ID:-${CLAUDE_CODE_SESSION_ID:-${CODEX_SESSION_ID:-}}}"

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE="$(git rev-parse --show-toplevel 2>/dev/null)" || {
    echo 'error: run this helper from inside a git worktree' >&2
    exit 1
}
DIR="$WORKTREE/.claude/skills/branch-context"
INDEX="$DIR/handoffs-index.md"
HAND_DIR="$DIR/handoffs"

if [ ! -d "$DIR" ]; then
    echo "error: branch context not found at $DIR" >&2
    exit 1
fi

validate_index_text() {
    local label="$1" value="$2"
    if [[ "$value" == *$'\n'* || "$value" == *$'\r'* ]] || \
        LC_ALL=C grep -q '[[:cntrl:]]' <<< "$value"; then
        echo "error: $label must not contain control characters" >&2
        exit 1
    fi
    if [[ "$value" == *'\'* ]]; then
        echo "error: $label must not contain backslashes" >&2
        exit 1
    fi
    if ! import_safe "$value"; then
        echo "error: $label must not contain an active @-import" >&2
        exit 1
    fi
}

# One implementation of the import test, in check-autoload-safety.sh. Duplicating
# the pattern here is how the two copies drifted the first time. The check needs a
# file to resolve relative paths against, so hand it one inside the live directory.
import_safe() {
    local value="$1" probe checker rc=0
    checker="$SELF_DIR/check-autoload-safety.sh"
    if [ ! -x "$checker" ]; then
        echo "error: cannot find $checker; refusing to write unchecked text" >&2
        exit 1
    fi
    probe="$(mktemp "$DIR/.autoload-check.XXXXXX")"
    printf '%s\n' "$value" > "$probe"
    "$checker" "$probe" >/dev/null 2>&1 || rc=1
    rm -f "$probe"
    return "$rc"
}

validate_index_text 'writer' "$WRITER"
validate_index_text 'summary' "$SUMMARY"
if [[ "$WRITER" == *']'* || "$WRITER" == *' · '* ]]; then
    echo "error: writer must not contain ']' or the index field separator" >&2
    exit 1
fi

LANE_ID="${HANDOFF_LANE:-${CLAUDE_CODE_HOST_SESSION_ID:-${CODEX_THREAD_ID:-}}}"
if [ -z "$LANE_ID" ] && [ -n "${TMUX:-}" ]; then
    tmux_lane="$(tmux display-message -p '#S' 2>/dev/null || true)"
    [ -n "$tmux_lane" ] && LANE_ID="tmux-$tmux_lane"
fi
if [ -z "$LANE_ID" ]; then
    echo "error: no lane identity. None of HANDOFF_LANE, CLAUDE_CODE_HOST_SESSION_ID, CODEX_THREAD_ID or a tmux session is set." >&2
    echo "Run this again with HANDOFF_LANE=<name>, and give the successor the same value." >&2
    exit 1
fi

mkdir -p "$HAND_DIR"

# The index, lane map, session ledger, and handoff file form one state transition.
# Serialize the complete transition so concurrent first writers cannot split it.
# The lock block below is identical to latest-handoff.sh's; keep them in sync.

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

TMP_INDEX=''
cleanup() {
    if [ -n "$TMP_INDEX" ]; then
        rm -f "$TMP_INDEX"
    fi
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

if [ ! -f "$INDEX" ]; then
    if [ -f "$DIR/handoffs-index.template.md" ]; then
        cp "$DIR/handoffs-index.template.md" "$INDEX"
    else
        printf '# Handoffs index\n\n<!-- entries below, newest at bottom -->\n' > "$INDEX"
    fi
fi

LEDGER="$HAND_DIR/.sessions"
LANES="$HAND_DIR/.lanes"

validate_state_id() {
    local label="$1" value="$2"
    validate_index_text "$label" "$value"
    if [[ "$value" =~ [[:space:]] || "$value" == *']'* || "$value" == *' · '* ]]; then
        echo "error: $label must be a single delimiter-safe field" >&2
        exit 1
    fi
}

validate_state_id 'lane id' "$LANE_ID"
if [ -n "$SESSION_ID" ]; then
    validate_state_id 'session id' "$SESSION_ID"
fi

LANE_LABEL=""
if [ -f "$LANES" ]; then
    LANE_LABEL="$(awk -v id="$LANE_ID" '$1 == id { $1 = ""; sub(/^ /, ""); print; exit }' "$LANES")"
fi
if [ -z "$LANE_LABEL" ]; then
    # unnamed lane → short, stable, unique label; rename it in .lanes any time
    base_label="$(printf '%s' "${LANE_ID#local_}" | cut -c1-8)"
    [ -z "$base_label" ] && base_label='lane'
    LANE_LABEL="$base_label"
    suffix=2
    while awk -v label="$LANE_LABEL" '$1 != "" { $1 = ""; sub(/^ /, ""); if ($0 == label) found = 1 } END { exit !found }' "$LANES" 2>/dev/null; do
        LANE_LABEL="${base_label}-${suffix}"
        suffix=$((suffix + 1))
    done
    echo "$LANE_ID $LANE_LABEL" >> "$LANES"
fi
validate_index_text 'lane label' "$LANE_LABEL"
if [[ "$LANE_LABEL" == *']'* || "$LANE_LABEL" == *' · '* ]]; then
    echo "error: lane label must not contain ']' or the index field separator" >&2
    exit 1
fi

# Same session already handed off → amend that entry instead of competing with it.
if [ -n "$SESSION_ID" ] && [ -f "$LEDGER" ]; then
    PRIOR="$(awk -v lane="$LANE_ID" -v session="$SESSION_ID" \
        '$1 == lane && $2 == session { print $3 }' "$LEDGER" | tail -1)"
    if [ -n "$PRIOR" ] && [ -f "$HAND_DIR/$PRIOR" ]; then
        DEST="$HAND_DIR/$PRIOR"
        if [ -n "$BODY_SRC" ]; then
            if [ ! -f "$BODY_SRC" ]; then
                echo "error: body file not found: $BODY_SRC" >&2
                exit 1
            fi
            cp "$BODY_SRC" "$DEST"
        fi
        PRIOR_TS="$(printf '%s' "$PRIOR" | sed -E 's/^([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]+Z)-.*/\1/')"
        TMP_INDEX="$(mktemp "$DIR/.handoffs-index.tmp.XXXXXX")"
        awk -F ' · ' -v f="handoffs/$PRIOR" -v line="## $PRIOR_TS · handoffs/$PRIOR · [$WRITER · lane-id:$LANE_ID · lane:$LANE_LABEL] $SUMMARY" \
            '$1 ~ /^## / && $2 == f { print line; next } { print }' "$INDEX" > "$TMP_INDEX"
        mv "$TMP_INDEX" "$INDEX"
        TMP_INDEX=''
        echo "Amended this session's existing handoff (one handoff per session) → $INDEX" >&2
        echo "Handoff file: $DEST" >&2
        echo "Successor lane: HANDOFF_LANE=$LANE_ID" >&2
        echo "$DEST"
        exit 0
    fi
fi

TS="$(date -u +%Y-%m-%dT%H%MZ)"
# slug: first ~40 chars of summary, safe filename
SLUG="$(printf '%s' "$SUMMARY" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//; s/-+/-/g' | cut -c1-40)"
[ -z "$SLUG" ] && SLUG="handoff"
FNAME="${TS}-${SLUG}.md"
DEST="$HAND_DIR/$FNAME"

if [ -n "$BODY_SRC" ] && [ ! -f "$BODY_SRC" ]; then
    echo "error: body file not found: $BODY_SRC" >&2
    exit 1
fi

reserve_dest() {
    (set -o noclobber; : > "$1") 2>/dev/null
}

if ! reserve_dest "$DEST"; then
    # A minute-level name already exists. Retry with seconds and then suffixes;
    # noclobber reserves each candidate atomically across concurrent sessions.
    TS="$(date -u +%Y-%m-%dT%H%M%SZ)"
    FNAME="${TS}-${SLUG}.md"
    DEST="$HAND_DIR/$FNAME"
    suffix=2
    while ! reserve_dest "$DEST"; do
        FNAME="${TS}-${SLUG}-${suffix}.md"
        DEST="$HAND_DIR/$FNAME"
        suffix=$((suffix + 1))
    done
fi

if [ -n "$BODY_SRC" ]; then
    cp "$BODY_SRC" "$DEST"
else
    cat > "$DEST" <<EOF
# Handoff · ${TS} · ${SUMMARY}

Before acting, load the skills branch-context SKILL.md step 5 names — always \`i-have-adhd\`.

## Done
- TODO

## Next
- TODO (first item = next agent starts here)

## Commitments & constraints carried forward
- TODO — quote VERBATIM, never paraphrase: user constraints not already in issue-brief.md
  (modality matters: "always X unless Y" != "always X"), and promises you made and have not
  kept, including ones made on the record in a PR/review comment. Write "none" only after
  actually sweeping the session for both.

## Key paths
- TODO

## Open questions
- none

## Branch-context pointers
- Brief: issue-brief.md — still valid?
- Decisions appended this session: none
- Related plan file (if any): none
EOF
fi

{
    echo ""
    echo "## $TS · handoffs/$FNAME · [$WRITER · lane-id:$LANE_ID · lane:$LANE_LABEL] $SUMMARY"
} >> "$INDEX"

if [ -n "$SESSION_ID" ]; then
    echo "$LANE_ID $SESSION_ID $FNAME" >> "$LEDGER"
fi

echo "Appended handoff index entry → $INDEX" >&2
echo "Handoff file: $DEST" >&2
echo "Successor lane: HANDOFF_LANE=$LANE_ID" >&2
# stdout: path only (for scripting)
echo "$DEST"
