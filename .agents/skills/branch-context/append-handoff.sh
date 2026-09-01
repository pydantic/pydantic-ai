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
# LANES. Several managers share this worktree. A handoff is written FOR ONE LANE
# and must only ever be read back by that same lane — reading a neighbour's board
# is how a manager adopts PRs it does not drive. The lane is the host conversation
# (`$CLAUDE_CODE_HOST_SESSION_ID` or `$CODEX_THREAD_ID`) when available. Override:
# `$HANDOFF_LANE`. Otherwise it falls back to the tmux session, then to "unlaned"
# (an unlaned handoff is never matched by a reader — deliberately).
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

if [ $# -lt 1 ]; then
    echo "Usage: $0 [--writer <name>] <one-line-summary> [body-file]" >&2
    exit 1
fi

SUMMARY="$1"
BODY_SRC="${2:-}"
SESSION_ID="${HANDOFF_SESSION_ID:-${CLAUDE_CODE_SESSION_ID:-${CODEX_SESSION_ID:-}}}"

DIR=".claude/skills/branch-context"
INDEX="$DIR/handoffs-index.md"
HAND_DIR="$DIR/handoffs"

if [ ! -d "$DIR" ]; then
    echo "error: $DIR not found. Are you at the worktree root?" >&2
    exit 1
fi

mkdir -p "$HAND_DIR"

if [ ! -f "$INDEX" ]; then
    if [ -f "$DIR/handoffs-index.template.md" ]; then
        cp "$DIR/handoffs-index.template.md" "$INDEX"
    else
        printf '# Handoffs index\n\n<!-- entries below, newest at bottom -->\n' > "$INDEX"
    fi
fi

LEDGER="$HAND_DIR/.sessions"
LANES="$HAND_DIR/.lanes"

LANE_ID="${HANDOFF_LANE:-${CLAUDE_CODE_HOST_SESSION_ID:-${CODEX_THREAD_ID:-}}}"
if [ -z "$LANE_ID" ] && [ -n "${TMUX:-}" ]; then
    LANE_ID="tmux-$(tmux display-message -p '#S' 2>/dev/null || echo unknown)"
fi
[ -z "$LANE_ID" ] && LANE_ID="unlaned"

LANE_LABEL=""
if [ -f "$LANES" ]; then
    LANE_LABEL="$(awk -v id="$LANE_ID" '$1 == id { $1 = ""; sub(/^ /, ""); print; exit }' "$LANES")"
fi
if [ -z "$LANE_LABEL" ]; then
    # unnamed lane → short, stable label; David can rename it in .lanes any time
    LANE_LABEL="$(printf '%s' "${LANE_ID#local_}" | cut -c1-8)"
    echo "$LANE_ID $LANE_LABEL" >> "$LANES"
fi

# Same session already handed off → amend that entry instead of competing with it.
if [ -n "$SESSION_ID" ] && [ -f "$LEDGER" ]; then
    PRIOR="$(awk -v s="$SESSION_ID" '$1 == s { $1 = ""; sub(/^ /, ""); print }' "$LEDGER" | tail -1)"
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
        TMP_INDEX="$(mktemp)"
        awk -v f="handoffs/$PRIOR" -v line="## $PRIOR_TS · handoffs/$PRIOR · [$WRITER · lane:$LANE_LABEL] $SUMMARY" \
            'index($0, f) { print line; next } { print }' "$INDEX" > "$TMP_INDEX"
        mv "$TMP_INDEX" "$INDEX"
        echo "Amended this session's existing handoff (one handoff per session) → $INDEX" >&2
        echo "Handoff file: $DEST" >&2
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

if [ -e "$DEST" ]; then
    # collision within the same minute — add seconds
    TS="$(date -u +%Y-%m-%dT%H%M%SZ)"
    FNAME="${TS}-${SLUG}.md"
    DEST="$HAND_DIR/$FNAME"
fi

if [ -n "$BODY_SRC" ]; then
    if [ ! -f "$BODY_SRC" ]; then
        echo "error: body file not found: $BODY_SRC" >&2
        exit 1
    fi
    cp "$BODY_SRC" "$DEST"
else
    cat > "$DEST" <<EOF
# Handoff · ${TS} · ${SUMMARY}

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
    echo "## $TS · handoffs/$FNAME · [$WRITER · lane:$LANE_LABEL] $SUMMARY"
} >> "$INDEX"

if [ -n "$SESSION_ID" ]; then
    echo "$SESSION_ID $FNAME" >> "$LEDGER"
fi

echo "Appended handoff index entry → $INDEX" >&2
echo "Handoff file: $DEST" >&2
# stdout: path only (for scripting)
echo "$DEST"
