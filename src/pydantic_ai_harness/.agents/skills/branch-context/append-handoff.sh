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
# One handoff per session, newest last. The reader reads the last entry in the
# index. To revise a handoff you already wrote this session, edit the file in
# place rather than calling this again.
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

TS="$(date -u +%Y-%m-%dT%H%MZ)"
# slug: first ~40 chars of summary, safe filename
SLUG="$(printf '%s' "$SUMMARY" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//; s/-+/-/g' | cut -c1-40)"
[ -z "$SLUG" ] && SLUG="handoff"
FNAME="${TS}-${SLUG}.md"
DEST="$HAND_DIR/$FNAME"

if [ -e "$DEST" ]; then
    # collision within the same minute -- add seconds
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
- TODO -- quote VERBATIM, never paraphrase: user constraints not already in issue-brief.md
  (modality matters: "always X unless Y" != "always X"), and promises you made and have not
  kept, including ones made on the record in a PR/review comment. Write "none" only after
  actually sweeping the session for both.

## Key paths
- TODO

## Open questions
- none

## Branch-context pointers
- Brief: issue-brief.md -- still valid?
- Decisions appended this session: none
- Related plan file (if any): none
EOF
fi

{
    echo ""
    echo "## $TS · handoffs/$FNAME · [$WRITER] $SUMMARY"
} >> "$INDEX"

echo "Appended handoff index entry → $INDEX" >&2
echo "Handoff file: $DEST" >&2
# stdout: path only (for scripting)
echo "$DEST"
