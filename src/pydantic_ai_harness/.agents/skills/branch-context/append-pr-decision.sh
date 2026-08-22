#!/bin/bash
# Append one decision entry to the adjacent pr-decisions.md.
#
# Preferred (named flags -- resistant to arg-order mistakes):
#   append-pr-decision.sh \
#     --title "thread #123: use kwargs not positional" \
#     --decision "Reviewer suggestion adopted: kwargs-only for new param" \
#     --why "Reduces breakage risk at call sites" \
#     --source "https://github.com/owner/repo/pull/4567#discussion_r987654" \
#     [--iter 3] [--supersedes "..."]
#
# Positional (compat):
#   append-pr-decision.sh <title> <decision> <why> <source-url> [iteration] [supersedes]
#
# Field order in the file is always: title, decision, why, source, iter.
# Do NOT pass iteration as the second positional arg -- that mangles the entry.

set -euo pipefail

TITLE=""
DECISION=""
WHY=""
SOURCE=""
ITER="-"
SUPERSEDES=""

usage() {
    cat >&2 <<'EOF'
Usage:
  append-pr-decision.sh --title T --decision D --why W --source URL [--iter N] [--supersedes S]
  append-pr-decision.sh <title> <decision> <why> <source-url> [iteration] [supersedes]

Named flags are preferred. Positional order is title, decision, why, source
(NOT title, iter, decision, ...).
EOF
    exit 1
}

if [[ $# -eq 0 ]]; then
    usage
fi

# Named-flag path if first arg starts with -
if [[ "${1:-}" == -* ]]; then
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help) usage ;;
            --title) TITLE="${2:?}"; shift 2 ;;
            --decision) DECISION="${2:?}"; shift 2 ;;
            --why) WHY="${2:?}"; shift 2 ;;
            --source) SOURCE="${2:?}"; shift 2 ;;
            --iter|--iteration) ITER="${2:?}"; shift 2 ;;
            --supersedes) SUPERSEDES="${2:?}"; shift 2 ;;
            *)
                echo "error: unknown option $1" >&2
                usage
                ;;
        esac
    done
else
    # Named-flag mode triggers only on a leading flag, so `"$title" --decision D ...` parses
    # as positional and writes the literal "--decision" into a field. Fail instead.
    for arg in "$@"; do
        if [[ "$arg" == --* ]]; then
            echo "error: '$arg' looks like a named flag, but this call parsed as positional." >&2
            echo "       Flags are only honoured when the FIRST argument is one." >&2
            echo "       Pass every field as a flag, or none of them." >&2
            usage
        fi
    done

    if [[ $# -lt 4 ]]; then
        usage
    fi
    TITLE="$1"
    DECISION="$2"
    WHY="$3"
    SOURCE="$4"
    ITER="${5:--}"
    SUPERSEDES="${6:-}"
fi

if [[ -z "$TITLE" || -z "$DECISION" || -z "$WHY" || -z "$SOURCE" ]]; then
    echo "error: --title, --decision, --why, and --source are required" >&2
    usage
fi

# Source is the load-bearing field -- an entry whose link doesn't resolve can't be traced back,
# and it's the field prose lands in when positional args go in the wrong order.
case "$SOURCE" in
    http://*|https://*) ;;
    *)
        echo "error: source must be a URL, got: $SOURCE" >&2
        echo "       Prose here usually means the fields landed out of order." >&2
        echo "       Positional order is: title, decision, why, source." >&2
        exit 1
        ;;
esac

# Static helpers can be reached through any harness skill root. Live state stays
# beside the invoked helper, so resolve the directory rather than assuming cwd.
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
FILE="$DIR/pr-decisions.md"
if [ ! -f "$FILE" ]; then
    echo "error: $FILE not found. Initialize branch context before appending a decision." >&2
    exit 1
fi

DATE="$(date -u +%Y-%m-%d)"

ENTRY="
## $DATE · $TITLE · iter $ITER
- Decision: $DECISION
- Why: $WHY
- Source: $SOURCE"
if [ -n "$SUPERSEDES" ]; then
    ENTRY="$ENTRY
- Supersedes: $SUPERSEDES"
fi

if ! printf '%s\n' "$ENTRY" >> "$FILE"; then
    echo "error: could not append decision to $FILE" >&2
    exit 1
fi

echo "Appended decision to $FILE"
