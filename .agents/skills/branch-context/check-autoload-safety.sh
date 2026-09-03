#!/bin/bash
# Reject @-import tokens in an autoloaded branch-context file.
#
# The thing that makes an @-token dangerous is that it RESOLVES to a file the
# harness will then read into context. A decorator resolves to nothing, and that
# is the whole difference between `@agent.tool` and `@AGENTS.md`. So the test is
# resolution, not spelling:
#
#   reject  a path-shaped token (a slash, a leading . or ~, an absolute path)
#   reject  a token naming a file that exists beside this file or at the repo root
#   reject  a token ending in a source or document extension, whether or not that
#           file exists yet -- resolution is checked once, and a later commit can
#           add the file an already-written brief points at
#   allow   anything else -- `@agent.tool`, `@dataclass`, `@field_validator`,
#           `@pytest.mark.parametrize`, a bare username
#
# Matching every `@` before a non-space character instead rejects this repo's own
# vocabulary, and the callers' remedy ("write it without the @") would then delete
# the decorator name from a brief about that decorator. Matching extensions ALONE
# goes wrong the other way: an extensionless file is importable too, so a planted
# `@payload` beside the brief would sail through. Both tests, not either.

set -euo pipefail

if [ "$#" -ne 1 ] || [ ! -f "$1" ]; then
    echo "Usage: $0 <autoloaded-markdown-file>" >&2
    exit 1
fi

FILE="$1"
DIR="$(cd "$(dirname "$FILE")" && pwd)"
ROOT="$(git -C "$DIR" rev-parse --show-toplevel 2>/dev/null || printf '%s' "$DIR")"

offenders=""
while IFS=: read -r lineno token; do
    # Markdown puts prose punctuation and closing code fences right after a token.
    token="$(printf '%s' "$token" | sed -E "s/[])}.,;:!?\"'\`*_]+$//")"
    path="${token#@}"
    [ -n "$path" ] || continue

    reason=""
    case "$path" in
        */*|/*|~*|.*) reason="path-shaped" ;;
    esac
    if [ -z "$reason" ] && printf '%s' "$path" | LC_ALL=C grep -qE \
        '\.(md|markdown|txt|rst|json|ya?ml|toml|ini|cfg|conf|env|lock|py|pyi|sh|zsh|bash|ipynb|csv|xml|html?)$'; then
        reason="names a file type an import can expand"
    fi
    if [ -z "$reason" ]; then
        for base in "$DIR" "$ROOT"; do
            if [ -e "$base/$path" ]; then
                reason="resolves to $base/$path"
                break
            fi
        done
    fi

    [ -n "$reason" ] && offenders="${offenders}${lineno}: ${token}  (${reason})
"
done < <(LC_ALL=C grep -on '@[^[:space:]][^[:space:]]*' "$FILE" || true)

if [ -n "$offenders" ]; then
    echo "error: $FILE contains an active @-import from untrusted text:" >&2
    printf '%s' "$offenders" >&2
    echo 'Write the reference without its leading @, then run this check again.' >&2
    exit 1
fi

echo "Autoload safety check passed: $FILE"
