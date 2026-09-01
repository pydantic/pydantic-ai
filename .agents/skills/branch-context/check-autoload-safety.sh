#!/bin/bash
# Reject active Claude Code @-import tokens in an autoloaded branch-context file.

set -euo pipefail

if [ "$#" -ne 1 ] || [ ! -f "$1" ]; then
    echo "Usage: $0 <autoloaded-markdown-file>" >&2
    exit 1
fi

FILE="$1"
if matches="$(LC_ALL=C grep -nE '@[^[:space:]]' "$FILE" || true)" && \
    [ -n "$matches" ]; then
    echo "error: $FILE contains active @-import syntax from untrusted text:" >&2
    echo "$matches" >&2
    echo 'Paraphrase the text without @, then run this check again.' >&2
    exit 1
fi

echo "Autoload safety check passed: $FILE"
