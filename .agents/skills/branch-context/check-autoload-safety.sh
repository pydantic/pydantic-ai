#!/bin/bash
# Reject active @-import tokens in an autoloaded branch-context file.
#
# An import needs a PATH after the @, so that is what this matches: a token
# carrying a slash, or one ending in a file extension. Matching every `@` before
# a non-space character instead would reject this repo's own vocabulary --
# `@agent.tool`, `@agent.output_validator`, `@dataclass`, `@field_validator` --
# and the callers' remedy ("write it without the @") would then delete the
# decorator name from a brief about that decorator.

set -euo pipefail

if [ "$#" -ne 1 ] || [ ! -f "$1" ]; then
    echo "Usage: $0 <autoloaded-markdown-file>" >&2
    exit 1
fi

FILE="$1"
EXT='md|markdown|txt|rst|json|ya?ml|toml|ini|cfg|conf|env|py|pyi|sh|zsh|bash|lock'
PATTERN="@[^[:space:]]*/|@[^[:space:]/]*\.($EXT)([^[:alnum:]]|\$)"
if matches="$(LC_ALL=C grep -nE "$PATTERN" "$FILE" || true)" && \
    [ -n "$matches" ]; then
    echo "error: $FILE contains an active @-import path from untrusted text:" >&2
    echo "$matches" >&2
    echo 'Write the path without its leading @, then run this check again.' >&2
    exit 1
fi

echo "Autoload safety check passed: $FILE"
