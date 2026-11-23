"""Fail when a change adds text after a coverage pragma on its line.

`agent_docs/index.md` asks for nothing after a `# pragma: no cover`, `# pragma: lax no cover` or
`# pragma: no branch` marker on its line but tooling directives such as `# pyright: ignore[...]`, with any reason
on its own comment line above. Existing lines with text after the marker are pervasive, so the rule counts what it
finds in each file, and a change may not raise a file's count. What the base already has is allowed, so existing
lines pass until a change adds to them. A line that moved between changed files, the same line removed from one
and added to another, is not counted.

During a merge, `--staged` reports a file only when its count is higher than in both parents.

Usage:
    uv run scripts/check_pragma_comments.py --staged                     # the index against HEAD, for pre-commit
    uv run scripts/check_pragma_comments.py --base REF                   # the working tree against REF
"""

from __future__ import annotations

import io
import re
import sys
import tokenize
from collections.abc import Sequence

from _ratchet import Finding, parse_args, print_growth, ratchet_growth

GUIDE = 'agent_docs/index.md'
ROOTS = ('.',)

_PRAGMA = re.compile(r'#[ \t]*pragma:[ \t]*(?:lax no cover|no cover|no branch)\b(?P<rest>.*)')
# The tooling directives that may follow the marker, each in its own `#` comment.
_DIRECTIVE = re.compile(r'[ \t]*(?:pyright|type|noqa|mypy|ruff|fmt|isort|complexipy|real-wait)\b')
_ADVICE = {
    'pragma-prose': (
        'leave nothing after the marker but tooling directives such as `# pyright: ignore[...]`, '
        'and put the reason on its own comment line above the statement'
    ),
}


def _scan(source: str | None) -> dict[str, list[Finding]]:
    if source is None:
        return {}
    found: list[Finding] = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            marker = _PRAGMA.search(token.string) if token.type == tokenize.COMMENT else None
            if marker is None:
                continue
            before, *directives = marker['rest'].split('#')
            if before.strip() or not all(_DIRECTIVE.match(directive) for directive in directives):
                found.append(Finding(token.start[0], token.line.strip()))
    except (SyntaxError, tokenize.TokenError):
        return {}
    return {'pragma-prose': found} if found else {}


def main(argv: Sequence[str] | None = None) -> int:
    """Run the check and return the exit code."""
    options = parse_args(argv, 'Check the text a change adds after coverage pragmas.')
    growth = ratchet_growth(options, ROOTS, _scan)
    print_growth(growth, _ADVICE, 'Pragma comments', GUIDE, options.github)
    return 1 if growth else 0


if __name__ == '__main__':
    sys.exit(main())
