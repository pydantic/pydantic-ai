#!/usr/bin/env python
"""Reject a spelling that the glossary's "Spellings" table replaces.

Fenced code blocks, inline code spans and link targets are skipped, so a path, a flag or a shell
variable keeps its spelling.

Usage:
    python scripts/check_glossary_spellings.py [--glossary PATH] FILE...
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_FENCED_BLOCK = re.compile(r'^[ \t]*(`{3,}|~{3,})[^\n]*\n.*?^[ \t]*\1[ \t]*$', re.MULTILINE | re.DOTALL)
# A code span may wrap onto the next line, but never across a blank line.
_CODE_SPAN = re.compile(r'(`+)(?:(?!\n[ \t]*\n)[\s\S])+?\1')
_LINK_TARGET = re.compile(r'\]\([^)\n]*\)')
_SPELLINGS_ROW = re.compile(r'^\|\s*([^|`]+?)\s*\|\s*(`.+`)\s*\|\s*$')


def _blank(match: re.Match[str]) -> str:
    """Blank out a match but keep its newlines, so line numbers still match the file."""
    return re.sub(r'[^\n]', ' ', match[0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--glossary', type=Path, default=Path('agent_docs/GLOSSARY.md'))
    parser.add_argument('files', nargs='*', type=Path)
    args = parser.parse_args()

    rejected: dict[re.Pattern[str], str] = {}
    in_spellings = False
    for line in args.glossary.read_text().splitlines():
        if line.startswith('## '):
            in_spellings = line == '## Spellings'
        elif in_spellings and (row := _SPELLINGS_ROW.match(line)):
            for spelling in re.findall(r'`([^`]+)`', row[2]):
                rejected[re.compile(rf'(?<![\w/-]){re.escape(spelling)}(?![\w/-]|\.\w)', re.IGNORECASE)] = row[1]
    if not rejected:
        print(f'{args.glossary}: no "## Spellings" table rows found', file=sys.stderr)
        return 1

    failed = False
    for path in args.files:
        text = path.read_text()
        for pattern in (_FENCED_BLOCK, _CODE_SPAN, _LINK_TARGET):
            text = pattern.sub(_blank, text)
        for lineno, line in enumerate(text.splitlines(), start=1):
            for pattern, term in rejected.items():
                for match in pattern.finditer(line):
                    print(f'{path}:{lineno}: write "{term}", not "{match[0]}" ({args.glossary} → Spellings)')
                    failed = True
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
