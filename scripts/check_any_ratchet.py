"""Fail when a change adds an `Any` or a `cast()` to a file of Pydantic AI Harness or its tests.

`src/pydantic_ai_harness/AGENTS.md` asks for no `Any` and no casts. Existing uses are pervasive, so each rule
counts what it finds in each file, and a change may not raise a file's count. What the base already has is
allowed, so existing uses pass until a change adds to them. A use that moved between changed files, the same line
removed from one and added to another, is not counted.

- `any`: a reference to `typing.Any`, by name or as an attribute of `typing` or `typing_extensions`. String
  annotations are not read.
- `cast`: a call of `typing.cast`, by name or as an attribute of `typing` or `typing_extensions`.

During a merge, `--staged` reports a rule in a file only when its count is higher than in both parents.

Usage:
    uv run scripts/check_any_ratchet.py --staged                     # the index against HEAD, for pre-commit
    uv run scripts/check_any_ratchet.py --base REF                   # the working tree against REF
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Sequence

from _ratchet import Finding, parse_args, print_growth, ratchet_growth

GUIDE = 'src/pydantic_ai_harness/AGENTS.md'
ROOTS = ('src/pydantic_ai_harness', 'tests/harness')

_TYPING_MODULES = frozenset({'typing', 'typing_extensions'})
_ADVICE = {
    'any': 'type the value precisely, with the type it has, a `TypeVar`, a `Protocol` or a `TypedDict`',
    'cast': 'narrow the value with `isinstance` or a type guard, or fix the type that made the cast necessary',
}


def _is_typing_name(node: ast.AST, name: str) -> bool:
    match node:
        case ast.Name(id=found):
            return found == name
        case ast.Attribute(attr=found, value=ast.Name(id=module)):
            return found == name and module in _TYPING_MODULES
        case _:
            return False


def _scan(source: str | None) -> dict[str, list[Finding]]:
    if source is None:
        return {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    lines = source.splitlines()
    found: dict[str, list[Finding]] = {'any': [], 'cast': []}
    for node in ast.walk(tree):
        if not isinstance(node, ast.expr):
            continue
        if _is_typing_name(node, 'Any'):
            rule = 'any'
        elif isinstance(node, ast.Call) and _is_typing_name(node.func, 'cast'):
            rule = 'cast'
        else:
            continue
        found[rule].append(Finding(node.lineno, lines[node.lineno - 1].strip()))
    return {rule: findings for rule, findings in found.items() if findings}


def main(argv: Sequence[str] | None = None) -> int:
    """Run the check and return the exit code."""
    options = parse_args(argv, 'Check the `Any` references and `cast()` calls a change adds to the harness.')
    growth = ratchet_growth(options, ROOTS, _scan)
    print_growth(growth, _ADVICE, 'Any ratchet', GUIDE, options.github)
    return 1 if growth else 0


if __name__ == '__main__':
    sys.exit(main())
