"""Fail when a change adds a hand-written resolve-and-confine check to library code.

A function that resolves a path (`realpath`, `resolve`) and compares it with a prefix (`is_relative_to`,
`startswith`, `commonpath`) is a containment check of its own. Confine a path with `confine` in
`pydantic_ai/workspaces/_confine.py`, and give its component a row in `tests/harness/test_rooted_components.py`.

The rule counts such functions in each file, and a change may not raise a file's count. What the base already has
is allowed, so existing checks pass until a change adds one. A function that matches but confines nothing, such as
one that compares a resolved path for another reason, is left out of the count by `# not-confinement: <reason>` on
its `def` line, or alone on the line above. Each such comment the change adds is reported, and one without a
reason fails. A function moved unchanged from one changed file to another is not counted, when the first file's
count falls and the function is not still there, under its own `def` line or renamed.

The rule cannot see a check split across two functions, one resolving and the other comparing.

During a merge, `--staged` reports a file only when its count is higher than in both parents.

Usage:
    uv run scripts/check_confine.py --staged                     # the index against HEAD, for pre-commit
    uv run scripts/check_confine.py --base REF                   # the working tree against REF
"""

from __future__ import annotations

import ast
import re
import sys
import textwrap
import tokenize
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace

from _ratchet import (
    Change,
    Exemption,
    Options,
    added_exemptions,
    added_findings,
    escape,
    exemptions,
    merge_head,
    parse_args,
    read_changes,
    report_exemptions,
    worse_than_both,
)

GUIDE = 'pydantic_ai_slim/pydantic_ai/workspaces/_confine.py'
ROOTS = ('pydantic_ai_slim/pydantic_ai', 'src/pydantic_ai_harness/pydantic_ai_harness')
EXEMPT = frozenset({GUIDE})

_RESOLVES = frozenset({'realpath', 'resolve'})
_COMPARES = frozenset({'is_relative_to', 'startswith', 'commonpath'})
_NOT_CONFINEMENT = re.compile(r'#[ \t]*not-confinement[ \t]*:(?P<reason>.*)')
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


@dataclass
class _Finding:
    line: int
    name: str
    text: str
    """The `def` line, stripped, which pairs it across the two sides of a file."""
    source: str
    """The whole function, dedented, which pairs it with the same function removed from another file."""

    @property
    def body(self) -> str:
        """The function less its `def` line, which pairs it with itself renamed in place."""
        return self.source.partition('\n')[2]


@dataclass
class _Scan:
    findings: list[_Finding] = field(default_factory=list[_Finding])
    exemptions: list[Exemption] = field(default_factory=list[Exemption])


@dataclass
class _Growth:
    path: str
    old: int
    new: int
    added: list[_Finding]
    """The findings the change added to this file, without those that moved here from another changed file."""


def _own_calls(function: ast.FunctionDef | ast.AsyncFunctionDef) -> Iterator[str]:
    """The names called in `function`'s own body, not in the functions, classes and lambdas it defines."""
    pending: list[ast.AST] = list(function.body)
    while pending:
        node = pending.pop()
        if isinstance(node, _SCOPES):
            continue
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                yield node.func.attr
            elif isinstance(node.func, ast.Name):
                yield node.func.id
        pending.extend(ast.iter_child_nodes(node))


def _scan(source: str | None, path: str) -> _Scan:
    if source is None or path in EXEMPT:
        return _Scan()
    try:
        tree = ast.parse(source)
        marked = exemptions(source, _NOT_CONFINEMENT)
    except (SyntaxError, tokenize.TokenError):
        return _Scan()
    exempt = {e.line for e in marked}
    lines = source.splitlines()
    findings: list[_Finding] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.lineno in exempt:
            continue
        calls = set(_own_calls(node))
        if calls & _RESOLVES and calls & _COMPARES:
            body = textwrap.dedent('\n'.join(lines[node.lineno - 1 : node.end_lineno]))
            findings.append(_Finding(node.lineno, node.name, lines[node.lineno - 1].strip(), body))
    return _Scan(sorted(findings, key=lambda f: f.line), marked)


def _growth(changes: list[Change]) -> tuple[list[_Growth], list[tuple[str, Exemption]]]:
    """Each file a change raised the count of, and each `# not-confinement` comment it added."""
    scans = [(c, _scan(c.old_source, c.old_path or ''), _scan(c.new_source, c.new_path or '')) for c in changes]
    # A function moves out of a file only when the file's count falls and nothing on its new side keeps the function's
    # `def` line or its body: one edited or renamed in place has not moved.
    moves: Counter[str] = Counter()
    for _, old, new in scans:
        budget = len(old.findings) - len(new.findings)
        lines = Counter(f.text for f in new.findings)
        bodies = Counter(f.body for f in new.findings)
        for f in old.findings:
            if lines[f.text]:
                lines[f.text] -= 1
            elif bodies[f.body]:
                bodies[f.body] -= 1
            elif budget > 0:
                budget -= 1
                moves[f.source] += 1
    growth: list[_Growth] = []
    added: list[tuple[str, Exemption]] = []
    for change, old, new in scans:
        path = change.new_path or ''
        added += added_exemptions(path, old.exemptions, new.exemptions)
        found = added_findings(new.findings, old.findings, moves, key=lambda f: f.text, moved_key=lambda f: f.source)
        if found is not None:
            growth.append(_Growth(path, len(old.findings), len(new.findings), found))
    return growth, added


def _print_growth(g: _Growth, github: bool) -> None:
    print(f'{g.path}: hand-written confine checks {g.old} -> {g.new}')
    for f in g.added:
        message = (
            f'`{f.name}` resolves a path and compares it with a prefix: confine it with `confine` from {GUIDE}, '
            'or mark it `# not-confinement: <reason>`'
        )
        print(f'  {g.path}:{f.line}: {message}')
        if github:
            print(
                f'::error file={escape(g.path, prop=True)},line={f.line},title=hand-written confine::{escape(message)}'
            )


def _check(options: Options) -> tuple[list[_Growth], list[tuple[str, Exemption]]]:
    return _growth(read_changes(options, ROOTS))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the check and return the exit code."""
    options = parse_args(argv, 'Check the hand-written resolve-and-confine checks a change adds.')
    growth, added = _check(options)
    theirs = merge_head() if options.staged else None
    if theirs:
        their_growth, _ = _check(replace(options, base=theirs))
        growth = worse_than_both(growth, their_growth, key=lambda g: g.path, before=lambda g: g.old)
    unexplained = report_exemptions(
        added,
        label='not confinement',
        missing='give a reason',
        title='not-confinement',
        guide=GUIDE,
        github=options.github,
    )
    for g in growth:
        _print_growth(g, options.github)
    return 1 if growth or unexplained else 0


if __name__ == '__main__':
    sys.exit(main())
