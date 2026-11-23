"""Fail when a change adds a hand-written resolve-and-confine check to library code.

A function that resolves a path (`realpath`, `resolve`) and compares it with a prefix (`is_relative_to`,
`startswith`, `commonpath`) is a containment check of its own. Each such check was written separately, and the
symlink-escape fixes kept landing one check at a time. `confine` in `pydantic_ai/workspaces/_confine.py` is the
one check, and `tests/harness/test_rooted_components.py` holds each rooted component to the same symlink cases.

The rule counts such functions in each file, and a change may not raise a file's count. What the base already has
is allowed, so existing checks pass until a change adds one. A function that matches but confines nothing, such as
one that compares a resolved path for another reason, is left out of the count by `# not-confinement: <reason>` on
its `def` line, or alone on the line above. Each such comment the change adds is reported, and one without a
reason fails. A function moved from one changed file to another, with the same `def` line, is not counted.

The rule cannot see a check split across two functions, one resolving and the other comparing.

During a merge, `--staged` reports a file only when its count is higher than in both parents.

Usage:
    uv run scripts/check_confine.py --staged                     # the index against HEAD, for pre-commit
    uv run scripts/check_confine.py --base REF                   # the working tree against REF
"""

from __future__ import annotations

import ast
import io
import re
import sys
import tokenize
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace

from _ratchet import Change, Options, escape, merge_head, parse_args, read_changes, worse_than_both

GUIDE = 'pydantic_ai_slim/pydantic_ai/workspaces/_confine.py'
ROOTS = ('pydantic_ai_slim/pydantic_ai', 'src/pydantic_ai_harness/pydantic_ai_harness')
EXEMPT = frozenset({GUIDE})

_RESOLVES = frozenset({'realpath', 'resolve'})
_COMPARES = frozenset({'is_relative_to', 'startswith', 'commonpath'})
_ESCAPE = re.compile(r'#[ \t]*not-confinement[ \t]*:(?P<reason>.*)')
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


@dataclass
class _Finding:
    line: int
    name: str
    text: str
    """The `def` line, stripped, which pairs it across the two sides of a change."""


@dataclass
class _Escape:
    line: int
    reason: str
    text: str


@dataclass
class _Scan:
    findings: list[_Finding] = field(default_factory=list[_Finding])
    escapes: list[_Escape] = field(default_factory=list[_Escape])


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


def _escapes(source: str) -> list[_Escape]:
    found: list[_Escape] = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        marker = _ESCAPE.match(token.string) if token.type == tokenize.COMMENT else None
        if marker:
            # A comment alone on its line marks the next line.
            line = token.start[0] + 1 if not token.line[: token.start[1]].strip() else token.start[0]
            found.append(_Escape(line, marker['reason'].strip(), token.line.strip()))
    return found


def _scan(source: str | None, path: str) -> _Scan:
    if source is None or path in EXEMPT:
        return _Scan()
    try:
        tree = ast.parse(source)
        escapes = _escapes(source)
    except (SyntaxError, tokenize.TokenError):
        return _Scan()
    escaped = {e.line for e in escapes}
    lines = source.splitlines()
    findings: list[_Finding] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.lineno in escaped:
            continue
        calls = set(_own_calls(node))
        if calls & _RESOLVES and calls & _COMPARES:
            findings.append(_Finding(node.lineno, node.name, lines[node.lineno - 1].strip()))
    return _Scan(sorted(findings, key=lambda f: f.line), escapes)


def _texts(findings: list[_Finding]) -> Counter[str]:
    return Counter(f.text for f in findings)


def _file_growth(path: str, found: list[_Finding], was: list[_Finding], moves: Counter[str]) -> _Growth | None:
    """How far a change raised a file's count, less the functions that moved in from another changed file."""
    fresh = _texts(found) - _texts(was)
    added: list[_Finding] = []
    moved = 0
    for f in found:
        if not fresh[f.text]:
            continue
        fresh[f.text] -= 1
        if moves[f.text]:
            moves[f.text] -= 1
            moved += 1
        else:
            added.append(f)
    if len(found) - len(was) - moved <= 0:
        return None
    return _Growth(path, len(was), len(found), added)


def _added_escapes(path: str, old: _Scan, new: _Scan) -> list[tuple[str, _Escape]]:
    before = Counter(e.text for e in old.escapes)
    added: list[tuple[str, _Escape]] = []
    for e in new.escapes:
        if before[e.text]:
            before[e.text] -= 1
        else:
            added.append((path, e))
    return added


def _growth(changes: list[Change]) -> tuple[list[_Growth], list[tuple[str, _Escape]]]:
    """Each file a change raised the count of, and each `# not-confinement` comment it added."""
    scans = [(c, _scan(c.old_source, c.old_path or ''), _scan(c.new_source, c.new_path or '')) for c in changes]
    moves: Counter[str] = Counter()
    for _, old, new in scans:
        moves.update(_texts(old.findings) - _texts(new.findings))
    growth: list[_Growth] = []
    escapes: list[tuple[str, _Escape]] = []
    for change, old, new in scans:
        path = change.new_path or ''
        escapes += _added_escapes(path, old, new)
        if grown := _file_growth(path, new.findings, old.findings, moves):
            growth.append(grown)
    return growth, escapes


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


def _report_escapes(escapes: list[tuple[str, _Escape]], github: bool) -> bool:
    """Print each `# not-confinement` comment the change added; return whether one gives no reason."""
    for path, e in escapes:
        level, message = ('warning', f'not confinement: {e.reason}') if e.reason else ('error', 'give a reason')
        print(f'{path}:{e.line}: {message}, see {GUIDE}')
        if github:
            print(f'::{level} file={escape(path, prop=True)},line={e.line},title=not-confinement::{escape(message)}')
    return any(not e.reason for _, e in escapes)


def _check(options: Options) -> tuple[list[_Growth], list[tuple[str, _Escape]]]:
    return _growth(read_changes(options, ROOTS))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the check and return the exit code."""
    options = parse_args(argv, 'Check the hand-written resolve-and-confine checks a change adds.')
    growth, escapes = _check(options)
    theirs = merge_head() if options.staged else None
    if theirs:
        their_growth, _ = _check(replace(options, base=theirs))
        growth = worse_than_both(growth, their_growth, key=lambda g: g.path, before=lambda g: g.old)
    unexplained = _report_escapes(escapes, options.github)
    for g in growth:
        _print_growth(g, options.github)
    return 1 if growth or unexplained else 0


if __name__ == '__main__':
    sys.exit(main())
