"""Scripts that find a sibling difference for the `sibling-handling` gate to judge.

Each locator yields pairs: code the diff added or changed, a sibling that should handle the same thing,
and the difference between them in one sentence. The decision model judges whether the difference
matters; it never searches for one itself.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from .prep import AGENT_INSTRUCTIONS, Function, SourceFile, python_functions

Locator = Literal['a', 'b', 'c', 'd', 'e']

_SIBLINGS_PER_METHOD = 2
_CONTEXT = 3
_DISPATCH_SPAN = 60

_ALTERNATION = re.compile(r'\((?:\?:)?([A-Za-z_]\w*(?:\|[A-Za-z_]\w*)+)\)')
_QUOTED_LIST = re.compile(r"""[\[({]\s*((?:['"]\w+['"]\s*,\s*)+['"]\w+['"])\s*,?\s*[\])}]""")
_QUOTED_UNION = re.compile(r"""^\s*(?:export\s+)?type\s+\w+\s*=\s*((?:['"]\w+['"]\s*\|\s*)+['"]\w+['"])""")
_LITERAL = re.compile(r'Literal\[([^\]]+)\]')
_QUOTED = re.compile(r"""['"](\w+)['"]""")
_DISPATCH = re.compile(r"""\bcase\s+((?:['"]\w+['"]\s*\|?\s*)+)|={2,3}\s*['"](\w+)['"]""")
# A call whose promise is discarded: `void` anywhere on the line, as in a JSX handler, or a bare statement.
_UNHANDLED_CALL = re.compile(r'(?:\bvoid\s+|^\s*)([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)\.\w+\(')
_SHELL_FENCE = re.compile(r'^\s*```(bash|sh|shell|console|zsh)?\s*$')
_INLINE_PLACEHOLDER = re.compile(r'"[^"\n]*(?:<[^<>\n]+>|\$\{?[A-Z_][A-Z0-9_]*\}?|\{\{?[\w.]+\}?\})[^"\n]*"')
_SAFE_ROUTING = re.compile(r"""<<-?\s*['"]\w+['"]|--body-file\b|--input\b|-F\s+\S+=@|\bcat\s+>""")
_RESET_METHOD = re.compile(r'reconnect|reset|resume|^a?close$')


@dataclass(frozen=True)
class Location:
    """Lines `start` to `end` of `path`, at the head unless said otherwise."""

    path: str
    start: int
    end: int


@dataclass(frozen=True)
class SiblingPair:
    """Code the diff touched, a sibling of it, and the difference a script found."""

    locator: Locator
    this: Location
    this_code: str
    sibling: Location
    sibling_code: str
    difference: str


def locate(files: list[SourceFile], root: Path) -> list[SiblingPair]:
    """Every pair the five locators find in the changed files, with unchanged siblings read from `root`."""
    return [
        *sibling_implementations(files, root),
        *enumeration_dispatch(files),
        *rejection_handling(files),
        *value_handling(files),
        *state_reset(files),
    ]


def sibling_implementations(files: list[SourceFile], root: Path) -> Iterator[SiblingPair]:
    """A. A changed method with fewer guard clauses, or fewer caught exceptions, than a same-named sibling."""
    for file in files:
        changed = [f for f in file.changed_functions() if f.owner]
        if not changed:
            continue
        candidates = list(_directory_methods(file, files, root))
        for method in changed:
            guards, excepts = _guards(method.node), _excepts(method.node)
            gaps: list[tuple[int, str, str, Function, str]] = []
            for path, text, sibling in candidates:
                if sibling.node.name != method.node.name or (path, sibling.qualname) == (file.path, method.qualname):
                    continue
                sibling_guards = _guards(sibling.node)
                missing_guards = [g for g in sibling_guards if g not in guards]
                missing_excepts = sorted(_excepts(sibling.node) - excepts)
                if missing_guards and len(sibling_guards) > len(guards):
                    gaps.append(
                        (len(missing_guards), path, text, sibling, f'starts with `if {"`, `if ".join(missing_guards)}`')
                    )
                elif missing_excepts:
                    gaps.append(
                        (len(missing_excepts), path, text, sibling, f'catches `{"`, `".join(missing_excepts)}`')
                    )
            for _, path, text, sibling, what in sorted(gaps, key=lambda gap: -gap[0])[:_SIBLINGS_PER_METHOD]:
                yield SiblingPair(
                    'a',
                    Location(file.path, method.start, method.end),
                    file.lines(method.start, method.end),
                    Location(path, sibling.start, sibling.end),
                    '\n'.join(text.splitlines()[sibling.start - 1 : sibling.end]),
                    f'`{sibling.qualname}` {what}; `{method.qualname}` does not.',
                )


def enumeration_dispatch(files: list[SourceFile]) -> Iterator[SiblingPair]:
    """B. A value the diff adds to an enumeration that the same file's dispatch on it has no branch for."""
    for file in files:
        if file.head is None or file.path.endswith('.md'):
            continue
        branches = _dispatch_branches(file.head)
        handled = {value for value, _ in branches}
        added = file.added
        for number, line in enumerate(file.head.splitlines(), start=1):
            if number not in added:
                continue
            for values in _enumerations(line):
                missing = [v for v in values if v not in handled]
                lines = [n for v, n in branches if v in values]
                if len(values) - len(missing) < 2 or not missing:
                    continue
                start, end = min(lines), min(max(lines), min(lines) + _DISPATCH_SPAN)
                yield SiblingPair(
                    'b',
                    Location(file.path, number, number),
                    file.lines(max(1, number - _CONTEXT), number + _CONTEXT),
                    Location(file.path, start, end),
                    file.lines(start, end),
                    f'The enumeration on line {number} lists `{"`, `".join(missing)}`, which no branch of the '
                    f'dispatch on lines {start}-{end} handles; the dispatch handles '
                    f'`{"`, `".join(v for v in values if v in handled)}`.',
                )


def rejection_handling(files: list[SourceFile]) -> Iterator[SiblingPair]:
    """C. An added TypeScript call left without `.catch` on a client whose other calls in the file have one."""
    for file in files:
        if file.head is None or not file.path.endswith(('.ts', '.tsx')):
            continue
        lines = file.head.splitlines()
        added = file.added
        for number, line in enumerate(lines, start=1):
            match = _UNHANDLED_CALL.search(line)
            statement = ' '.join(lines[number - 1 : number + 2])
            if number not in added or not match or '.catch(' in statement or '.then(' in statement:
                continue
            client = match.group(1)
            # The sibling is a call on the same client, starting on its own line, with `.catch` chained onto it
            # within the next two lines.
            call = re.compile(rf'\b{re.escape(client)}\.\w+\(')
            caught = re.compile(rf'{re.escape(client)}\.\w+\(.*?\)\s*\.catch\(')
            sibling = next(
                (
                    n
                    for n, other in enumerate(lines, start=1)
                    if n != number
                    and (found := call.search(other))
                    and caught.match(' '.join([other[found.start() :], *lines[n : n + 2]]))
                ),
                None,
            )
            if sibling is None:
                continue
            yield SiblingPair(
                'c',
                Location(file.path, number, number),
                file.lines(max(1, number - _CONTEXT), number + _CONTEXT),
                Location(file.path, sibling, sibling),
                file.lines(max(1, sibling - _CONTEXT), sibling + _CONTEXT),
                f'The call to `{client}` on line {number} has no `.catch` and is not awaited; '
                f'the call to `{client}` on line {sibling} handles its rejection.',
            )


def value_handling(files: list[SourceFile]) -> Iterator[SiblingPair]:
    """D. A shell step in agent instructions that double-quotes a placeholder another step routes safely."""
    for file in files:
        if file.head is None or not AGENT_INSTRUCTIONS.search(file.path):
            continue
        blocks = _shell_blocks(file.head)
        safe = [(start, end) for start, end in blocks if _SAFE_ROUTING.search(file.lines(start, end))]
        if not safe:
            continue
        added = file.added
        for start, end in blocks:
            inline = next(
                (
                    (n, found.group(0))
                    for n in range(start, end + 1)
                    if n in added and (found := _INLINE_PLACEHOLDER.search(file.lines(n, n)))
                ),
                None,
            )
            if (start, end) in safe or inline is None:
                continue
            number, placeholder = inline
            safe_start, safe_end = min(safe, key=lambda block: abs(block[0] - number))
            yield SiblingPair(
                'd',
                Location(file.path, start, end),
                file.lines(start, end),
                Location(file.path, safe_start, safe_end),
                file.lines(safe_start, safe_end),
                f'Line {number} passes `{placeholder}` inline in double quotes; the step on lines '
                f'{safe_start}-{safe_end} passes its value through a quoted heredoc or a file.',
            )


def state_reset(files: list[SourceFile]) -> Iterator[SiblingPair]:
    """E. A field the diff adds in `__init__` that the class's reset-like methods never reassign."""
    for file in files:
        if file.head is None or not file.path.endswith('.py'):
            continue
        added = file.added
        methods = file.head_functions()
        for init in (f for f in methods if f.node.name == '__init__' and f.owner):
            resets = [f for f in methods if f.owner == init.owner and _RESET_METHOD.search(f.node.name)]
            if not resets:
                continue
            first_lines = dict(reversed(_self_assignments(init.node)))
            for attribute, line in first_lines.items():
                if line not in added:
                    continue
                stale = [f for f in resets if attribute not in {a for a, _ in _self_assignments(f.node)}]
                if stale:
                    reset = stale[0]
                    yield SiblingPair(
                        'e',
                        Location(file.path, line, line),
                        file.lines(init.start, init.end),
                        Location(file.path, reset.start, reset.end),
                        file.lines(reset.start, reset.end),
                        f'`{init.qualname}` sets `self.{attribute}` on line {line}; '
                        f'`{reset.qualname}` never reassigns it.',
                    )


def _directory_methods(file: SourceFile, files: list[SourceFile], root: Path) -> Iterator[tuple[str, str, Function]]:
    """Methods in every Python file of `file`'s directory: the head of changed files, `root` for the rest."""
    directory = PurePosixPath(file.path).parent
    changed = {f.path: f.head for f in files if f.head is not None}
    local = root / directory
    paths = {str(directory / p.name) for p in local.glob('*.py')} if local.is_dir() else set[str]()
    paths |= {p for p in changed if PurePosixPath(p).parent == directory and p.endswith('.py')}
    for path in sorted(paths):
        text = changed[path] if path in changed else (root / path).read_text(encoding='utf-8')
        yield from ((path, text, f) for f in python_functions(text) if f.owner)


def _guards(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """The conditions of the `if ...: raise` or `if ...: return` statements that open the body."""
    guards: list[str] = []
    first = node.body[0]
    has_docstring = isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
    for statement in node.body[1:] if has_docstring else node.body:
        if not (
            isinstance(statement, ast.If)
            and not statement.orelse
            and isinstance(statement.body[-1], ast.Raise | ast.Return)
        ):
            break
        guards.append(ast.unparse(statement.test))
    return guards


def _excepts(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    return {ast.unparse(h.type) for h in ast.walk(node) if isinstance(h, ast.ExceptHandler) and h.type is not None}


def _self_assignments(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[tuple[str, int]]:
    found: list[tuple[str, int]] = []
    for child in ast.walk(node):
        targets = (
            child.targets
            if isinstance(child, ast.Assign)
            else [child.target]
            if isinstance(child, ast.AnnAssign | ast.AugAssign)
            else []
        )
        found.extend(
            (t.attr, t.lineno)
            for t in targets
            if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == 'self'
        )
    return found


def _enumerations(line: str) -> list[list[str]]:
    """The value sets a line spells out: a regex alternation, a list, tuple or set of strings, or a `Literal`."""
    found = [m.group(1).split('|') for m in _ALTERNATION.finditer(line)]
    found += [_QUOTED.findall(m.group(1)) for m in _QUOTED_LIST.finditer(line)]
    found += [_QUOTED.findall(m.group(1)) for m in _LITERAL.finditer(line)]
    found += [_QUOTED.findall(m.group(1)) for m in _QUOTED_UNION.finditer(line)]
    return [values for values in found if len(values) >= 2]


def _dispatch_branches(text: str) -> list[tuple[str, int]]:
    """Each string a `case` or `==`/`===` comparison in `text` dispatches on, with its line."""
    branches: list[tuple[str, int]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        for match in _DISPATCH.finditer(line):
            values = _QUOTED.findall(match.group(1)) if match.group(1) else [match.group(2)]
            branches.extend((value, number) for value in values)
    return branches


def _shell_blocks(text: str) -> list[tuple[int, int]]:
    """The 1-based line spans of the bodies of fenced shell blocks, an untagged fence counted as shell."""
    blocks: list[tuple[int, int]] = []
    opened: int | None = None
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if opened is None and stripped.startswith('```'):
            opened = number if _SHELL_FENCE.match(line) else -number
        elif opened is not None and stripped == '```':
            if opened > 0 and number - 1 >= opened + 1:
                blocks.append((opened + 1, number - 1))
            opened = None
    return blocks
