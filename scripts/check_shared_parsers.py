"""Fail when a change parses a `Retry-After` date, a media type's parameters or a usage count by hand.

Each of these has one shared parser, and each hand-written copy has needed its own fix for the same input:

- `retry-after`: a call to `parsedate_to_datetime` outside `pydantic_ai/exceptions.py`. Read a `Retry-After` value
  with `_parse_retry_after_seconds` and `_parse_retry_after_date` there, and decide at the call site what a negative
  or passed wait means.
- `media-type`: a `split(';')`, `rsplit(';')` or `partition(';')` of a value whose expression names a media type,
  a MIME type or a content type. Drop the parameters with `pydantic_ai._utils.bare_media_type`, and lowercase the
  result where the comparison is case-insensitive.
- `usage-count`: an `isinstance(value, int)` with no `not isinstance(value, bool)` beside it, in a function under
  `pydantic_ai/models/` that builds a `RequestUsage`. `bool` subclasses `int`, so a provider's flag counts as 1:
  use `pydantic_ai._utils.is_usage_count`.

Each rule counts what it finds in each file, and a change may not raise a file's count. What the base already has
is allowed. During a merge, `--staged` reports a rule in a file only when its count is higher than in both parents.
The base-ref comparison is `scripts/_ratchet.py`.

Usage:
    uv run scripts/check_shared_parsers.py --staged                     # the index against HEAD, for pre-commit
    uv run scripts/check_shared_parsers.py --base REF                   # the working tree against REF
"""

from __future__ import annotations

import ast
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Literal

from _ratchet import Change, escape, merge_head, parse_args, read_changes, worse_than_both

ROOTS = ('pydantic_ai_slim', 'pydantic_graph', 'pydantic_evals', 'src', 'clai')

Rule = Literal['retry-after', 'media-type', 'usage-count']

_ADVICE: dict[Rule, str] = {
    'retry-after': (
        'read a `Retry-After` value with `_parse_retry_after_seconds` and `_parse_retry_after_date` in '
        '`pydantic_ai/exceptions.py`, and decide here what a negative or passed wait means'
    ),
    'media-type': (
        'drop the parameters with `pydantic_ai._utils.bare_media_type`, and lowercase the result where the '
        'comparison is case-insensitive'
    ),
    'usage-count': '`bool` subclasses `int`, so a provider flag counts as 1: use `pydantic_ai._utils.is_usage_count`',
}

_RETRY_AFTER_HOME = 'pydantic_ai_slim/pydantic_ai/exceptions.py'
_MEDIA_TYPE_HOME = ('pydantic_ai_slim/pydantic_ai/_utils.py', 'bare_media_type')
_MODELS = 'pydantic_ai_slim/pydantic_ai/models/'
_MEDIA_TYPE_NAME = re.compile(r'(?i)media|mime|content.?type')
_SPLITS = frozenset({'split', 'rsplit', 'partition'})


@dataclass
class _Finding:
    line: int
    text: str


@dataclass
class _Growth:
    path: str
    rule: Rule
    old: int
    new: int
    added: list[_Finding] = field(default_factory=list[_Finding])


class _Finder(ast.NodeVisitor):
    """Collects the findings in one module."""

    def __init__(self, path: str, lines: list[str]) -> None:
        self.path = path
        self.lines = lines
        self.functions: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
        self.found: dict[Rule, list[_Finding]] = {}

    def add(self, rule: Rule, node: ast.expr) -> None:
        self.found.setdefault(rule, []).append(_Finding(node.lineno, self.lines[node.lineno - 1].strip()))

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.functions.append(node)
        self.generic_visit(node)
        self.functions.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.functions.append(node)
        self.generic_visit(node)
        self.functions.pop()

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else None
        if name == 'parsedate_to_datetime' and self.path != _RETRY_AFTER_HOME:
            self.add('retry-after', node)
        elif name in _SPLITS and isinstance(func, ast.Attribute) and self.media_type_split(node, func):
            self.add('media-type', node)
        elif name == 'isinstance' and self.usage_count_filter(node):
            self.add('usage-count', node)
        self.generic_visit(node)

    def media_type_split(self, node: ast.Call, func: ast.Attribute) -> bool:
        separator = node.args[0] if node.args else None
        if not (isinstance(separator, ast.Constant) and separator.value == ';'):
            return False
        if (self.path, self.functions[-1].name if self.functions else None) == _MEDIA_TYPE_HOME:
            return False
        return bool(_MEDIA_TYPE_NAME.search(ast.unparse(func.value)))

    def usage_count_filter(self, node: ast.Call) -> bool:
        if not self.path.startswith(_MODELS) or not self.functions or len(node.args) != 2:
            return False
        if 'RequestUsage' not in ast.unparse(self.functions[-1]):
            return False
        if 'int' not in {n.id for n in ast.walk(node.args[1]) if isinstance(n, ast.Name)}:
            return False
        subject = ast.unparse(node.args[0])
        return not any(_excludes_bool(conjunction, subject) for conjunction in self.conjunctions(node))

    def conjunctions(self, node: ast.Call) -> list[ast.BoolOp]:
        """The `and` expressions in the enclosing function that `node` is a direct operand of."""
        function = self.functions[-1]
        return [
            n
            for n in ast.walk(function)
            if isinstance(n, ast.BoolOp) and isinstance(n.op, ast.And) and any(v is node for v in n.values)
        ]


def _excludes_bool(conjunction: ast.BoolOp, subject: str) -> bool:
    """Whether `conjunction` also says `not isinstance(<subject>, bool)`."""
    for value in conjunction.values:
        if isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.Not) and isinstance(value.operand, ast.Call):
            call = value.operand
            if (
                isinstance(call.func, ast.Name)
                and call.func.id == 'isinstance'
                and len(call.args) == 2
                and ast.unparse(call.args[0]) == subject
                and ast.unparse(call.args[1]) == 'bool'
            ):
                return True
    return False


def _scan(source: str | None, path: str) -> dict[Rule, list[_Finding]]:
    if source is None:
        return {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    finder = _Finder(path, source.splitlines())
    finder.visit(tree)
    return finder.found


def _growth(changes: list[Change]) -> list[_Growth]:
    """Each rule a change raised the count of in a file."""
    growth: list[_Growth] = []
    for change in changes:
        path = change.new_path or ''
        old = _scan(change.old_source, change.old_path or '')
        for rule, found in _scan(change.new_source, path).items():
            was = old.get(rule, [])
            if len(found) > len(was):
                before = {f.text for f in was}
                growth.append(_Growth(path, rule, len(was), len(found), [f for f in found if f.text not in before]))
    return growth


def _print_growth(g: _Growth, github: bool) -> None:
    print(f'{g.path}: {g.rule} {g.old} -> {g.new}')
    for f in g.added:
        print(f'    {g.path}:{f.line}: {f.text}')
        if github:
            title = escape(f'Shared parsers: {g.rule} {g.old} -> {g.new}', prop=True)
            print(f'::error file={escape(g.path, prop=True)},line={f.line},title={title}::{escape(_ADVICE[g.rule])}')


def main(argv: Sequence[str] | None = None) -> int:
    """Run the check and return the exit code."""
    options = parse_args(argv, 'Check that a change parses shared external input through its shared parser.')
    growth = _growth(read_changes(options, ROOTS))
    theirs = merge_head() if options.staged else None
    if theirs:
        their_growth = _growth(read_changes(replace(options, base=theirs), ROOTS))
        growth = worse_than_both(growth, their_growth, key=lambda g: (g.path, g.rule), before=lambda g: g.old)
    for g in growth:
        _print_growth(g, options.github)
    rules: set[Rule] = {g.rule for g in growth}
    if rules:
        print()
    for rule in sorted(rules):
        print(f'{rule}: {_ADVICE[rule]}')
    return 1 if growth else 0


if __name__ == '__main__':
    sys.exit(main())
