"""Fail when a change adds a write to a `ToolManager`'s retry records outside `pydantic_ai_slim/pydantic_ai/tool_manager.py`.

A `ToolManager` carries each tool's retry count from one run step to the next. It derives the count from the tools
that failed and succeeded in the step (`failed_tools`, `succeeded_tools`) and the counts in `RunContext.retries`, and
it keeps the run's free availability refusals in `availability_refused`. Code that writes these by hand counts a call
differently from how the manager counts a direct one. A tool that dispatches other tools gets its manager from
`ToolManager.for_nested_calls`, which shares the step's records, and the manager charges the retries itself.

The rule counts the writes in each file under `pydantic_ai_slim/pydantic_ai` and `src`, and a change may not raise a
file's count. What the base already has is allowed, so existing writes pass until a change adds to them. A write is:

- a call of a mutating method on one of those attributes, such as `manager.failed_tools.add(name)`;
- an assignment to one of those attributes or to an item of one, or the deletion of either;
- one of their names passed as a keyword to `ToolManager(...)`, `RunContext(...)` or `replace(...)`.

A write that moved between changed files, the same line removed from one and added to another, is not counted.

During a merge, `--staged` reports a file only when its count is higher than in both parents.

Usage:
    uv run scripts/check_tool_manager_state.py --staged             # the index against HEAD, for pre-commit
    uv run scripts/check_tool_manager_state.py --base REF           # the working tree against REF
"""

from __future__ import annotations

import ast
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace

from _ratchet import Options, escape, merge_head, parse_args, read_changes, worse_than_both

OWNER = 'pydantic_ai_slim/pydantic_ai/tool_manager.py'
ROOTS = ('pydantic_ai_slim/pydantic_ai', 'src')
STATE = frozenset({'failed_tools', 'succeeded_tools', 'availability_refused', 'retries'})

_MUTATORS = frozenset(
    {
        'add',
        'clear',
        'difference_update',
        'discard',
        'intersection_update',
        'pop',
        'popitem',
        'remove',
        'setdefault',
        'symmetric_difference_update',
        'update',
    }
)
_CONSTRUCTORS = frozenset({'ToolManager', 'RunContext', 'replace'})
_ADVICE = (
    f"only `{OWNER}` writes a tool manager's retry records: build a nested manager with "
    '`ToolManager.for_nested_calls` and let it charge retries, or add the operation you need to `ToolManager`'
)


@dataclass
class _Finding:
    line: int
    text: str


@dataclass
class _Growth:
    path: str
    old: int
    new: int
    added: list[_Finding]
    """The writes the change added to this file, without those that moved here from another changed file."""


def _is_state(node: ast.expr) -> bool:
    """Whether `node` is one of the record attributes, or an item of one."""
    target = node.value if isinstance(node, ast.Subscript) else node
    return isinstance(target, ast.Attribute) and target.attr in STATE


def _writes(source: str | None, path: str) -> list[_Finding]:
    """The writes to a tool manager's retry records in a module, in the order `ast.walk` meets them."""
    if source is None or path == OWNER:
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    lines = source.splitlines()
    found: list[_Finding] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Call, ast.Assign, ast.Delete, ast.AugAssign, ast.AnnAssign)):
            continue
        match node:
            case ast.Call(func=ast.Attribute(attr=method, value=receiver)) if method in _MUTATORS:
                writes = _is_state(receiver)
            case ast.Call(func=func):
                callee = func.value if isinstance(func, ast.Subscript) else func
                name = (
                    callee.attr
                    if isinstance(callee, ast.Attribute)
                    else callee.id
                    if isinstance(callee, ast.Name)
                    else None
                )
                writes = name in _CONSTRUCTORS and any(kw.arg in STATE for kw in node.keywords)
            case ast.Assign(targets=targets) | ast.Delete(targets=targets):
                writes = any(_is_state(target) for target in targets)
            case ast.AugAssign(target=target) | ast.AnnAssign(target=target):
                writes = _is_state(target)
        if writes:
            found.append(_Finding(node.lineno, lines[node.lineno - 1].strip()))
    return found


def _texts(findings: list[_Finding]) -> Counter[str]:
    return Counter(f.text for f in findings)


def _check(options: Options) -> list[_Growth]:
    """Each changed file whose count of writes the change raised."""
    scans = [
        (
            change.new_path or '',
            _writes(change.old_source, change.old_path or ''),
            _writes(change.new_source, change.new_path or ''),
        )
        for change in read_changes(options, ROOTS)
    ]
    moves: Counter[str] = Counter()
    for _, old, new in scans:
        moves.update(_texts(old) - _texts(new))
    growth: list[_Growth] = []
    for path, old, new in scans:
        fresh = _texts(new) - _texts(old)
        added: list[_Finding] = []
        moved = 0
        for finding in new:
            if not fresh[finding.text]:
                continue
            fresh[finding.text] -= 1
            if moves[finding.text]:
                moves[finding.text] -= 1
                moved += 1
            else:
                added.append(finding)
        if len(new) - len(old) - moved > 0:
            growth.append(_Growth(path, len(old), len(new), added))
    return growth


def main(argv: Sequence[str] | None = None) -> int:
    """Run the check and return the exit code."""
    options = parse_args(argv, "Check the writes to a tool manager's retry records a change adds.")
    growth = _check(options)
    theirs = merge_head() if options.staged else None
    if theirs:
        growth = worse_than_both(
            growth, _check(replace(options, base=theirs)), key=lambda g: g.path, before=lambda g: g.old
        )
    for g in growth:
        print(f'{g.path}: tool-manager-state {g.old} -> {g.new}')
        for f in g.added:
            print(f'    {g.path}:{f.line}: {f.text}')
            if options.github:
                title = escape(f'Tool manager state: {g.old} -> {g.new}', prop=True)
                print(f'::error file={escape(g.path, prop=True)},line={f.line},title={title}::{escape(_ADVICE)}')
    if growth:
        print()
        print(f'tool-manager-state: {_ADVICE}')
    return 1 if growth else 0


if __name__ == '__main__':
    sys.exit(main())
