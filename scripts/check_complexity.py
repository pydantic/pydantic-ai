"""Fail when a change makes a function's cognitive complexity exceed the limit, or grow while above it.

Functions already above the limit pass until a change makes them more complex. A function that is new
above the limit fails, unless the same change removed a function of the same name that was at least as
complex (a move), or one of exactly the same complexity (a rename).

During a merge, `--staged` reports a function only when it is worse than in both parents.

Each failure lists complexipy's suggestions for the function and links the cases in `agent_docs/complexity.md`.
The limit and the checked roots come from `[tool.complexipy]` in `pyproject.toml`.

Usage:
    uv run scripts/check_complexity.py --staged                     # the index against HEAD, for pre-commit
    uv run scripts/check_complexity.py --base REF [FILE...]         # the working tree against REF
"""

from __future__ import annotations

import argparse
import ast
import io
import os
import re
import subprocess
import sys
import tokenize
import tomllib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import takewhile
from pathlib import Path
from typing import Literal

from complexipy import FunctionComplexity, code_complexity

GUIDE = 'agent_docs/complexity.md'

# complexipy drops a function for either marker, in any case, with its words whole and separated by spaces or
# tabs only, on its own line above the function or on one of its decorator or signature lines. A marker followed
# by a `[rule, ...]` list only silences those suggestions. Only a `(reason)` directly after the marker counts as
# a reason.
_IGNORE = re.compile(
    r'#[ \t]*(?:complexipy[ \t]*:[ \t]*ignore|noqa[ \t]*:[ \t]*complexipy)\b(?![ \t]*\[)[ \t]*(?:\((?P<reason>[^)]*)\))?',
    re.IGNORECASE,
)

_Failure = Literal['new', 'crossed', 'worsened']

_FAILURES: dict[_Failure, tuple[str, str]] = {
    'new': ('new-function', 'new function over the limit'),
    'crossed': ('crossed-the-limit', 'this change took it over the limit'),
    'worsened': ('existing-debt', 'already over the limit, and this change made it worse'),
}

# complexipy's refactoring rules, each pointed at the fix in the guide that fits this repo.
_RULES = {
    'C001': ('guard-clauses', 'return early for the cases you can rule out, so the main path is not nested'),
    'C002': ('guard-clauses', 'skip unwanted items with `continue` at the top of the loop'),
    'C003': ('extract-a-method', 'move the block into a private method, or a function next to its caller'),
    'C004': ('dispatch-on-type', 'turn the `elif` chain into `match`, with one method per branch'),
    'C005': ('extract-a-method', 'name the compound condition as a predicate method'),
    'C007': ('merge-nested-ifs', 'merge the nested `if`s into one `if a and b:`'),
    'C011': ('flatten-nested-try', 'move the inner `try` into its own method'),
}

_DEF = re.compile(r'\s*(?:async\s+)?def\s')
_ROLE_DECORATOR = re.compile(r'\s*@(?:\w+\.)*(overload|setter|deleter)\s*(?:#.*)?$')


@dataclass
class _Options:
    base: str
    staged: bool
    files: list[str]
    github: bool


@dataclass
class _Change:
    old_path: str | None
    new_path: str | None
    old_source: str | None = None
    new_source: str | None = None


@dataclass
class _Offender:
    path: str
    function: FunctionComplexity
    old: int | None
    failure: _Failure


@dataclass
class _Removed:
    path: str
    name: str
    complexity: int


def _git(*args: str, stdin: bytes | None = None) -> bytes:
    return subprocess.run(('git', *args), input=stdin, capture_output=True, check=True).stdout


def _read_objects(specs: Sequence[str]) -> list[str]:
    """Read `rev:path` blobs through one `git cat-file --batch`."""
    if not specs:
        return []
    out = _git('cat-file', '--batch', stdin=''.join(f'{spec}\n' for spec in specs).encode())
    blobs: list[str] = []
    pos = 0
    for _ in specs:
        end = out.index(b'\n', pos)
        size = int(out[pos:end].split()[2])
        blobs.append(out[end + 1 : end + 1 + size].decode('utf-8', errors='replace'))
        pos = end + 1 + size + 1
    return blobs


def _changed_files(revisions: Sequence[str], roots: Sequence[str]) -> list[_Change]:
    """Every changed `.py` file under `roots`, with renames paired."""
    fields = _git('diff', '--name-status', '-z', '-M', *revisions, '--', *roots).decode().split('\0')
    changes: list[_Change] = []
    i = 0
    while i < len(fields) - 1:
        status, path = fields[i], fields[i + 1]
        if status.startswith('R'):
            changes.append(_Change(path, fields[i + 2]))
            i += 3
            continue
        changes.append(_Change(None if status == 'A' else path, None if status == 'D' else path))
        i += 2
    return [c for c in changes if (c.new_path or c.old_path or '').endswith('.py')]


def _untracked_files(roots: Sequence[str]) -> list[_Change]:
    paths = _git('ls-files', '--others', '--exclude-standard', '-z', '--', *roots).decode().split('\0')
    return [_Change(None, path) for path in paths if path.endswith('.py')]


def _select(changes: list[_Change], only: Sequence[str]) -> list[_Change]:
    if not only:
        return changes
    wanted = {os.path.relpath(path) for path in only}
    return [c for c in changes if c.new_path in wanted or c.old_path in wanted]


def _load(changes: list[_Change], base: str, staged: bool) -> None:
    """Read both sides of each change, the new side from the index when `staged`, else from the working tree."""
    olds = iter(_read_objects([f'{base}:{c.old_path}' for c in changes if c.old_path]))
    news = iter(_read_objects([f':{c.new_path}' for c in changes if c.new_path] if staged else []))
    for c in changes:
        c.old_source = next(olds) if c.old_path else None
        if c.new_path:
            c.new_source = next(news) if staged else Path(c.new_path).read_text(encoding='utf-8')


def _functions(source: str | None, *, no_ignore: bool = False) -> dict[str, FunctionComplexity]:
    """Each function by name, with a property's setter and deleter keyed apart and `@overload` stubs left out."""
    try:
        found = code_complexity(source, no_ignore=no_ignore).functions if source is not None else []
    except ValueError:
        found = []
    repeated = {name for name, count in Counter(function.name for function in found).items() if count > 1}
    lines = (source or '').splitlines() if repeated else []
    by_name: dict[str, FunctionComplexity] = {}
    for function in found:
        role = _decorator_role(lines, function) if function.name in repeated else ''
        if role == 'overload':
            continue
        key = f'{function.name}.{role}' if role else function.name
        # Any other repeated name keeps its most complex definition.
        if key not in by_name or function.complexity > by_name[key].complexity:
            by_name[key] = function
    return by_name


def _decorator_role(lines: list[str], function: FunctionComplexity) -> str:
    """`overload`, `setter` or `deleter` when one of the function's decorators makes it one, else `''`."""
    decorators = takewhile(lambda line: not _DEF.match(line), lines[function.line_start - 1 :])
    return next((role[1] for line in decorators if (role := _ROLE_DECORATOR.match(line))), '')


def _classify(before: int | None, after: int, limit: int) -> _Failure | None:
    if after <= limit or (before is not None and after <= before):
        return None
    return 'new' if before is None else 'crossed' if before <= limit else 'worsened'


def _offenders_in(change: _Change, limit: int) -> tuple[list[_Offender], list[_Removed]]:
    # The base and the removals are scored with ignores off, so adding or removing an ignore neither hides a
    # function nor makes one look new. Only the change's own result honors them.
    old, new = _functions(change.old_source, no_ignore=True), _functions(change.new_source)
    kept = _functions(change.new_source, no_ignore=True)
    removed = [_Removed(change.old_path or '', f.name, f.complexity) for name, f in old.items() if name not in kept]
    offenders: list[_Offender] = []
    for name, function in new.items():
        before = old[name].complexity if name in old else None
        failure = _classify(before, function.complexity, limit)
        if failure:
            offenders.append(_Offender(change.new_path or '', function, before, failure))
    return offenders, removed


def _leaf(name: str) -> str:
    return name.rsplit('::', 1)[-1]


def _same_name(offender: _Offender, pool: list[_Removed]) -> _Removed | None:
    """The most complex removed function of the same name: a move, or a rename of its class."""
    leaf = _leaf(offender.function.name)
    return max((r for r in pool if _leaf(r.name) == leaf), key=lambda r: r.complexity, default=None)


def _same_score(offender: _Offender, pool: list[_Removed]) -> _Removed | None:
    """A removed function of exactly the same complexity: a rename."""
    return next((r for r in pool if r.complexity == offender.function.complexity), None)


def _predecessors(offenders: list[_Offender], pool: list[_Removed]) -> dict[int, _Removed]:
    """The removed function each new function continues, by offender index.

    Names pair first, so a rename cannot take the predecessor of a move.
    """
    found: dict[int, _Removed] = {}
    for match in (_same_name, _same_score):
        for i, o in enumerate(offenders):
            source = match(o, pool) if o.failure == 'new' and i not in found else None
            if source is not None:
                pool.remove(source)
                found[i] = source
    return found


def _collect(changes: list[_Change], limit: int) -> list[_Offender]:
    """Every function the change made worse, judging a moved or renamed function against its predecessor."""
    offenders: list[_Offender] = []
    pool: list[_Removed] = []
    for change in changes:
        found, removed = _offenders_in(change, limit)
        offenders += found
        pool += removed
    predecessors = _predecessors(offenders, pool)
    kept: list[_Offender] = []
    for i, o in enumerate(offenders):
        source = predecessors.get(i)
        if source is not None:
            failure = _classify(source.complexity, o.function.complexity, limit)
            if failure is None:
                print(f'note: {o.path} {o.function.name} replaces {source.path} {source.name}, so it is not counted')
                continue
            o.old, o.failure = source.complexity, failure
        kept.append(o)
    return kept


def _definitions(source: str) -> dict[tuple[str, int], ast.FunctionDef | ast.AsyncFunctionDef]:
    """The functions complexipy scores, every one not nested in another function, by qualified name and first line."""
    found: dict[tuple[str, int], ast.FunctionDef | ast.AsyncFunctionDef] = {}
    scopes: list[tuple[str, ast.AST]] = [('', ast.parse(source))]
    for prefix, scope in scopes:
        for node in ast.iter_child_nodes(scope):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                first = node.decorator_list[0].lineno if node.decorator_list else node.lineno
                found[(f'{prefix}{node.name}', first)] = node
            else:
                # A class adds to the qualified name; an `if`, `try` or other block does not.
                scopes.append((f'{prefix}{node.name}::' if isinstance(node, ast.ClassDef) else prefix, node))
    return found


def _ignored(source: str | None) -> dict[str, FunctionComplexity]:
    """The functions complexipy drops for an ignore marker, by name."""
    if not source or not _IGNORE.search(source):
        return {}
    scored = _functions(source)
    return {name: f for name, f in _functions(source, no_ignore=True).items() if name not in scored}


def _marker_reasons(source: str) -> dict[int, str]:
    """The reason each ignore comment gives, by line."""
    reasons: dict[int, str] = {}
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        marker = _IGNORE.search(token.string) if token.type == tokenize.COMMENT else None
        if marker:
            reasons[token.start[0]] = (marker['reason'] or '').strip()
    return reasons


@dataclass
class _Ignore:
    path: str
    line: int
    reason: str
    added: bool
    """Whether complexipy scored the function before this change."""


def _ignores(change: _Change) -> list[_Ignore]:
    """Each function complexipy drops in the changed file, with the line and reason of its marker."""
    ignored = _ignored(change.new_source)
    if not ignored:
        return []
    before = _ignored(change.old_source)
    source = change.new_source or ''
    reasons, definitions = _marker_reasons(source), _definitions(source)
    found: list[_Ignore] = []
    for name, function in ignored.items():
        node = definitions[(function.name, function.line_start)]
        # From the line above the first decorator to the end of the signature.
        header = range(function.line_start - 1, max(node.lineno, node.body[0].lineno - 1) + 1)
        line = next((n for n in header if n in reasons), function.line_start)
        found.append(_Ignore(change.new_path or '', line, reasons.get(line, ''), name not in before))
    return found


def _escape(value: str, prop: bool = False) -> str:
    value = value.replace('%', '%25').replace('\r', '%0D').replace('\n', '%0A')
    return value.replace(':', '%3A').replace(',', '%2C') if prop else value


def _advice(o: _Offender) -> list[str]:
    lines: list[str] = []
    for plan in o.function.refactor_plans[:3]:
        anchor, text = _RULES.get(plan.rule_id, ('fixes', plan.title))
        lines.append(
            f'{plan.rule_id} at line {plan.line_start}, about -{plan.estimated_reduction}: {text} ({GUIDE}#{anchor})'
        )
    lines.append(f'pick the case whose shape matches the function, and apply its fix ({GUIDE}#cases)')
    return lines


def _delta(o: _Offender) -> str:
    return f'{o.old} -> {o.function.complexity}' if o.old is not None else str(o.function.complexity)


def _print_offender(o: _Offender, limit: int, github: bool) -> None:
    f = o.function
    anchor, failure = _FAILURES[o.failure]
    print(f'{o.path}:{f.line_start}: {f.name} {_delta(o)}: {failure} ({GUIDE}#{anchor})')
    for line in _advice(o):
        print(f'    {line}')
    if github:
        title = _escape(f'Cognitive complexity {_delta(o)} (limit {limit})', prop=True)
        body = _escape('\n'.join([f'{failure}. See {GUIDE}#{anchor}', *_advice(o)]))
        print(f'::error file={_escape(o.path, prop=True)},line={f.line_start},title={title}::{body}')


def _write_summary(offenders: list[_Offender], limit: int, summary: str) -> None:
    rows = [
        f'### Cognitive complexity (limit {limit})',
        '',
        '| Function | Change | Failure |',
        '|---|---|---|',
    ]
    for o in offenders:
        function = f'`{o.path}:{o.function.line_start}` `{o.function.name}`'
        rows.append(f'| {function} | {_delta(o)} | {_FAILURES[o.failure][1]} |')
    with open(summary, 'a', encoding='utf-8') as out:
        out.write('\n'.join(rows) + '\n')


def _report_ignores(ignores: list[_Ignore], github: bool) -> bool:
    """Print each ignore that gives no reason, and each one the change added; return whether one gives no reason."""
    for ignore in ignores:
        if ignore.reason and not ignore.added:
            continue
        level, message = (
            ('warning', f'ignored: {ignore.reason}') if ignore.reason else ('error', 'give the ignore a reason')
        )
        print(f'{ignore.path}:{ignore.line}: complexipy {message}, see {GUIDE}#ignoring-a-function')
        if github:
            location = f'file={_escape(ignore.path, prop=True)},line={ignore.line}'
            print(f'::{level} {location},title=complexipy ignore::{_escape(message)}')
    return any(not ignore.reason for ignore in ignores)


def _parse_args(argv: Sequence[str] | None) -> _Options:
    parser = argparse.ArgumentParser(description='Check the cognitive complexity a change adds.')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--staged', action='store_true', help='compare the index with HEAD')
    source.add_argument('--base', help='compare the working tree with this revision')
    parser.add_argument('--format', choices=('text', 'github'), default='text')
    parser.add_argument('files', nargs='*', help='only check these paths')
    args = parser.parse_args(argv)
    staged: bool = args.staged
    base: str | None = args.base
    files: list[str] = args.files
    output_format: str = args.format
    # The group is required, so `base` is set whenever `--staged` is not.
    return _Options(base or 'HEAD', staged, files, output_format == 'github')


def _config() -> tuple[int, list[str]]:
    """The limit and the checked roots, from `[tool.complexipy]` in `pyproject.toml`."""
    config = tomllib.loads(Path('pyproject.toml').read_text(encoding='utf-8'))['tool']['complexipy']
    limit = config['max-complexity-allowed']
    if not isinstance(limit, int) or not isinstance(config['paths'], list):
        raise SystemExit('`[tool.complexipy]` needs an integer `max-complexity-allowed` and a list of `paths`')
    roots = [root for root in config['paths'] if isinstance(root, str)]
    if len(roots) != len(config['paths']):
        raise SystemExit('`[tool.complexipy] paths` must be strings')
    return limit, roots


def _changes(options: _Options, roots: Sequence[str]) -> list[_Change]:
    if options.staged:
        return _changed_files(['--cached', options.base], roots)
    return _changed_files([options.base], roots) + _untracked_files(roots)


def _offenders(options: _Options, roots: Sequence[str], limit: int) -> tuple[list[_Change], list[_Offender]]:
    changes = _select(_changes(options, roots), options.files)
    _load(changes, options.base, options.staged)
    return changes, _collect(changes, limit)


def _merge_head() -> str | None:
    """The commit a merge in progress brings in, or `None` outside a merge."""
    found = subprocess.run(('git', 'rev-parse', '-q', '--verify', 'MERGE_HEAD'), capture_output=True, text=True)
    return found.stdout.strip() or None


def main(argv: Sequence[str] | None = None) -> int:
    """Run the check and return the exit code."""
    options = _parse_args(argv)
    limit, roots = _config()

    changes, offenders = _offenders(options, roots, limit)
    merge_head = _merge_head() if options.staged else None
    if merge_head:
        # Compared with `HEAD` alone, a merge would be blamed for every function the merged branch made worse.
        # Each function that is worse than in both parents is reported against the more complex parent.
        _, theirs = _offenders(replace(options, base=merge_head), roots, limit)
        by_name = {(o.path, o.function.name): o for o in theirs}
        offenders = [
            max(o, by_name[o.path, o.function.name], key=lambda offender: offender.old or 0)
            for o in offenders
            if (o.path, o.function.name) in by_name
        ]
    unexplained = _report_ignores([ignore for change in changes for ignore in _ignores(change)], options.github)
    if offenders:
        print(f'\ncognitive complexity limit is {limit}; {len(offenders)} function(s) got worse ({GUIDE}):\n')
    for o in offenders:
        _print_offender(o, limit, options.github)
    summary = os.environ.get('GITHUB_STEP_SUMMARY')
    if options.github and offenders and summary:
        _write_summary(offenders, limit, summary)
    return 1 if offenders or unexplained else 0


if __name__ == '__main__':
    sys.exit(main())
