"""Fail when a change makes a function's cognitive complexity exceed the limit, or grow while above it.

Functions already above the limit pass until a change makes them more complex. A function that is new
above the limit fails, unless the same change removed a function of the same name that was at least as
complex (a move), or one of exactly the same complexity (a rename).

Each failure names the shape of the function, its case in `agent_docs/complexity.md`, and the
complexipy suggestions for it. The limit and the checked roots come from `[tool.complexipy]` in
`pyproject.toml`.

Usage:
    uv run scripts/check_complexity.py --staged                     # the index against HEAD, for pre-commit
    uv run scripts/check_complexity.py --base REF [FILE...]         # the working tree against REF
    uv run scripts/check_complexity.py --base REF --head REF2       # one commit against another
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
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from complexipy import FunctionComplexity, code_complexity

GUIDE = 'agent_docs/complexity.md'

# complexipy drops a function for either marker, in any case and with any text after it, on the line above
# the function or on one of its decorator or signature lines. Only a `(reason)` directly after the marker
# counts as a reason.
_IGNORE = re.compile(
    r'#\s*(?:complexipy\s*:\s*ignore|noqa\s*:\s*complexipy)[^\s(]*\s*(?:\((?P<reason>[^)]*)\))?', re.IGNORECASE
)

_Failure = Literal['new', 'crossed', 'worsened']
_Case = Literal[
    'closure-hosted',
    'orchestration',
    'name-dispatch',
    'stream-state-machine',
    'tree-walk',
    'presentation',
    'vocab-translator',
    'history-pass',
    'part-ladder',
    'precedence-resolution',
    'check-chain',
    'algorithm',
]

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

_CASES: dict[_Case, str] = {
    'closure-hosted': 'hoist the nested function to a method or a module function, passing what it captures',
    'orchestration': 'move the pure steps out, and keep every `yield`, cleanup and cancellation path in place',
    'name-dispatch': 'dispatch with `match` on the key, ending in `case _:` that raises or calls `assert_never`',
    'stream-state-machine': 'keep stream state on fields, and handle each event in a method returning its events',
    'tree-walk': 'separate the traversal from one handler method per node kind',
    'presentation': 'build each row, card or frame in its own method',
    'vocab-translator': '`match` over the closed union, ending in `assert_never`, with one builder per variant',
    'history-pass': 'build an index first, then rebuild with a comprehension and `dataclasses.replace`',
    'part-ladder': '`match` on the part type, and map each part in its own method',
    'precedence-resolution': 'resolve each layer in its own named method',
    'check-chain': 'group the checks into named predicates, one per section',
    'algorithm': 'use guard clauses, and name the conditions as predicates',
}

_PART = re.compile(
    r'(Part|PartDelta|ModelRequest|ModelResponse|ModelMessage|Url|BinaryContent|Block|Param|Content|Item)$'
)
_EVENT = re.compile(r'(Event|Chunk|Delta|Frame)$')
_TOOL = re.compile(r'(Tool|ToolChoice)$')
_NESTED_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
_MESSAGES = re.compile(r'message|parts|history')
_MESSAGE_TYPES = re.compile(r'ModelMessage|ModelRequest|ModelResponse')
_CONCURRENCY_CALLS = frozenset(
    {'create_task', 'create_task_group', 'TaskGroup', 'start_soon', 'gather', 'wait', 'cancel', 'shield', 'CancelScope'}
    | {'move_on_after', 'fail_after', 'ExitStack', 'AsyncExitStack'}
)
_RICH_CALLS = frozenset({'Table', 'Text', 'Panel', 'Group', 'Columns', 'Syntax', 'Markdown', 'add_row', 'add_column'})

_STREAM_OWNER = re.compile(r'StreamedResponse$|Connection$|SessionCore$|EventStream$|Renderer$')
_STREAM_NAME = re.compile(r'event|frame|chunk|^apply$|observe|dispatch|_delta$|_command$|receive|scan')
_DISPATCH_NAME = re.compile(r'^infer_|_command$|handle_key|_route$|_errors?$|_extract$|_translate$')
_TREE_NAME = re.compile(
    r'transform|walk|schema|xml|_gen$|graph|tree|flatten|_expand|_type_to|externalize|_restore|_strip'
)
_PRESENTATION_PATH = re.compile(r'pydantic_clai2/ui/|/reporting/|/_cli/')
_PRESENTATION_NAME = re.compile(r'render|frame|rows|panel|table|menu|picker|choose|completions')
_VOCAB_NAME = re.compile(
    r'tool_choice|native_tools|_config$|_params$|settings|thinking|cache_control|cache_points|options'
)
_HISTORY_NAME = re.compile(
    r'^_?(drop|merge|repair|order|synthesize|move|wrap|replace|prune|clean|narrow|dangling|group|split|strip|'
    r'deliver|update|discovered|reevaluate|trim|find|collect|remove|iter|without|fit|rebuild|is_|visible|include)'
)
_PRECEDENCE_NAME = re.compile(r'__init__|resolve|override|_prepare|_build|get_model_settings|_merge_|configure')


@dataclass
class _Options:
    base: str
    head: str | None
    """A revision, `''` for the index, or `None` for the working tree."""
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
    case: _Case = 'algorithm'


@dataclass
class _Removed:
    path: str
    name: str
    complexity: int


@dataclass
class _Shape:
    """What the case tests read off a function: its names, and tallies of the constructs in its own body."""

    path: str
    owner: str
    name: str
    closure_share: float = 0.0
    is_async_generator: bool = False
    returns_messages: bool = False
    recursive: bool = False
    concurrency: int = 0
    part_types: int = 0
    event_types: int = 0
    tool_types: int = 0
    type_tests: int = 0
    string_tests: int = 0
    ternaries: int = 0
    rich_calls: int = 0
    raises: int = 0
    ifs: int = 0
    top_level_ifs: int = 0
    message_loops: int = 0


# The first case whose test holds names the function, so the narrower shapes come first.
_CASE_TESTS: tuple[tuple[_Case, Callable[[_Shape], bool]], ...] = (
    ('closure-hosted', lambda s: s.closure_share >= 0.5),
    ('orchestration', lambda s: s.concurrency > 0),
    ('name-dispatch', lambda s: bool(_DISPATCH_NAME.search(s.name)) and s.string_tests + s.type_tests >= 4),
    (
        'stream-state-machine',
        lambda s: (
            s.event_types >= 2
            or bool(_STREAM_OWNER.search(s.owner))
            and (s.is_async_generator or bool(_STREAM_NAME.search(s.name)))
        ),
    ),
    ('tree-walk', lambda s: s.recursive or bool(_TREE_NAME.search(s.name))),
    (
        'presentation',
        lambda s: bool(_PRESENTATION_PATH.search(s.path) or _PRESENTATION_NAME.search(s.name)) or s.rich_calls >= 3,
    ),
    ('vocab-translator', lambda s: s.tool_types > 0 or bool(_VOCAB_NAME.search(s.name))),
    (
        'history-pass',
        lambda s: (
            (s.part_types >= 2 or s.message_loops > 0) and (s.returns_messages or bool(_HISTORY_NAME.search(s.name)))
        ),
    ),
    ('part-ladder', lambda s: s.part_types >= 2 or s.message_loops > 0),
    ('name-dispatch', lambda s: s.string_tests >= 5 and s.string_tests >= 0.6 * s.ifs),
    ('precedence-resolution', lambda s: bool(_PRECEDENCE_NAME.search(s.name)) or s.ternaries >= 4),
    ('orchestration', lambda s: s.is_async_generator),
    ('check-chain', lambda s: s.raises >= 2 or s.top_level_ifs >= max(0.4 * s.ifs, 1)),
    ('algorithm', lambda s: True),
)


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


def _load(changes: list[_Change], base: str, head: str | None) -> None:
    """Read both sides of each change; `head` is a revision, `''` for the index, or `None` for the working tree."""
    olds = iter(_read_objects([f'{base}:{c.old_path}' for c in changes if c.old_path]))
    news = iter(_read_objects([f'{head}:{c.new_path}' for c in changes if c.new_path] if head is not None else []))
    for c in changes:
        c.old_source = next(olds) if c.old_path else None
        if c.new_path:
            c.new_source = next(news) if head is not None else Path(c.new_path).read_text(encoding='utf-8')


def _functions(source: str | None, *, no_ignore: bool = False) -> dict[str, FunctionComplexity]:
    """The most complex function per name, so `@overload` stubs do not hide their implementation."""
    by_name: dict[str, FunctionComplexity] = {}
    try:
        found = code_complexity(source, no_ignore=no_ignore).functions if source is not None else []
    except ValueError:
        found = []
    for function in found:
        if function.name not in by_name or function.complexity > by_name[function.name].complexity:
            by_name[function.name] = function
    return by_name


def _classify(before: int | None, after: int, limit: int) -> _Failure | None:
    if after <= limit or (before is not None and after <= before):
        return None
    return 'new' if before is None else 'crossed' if before <= limit else 'worsened'


def _offenders_in(change: _Change, limit: int) -> tuple[list[_Offender], list[_Removed]]:
    # The base and the removals are scored with ignores off, so adding or removing an ignore neither hides a
    # function nor makes one look new. Only the change's own result honors them.
    old, new = _functions(change.old_source, no_ignore=True), _functions(change.new_source)
    kept = _functions(change.new_source, no_ignore=True)
    removed = [_Removed(change.old_path or '', name, f.complexity) for name, f in old.items() if name not in kept]
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


def _type_names(node: ast.expr) -> list[str]:
    match node:
        case ast.Tuple(elts=elts):
            return [name for elt in elts for name in _type_names(elt)]
        case ast.Name(id=name) | ast.Attribute(attr=name):
            return [name]
        case _:
            return []


def _callee(node: ast.expr) -> str:
    match node:
        case ast.Name(id=name) | ast.Attribute(attr=name):
            return name
        case _:
            return ''


def _is_string(node: ast.expr) -> bool:
    match node:
        case ast.Constant(value=str()):
            return True
        case ast.Tuple(elts=[_, *_] as elts) | ast.Set(elts=[_, *_] as elts) | ast.List(elts=[_, *_] as elts):
            return all(_is_string(elt) for elt in elts)
        case _:
            return False


def _own_nodes(node: ast.AST) -> Iterator[ast.AST]:
    """The nodes of a function's body, without the bodies of the functions and classes it defines."""
    for child in ast.iter_child_nodes(node):
        yield child
        if not isinstance(child, (*_NESTED_FUNCTIONS, ast.ClassDef)):
            yield from _own_nodes(child)


def _tally_types(shape: _Shape, types: ast.expr) -> None:
    for name in _type_names(types):
        shape.type_tests += 1
        shape.part_types += bool(_PART.search(name))
        shape.event_types += bool(_EVENT.search(name))
        shape.tool_types += bool(_TOOL.search(name))


def _tally_call(shape: _Shape, call: ast.Call) -> None:
    callee = _callee(call.func)
    match call:
        case ast.Call(func=ast.Name(id='isinstance'), args=[_, types]):
            _tally_types(shape, types)
        case ast.Call(func=ast.Name() | ast.Attribute(value=ast.Name(id='self' | 'cls'))) if callee == shape.name:
            shape.recursive = True
        case ast.Call(func=ast.Attribute(attr='startswith' | 'endswith'), args=[ast.Constant(value=str()), *_]):
            shape.string_tests += 1
        case _:
            shape.concurrency += callee in _CONCURRENCY_CALLS
            shape.rich_calls += callee in _RICH_CALLS


def _tally(shape: _Shape, node: ast.AST) -> None:
    match node:
        case ast.Call():
            _tally_call(shape, node)
        case ast.MatchClass(cls=cls):
            _tally_types(shape, cls)
        case ast.Compare(left=left, comparators=comparators):
            shape.string_tests += sum(_is_string(side) for side in (left, *comparators))
        case ast.Try(finalbody=[_, *_]) | ast.TryStar(finalbody=[_, *_]) | ast.AsyncWith():
            shape.concurrency += 1
        case ast.Yield() | ast.YieldFrom():
            shape.is_async_generator = True
        case ast.For(iter=items) | ast.AsyncFor(iter=items):
            shape.message_loops += bool(_MESSAGES.search(ast.unparse(items)))
        case ast.IfExp():
            shape.ternaries += 1
        case ast.Raise():
            shape.raises += 1
        case ast.If():
            shape.ifs += 1
        case _:
            pass


def _shape(path: str, function: FunctionComplexity, node: ast.FunctionDef | ast.AsyncFunctionDef) -> _Shape:
    owner, _, name = function.name.rpartition('::')
    shape = _Shape(path, owner, name)
    for child in _own_nodes(node):
        _tally(shape, child)
    shape.is_async_generator &= isinstance(node, ast.AsyncFunctionDef)
    shape.returns_messages = node.returns is not None and bool(_MESSAGE_TYPES.search(ast.unparse(node.returns)))
    shape.top_level_ifs = sum(isinstance(statement, ast.If) for statement in node.body)
    nested = [
        (n.lineno, n.end_lineno or n.lineno)
        for n in ast.walk(node)
        if n is not node and isinstance(n, _NESTED_FUNCTIONS)
    ]
    in_nested = sum(lc.complexity for lc in function.line_complexities if any(a <= lc.line <= b for a, b in nested))
    shape.closure_share = in_nested / max(function.complexity, 1)
    return shape


def _case(shape: _Shape) -> _Case:
    return next(case for case, test in _CASE_TESTS if test(shape))


def _definitions(source: str) -> dict[tuple[str, int], ast.FunctionDef | ast.AsyncFunctionDef]:
    """The module- and class-level functions complexipy scores, by qualified name and first line."""
    found: dict[tuple[str, int], ast.FunctionDef | ast.AsyncFunctionDef] = {}
    scopes: list[tuple[str, list[ast.stmt]]] = [('', ast.parse(source).body)]
    for prefix, body in scopes:
        for statement in body:
            if isinstance(statement, ast.ClassDef) and not prefix:
                scopes.append((f'{statement.name}::', statement.body))
            elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                first = statement.decorator_list[0].lineno if statement.decorator_list else statement.lineno
                found[(f'{prefix}{statement.name}', first)] = statement
    return found


def _name_cases(offenders: list[_Offender], changes: list[_Change]) -> None:
    sources = {c.new_path: c.new_source for c in changes if c.new_path and c.new_source}
    definitions = {path: _definitions(sources[path]) for path in {o.path for o in offenders}}
    for o in offenders:
        node = definitions[o.path][(o.function.name, o.function.line_start)]
        o.case = _case(_shape(o.path, o.function, node))


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


def _added_ignores(change: _Change) -> list[tuple[str, int, str]]:
    """Each function this change makes complexipy drop, with the line and reason of its marker."""
    before = _ignored(change.old_source)
    added = [f for name, f in _ignored(change.new_source).items() if name not in before]
    if not added:
        return []
    source = change.new_source or ''
    reasons, definitions = _marker_reasons(source), _definitions(source)
    found: list[tuple[str, int, str]] = []
    for function in added:
        node = definitions[(function.name, function.line_start)]
        # From the line above the first decorator to the end of the signature.
        header = range(function.line_start - 1, max(node.lineno, node.body[0].lineno - 1) + 1)
        line = next((n for n in header if n in reasons), function.line_start)
        found.append((change.new_path or '', line, reasons.get(line, '')))
    return found


def _escape(value: str, prop: bool = False) -> str:
    value = value.replace('%', '%25').replace('\r', '%0D').replace('\n', '%0A')
    return value.replace(':', '%3A').replace(',', '%2C') if prop else value


def _advice(o: _Offender) -> list[str]:
    lines = [f'looks like {o.case}: {_CASES[o.case]} ({GUIDE}#{o.case})']
    for plan in o.function.refactor_plans[:3]:
        anchor, text = _RULES.get(plan.rule_id, ('fixes', plan.title))
        lines.append(
            f'{plan.rule_id} at line {plan.line_start}, about -{plan.estimated_reduction}: {text} ({GUIDE}#{anchor})'
        )
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
        '| Function | Change | Failure | Case |',
        '|---|---|---|---|',
    ]
    for o in offenders:
        function = f'`{o.path}:{o.function.line_start}` `{o.function.name}`'
        rows.append(f'| {function} | {_delta(o)} | {_FAILURES[o.failure][1]} | [{o.case}]({GUIDE}#{o.case}) |')
    with open(summary, 'a', encoding='utf-8') as out:
        out.write('\n'.join(rows) + '\n')


def _report_ignores(added: list[tuple[str, int, str]], github: bool) -> bool:
    """Print each added ignore; return whether one of them gives no reason."""
    for path, line, reason in added:
        level, message = ('warning', f'ignored: {reason}') if reason else ('error', 'give the ignore a reason')
        print(f'{path}:{line}: complexipy {message}, see {GUIDE}#ignoring-a-function')
        if github:
            print(f'::{level} file={_escape(path, prop=True)},line={line},title=complexipy ignore::{_escape(message)}')
    return any(not reason for _, _, reason in added)


def _parse_args(argv: Sequence[str] | None) -> _Options:
    parser = argparse.ArgumentParser(description='Check the cognitive complexity a change adds.')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--staged', action='store_true', help='compare the index with HEAD')
    source.add_argument('--base', help='compare the working tree, or `--head`, with this revision')
    parser.add_argument('--head', help='compare this revision instead of the working tree')
    parser.add_argument('--format', choices=('text', 'github'), default='text')
    parser.add_argument('files', nargs='*', help='only check these paths')
    args = parser.parse_args(argv)
    staged: bool = args.staged
    base: str | None = args.base
    head: str | None = args.head
    files: list[str] = args.files
    output_format: str = args.format
    if head and not base:
        parser.error('--head needs --base')
    # The group is required, so `base` is set whenever `--staged` is not.
    return _Options(base or 'HEAD', '' if staged else head, files, output_format == 'github')


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
    if options.head == '':
        return _changed_files(['--cached', options.base], roots)
    if options.head is not None:
        return _changed_files([options.base, options.head], roots)
    return _changed_files([options.base], roots) + _untracked_files(roots)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the check and return the exit code."""
    options = _parse_args(argv)
    limit, roots = _config()

    changes = _select(_changes(options, roots), options.files)
    _load(changes, options.base, options.head)
    offenders = _collect(changes, limit)
    _name_cases(offenders, changes)
    added = [marker for change in changes for marker in _added_ignores(change)]
    unexplained = _report_ignores(added, options.github)
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
