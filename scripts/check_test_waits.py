"""Fail when a change adds a real-clock wait, a private readiness timeout or a probed port to a test under `tests/`.

A wait on a real timer fails once a runner stalls for longer than it, and a port read from a socket that is already
closed is free only until another process binds it. Each rule counts what it finds in each file, and a change may not
raise a file's count. What the base already has is allowed, so existing debt passes until a change adds to it.

- `readiness-constant`: a module that defines its own `READINESS_WAIT_TIMEOUT` instead of importing the shared one
  from `tests/_timeouts.py`.
- `probe-port`: a port read from a socket that a `with` block closes before the port is used, outside
  `tests/_ports.py`.
- `literal-wait`: a number, or a constant the module defines, passed as the delay or timeout of `sleep`,
  `fail_after`, `move_on_after`, `asyncio.timeout` or `asyncio.wait_for`, or of a `.wait()`, `.join()` or
  `.result(timeout=)` call. A delay of zero is a checkpoint, not a wait.
- `sleep-loop`: a loop whose body sleeps, so it polls. A `sleep(0)` counts here too: the loop then polls until
  another task has run. Its sleeps are not also counted as `literal-wait`.

A wait that moved between changed files, the same line removed from one and added to another, is not counted.
A wait or loop that is about real time is left out of the count by `# real-wait: <reason>` on the line it starts on,
or alone on the line above. Each such comment the change adds is reported, and one without a reason fails.

During a merge, `--staged` reports a rule in a file only when its count is higher than in both parents.

Usage:
    uv run scripts/check_test_waits.py --staged                     # the index against HEAD, for pre-commit
    uv run scripts/check_test_waits.py --base REF                   # the working tree against REF
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
from typing import Literal

from _ratchet import Change, Options, escape, merge_head, parse_args, read_changes, worse_than_both

GUIDE = 'tests/AGENTS.md#test-cost'
ROOTS = ('tests',)

Rule = Literal['readiness-constant', 'probe-port', 'literal-wait', 'sleep-loop']

_ADVICE: dict[Rule, str] = {
    'readiness-constant': (
        'import `READINESS_WAIT_TIMEOUT` from `tests/_timeouts.py` instead of defining a shorter one: '
        'a stalled runner can outlast it'
    ),
    'probe-port': (
        'the port is free only until something else binds it: take one from `reserve_loopback_port()` '
        'in `tests/_ports.py`, or hand the bound socket to the server'
    ),
    'literal-wait': (
        'a wait on the real clock: order the steps with an `Event` and bound the wait with '
        '`READINESS_WAIT_TIMEOUT` from `tests/_timeouts.py`, or inject the delay'
    ),
    'sleep-loop': (
        'a loop that sleeps polls, on the real clock or, with `sleep(0)`, on the scheduler: '
        'wait on an `Event`, a stream or the condition itself'
    ),
}
_ESCAPABLE: frozenset[Rule] = frozenset({'literal-wait', 'sleep-loop'})
_EXEMPT: dict[Rule, str] = {'readiness-constant': 'tests/_timeouts.py', 'probe-port': 'tests/_ports.py'}

_READINESS = re.compile(r'_?READINESS_WAIT_TIMEOUT')
_REAL_WAIT = re.compile(r'#[ \t]*real-wait[ \t]*:(?P<reason>.*)')

_CLOCKS = ('asyncio', 'anyio', 'trio', 'time')
_SLEEPS = frozenset(f'{module}.sleep' for module in _CLOCKS)
# Waiters whose delay is the first argument, by qualified name.
_DELAY_FIRST = _SLEEPS | {
    'anyio.fail_after',
    'anyio.move_on_after',
    'trio.fail_after',
    'trio.move_on_after',
    'asyncio.timeout',
}
_DELAY_KEYWORDS = ('delay', 'seconds')
# Methods whose timeout is the given positional argument, or `timeout=` when the index is `None`, on any receiver.
_TIMEOUT_METHODS: dict[str, int | None] = {'wait': 0, 'join': 0, 'result': None}


@dataclass
class _Finding:
    line: int
    text: str
    """The source line it starts on, stripped, which pairs it across the two sides of a change."""


@dataclass
class _Escape:
    line: int
    reason: str
    text: str


@dataclass
class _Scan:
    findings: dict[Rule, list[_Finding]] = field(default_factory=dict[Rule, list[_Finding]])
    escapes: list[_Escape] = field(default_factory=list[_Escape])


@dataclass
class _Growth:
    path: str
    rule: Rule
    old: int
    new: int
    added: list[_Finding]
    """The findings the change added to this file, without those that moved here from another changed file."""


class _Finder:
    """Collects the findings in one module."""

    def __init__(self, tree: ast.Module, lines: list[str], path: str) -> None:
        self.lines = lines
        self.path = path
        self.names = _imported_names(tree)
        self.constants = _module_constants(tree)
        self.found: dict[Rule, list[_Finding]] = {}
        self.looped: set[int] = set()
        """The ids of sleep calls already counted as part of a `sleep-loop`."""
        self.tree = tree

    def add(self, rule: Rule, node: ast.stmt | ast.expr) -> None:
        if self.path != _EXEMPT.get(rule):
            self.found.setdefault(rule, []).append(_Finding(node.lineno, self.lines[node.lineno - 1].strip()))

    def qualified(self, call: ast.Call) -> str | None:
        """`module.function` for a call to a function of an imported module, else `None`."""
        func = call.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            module = self.names.get(func.value.id)
            return f'{module}.{func.attr}' if module else None
        if isinstance(func, ast.Name):
            return self.names.get(func.id)
        return None

    def run(self) -> dict[Rule, list[_Finding]]:
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
                self.loop(node)
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                self.readiness(node)
            elif isinstance(node, ast.Call) and id(node) not in self.looped:
                self.wait(node)
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                self.probe(node)
        return self.found

    def loop(self, node: ast.For | ast.AsyncFor | ast.While) -> None:
        own = _own_nodes(node.body, _SCOPES + _LOOPS)
        sleeps = [n for n in own if isinstance(n, ast.Call) and self.qualified(n) in _SLEEPS]
        if sleeps:
            self.looped.update(id(n) for n in sleeps)
            self.add('sleep-loop', node)

    def readiness(self, node: ast.Assign | ast.AnnAssign) -> None:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(t, ast.Name) and _READINESS.fullmatch(t.id) for t in targets):
            self.add('readiness-constant', node)

    def wait(self, node: ast.Call) -> None:
        delay = self.delay(node)
        if delay is not None and self.real(delay):
            self.add('literal-wait', node)

    def delay(self, node: ast.Call) -> ast.expr | None:
        """The argument a waiter takes as its delay or timeout, or `None` when `node` is not a waiter."""
        keywords = {kw.arg: kw.value for kw in node.keywords if kw.arg}
        name = self.qualified(node)
        if name in _DELAY_FIRST:
            return node.args[0] if node.args else next((keywords[k] for k in _DELAY_KEYWORDS if k in keywords), None)
        if name == 'asyncio.wait_for':
            return keywords.get('timeout', node.args[1] if len(node.args) > 1 else None)
        if isinstance(node.func, ast.Attribute) and node.func.attr in _TIMEOUT_METHODS:
            index = _TIMEOUT_METHODS[node.func.attr]
            positional = node.args[index] if index is not None and len(node.args) > index else None
            return keywords.get('timeout', positional)
        return None

    def real(self, delay: ast.expr) -> bool:
        """Whether `delay` is a positive number written in this module."""
        value = self.constants.get(delay.id) if isinstance(delay, ast.Name) else _number(delay)
        return value is not None and value > 0

    def probe(self, node: ast.With | ast.AsyncWith) -> None:
        for item in node.items:
            call, alias = item.context_expr, item.optional_vars
            if not (isinstance(call, ast.Call) and self.qualified(call) == 'socket.socket'):
                continue
            if not isinstance(alias, ast.Name):
                continue
            own = _own_nodes(node.body, _SCOPES)
            bind = next((n for n in own if isinstance(n, ast.Call) and _binds_any_port(n, alias.id)), None)
            if bind is not None and self.escapes(node, alias.id):
                self.add('probe-port', bind)

    def escapes(self, node: ast.With | ast.AsyncWith, sock: str) -> bool:
        """Whether the port leaves the `with` block, which closes the socket, while it is still needed.

        A `yield` inside the block keeps the socket open for as long as the port is used, as a fixture does. A
        `return` that reads the port hands it out as the socket closes. A name read from `getsockname()` that a call
        takes, or that the function returns or yields, after the block uses it too late; an `assert` that only reads
        it does not.
        """
        body = list(_own_nodes(node.body, _SCOPES))
        if any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in body):
            return False
        ports: set[str] = set()
        for n in body:
            if isinstance(n, (ast.Assign, ast.AnnAssign)) and n.value is not None and _reads_port(n.value, sock):
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                stored = (t for target in targets for t in ast.walk(target) if isinstance(t, ast.Name))
                ports.update(t.id for t in stored if isinstance(t.ctx, ast.Store))
        for n in body:
            if isinstance(n, ast.Return) and n.value is not None:
                read = {r.id for r in ast.walk(n.value) if isinstance(r, ast.Name)}
                if _reads_port(n.value, sock) or read & ports:
                    return True
        end = node.end_lineno or node.lineno
        scope = _enclosing_scope(self.tree, node)
        rebinds = _rebinds(scope, node, ports)
        return any(
            name.id in ports and name.lineno > end and not any(end < line < name.lineno for line in rebinds[name.id])
            for name in _handed_on(scope)
        )


_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
_LOOPS = (ast.For, ast.AsyncFor, ast.While)


def _own_nodes(body: Sequence[ast.stmt], opaque: tuple[type[ast.AST], ...]) -> Iterator[ast.AST]:
    """Every node in `body`, without looking inside the nodes of an `opaque` type."""
    stack: list[ast.AST] = list(body)
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, opaque):
            stack.extend(ast.iter_child_nodes(node))


def _imported_names(tree: ast.Module) -> dict[str, str]:
    """Each name the module binds to a module, or to a function of one, with what it stands for."""
    names: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                # `import a.b` binds `a`.
                bound = alias.name if alias.asname else alias.name.partition('.')[0]
                names[alias.asname or bound] = bound
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for alias in node.names:
                names[alias.asname or alias.name] = f'{node.module}.{alias.name}'
    return names


def _module_constants(tree: ast.Module) -> dict[str, float]:
    """The number each module-level name is assigned."""
    constants: dict[str, float] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        value = _number(node.value)
        for target in targets:
            if isinstance(target, ast.Name) and value is not None:
                constants[target.id] = value
    return constants


def _number(node: ast.expr | None) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    return None


def _binds_any_port(node: ast.Call, sock: str) -> bool:
    """Whether `node` is `sock.bind((host, 0))`, which asks the system for a free port."""
    if not (isinstance(node.func, ast.Attribute) and node.func.attr == 'bind'):
        return False
    receiver, address = node.func.value, node.args[0] if node.args else None
    if not (isinstance(receiver, ast.Name) and receiver.id == sock and isinstance(address, ast.Tuple)):
        return False
    port = address.elts[1] if len(address.elts) == 2 else None
    return _number(port) == 0


def _reads_port(node: ast.expr, sock: str) -> bool:
    return any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == 'getsockname'
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == sock
        for n in ast.walk(node)
    )


def _enclosing_scope(tree: ast.Module, target: ast.AST) -> ast.AST:
    """The innermost function that contains `target`, or the module."""
    scope: ast.AST = tree
    for node in ast.walk(tree):
        is_function = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        if is_function and any(n is target for n in ast.walk(node)):
            scope = node
    return scope


def _rebinds(scope: ast.AST, node: ast.stmt, names: set[str]) -> dict[str, list[int]]:
    """The lines where a statement after `node`, beside it, assigns each of `names` again.

    A read after such a line is not the port. Only an assignment in the same statement list counts: one under a
    condition or in another function may not run.
    """
    lines: dict[str, list[int]] = {name: [] for name in names}
    for body in _statement_lists(scope):
        if not any(statement is node for statement in body):
            continue
        for statement in body[body.index(node) + 1 :]:
            if isinstance(statement, ast.Assign):
                targets = statement.targets
            elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
                targets = [statement.target]
            else:
                continue
            for target in (n for t in targets for n in ast.walk(t)):
                if isinstance(target, ast.Name) and target.id in names:
                    # A read in a multi-line right-hand side still sees the port, so count from the statement's end.
                    lines[target.id].append(statement.end_lineno or statement.lineno)
    return lines


def _statement_lists(scope: ast.AST) -> Iterator[list[ast.stmt]]:
    """Every list of statements in `scope`: each block's body, `else` and `finally`."""
    for node in ast.walk(scope):
        match node:
            case ast.For() | ast.AsyncFor() | ast.While() | ast.If():
                yield from (node.body, node.orelse)
            case ast.Try() | ast.TryStar():
                yield from (node.body, node.orelse, node.finalbody)
            case (
                ast.Module()
                | ast.FunctionDef()
                | ast.AsyncFunctionDef()
                | ast.ClassDef()
                | ast.With()
                | ast.AsyncWith()
                | ast.ExceptHandler()
                | ast.match_case()
            ):
                yield node.body
            case _:
                pass


def _handed_on(scope: ast.AST) -> Iterator[ast.Name]:
    """Every name read in a call's arguments, or in a value `scope` returns or yields, outside `assert` statements."""
    asserted = {id(n) for a in ast.walk(scope) if isinstance(a, ast.Assert) for n in ast.walk(a)}
    for node in ast.walk(scope):
        if id(node) in asserted:
            continue
        if isinstance(node, ast.Call):
            values = [*node.args, *(kw.value for kw in node.keywords)]
        elif isinstance(node, (ast.Return, ast.Yield)) and node.value is not None:
            values = [node.value]
        else:
            continue
        for value in values:
            yield from (n for n in ast.walk(value) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load))


def _escapes(source: str) -> list[_Escape]:
    found: list[_Escape] = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        marker = _REAL_WAIT.match(token.string) if token.type == tokenize.COMMENT else None
        if marker:
            alone = not token.line[: token.start[1]].strip()
            # A comment alone on its line marks the next line.
            line = token.start[0] + 1 if alone else token.start[0]
            found.append(_Escape(line, marker['reason'].strip(), token.line.strip()))
    return found


def _scan(source: str | None, path: str) -> _Scan:
    if source is None:
        return _Scan()
    try:
        tree = ast.parse(source)
        escapes = _escapes(source)
    except (SyntaxError, tokenize.TokenError):
        return _Scan()
    findings = _Finder(tree, source.splitlines(), path).run()
    escaped = {e.line for e in escapes}
    for rule in _ESCAPABLE & findings.keys():
        findings[rule] = [f for f in findings[rule] if f.line not in escaped]
    return _Scan({rule: found for rule, found in findings.items() if found}, escapes)


def _texts(findings: list[_Finding]) -> Counter[str]:
    return Counter(f.text for f in findings)


def _rule_growth(
    path: str, rule: Rule, found: list[_Finding], was: list[_Finding], moves: Counter[str]
) -> _Growth | None:
    """How far a change raised `rule`'s count in a file, less the findings that moved in from another file.

    `moves` holds the findings, by text, that the change removed from the other changed files; each one this file
    gained is taken out of it.
    """
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
    return _Growth(path, rule, len(was), len(found), added)


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
    """Each rule a change raised the count of in a file, and each `# real-wait` comment it added."""
    scans = [(c, _scan(c.old_source, c.old_path or ''), _scan(c.new_source, c.new_path or '')) for c in changes]
    # Only a wait may move, so a probe or constant removed from one file does not excuse one added to another.
    moves: dict[Rule, Counter[str]] = {rule: Counter() for rule in _ESCAPABLE}
    for _, old, new in scans:
        for rule in _ESCAPABLE & old.findings.keys():
            moves[rule].update(_texts(old.findings[rule]) - _texts(new.findings.get(rule, [])))
    growth: list[_Growth] = []
    escapes: list[tuple[str, _Escape]] = []
    for change, old, new in scans:
        path = change.new_path or ''
        escapes += _added_escapes(path, old, new)
        for rule, found in new.findings.items():
            grown = _rule_growth(path, rule, found, old.findings.get(rule, []), moves.get(rule, Counter()))
            if grown:
                growth.append(grown)
    return growth, escapes


def _print_growth(g: _Growth, github: bool) -> None:
    print(f'{g.path}: {g.rule} {g.old} -> {g.new}')
    for f in g.added:
        print(f'    {g.path}:{f.line}: {f.text}')
        if github:
            title = escape(f'Test waits: {g.rule} {g.old} -> {g.new}', prop=True)
            print(f'::error file={escape(g.path, prop=True)},line={f.line},title={title}::{escape(_ADVICE[g.rule])}')


def _print_advice(rules: set[Rule]) -> None:
    print()
    for rule in sorted(rules):
        print(f'{rule}: {_ADVICE[rule]}')
    if rules & _ESCAPABLE:
        print('A wait that is about real time can say so with `# real-wait: <reason>`.')
    print(f'See {GUIDE}.')


def _report_escapes(escapes: list[tuple[str, _Escape]], github: bool) -> bool:
    """Print each `# real-wait` comment the change added; return whether one gives no reason."""
    for path, e in escapes:
        level, message = ('warning', f'real wait: {e.reason}') if e.reason else ('error', 'give the real wait a reason')
        print(f'{path}:{e.line}: {message}, see {GUIDE}')
        if github:
            print(f'::{level} file={escape(path, prop=True)},line={e.line},title=real-wait::{escape(message)}')
    return any(not e.reason for _, e in escapes)


def _check(options: Options) -> tuple[list[_Growth], list[tuple[str, _Escape]]]:
    return _growth(read_changes(options, ROOTS))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the check and return the exit code."""
    options = parse_args(argv, 'Check the real-clock waits a change adds to the tests.')
    growth, escapes = _check(options)
    theirs = merge_head() if options.staged else None
    if theirs:
        their_growth, _ = _check(replace(options, base=theirs))
        growth = worse_than_both(growth, their_growth, key=lambda g: (g.path, g.rule), before=lambda g: g.old)
    unexplained = _report_escapes(escapes, options.github)
    for g in growth:
        _print_growth(g, options.github)
    if growth:
        _print_advice({g.rule for g in growth})
    return 1 if growth or unexplained else 0


if __name__ == '__main__':
    sys.exit(main())
