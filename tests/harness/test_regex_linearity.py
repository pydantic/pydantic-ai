"""Every regular expression the harness and pax ship scans repeated adversarial input at a steady cost per character.

`re` backtracks, so a pattern whose match attempts re-read the same input, from each offset of a run or once per
way of splitting it, takes quadratic or exponential time on input built to provoke it. A harness pattern often runs
on text the model or a tool produced, and it runs on the event loop. So does a pax pattern.

The patterns are found in the package source, so a new one is checked without being listed here: every call to a
`re` function is read, and its pattern is worked out from the literals, f-strings, `'|'.join(...)` calls,
`.pattern` of a compiled pattern, and names bound to them in its module or function. `re.escape(...)` of runtime text stands in as `x`: escaped text
matches only itself, so the pattern around it is what can backtrack. Each call whose pattern only exists at
runtime says where its pattern comes from in `RUNTIME_PATTERNS`, and the built-in tables those calls compile are
checked through `PATTERN_TABLES`.

Each pattern is run the way its calls run it. `re.match` and `re.fullmatch` try one match from the start.
`re.search` tries each offset until one matches, so it re-reads a run from each offset only where every attempt in
the run fails; an alternative that matches where the others fail ends the search there. Every other call goes on from
each match it finds. A compiled pattern bound to a module-level name runs the way the methods called on that name do, in any
scanned module, the most expensive of them; any other use of the name, or none, counts as going on.

Each pattern is then scanned by `tests/harness/_regex_steps.py`, which counts the steps `re`'s backtracking takes
rather than timing it. `linearity_failure` says what it asserts, on inputs built from each pattern's own literals and
repeats: a pass shows those inputs scan in linear time, not that every input does. A failure names the
repeated unit and the step counts. The usual fix keeps a failed attempt from restarting inside the run it already
read, with a lookbehind such as `(?<![\\w-])` before the repeat, or bounds what the repeat can cross.
"""

from __future__ import annotations

import ast
import itertools
import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass

import pytest
from inline_snapshot import snapshot

from pydantic_ai_harness.artifact_scanning import DEFAULT_ARTIFACT_PATTERNS
from pydantic_ai_harness.guardrails.detectors import DEFAULT_PII_PATTERNS, DEFAULT_SECRET_PATTERNS
from pydantic_ai_harness.require_verification import DEFAULT_EDIT_COMMANDS, DEFAULT_VERIFICATION_COMMANDS

from .._source_roots import REPO_ROOT, shipped_packages
from ._regex_steps import Mode, Shape, StepCounter, linearity_failure, shape

_PACKAGES = [
    REPO_ROOT / package.path
    for package in shipped_packages()
    if package.name in ('pydantic_ai_harness', 'pydantic_ai_pax')
]
_ONE_MATCH: dict[str, Mode] = {'search': 'search', 'match': 'match', 'fullmatch': 'fullmatch'}
"""The `re` functions and methods that look for one match, by name."""
_COST: dict[Mode, int] = {'match': 0, 'fullmatch': 1, 'search': 2, 'scan': 3}
"""How much of the input each mode can re-read: `fullmatch` reads on past an end `match` accepts."""
_FLAGS_ARGUMENT = {
    'compile': 1,
    'search': 2,
    'match': 2,
    'fullmatch': 2,
    'findall': 2,
    'finditer': 2,
    'split': 3,
    'sub': 4,
    'subn': 4,
}
"""Each `re` function that takes a pattern, and the position of its `flags` argument."""
_ESCAPED = 'x'
"""What `re.escape(...)` of runtime text stands in as."""

PATTERN_TABLES: dict[str, dict[str, tuple[str, int]]] = {
    'DEFAULT_SECRET_PATTERNS': {name: (pattern, 0) for name, pattern in DEFAULT_SECRET_PATTERNS.items()},
    'DEFAULT_PII_PATTERNS': {name: (pattern, 0) for name, pattern in DEFAULT_PII_PATTERNS.items()},
    'DEFAULT_ARTIFACT_PATTERNS': {
        name: (pattern, re.IGNORECASE) for name, (_, pattern) in DEFAULT_ARTIFACT_PATTERNS.items()
    },
    'DEFAULT_VERIFICATION_COMMANDS': {
        str(index): (pattern, 0) for index, pattern in enumerate(DEFAULT_VERIFICATION_COMMANDS)
    },
    'DEFAULT_EDIT_COMMANDS': {str(index): (pattern, 0) for index, pattern in enumerate(DEFAULT_EDIT_COMMANDS)},
}
"""Every `DEFAULT_*_PATTERNS` table the package defines, and the default command patterns `RequireVerification`
compiles, with the flags the package compiles them with."""

RUNTIME_PATTERNS: dict[str, tuple[str, ...]] = {
    'pydantic_ai_harness/artifact_scanning/_scanner.py::ArtifactScanner.__post_init__': (
        'a `DEFAULT_ARTIFACT_PATTERNS` entry',
        "the user's `extra_patterns`",
    ),
    'pydantic_ai_harness/browser_use/_toolset.py::BrowserUseToolset.__init__': (
        'an alternation of escaped secret placeholders',
    ),
    'pydantic_ai_harness/file_system/_toolset.py::FileSystemToolset._search_files': ("the model's search pattern",),
    'pydantic_ai_harness/guardrails/detectors.py::_compile': (
        "a `DEFAULT_SECRET_PATTERNS` or `DEFAULT_PII_PATTERNS` entry, or the user's `extra`",
    ),
    'pydantic_ai_harness/permissions/_capability.py::Permissions.__post_init__': ("the user's `ask_commands`",),
    'pydantic_ai_harness/playwright/_toolset.py::_without_endpoint_credentials': (
        'an alternation of the escaped `cdp_url` credential values',
    ),
    'pydantic_ai_harness/require_verification/_capability.py::RequireVerification.__post_init__': (
        "a `DEFAULT_VERIFICATION_COMMANDS` entry, or the user's `verification_commands`",
        "a `DEFAULT_EDIT_COMMANDS` entry, or the user's `edit_commands`",
    ),
    'pydantic_ai_pax/builtin_plugins/sandbox/__init__.py::_ConversationSandbox.request_env': (
        '`KEY_NAME`, imported from `pydantic_ai_pax/models/profiles.py`',
    ),
    'pydantic_ai_pax/config/api_keys.py::normalize_name': (
        '`KEY_NAME`, imported from `pydantic_ai_pax/models/profiles.py`',
    ),
}
"""Calls whose pattern the scan can't work out, keyed by `<file>::<function>`: where each call's pattern comes from, in
source order. A built-in table they compile is checked through `PATTERN_TABLES`; a caller's own pattern can't be, and
the scan doesn't follow an import."""

CHECKER_LIMITS: dict[str, str] = {
    'pydantic_ai_pax/ui/rendering/command_syntax.py::_HEREDOC': (
        'its backreference repeats the optional quote, and `StepCounter` runs no backreference'
    ),
}
"""Patterns `StepCounter` can't run, by label, each an expected failure with the reason."""


@dataclass(frozen=True)
class _ElementOf:
    """A comprehension target, bound to each element of a literal sequence in turn."""

    sequence: ast.expr


_Binding = ast.expr | _ElementOf | None
"""What a name is bound to in a scope; `None` when it can't be worked out statically, such as a parameter."""


class _Unresolved(Exception):
    """The expression's value isn't known until runtime."""


def _bindings(scope: ast.AST) -> dict[str, _Binding]:
    """The names a module or function binds in its own scope, not in the functions and classes nested in it."""
    bindings: dict[str, _Binding] = {}
    if isinstance(scope, ast.FunctionDef | ast.AsyncFunctionDef):
        arguments = scope.args
        for argument in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
            arguments.vararg,
            arguments.kwarg,
        ):
            if argument is not None:
                bindings[argument.arg] = None
    pending = list(ast.iter_child_nodes(scope))
    while pending:
        node = pending.pop()
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda):
            continue
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            bindings[name] = None if name in bindings else node.value
            pending.append(node.value)
            continue
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            name = node.target.id
            bindings[name] = None if name in bindings else node.value
            pending.append(node.value)
            continue
        if isinstance(node, ast.comprehension) and isinstance(node.target, ast.Name):
            bindings[node.target.id] = _ElementOf(node.iter)
            pending.extend([node.iter, *node.ifs])
            continue
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name):
            name = node.target.id
            bindings[name] = None if name in bindings else _ElementOf(node.iter)
            pending.extend([node.iter, *node.body, *node.orelse])
            continue
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(targets := node.targets[0], ast.Tuple)
            and isinstance(node.value, ast.Tuple)
            and len(targets.elts) == len(node.value.elts)
        ):
            for target, value in zip(targets.elts, node.value.elts):
                if isinstance(target, ast.Name):
                    bindings[target.id] = None if target.id in bindings else value
                else:
                    pending.append(target)
            pending.append(node.value)
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bindings[node.id] = None
        pending.extend(ast.iter_child_nodes(node))
    return bindings


def _lookup(name: str, scopes: list[dict[str, _Binding]]) -> _Binding:
    for scope in scopes:
        if name in scope:
            return scope[name]
    raise _Unresolved(name)


def _literal_container(node: ast.expr, scopes: list[dict[str, _Binding]]) -> ast.expr:
    """The display a name refers to, following names bound to names."""
    while isinstance(node, ast.Name):
        binding = _lookup(node.id, scopes)
        if not isinstance(binding, ast.expr):
            raise _Unresolved(node.id)
        node = binding
    return node


def _elements(node: ast.expr, scopes: list[dict[str, _Binding]]) -> list[ast.expr]:
    """The elements of a tuple, list or set display, or a dict display's keys, as iterating it gives them."""
    container = _literal_container(node, scopes)
    if isinstance(container, ast.Dict):
        return [key for key in container.keys if key is not None]
    if isinstance(container, ast.Tuple | ast.List | ast.Set):
        return container.elts
    raise _Unresolved(ast.unparse(node))


def _subscript(node: ast.Subscript, scopes: list[dict[str, _Binding]]) -> ast.expr:
    """The element a subscript of a dict, tuple or list display picks."""
    container = _literal_container(node.value, scopes)
    if isinstance(container, ast.Subscript):
        container = _subscript(container, scopes)
    entries: dict[str | int, ast.expr]
    if isinstance(container, ast.Dict):
        entries = {
            key: value
            for key_node, value in zip(container.keys, container.values)
            if key_node
            for key in _fold(key_node, scopes)
        }
    elif isinstance(container, ast.Tuple | ast.List):
        entries = {index: element for index, element in enumerate(container.elts)}
    else:
        raise _Unresolved(ast.unparse(node))
    try:
        [key] = _fold(node.slice, scopes)
        return entries[key]
    except (ValueError, KeyError) as error:
        raise _Unresolved(ast.unparse(node)) from error


def _named(name: str, scopes: list[dict[str, _Binding]]) -> list[str | int]:
    """Every value the name `name` can take."""
    binding = _lookup(name, scopes)
    if isinstance(binding, _ElementOf):
        return [value for element in _elements(binding.sequence, scopes) for value in _fold(element, scopes)]
    if binding is None:
        raise _Unresolved(name)
    return _fold(binding, scopes)


def _compiled_pattern(node: ast.Attribute, name: str, scopes: list[dict[str, _Binding]]) -> ast.expr:
    """The pattern `name.pattern` reads: the one `name` was compiled from."""
    compiled = _lookup(name, scopes)
    if not (isinstance(compiled, ast.Call) and ast.unparse(compiled.func) == 're.compile'):
        raise _Unresolved(ast.unparse(node))
    pattern = _argument(compiled, 0, 'pattern')
    assert pattern is not None
    return pattern


def _strings(values: list[str | int]) -> list[str]:
    strings = [value for value in values if isinstance(value, str)]
    assert len(strings) == len(values), f'expected strings, got {values!r}'
    return strings


def _ints(values: list[str | int]) -> list[int]:
    ints = [value for value in values if isinstance(value, int)]
    assert len(ints) == len(values), f'expected flags, got {values!r}'
    return ints


def _fold(node: ast.expr, scopes: list[dict[str, _Binding]]) -> list[str | int]:
    """Every value `node` can take, worked out from the source alone."""
    match node:
        case ast.Constant(value=str() | int() as value):
            return [value]
        case ast.JoinedStr(values=values):
            texts = ['']
            for value in values:
                if isinstance(value, ast.FormattedValue) and (value.conversion != -1 or value.format_spec):
                    raise _Unresolved(ast.unparse(value))
                part = _strings(_fold(value.value if isinstance(value, ast.FormattedValue) else value, scopes))
                texts = [text + piece for text in texts for piece in part]
            return list(texts)
        case ast.BinOp(left=left, op=ast.Add(), right=right):
            return [a + b for a in _strings(_fold(left, scopes)) for b in _strings(_fold(right, scopes))]
        case ast.BinOp(left=left, op=ast.BitOr(), right=right):
            return [a | b for a in _ints(_fold(left, scopes)) for b in _ints(_fold(right, scopes))]
        case ast.Attribute(value=ast.Name(id='re'), attr=attribute) if isinstance(getattr(re, attribute, None), int):
            flag: int = getattr(re, attribute)
            return [flag]
        case ast.Attribute(value=ast.Name(id=name), attr='pattern'):
            return _fold(_compiled_pattern(node, name, scopes), scopes)
        case ast.Name(id=name):
            return _named(name, scopes)
        case ast.Subscript():
            return _fold(_subscript(node, scopes), scopes)
        case ast.Call(func=ast.Attribute(value=ast.Name(id='re'), attr='escape')):
            return [_ESCAPED]
        case ast.Call(func=ast.Attribute(value=separator, attr='join'), args=[sequence]):
            choices = [_strings(_fold(element, scopes)) for element in _elements(sequence, scopes)]
            joiners = _strings(_fold(separator, scopes))
            return [joiner.join(parts) for joiner in joiners for parts in itertools.product(*choices)]
        case ast.IfExp(body=body, orelse=orelse):
            return [*_fold(body, scopes), *_fold(orelse, scopes)]
        case _:
            raise _Unresolved(ast.unparse(node))


@dataclass(frozen=True)
class CallSite:
    location: str
    """`<file>::<qualified name>`: the function the call is in, or the name a module-level call is assigned to."""
    patterns: list[tuple[str, int]] | None
    """The patterns and flags the call can compile, or `None` if they only exist at runtime."""
    mode: Mode = 'scan'
    """How the call, or the methods called on the compiled pattern it binds, run the pattern."""


def _argument(call: ast.Call, position: int, keyword: str) -> ast.expr | None:
    if len(call.args) > position:
        return call.args[position]
    return next((k.value for k in call.keywords if k.arg == keyword), None)


def _patterns(call: ast.Call, function: str, scopes: list[dict[str, _Binding]]) -> list[tuple[str, int]] | None:
    pattern = _argument(call, 0, 'pattern')
    flags = _argument(call, _FLAGS_ARGUMENT[function], 'flags')
    assert pattern is not None
    try:
        if isinstance(pattern, ast.Constant) and isinstance(pattern.value, bytes):
            # A bytes pattern runs on bytes as its latin-1 decoding runs, with ASCII classes, on the decoded text.
            patterns, text_flag = [pattern.value.decode('latin-1')], re.ASCII
        else:
            patterns, text_flag = _strings(_fold(pattern, scopes)), 0
        flag_values = [0] if flags is None else _ints(_fold(flags, scopes))
    except _Unresolved:
        return None
    return [(text, flag | text_flag) for text in patterns for flag in flag_values]


def _modes_of_names(trees: list[ast.Module]) -> dict[str, Mode]:
    """How each name runs the pattern it holds: through its `search`, `match` or `fullmatch` method, or any other way."""
    calls = {id(node.func) for tree in trees for node in ast.walk(tree) if isinstance(node, ast.Call)}
    methods = {
        id(node.value): node
        for tree in trees
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
    }
    modes: dict[str, Mode] = {}
    for tree in trees:
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)):
                continue
            method = methods.get(id(node))
            used = _ONE_MATCH.get(method.attr, 'scan') if method is not None and id(method) in calls else 'scan'
            previous = modes.get(node.id, used)
            modes[node.id] = previous if _COST[previous] >= _COST[used] else used
    return modes


def _call_sites_in(relative: str, tree: ast.Module, modes: dict[str, Mode]) -> Iterator[CallSite]:
    """Every call to a `re` function that takes a pattern, in one module; `modes` is `_modes_of_names`'s."""

    def visit(node: ast.AST, qualname: str, scopes: list[dict[str, _Binding]]) -> Iterator[CallSite]:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                name = f'{qualname}.{child.name}' if qualname else child.name
                inner = scopes if isinstance(child, ast.ClassDef) else [_bindings(child), *scopes]
                yield from visit(child, name, inner)
                continue
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and isinstance(child.func.value, ast.Name)
                and child.func.value.id == 're'
                and child.func.attr in _FLAGS_ARGUMENT
            ):
                function = child.func.attr
                mode = _ONE_MATCH.get(function, 'scan')
                if (
                    function == 'compile'
                    and isinstance(node, ast.Assign)
                    and node.value is child
                    and [target.id for target in node.targets if isinstance(target, ast.Name)] == [qualname]
                ):
                    mode = modes.get(qualname, 'scan')
                yield CallSite(f'{relative}::{qualname}', _patterns(child, function, scopes), mode)
            assigned = (
                child.targets[0].id
                if isinstance(child, ast.Assign) and len(child.targets) == 1 and isinstance(child.targets[0], ast.Name)
                else None
            )
            yield from visit(child, assigned if assigned and not qualname else qualname, scopes)

    yield from visit(tree, '', [_bindings(tree)])


def _hidden_re_uses(tree: ast.Module) -> list[str]:
    """Uses of `re` that `_call_sites_in` can't see: imported names, a renamed or rebound `re`, a function passed on."""
    attribute_values = {id(node.value) for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    called = {id(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    hidden: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in ('re', 'regex'):
            hidden.append(f'from {node.module} import ...')
        elif isinstance(node, ast.Import):
            hidden += [f'import {a.name} as {a.asname}' for a in node.names if a.name == 're' and a.asname]
            hidden += ['import regex' for a in node.names if a.name == 'regex']
        elif isinstance(node, ast.Name) and node.id == 're' and id(node) not in attribute_values:
            hidden.append(f'`re` itself, at line {node.lineno}')
        elif (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == 're'
            and node.attr in _FLAGS_ARGUMENT
            and id(node) not in called
        ):
            hidden.append(f'`re.{node.attr}` uncalled, at line {node.lineno}')
    return hidden


_SOURCES = {
    path.relative_to(package.parent).as_posix(): path.read_text(encoding='utf-8')
    for package in _PACKAGES
    for path in sorted(package.rglob('*.py'))
}
# Parsed at collection, once: a module that never names `re` or `regex` can't call either, so it isn't parsed.
_MODULES = [
    (relative, ast.parse(source)) for relative, source in _SOURCES.items() if re.search(r'\b(?:re|regex)\b', source)
]
_MODES = _modes_of_names([tree for _, tree in _MODULES])
_CALL_SITES = [site for relative, tree in _MODULES for site in _call_sites_in(relative, tree, _MODES)]


def _checked_patterns() -> dict[str, tuple[str, int, Mode]]:
    checked: dict[str, tuple[str, int, Mode]] = {}
    seen: Counter[str] = Counter()
    for site in _CALL_SITES:
        for pattern, flags in site.patterns or ():
            seen[site.location] += 1
            label = f'{site.location}#{seen[site.location]}' if seen[site.location] > 1 else site.location
            checked[label] = (pattern, flags, site.mode)
    for table_name, table in PATTERN_TABLES.items():
        checked.update({f'{table_name}[{name!r}]': (*pattern, 'scan') for name, pattern in table.items()})
    # A pattern several calls run is checked once, in the most expensive of their modes.
    unique: dict[tuple[str, int], tuple[str, Mode]] = {}
    for label, (pattern, flags, mode) in checked.items():
        first = unique.setdefault((pattern, flags), (label, mode))
        if _COST[mode] > _COST[first[1]]:
            unique[pattern, flags] = (first[0], mode)
    return {label: (pattern, flags, mode) for (pattern, flags), (label, mode) in unique.items()}


@pytest.mark.parametrize(
    ('pattern', 'flags', 'mode'),
    [
        pytest.param(
            *p,
            id=label,
            marks=[pytest.mark.xfail(raises=NotImplementedError, strict=True, reason=CHECKER_LIMITS[label])]
            if label in CHECKER_LIMITS
            else [],
        )
        for label, p in _checked_patterns().items()
    ],
)
def test_scanning_is_linear(pattern: str, flags: int, mode: Mode):
    assert linearity_failure(pattern, flags, mode) is None


def test_every_checker_limit_is_a_checked_pattern():
    assert CHECKER_LIMITS.keys() <= _checked_patterns().keys()


def test_every_runtime_pattern_says_where_it_comes_from():
    """Each call whose pattern only exists at runtime has an entry, and each entry still has its call."""
    unresolved = Counter(site.location for site in _CALL_SITES if site.patterns is None)

    assert dict(unresolved) == {location: len(sources) for location, sources in RUNTIME_PATTERNS.items()}


def test_every_pattern_table_is_checked():
    """A new table reaches `re` through a call `RUNTIME_PATTERNS` already lists, so it has to be listed by name."""
    tables = {name for source in _SOURCES.values() for name in re.findall(r'^(DEFAULT_\w+_PATTERNS)\b', source, re.M)}

    assert sorted(tables) == sorted(name for name in PATTERN_TABLES if name.endswith('_PATTERNS'))


def test_re_functions_are_only_called_directly():
    """A `from re import compile`, a renamed `re` or `re.compile` passed on would hide calls from the scan above."""
    assert [(relative, hidden) for relative, tree in _MODULES for hidden in _hidden_re_uses(tree)] == []


_UNRESOLVED_SOURCE = """\
import re
from elsewhere import IMPORTED
T = {'a': 'x'}

def f(pattern, table, names):
    re.compile(pattern)
    [re.search(value, key) for key, value in table.items()]
    re.compile(IMPORTED)
    re.compile('|'.join(names))
    re.compile('|'.join(name for name in 'ab'))
    re.compile(f'{pattern!r}')
    re.compile(table()[0])
    re.compile(str(pattern))
    re.compile(T['missing'])
    re.compile(T['a' if pattern else 'b'])
    re.compile(T.pattern)
"""


@pytest.mark.parametrize(
    ('source', 'sites'),
    [
        pytest.param(
            "import re\n_A = re.compile(r'a+' 'b', re.I)\n",
            [CallSite('m.py::_A', [('a+b', re.IGNORECASE)])],
            id='module-constant',
        ),
        pytest.param(
            "import re\n_P = 'x'\n_N = ('a', 'b')\n_K = {'c': 1, 'd': 2}\n_A = re.compile(rf'{_P}(?:{\"|\".join(_N)})|' + '|'.join(_K))\n",
            [CallSite('m.py::_A', [('x(?:a|b)|c|d', 0)])],
            id='f-string-join-and-concatenation',
        ),
        pytest.param(
            "import re\nT: dict[str, tuple[str, str]] = {'j': ('t', 'b'), 'k': ('t', 'a*')}\n_A = re.compile(T['k'][1])\n",
            [CallSite('m.py::_A', [('a*', 0)])],
            id='table-entry',
        ),
        pytest.param(
            'import re\ndef f(command):\n    ps = [r"^a", r"^b"]\n    return any(re.match(p, command) for p in ps)\n',
            [CallSite('m.py::f', [('^a', 0), ('^b', 0)], 'match')],
            id='comprehension-over-local-list',
        ),
        pytest.param(
            "import re\nclass C:\n    def f(self, word, strict):\n        return re.sub(rf'\\b{re.escape(word)}' if strict else re.escape(word), '', 'text', flags=re.I | re.M)\n",
            [CallSite('m.py::C.f', [('\\bx', re.I | re.M), ('x', re.I | re.M)])],
            id='escaped-runtime-text',
        ),
        pytest.param(
            "import re\n_A = re.compile(r'a(?!b)')\n_B = re.compile(rf'{_A.pattern}|c', re.I)\n",
            [CallSite('m.py::_A', [('a(?!b)', 0)]), CallSite('m.py::_B', [('a(?!b)|c', re.IGNORECASE)])],
            id='compiled-pattern',
        ),
        pytest.param(
            "import re\n_A = re.compile('a+')\n_B = re.compile('b+')\n_C = re.compile('c+')\n_D = re.compile('d+')\n"
            "_E = re.compile('e+')\n"
            'def f(s):\n    return _A.match(s), _B.fullmatch(s), _B.match(s), _C.match(s), _C.sub("", s), [_D]'
            ', _E.search(s), _E.match(s), re.search("x+", s)\n',
            [
                CallSite('m.py::_A', [('a+', 0)], 'match'),
                CallSite('m.py::_B', [('b+', 0)], 'fullmatch'),
                CallSite('m.py::_C', [('c+', 0)], 'scan'),
                CallSite('m.py::_D', [('d+', 0)], 'scan'),
                CallSite('m.py::_E', [('e+', 0)], 'search'),
                CallSite('m.py::f', [('x+', 0)], 'search'),
            ],
            id='compiled-pattern-mode-from-its-methods',
        ),
        pytest.param(
            "import re\nA, B = 'a', 'b'\ndef f(s):\n    for p in ('x', 'y'):\n        re.match(p, s)\n"
            "    re.compile(f'{A}{B}')\n",
            [CallSite('m.py::f', [('x', 0), ('y', 0)], 'match'), CallSite('m.py::f', [('ab', 0)])],
            id='loop-over-a-literal-and-unpacked-names',
        ),
        pytest.param(
            "import re\n_A = re.compile(rb'\\x1b\\w+')\n",
            [CallSite('m.py::_A', [('\\x1b\\w+', re.ASCII)])],
            id='bytes-pattern',
        ),
        pytest.param(
            _UNRESOLVED_SOURCE,
            [CallSite('m.py::f', None), CallSite('m.py::f', None, 'search'), *[CallSite('m.py::f', None)] * 9],
            id='runtime-patterns',
        ),
    ],
)
def test_the_call_site_scan(source: str, sites: list[CallSite]):
    tree = ast.parse(source)
    assert list(_call_sites_in('m.py', tree, _modes_of_names([tree]))) == sites


def test_the_hidden_use_scan():
    source = 'import re\nimport re as regexp\nfrom re import compile\nimport regex\n_re = re\nmap(re.compile, [])\nre.compile("a")\n'

    assert sorted(_hidden_re_uses(ast.parse(source))) == snapshot(
        [
            '`re.compile` uncalled, at line 6',
            '`re` itself, at line 5',
            'from re import ...',
            'import re as regexp',
            'import regex',
        ]
    )


@pytest.mark.parametrize(
    ('pattern', 'failure'),
    [
        pytest.param(r'\w+@', snapshot("'0' * 24: over 900 steps"), id='restart-in-a-run'),
        pytest.param(
            r'\bkey-[\w-]+\.',
            snapshot("'key-' * 12, 24, 48: [914, 3554, 14018] steps"),
            id='restart-at-each-word-in-a-run',
        ),
        pytest.param(r'<.*?>', snapshot("'<' * 96: over 6208 steps"), id='lazy-to-a-missing-end'),
        pytest.param(r'\s*\s*x', snapshot("'\\t' * 24: over 2000 steps"), id='adjacent-repeats'),
        pytest.param(r'(a+)+b', snapshot("'a' * 24: over 1500 steps"), id='nested-repeats'),
        pytest.param(r'(?:a|a){0,20}b', snapshot("'a' * 48: over 17248 steps"), id='bounded-alternation'),
        pytest.param(r'\d+(?:\.\d+)+|\d', snapshot("'0' * 96: over 12416 steps"), id='restart-after-each-match'),
    ],
)
def test_a_super_linear_pattern_fails(pattern: str, failure: str):
    """The check catches each way a backtracking pattern goes super-linear, with step counts that show it."""
    assert linearity_failure(pattern, 0) == failure


@pytest.mark.parametrize(
    ('pattern', 'mode'),
    [
        pytest.param(r'\$\'(?:[^\'\\]|\\.)*\'|\'[^\']*\'|"(?:[^"\\]|\\.)*"|\\.', 'scan', id='matches-spanning-units'),
        pytest.param(r'\d+(?:\.\d+)+|\d', 'search', id='search-ended-by-a-later-alternative'),
    ],
)
def test_a_linear_pattern_passes(pattern: str, mode: Mode):
    """Linear patterns that pass only because the check runs them as `re` does.

    A match can span two units (`$'` then `$'`), so each repeat count is a multiple of 12, and the scans at all three
    sizes end the same way. A search for a dotted version or one digit reads a digit run once: the one-digit
    alternative then matches at the run's start and ends the search, where `re.finditer` reads the run again from every
    offset.
    """
    assert linearity_failure(pattern, 0, mode) is None


@pytest.mark.parametrize(
    ('pattern', 'flags', 'text'),
    [
        pytest.param(r'ab|a[^b]', 0, 'abacad', id='branch-and-negated-literal'),
        pytest.param(r'a+', 0, 'aAa', id='case-sensitive'),
        pytest.param(r'[a-c]x|(?-i:[a-c])y|(?i:K)k', re.IGNORECASE, 'BXcx Ay ay kKKk', id='scoped-flags'),
        pytest.param(r'(?a:\w+)', 0, 'aé b', id='scoped-ascii'),
        pytest.param(r'.+', 0, 'a\nbc', id='any'),
        pytest.param(r'.+', re.DOTALL, 'a\nbc', id='dotall'),
        pytest.param(r'^a|a$|\Ab|b\Z', 0, 'ab\nab\n', id='anchors'),
        pytest.param(r'^a|a$', re.MULTILINE, 'ab\nba\na', id='multiline-anchors'),
        pytest.param(r'\bab\b', 0, 'ab cab ab', id='word-boundary'),
        pytest.param(r'(?<=a)b|(?<!c)d|x(?=y)|z(?!w)', 0, 'ab cd ad xy xz zw z', id='lookarounds'),
        pytest.param(r'a{2,3}?b|c*?d|e++e|f?+g', 0, 'aab aaab ccd eee fg', id='lazy-and-possessive'),
        pytest.param(r'(?>a+)b|(?:ab|a)+c|(?:x|)*y|(?:p|q)*?r|(?:m|n)++m', 0, 'aab ababc xxy pqr mnm', id='groups'),
        pytest.param(r'(?:ab){1,2}+', 0, 'ababab', id='possessive-group-at-its-bound'),
        pytest.param(r'x*', 0, 'axxb', id='empty-matches'),
        pytest.param(r'(?:a?){2,3}b', 0, 'b ab', id='empty-iterations'),
        pytest.param(r'(?:|a)*', 0, 'xaa', id='empty-first-alternative'),
        pytest.param(r'(?:ab)*?c|(?:|b)*?d', 0, 'ababc bd d', id='lazy-group-iterations'),
        pytest.param(r'(?:a?){2}+|(?:b?){0,3}+c', 0, 'x bbc', id='empty-possessive-iterations'),
    ],
)
def test_the_step_counter_matches_like_re(pattern: str, flags: int, text: str):
    """The counter is only evidence if it backtracks the way `re` does: it finds the same matches."""
    assert StepCounter(pattern, flags).scan(text, budget=10_000).spans == [
        m.span() for m in re.finditer(pattern, text, flags)
    ]


def test_the_step_counter_refuses_a_construct_it_does_not_model():
    with pytest.raises(NotImplementedError, match='GROUPREF'):
        StepCounter(r'(a)\1')


@pytest.mark.parametrize(
    ('pattern', 'expected'),
    [
        pytest.param(
            r'\beyJ[a-z_-]{2,}\.x',
            snapshot(
                Shape(units=['-', '.x', '_', 'a', 'eyJ', 'eyJ-', 'eyJ_', 'eyJa'], reach=7, unbounded=True, nodes=8)
            ),
            id='literal-before-a-repeat',
        ),
        pytest.param(
            r'\$\{?[A-Z]*(?:KEY|ID)',
            snapshot(Shape(units=['$', '$A', '$ID', '$KEY', '${', 'A', '{'], reach=6, unbounded=True, nodes=11)),
            id='optional-repeat-keeps-the-prefix',
        ),
        pytest.param(
            r'".*?"[^"]+[\w-]',
            snapshot(
                Shape(
                    units=['"', '""', '""0', '""A', '""a', '"0', '"A', '"a', '-', '0', 'A', 'a'],
                    reach=5,
                    unbounded=True,
                    nodes=7,
                )
            ),
            id='wildcards-and-categories',
        ),
        pytest.param(
            r'(?<!-)(?=\w)(?>ab|c)(?i:d){1,3}',
            snapshot(Shape(units=['-', 'a', 'ab', 'c', 'd'], reach=6, unbounded=False, nodes=12)),
            id='groups-and-lookarounds',
        ),
    ],
)
def test_the_pump_units(pattern: str, expected: Shape):
    assert shape(pattern, 0) == expected
