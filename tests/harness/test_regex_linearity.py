"""Every regular expression `pydantic_ai_harness` ships scans repeated adversarial input in linear time.

`re` backtracks, so a pattern whose match attempts re-read the same input, from each offset of a run or once per
way of splitting it, takes quadratic or exponential time on input built to provoke it. A harness pattern often runs
on text the model or a tool produced, and it runs on the event loop.

The patterns are found in the package source, so a new one is checked without being listed here: every call to a
`re` function is read, and its pattern is worked out from the literals, f-strings, `'|'.join(...)` calls,
`.pattern` of a compiled pattern, and names bound to them in its module or function. `re.escape(...)` of runtime text stands in as `x`: escaped text
matches only itself, so the pattern around it is what can backtrack. Each call whose pattern only exists at
runtime says where its pattern comes from in `RUNTIME_PATTERNS`, and the built-in tables those calls compile are
checked through `PATTERN_TABLES`.

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
from pathlib import Path

import pytest
from inline_snapshot import snapshot

import pydantic_ai_harness
from pydantic_ai_harness.artifact_scanning import DEFAULT_ARTIFACT_PATTERNS
from pydantic_ai_harness.guardrails.detectors import DEFAULT_PII_PATTERNS, DEFAULT_SECRET_PATTERNS
from pydantic_ai_harness.require_verification import DEFAULT_EDIT_COMMANDS, DEFAULT_VERIFICATION_COMMANDS

from ._regex_steps import GROWTH_LIMIT, Shape, StepCounter, linearity_failure, shape

_PACKAGE = Path(pydantic_ai_harness.__file__).parent
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
    'artifact_scanning/_scanner.py::ArtifactScanner.__post_init__': (
        'a `DEFAULT_ARTIFACT_PATTERNS` entry',
        "the user's `extra_patterns`",
    ),
    'browser_use/_toolset.py::BrowserUseToolset.__init__': ('an alternation of escaped secret placeholders',),
    'file_system/_toolset.py::FileSystemToolset._search_files': ("the model's search pattern",),
    'guardrails/detectors.py::_compile': (
        "a `DEFAULT_SECRET_PATTERNS` or `DEFAULT_PII_PATTERNS` entry, or the user's `extra`",
    ),
    'permissions/_capability.py::Permissions.__post_init__': ("the user's `ask_commands`",),
    'playwright/_toolset.py::_without_endpoint_credentials': (
        'an alternation of the escaped `cdp_url` credential values',
    ),
    'require_verification/_capability.py::RequireVerification.__post_init__': (
        "a `DEFAULT_VERIFICATION_COMMANDS` entry, or the user's `verification_commands`",
        "a `DEFAULT_EDIT_COMMANDS` entry, or the user's `edit_commands`",
    ),
}
"""Calls whose pattern only exists at runtime, keyed by `<file>::<function>`: where each call's pattern comes from, in
source order. A built-in table they compile is checked through `PATTERN_TABLES`; a caller's own pattern can't be."""


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


def _argument(call: ast.Call, position: int, keyword: str) -> ast.expr | None:
    if len(call.args) > position:
        return call.args[position]
    return next((k.value for k in call.keywords if k.arg == keyword), None)


def _patterns(call: ast.Call, function: str, scopes: list[dict[str, _Binding]]) -> list[tuple[str, int]] | None:
    pattern = _argument(call, 0, 'pattern')
    flags = _argument(call, _FLAGS_ARGUMENT[function], 'flags')
    assert pattern is not None
    try:
        patterns = _strings(_fold(pattern, scopes))
        flag_values = [0] if flags is None else _ints(_fold(flags, scopes))
    except _Unresolved:
        return None
    return [(text, flag) for text in patterns for flag in flag_values]


def _call_sites_in(relative: str, tree: ast.Module) -> Iterator[CallSite]:
    """Every call to a `re` function that takes a pattern, in one module."""

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
                yield CallSite(f'{relative}::{qualname}', _patterns(child, child.func.attr, scopes))
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
    path.relative_to(_PACKAGE).as_posix(): path.read_text(encoding='utf-8') for path in sorted(_PACKAGE.rglob('*.py'))
}
# Parsed at collection, once: a module that never names `re` or `regex` can't call either, so it isn't parsed.
_MODULES = [
    (relative, ast.parse(source)) for relative, source in _SOURCES.items() if re.search(r'\b(?:re|regex)\b', source)
]
_CALL_SITES = [site for relative, tree in _MODULES for site in _call_sites_in(relative, tree)]


def _checked_patterns() -> dict[str, tuple[str, int]]:
    checked: dict[str, tuple[str, int]] = {}
    seen: Counter[str] = Counter()
    for site in _CALL_SITES:
        for pattern in site.patterns or ():
            seen[site.location] += 1
            checked[f'{site.location}#{seen[site.location]}' if seen[site.location] > 1 else site.location] = pattern
    for table_name, table in PATTERN_TABLES.items():
        checked.update({f'{table_name}[{name!r}]': pattern for name, pattern in table.items()})
    unique: dict[tuple[str, int], str] = {}
    for label, pattern in checked.items():
        unique.setdefault(pattern, label)
    return {label: pattern for pattern, label in unique.items()}


@pytest.mark.parametrize(('pattern', 'flags'), [pytest.param(*p, id=label) for label, p in _checked_patterns().items()])
def test_scanning_is_linear(pattern: str, flags: int):
    assert linearity_failure(pattern, flags) is None


@pytest.mark.parametrize(
    ('prefix', 'repeated', 'suffix'),
    [
        pytest.param('', 'black/', 'black x', id='tool-path'),
        pytest.param('uv run ', '-a/black ', 'x', id='runner-flags'),
    ],
)
def test_the_default_formatter_check_is_linear(prefix: str, repeated: str, suffix: str):
    """`DEFAULT_VERIFICATION_COMMANDS[0]` stays linear on two formatter commands `linearity_failure` doesn't build.

    A tool name repeated through a path or through runner flags gives the formatter alternative's prefix a place to
    end at each `/` or flag, and the `--check` lookahead must not re-read the rest of the step from each one.
    """
    counter = StepCounter(DEFAULT_VERIFICATION_COMMANDS[0])
    steps = [counter.scan(prefix + repeated * n + suffix, budget=1_000_000).steps for n in (32, 64, 128)]

    assert steps[2] - steps[1] <= GROWTH_LIMIT * (steps[1] - steps[0])


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
            [CallSite('m.py::f', [('^a', 0), ('^b', 0)])],
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
        pytest.param(_UNRESOLVED_SOURCE, [CallSite('m.py::f', None)] * 11, id='runtime-patterns'),
    ],
)
def test_the_call_site_scan(source: str, sites: list[CallSite]):
    assert list(_call_sites_in('m.py', ast.parse(source))) == sites


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
        pytest.param(r'\w+@', snapshot("'0' * 36: over 1332 steps"), id='restart-in-a-run'),
        pytest.param(
            r'\bkey-[\w-]+\.',
            snapshot("'key-' * 6, 12, 24: [242, 914, 3554] steps"),
            id='restart-at-each-word-in-a-run',
        ),
        pytest.param(r'<.*?>', snapshot("'<' * 76: over 4928 steps"), id='lazy-to-a-missing-end'),
        pytest.param(r'\s*\s*x', snapshot("'\\t' * 19: over 1600 steps"), id='adjacent-repeats'),
        pytest.param(r'(a+)+b', snapshot("'a' * 18: over 1140 steps"), id='nested-repeats'),
        pytest.param(r'(?:a|a){0,20}b', snapshot("'a' * 37: over 13376 steps"), id='bounded-alternation'),
        # A one-character lookbehind only lets an attempt start after a character it admits, so its pump unit carries
        # one. `DEFAULT_VERIFICATION_COMMANDS[0]`, before its formatter alternative was split out, restarts at each
        # `black` of `  black  black`, and `blackblack` would never find that.
        pytest.param(
            r'(?<!\S)(?:(?:uv|poetry|pipenv|hatch|pdm)\s+run(?:\s+-\S+)*\s+)?(?:(?<!\S)(?:\S*/)?python3?\s+-m\s+)?'
            r'(?<!\S)(?:\S*/)?'
            r'(?:pytest|py\.test|unittest|tox|nox|mypy|pyright|basedpyright|ty\s+check|ruff\s+check'
            r'|(?:ruff\s+format|black|isort)(?=[^;&|\n]*\s--(?:check|diff)))\b',
            snapshot("'  black' * 10, 20, 40: [2358, 6973, 23103] steps"),
            id='restart-at-each-character-a-lookbehind-admits',
        ),
        pytest.param(
            r'(?<!\S)(?:ab)*(?:ab)*c',
            snapshot("'ab' * 11, 22, 44: [382, 1240, 4408] steps"),
            id='adjacent-repeats-after-a-lookbehind',
        ),
    ],
)
def test_a_super_linear_pattern_fails(pattern: str, failure: str):
    """The check catches each way a backtracking pattern goes super-linear, with step counts that show it."""
    assert linearity_failure(pattern, 0) == failure


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
    """The counter is only evidence if it backtracks the way `re` does: it finds the same matches.

    That holds for a scan of the whole text, and for a search or match from each position in it.
    """
    counter, compiled = StepCounter(pattern, flags), re.compile(pattern, flags)
    assert counter.scan(text, budget=10_000).spans == [m.span() for m in compiled.finditer(text)]
    positions = range(len(text) + 1)
    assert [counter.search(text, 10_000, p).spans for p in positions] == [
        [m.span()] if (m := compiled.search(text, p)) else [] for p in positions
    ]
    assert [counter.match(text, 10_000, p).spans for p in positions] == [
        [m.span()] if (m := compiled.match(text, p)) else [] for p in positions
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
            snapshot(Shape(units=['-', 'a', 'aab', 'ab', 'ac', 'c', 'd'], reach=6, unbounded=False, nodes=12)),
            id='groups-and-lookarounds',
        ),
        pytest.param(
            r'(?<!\S)a(?<!\S)b',
            snapshot(Shape(units=[' a b', 'a', 'a b', 'ab'], reach=2, unbounded=False, nodes=6)),
            id='admitting-from-each-lookbehind-on',
        ),
    ],
)
def test_the_pump_units(pattern: str, expected: Shape):
    assert shape(pattern, 0) == expected
