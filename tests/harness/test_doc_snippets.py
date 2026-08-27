"""Static validation of the code snippets shown in the docs.

Every Python snippet in a capability `README.md` (GitHub/PyPI) and in the flat
`docs/<capability>.md` pages (the unified docs site) is checked for the two
failure modes a reader hits immediately:

- **it does not parse** -- a syntax error means the snippet cannot run at all;
- **it imports a harness symbol that does not exist** -- a stale module path or a
  renamed/removed name (e.g. a snippet still importing from
  `pydantic_ai_harness.experimental.<graduated>`).

This is the *static* half of doc-snippet testing. It deliberately does not
execute the snippets -- most build an `Agent` and call `.run()`, which needs a
model -- so it stays fast and needs no mocking. Running snippets against a mocked
model is a separate concern (see `test_readme_quick_start.py` for that shape).

Illustrative signature blocks (API-reference pseudo-code with type annotations or
a bare `*`, which is not runnable Python) opt out with a `{test="skip"}` fence
directive.

A fence marked `{names="defined"}` opts in to a third check: it must not use a
name it never binds. That is the failure a reader hits when they copy the block
into a file of their own, and it is invisible to the two checks above. The mark
is per fence because standing alone is a property of the block -- a page may
deliberately continue an earlier block's namespace.

The check reads names only. A snippet calling a method that has since been
renamed is an attribute access, which neither this nor `_snippet_problem` sees,
and nothing in this repo type-checks a snippet's annotations. A star import
blanks it out entirely: ruff downgrades `F821` to `F405` while one is in scope.
"""

from __future__ import annotations as _annotations

import ast
import importlib
import os
import re
import subprocess
import warnings
from collections.abc import Iterable
from pathlib import Path

import pytest
from _pytest.mark import ParameterSet
from pytest_examples import CodeExample, find_examples
from ruff.__main__ import find_ruff_bin  # pyright: ignore[reportMissingTypeStubs]

_ROOT = Path(__file__).parent.parent
_HARNESS = 'pydantic_ai_harness'


def _harness_import_targets(tree: ast.AST) -> Iterable[tuple[str, str | None]]:
    """`(module, name)` for every `pydantic_ai_harness` symbol a snippet imports.

    `name` is `None` for a plain `import pydantic_ai_harness.x` or a star import,
    where only the module's existence can be checked.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ''
            if module == _HARNESS or module.startswith(f'{_HARNESS}.'):
                for alias in node.names:
                    yield module, None if alias.name == '*' else alias.name
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == _HARNESS or alias.name.startswith(f'{_HARNESS}.'):
                    yield alias.name, None


def _is_missing_harness_module(exc_name: str | None) -> bool:
    """True when an ImportError is a genuinely absent harness module, not a missing extra.

    A missing optional dependency (e.g. `acp` in the `slim` CI job) raises
    `ModuleNotFoundError` naming the third-party package, not the harness module --
    the harness module exists, its extra just isn't installed.
    """
    return exc_name is not None and exc_name.startswith(_HARNESS)


def _snippet_problem(source: str) -> str | None:
    """Return why a snippet is invalid, or `None` if it parses and its harness imports resolve."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return f'does not parse: {exc.msg} (line {exc.lineno})'

    for module, name in _harness_import_targets(tree):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')  # a deprecated shim path still resolves; existence is what we check
                imported = importlib.import_module(module)
        except ImportError as exc:
            if _is_missing_harness_module(exc.name):
                return f'imports `{module}`, which does not exist: {exc}'
            continue  # missing optional extra in this environment; the harness module exists
        if name is not None:
            try:
                exists = hasattr(imported, name)
            except ImportError:
                # a lazy top-level export whose optional extra isn't installed in this environment;
                # the name resolving far enough to demand the extra proves it exists
                continue
            if not exists:
                return f'imports `{name}` from `{module}`, but that name does not exist'
    return None


def _doc_snippets() -> Iterable[ParameterSet]:
    # `find_examples` yields only Python fenced blocks and wants paths relative to
    # the cwd, so pin it to the repo root (matches `test_skill_examples.py`).
    os.chdir(_ROOT)
    readmes = sorted(str(p.relative_to(_ROOT)) for p in _ROOT.glob(f'{_HARNESS}/**/README.md'))
    for ex in find_examples(*readmes, 'docs'):
        yield pytest.param(ex, id=f'{ex.path}:{ex.start_line}')


@pytest.mark.parametrize('example', _doc_snippets())
def test_doc_snippet_valid(example: CodeExample) -> None:
    if example.prefix_settings().get('test', '').startswith('skip'):
        pytest.skip('illustrative signature block; not runnable Python')
    problem = _snippet_problem(example.source)
    assert problem is None, (
        f'{example.path}:{example.start_line} {problem}. '
        'Fix the snippet, or mark the fence `{test="skip"}` if it is illustrative signature pseudo-code.'
    )


def test_doc_snippets_discovered() -> None:
    # Guard against a discovery break silently making the check vacuous.
    assert sum(1 for _ in _doc_snippets()) >= 100


_F821 = re.compile(r'^-:(\d+):(\d+): (F821 .+)$', re.MULTILINE)


def _undefined_names(example: CodeExample) -> list[str]:
    """`path:line:column` and ruff's message for every name the snippet uses but never binds.

    Ruff rather than `exec`, because an `exec` sees neither shape of this bug.
    Annotations: a plain `exec` inherits this module's `from __future__ import
    annotations`, so they stay strings, and compiling with `dont_inherit=True`
    only moves the blind spot to 3.14, where PEP 649 defers them anyway. Bodies:
    defining a function does not run it, on any version. `--isolated` keeps the
    repo's own ruff config out of the verdict, so a snippet is judged the way a
    reader's fresh file would be.
    """
    result = subprocess.run(
        [
            find_ruff_bin(),
            'check',
            '--isolated',
            '--no-cache',
            '--select',
            'F821',
            '--target-version',
            'py310',
            '--output-format',
            'concise',
            '-',
        ],
        input=example.source,
        capture_output=True,
        text=True,
        # A snippet measures ~12ms, so this cannot flake; it bounds a wedged ruff well
        # under the CI job's own timeout.
        timeout=60,
    )
    if result.returncode > 1:  # pragma: no cover - ruff itself failed, not the snippet
        raise RuntimeError(f'ruff failed on {example.path}:{example.start_line}: {result.stderr or result.stdout}')
    # ruff numbers the snippet from 1; the fence line is `start_line`, so its first
    # line of code is the next one. `indent` is what `find_examples` dedented away.
    problems = [
        f'{example.path}:{example.start_line + int(row)}:{int(column) + example.indent} {message}'
        for row, column, message in _F821.findall(result.stdout)
    ]
    if result.returncode == 1 and not problems:
        raise RuntimeError(f'ruff reported a violation this cannot read: {result.stdout}')
    return problems


def _name_checked_snippets() -> Iterable[ParameterSet]:
    for parameter in _doc_snippets():
        example = parameter.values[0]
        if isinstance(example, CodeExample) and example.prefix_settings().get('names') == 'defined':
            yield parameter


def test_name_checked_snippets_discovered() -> None:
    """Every block annotating with `SpendLimits[None]` carries the fence directive.

    A count floor does not guard this. Eight blocks are marked and only four were ever
    broken, so a floor of four is met by the four that were always fine, and #690's own
    bug ships green. `prefix_settings()` is a free-form `key="value"` parse with no key
    validation, so a misspelled directive reads as no directive at all.
    """
    marked = {parameter.id for parameter in _name_checked_snippets()}
    annotating: set[str] = set()
    for parameter in _doc_snippets():
        example = parameter.values[0]
        if isinstance(example, CodeExample) and 'SpendLimits[None]' in example.source:
            annotating.add(f'{example.path}:{example.start_line}')
    assert annotating, 'discovery found no `SpendLimits[None]` blocks at all; the check has gone vacuous'
    assert annotating <= marked, (
        f'`SpendLimits[None]` blocks with no `{{names="defined"}}` fence directive: {sorted(annotating - marked)}'
    )


@pytest.mark.parametrize('example', _name_checked_snippets())
def test_name_checked_snippet_binds_every_name(example: CodeExample) -> None:
    problems = _undefined_names(example)
    assert not problems, (
        '\n'.join(problems) + '\nImport or define the name in the snippet, '
        'or drop the `{names="defined"}` fence directive if the block continues an earlier one.'
    )


def test_undefined_names_detects_both_shapes() -> None:
    annotated = CodeExample.create('def report(limits: SpendLimits[None]) -> None: ...\n', start_line=40)
    assert _undefined_names(annotated) == ['testing.md:41:20 F821 Undefined name `SpendLimits`']
    in_body = CodeExample.create('async def start() -> None:\n    await workflow_handle.execute()\n', start_line=40)
    assert _undefined_names(in_body) == ['testing.md:42:11 F821 Undefined name `workflow_handle`']
    assert _undefined_names(CodeExample.create('x = 1\nprint(x)\n')) == []
    # `find_examples` dedents an indented fence, so the column has to be shifted back onto it.
    indented = CodeExample.create('print(undefined_here)\n', start_line=40, indent=2)
    assert _undefined_names(indented) == ['testing.md:41:9 F821 Undefined name `undefined_here`']


def test_undefined_names_refuses_a_verdict_it_cannot_read() -> None:
    # ruff exits 1 on a snippet it cannot parse and emits no `F821` at all, because it
    # cannot resolve names in a file it cannot parse. Returning `[]` would report that
    # snippet clean -- and so would an output format that has moved.
    with pytest.raises(RuntimeError, match='cannot read'):
        _undefined_names(CodeExample.create('x: Foo = 1\ndef (:\n'))


def test_snippet_problem_detects_each_failure_mode() -> None:
    # Valid: harness imports that resolve, star imports, plain imports, and non-harness imports.
    assert _snippet_problem('from pydantic_ai_harness import CodeMode') is None
    assert _snippet_problem('import pydantic_ai_harness.code_mode') is None
    assert _snippet_problem('from pydantic_ai_harness.tool_output_limits import *') is None
    assert _snippet_problem('from os import path\nimport sys') is None
    # Invalid: syntax, a module that does not exist, and a name that does not exist.
    assert 'does not parse' in (_snippet_problem('def (:') or '')
    assert 'does not exist' in (_snippet_problem('from pydantic_ai_harness.nope import X') or '')
    assert 'does not exist' in (_snippet_problem('from pydantic_ai_harness import NoSuchCapability') or '')


def test_missing_harness_module_classification() -> None:
    assert _is_missing_harness_module('pydantic_ai_harness.experimental.nope') is True
    assert _is_missing_harness_module('acp') is False  # a missing extra, not a harness module
    assert _is_missing_harness_module(None) is False


def test_missing_optional_extra_is_not_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate the `slim` environment: the harness module exists, but importing it
    # fails because its third-party extra is absent. That is not a broken snippet.
    def _extra_missing(module: str) -> object:
        raise ModuleNotFoundError("No module named 'acp'", name='acp')

    monkeypatch.setattr(importlib, 'import_module', _extra_missing)
    assert _snippet_problem('from pydantic_ai_harness.experimental.acp import run_acp_stdio') is None
