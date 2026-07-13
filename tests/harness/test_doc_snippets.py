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
"""

from __future__ import annotations as _annotations

import ast
import importlib
import os
import warnings
from collections.abc import Iterable
from pathlib import Path

import pytest
from _pytest.mark import ParameterSet
from pytest_examples import CodeExample, find_examples

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
        if name is not None and not hasattr(imported, name):
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


def test_snippet_problem_detects_each_failure_mode() -> None:
    # Valid: harness imports that resolve, star imports, plain imports, and non-harness imports.
    assert _snippet_problem('from pydantic_ai_harness import CodeMode') is None
    assert _snippet_problem('import pydantic_ai_harness.code_mode') is None
    assert _snippet_problem('from pydantic_ai_harness.overflowing_tool_output import *') is None
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
