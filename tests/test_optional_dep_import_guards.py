"""Regression tests for #4927: optional-dep import guards must use ModuleNotFoundError.

Each model/provider file optionally guards a third-party import like::

    try:
        from mistralai import ...
    except ModuleNotFoundError as e:
        raise ImportError('Please install `mistralai` ...') from e

The original code caught ``ImportError`` instead of ``ModuleNotFoundError``. That
swallowed *name* import errors (e.g. ``cannot import name 'UNSET' from 'mistralai'``
when the upstream package renames a symbol) and reported them as if the package
itself were missing — a frustrating dead-end for users on the wrong upstream
version. ``ModuleNotFoundError`` is the narrower subclass that fires only when
the module truly isn't importable.

These tests assert the convention statically so newly-added providers can't
quietly regress.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / 'pydantic_ai_slim' / 'pydantic_ai' / 'models'
PROVIDERS_DIR = ROOT / 'pydantic_ai_slim' / 'pydantic_ai' / 'providers'


def _files_with_optional_dep_guard() -> Iterator[tuple[Path, ast.ExceptHandler]]:
    """Yield every (file, except-handler) pair where the handler re-raises an
    'install ...optional group' ImportError."""
    for d in (MODELS_DIR, PROVIDERS_DIR):
        for f in sorted(d.glob('*.py')):
            try:
                tree = ast.parse(f.read_text())
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Try):
                    continue
                for handler in node.handlers:
                    # Skip nested or class-attribute Try blocks; we want module-level guards
                    if not _reraises_install_hint(handler):
                        continue
                    yield f, handler


def _reraises_install_hint(handler: ast.ExceptHandler) -> bool:
    """True iff the handler's body raises ImportError with an 'optional group' hint."""
    for stmt in ast.walk(handler):
        if isinstance(stmt, ast.Raise) and isinstance(stmt.exc, ast.Call):
            func = stmt.exc.func
            if isinstance(func, ast.Name) and func.id == 'ImportError':
                for arg in stmt.exc.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and 'optional group' in arg.value:
                        return True
    return False


GUARD_FILES: list[tuple[Path, ast.ExceptHandler]] = list(_files_with_optional_dep_guard())


@pytest.mark.parametrize(
    'path,handler',
    GUARD_FILES,
    ids=[f'{p.parent.name}/{p.name}' for p, _ in GUARD_FILES],
)
def test_optional_dep_guard_uses_module_not_found_error(path: Path, handler: ast.ExceptHandler) -> None:
    """Every optional-dep guard must catch ModuleNotFoundError, not bare ImportError.

    Catching ImportError swallows 'cannot import name X' too, which is what bit
    users in #4927. ModuleNotFoundError is the narrower subclass.
    """
    exc_type = handler.type
    assert exc_type is not None, (
        f'{path.name}: optional-dep guard uses bare `except:`, expected `except ModuleNotFoundError`'
    )
    # `except Name` (e.g. `ModuleNotFoundError`) vs `except module.Attr`.
    name: str | None = None
    if isinstance(exc_type, ast.Name):
        name = exc_type.id
    elif isinstance(exc_type, ast.Attribute):
        name = exc_type.attr
    assert name == 'ModuleNotFoundError', (
        f'{path.name} line {handler.lineno}: optional-dep guard catches '
        f'`{name}`, expected `ModuleNotFoundError`. Catching ImportError '
        f'swallows name-import errors and misleads users about which '
        f'upstream version they have (see #4927).'
    )


def test_at_least_one_guard_was_found() -> None:
    """Smoke test: if a refactor moves all guarded imports elsewhere, the test
    above silently passes with 0 cases. This guards against that."""
    assert len(GUARD_FILES) > 5, (
        f'Expected to find optional-dep guards across models/ and providers/, '
        f'only found {len(GUARD_FILES)}. Either the test discovery broke or the '
        f'guard pattern was refactored away.'
    )
